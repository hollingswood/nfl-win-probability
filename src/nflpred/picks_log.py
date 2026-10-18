"""Analyst-picks logger (logging only): explicit published picks for NFL and college games, with the time we first saw
them, so they can be scored against the closing line at season's end (picks_rules.json).

Prior evidence is against it (tout and expert picks, and fading the public, have not beaten the close in published
studies; reports/NFL and college betting edges.md), so this is a cheap test, not a signal. Nothing here bets.

Sources: public RSS feeds; articles whose titles look like picks ("picks", "best bets", "predictions", "against the
spread") are fetched in full (publisher pages only) and read by Claude Haiku, which returns only picks stated in the text
with their line. Saved to history/picks_log.jsonl (one row per pick) and history/picks_seen.json.
"""
from __future__ import annotations

import hashlib
import html  # noqa: F401
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import news_llm as N
from . import picks_check as PC

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
FEEDS = {
    "cbs_nfl": "https://www.cbssports.com/rss/headlines/nfl/",
    "cbs_cfb": "https://www.cbssports.com/rss/headlines/college-football/",
    "yahoo_nfl": "https://sports.yahoo.com/nfl/rss/",
    "yahoo_cfb": "https://sports.yahoo.com/college-football/rss/",
    "espn_nfl": "https://www.espn.com/espn/rss/nfl/news",
    "espn_cfb": "https://www.espn.com/espn/rss/ncf/news",
    "pft": "https://www.nbcsports.com/profootballtalk.rss",
    # betting sites (added 2026-10-05; reachability is recorded in history/picks_probe.json each run).
    # Not read: Action Network (robots.txt disallows automated access), Reddit (API needs a login, blocks cloud servers)
    "vsin": "https://vsin.com/feed/",
    "pickswise": "https://www.pickswise.com/feed/",
    "covers_nfl": "https://www.covers.com/rss/nfl",
    "oddsshark": "https://www.oddsshark.com/rss.xml",
    # added 2026-10-09 (reachability checked live in history/picks_probe.json; dead ones cost nothing)
    "usatoday_sbw": "https://sportsbookwire.usatoday.com/feed/",
    "thelines": "https://www.thelines.com/feed/",
    "pfn": "https://www.profootballnetwork.com/feed/",
    "sdsouth": "https://www.saturdaydownsouth.com/feed/",
    "fox_nfl": "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/nfl",
    "fox_cfb": "https://api.foxsports.com/v2/content/optimized-rss?partnerKey=MB0Wehpmuj2lUhuRhQaafhBjAJqaPU244mlTDK1i&size=30&tags=fs/college-football",
    "sbr": "https://www.sportsbookreview.com/feed/",
    "bettingpros": "https://www.bettingpros.com/articles/feed/",
    "si_nfl": "https://www.si.com/nfl/.rss/full/",
    "sportingnews": "https://www.sportingnews.com/us/nfl/rss",
}
BLOCKED = ("actionnetwork.com", "reddit.com")
BET_POST = re.compile(r"([+-]\d{1,2}(\.5)?\b|\bML\b|\bmoneyline\b|\b[ou]\s?\d{2}(\.5)?\b|\bover\b|\bunder\b|\b\d(\.\d)?u\b|\bunits?\b|\bATS\b)", re.I)
PICKER_BIO = re.compile(r"(\bpicks?\b|handicap|betting|bettor|\bbets?\b|\bunits?\b|\bATS\b|sharp|wager|sportsbook|odds)", re.I)
PICKER_SPORT = re.compile(r"(\bNFL\b|\bCFB\b|college football|\bfootball\b|\bNCAAF\b)", re.I)
PICKER_QUERIES = ["NFL picks", "college football picks", "CFB picks", "NFL best bets", "NFL betting", "college football betting",
                  "CFB betting", "football handicapper", "NFL ATS", "sports betting football", "NFL props", "betting analyst NFL"]
PICKY = re.compile(r"\b(picks?|best bets?|bets?|predictions?|prediction|against the spread|ATS|locks?|upset picks|expert|odds|spread|parlays?|leans?|props?|over/under)\b", re.I)
INSTR = """Below is the text of a sports article. Extract every EXPLICIT pick the author or a named analyst makes on an NFL or
college football game: a side against the spread, a moneyline winner pick, or an over/under. Only picks stated in the text;
skip straight-up win predictions that give no betting line unless they say "moneyline". Return ONLY JSON:
{"picks": [{"analyst": str or null, "outlet": str or null, "sport": "nfl" or "cfb", "away_team": str, "home_team": str,
  "market": "spread" | "moneyline" | "total", "pick": team name for spread/moneyline, or "over"/"under",
  "line": number or null (spread from the picked team's side, e.g. -3.5; or the total), "confidence_words": quote,
  "reason": why}]}
Use full team names. Return {"picks": []} if there are none.
Lines starting with [PICKS TABLE] are rows of a picks table, already matched to each analyst: the row label gives the
market ("against the spread" rows: the team named is that analyst's side at the row's number, from the side of the team in
the label; "straight up" rows: moneyline winner; "total" rows: over/under). Copy them exactly; never shift a pick to a
neighbouring analyst. Only game lines: skip team totals, player props, first-half/quarter bets and parlay legs.
"confidence_words": the sentence(s) where the analyst states the pick or its main reason, copied WORD FOR WORD from the
article (up to 2 sentences; "" if there is no such sentence).
"reason": 1-2 plain sentences (max 45 words) summarizing WHY this analyst likes this pick, using only reasons the article
gives for it (matchups, injuries, stats, trends, coaching, weather, line value). For a table row, use that analyst's own
write-up elsewhere in the article if it has one. "" if the article gives no reason; never invent one."""


def _text(url: str, timeout: float = 20, limit: int = 90000) -> str:
    """Article text for the AI; picks tables are rewritten as explicit analyst -> pick rows (picks_check layer 1)."""
    from .picks_check import html_to_text
    req = urllib.request.Request(url, headers=N.UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(2_000_000).decode("utf-8", "ignore")
    return html_to_text(raw, limit)


CHUNK = 14000


def chunks(text: str, size: int = CHUNK, overlap: int = 600, most: int = 7) -> list[str]:
    """Long expert-picks pages (CBS lists every game with several analysts each) are read in pieces: one call on the
    first 15,000 characters used to miss most of the page and cut the answer off."""
    out, i = [], 0
    while i < len(text) and len(out) < most:
        out.append(text[i:i + size]); i += size - overlap
    return out or [""]


def pick_key(p: dict) -> tuple:
    return (str(p.get("analyst") or "").strip().lower(), str(p.get("away_team") or "").lower(), str(p.get("home_team") or "").lower(),
            p.get("market"), str(p.get("pick") or "").strip().lower())


def extract(text: str, api_key: str, llm=None) -> list[dict]:
    llm = llm or call_claude
    got, keys = [], set()
    for part in chunks(text):
        for p in llm(part, api_key):
            k = pick_key(p)
            if k not in keys:
                keys.add(k); got.append(p)
    return got


def call_claude(text: str, api_key: str, timeout: float = 120) -> list[dict]:
    payload = {"model": N.MODEL, "max_tokens": 8000, "system": "You extract published betting picks. Never invent picks.",
               "messages": [{"role": "user", "content": INSTR + "\n\nARTICLE:\n" + text}]}
    req = urllib.request.Request(N.API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    out = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", out, re.S)
    try:
        picks = json.loads(m.group(0)).get("picks", []) if m else []
    except Exception:
        picks = []
    return [p for p in picks if isinstance(p, dict) and p.get("market") in ("spread", "moneyline", "total") and p.get("pick")]


def bluesky_pickers(hist: Path = HIST, get=None, now: datetime | None = None, max_age_days: int = 7, limit: int = 80) -> dict:
    """Bluesky accounts that post betting picks (bio mentions picks / capper / betting; >= 300 followers), refreshed
    weekly into history/bluesky_pickers.json (reviewable; delete a handle there to drop it)."""
    from . import news_sources as NS
    get = get or NS._get
    now = now or datetime.now(timezone.utc)
    path = hist / "bluesky_pickers.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    if cur.get("_refreshed") and now - datetime.fromisoformat(cur["_refreshed"]) < timedelta(days=max_age_days):
        return {k: v for k, v in cur.items() if not k.startswith("_")}
    acc = {}
    for q in PICKER_QUERIES:
        try:
            res = get("app.bsky.actor.searchActors", {"q": q, "limit": 50}).get("actors", [])
        except Exception:
            continue
        for a in res:
            bio = f"{a.get('displayName') or ''} {a.get('description') or ''}"
            if PICKER_BIO.search(bio) and PICKER_SPORT.search(bio) and a.get("handle") not in acc:
                acc[a["handle"]] = {"name": a.get("displayName"), "how": q}
    for h in list(acc):
        try:
            acc[h]["followers"] = get("app.bsky.actor.getProfile", {"actor": h}).get("followersCount", 0)
        except Exception:
            acc[h]["followers"] = 0
        if acc[h]["followers"] < 150:
            del acc[h]
    acc = dict(sorted(acc.items(), key=lambda kv: -kv[1]["followers"])[:limit])
    hist.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_refreshed": now.isoformat(timespec="minutes"), **acc}, indent=1))
    return acc


def scan(now: datetime | None = None, api_key: str | None = None, fetcher=N.fetch_feed, texter=_text, llm=call_claude,
         max_articles: int = 30, bluesky=None) -> dict:
    now = now or datetime.now(timezone.utc)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    rep = {"feeds": {}, "articles": 0, "picks": 0}
    if not api_key:
        rep["skipped"] = "ANTHROPIC_API_KEY not set"; return rep
    if now.hour % 3 != 0 and texter is _text:      # live runs: every 3 hours (picks articles change slowly)
        rep["skipped"] = "not due this hour"; return rep
    seen_p = HIST / "picks_seen.json"
    seen = set(json.loads(seen_p.read_text())) if seen_p.exists() else set()
    # articles are re-read (picks pages get more analysts added over the week): up to 4 reads, 12+ h apart, within 4 days
    reads_p = HIST / "picks_reads.json"
    reads = json.loads(reads_p.read_text()) if reads_p.exists() else {}
    def reread_due(k):
        r = reads.get(k)
        if not r:
            return False
        first, last = datetime.fromisoformat(r["first"]), datetime.fromisoformat(r["last"])
        return r.get("n", 1) < 4 and now - first <= timedelta(days=4) and now - last >= timedelta(hours=12)
    logged_keys = set()
    lp = HIST / "picks_log.jsonl"
    if lp.exists():
        for line in lp.read_text().splitlines()[-6000:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            logged_keys.add(pick_key(row))
            if not reads_p.exists() and row.get("link") and row.get("source") != "bluesky" and row.get("seen_at", "") >= (now - timedelta(days=4)).isoformat():
                k0 = hashlib.sha1(row["link"].encode()).hexdigest()[:16]     # first run: articles read under the old 15k-char limit
                reads.setdefault(k0, {"first": row["seen_at"], "last": row["seen_at"], "n": 1, "link": row["link"],
                                      "source": row.get("source"), "title": row.get("title"), "published": row.get("published")})
    cands = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url); rep["feeds"][name] = len(items)
        except Exception as e:
            rep["feeds"][name] = f"failed: {str(e)[:60]}"; continue
        for it in items:
            k = hashlib.sha1((it.get("link") or it.get("title", "")).encode()).hexdigest()[:16]
            if (k in seen and not reread_due(k)) or k in {c["key"] for c in cands} or not PICKY.search(it.get("title", "")) \
                    or "news.google.com" in (it.get("link") or "") or any(b in (it.get("link") or "") for b in BLOCKED):
                continue
            cands.append(dict(it, source=name, key=k))
    # order: articles never read, then (picks_check layer 4) articles with picks read before the current checks (newest
    # first, up to 20 a run, 8 days back: a wrong pick has to drop even after its game was graded), then regular re-reads.
    # 2026-10-18: regular re-reads used to come first and filled every run's 30 slots, so old-check articles never got read
    new_c = [c for c in cands if c["key"] not in reads]
    due_c = [c for c in cands if c["key"] in reads]
    old = sorted((r.get("first", ""), k) for k, r in reads.items()
                 if r.get("checks") != PC.VERSION and r.get("picks_found") and r.get("link")
                 and now - datetime.fromisoformat(r["first"]) <= timedelta(days=8))
    ver_c = []
    for _, k in reversed(old[-20:]):
        r = reads[k]
        ver_c.append({"link": r["link"], "key": k, "source": r.get("source") or "reread", "title": r.get("title"),
                      "published": r.get("published")})
    for k, r in reads.items():      # due re-reads even after the article left its RSS feed
        if reread_due(k) and k not in {c["key"] for c in due_c} and r.get("link"):
            due_c.append({"link": r["link"], "key": k, "source": r.get("source") or "reread", "title": r.get("title"),
                          "published": r.get("published")})
    cands, ks = [], set()
    for c in new_c + ver_c + due_c:
        if c["key"] not in ks:
            ks.add(c["key"]); cands.append(c)
    rows, rej_rows = [], []
    why_p = HIST / "picks_reasons.json"
    why = json.loads(why_p.read_text()) if why_p.exists() else {}
    for it in cands[:max_articles]:
        seen.add(it["key"])
        r = reads.setdefault(it["key"], {"first": now.isoformat(timespec="minutes"), "n": 0, "link": it.get("link"),
                                         "source": it.get("source"), "title": it.get("title"), "published": it.get("published")})
        r["last"], r["n"] = now.isoformat(timespec="minutes"), r.get("n", 0) + 1
        try:
            text = texter(it["link"])
            picks, rejected = PC.validate(extract(text, api_key, llm), text)     # picks_check layer 2
        except Exception as e:
            rep.setdefault("errors", []).append(str(e)[:100]); continue
        rep["articles"] += 1
        r["keys"] = sorted("|".join(map(str, pick_key(p))) for p in picks)      # layer 4: this read's full pick list
        r["checks"] = PC.VERSION
        for p in picks:                     # why each pick was made, also for picks logged on an earlier read
            if p.get("reason") or p.get("confidence_words"):
                kk = "|".join(map(str, pick_key(p)))
                old = why.get(kk) or {}
                if len(p.get("reason") or "") >= len(old.get("reason") or ""):
                    why[kk] = {"reason": p.get("reason") or "", "quote": (p.get("confidence_words") or "")[:400],
                               "at": now.isoformat(timespec="minutes")}
        for x in rejected:
            rej_rows.append({"seen_at": now.isoformat(timespec="minutes"), "link": it.get("link"), "title": it.get("title"), **x})
        new = [p for p in picks if pick_key(p) not in logged_keys]   # a re-read logs only picks we hadn't seen
        r["picks_found"], r["picks_new"] = len(picks), len(new)
        for p in new:
            logged_keys.add(pick_key(p))
            rows.append({"seen_at": now.isoformat(timespec="minutes"), "published": it.get("published"), "source": it["source"],
                         "title": it.get("title"), "link": it.get("link"), **p})
    # Bluesky accounts that post picks: recent posts that look like a bet, read in batches (analyst = @handle)
    if bluesky is None and texter is not _text:
        bluesky = False
    if bluesky is not False:
        try:
            from . import news_sources as NS
            accts = bluesky_pickers() if bluesky is None else bluesky
            items, brep = NS.bluesky_items({h: {} for h in accts}, per_account=20)
            rep["feeds"]["bluesky"] = brep
            cutoff = (now - timedelta(days=2)).isoformat()
            posts = []
            for it in items:
                k = hashlib.sha1(it["link"].encode()).hexdigest()[:16]
                if k in seen or (it.get("published") or "") < cutoff or not BET_POST.search(it.get("summary", "")):
                    continue
                seen.add(k); posts.append(it)
            for i in range(0, min(len(posts), 120), 30):
                batch = posts[i:i + 30]
                text = "Bluesky posts. Each post's author is the @handle in brackets; use it as the analyst and 'Bluesky' as outlet.\n\n" + \
                    "\n\n".join(f"[@{it['outlet'].split(':', 1)[1]}] ({it.get('published')}) {it['summary']}" for it in batch)
                try:
                    ok, bad = PC.validate(llm(text, api_key), text)
                    rej_rows.extend({"seen_at": now.isoformat(timespec="minutes"), "source": "bluesky", **x} for x in bad)
                    for p in ok:
                        h = str(p.get("analyst") or "").lstrip("@")
                        src = next((it for it in batch if it["outlet"].endswith(":" + h)), None)
                        rows.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published") if src else None,
                                     "source": "bluesky", "title": (src or {}).get("title"), "link": (src or {}).get("link"),
                                     **p, "analyst": "@" + h if h else None, "outlet": f"bsky:{h}" if h else "Bluesky"})
                except Exception as e:
                    rep.setdefault("errors", []).append(str(e)[:100])
        except Exception as e:
            rep["feeds"]["bluesky"] = f"failed: {str(e)[:80]}"
    if rej_rows:
        with (HIST / "picks_rejected.jsonl").open("a") as f:
            for r in rej_rows:
                f.write(json.dumps(r) + "\n")
    rep["rejected"] = len(rej_rows)
    if rows:
        with (HIST / "picks_log.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
    cut = (now - timedelta(days=10)).isoformat()
    reads_p.write_text(json.dumps({k: v for k, v in reads.items() if v.get("last", "") >= cut}, indent=0))
    wcut = (now - timedelta(days=120)).isoformat()
    why_p.write_text(json.dumps({k: v for k, v in why.items() if v.get("at", "") >= wcut}, indent=0))
    rep["picks"] = len(rows)
    if texter is _text:   # live runs: which feeds Actions can reach, readable without log access
        (HIST / "picks_probe.json").write_text(json.dumps({"checked_at": now.isoformat(timespec="minutes"), "feeds": rep["feeds"]}, indent=1))
    return rep


if __name__ == "__main__":
    print(json.dumps(scan(), indent=1))
    try:   # score logged picks + hot-pickers test (picks_hot_rules.json) whenever new results may be in
        from . import picks_score
        res = picks_score.run()
        print("picks scored:", res["picks_graded"], "hot test:", json.dumps(res["hot_test"]))
    except Exception as e:
        print("picks scoring failed:", e)
    sys.exit(0)
