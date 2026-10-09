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
import html
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import news_llm as N

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
  "line": number or null (spread from the picked team's side, e.g. -3.5; or the total), "confidence_words": short quote}]}
Use full team names. Return {"picks": []} if there are none."""


def _text(url: str, timeout: float = 20, limit: int = 15000) -> str:
    req = urllib.request.Request(url, headers=N.UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(600_000).decode("utf-8", "ignore")
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    t = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", t)[:limit]


def call_claude(text: str, api_key: str, timeout: float = 90) -> list[dict]:
    payload = {"model": N.MODEL, "max_tokens": 3000, "system": "You extract published betting picks. Never invent picks.",
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
    cands = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url); rep["feeds"][name] = len(items)
        except Exception as e:
            rep["feeds"][name] = f"failed: {str(e)[:60]}"; continue
        for it in items:
            k = hashlib.sha1((it.get("link") or it.get("title", "")).encode()).hexdigest()[:16]
            if k in seen or k in {c["key"] for c in cands} or not PICKY.search(it.get("title", "")) or "news.google.com" in (it.get("link") or "") \
                    or any(b in (it.get("link") or "") for b in BLOCKED):
                continue
            cands.append(dict(it, source=name, key=k))
    rows = []
    for it in cands[:max_articles]:
        seen.add(it["key"])
        try:
            picks = llm(texter(it["link"]), api_key)
        except Exception as e:
            rep.setdefault("errors", []).append(str(e)[:100]); continue
        rep["articles"] += 1
        for p in picks:
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
                    for p in llm(text, api_key):
                        h = str(p.get("analyst") or "").lstrip("@")
                        src = next((it for it in batch if it["outlet"].endswith(":" + h)), None)
                        rows.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published") if src else None,
                                     "source": "bluesky", "title": (src or {}).get("title"), "link": (src or {}).get("link"),
                                     **p, "analyst": "@" + h if h else None, "outlet": f"bsky:{h}" if h else "Bluesky"})
                except Exception as e:
                    rep.setdefault("errors", []).append(str(e)[:100])
        except Exception as e:
            rep["feeds"]["bluesky"] = f"failed: {str(e)[:80]}"
    if rows:
        with (HIST / "picks_log.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
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
