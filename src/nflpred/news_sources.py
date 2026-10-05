"""More news sources for the AI news reader, and a per-source accuracy record.

Sources added 2026-10-04:
  * Bluesky (public AppView API, no login): NFL reporters/insiders and official team accounts. Accounts =
    a seed list of known beat writers + accounts discovered weekly with app.bsky.actor.searchActors
    ("<team> reporter/beat/insider", bio must name the team and a reporter word, >= MIN_FOLLOWERS) + the 32 team
    website domains tried as Bluesky handles (teams that verified with their domain). Saved to
    history/bluesky_accounts.json (reviewable; delete a handle there to drop it, add one to SEED to force it).
  * Official team websites (injury/practice reports): Google News RSS restricted to the 32 team domains.

Source accuracy (history/news_sources.json, refreshed after each audit):
  Every graded audit unit credits each source that carried that signal (correct / wrong). Wrong units are
  classified once by Claude (history/news_errors.json): "misread" (the item does not say what we logged: the AI's
  fault, not the source's), "stale_or_hedged" (the item said it, but it was conditional or old news that changed),
  "source_wrong" (the item stated it plainly and it did not happen). A source's accuracy counts correct vs
  stale_or_hedged + source_wrong, with a Beta(4, 1) prior (80%). A source with >= MUTE_MIN_N graded signals and a
  posterior below MUTE_BELOW is muted: its new signals are still logged, marked source_muted, and the dashboard
  hides them. Weights are labels only; the frozen news_rules.json promotion rule is not changed by them.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BSKY = "https://public.api.bsky.app/xrpc/"
UA = {"User-Agent": "nfl-win-probability news reader (github.com/hollingswood/nfl-win-probability)"}
MIN_FOLLOWERS = 1500
PRIOR = (4.0, 1.0)            # Beta prior: 4 correct, 1 wrong
MUTE_MIN_N, MUTE_BELOW = 10, 0.60
MAX_ACCOUNTS = 160

TEAM_SITES = {"ARI": "azcardinals.com", "ATL": "atlantafalcons.com", "BAL": "baltimoreravens.com", "BUF": "buffalobills.com",
              "CAR": "panthers.com", "CHI": "chicagobears.com", "CIN": "bengals.com", "CLE": "clevelandbrowns.com",
              "DAL": "dallascowboys.com", "DEN": "denverbroncos.com", "DET": "detroitlions.com", "GB": "packers.com",
              "HOU": "houstontexans.com", "IND": "colts.com", "JAX": "jaguars.com", "KC": "chiefs.com", "LV": "raiders.com",
              "LAC": "chargers.com", "LA": "therams.com", "MIA": "miamidolphins.com", "MIN": "vikings.com", "NE": "patriots.com",
              "NO": "neworleanssaints.com", "NYG": "giants.com", "NYJ": "newyorkjets.com", "PHI": "philadelphiaeagles.com",
              "PIT": "steelers.com", "SF": "49ers.com", "SEA": "seahawks.com", "TB": "buccaneers.com", "TEN": "tennesseetitans.com",
              "WAS": "commanders.com"}
TEAM_NAMES = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
              "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
              "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LV": "Raiders", "LAC": "Chargers", "LA": "Rams",
              "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
              "PHI": "Eagles", "PIT": "Steelers", "SF": "49ers", "SEA": "Seahawks", "TB": "Buccaneers", "TEN": "Titans",
              "WAS": "Commanders"}
# known NFL reporters on Bluesky (seen in public follow lists 2026-10-04); discovery adds more
SEED = {"jordanraanan.bsky.social": "NYG", "joebuscaglia.bsky.social": "BUF", "davebirkett.bsky.social": "DET",
        "nateatkins.bsky.social": "LA", "darrenurban.bsky.social": "ARI", "marcraimondi.bsky.social": "ATL",
        "giana-jade.bsky.social": "BAL", "antwanstaley.bsky.social": "NYJ", "patriciatraina.bsky.social": "NYG",
        "wyche89.bsky.social": None, "sethwickersham.bsky.social": None, "nflnetwork.bsky.social": None,
        "rotopat.bsky.social": None, "thomasgower.bsky.social": "TEN"}
FOOTBALL = re.compile(r"\b(NFL|football)\b", re.I)   # drops same-nickname baseball/NBA writers (SF Giants, etc.)
REPORTER = re.compile(r"\b(reporter|beat|covers?|covering|writer|insider|correspondent|columnist|analyst|radio|host)\b", re.I)


def team_site_feeds() -> dict:
    """Google News RSS limited to official team sites, in 4 groups of 8 teams (query length)."""
    doms = list(TEAM_SITES.values())
    out = {}
    for i in range(0, 32, 8):
        q = "(" + " OR ".join(f"site:{d}" for d in doms[i:i + 8]) + ") (injury OR practice OR status OR ruled OR start) when:2d"
        out[f"team_sites_{i // 8 + 1}"] = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
                                          + "&hl=en-US&gl=US&ceid=US:en")
    return out


def _get(method: str, params: dict, timeout: float = 15) -> dict:
    req = urllib.request.Request(BSKY + method + "?" + urllib.parse.urlencode(params), headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def discover(history_dir: Path, get=_get, now: datetime | None = None, max_age_days: int = 7) -> dict:
    """Refresh history/bluesky_accounts.json at most weekly. Returns {handle: {team, name, followers, how}}."""
    now = now or datetime.now(timezone.utc)
    path = history_dir / "bluesky_accounts.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    if cur.get("_refreshed") and now - datetime.fromisoformat(cur["_refreshed"]) < timedelta(days=max_age_days):
        return {k: v for k, v in cur.items() if not k.startswith("_")}
    acc = {h: {"team": t, "how": "seed"} for h, t in SEED.items()}
    for team, dom in TEAM_SITES.items():   # official team accounts verified by domain
        try:
            p = get("app.bsky.actor.getProfile", {"actor": dom})
            acc[p["handle"]] = {"team": team, "name": p.get("displayName"), "followers": p.get("followersCount"), "how": "team_domain"}
        except Exception:
            pass
    for team, nm in TEAM_NAMES.items():
        for q in (f"{nm} reporter", f"{nm} beat"):
            try:
                res = get("app.bsky.actor.searchActors", {"q": q, "limit": 25}).get("actors", [])
            except Exception:
                continue
            for a in res:
                bio = (a.get("description") or "") + " " + (a.get("displayName") or "")
                if (nm.lower() in bio.lower() and REPORTER.search(bio) and FOOTBALL.search(bio)
                        and a.get("handle") not in acc):
                    acc[a["handle"]] = {"team": team, "name": a.get("displayName"), "how": "search"}
    # follower counts for search hits (searchActors does not return them); keep the biggest
    for h, v in list(acc.items()):
        if v.get("how") == "search":
            try:
                v["followers"] = get("app.bsky.actor.getProfile", {"actor": h}).get("followersCount", 0)
            except Exception:
                v["followers"] = 0
            if v["followers"] < MIN_FOLLOWERS:
                del acc[h]
    ranked = sorted(acc.items(), key=lambda kv: (kv[1].get("how") == "search", -(kv[1].get("followers") or 0)))[:MAX_ACCOUNTS]
    acc = dict(ranked)
    history_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_refreshed": now.isoformat(timespec="minutes"), **acc}, indent=1))
    return acc


def bluesky_items(accounts: dict, get=_get, per_account: int = 15) -> tuple[list[dict], dict]:
    """Recent original posts (no replies) of each account as feed items. -> (items, report)."""
    items, rep = [], {"accounts": len(accounts), "ok": 0, "failed": 0}
    for h, meta in accounts.items():
        try:
            feed = get("app.bsky.feed.getAuthorFeed", {"actor": h, "limit": per_account, "filter": "posts_no_replies"}).get("feed", [])
            rep["ok"] += 1
        except Exception:
            rep["failed"] += 1
            continue
        for f in feed:
            p = f.get("post") or {}
            if (f.get("reason") or {}).get("$type", "").endswith("reasonRepost"):
                continue
            rec = p.get("record") or {}
            text = re.sub(r"\s+", " ", rec.get("text") or "").strip()
            if not text:
                continue
            ext = ((p.get("embed") or {}).get("external") or {})
            rkey = (p.get("uri") or "").rsplit("/", 1)[-1]
            items.append({"title": text[:140], "link": f"https://bsky.app/profile/{h}/post/{rkey}",
                          "published": (rec.get("createdAt") or "")[:16] + "+00:00" if rec.get("createdAt") else None,
                          "summary": (text + (" | " + ext.get("title", "") if ext.get("title") else ""))[:600],
                          "outlet": f"bsky:{h}" + (f" ({meta.get('team')})" if meta.get("team") else "")})
    return items, rep


# ------------------------------------------------------------------------------------------- source accuracy
def source_key(sig: dict) -> str:
    """Who said it: Bluesky handle / Google News outlet / feed name. Google titles end in ' - Outlet'."""
    o = sig.get("outlet")
    if o:
        return o.split(" (")[0]
    src = sig.get("source") or "unknown"
    if src.startswith(("google", "team_sites")):
        m = re.search(r" - ([^-]{2,60})$", sig.get("title") or "")
        if m:
            return m.group(1).strip()
    return src


def error_id(ev: dict) -> str:
    return f"{ev.get('link') or ev.get('title')}|{ev.get('player')}|{ev.get('signal')}"


CLASSIFY = """A news-reading AI logged NFL availability signals that turned out wrong. For each case decide WHY, from the
item text alone:
 "misread": the title/quote does not actually say the logged signal (the AI misread or mis-assigned player/team),
 "stale_or_hedged": the item says it, but conditionally ("expected", "hopes", "could") or it was later superseded,
 "source_wrong": the item states it plainly and definitively, and it did not happen.
Return ONLY a JSON object {"<case id>": "misread" | "stale_or_hedged" | "source_wrong", ...}."""


def classify_errors(history_dir: Path, api_key: str, llm=None, model: str = "claude-sonnet-5") -> dict:
    """Classify each signal behind a 'wrong' audit unit once; cached in history/news_errors.json."""
    a_p, e_p = history_dir / "news_audit.json", history_dir / "news_errors.json"
    if not a_p.exists():
        return {}
    errs = json.loads(e_p.read_text()) if e_p.exists() else {}
    todo = []
    for r in json.loads(a_p.read_text()).get("rows", []):
        if r.get("verdict") != "wrong":
            continue
        for ev in r.get("evidence", []):
            k = error_id(ev)
            if k not in errs and (k, ev.get("title")) not in {(t[0], t[1].get("title")) for t in todo}:
                todo.append((k, ev, r))
    if not todo:
        return errs
    body = "\n\n".join(f"[{i}] logged: {ev.get('player')} ({r.get('team')}) = {ev.get('signal')}; truth: {r.get('truth')}; "
                       f"seen {ev.get('seen_at')}, source {source_key(ev)}\ntitle: {ev.get('title')}\nquote: {ev.get('quote')}"
                       for i, (k, ev, r) in enumerate(todo[:40]))
    call = llm or (lambda prompt: _claude(prompt, api_key, model))
    try:
        out = call(CLASSIFY + "\n\nCASES:\n" + body)
    except Exception as e:
        print("news error classification failed:", e)
        errs["_status"] = f"failed {datetime.now(timezone.utc).isoformat(timespec='minutes')}: {str(e)[:200]}"
        e_p.write_text(json.dumps(errs, indent=1))
        return errs
    out = {re.sub(r"\D", "", str(k)): v for k, v in (out or {}).items()}   # "[3]" / "case 3" -> "3"
    errs["_status"] = f"ok {datetime.now(timezone.utc).isoformat(timespec='minutes')}: {len(out)} answers for {min(len(todo), 40)} cases"
    for i, (k, ev, r) in enumerate(todo[:40]):
        v = out.get(str(i))
        if v in ("misread", "stale_or_hedged", "source_wrong"):
            errs[k] = {"why": v, "source": source_key(ev), "classified_at": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    e_p.write_text(json.dumps(errs, indent=1))
    return errs


def _claude(prompt: str, api_key: str, model: str) -> dict:
    payload = {"model": model, "max_tokens": 2000, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode(), method="POST",
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.load(r)
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def source_table(audit: dict, errors: dict) -> dict:
    """Per-source record and weight from audit rows (each graded unit credits each distinct source once)."""
    tab: dict = {}
    for r in audit.get("rows", []):
        if r.get("verdict") not in ("correct", "wrong"):
            continue
        done = set()
        for ev in r.get("evidence", []):
            s = source_key(ev)
            if s in done:
                continue
            done.add(s)
            t = tab.setdefault(s, {"n": 0, "correct": 0, "misread": 0, "stale_or_hedged": 0, "source_wrong": 0, "unclassified": 0})
            t["n"] += 1
            if r["verdict"] == "correct":
                t["correct"] += 1
            else:
                t[(errors.get(error_id(ev)) or {}).get("why", "unclassified")] += 1
    for s, t in tab.items():
        bad = t["stale_or_hedged"] + t["source_wrong"] + t["unclassified"]   # unclassified counts against until known
        judged = t["correct"] + bad
        t["accuracy"] = round(t["correct"] / judged, 3) if judged else None
        t["weight"] = round((t["correct"] + PRIOR[0]) / (judged + PRIOR[0] + PRIOR[1]), 3)
        t["muted"] = bool(judged >= MUTE_MIN_N and t["weight"] < MUTE_BELOW)
    return dict(sorted(tab.items(), key=lambda kv: -kv[1]["n"]))


def refresh(history_dir: Path, api_key: str | None) -> dict:
    """Classify new errors (needs the API key) and rewrite history/news_sources.json."""
    a_p = history_dir / "news_audit.json"
    if not a_p.exists():
        return {}
    errs = classify_errors(history_dir, api_key) if api_key else (
        json.loads((history_dir / "news_errors.json").read_text()) if (history_dir / "news_errors.json").exists() else {})
    tab = source_table(json.loads(a_p.read_text()), errs)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "rule": f"weight = (correct + {PRIOR[0]:g}) / (judged + {PRIOR[0] + PRIOR[1]:g}); misreads are the AI's fault and not "
                   f"counted; muted if >= {MUTE_MIN_N} judged and weight < {MUTE_BELOW}", "sources": tab}
    (history_dir / "news_sources.json").write_text(json.dumps(out, indent=1))
    return out


def weights(history_dir: Path) -> dict:
    p = history_dir / "news_sources.json"
    return json.loads(p.read_text()).get("sources", {}) if p.exists() else {}
