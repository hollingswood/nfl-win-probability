"""College football AI news reader (logging only, like the NFL one): free feeds -> Claude Haiku -> structured
QB / availability signals and team context, logged with the time we first saw them.

history/cfb/news_llm.jsonl (availability), history/cfb/news_context.jsonl (team context), history/cfb/news_seen.json.
Team names must be CFBD school names (validated against the FBS list). Runs from the hourly odds watch on the same
cadence as the college odds (hourly Thu-Sat, every 3 h otherwise) to keep the API cost to a few dollars a month.
Nothing here places or changes a bet.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nflpred import news_llm as N

from . import data as D

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history" / "cfb"
Q = lambda s: "https://news.google.com/rss/search?q=" + urllib.parse.quote(s) + "&hl=en-US&gl=US&ceid=US:en"
FEEDS = {
    "espn_cfb": "https://www.espn.com/espn/rss/ncf/news",
    "cbs_cfb": "https://www.cbssports.com/rss/headlines/college-football/",
    "yahoo_cfb": "https://sports.yahoo.com/college-football/rss/",
    "google_cfb_qb": Q("college football quarterback (start OR benched OR injury OR \"ruled out\" OR \"out for\" OR suspended) when:1d"),
    "google_cfb_injury": Q("college football (injury OR \"ruled out\" OR questionable OR \"opt out\" OR suspended OR \"availability report\") when:1d"),
    "google_cfb_context": Q("college football (\"offensive coordinator\" OR \"play-caller\" OR fired OR \"interim coach\" OR illness OR flu) when:1d"),
    "google_cfb_discipline": Q("college football (suspended OR suspension OR dismissed OR arrested OR \"violation of team rules\") when:1d"),
    "google_cfb_eligibility": Q("college football (eligible OR ineligible OR \"transfer portal\" OR \"opts out\" OR redshirt OR \"NCAA ruling\") when:1d"),
    "google_cfb_depth": Q("college football (\"depth chart\" OR \"starting quarterback\" OR \"QB1\" OR \"true freshman\" OR \"will start\") when:1d"),
    "google_cfb_market": Q("college football (\"line move\" OR \"sharp money\" OR \"sharp action\" OR \"betting splits\" OR \"odds shift\") when:1d"),
    "google_cfb_logistics": Q("college football (\"kickoff time\" OR postponed OR relocated OR \"weather delay\" OR hurricane OR \"travel delay\") when:1d"),
}

CONTEXT = ["coach_fired_or_interim", "play_caller_change", "scheme_or_role_change", "suspension_or_discipline",
           "eligibility_or_transfer", "opt_out_or_resting", "illness_outbreak", "weather_or_logistics",
           "kickoff_time_or_venue_change", "motivation", "depth_chart_change", "kicker_or_special_teams",
           "offensive_line_shuffle", "market_report", "other"]
INSTR = """For each news item below, extract (A) every concrete player-availability or starter signal and
(B) team-level context that could move a point spread or total this week.
Return ONLY a JSON object {"availability": [...], "context": [...]} (lists may be empty).
(A) availability element:
{"item": <item number>, "player": str, "team": str, "position": str or null,
 "signal": one of ["out", "out_for_season", "doubtful", "questionable", "game_time_decision", "expected_to_play",
                   "will_start", "benched", "returning", "suspended", "dismissed", "transferred_or_ineligible",
                   "opted_out", "other"],
 "is_starting_qb_news": true/false, "game_week_relevant": true/false,
 "certainty": number 0-1 (how definite the wording is: "ruled out" 1.0, "expected to" 0.7, "could" 0.4),
 "quote": short exact phrase supporting it}
(B) context element:
{"item": <item number>, "team": str, "category": one of """ + json.dumps(CONTEXT) + """,
 "summary": one short factual sentence, "player": str or null, "game_week_relevant": true/false,
 "direction": "helps" | "hurts" | "unclear" (for that team this week, only if the item itself implies it),
 "certainty": number 0-1, "quote": short exact phrase supporting it,
 "line_move": for market_report only: {"from": number or null, "to": number or null, "market": "spread"|"total"|"moneyline"|null}}
Context examples: head coach fired / interim named, new play-caller, several players suspended, a key player ruled
ineligible or entering the portal, starters resting or opting out, illness in the locker room, hurricane/relocated game,
kickoff time moved, rivalry/bowl-eligibility/look-ahead spot stated by a reporter, QB depth chart changed, kicker hurt,
offensive-line starters out, and betting-market reports (line moved, sharp money, betting splits) WITH numbers if given.
Skip recaps, recruiting, rankings talk and fantasy advice without news. Never guess."""


def fbs_teams() -> list[str]:
    now = datetime.now(timezone.utc)
    y = now.year if now.month >= 7 else now.year - 1
    for yy in (y, y - 1):
        t = D._load(f"teams_fbs_{yy}.json.gz")
        if t:
            return sorted(x["school"] for x in t)
    return []


_NAMES: dict | None = None


def team_names() -> dict:
    """school -> lowercase names a story might use (school, abbreviation, alternate names, mascot)."""
    global _NAMES
    if _NAMES is None:
        _NAMES = {}
        now = datetime.now(timezone.utc)
        y = now.year if now.month >= 7 else now.year - 1
        for yy in (y, y - 1):
            for x in D._load(f"teams_fbs_{yy}.json.gz") or []:
                n = {x["school"], x.get("abbreviation"), x.get("mascot"), *(x.get("alternateNames") or [])}
                _NAMES.setdefault(x["school"], set()).update(str(v).lower() for v in n if v and len(str(v)) >= 3)
    return _NAMES


def names_team(r: dict, team: str) -> bool:
    """Team-level context must name its team in the story: the AI has to pick an FBS school, so a story about an FCS
    team (e.g. UMaine) could be filed under a look-alike FBS school (Massachusetts). Unknown teams pass."""
    names = team_names().get(team)
    if not names:
        return True
    text = f" {r.get('title') or ''} {r.get('summary') or ''} ".lower()
    return any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", text) for n in names)


def due(now: datetime) -> bool:
    from .odds_live import due as odds_due
    return odds_due(now)


def call_claude(items: list[dict], api_key: str, teams: list[str], timeout: float = 90):
    system = ("You extract COLLEGE FOOTBALL (FBS) injury, quarterback and availability news for a betting model. "
              "Only report facts stated in the items; never guess. The team field must be EXACTLY one of these school "
              "names: " + "; ".join(teams) + ". Players transfer every year: take each player's CURRENT school from the "
              "item itself, never from memory; in a story about a matchup, check which of the two teams the player is on. "
              "If the item does not make the player's current team clear, skip that player. If the team is not one of these "
              "FBS schools (an FCS or lower-division team such as UMaine or Bryant), skip the item: never substitute a "
              "school with a similar name. "
              "Skip NFL, recruiting and high-school items, and recaps of games already played.")
    instr = INSTR
    body = "\n\n".join(f"[{i}] ({it['source']}, {it.get('published') or 'time unknown'}) {it['title']}\n{it['summary']}"
                       for i, it in enumerate(items))
    payload = {"model": N.MODEL, "max_tokens": 4000, "system": system,
               "messages": [{"role": "user", "content": instr + "\n\nITEMS:\n" + body}]}
    req = urllib.request.Request(N.API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    N.log_usage("cfb_news", resp, {"items": len(items)})
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    ts = set(teams)
    av = [x for x in (obj.get("availability") or []) if isinstance(x, dict) and x.get("team") in ts and x.get("player")]
    cx = [x for x in (obj.get("context") or []) if isinstance(x, dict) and x.get("team") in ts and x.get("category") in CONTEXT]
    return av, cx


def scan(now: datetime | None = None, api_key: str | None = None, fetcher=N.fetch_feed, llm=call_claude,
         max_items: int = 180, force: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    rep = {"feeds": {}, "new_items": 0, "signals": 0, "context_items": 0}
    if not api_key:
        rep["skipped"] = "ANTHROPIC_API_KEY not set"; return rep
    if not (force or due(now)):
        rep["skipped"] = "not due this hour"; return rep
    teams = fbs_teams()
    HIST.mkdir(parents=True, exist_ok=True)
    seen_p = HIST / "news_seen.json"
    seen = set(json.loads(seen_p.read_text())) if seen_p.exists() else set()
    new = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url); rep["feeds"][name] = len(items)
        except Exception as e:
            rep["feeds"][name] = f"failed: {str(e)[:80]}"; continue
        for it in items:
            k = N._key(it)
            if k not in seen and k not in {x["key"] for x in new}:
                new.append(dict(it, source=name, key=k))
    rep["new_items"] = len(new)
    new.sort(key=lambda x: x.get("published") or "", reverse=True)
    sig, ctx = [], []
    for i in range(0, min(len(new), max_items), 30):
        batch = new[i:i + 30]
        try:
            av, cx = llm(batch, api_key, teams)
            for kind, rows, out in (("av", av, sig), ("cx", cx, ctx)):
                for s in rows:
                    idx = s.get("item")
                    src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                    keep = ("player", "team", "position", "signal", "is_starting_qb_news", "game_week_relevant", "certainty", "quote") if kind == "av" \
                        else ("team", "category", "summary", "player", "direction", "game_week_relevant", "certainty", "quote", "line_move")
                    row = {"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                           "source": src.get("source"), "link": src.get("link"), "title": src.get("title"),
                           **{k: s.get(k) for k in keep}}
                    try:   # roster check (cfbpred.roster); the AI's own team field is kept as-is
                        from . import roster as R
                        row["verify"] = R.verify(row.get("player"), row.get("team"), now.year if now.month >= 7 else now.year - 1)
                    except Exception:
                        pass
                    out.append(row)
            seen.update(x["key"] for x in batch)
        except Exception as e:
            rep.setdefault("errors", []).append(str(e)[:160])
    for path, rows in ((HIST / "news_llm.jsonl", sig), (HIST / "news_context.jsonl", ctx)):
        if rows:
            with path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
    rep["signals"], rep["context_items"] = len(sig), len(ctx)
    rep["qb_signals"] = [f"{s['player']} ({s['team']}): {s['signal']}" for s in sig if s.get("is_starting_qb_news")][:20]
    return rep


def recent_by_team(days: int = 7, now: datetime | None = None, idx: dict | None = None, cap: int | None = 6) -> dict:
    """team -> recent availability/context items (newest first), for the college page. Items naming a player are
    filed under the team the roster check finds (cfbpred.roster: this season's passing stats, CFBD roster, transfer
    portal); items whose player can't be placed on a team are left off the page."""
    from . import roster as R
    now = now or datetime.now(timezone.utc)
    season = now.year if now.month >= 7 else now.year - 1
    out: dict = {}
    for name, kind in (("news_llm.jsonl", "av"), ("news_context.jsonl", "cx")):
        p = HIST / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[-3000:]:
            try:
                r = json.loads(line)
                t = datetime.fromisoformat(r["seen_at"])
            except Exception:
                continue
            if (now - t).days > days or not r.get("game_week_relevant", True):
                continue
            v = R.verify(r.get("player"), r.get("team"), season, idx)    # re-checked on every build (rosters update)
            if not v["team_verified"] or (kind == "cx" and not r.get("player") and not names_team(r, v["team_verified"])):
                continue
            out.setdefault(v["team_verified"], []).append({**r, "kind": kind, "verify": v})
    for t in out:   # newest first; one line per player+signal (or per context category without a player)
        keep, dup = [], set()
        for r in sorted(out[t], key=lambda r: r["seen_at"], reverse=True):
            k = (R.norm(r.get("player")), r.get("signal") or r.get("category"))
            if k not in dup:
                dup.add(k); keep.append(r)
        out[t] = keep[:cap] if cap else keep
    return out


if __name__ == "__main__":
    print(json.dumps(scan(force="--force" in sys.argv), indent=2))
