"""AI news reader: pull free NFL news feeds, have Claude (Haiku) turn new items into structured
injury / quarterback signals, and log them with the time we first saw them.

Why: research showed the early-week line misses QB/injury news by ~2 points until the news
settles; the only way to capture that is to know first. This module LOGS signals
(history/news_llm.jsonl) so we can measure, against our saved odds snapshots, whether we see news
before the line moves. It does not place or change any bets.

Runs only when ANTHROPIC_API_KEY is set. Only items not seen before are sent to the model
(history/news_seen.json), so cost stays at a few dollars a month.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

MODEL = "claude-haiku-4-5"
API_URL = "https://api.anthropic.com/v1/messages"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36", "Accept": "application/rss+xml, application/xml, text/xml, */*"}

FEEDS = {
    "pft": "https://www.nbcsports.com/profootballtalk.rss",
    "pft_legacy": "https://profootballtalk.nbcsports.com/feed/",
    "espn": "https://www.espn.com/espn/rss/nfl/news",
    "cbs": "https://www.cbssports.com/rss/headlines/nfl/",
    "yahoo": "https://sports.yahoo.com/nfl/rss/",
    "google_injury": "https://news.google.com/rss/search?q=" + urllib.parse.quote(
        "NFL (injury OR quarterback OR \"ruled out\" OR questionable OR \"will start\") when:1d") + "&hl=en-US&gl=US&ceid=US:en",
    "google_qb": "https://news.google.com/rss/search?q=" + urllib.parse.quote(
        "NFL quarterback start OR benched OR concussion OR \"injured reserve\" when:1d") + "&hl=en-US&gl=US&ceid=US:en",
    "google_context": "https://news.google.com/rss/search?q=" + urllib.parse.quote(
        "NFL (\"play-caller\" OR \"play calling\" OR fired OR \"snap count\" OR illness OR flu OR \"rest starters\" "
        "OR holdout OR \"offensive line\" OR kicker) when:1d") + "&hl=en-US&gl=US&ceid=US:en",
}

CONTEXT_CATEGORIES = ["play_caller_change", "coach_fired_or_hired", "scheme_or_role_change", "snap_limit",
                      "illness_outbreak", "travel_or_weather_disruption", "resting_starters", "motivation_or_locker_room",
                      "contract_holdout", "kicker_or_special_teams", "offensive_line_shuffle", "other"]

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]

SYSTEM = ("You extract NFL injury and depth-chart news for a betting model. Only report facts stated in the "
          "items; never guess. Teams must use these abbreviations: " + ", ".join(TEAMS) + ".")

INSTRUCTIONS = """For each news item below, extract (A) every concrete player-availability or starter signal and
(B) team-level context that could change how a team plays this week.
Return ONLY a JSON object {"availability": [...], "context": [...]} (lists may be empty).
(A) availability element:
{"item": <item number>, "player": str, "team": str, "position": str or null,
 "signal": one of ["out", "doubtful", "questionable", "game_time_decision", "expected_to_play", "will_start",
                   "benched", "injured_reserve", "returning", "limited_practice", "full_practice", "did_not_practice",
                   "suspended", "released_or_traded", "other"],
 "is_starting_qb_news": true/false, "game_week_relevant": true/false,
 "certainty": number 0-1 (how definite the wording is: "ruled out" 1.0, "expected to" 0.7, "could" 0.4),
 "quote": short exact phrase supporting it}
(B) context element:
{"item": <item number>, "team": str, "category": one of """ + json.dumps(CONTEXT_CATEGORIES) + """,
 "summary": one short factual sentence, "player": str or null, "game_week_relevant": true/false,
 "direction": "helps" | "hurts" | "unclear" (for that team this week, only if the item itself implies it),
 "certainty": number 0-1, "quote": short exact phrase supporting it}
Examples of context: new offensive play-caller, coordinator fired, starter on a snap count / limited workload,
illness going through the locker room, travel delay or relocated game, team expected to rest starters,
contract holdout, kicker injured/released, multiple offensive-line starters changed.
Skip items with neither (trade rumors, game recaps, fantasy advice without news). Never guess."""


def fetch_feed(url: str, timeout: float = 20) -> list[dict]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.iter("item"):
        g = lambda tag: (it.findtext(tag) or "").strip()
        pub = g("pubDate")
        try:
            pub_iso = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat(timespec="minutes") if pub else None
        except Exception:
            pub_iso = None
        desc = re.sub(r"<[^>]+>", " ", g("description"))
        out.append({"title": g("title"), "link": g("link"), "published": pub_iso,
                    "summary": re.sub(r"\s+", " ", desc)[:600]})
    return out


def probe() -> dict:
    """Which feeds are reachable from here (run on GitHub to see what Actions can reach)."""
    res = {}
    for name, url in FEEDS.items():
        try:
            items = fetch_feed(url)
            res[name] = f"ok ({len(items)} items)"
        except Exception as e:
            res[name] = f"failed: {str(e)[:120]}"
    return res


def _key(item: dict) -> str:
    return hashlib.sha1((item.get("link") or item.get("title", "")).encode()).hexdigest()[:16]


def call_claude(items: list[dict], api_key: str, timeout: float = 60) -> list[dict]:
    body = "\n\n".join(f"[{i}] ({it['source']}, {it.get('published') or 'time unknown'}) {it['title']}\n{it['summary']}"
                       for i, it in enumerate(items))
    payload = {"model": MODEL, "max_tokens": 4000, "system": SYSTEM,
               "messages": [{"role": "user", "content": INSTRUCTIONS + "\n\nITEMS:\n" + body}]}
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    out, ctx = [], []
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        if isinstance(obj, dict):
            out, ctx = obj.get("availability") or [], obj.get("context") or []
    except Exception:
        m = re.search(r"\[.*\]", text, re.S)   # older array-only answers
        try:
            out = json.loads(m.group(0)) if m else []
        except Exception:
            out = []
    LAST_CONTEXT[:] = [x for x in ctx if isinstance(x, dict) and x.get("team") in TEAMS
                       and x.get("category") in CONTEXT_CATEGORIES]
    return [x for x in out if isinstance(x, dict) and x.get("team") in TEAMS and x.get("player")]


LAST_CONTEXT: list = []   # context items from the most recent call_claude (read by scan)


def scan(history_dir: Path, api_key: str | None = None, now: datetime | None = None, fetcher=fetch_feed,
         llm=call_claude, max_items: int = 150, roster: dict | None = None) -> dict:
    """New feed items -> AI signals appended to history/news_llm.jsonl. Each signal is checked against the
    latest roster snapshot (history/roster.json, saved by full runs; roster.py): `verify` + `team_verified`
    are added, the AI's own fields are kept unchanged. Without a snapshot the check is left to roster.backfill."""
    from . import roster as roster_lib
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    report = {"feeds": {}, "new_items": 0, "signals": 0}
    if not api_key:
        report["skipped"] = "ANTHROPIC_API_KEY not set"
        return report
    now = now or datetime.now(timezone.utc)
    roster = roster if roster is not None else roster_lib.load(history_dir)
    seen_p, log_p = history_dir / "news_seen.json", history_dir / "news_llm.jsonl"
    seen = set(json.loads(seen_p.read_text())) if seen_p.exists() else set()
    new = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url)
            report["feeds"][name] = len(items)
        except Exception as e:
            report["feeds"][name] = f"failed: {str(e)[:80]}"
            continue
        for it in items:
            k = _key(it)
            if k not in seen and k not in {x["key"] for x in new}:
                new.append(dict(it, source=name, key=k))
    report["new_items"] = len(new)
    signals, contexts = [], []
    # newest first; only items actually sent to the model are marked seen (the rest wait for next run)
    new.sort(key=lambda x: x.get("published") or "", reverse=True)
    for i in range(0, min(len(new), max_items), 30):  # batches of 30 items
        batch = new[i:i + 30]
        try:
            for s in llm(batch, api_key):
                if not isinstance(s, dict) or s.get("team") not in TEAMS or not s.get("player"):
                    continue
                idx = s.get("item")
                src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                signals.append(roster_lib.attach({
                    "seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                    "source": src.get("source"), "link": src.get("link"), "title": src.get("title"),
                    **{k: s.get(k) for k in ("player", "team", "position", "signal", "is_starting_qb_news",
                                             "game_week_relevant", "certainty", "quote")}}, roster))
            for c in LAST_CONTEXT:
                idx = c.get("item")
                src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                contexts.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                                 "source": src.get("source"), "link": src.get("link"), "title": src.get("title"),
                                 **{k: c.get(k) for k in ("team", "category", "summary", "player", "direction",
                                                          "game_week_relevant", "certainty", "quote")}})
            LAST_CONTEXT.clear()
            seen.update(x["key"] for x in batch)
        except Exception as e:
            report.setdefault("errors", []).append(str(e)[:160])
    history_dir.mkdir(parents=True, exist_ok=True)
    if signals:
        with log_p.open("a") as f:
            for s in signals:
                f.write(json.dumps(s) + "\n")
    if contexts:  # team-level context (play-caller changes, snap limits, illness...): logged for later testing
        with (history_dir / "news_context.jsonl").open("a") as f:
            for c in contexts:
                f.write(json.dumps(c) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
    report["signals"] = len(signals)
    report["context_items"] = len(contexts)
    report["qb_signals"] = [f"{s['player']} ({s['team']}): {s['signal']}" for s in signals if s.get("is_starting_qb_news")][:20]
    report["roster_check"] = {st: sum((s.get("verify") or {}).get("roster") == st for s in signals)
                              for st in ("match", "team_mismatch", "not_found", "ambiguous")} if roster else "no roster snapshot yet"
    return report


def main():
    import sys
    root = Path(__file__).resolve().parents[2]
    if "--probe" in sys.argv:
        res = probe()
        res["_checked_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
        (root / "history").mkdir(exist_ok=True)
        (root / "history" / "news_probe.json").write_text(json.dumps(res, indent=2))  # readable without log access
        print(json.dumps(res, indent=2))
        return
    print(json.dumps(scan(root / "history"), indent=2))
    try:  # rows logged before a roster snapshot existed
        from . import roster as roster_lib
        print("roster backfill:", roster_lib.backfill(root / "history", roster_lib.load(root / "history")))
    except Exception as e:
        print("roster backfill failed:", e)


if __name__ == "__main__":
    main()
