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
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

MODEL = "claude-haiku-4-5"
PROMPT_VERSION = 3   # 2 (2026-10-12): next-game-only statements, practice != returning, full names
#                      3 (2026-10-30): + (C) per-team outlook (sentiment, point impact, novelty, factors) -> news_outlook.jsonl
API_URL = "https://api.anthropic.com/v1/messages"
# USD per million tokens (Claude Haiku 4.5: $1 input, $5 output; cache reads 10%, cache writes 125%)
PRICE = {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25}


class Seen(dict):
    """'Already read' ids -> day first seen. Trimmed by AGE when saved, never by id order.
    2026-10-22: the old `sorted(seen)[-20000:]` kept the 20,000 alphabetically LAST ids once full, so every new item
    whose id sorted below the creeping cutoff (~15% of them) was forgotten as it was saved and re-sent to Claude on
    every run (~160 items a run, ~90 runs a day): most of the automation's Claude spend."""

    def __init__(self, path: Path, today: str | None = None):
        from datetime import datetime, timezone
        super().__init__()
        self.today = today or datetime.now(timezone.utc).date().isoformat()
        try:
            raw = json.loads(path.read_text()) if path.exists() else {}
        except ValueError:
            raw = {}
        if isinstance(raw, list):                  # the old id list: start every id's clock today
            raw = {str(k): self.today for k in raw}
        self.update_from(raw)

    def update_from(self, d: dict) -> None:
        for k, v in d.items():
            if k not in self or str(v) < str(self[k]):
                dict.__setitem__(self, k, str(v))

    def add(self, k: str) -> None:
        self.setdefault(k, self.today)

    def update(self, keys) -> None:               # set-style: seen.update(ids)
        for k in keys:
            self.add(k)

    def save(self, path: Path, keep_days: int = 30) -> None:
        from datetime import date, timedelta
        cut = (date.fromisoformat(self.today) - timedelta(days=keep_days)).isoformat()
        path.write_text(json.dumps({k: v for k, v in sorted(self.items()) if v >= cut}, separators=(",", ":")))


def claude_request(payload: dict, api_key: str, component: str, timeout: float = 60, extra: dict | None = None) -> dict:
    """POST to the Messages API, log tokens + cost, and flag 'out of credits' replies (claude_budget alert)."""
    import urllib.error
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")
        except Exception:
            pass
        if "credit balance" in body.lower() or "billing" in body.lower():
            try:
                from . import claude_budget
                claude_budget.out_of_credits(component, body)
            except Exception as ex:
                print("credit alert failed:", ex)
        raise RuntimeError(f"Claude API {e.code}: {body[:200]}") from e
    log_usage(component, resp, extra)
    return resp


def log_usage(component: str, resp: dict, extra: dict | None = None) -> None:
    """One line per Claude call -> history/claude_usage.jsonl (tokens and cost per component; never raises)."""
    try:
        from datetime import datetime, timezone
        u = resp.get("usage") or {}
        row = {"at": datetime.now(timezone.utc).isoformat(timespec="minutes"), "component": component,
               "model": resp.get("model"), "in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
               "cache_read": u.get("cache_read_input_tokens", 0), "cache_write": u.get("cache_creation_input_tokens", 0),
               "stop": resp.get("stop_reason"), **(extra or {})}
        row["usd"] = round((row["in"] * PRICE["input"] + row["out"] * PRICE["output"] + row["cache_read"] * PRICE["cache_read"]
                            + row["cache_write"] * PRICE["cache_write"]) / 1e6, 5)
        p = Path(__file__).resolve().parents[2] / "history" / "claude_usage.jsonl"
        with p.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        print("usage log failed:", e)
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
from .news_sources import team_site_feeds  # noqa: E402  official team websites (injury / practice reports)
FEEDS.update(team_site_feeds())

CONTEXT_CATEGORIES = ["play_caller_change", "coach_fired_or_hired", "scheme_or_role_change", "snap_limit",
                      "illness_outbreak", "travel_or_weather_disruption", "resting_starters", "motivation_or_locker_room",
                      "contract_holdout", "kicker_or_special_teams", "offensive_line_shuffle", "other"]

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]

SYSTEM = ("You extract NFL injury and depth-chart news for a betting model. Only report facts stated in the "
          "items; never guess. Teams must use these abbreviations: " + ", ".join(TEAMS) + ".")

INSTRUCTIONS = """For each news item below, extract (A) every concrete player-availability or starter signal and
(B) team-level context that could change how a team plays this week.
Return ONLY a JSON object {"availability": [...], "context": [...], "outlook": [...]} (lists may be empty).
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
(C) outlook element: one per team the item says something about for its NEXT game (any position, any topic):
{"item": <item number>, "team": str, "sentiment": integer -2..2 (how the item portrays this team's chances this week:
 -2 clearly worse, 0 neutral, 2 clearly better), "impact_pts": number (your estimate of the effect on this team's
 point margin this week, e.g. -3 for a starting QB out with a weak backup, -0.5 for a depth receiver out, +1 for a
 key starter returning; 0 if negligible), "novelty": "new" | "expected" | "stale" (new = not yet widely known when
 published, expected = already announced or obvious from earlier reports, stale = old news restated),
 "factors": list of any of ["injury", "depth", "coaching", "scheme", "motivation", "morale", "travel", "weather",
 "rest", "matchup", "form", "discipline", "other"], "certainty": number 0-1, "quote": short exact phrase}
Examples of context: new offensive play-caller, coordinator fired, starter on a snap count / limited workload,
illness going through the locker room, travel delay or relocated game, team expected to rest starters,
contract holdout, kicker injured/released, multiple offensive-line starters changed.
Rules (v2, 2026-10-12, from the news audit's wrong calls):
- Only statements about the team's NEXT game. A description of a game already played or in progress ("started",
  "was without", "benched in the third quarter") is not a signal for next week: skip it, or set
  game_week_relevant=false if you report it.
- "returning" means he will PLAY in the next game after missing time. "Returns to practice" / "designated to return"
  is limited_practice or full_practice, never returning or expected_to_play.
- player must be the FULL first and last name as the item gives it; if the item only gives a surname, skip it unless
  the item makes the full name unambiguous.
- is_starting_qb_news is true only for the team's starting quarterback or the player named to replace him.
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
        row = {"title": g("title"), "link": g("link"), "published": pub_iso, "summary": re.sub(r"\s+", " ", desc)[:600]}
        if g("source"):  # Google News: the outlet that published it
            row["outlet"] = g("source")
        out.append(row)
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
    try:
        from . import news_sources as NS
        items, rep = NS.bluesky_items(dict(list(NS.SEED.items())[:3]))
        res["bluesky"] = f"ok ({len(items)} posts from {rep['ok']} accounts)" if rep["ok"] else "failed: no account reachable"
    except Exception as e:
        res["bluesky"] = f"failed: {str(e)[:120]}"
    return res


def _key(item: dict) -> str:
    return hashlib.sha1((item.get("link") or item.get("title", "")).encode()).hexdigest()[:16]


def call_claude(items: list[dict], api_key: str, timeout: float = 60) -> list[dict]:
    body = "\n\n".join(f"[{i}] ({it.get('outlet') or it['source']}, {it.get('published') or 'time unknown'}) {it['title']}\n{it['summary']}"
                       for i, it in enumerate(items))
    payload = {"model": MODEL, "max_tokens": 8000, "system": SYSTEM,
               "messages": [{"role": "user", "content": INSTRUCTIONS + "\n\nITEMS:\n" + body}]}
    resp = claude_request(payload, api_key, "nfl_news", timeout, {"items": len(items)})
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    out, ctx = [], []
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
        if isinstance(obj, dict):
            out, ctx = obj.get("availability") or [], obj.get("context") or []
            LAST_OUTLOOK[:] = [x for x in obj.get("outlook") or [] if isinstance(x, dict) and x.get("team") in TEAMS]
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
LAST_OUTLOOK: list = []   # (C) per-team outlook items from the most recent call_claude


def scan(history_dir: Path, api_key: str | None = None, now: datetime | None = None, fetcher=fetch_feed,
         llm=call_claude, max_items: int = 200, roster: dict | None = None, bluesky=None) -> dict:
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
    seen = Seen(seen_p, now.date().isoformat())
    new = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url)
            report["feeds"][name] = len(items)
        except Exception as e:
            report["feeds"][name] = f"failed: {str(e)[:80]}"
            continue
        stale = (now - timedelta(days=20)).isoformat()      # older than the seen list's memory: never re-read
        for it in items:
            k = _key(it)
            if (it.get("published") or "9") < stale:
                continue
            if k not in seen and k not in {x["key"] for x in new}:
                new.append(dict(it, source=name, key=k))
    # Bluesky reporters and team accounts (public API, no login); bluesky=False turns it off (tests)
    from . import news_sources as NS
    if bluesky is None and fetcher is not fetch_feed:
        bluesky = False   # injected test feeds: no live Bluesky calls
    if bluesky is not False:
        try:
            accounts = NS.discover(history_dir) if bluesky is None else bluesky
            items, rep = NS.bluesky_items(accounts)
            report["feeds"]["bluesky"] = rep
            cutoff = (now - timedelta(days=2)).isoformat()
            for it in items:
                k = _key(it)
                if (it.get("published") or "") >= cutoff and k not in seen and k not in {x["key"] for x in new}:
                    new.append(dict(it, source="bluesky", key=k))
        except Exception as e:
            report["feeds"]["bluesky"] = f"failed: {str(e)[:80]}"
    report["new_items"] = len(new)
    sw = NS.weights(history_dir)
    signals, contexts, outlooks = [], [], []
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
                    "seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"), "prompt_version": PROMPT_VERSION,
                    "source": src.get("source"), "outlet": src.get("outlet"), "link": src.get("link"), "title": src.get("title"),
                    **{k: s.get(k) for k in ("player", "team", "position", "signal", "is_starting_qb_news",
                                             "game_week_relevant", "certainty", "quote")}}, roster))
                w = sw.get(NS.source_key(signals[-1]))
                if w:
                    signals[-1]["source_weight"], signals[-1]["source_muted"] = w["weight"], w["muted"]
            for c in LAST_CONTEXT:
                idx = c.get("item")
                src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                contexts.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                                 "source": src.get("source"), "outlet": src.get("outlet"), "link": src.get("link"), "title": src.get("title"),
                                 **{k: c.get(k) for k in ("team", "category", "summary", "player", "direction",
                                                          "game_week_relevant", "certainty", "quote")}})
            LAST_CONTEXT.clear()
            for o in LAST_OUTLOOK:
                idx = o.get("item")
                src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                try:
                    sent, imp = int(o.get("sentiment") or 0), float(o.get("impact_pts") or 0)
                except (TypeError, ValueError):
                    continue
                outlooks.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                                 "source": src.get("source"), "outlet": src.get("outlet"), "link": src.get("link"),
                                 "title": src.get("title"), "team": o["team"], "sentiment": max(-2, min(2, sent)),
                                 "impact_pts": max(-10.0, min(10.0, imp)), "novelty": o.get("novelty"),
                                 "factors": [f for f in (o.get("factors") or []) if isinstance(f, str)][:6],
                                 "certainty": o.get("certainty"), "quote": o.get("quote"), "prompt_version": PROMPT_VERSION})
            LAST_OUTLOOK.clear()
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
    if outlooks:  # (C) per-team outlook: sentiment / point impact / novelty (news_features.py turns it into game features)
        with (history_dir / "news_outlook.jsonl").open("a") as f:
            for o in outlooks:
                f.write(json.dumps(o) + "\n")
    seen.save(seen_p)
    report["outlook_items"] = len(outlooks)
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
    try:  # Claude credit budget: phone alert at 50/75/90% of the last top-up (claude_budget.json)
        from . import claude_budget
        print("claude budget:", json.dumps(claude_budget.check(root / "history")))
    except Exception as e:
        print("claude budget check failed:", e)
    try:  # per-source accuracy (classifies new wrong calls once, then reweights sources)
        from . import news_sources as NS
        out = NS.refresh(root / "history", os.environ.get("ANTHROPIC_API_KEY"))
        top = list((out.get("sources") or {}).items())[:12]
        print("sources:", json.dumps({k: (v["n"], v["accuracy"], v["weight"], v["muted"]) for k, v in top}))
    except Exception as e:
        print("source accuracy failed:", e)


if __name__ == "__main__":
    main()
