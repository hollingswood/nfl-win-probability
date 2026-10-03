"""Futures logging (Super Bowl winner and any other NFL outright market The Odds API lists).

Goal: measure how fast futures prices react to big news (a starting QB ruled out, traded, benched)
compared with the game lines, to see whether a slow-reacting book can be caught. Logging only; no bets.

Runs from the hourly workflow (python -m nflpred.futures):
  * every `every_hours` (default 6) a full snapshot, and
  * an extra snapshot when the AI news reader logged a new starting-QB signal since the last one
    (at most one extra per hour), so reaction speed can be measured.
Each snapshot: regions us,us2,us_ex (US books + exchanges) plus Pinnacle = ~4 credits per outright market.
Saved to history/futures/futures_<UTC timestamp>.json.gz as {sport_key: [events]}.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
OUT_DIR = HIST / "futures"
STATE = HIST / "futures_state.json"
API = "https://api.the-odds-api.com/v4"
FALLBACK_KEYS = ["americanfootball_nfl_super_bowl_winner"]
EVERY_HOURS = 6
QB_SIGNALS = {"out", "questionable", "game_time_decision", "did_not_practice", "injured_reserve",
              "released_or_traded", "suspended", "will_start", "returning"}  # news_llm signal values


def _get(path: str, params: dict, timeout: float = 20):
    req = urllib.request.Request(f"{API}{path}?{urllib.parse.urlencode(params)}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        hdr = {k: r.headers.get(k) for k in ("x-requests-remaining", "x-requests-used", "x-requests-last")}
        return json.loads(r.read()), hdr


def outright_keys(key: str) -> list[str]:
    """NFL outright sport keys currently listed (the /sports endpoint costs no credits)."""
    try:
        sports, _ = _get("/sports", {"apiKey": key, "all": "false"})
        ks = sorted(s["key"] for s in sports if s.get("has_outrights") and s["key"].startswith("americanfootball_nfl"))
        return ks or FALLBACK_KEYS
    except Exception as e:
        print("futures: sports list failed, using fallback:", e)
        return FALLBACK_KEYS


def fetch(key: str) -> tuple[dict, dict]:
    snap, hdr = {}, {}
    for sk in outright_keys(key):
        evs = {}
        for extra in ({"regions": "us,us2,us_ex"}, {"bookmakers": "pinnacle"}):
            try:
                data, hdr = _get(f"/sports/{sk}/odds", {"apiKey": key, "markets": "outrights", "oddsFormat": "american", **extra})
            except Exception as e:
                print(f"futures: {sk} {extra} failed: {e}")
                continue
            for ev in data:  # merge Pinnacle into the same event
                e = evs.setdefault(ev["id"], {**ev, "bookmakers": []})
                e["bookmakers"] += ev.get("bookmakers", [])
        snap[sk] = list(evs.values())
    return snap, hdr


def _new_qb_signal(since: datetime | None) -> str | None:
    p = HIST / "news_llm.jsonl"
    if not p.exists():
        return None
    for line in reversed(p.read_text().splitlines()[-400:]):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        seen = datetime.fromisoformat(row.get("seen_at", "1970-01-01T00:00+00:00"))
        if since and seen <= since:
            break
        if row.get("is_starting_qb_news") and row.get("signal") in QB_SIGNALS:
            return f"{row.get('player')} ({row.get('team')}) {row.get('signal')}"
    return None


def due(now: datetime, state: dict) -> str | None:
    last = datetime.fromisoformat(state["last"]) if state.get("last") else None
    if last is None or now - last >= timedelta(hours=EVERY_HOURS) - timedelta(minutes=10):
        return "scheduled"
    if now - last >= timedelta(minutes=50):
        sig = _new_qb_signal(last)
        if sig:
            return f"news: {sig}"
    return None


def run(now: datetime | None = None, force: bool = False) -> Path | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("futures: no ODDS_API_KEY")
        return None
    now = now or datetime.now(timezone.utc)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    why = "manual" if force else due(now, state)
    if not why:
        print("futures: not due")
        return None
    snap, hdr = fetch(key)
    if not snap:
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"futures_{now:%Y-%m-%dT%H%M}.json.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"fetched_at": now.isoformat(timespec="minutes"), "reason": why, "markets": snap}))
    state.update(last=now.isoformat(timespec="minutes"), last_reason=why, credits=hdr)
    STATE.write_text(json.dumps(state, indent=1))
    n = sum(len(v) for v in snap.values())
    print(f"futures: saved {path.name} ({why}; {len(snap)} markets, {n} events; credits left {hdr.get('x-requests-remaining')})")
    return path


if __name__ == "__main__":
    run(force="--force" in sys.argv)
