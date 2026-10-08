"""Live college football odds snapshots (logging only, for now): history/cfb/odds_<UTC>.json.gz.

Cadence (called from the hourly odds watch): every hour Thursday 12:00 UTC to Sunday 08:00 UTC (game
days and the late-week market), every 3rd hour otherwise. Each snapshot = us,us2 h2h/spreads/totals
(6 credits) + Pinnacle and exchanges (3 credits): ~4-5k credits a month.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from nflpred import odds as O

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history" / "cfb"
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"


def due(now: datetime) -> bool:
    wd, h = now.weekday(), now.hour          # Mon=0
    busy = (wd == 3 and h >= 12) or wd in (4, 5) or (wd == 6 and h < 8)
    # hourly when Pinnacle first posts college lines (Sun 16-24 UTC, Mon 13-17 UTC): the richest shop window in
    # 2021-25 (Sunday snapshots +5.3% CLV vs +2.4% overall; output/research/round3/cfb_shop_segments.md)
    posting = (wd == 6 and h >= 16) or (wd == 0 and 13 <= h < 17)
    return busy or posting or h % 3 == 0


def fetch(key: str) -> list[dict]:
    base = {"apiKey": key, "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    events = O._get({**base, "regions": "us,us2"}, 30, url=URL)
    try:
        extra = {e["id"]: e for e in O._get({**base, "bookmakers": O.EXTRA_BOOKS}, 30, url=URL)}
        for e in events:
            if e.get("id") in extra:
                e["bookmakers"] = e.get("bookmakers", []) + extra[e["id"]].get("bookmakers", [])
    except Exception as ex:
        print("cfb odds: pinnacle/exchanges skipped:", ex)
    return events


def run(now: datetime | None = None, force: bool = False) -> Path | None:
    now = now or datetime.now(timezone.utc)
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("cfb odds: no ODDS_API_KEY")
        return None
    if not (force or due(now)):
        print("cfb odds: not due this hour")
        return None
    events = fetch(key)
    HIST.mkdir(parents=True, exist_ok=True)
    path = HIST / f"odds_{now:%Y-%m-%dT%H%M}.json.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(events))
    books = {b["key"] for e in events for b in e.get("bookmakers", [])}
    print(f"cfb odds: saved {path.name}: {len(events)} games, {len(books)} books; credits left {O.CREDITS.get('x-requests-remaining')}")
    return path


if __name__ == "__main__":
    run(force="--force" in sys.argv)
