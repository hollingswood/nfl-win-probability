"""Backfill historical NFL odds from The Odds API (paid plans only; data from June 2020).

Why: the spread track was backtested against CLOSING lines, which are the hardest to beat and
not the lines we actually bet. This pulls the lines that were on the board at the same moments
the live pipeline runs, so the pre-registered rules can be replayed exactly on 2020-2025:

  * every day at 14:10 UTC (the 7:10am Arizona daily run)
  * Fridays at 21:40 UTC (the Friday final-injury-report run)
  * 75 minutes before each distinct kickoff time (the game-day runs)

only on days with a game in the next 9 days. Each snapshot costs 10 x markets x regions credits
(h2h + spreads, us + us2 = 40). Output: one gzipped CSV per season in data/historical_odds/, one
row per (snapshot, game, book). Resumable: finished snapshots are listed in done_<season>.txt, so
re-running only fetches what's missing. Stops before the account drops below --reserve credits.

    python -m nflpred.odds_history --seasons 2020-2025 --dry-run     # count snapshots and credits
    ODDS_API_KEY=... python -m nflpred.odds_history --seasons 2020-2025
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .odds import TEAM_ABBR
from .weather import _kickoff_utc

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "historical_odds"
URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
FIELDS = ["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "book_title",
          "last_update", "ml_home", "ml_away", "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"]
FIRST_AVAILABLE = datetime(2020, 6, 6, tzinfo=timezone.utc)


def plan_snapshots(games: pd.DataFrame, season: int, horizon_days: int = 9) -> list[datetime]:
    g = games[games["season"] == season]
    kicks = sorted({_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()})
    if not kicks:
        return []
    times = set()
    day = (kicks[0] - timedelta(days=horizon_days)).date()
    while day <= kicks[-1].date():
        base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        times.add(base + timedelta(hours=14, minutes=10))
        if base.weekday() == 4:
            times.add(base + timedelta(hours=21, minutes=40))
        day += timedelta(days=1)
    times |= {k - timedelta(minutes=75) for k in kicks}
    # keep only moments with an unplayed game within the horizon
    keep = [t for t in sorted(times)
            if any(t < k <= t + timedelta(days=horizon_days) for k in kicks) and t >= FIRST_AVAILABLE]
    return keep


def rows_from_snapshot(payload: dict, requested: datetime) -> list[dict]:
    rows = []
    for ev in payload.get("data", []):
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        for bk in ev.get("bookmakers", []):
            mk = {m["key"]: m for m in bk.get("markets", [])}
            r = {"requested_ts": requested.isoformat(), "snapshot_ts": payload.get("timestamp"),
                 "event_id": ev.get("id"), "commence_time": ev["commence_time"], "home": home, "away": away,
                 "book": bk.get("key"), "book_title": bk.get("title"), "last_update": bk.get("last_update")}
            if "h2h" in mk:
                px = {o["name"]: o["price"] for o in mk["h2h"]["outcomes"]}
                r["ml_home"], r["ml_away"] = px.get(ev["home_team"]), px.get(ev["away_team"])
            if "spreads" in mk:
                sp = {o["name"]: o for o in mk["spreads"]["outcomes"]}
                ho, ao = sp.get(ev["home_team"]) or {}, sp.get(ev["away_team"]) or {}
                r["sp_home_point"], r["sp_home_price"] = ho.get("point"), ho.get("price")
                r["sp_away_point"], r["sp_away_price"] = ao.get("point"), ao.get("price")
            rows.append(r)
    return rows


def fetch(key: str, when: datetime, regions: str, markets: str, timeout: float = 30):
    q = urllib.parse.urlencode({"apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american",
                                "date": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=timeout) as r:
                remaining = r.headers.get("x-requests-remaining")
                return json.load(r), (float(remaining) if remaining is not None else None)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 422):
                raise RuntimeError(f"{e.code}: {e.read().decode()[:300]}") from e
            time.sleep(2 ** attempt * 2)
        except Exception:
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f"failed after retries: {when}")


def _append(path: Path, rows: list[dict]):
    new = not path.exists()
    with gzip.open(path, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def backfill(games, seasons, key, regions="us,us2", markets="h2h,spreads", reserve=1000, out_dir=OUT_DIR,
             dry_run=False, fetcher=fetch):
    cost = 10 * len(markets.split(",")) * len(regions.split(","))
    out_dir.mkdir(parents=True, exist_ok=True)
    total = {"snapshots": 0, "credits": 0, "fetched": 0, "rows": 0, "remaining": None, "stopped": None}
    for s in seasons:
        done_path = out_dir / f"done_{s}.txt"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = [t for t in plan_snapshots(games, s) if t.isoformat() not in done]
        total["snapshots"] += len(todo)
        total["credits"] += len(todo) * cost
        print(f"{s}: {len(todo)} snapshots to fetch (~{len(todo) * cost:,} credits)")
        if dry_run:
            continue
        for t in todo:
            if total["remaining"] is not None and total["remaining"] - cost < reserve:
                total["stopped"] = f"reserve of {reserve} credits reached"
                print("stopping:", total["stopped"])
                return total
            payload, remaining = fetcher(key, t, regions, markets)
            total["remaining"] = remaining
            rows = rows_from_snapshot(payload, t)
            if rows:
                _append(out_dir / f"nfl_odds_{s}.csv.gz", rows)
            with done_path.open("a") as f:
                f.write(t.isoformat() + "\n")
            total["fetched"] += 1
            total["rows"] += len(rows)
            if total["fetched"] % 50 == 0:
                print(f"  {total['fetched']} fetched, {total['rows']:,} rows, {remaining} credits left")
    return total


def _seasons(arg: str) -> list[int]:
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in arg.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2020-2025")
    ap.add_argument("--regions", default="us,us2")
    ap.add_argument("--reserve", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from .data import load_schedules
    games = load_schedules()
    key = os.environ.get("ODDS_API_KEY", "")
    if not a.dry_run and not key:
        raise SystemExit("ODDS_API_KEY not set")
    res = backfill(games, sorted(_seasons(a.seasons), reverse=True),  # newest first
                   key, regions=a.regions, reserve=a.reserve, dry_run=a.dry_run)
    print(json.dumps(res))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Historical odds {'(dry run)' if a.dry_run else ''}\n\n```\n{json.dumps(res, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
