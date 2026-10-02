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
TOTAL_FIELDS = ["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "book_title",
                "last_update", "tot_point", "tot_over_price", "tot_under_price"]
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


def _daily(games, season, horizon_days, add):
    """Walk the season's days; add(base_midnight_utc) returns datetimes for that day."""
    g = games[games["season"] == season]
    kicks = sorted({_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()})
    if not kicks:
        return [], []
    times = set()
    day = (kicks[0] - timedelta(days=horizon_days)).date()
    while day <= kicks[-1].date():
        times |= set(add(datetime(day.year, day.month, day.day, tzinfo=timezone.utc)))
        day += timedelta(days=1)
    return times, kicks


def _keep(times, kicks, horizon_days=9):
    return [t for t in sorted(times)
            if any(t < k <= t + timedelta(days=horizon_days) for k in kicks) and t >= FIRST_AVAILABLE]


def plan_totals(games, season, horizon_days=9):
    """Totals: Tuesday 7:10am AZ, Friday report run, 75 min before each kickoff."""
    times, kicks = _daily(games, season, horizon_days, lambda b: (
        [b + timedelta(hours=14, minutes=10)] if b.weekday() == 1 else []) + (
        [b + timedelta(hours=21, minutes=40)] if b.weekday() == 4 else []))
    return _keep(set(times) | {k - timedelta(minutes=75) for k in kicks}, kicks, horizon_days)


def plan_openers(games, season, horizon_days=9):
    """Next week's opening lines: Sunday 7:30pm ET and midnight ET (23:30 and 04:00 UTC)."""
    times, kicks = _daily(games, season, horizon_days, lambda b: (
        [b + timedelta(hours=23, minutes=30)] if b.weekday() == 6 else []) + (
        [b + timedelta(hours=4)] if b.weekday() == 0 else []))
    return _keep(times, kicks, horizon_days)


HOURLY_SEASONS, HOURLY_WEEKS = (2025,), range(3, 13)


def plan_hourly(games, season, horizon_days=9):
    """Hourly sample (does checking more often catch more soft prices?): 2025 weeks 3-12,
    every hour from Saturday 15:00 UTC to Sunday 16:00 UTC."""
    if season not in HOURLY_SEASONS:
        return []
    g = games[(games["season"] == season) & games["week"].isin(list(HOURLY_WEEKS)) & (games["weekday"] == "Sunday")]
    kicks = sorted({_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()})
    times = set()
    for sunday in {d.date() for d in pd.to_datetime(g["gameday"])}:
        start = datetime(sunday.year, sunday.month, sunday.day, tzinfo=timezone.utc) - timedelta(hours=9)
        times |= {start + timedelta(hours=h) for h in range(26)}
    return _keep(times, kicks, horizon_days)


def plan_pinnacle(games, season, horizon_days=9):
    """Pinnacle check (is it a better 'sharp' reference than LowVig/BetOnline?): 2024-2025 only,
    Friday report run + 75 min before each kickoff (the snapshots where moneyline v2 can bet)."""
    if season not in (2024, 2025):
        return []
    times, kicks = _daily(games, season, horizon_days, lambda b: (
        [b + timedelta(hours=21, minutes=40)] if b.weekday() == 4 else []))
    return _keep(set(times) | {k - timedelta(minutes=75) for k in kicks}, kicks, horizon_days)


PROP_FIELDS = ["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "market",
               "player", "point", "over_price", "under_price"]
EVENT_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{eid}/odds"


def backfill_props(seasons, key, market="player_reception_yds", regions="us", reserve=500, out_dir=None,
                   dry_run=False, fetcher=None, times=("early", "close")):
    """Player props (or alternate_spreads: rows are team=player, point, price=over_price) via the per-EVENT historical endpoint (data from May 2023; 10 credits per market per
    region per event-snapshot). Two snapshots per game: Friday 21:40 UTC (or 24 h before a non-Sunday
    kickoff) and 75 min before kickoff (the close, for CLV). Event ids come from the side-odds files."""
    out_dir = out_dir or OUT_DIR / ("alternates" if market.startswith("alternate") else
                                    "derivatives" if ("_h1" in market or "team_totals" in market) else "props")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = {"snapshots": 0, "credits": 0, "fetched": 0, "rows": 0, "remaining": None, "stopped": None}
    for s in seasons:
        f = OUT_DIR / f"nfl_odds_{s}.csv.gz"
        if not f.exists() or s < 2023:
            continue
        raw = pd.read_csv(f, usecols=["event_id", "commence_time", "home", "away", "requested_ts"])
        # one event per real game: the feed sometimes carries duplicate/rescheduled event ids — keep the id
        # seen in the most snapshots, at its latest commence time
        n = raw.groupby("event_id").requested_ts.nunique().rename("n")
        ev = raw.drop_duplicates("event_id", keep="last").join(n, on="event_id")
        ev["day"] = pd.to_datetime(ev.commence_time, utc=True).dt.tz_convert("America/New_York").dt.date
        ev = ev.sort_values("n").drop_duplicates(["home", "away", "day"], keep="last")
        ev = ev[pd.to_datetime(ev.commence_time, utc=True) >= pd.Timestamp("2023-05-03", tz="UTC")]
        done_path = out_dir / f"done_{market.replace(',', '+')}_{s}.txt"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = []
        for r in ev.itertuples():
            kick = pd.Timestamp(r.commence_time).to_pydatetime()
            if kick.weekday() == 6:   # Sunday game -> Friday 21:40 UTC
                early = datetime(kick.year, kick.month, kick.day, 21, 40, tzinfo=timezone.utc) - timedelta(days=2)
            else:
                early = kick - timedelta(hours=24)
            # "open": Tuesday 14:10 UTC of game week (soft opening prices, before injury news settles)
            tue = datetime(kick.year, kick.month, kick.day, 14, 10, tzinfo=timezone.utc) - timedelta(
                days=(kick.weekday() - 1) % 7 or 7)
            opts = {"open": tue, "early": early, "close": kick - timedelta(minutes=75)}
            for t in (opts[x] for x in times if opts[x] < kick):
                tag = f"{r.event_id}|{t.isoformat()}"
                if tag not in done:
                    todo.append((r, t, tag))
        cost = 10 * len(regions.split(",")) * len(market.split(","))
        total["snapshots"] += len(todo)
        total["credits"] += len(todo) * cost
        print(f"{s}: {len(todo)} event-snapshots for {market} (~{len(todo) * cost:,} credits)")
        if dry_run:
            continue
        for r, t, tag in todo:
            if total["remaining"] is not None and total["remaining"] - cost < reserve:
                total["stopped"] = f"reserve of {reserve} credits reached"
                return total
            q = urllib.parse.urlencode({"apiKey": key, "regions": regions, "markets": market, "oddsFormat": "american",
                                        "date": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
            try:
                if fetcher:
                    payload, remaining = fetcher(r.event_id, t)
                else:
                    with urllib.request.urlopen(f"{EVENT_URL.format(eid=r.event_id)}?{q}", timeout=30) as resp:
                        remaining = resp.headers.get("x-requests-remaining")
                        remaining = float(remaining) if remaining is not None else None
                        payload = json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise
                payload, remaining = {"data": {}}, total["remaining"]   # 404/422: event not in archive
            total["remaining"] = remaining
            d = payload.get("data") or {}
            rows = []
            for bk in d.get("bookmakers", []):
                for m in bk.get("markets", []):
                    if m.get("key", "").startswith(("alternate_spreads", "spreads")):
                        # every alternate line: team, point, price (outcomes come in home/away pairs)
                        for o in m.get("outcomes", []):
                            rows.append({"requested_ts": t.isoformat(), "snapshot_ts": payload.get("timestamp"),
                                         "event_id": r.event_id, "commence_time": r.commence_time, "home": r.home,
                                         "away": r.away, "book": bk.get("key"), "market": m.get("key"),
                                         "player": o.get("name"), "point": o.get("point"), "over_price": o.get("price")})
                        continue
                    by_player = {}
                    for o in m.get("outcomes", []):
                        p = by_player.setdefault(o.get("description"), {})
                        p["point"] = o.get("point")
                        p["over_price" if o.get("name") == "Over" else "under_price"] = o.get("price")
                    for player, p in by_player.items():
                        rows.append({"requested_ts": t.isoformat(), "snapshot_ts": payload.get("timestamp"),
                                     "event_id": r.event_id, "commence_time": r.commence_time, "home": r.home,
                                     "away": r.away, "book": bk.get("key"), "market": m.get("key"), "player": player, **p})
            if rows:
                _append(out_dir / f"{market.replace(',', '+')}_{s}.csv.gz", rows, PROP_FIELDS)
            with done_path.open("a") as fh:
                fh.write(tag + "\n")
            total["fetched"] += 1
            total["rows"] += len(rows)
            if total["fetched"] % 100 == 0:
                print(f"  {total['fetched']} fetched, {total['rows']:,} rows, {remaining} credits left")
    return total


def plan_dense(games, season, horizon_days=9):
    """Hourly snapshots (:10 past each hour) on every day with an unplayed game in the next 9 days:
    the dense price history needed to model how long soft prices last and to train a true-price model."""
    times, kicks = _daily(games, season, horizon_days, lambda b: [b + timedelta(hours=h, minutes=10) for h in range(24)])
    return _keep(times, kicks, horizon_days)


DENSE_FIELDS = FIELDS + ["tot_point", "tot_over_price", "tot_under_price"]


PLANS = {  # name: (planner, markets, regions, subdirectory, fields)
    "main": (None, "h2h,spreads", "us,us2", "", FIELDS),
    "totals": (plan_totals, "totals", "us", "totals", TOTAL_FIELDS),
    "openers": (plan_openers, "h2h,spreads", "us", "openers", FIELDS),
    "hourly": (plan_hourly, "h2h,spreads", "us", "hourly", FIELDS),
    "pinnacle": (plan_pinnacle, "h2h", "bookmakers:pinnacle", "pinnacle", FIELDS),
    "dense": (plan_dense, "h2h,spreads,totals", "us,us2", "dense", DENSE_FIELDS),
    "dense_pin": (plan_dense, "h2h,spreads,totals", "bookmakers:pinnacle", "dense_pin", DENSE_FIELDS),
}


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
            if "totals" in mk:
                tt = {o["name"]: o for o in mk["totals"]["outcomes"]}
                ov, un = tt.get("Over") or {}, tt.get("Under") or {}
                r["tot_point"] = ov.get("point", un.get("point"))
                r["tot_over_price"], r["tot_under_price"] = ov.get("price"), un.get("price")
            rows.append(r)
    return rows


def fetch(key: str, when: datetime, regions: str, markets: str, timeout: float = 30):
    p = {"apiKey": key, "markets": markets, "oddsFormat": "american", "date": when.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if regions.startswith("bookmakers:"):
        p["bookmakers"] = regions.split(":", 1)[1]     # up to 10 books cost the same as one region
    else:
        p["regions"] = regions
    q = urllib.parse.urlencode(p)
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


def remaining_credits(key: str) -> float | None:
    """The /sports endpoint is free (costs 0) and reports the account's remaining credits."""
    try:
        with urllib.request.urlopen(f"https://api.the-odds-api.com/v4/sports?apiKey={key}", timeout=20) as r:
            v = r.headers.get("x-requests-remaining")
            return float(v) if v is not None else None
    except Exception as e:
        print("credit check failed:", e)
        return None


def _append(path: Path, rows: list[dict], fields=FIELDS):
    new = not path.exists()
    with gzip.open(path, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def backfill(games, seasons, key, regions="us,us2", markets="h2h,spreads", reserve=1000, out_dir=OUT_DIR,
             dry_run=False, fetcher=fetch, planner=None, fields=FIELDS):
    planner = planner or plan_snapshots
    n_reg = 1 if regions.startswith("bookmakers:") else len(regions.split(","))
    cost = 10 * len(markets.split(",")) * n_reg
    out_dir.mkdir(parents=True, exist_ok=True)
    total = {"snapshots": 0, "credits": 0, "fetched": 0, "rows": 0, "remaining": None, "stopped": None}
    for s in seasons:
        done_path = out_dir / f"done_{s}.txt"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = [t for t in planner(games, s) if t.isoformat() not in done]
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
                name = f"nfl_odds_{s}_{t:%Y-%m}.csv.gz" if planner is plan_dense else f"nfl_odds_{s}.csv.gz"
                _append(out_dir / name, rows, fields)  # dense: monthly files keep each under GitHub's size limit
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
    ap.add_argument("--plan", default="main", help="main/totals/openers/hourly/pinnacle or props[:market_key]")
    ap.add_argument("--regions", default=None, help="override the plan's regions")
    ap.add_argument("--reserve", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from .data import load_schedules
    games = load_schedules()
    key = os.environ.get("ODDS_API_KEY", "")
    if not a.dry_run and not key:
        raise SystemExit("ODDS_API_KEY not set")
    if key:
        print(f"credits remaining before this run: {remaining_credits(key)}")
    if a.plan.startswith("props"):
        parts = a.plan.split(":")
        market = parts[1] if len(parts) > 1 else "player_reception_yds"
        times = tuple(parts[2].split("+")) if len(parts) > 2 else ("early", "close")
        res = backfill_props(sorted(_seasons(a.seasons), reverse=True), key, market=market, reserve=a.reserve,
                             dry_run=a.dry_run, times=times)
        res["plan"] = a.plan
        print(json.dumps(res))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write(f"### Historical odds: {a.plan} {'(dry run)' if a.dry_run else ''}\n\n```\n{json.dumps(res, indent=2)}\n```\n")
        return
    planner, markets, regions, sub, fields = PLANS[a.plan]
    res = backfill(games, sorted(_seasons(a.seasons), reverse=True),  # newest first
                   key, regions=a.regions or regions, markets=markets, reserve=a.reserve, dry_run=a.dry_run,
                   out_dir=OUT_DIR / sub if sub else OUT_DIR, planner=planner, fields=fields)
    res["plan"] = a.plan
    print(json.dumps(res))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(f"### Historical odds: {a.plan} {'(dry run)' if a.dry_run else ''}\n\n```\n{json.dumps(res, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
