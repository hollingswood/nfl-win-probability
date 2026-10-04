"""Historical college football odds from The Odds API (americanfootball_ncaaf), for the price-based tests
that worked in the NFL (soft books vs sharp, moneyline vs spread, opener drift, line shopping).

Snapshots on every day within 8 days of an FBS game: 16:10 and 23:10 UTC (Sunday 23:10 catches the openers),
plus Saturday (UTC) 13:10 / 19:10 and Sunday 01:10 / 03:10 for game-day closes. ~380 snapshots a season.
  plan cfb     : regions us,us2, markets h2h,spreads,totals   (60 credits per snapshot)
  plan cfb_pin : Pinnacle only, same markets                   (30 credits per snapshot)
Files: data/historical_odds/cfb[/_pin]/cfb_odds_<season>.csv.gz (+ done_<season>.txt, resumable).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nflpred import odds_history as NH
from . import data as D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "historical_odds"
URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_ncaaf/odds"
FIELDS = ["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "last_update",
          "ml_home", "ml_away", "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price",
          "tot_point", "tot_over_price", "tot_under_price"]
PLANS = {"cfb": ("us,us2", "cfb"), "cfb_pin": ("bookmakers:pinnacle", "cfb_pin"), "cfb_deriv": (None, "cfb_deriv")}
EVENT_URL = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_ncaaf/events/{eid}/odds"
DERIV_MARKETS = "spreads_h1,totals_h1,team_totals"
DERIV_BOOKS = "pinnacle,draftkings,fanduel,espnbet,betmgm,williamhill_us,betrivers,fanatics,hardrockbet,ballybet"   # 10 books = 1 region
DERIV_FIELDS = ["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "market", "name",
                "description", "point", "price", "last_update"]


def backfill_derivatives(seasons, key, reserve=1000, dry_run=False, getter=None):
    """First-half spreads/totals and team totals via the per-EVENT historical endpoint (from May 2023): FBS games that
    Pinnacle listed (cfb_pin files), two snapshots each: 48 h before kickoff and 75 min before (the close).
    30 credits per call (3 markets x 1 region-equivalent of 10 books)."""
    import urllib.error, urllib.parse, urllib.request
    out_dir = OUT / "cfb_deriv"
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = 30
    total = {"calls": 0, "credits": 0, "fetched": 0, "rows": 0, "remaining": None, "stopped": None, "plan": "cfb_deriv"}
    for s in seasons:
        f = OUT / "cfb_pin" / f"cfb_odds_{s}.csv.gz"
        if not f.exists() or s < 2023:
            continue
        ev = pd.read_csv(f, usecols=["event_id", "commence_time", "home", "away"]).drop_duplicates("event_id", keep="last")
        ev = ev[pd.to_datetime(ev.commence_time, utc=True) >= pd.Timestamp("2023-05-03", tz="UTC")]
        done_path = out_dir / f"done_{s}.txt"
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = []
        for r in ev.itertuples():
            kick = pd.Timestamp(r.commence_time).to_pydatetime()
            if kick > datetime.now(timezone.utc):
                continue
            for t in (kick - timedelta(hours=48), kick - timedelta(minutes=75)):
                tag = f"{r.event_id}|{t.isoformat()}"
                if tag not in done:
                    todo.append((r, t, tag))
        total["calls"] += len(todo); total["credits"] += len(todo) * cost
        print(f"{s}: {len(todo)} event-snapshots (~{len(todo) * cost:,} credits)", flush=True)
        if dry_run:
            continue
        for r, t, tag in todo:
            if NH._out_of_time():
                total["stopped"] = "time limit (re-run to resume)"; return total
            if total["remaining"] is not None and total["remaining"] - cost < reserve:
                total["stopped"] = f"reserve of {reserve} credits reached"; return total
            q = urllib.parse.urlencode({"apiKey": key, "bookmakers": DERIV_BOOKS, "markets": DERIV_MARKETS, "oddsFormat": "american",
                                        "date": t.strftime("%Y-%m-%dT%H:%M:%SZ")})
            try:
                if getter:
                    payload, remaining = getter(r.event_id, t)
                else:
                    with urllib.request.urlopen(f"{EVENT_URL.format(eid=r.event_id)}?{q}", timeout=30) as resp:
                        remaining = resp.headers.get("x-requests-remaining")
                        remaining = float(remaining) if remaining is not None else None
                        payload = json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise
                payload, remaining = {"data": {}}, total["remaining"]
            total["remaining"] = remaining
            d = payload.get("data") or {}
            rr = [{"requested_ts": t.isoformat(), "snapshot_ts": payload.get("timestamp"), "event_id": r.event_id,
                   "commence_time": r.commence_time, "home": r.home, "away": r.away, "book": bk.get("key"), "market": m.get("key"),
                   "name": o.get("name"), "description": o.get("description"), "point": o.get("point"), "price": o.get("price"),
                   "last_update": m.get("last_update") or bk.get("last_update")}
                  for bk in d.get("bookmakers", []) for m in bk.get("markets", []) for o in m.get("outcomes", [])]
            if rr:
                NH._append(out_dir / f"cfb_deriv_{s}.csv.gz", rr, DERIV_FIELDS)
            with done_path.open("a") as fh:
                fh.write(tag + "\n")
            total["fetched"] += 1; total["rows"] += len(rr)
            if total["fetched"] % 100 == 0:
                print(f"  {total['fetched']} fetched, {total['rows']:,} rows, {remaining} credits left", flush=True)
    return total


def plan(season: int, now: datetime | None = None) -> list[datetime]:
    g = D.games([season])
    g = g[(g.home_div == "fbs") | (g.away_div == "fbs")]
    kicks = sorted(t.to_pydatetime() for t in g["start"].dropna())
    now = now or datetime.now(timezone.utc)
    if not kicks:
        return []
    out = set()
    day = (kicks[0] - timedelta(days=8)).replace(hour=0, minute=0, second=0, microsecond=0)
    last = kicks[-1]
    import bisect
    while day <= last:
        hours = [16, 23]
        if day.weekday() == 5:
            hours += [13, 19]
        if day.weekday() == 6:
            hours += [1, 3]
        for h in hours:
            t = day + timedelta(hours=h, minutes=10)
            i = bisect.bisect_left(kicks, t)           # next kickoff after t
            if i < len(kicks) and kicks[i] - t <= timedelta(days=8) and t < now - timedelta(hours=1):
                out.add(t)
        day += timedelta(days=1)
    return sorted(out)


def rows(payload: dict, requested: datetime) -> list[dict]:
    out = []
    for ev in payload.get("data", []):
        for bk in ev.get("bookmakers", []):
            mk = {m["key"]: m for m in bk.get("markets", [])}
            r = {"requested_ts": requested.isoformat(), "snapshot_ts": payload.get("timestamp"), "event_id": ev.get("id"),
                 "commence_time": ev["commence_time"], "home": ev["home_team"], "away": ev["away_team"],
                 "book": bk.get("key"), "last_update": bk.get("last_update")}
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
            out.append(r)
    return out


def backfill(seasons, key, plan_name="cfb", reserve=1000, dry_run=False, fetcher=None):
    regions, sub = PLANS[plan_name]
    n_reg = 1 if regions.startswith("bookmakers:") else len(regions.split(","))
    cost = 10 * 3 * n_reg
    out_dir = OUT / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    NH.URL, url0 = URL, NH.URL      # reuse the NFL fetcher against the NCAAF endpoint
    fetcher = fetcher or NH.fetch
    total = {"snapshots": 0, "credits": 0, "fetched": 0, "rows": 0, "remaining": None, "stopped": None, "plan": plan_name}
    try:
        for s in seasons:
            done_path = out_dir / f"done_{s}.txt"
            done = set(done_path.read_text().split()) if done_path.exists() else set()
            todo = [t for t in plan(s) if t.isoformat() not in done]
            total["snapshots"] += len(todo)
            total["credits"] += len(todo) * cost
            print(f"{s}: {len(todo)} snapshots (~{len(todo) * cost:,} credits)", flush=True)
            if dry_run:
                continue
            for t in todo:
                if NH._out_of_time():
                    total["stopped"] = "time limit (re-run to resume)"
                    return total
                if total["remaining"] is not None and total["remaining"] - cost < reserve:
                    total["stopped"] = f"reserve of {reserve} credits reached"
                    return total
                payload, remaining = fetcher(key, t, regions, "h2h,spreads,totals")
                total["remaining"] = remaining
                rr = rows(payload, t)
                if rr:
                    NH._append(out_dir / f"cfb_odds_{s}.csv.gz", rr, FIELDS)
                with done_path.open("a") as f:
                    f.write(t.isoformat() + "\n")
                total["fetched"] += 1
                total["rows"] += len(rr)
                if total["fetched"] % 50 == 0:
                    print(f"  {total['fetched']} fetched, {total['rows']:,} rows, {remaining} credits left", flush=True)
    finally:
        NH.URL = url0
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2025")
    ap.add_argument("--plan", default="cfb", choices=list(PLANS))
    ap.add_argument("--reserve", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    seasons = NH._seasons(a.seasons)
    key = os.environ.get("ODDS_API_KEY", "")
    if not a.dry_run and not key:
        raise SystemExit("ODDS_API_KEY not set")
    if a.plan == "cfb_deriv":
        res = backfill_derivatives(sorted(seasons, reverse=True), key, a.reserve, a.dry_run)
    else:
        res = backfill(sorted(seasons, reverse=True), key, a.plan, a.reserve, a.dry_run)
    print(json.dumps(res))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### CFB historical odds: {a.plan} {'(dry run)' if a.dry_run else ''}\n\n```\n{json.dumps(res, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
