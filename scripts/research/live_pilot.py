"""NFL live (in-game) moneyline pilot, 2022-25. PRE-DECLARED 2026-10-04 before looking at any in-game price.

Data: in-game rows of the hourly odds snapshots (data/historical_odds/dense, requested after kickoff), only quotes the book
updated within 3 minutes of the snapshot (fresh). Game state from nflverse play-by-play at the snapshot time (last play
with time_of_day <= snapshot, within 10 minutes). Win-probability model: nflfastR vegas_home_wp (uses the pregame line).
Market live probability: median no-vig home probability across fresh books (>= 2 books).

Questions (descriptive): log loss of the live market vs vegas_home_wp on the final result; calibration of the market.
Rules (1 bet per game, first qualifying snapshot, moneyline at the best price among Arizona books, 1 unit):
  L1 model gap: vegas_home_wp - market >= 0.06 -> home; <= -0.06 -> away.
  L2 same with 0.10.
  L3 fade early overreaction: quarters 1-2, market moved >= 0.15 from the last pregame consensus toward a team -> bet the other team.
  L4 back late underreaction: quarters 3-4, market moved >= 0.15 since the previous in-game snapshot... (needs consecutive in-game
     snapshots of the same game; reported only if >= 50 cases) -> bet in the direction of the move.
Pass for any follow-up: ROI > 0 with p < 0.05/4 (one-sided, binomial-like via mean/se of unit profits). Hourly snapshots are far
too coarse for live trading, so a pass would justify buying dense live data, not betting.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nflpred import data as ND  # noqa: E402

AZ = ["draftkings", "fanduel", "espnbet", "barstool", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]


def imp(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def main():
    fs = sorted(glob.glob(str(ROOT / "data/historical_odds/dense/nfl_odds_*.csv.gz")))
    O = pd.concat([pd.read_csv(f, usecols=["requested_ts", "snapshot_ts", "event_id", "commence_time", "home", "away", "book", "last_update", "ml_home", "ml_away"])
                   .assign(season=int(Path(f).name[9:13])) for f in fs], ignore_index=True)
    O["t"] = pd.to_datetime(O.requested_ts, utc=True); O["ko"] = pd.to_datetime(O.commence_time, utc=True)
    O["age_min"] = (pd.to_datetime(O.snapshot_ts, utc=True) - pd.to_datetime(O.last_update, utc=True)).dt.total_seconds() / 60
    O = O.dropna(subset=["ml_home", "ml_away"])
    a, b = imp(O.ml_home), imp(O.ml_away)
    O["p"] = a / (a + b)
    pre = O[O.t < O.ko].sort_values("t").groupby("event_id").apply(lambda d: d[d.t == d.t.max()].p.median()).rename("p_pre")
    L = O[(O.t > O.ko) & (O.t < O.ko + pd.Timedelta(hours=4)) & (O.age_min <= 3)]
    snap = L.groupby(["event_id", "t"]).agg(p_mkt=("p", "median"), n=("book", "nunique"), home=("home", "first"), away=("away", "first"),
                                             ko=("ko", "first"), season=("season", "first")).reset_index()
    snap = snap[snap.n >= 2]
    best = L[L.book.isin(AZ)].groupby(["event_id", "t"]).agg(best_home=("ml_home", "max"), best_away=("ml_away", "max")).reset_index()
    snap = snap.merge(best, on=["event_id", "t"], how="inner").join(pre, on="event_id")
    print("in-game snapshots with >= 2 fresh books:", len(snap), "games:", snap.event_id.nunique(), flush=True)
    sched = ND.load_schedules()
    s = sched[sched.result.notna()][["game_id", "season", "home_team", "away_team", "gameday", "result"]]
    snap = snap.merge(s, left_on=["season", "home", "away"], right_on=["season", "home_team", "away_team"])
    snap = snap[(pd.to_datetime(snap.gameday).dt.tz_localize("UTC") - snap.ko.dt.floor("D")).abs() <= pd.Timedelta(days=2)]
    cols = ["game_id", "time_of_day", "vegas_home_wp", "qtr", "game_seconds_remaining", "total_home_score", "total_away_score"]
    pbp = pd.concat([ND.load_pbp([y])[cols] for y in range(2022, 2026)])
    pbp = pbp.dropna(subset=["time_of_day", "vegas_home_wp"])
    pbp["tod"] = pd.to_datetime(pbp.time_of_day, utc=True, format="mixed")
    pbp = pbp.sort_values("tod")
    snap = snap.sort_values("t")
    M = pd.merge_asof(snap, pbp, left_on="t", right_on="tod", by="game_id", direction="backward", tolerance=pd.Timedelta(minutes=10))
    M = M.dropna(subset=["vegas_home_wp"])
    M = M[M.qtr <= 4]
    M["home_win"] = (M.result > 0).astype(float)
    M = M[M.result != 0]
    def ll(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-(M.home_win * np.log(p) + (1 - M.home_win) * np.log(1 - p)).mean())
    out = {"snapshots": int(len(M)), "games": int(M.game_id.nunique()),
           "log_loss_market": round(ll(M.p_mkt), 4), "log_loss_nflfastR": round(ll(M.vegas_home_wp), 4),
           "log_loss_avg": round(ll(0.5 * (M.p_mkt + M.vegas_home_wp)), 4),
           "brier_market": round(float(((M.p_mkt - M.home_win) ** 2).mean()), 4), "brier_nflfastR": round(float(((M.vegas_home_wp - M.home_win) ** 2).mean()), 4),
           "by_quarter": {int(q): {"n": int(len(g)), "ll_market": round(float(-(g.home_win * np.log(np.clip(g.p_mkt, 1e-4, 1)) + (1 - g.home_win) * np.log(np.clip(1 - g.p_mkt, 1e-4, 1))).mean()), 4),
                                   "ll_nflfastR": round(float(-(g.home_win * np.log(np.clip(g.vegas_home_wp, 1e-4, 1)) + (1 - g.home_win) * np.log(np.clip(1 - g.vegas_home_wp, 1e-4, 1))).mean()), 4)}
                          for q, g in M.groupby("qtr")}}
    bins = pd.cut(M.p_mkt, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    out["market_calibration"] = {str(k): {"n": int(len(g)), "pred": round(float(g.p_mkt.mean()), 3), "actual": round(float(g.home_win.mean()), 3)} for k, g in M.groupby(bins, observed=True)}
    def rule(name, mask, home_side):
        d = M[mask].copy(); d["bet_home"] = home_side[mask]
        d = d.sort_values("t").groupby("game_id").head(1)
        price = np.where(d.bet_home, d.best_home, d.best_away)
        won = np.where(d.bet_home, d.home_win == 1, d.home_win == 0)
        ok = (price >= -1000) & (price <= 1000)
        pnl = np.where(won, dec(price) - 1, -1.0)[ok]
        n = len(pnl)
        m, se = (pnl.mean(), pnl.std() / math.sqrt(n)) if n > 1 else (float("nan"), float("nan"))
        out[name] = {"bets": int(n), "win_rate": round(float(won[ok].mean()), 3) if n else None, "roi": round(float(m), 4) if n else None,
                     "roi_se": round(float(se), 4) if n > 1 else None, "p": round(0.5 * math.erfc((m / se) / math.sqrt(2)), 4) if n > 1 and se > 0 else None,
                     "avg_price": round(float(np.median(price[ok])), 0) if n else None}
        out[name]["pass"] = bool(n >= 50 and out[name]["roi"] > 0 and (out[name]["p"] or 1) < 0.05 / 4)
    gap = M.vegas_home_wp - M.p_mkt
    rule("L1 nflfastR gap >= 0.06", gap.abs() >= 0.06, gap > 0)
    rule("L2 nflfastR gap >= 0.10", gap.abs() >= 0.10, gap > 0)
    mv = M.p_mkt - M.p_pre
    rule("L3 fade early moves >= 0.15 (Q1-Q2)", (M.qtr <= 2) & (mv.abs() >= 0.15), mv < 0)
    M2 = M.sort_values("t").copy()
    M2["prev_p"] = M2.groupby("game_id").p_mkt.shift(1)
    M = M2
    mv2 = M.p_mkt - M.prev_p
    if ((M.qtr >= 3) & (mv2.abs() >= 0.15)).sum() >= 50:
        rule("L4 follow late moves >= 0.15 (Q3-Q4)", (M.qtr >= 3) & (mv2.abs() >= 0.15), mv2 > 0)
    else:
        out["L4 follow late moves >= 0.15 (Q3-Q4)"] = {"bets": int(((M.qtr >= 3) & (mv2.abs() >= 0.15)).sum()), "note": "fewer than 50 cases; not evaluated"}
    (ROOT / "output" / "research" / "live_pilot.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
