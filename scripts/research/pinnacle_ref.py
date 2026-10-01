"""Pinnacle as the sharp reference for moneyline v2 (C4) and C2 -- descriptive measurement, 2024-25.

2024-25 overlaps the 2023-25 holdout already used once for the frozen candidates, so nothing here
is a fresh validation; thresholds are NOT tuned here (v2's 2% / C2's 3% are kept, only the fair
probability reference is swapped).

Q1: at the same snapshots (Friday 21:40 UTC run and ~75 min before kickoff) compare no-vig home
    probability from Pinnacle vs our sharp median vs all-book consensus as predictors of
    (a) the nflverse closing no-vig probability and our own last pre-kickoff snapshot,
    (b) game results (log loss). Plus Pinnacle coverage and quote staleness.
Q2: C2 / C4 at the Pinnacle snapshots with fair prob from sharp / Pinnacle / average of both.

Run: EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/pinnacle_ref.py
Writes output/research/pinnacle_ref.{md,json}.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edge_lab as E  # noqa: E402
import replay_early_lines as R  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "research"
SEASONS = (2024, 2025)
REFS = ("sharp", "pin", "avg")


def load_pinnacle() -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / "data" / "historical_odds" / "pinnacle" / "nfl_odds_*.csv.gz")))
    o = pd.concat([pd.read_csv(f).assign(season=int(Path(f).name[9:13])) for f in fs], ignore_index=True)
    o = o[o.book == "pinnacle"].copy()
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    o["stale_min"] = (pd.to_datetime(o.snapshot_ts, utc=True) - pd.to_datetime(o.last_update, utc=True)).dt.total_seconds() / 60
    return o


def requested_times() -> set:
    ts = set()
    for f in glob.glob(str(ROOT / "data" / "historical_odds" / "pinnacle" / "done_*.txt")):
        ts |= {pd.Timestamp(x.strip()) for x in Path(f).read_text().split() if x.strip()}
    return ts


def games() -> pd.DataFrame:
    wf = R.walk_forward()
    wf = wf[wf.season.isin(SEASONS)].copy()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in wf.itertuples()], utc=True)
    return wf


def pin_snapshots(wf) -> pd.DataFrame:
    o = R.match_games(load_pinnacle(), wf).merge(wf[["game_id", "kick"]], on="game_id")
    o = o[(o.requested_ts < o.kick) & o.ml_home.notna() & o.ml_away.notna()]
    o["p_pin"] = E._nv(o.ml_home.values, o.ml_away.values)
    return o.groupby(["game_id", "requested_ts"]).agg(p_pin=("p_pin", "median"), pin_stale_min=("stale_min", "max")).reset_index()


def staleness_main(wf) -> dict:
    o = R.match_games(R.load_odds(), wf)
    o = o[o.season.isin(SEASONS)]
    s = (pd.to_datetime(o.snapshot_ts, utc=True) - pd.to_datetime(o.last_update, utc=True)).dt.total_seconds() / 60
    return {b: round(float(s[o.book == b].median()), 1) for b in ("lowvig", "betonlineag", "draftkings", "fanduel")}


def snap_type(ts: pd.Series, hours_before: pd.Series) -> pd.Series:
    fri = (ts.dt.dayofweek == 4) & (ts.dt.hour == 21) & (ts.dt.minute == 40)
    return np.where(fri, "fri2140", np.where(hours_before <= 2.0, "t75", "other"))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def q1(snaps: pd.DataFrame, close: pd.DataFrame) -> dict:
    s = snaps.merge(close[["game_id", "p_close_all", "p_close_sharp", "p_close_pin"]], on="game_id", how="left")
    s["y"] = np.where(s.home_score > s.away_score, 1.0, np.where(s.home_score < s.away_score, 0.0, np.nan))
    out = {}
    for typ in ("fri2140", "other", "t75"):
        d = s[(s.typ == typ) & (s.hours_before <= 7 * 24)].sort_values("requested_ts")
        rec = {"games_with_snapshot": int(d.game_id.nunique()),
               "pin_missing_pct_latest": round(float(d.groupby("game_id").tail(1).p_pin.isna().mean() * 100), 2),
               "sharp_missing_pct_latest": round(float(d.groupby("game_id").tail(1).p_sharp.isna().mean() * 100), 2)}
        d = d[d.p_pin.notna() & d.p_sharp.notna() & d.p_cons.notna()].groupby("game_id").tail(1)   # one snapshot per game
        rec["games"] = int(len(d))
        # two-predictor: (nflverse close - all-book) on (pin - all-book, sharp - all-book), no intercept
        X = np.column_stack([d.p_pin - d.p_cons, d.p_sharp - d.p_cons])
        yv = (d.vegas_home_prob - d.p_cons).values
        ok = np.isfinite(yv) & np.isfinite(X).all(1)
        if ok.sum() > 20:
            X, yv = X[ok], yv[ok]
            beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
            res = yv - X @ beta
            cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(yv) - 2)
            rec["joint_vs_nflverse_close"] = {"b_pin": round(float(beta[0]), 3), "se_pin": round(float(np.sqrt(cov[0, 0])), 3),
                                              "b_sharp": round(float(beta[1]), 3), "se_sharp": round(float(np.sqrt(cov[1, 1])), 3)}
        preds = {"pinnacle": d.p_pin, "sharp_median": d.p_sharp, "all_book": d.p_cons, "avg_pin_sharp": d.p_avg}
        targets = {"nflverse_close": d.vegas_home_prob, "own_close_all": d.p_close_all, "own_close_sharp": d.p_close_sharp,
                   "own_close_pinnacle": d.p_close_pin}
        for tn, tv in targets.items():
            ok = tv.notna()
            rec[f"rmse_vs_{tn}"] = {k: round(float(np.sqrt(((v[ok] - tv[ok]) ** 2).mean())) * 100, 3) for k, v in preds.items()}
            rec[f"rmse_vs_{tn}"]["n"] = int(ok.sum())
            # incremental info: regress (target - sharp) on (pin - sharp). slope 1 => Pinnacle is the better estimate
            x, yv = (d.p_pin - d.p_sharp)[ok].values, (tv - d.p_sharp)[ok].values
            if len(x) > 10 and x.std() > 0:
                b = (x * yv).sum() / (x * x).sum()
                res = yv - b * x
                se = math.sqrt((res ** 2).sum() / (len(x) - 1) / (x * x).sum())
                rec[f"slope_{tn}"] = {"b": round(b, 3), "se": round(se, 3)}
        g = d[d.y.notna()]
        rec["logloss"] = {k: round(float(ll(v[g.index], g.y).mean()), 5) for k, v in preds.items()}
        rec["logloss"]["nflverse_close"] = round(float(ll(g.vegas_home_prob, g.y).mean()), 5)
        rec["logloss"]["n"] = int(len(g))
        diff = ll(g.p_pin, g.y) - ll(g.p_sharp, g.y)
        rec["logloss_pin_minus_sharp"] = {"mean": round(float(diff.mean()), 5), "se": round(float(diff.std(ddof=1) / math.sqrt(len(diff))), 5)}
        diff = ll(g.p_pin, g.y) - ll(g.p_cons, g.y)
        rec["logloss_pin_minus_allbook"] = {"mean": round(float(diff.mean()), 5), "se": round(float(diff.std(ddof=1) / math.sqrt(len(diff))), 5)}
        rec["mean_abs_pin_minus_sharp_pp"] = round(float((d.p_pin - d.p_sharp).abs().mean() * 100), 3)
        out[typ] = rec
    return out


def bet_stats(b: pd.DataFrame, close: pd.DataFrame) -> dict:
    s = E.stats(b, "ml")
    if not s.get("bets"):
        return s
    b = b.merge(close[["game_id", "p_close_all", "p_close_sharp", "p_close_pin"]], on="game_id", how="left")
    for c in ("p_close_all", "p_close_sharp", "p_close_pin"):
        p = np.where(b.side == "home", b[c], 1 - b[c])
        v = (b.ml_dec * p - 1).dropna()
        v = v[np.isfinite(v)]
        s[f"clv_{c[2:]}"] = {"mean": round(float(v.mean()), 4), "t": round(float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))), 2) if len(v) > 2 else 0,
                              "beat": round(float((v > 0).mean()), 3), "n": int(len(v))}
    s["snap_mix"] = b.typ.value_counts().to_dict()
    s["by_season"] = {int(y): {"bets": len(g), "clv": round(float(g.ml_clv.mean()), 4), "roi": round(float(g.ml_pnl.mean()), 4)}
                      for y, g in b.groupby("season")}
    return s


def with_ref(t: pd.DataFrame, ref: str) -> pd.DataFrame:
    tt = t.copy()
    tt["p_sharp"] = {"sharp": t.p_sharp, "pin": t.p_pin, "avg": t.p_avg}[ref]
    tt["p_sharp_side"] = np.where(tt.side == "home", tt.p_sharp, 1 - tt.p_sharp)
    return tt


def main():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("set EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES (descriptive study on the already-used holdout)")
    wf = games()
    t = E.load(holdout=True)
    t = t[t.season.isin(SEASONS)].copy()
    pin = pin_snapshots(wf)
    close = E.closing_fair(SEASONS)
    lastpin = pin.sort_values("requested_ts").merge(wf[["game_id", "kick"]], on="game_id")
    lastpin = lastpin[(lastpin.kick - lastpin.requested_ts) <= pd.Timedelta(hours=2)].groupby("game_id").tail(1)
    close = close.merge(lastpin[["game_id", "p_pin"]].rename(columns={"p_pin": "p_close_pin"}), on="game_id", how="left")

    t = t.merge(pin, on=["game_id", "requested_ts"], how="left")
    t["p_avg"] = (t.p_pin + t.p_sharp) / 2
    t["typ"] = snap_type(t.requested_ts, t.hours_before)
    snap_cols = ["game_id", "requested_ts", "season", "typ", "hours_before", "p_cons", "p_sharp", "p_pin", "p_avg", "n_books",
                 "vegas_home_prob", "home_score", "away_score", "eligible", "pin_stale_min"]
    snaps = t[snap_cols].drop_duplicates(["game_id", "requested_ts"])

    # ---- coverage: main-file snapshots at the times Pinnacle was requested
    req = requested_times()
    at_req = snaps[snaps.requested_ts.isin(req)]
    cov = {"requested_times": len(req), "main_snapshots_at_those_times": int(len(at_req)),
           "pinnacle_present": int(at_req.p_pin.notna().sum()),
           "pinnacle_missing_pct": round(float(at_req.p_pin.isna().mean() * 100), 2),
           "sharp_missing_pct": round(float(at_req.p_sharp.isna().mean() * 100), 2),
           "by_type": {k: {"n": int(len(g)), "pin_missing_pct": round(float(g.p_pin.isna().mean() * 100), 2),
                           "sharp_missing_pct": round(float(g.p_sharp.isna().mean() * 100), 2)} for k, g in at_req.groupby("typ")},
           "games": int(snaps.game_id.nunique()),
           "games_with_any_pinnacle": int(snaps[snaps.p_pin.notna()].game_id.nunique()),
           "sharp_books_actually_in_data": sorted(set(R.load_odds().query("season in @SEASONS").book) & E.SHARP),
           "median_quote_age_min": {"pinnacle": round(float(snaps.pin_stale_min.median()), 1), **staleness_main(wf)}}

    res = {"coverage": cov, "q1": q1(snaps, close)}

    # ---- Q2: C2 / C4 at Pinnacle snapshots, fair prob from each reference (same universe for all three)
    u = t[t.p_pin.notna() & t.p_sharp.notna()]
    q2 = {}
    for ref in REFS:
        C = E.candidates(with_ref(u, ref))
        for cid in ("C2_ml_sharp_dog", "C4_ml_sharp_and_model"):
            q2[f"{cid}|{ref}"] = bet_stats(C[cid][0], close)
    # reference rows: original rules on ALL 2024-25 snapshots (not just Pinnacle times)
    C = E.candidates(t)
    for cid in ("C2_ml_sharp_dog", "C4_ml_sharp_and_model"):
        q2[f"{cid}|sharp_all_snapshots"] = bet_stats(C[cid][0], close)
    # overlap between sharp- and Pinnacle-selected C4 bets
    a = E.candidates(with_ref(u, "sharp"))["C4_ml_sharp_and_model"][0]
    b = E.candidates(with_ref(u, "pin"))["C4_ml_sharp_and_model"][0]
    ka, kb = set(zip(a.game_id, a.side)), set(zip(b.game_id, b.side))
    q2["C4_overlap"] = {"sharp_only": len(ka - kb), "pin_only": len(kb - ka), "both": len(ka & kb)}
    for lab, keys, src in (("C4_sharp_only_bets", ka - kb, a), ("C4_pin_only_bets", kb - ka, b)):
        sel = src[[k in keys for k in zip(src.game_id, src.side)]]
        q2[lab] = bet_stats(sel, close)
    res["q2"] = q2

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pinnacle_ref.json").write_text(json.dumps(res, indent=2, default=str))
    write_md(res)
    print(json.dumps(res, indent=1, default=str))


def write_md(res):
    L = ["# Pinnacle as the sharp reference (2024-25, descriptive)", "",
         "2024-25 overlaps the 2023-25 holdout already used once; this is a measurement study, not a validation. "
         "No thresholds were tuned. Script: `scripts/research/pinnacle_ref.py`.", "", "## Coverage", ""]
    c = res["coverage"]
    L += [f"- Sharp books actually present in the main odds files: {', '.join(c['sharp_books_actually_in_data'])} "
          "(circasports and bookmaker never appear, so 'sharp median' = BetOnline/LowVig, one book group).",
          f"- All main snapshots at Pinnacle request times (any horizon, incl. lines weeks ahead; per-game coverage is in Q1 headers): {c['main_snapshots_at_those_times']}; Pinnacle missing "
          f"{c['pinnacle_missing_pct']}% (sharp missing {c['sharp_missing_pct']}%). By type: "
          + "; ".join(f"{k}: n={v['n']}, pin missing {v['pin_missing_pct']}%, sharp missing {v['sharp_missing_pct']}%" for k, v in c["by_type"].items()),
          f"- Median quote age (snapshot - last_update, min): {c['median_quote_age_min']}", "", "## Q1: accuracy at the same snapshots", ""]
    for typ, r in res["q1"].items():
        L += [f"### {typ} (games={r['games']}; latest snapshot <=7d before kick: Pinnacle missing {r['pin_missing_pct_latest']}%, "
              f"sharp missing {r['sharp_missing_pct_latest']}% of {r['games_with_snapshot']} games)", "",
              f"Joint regression (nflverse close - all-book) on (pin - all-book, sharp - all-book): {r.get('joint_vs_nflverse_close')}", "", "| target (RMSE, pp) | pinnacle | sharp median | all-book | avg pin+sharp | n | slope b (se) |",
              "|---|---|---|---|---|---|---|"]
        for tn in ("nflverse_close", "own_close_all", "own_close_sharp", "own_close_pinnacle"):
            x = r[f"rmse_vs_{tn}"]
            sl = r.get(f"slope_{tn}", {})
            L.append(f"| {tn} | {x['pinnacle']} | {x['sharp_median']} | {x['all_book']} | {x['avg_pin_sharp']} | {x['n']} | {sl.get('b')} ({sl.get('se')}) |")
        lg = r["logloss"]
        L += ["", f"Log loss on results (n={lg['n']}): pinnacle {lg['pinnacle']}, sharp {lg['sharp_median']}, all-book {lg['all_book']}, "
              f"avg {lg['avg_pin_sharp']}, nflverse close {lg['nflverse_close']}. Pin - sharp = {r['logloss_pin_minus_sharp']['mean']} "
              f"± {r['logloss_pin_minus_sharp']['se']}; pin - all-book = {r['logloss_pin_minus_allbook']['mean']} ± {r['logloss_pin_minus_allbook']['se']}. "
              f"Mean |pin - sharp| = {r['mean_abs_pin_minus_sharp_pp']} pp.", "",
              "Slope b: regression of (target - sharp) on (pinnacle - sharp) through the origin; b=1 means the target moved all the way "
              "to Pinnacle, b=0 means Pinnacle adds nothing beyond the sharp median. 'other' = a Pinnacle request time that was "
              "another game's T-75 (e.g. Sunday 15:45 UTC for a 20:20 kickoff). At t75 the own-close targets are the same snapshot, so they are not forecasts.", ""]
    L += ["## Q2: C2 / C4 at Pinnacle snapshots, by fair-probability reference", "",
          "| rule / ref | bets | CLV nflverse (t) | beat | CLV own close all (t) | CLV own close sharp (t) | CLV own close Pinnacle (t) | ROI ± SE | snaps |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in res["q2"].items():
        if k == "C4_overlap" or not s.get("bets"):
            continue
        f = lambda d: f"{d['mean']:+.4f} ({d['t']:+.2f})"
        L.append(f"| {k} | {s['bets']} | {s['clv']:+.4f} ({s['clv_t']:+.2f}) | {s['beat_close']:.2f} | {f(s['clv_close_all'])} | "
                 f"{f(s['clv_close_sharp'])} | {f(s['clv_close_pin'])} | {s['roi']:+.3f} ± {s['roi_se']:.3f} | {s['snap_mix']} |")
    L += ["", "CLV vs own close Pinnacle is circular for the Pinnacle reference (and vs own close sharp for the sharp reference), "
          "and at t75 the own close IS the bet snapshot. CLV vs nflverse close is the neutral yardstick.", "",
          f"C4 overlap (game, side) between sharp- and Pinnacle-referenced selections: {res['q2']['C4_overlap']}", ""]
    (OUT / "pinnacle_ref.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
