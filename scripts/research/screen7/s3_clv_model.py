"""S3: predict the Tuesday -> close line move and bet only the top decile (pre-declared 2026-10-03).
Data: hourly odds 2022-2025 (US books + Pinnacle). Early snapshot = Tuesday 14:10 UTC of game week
(kickoff 3.5-7 days later); close = last snapshot before kickoff. Points are home spreads (negative = home favored).
Target: move = consensus home point at close - at Tuesday (in points; positive = market moved toward the away team).
Features known on Tuesday: model gap (walk-forward model margin + Tuesday home point), preseason gap (S6),
Pinnacle - consensus point gap, book dispersion, |spread|, total, week, Tuesday ML-implied vs spread-implied gap.
Model: ridge, walk-forward by season (train on all earlier seasons; first test season 2023).
Bet: top 10% of |predicted move| in each test season, side the move favors, at the best Tuesday price among
DraftKings/FanDuel/theScore; scored by CLV in points (our-way move) and ATS at that price.
Pass: one-sided p < 0.007 on mean CLV points AND ATS ROI not negative."""
import glob, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
MINE = {"draftkings", "fanduel", "espnbet"}

def dec(a):
    a = np.asarray(a, float); return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))

def load(sub):
    fs = sorted(glob.glob(str(ROOT / f"data/historical_odds/{sub}/nfl_odds_*.csv.gz")))
    O = pd.concat([pd.read_csv(f, usecols=["requested_ts", "commence_time", "home", "away", "book", "ml_home", "ml_away",
                                           "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price", "tot_point"]) for f in fs], ignore_index=True)
    O["t"] = pd.to_datetime(O.requested_ts, utc=True); O["ko"] = pd.to_datetime(O.commence_time, utc=True)
    O = O[(O.t < O.ko) & O.sp_home_point.notna()]
    O["home"] = O.home.replace(FIX); O["away"] = O.away.replace(FIX)
    return O

def main():
    O = load("dense"); P = load("dense_pin")
    O["hours"] = (O.ko - O.t).dt.total_seconds() / 3600
    # Tuesday 14:10 UTC snapshot of game week
    tue = O[(O.t.dt.weekday == 1) & (O.t.dt.hour == 14) & (O.hours.between(84, 170))]
    key = ["home", "away", "ko"]
    agg = lambda d: d.groupby(key).agg(pt=("sp_home_point", "median"), disp=("sp_home_point", "std"), tot=("tot_point", "median"),
                                       mlh=("ml_home", "median"), mla=("ml_away", "median")).reset_index()
    T = agg(tue)
    O["tmax"] = O.groupby(key).t.transform("max"); Cl = agg(O[O.t == O.tmax]).rename(columns={"pt": "pt_close"})
    ptue = P[(P.t.dt.weekday == 1) & (P.t.dt.hour == 14)].groupby(key).sp_home_point.median().rename("pin").reset_index()
    D = T.merge(Cl[key + ["pt_close"]], on=key).merge(ptue, on=key, how="left")
    # best Tuesday price at Tyler's books for each side at the consensus number
    mine = tue[tue.book.isin(MINE)]
    bh = mine.assign(d=dec(mine.sp_home_price)).sort_values(["sp_home_point", "d"], ascending=[False, False]).groupby(key).head(1)[key + ["sp_home_point", "sp_home_price"]]
    ba = mine.assign(d=dec(mine.sp_away_price)).sort_values(["sp_away_point", "d"], ascending=[False, False]).groupby(key).head(1)[key + ["sp_away_point", "sp_away_price"]]
    D = D.merge(bh, on=key, how="left").merge(ba, on=key, how="left")
    # games + model + preseason
    B = pd.read_parquet(ROOT / "data/screen_base.parquet")
    B["home_team"] = B.home_team.replace(FIX); B["away_team"] = B.away_team.replace(FIX)
    B["kd"] = pd.to_datetime(B.gameday).dt.date
    D["kd"] = D.ko.dt.tz_convert("America/New_York").dt.date
    D = D.merge(B[["game_id", "season", "week", "home_team", "away_team", "kd", "mu_model", "home_score", "away_score"]],
                left_on=["home", "away", "kd"], right_on=["home_team", "away_team", "kd"])
    W = pd.read_csv(ROOT / "data/research_futures/win_totals_2013_2026.csv")[["season", "team", "line_adj"]]
    W["team"] = W.team.replace(FIX)
    D = D.merge(W.rename(columns={"team": "home", "line_adj": "wt_h"}), on=["season", "home"], how="left").merge(
        W.rename(columns={"team": "away", "line_adj": "wt_a"}), on=["season", "away"], how="left")
    D["pre_margin"] = 2.48 + 2.085 * (D.wt_h - D.wt_a)
    TU = pd.read_parquet(ROOT / "data/screen_tuesday.parquet")[["game_id", "mu_tue", "qb_same"]]
    D = D.merge(TU, on="game_id", how="left")
    import os
    if os.environ.get("S3_TUESDAY", "1") == "1":     # Tuesday-information model; drop games with a starting-QB change
        D["mu_model"] = D.mu_tue
        D = D[D.qb_same.fillna(False).astype(bool)]
    D["move"] = D.pt_close - D.pt
    D["f_model"] = D.mu_model + D.pt                 # model home margin minus market home margin (home point = -margin)
    D["f_pre"] = np.where(D.week <= 8, D.pre_margin + D.pt, 0.0)
    D["f_pin"] = (D.pin - D.pt).fillna(0.0)
    ih = np.where(D.mlh > 0, 100 / (D.mlh + 100), -D.mlh / (-D.mlh + 100)); ia = np.where(D.mla > 0, 100 / (D.mla + 100), -D.mla / (-D.mla + 100))
    D["f_mlgap"] = (ih / (ih + ia)) - 0.5 + D.pt * 0.03   # crude: ML-implied vs spread-implied (≈3% per point near PK)
    D["f_disp"] = D.disp.fillna(0); D["f_abs"] = D.pt.abs(); D["f_tot"] = D.tot.fillna(D.tot.median())
    feats = ["f_model", "f_pre", "f_pin", "f_mlgap", "f_disp", "f_abs", "f_tot"]
    D = D.dropna(subset=["move", "sp_home_price", "sp_away_price"])
    out = {"games": int(len(D)), "by_season_n": D.groupby("season").size().to_dict(), "corr_with_move": {f: round(float(np.corrcoef(D[f], D.move)[0, 1]), 3) for f in feats}}
    res = []
    for test in (2023, 2024, 2025):
        tr, te = D[D.season < test], D[D.season == test].copy()
        X = np.column_stack([np.ones(len(tr))] + [tr[f] for f in feats]); A = X.T @ X + 5 * np.eye(X.shape[1]); A[0, 0] -= 5
        beta = np.linalg.solve(A, X.T @ tr.move.to_numpy())
        te["pred"] = np.column_stack([np.ones(len(te))] + [te[f] for f in feats]) @ beta
        cut = te.pred.abs().quantile(0.9)
        b = te[te.pred.abs() >= cut].copy()
        # predicted move > 0 = home point rises (-3 -> -1): the market moves toward the AWAY... no: the HOME side gets
        # fewer points later? A rising home point (-3 -> -1) means home is LESS favored at the close, so the home
        # bettor would get a better number later and the AWAY number (+3 -> +1) is better TODAY -> bet AWAY now.
        side_home = b.pred < 0
        clv = np.where(side_home, -b.move, b.move)   # points gained vs the close for the side we bet
        point = np.where(side_home, b.sp_home_point, b.sp_away_point); price = np.where(side_home, b.sp_home_price, b.sp_away_price)
        margin = (b.home_score - b.away_score).to_numpy()
        r = np.where(side_home, margin + point, -margin + point)
        pnl = np.where(r > 0, dec(price) - 1, np.where(r < 0, -1, 0))
        res.append(pd.DataFrame({"season": test, "clv_pts": clv, "pnl": pnl, "r": r}))
        out[f"beta_{test}"] = dict(zip(["const"] + feats, np.round(beta, 3).tolist()))
    R = pd.concat(res)
    m, se = R.clv_pts.mean(), R.clv_pts.std() / math.sqrt(len(R))
    out["top_decile"] = {"bets": int(len(R)), "mean_clv_pts": round(float(m), 3), "se": round(float(se), 3),
                         "p_clv": round(0.5 * math.erfc((m / se) / math.sqrt(2)), 4), "share_clv_positive": round(float((R.clv_pts > 0).mean()), 3),
                         "share_clv_negative": round(float((R.clv_pts < 0).mean()), 3),
                         "ats_cover": round(float((R.r > 0).sum() / max(1, (R.r != 0).sum())), 3), "roi": round(float(R.pnl.mean()), 4),
                         "roi_se": round(float(R.pnl.std() / math.sqrt(len(R))), 4),
                         "by_season": R.groupby("season").agg(n=("clv_pts", "size"), clv=("clv_pts", "mean"), roi=("pnl", "mean")).round(3).to_dict("index")}
    (ROOT / "output/research/screen7/s3_clv_model.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))

if __name__ == "__main__":
    main()
