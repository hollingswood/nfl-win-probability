"""One-shot holdout of the three frozen CFB rules (output/research/cfb/frozen_rules.json)."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
import alt_models as AM, fast
from cfbpred import data as D

AM.SEASONS = range(2014, 2027)
OUT = ROOT / "output/research/cfb"

def features():
    g, tal, ret = fast.prep()
    adv = AM.adv_long(); F = AM.adjusted(g, adv)
    base = fast.run(g, tal, ret, last_seasons=(2014, 2026)).rename(columns={"m_pred": "margin_ridge"})
    G = g.merge(F, on="game_id", how="left").merge(base, on="game_id", how="left").merge(D.lines(range(2014, 2027)), on="game_id", how="left")
    tz = tal.copy(); tz["tz"] = tz.groupby("season").talent.transform(lambda x: (x - x.mean()) / x.std())
    for side in ("home", "away"):
        G = G.merge(tz[["season", "team", "tz"]].rename(columns={"team": side, "tz": f"tz_{side[0]}"}), on=["season", side], how="left")
        G = G.merge(ret.rename(columns={"team": side, "percentPPA": f"ret_{side[0]}"}), on=["season", side], how="left")
    G["v"] = -G.spread_close; G["vo"] = -G.spread_open; G["nh"] = (~G.neutral).astype(float)
    G["elo"] = (G.home_elo - G.away_elo) / 25.0
    for s in AM.STATS:
        G[f"{s}_d"] = G[f"{s}_h"] - G[f"{s}_a"]; G[f"{s}_s"] = G[f"{s}_h"] + G[f"{s}_a"]
    G["tz_d"] = G.tz_h.fillna(0) - G.tz_a.fillna(0); G["ret_d"] = G.ret_h.fillna(0.5) - G.ret_a.fillna(0.5)
    G.to_parquet(OUT / "features_all.parquet")
    return G

ALLF = sorted({"margin_ridge", "elo", "nh", "pts_d", "ppa_d", "sr_d", "expl_d", "rush_ppa_d", "pass_ppa_d", "line_yds_d",
               "stuff_d", "power_d", "second_lvl_d", "open_field_d", "sd_ppa_d", "pd_ppa_d", "tz_d", "ret_d"}) + ["week"]

def ridge(fit, cols, y, lam=1.0):
    X = np.column_stack([np.ones(len(fit))] + [fit[c] for c in cols]); A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, X.T @ y)

def pred(df, cols, b): return np.column_stack([np.ones(len(df))] + [df[c] for c in cols]) @ b

def binom_p(w, n, p0=0.5238):
    z = (w - n * p0) / math.sqrt(n * p0 * (1 - p0)); return 0.5 * math.erfc(z / math.sqrt(2))

def report(name, wins, losses, extra=""):
    n = wins + losses; r = wins / n if n else float("nan"); p = binom_p(wins, n) if n else 1
    return f"| {name} | {n} | {wins}-{losses} | {r:.3f} | {p:.3f} | {'PASS' if (r > 0.5238 and p < 0.0167) else 'fail'} | {extra} |"

def main():
    G = features()
    F = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.margin.notna()].dropna(subset=ALLF)
    out = ["# CFB frozen rules: one-shot holdout\n", "| rule | bets | W-L | cover/hit | p (vs 52.38%) | result | notes |", "|---|---|---|---|---|---|---|"]
    by_season = []
    # R1
    dev = F[F.season.between(2014, 2023)]; b = ridge(dev, ALLF, dev.margin.to_numpy())
    H = F[F.season.isin([2024, 2025]) & F.vo.notna() & F.v.notna()].copy(); H["p"] = pred(H, ALLF, b)
    gap = H.p - H.vo; m = np.abs(gap) >= 5; s = np.sign(gap[m]); cv = s * (H.margin[m] - H.vo[m]); mv = s * (H.v[m] - H.vo[m])
    out.append(report("R1 opener spread (|model−open| ≥ 5)", int((cv > 0).sum()), int((cv < 0).sum()),
                      f"line moved our way {np.mean(mv > 0):.0%} / against {np.mean(mv < 0):.0%}, mean {mv.mean():+.2f} pts; cover vs close {np.mean((s * (H.margin[m] - H.v[m])) > 0) / max(1e-9, np.mean((s * (H.margin[m] - H.v[m])) != 0)):.3f}"))
    for y in (2024, 2025):
        k = m & (H.season == y); ss = np.sign(gap[k]); c = ss * (H.margin[k] - H.vo[k])
        by_season.append(f"R1 {y}: {int(k.sum())} bets, cover {(c > 0).sum() / max(1, (c != 0).sum()):.3f}")
    # R2
    import lightgbm as lgb
    fit = F[F.season.between(2014, 2018) & F.v.notna()]; val = F[F.season.between(2019, 2021) & F.v.notna()]
    r = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=80, verbose=-1)
    r.fit(fit[ALLF + ["v"]], fit.margin - fit.v); thr = float(np.quantile(np.abs(r.predict(val[ALLF + ["v"]])), 0.8))
    dv = F[F.season.between(2014, 2021) & F.v.notna()]
    r2 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=80, verbose=-1)
    r2.fit(dv[ALLF + ["v"]], dv.margin - dv.v)
    H2 = F[F.season.between(2022, 2025) & F.v.notna()].copy(); rp = r2.predict(H2[ALLF + ["v"]])
    m2 = np.abs(rp) >= thr; c2 = np.sign(rp[m2]) * (H2.margin[m2] - H2.v[m2])
    out.append(report("R2 close residual (LightGBM, top-20% threshold)", int((c2 > 0).sum()), int((c2 < 0).sum()), f"threshold {thr:.2f} pts"))
    for y in range(2022, 2026):
        k = m2 & (H2.season == y).to_numpy(); c = np.sign(rp[k]) * (H2.margin.to_numpy()[k] - H2.v.to_numpy()[k])
        by_season.append(f"R2 {y}: {int(k.sum())} bets, cover {(c > 0).sum() / max(1, (c != 0).sum()):.3f}")
    # R3
    tc = ["pts_s", "plays_s", "ppa_s", "sr_s", "expl_s"]
    T = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.total.notna() & G.total_close.notna()].dropna(subset=tc)
    tf = T[T.season.between(2014, 2021)]; bt = np.linalg.lstsq(np.column_stack([np.ones(len(tf))] + [tf[c] for c in tc]), tf.total, rcond=None)[0]
    TH = T[T.season.between(2022, 2025)].copy(); tp = pred(TH, tc, bt); d = tp - TH.total_close; m3 = np.abs(d) >= 3
    c3 = np.sign(d[m3]) * (TH.total[m3] - TH.total_close[m3])
    out.append(report("R3 close totals (|pred−close| ≥ 3)", int((c3 > 0).sum()), int((c3 < 0).sum()), f"overs {np.mean(d[m3] > 0):.0%} of bets"))
    for y in range(2022, 2026):
        k = m3 & (TH.season == y); c = np.sign(d[k]) * (TH.total[k] - TH.total_close[k])
        by_season.append(f"R3 {y}: {int(k.sum())} bets, hit {(c > 0).sum() / max(1, (c != 0).sum()):.3f}")
    out += ["", "By season:", ""] + [f"- {x}" for x in by_season]
    (OUT / "holdout.md").write_text("\n".join(out) + "\n"); print("\n".join(out))

if __name__ == "__main__":
    main()
