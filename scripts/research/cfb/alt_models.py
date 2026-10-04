"""Alternative college football rating models, compared on DEVELOPMENT seasons only.

Close-line tests: fit 2014-2018, validate 2019-2021 (2022-25 untouched).
Opener tests: fit 2021-2022, validate 2023 (2024-25 untouched).

Walk-forward, opponent-adjusted ratings for many per-game stats (CFBD advanced box score, garbage time
excluded): for each stat, team_offense_stat = mu + O_team + D_opponent + h*home, ridge-shrunk toward
0.6 x last season. Every game is described only by ratings fit on earlier week slots.

Models compared (each mapped to a margin by a regression fit on 2014-18):
  margin_ridge  - points margin ridge (baseline, fast.py)
  elo           - CFBD pregame Elo difference
  points        - offense/defense points ratings
  ppa           - expected points added per play
  success       - success rate
  explosive     - explosiveness (PPA on successful plays)
  rush_pass     - rushing and passing PPA separately
  line          - line yards, stuff rate, power success, second-level and open-field yards
  downs         - standard-downs and passing-downs PPA
  talent_ret    - 247 talent composite + returning production (preseason only)
  all_linear    - every feature, ridge
  all_gbm       - every feature, LightGBM
Writes output/research/cfb/alt_models.md and alt_features.parquet.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from cfbpred import data as D  # noqa: E402
import fast  # noqa: E402

OUT = ROOT / "output" / "research" / "cfb"
SEASONS = range(2014, 2024)        # features needed for dev only (opener dev runs to 2023)
RAW = ROOT / "data" / "cfb" / "raw"


def adv_long():
    """Per team-game advanced stats (offense side; defense side is the opponent's offense)."""
    rows = []
    for y in SEASONS:
        for st in ("regular", "postseason"):
            p = RAW / f"stats_game_advanced_{y}_{st}.json.gz"
            if not p.exists():
                continue
            for r in json.load(gzip.open(p, "rt")):
                o = r.get("offense") or {}
                g = lambda d, k: (d or {}).get(k)
                rows.append({"game_id": r["gameId"], "team": r["team"], "ppa": o.get("ppa"), "sr": o.get("successRate"),
                             "expl": o.get("explosiveness"), "plays": o.get("plays"),
                             "rush_ppa": g(o.get("rushingPlays"), "ppa"), "pass_ppa": g(o.get("passingPlays"), "ppa"),
                             "line_yds": o.get("lineYards"), "stuff": o.get("stuffRate"), "power": o.get("powerSuccess"),
                             "second_lvl": o.get("secondLevelYards"), "open_field": o.get("openFieldYards"),
                             "sd_ppa": g(o.get("standardDowns"), "ppa"), "pd_ppa": g(o.get("passingDowns"), "ppa")})
    return pd.DataFrame(rows)


STATS = ["pts", "ppa", "sr", "expl", "plays", "rush_ppa", "pass_ppa", "line_yds", "stuff", "power", "second_lvl",
         "open_field", "sd_ppa", "pd_ppa"]


def adjusted(g: pd.DataFrame, adv: pd.DataFrame, lam=5.0) -> pd.DataFrame:
    """Walk-forward opponent-adjusted expectations of each stat for both teams of every game."""
    teams = sorted(set(g.home) | set(g.away)); ix = {t: i for i, t in enumerate(teams)}; k = len(teams)
    # long table: one row per (game, offense team)
    L = []
    for r in g.itertuples():
        L.append((r.game_id, r.season, r.slot, r.home, r.away, 0 if r.neutral else 1, r.home_pts))
        L.append((r.game_id, r.season, r.slot, r.away, r.home, 0, r.away_pts))
    L = pd.DataFrame(L, columns=["game_id", "season", "slot", "team", "opp", "home", "pts"])
    L = L.merge(adv, on=["game_id", "team"], how="left")
    out = {}
    last = {s: None for s in STATS}
    for season in SEASONS:
        S = L[L.season == season]
        G = g[g.season == season]
        cur_last = {}
        for stat in STATS:
            prior = np.zeros(2 * k + 2)
            if last[stat] is not None:
                prior[:2 * k] = 0.6 * last[stat]
            A = np.zeros((2 * k + 2, 2 * k + 2)); A[np.arange(2 * k), np.arange(2 * k)] = lam
            A[2 * k, 2 * k] = A[2 * k + 1, 2 * k + 1] = 1e-6
            b = np.zeros(2 * k + 2); R = prior.copy()
            mean0 = S[stat].mean() if stat in S else np.nan
            if np.isnan(mean0):
                continue
            R[2 * k] = mean0
            for sl in sorted(G.slot.unique()):
                gg = G[G.slot == sl]
                h = gg.home.map(ix).to_numpy(); a = gg.away.map(ix).to_numpy(); nh = (~gg.neutral).to_numpy(float)
                eh = R[2 * k] + R[h] + R[k + a] + nh * R[2 * k + 1]
                ea = R[2 * k] + R[a] + R[k + h]
                for gid, x1, x2 in zip(gg.game_id, eh, ea):
                    out.setdefault(gid, {})[f"{stat}_h"] = x1
                    out[gid][f"{stat}_a"] = x2
                done = S[(S.slot == sl) & S[stat].notna()]
                if done.empty:
                    continue
                ti = done.team.map(ix).to_numpy(); oi = done.opp.map(ix).to_numpy()
                y = done[stat].to_numpy(float)
                for t, o, hm, yy in zip(ti, oi, done.home.to_numpy(float), y):
                    idx = [t, k + o, 2 * k, 2 * k + 1]; x = np.array([1.0, 1.0, 1.0, hm])
                    resid = yy - x @ prior[idx]
                    A[np.ix_(idx, idx)] += np.outer(x, x); b[idx] += x * resid
                if len(done) > 30:
                    R = prior + np.linalg.solve(A, b)
            cur_last[stat] = R[:2 * k]
        last.update(cur_last)
        print("adjusted", season, flush=True)
    F = pd.DataFrame.from_dict(out, orient="index"); F.index.name = "game_id"
    return F.reset_index()


def main():
    g, tal, ret = fast.prep()
    g = g[g.season.between(min(SEASONS), max(SEASONS))].copy()
    adv = adv_long()
    F = adjusted(g, adv)
    base = fast.run(g, tal, ret, last_seasons=(min(SEASONS), max(SEASONS))).rename(columns={"m_pred": "margin_ridge"})
    lines = D.lines(SEASONS)
    G = g.merge(F, on="game_id", how="left").merge(base, on="game_id", how="left").merge(lines, on="game_id", how="left")
    tz = tal.copy(); tz["tz"] = tz.groupby("season").talent.transform(lambda x: (x - x.mean()) / x.std())
    for side in ("home", "away"):
        G = G.merge(tz[["season", "team", "tz"]].rename(columns={"team": side, "tz": f"tz_{side[0]}"}), on=["season", side], how="left")
        G = G.merge(ret.rename(columns={"team": side, "percentPPA": f"ret_{side[0]}"}), on=["season", side], how="left")
    G = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.margin.notna()].copy()
    G["v"] = -G.spread_close; G["vo"] = -G.spread_open; G["nh"] = (~G.neutral).astype(float)
    G["elo"] = (G.home_elo - G.away_elo) / 25.0
    for s in STATS:
        G[f"{s}_d"] = G[f"{s}_h"] - G[f"{s}_a"]; G[f"{s}_s"] = G[f"{s}_h"] + G[f"{s}_a"]
    G["tz_d"] = G.tz_h.fillna(0) - G.tz_a.fillna(0); G["ret_d"] = G.ret_h.fillna(0.5) - G.ret_a.fillna(0.5)
    G.to_parquet(OUT / "alt_features.parquet")
    sets = {
        "margin_ridge": ["margin_ridge"], "elo": ["elo", "nh"], "points": ["pts_d", "nh"], "ppa": ["ppa_d", "nh"],
        "success": ["sr_d", "nh"], "explosive": ["expl_d", "nh"], "rush_pass": ["rush_ppa_d", "pass_ppa_d", "nh"],
        "line": ["line_yds_d", "stuff_d", "power_d", "second_lvl_d", "open_field_d", "nh"],
        "downs": ["sd_ppa_d", "pd_ppa_d", "nh"], "talent_ret": ["tz_d", "ret_d", "nh"],
    }
    allf = sorted({c for v in sets.values() for c in v}) + ["week"]
    sets["all_linear"] = allf
    fit = G[G.season.between(2014, 2018) & G.v.notna()].dropna(subset=allf)
    val = G[G.season.between(2019, 2021) & G.v.notna()].dropna(subset=allf)
    lines_out = ["# Alternative college football rating models (development seasons only)\n",
                 f"Fit on 2014-18 ({len(fit)} FBS games), validated on 2019-21 ({len(val)}). 2022-25 not examined.\n",
                 "| model | MAE vs result | MAE of close | info beyond close (coef on model−close) | \\|pred−close\\|≥3: bets, cover | ≥5: bets, cover |",
                 "|---|---|---|---|---|---|"]
    res = {}

    def ats(pred, D_, k):
        d = pred - D_.v; m = np.abs(d) >= k; r = np.sign(d[m]) * (D_.margin[m] - D_.v[m])
        c, l = int((r > 0).sum()), int((r < 0).sum())
        return f"{int(m.sum())}, {c / max(1, c + l):.3f}"

    for name, cols in sets.items():
        X = np.column_stack([np.ones(len(fit))] + [fit[c] for c in cols])
        lam = 1.0 if name == "all_linear" else 0.0
        A = X.T @ X + lam * np.eye(X.shape[1]); A[0, 0] -= lam
        beta = np.linalg.solve(A, X.T @ fit.margin.to_numpy())
        pv = np.column_stack([np.ones(len(val))] + [val[c] for c in cols]) @ beta
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(val)), val.v, pv - val.v]), val.margin, rcond=None)[0][2]
        res[name] = pv
        lines_out.append(f"| {name} | {np.abs(val.margin - pv).mean():.2f} | {np.abs(val.margin - val.v).mean():.2f} | {coef:+.3f} | {ats(pv, val, 3)} | {ats(pv, val, 5)} |")
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1)
        m.fit(fit[allf], fit.margin)
        pv = m.predict(val[allf]); res["all_gbm"] = pv
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(val)), val.v, pv - val.v]), val.margin, rcond=None)[0][2]
        lines_out.append(f"| all_gbm | {np.abs(val.margin - pv).mean():.2f} | {np.abs(val.margin - val.v).mean():.2f} | {coef:+.3f} | {ats(pv, val, 3)} | {ats(pv, val, 5)} |")
        # residual model: can any feature set predict (result - close)?
        r = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=80, verbose=-1)
        r.fit(fit[allf + ["v"]], fit.margin - fit.v)
        rp = r.predict(val[allf + ["v"]])
        cc = np.corrcoef(rp, val.margin - val.v)[0, 1]
        rr = np.sign(rp) * (val.margin - val.v); top = np.abs(rp) >= np.quantile(np.abs(rp), 0.8)
        c, l = int((rr[top] > 0).sum()), int((rr[top] < 0).sum())
        lines_out.append(f"\nResidual model (LightGBM predicting result − close from every feature + the close): validation corr {cc:+.3f}; top 20% most confident picks cover {c / max(1, c + l):.3f} ({c}-{l}).")
    except ImportError:
        pass
    # totals
    tf = G[G.season.between(2014, 2018) & G.total_close.notna()].dropna(subset=["pts_s", "plays_s", "ppa_s"])
    tv = G[G.season.between(2019, 2021) & G.total_close.notna()].dropna(subset=["pts_s", "plays_s", "ppa_s"])
    tcols = ["pts_s", "plays_s", "ppa_s", "sr_s", "expl_s"]
    X = np.column_stack([np.ones(len(tf))] + [tf[c] for c in tcols]); bt = np.linalg.lstsq(X, tf.total, rcond=None)[0]
    tp = np.column_stack([np.ones(len(tv))] + [tv[c] for c in tcols]) @ bt
    d = tp - tv.total_close; m3 = np.abs(d) >= 3; rr = np.sign(d[m3]) * (tv.total[m3] - tv.total_close[m3])
    lines_out.append(f"\nTotals (points + pace + PPA + success + explosiveness sums, fit 2014-18): MAE {np.abs(tv.total - tp).mean():.2f} vs close {np.abs(tv.total - tv.total_close).mean():.2f}; |pred−close|≥3: {int(m3.sum())} bets, hit {(rr > 0).sum() / max(1, (rr != 0).sum()):.3f}.")
    # openers: fit 2021-22, validate 2023
    O = G[G.season.between(2021, 2023) & G.vo.notna() & G.v.notna()].dropna(subset=allf).copy()
    O["move"] = O.v - O.vo
    of, ov = O[O.season <= 2022], O[O.season == 2023]
    lines_out.append("\n## Opening lines (fit 2021-22, validate 2023)\n")
    lines_out.append("| model | corr(model − open, close − open) | \\|model−open\\|≥5: bets, line moved our way / against, cover vs open |")
    lines_out.append("|---|---|---|")
    for name, cols in list(sets.items()):
        X = np.column_stack([np.ones(len(of))] + [of[c] for c in cols])
        beta = np.linalg.lstsq(X, of.margin, rcond=None)[0]
        pv = np.column_stack([np.ones(len(ov))] + [ov[c] for c in cols]) @ beta
        gap = pv - ov.vo; cc = np.corrcoef(gap, ov.move)[0, 1]
        m5 = np.abs(gap) >= 5; s = np.sign(gap[m5]); mv = s * ov.move[m5]; cv = s * (ov.margin[m5] - ov.vo[m5])
        lines_out.append(f"| {name} | {cc:+.3f} | {int(m5.sum())}, {np.mean(mv > 0):.0%} / {np.mean(mv < 0):.0%}, {(cv > 0).sum() / max(1, (cv != 0).sum()):.3f} |")
    (OUT / "alt_models.md").write_text("\n".join(lines_out) + "\n")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
