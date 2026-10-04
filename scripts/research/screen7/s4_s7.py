"""S4 (do our errors overlap the market's?) and S7 (week-to-week adaptive model+market blend), NFL 2015-2025.
Market = nflverse closing spread_line (+ = home favored) and vegas_home_prob (closing moneyline, no-vig)."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "research" / "screen7"

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(y, p): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def main():
    D = pd.read_parquet(ROOT / "data/screen_base.parquet")
    D = D[D.spread_line.notna() & D.vegas_home_prob.notna() & D.home_score.notna()].copy()
    D["gameday"] = pd.to_datetime(D.gameday)
    D["margin"] = D.home_score - D.away_score
    D["y"] = (D.margin > 0).astype(float)
    D = D[D.margin != 0]
    D["v"] = D.spread_line
    D["gap"] = D.mu_model - D.v
    D["res"] = D.margin - D.v
    R = {"S4": {}, "S7": {}}
    # ---------------- S4: error overlap
    rows = []
    for s, d in D.groupby("season"):
        em, ev = d.margin - d.mu_model, d.res
        b = np.polyfit(d.gap, d.res, 1)[0]
        rows.append({"season": int(s), "n": len(d), "corr_errors": round(float(np.corrcoef(em, ev)[0, 1]), 3),
                     "beta_gap": round(float(b), 3), "mae_model": round(float(em.abs().mean()), 2), "mae_close": round(float(ev.abs().mean()), 2)})
    all_b = np.polyfit(D.gap, D.res, 1)[0]
    se = float(np.std(D.res - all_b * D.gap) / (np.std(D.gap) * math.sqrt(len(D))))
    R["S4"] = {"by_season": rows, "beta_all": round(float(all_b), 3), "beta_se": round(se, 3),
               "corr_errors_all": round(float(np.corrcoef(D.margin - D.mu_model, D.res)[0, 1]), 3),
               "note": "beta_gap = how much of (model - close) shows up in (result - close). 0 = the model adds nothing the close lacks; 1 = the close ignores it entirely."}
    # where (if anywhere) does the model add information? pre-declared slices
    sl = {"weeks 1-4": D.week <= 4, "weeks 5-9": D.week.between(5, 9), "weeks 10-18": D.week.between(10, 18),
          "playoffs": D.game_type != "REG", "QB change either side": (D.get("home_qb_change", 0).fillna(0) + D.get("away_qb_change", 0).fillna(0)) > 0,
          "big gap |gap|>=3": D.gap.abs() >= 3, "home fav": D.v > 0, "home dog": D.v < 0}
    R["S4"]["slices"] = {}
    for k, m in sl.items():
        d = D[m]
        if len(d) < 60:
            continue
        b = np.polyfit(d.gap, d.res, 1)[0]
        se = float(np.std(d.res - b * d.gap) / (np.std(d.gap) * math.sqrt(len(d))))
        R["S4"]["slices"][k] = {"n": int(len(d)), "beta": round(float(b), 3), "se": round(se, 3), "z": round(float(b / se), 2)}
    # ---------------- S7: adaptive blend, refit every week on the trailing window
    D = D.sort_values("gameday")
    out = []
    for (s, w), d in D[D.season >= 2015].groupby(["season", "week"], sort=False):
        start = d.gameday.min()
        for win_name, yrs in (("trail1", 1), ("trail3", 3), ("all", 99)):
            tr = D[(D.gameday < start) & (D.gameday >= start - pd.Timedelta(days=365 * yrs + 30))]
            # probability blend (logistic on logits) and margin blend (OLS)
            X = np.column_stack([np.ones(len(tr)), logit(tr.p_model), logit(tr.vegas_home_prob)])
            beta = np.zeros(3); beta[2] = 1.0
            for _ in range(25):   # Newton steps
                p = 1 / (1 + np.exp(-X @ beta)); W = p * (1 - p)
                H = X.T @ (X * W[:, None]) + 1e-3 * np.eye(3); g = X.T @ (tr.y - p) - 1e-3 * beta
                beta = beta + np.linalg.solve(H, g)
            Xt = np.column_stack([np.ones(len(d)), logit(d.p_model), logit(d.vegas_home_prob)])
            pb = 1 / (1 + np.exp(-Xt @ beta))
            bm = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), tr.mu_model, tr.v]), tr.margin, rcond=None)[0]
            mb = bm[0] + bm[1] * d.mu_model + bm[2] * d.v
            for i, (gid, p_, m_) in enumerate(zip(d.game_id, pb, mb)):
                out.append({"game_id": gid, "season": s, "week": w, "window": win_name, "p_blend": float(p_), "m_blend": float(m_),
                            "w_model": float(beta[1]), "w_vegas": float(beta[2]), "m_w_model": float(bm[1])})
    B = pd.DataFrame(out).merge(D[["game_id", "y", "vegas_home_prob", "p_model", "margin", "v"]], on="game_id")
    s7 = {}
    for win, b in B.groupby("window"):
        per = {int(s): {"blend": round(ll(x.y, x.p_blend), 4), "vegas": round(ll(x.y, x.vegas_home_prob), 4)} for s, x in b.groupby("season")}
        diff = [ll(x.y, x.vegas_home_prob) - ll(x.y, x.p_blend) for _, x in b.groupby("season")]
        # per-game log-loss difference for a paired test
        d_i = -(b.y * np.log(b.vegas_home_prob) + (1 - b.y) * np.log(1 - b.vegas_home_prob)) + (b.y * np.log(b.p_blend) + (1 - b.y) * np.log(1 - b.p_blend))
        z = float(d_i.mean() / (d_i.std() / math.sqrt(len(d_i))))
        ats = {}
        for k in (1.0, 2.0, 3.0):
            g = b.m_blend - b.v; m = g.abs() >= k; r = np.sign(g[m]) * (b.margin[m] - b.v[m])
            c, l = int((r > 0).sum()), int((r < 0).sum())
            ats[f">={k:g}"] = {"bets": c + l, "cover": round(c / max(1, c + l), 3)}
        s7[win] = {"log_loss_blend": round(ll(b.y, b.p_blend), 4), "log_loss_vegas": round(ll(b.y, b.vegas_home_prob), 4),
                   "seasons_blend_better": int(sum(x > 0 for x in diff)), "seasons": len(diff), "z_blend_beats_vegas": round(z, 2),
                   "avg_model_weight": round(float(b.w_model.mean()), 3), "last_model_weight": round(float(b.sort_values(["season", "week"]).w_model.iloc[-1]), 3),
                   "ats_vs_close": ats, "by_season": per}
    R["S7"] = s7
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s4_s7.json").write_text(json.dumps(R, indent=1))
    print(json.dumps({"S4": {k: R["S4"][k] for k in ("beta_all", "beta_se", "corr_errors_all", "slices")},
                      "S7": {k: {kk: v[kk] for kk in ("log_loss_blend", "log_loss_vegas", "seasons_blend_better", "seasons", "z_blend_beats_vegas", "avg_model_weight", "last_model_weight", "ats_vs_close")} for k, v in s7.items()}}, indent=1))
    print("S4 by season:", [(r["season"], r["beta_gap"], r["corr_errors"]) for r in rows])

if __name__ == "__main__":
    main()
