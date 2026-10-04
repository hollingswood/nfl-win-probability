"""S6: preseason prior in weeks 2-8 (Fodor, Patterson & Shank 2025: closing lines stay anchored to preseason
Super Bowl odds; betting with preseason expectations paid in weeks 2-8).

Pre-declared before looking at results (2026-10-03):
  preseason implied home margin = a + c*(win_total_home - win_total_away), fit on WEEK-1 closing spreads 2003-2012
  signal (weeks 2-8, regular season): gap = preseason_implied - closing spread_line (both home margins)
  bet the side the preseason number favors when |gap| >= 3 (primary); 2 and 4 reported
  graded vs the closing number at the closing price (nflverse; -110 when missing)
  screen seasons 2013-2025 (Fodor et al. sample ends 2023: 2024-25 are post-publication)
  pass: one-sided p < 0.007 (7 strategies, Bonferroni) on ROI > 0 AND positive in most seasons
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]

def dec(a):
    a = pd.to_numeric(a, errors="coerce").fillna(-110).to_numpy(float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))

def main():
    G = pd.read_parquet(ROOT / "data/raw/games.parquet")
    G = G[(G.game_type == "REG") & G.spread_line.notna() & G.home_score.notna()].copy()
    W = pd.read_csv(ROOT / "data/research_futures/win_totals_2013_2026.csv")[["season", "team", "line_adj"]]
    fix = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
    W["team"] = W.team.replace(fix)
    for c in ("home_team", "away_team"):
        G[c] = G[c].replace(fix)
    G = G.merge(W.rename(columns={"team": "home_team", "line_adj": "wt_h"}), on=["season", "home_team"], how="left")
    G = G.merge(W.rename(columns={"team": "away_team", "line_adj": "wt_a"}), on=["season", "away_team"], how="left")
    G = G[G.wt_h.notna() & G.wt_a.notna()]
    G["neutral"] = (G.location == "Neutral").astype(float)
    wk1 = G[(G.week == 1) & G.season.between(2003, 2012)]
    X = np.column_stack([np.ones(len(wk1)) - wk1.neutral, wk1.wt_h - wk1.wt_a])
    a, c = np.linalg.lstsq(X, wk1.spread_line, rcond=None)[0]
    G["pre"] = a * (1 - G.neutral) + c * (G.wt_h - G.wt_a)
    G["gap"] = G.pre - G.spread_line
    G["res"] = (G.home_score - G.away_score) - G.spread_line
    S = G[G.week.between(2, 8) & G.season.between(2013, 2025)].copy()
    out = {"fit": {"hfa": round(a, 2), "pts_per_win_diff": round(c, 3), "n_week1": len(wk1)}, "rules": {}}
    for k in (2, 3, 4):
        d = S[S.gap.abs() >= k]; s = np.sign(d.gap); r = s * d.res
        pr = np.where(s > 0, dec(d.home_spread_odds), dec(d.away_spread_odds))
        pnl = np.where(r > 0, pr - 1, np.where(r < 0, -1, 0))
        m, se = pnl.mean(), pnl.std() / math.sqrt(len(pnl))
        p = 0.5 * math.erfc((m / se) / math.sqrt(2))
        per = pd.Series(pnl, index=d.season).groupby(level=0).mean()
        cov = (r > 0).sum() / max(1, (r != 0).sum())
        out["rules"][f"|gap|>={k}"] = {"bets": int(len(d)), "cover": round(float(cov), 3), "roi": round(float(m), 4), "se": round(float(se), 4),
                                      "p_one_sided": round(float(p), 4), "seasons_positive": f"{int((per > 0).sum())}/{len(per)}",
                                      "post_publication_2024_25_roi": round(float(pnl[d.season.to_numpy() >= 2024].mean()), 4) if (d.season >= 2024).any() else None,
                                      "pass": bool(p < 0.007 and m > 0 and (per > 0).sum() > len(per) / 2),
                                      "by_season": {int(k2): round(float(v), 3) for k2, v in per.items()},
                                      "side_preseason_favors_dog_share": round(float(np.mean((s > 0) == (d.spread_line < 0))), 3)}
    # also: weeks 9-18 as a placebo (the paper says the effect fades)
    P = G[G.week.between(9, 18) & G.season.between(2013, 2025)]
    d = P[P.gap.abs() >= 3]; s = np.sign(d.gap); r = s * d.res
    out["placebo_weeks_9_18_gap3"] = {"bets": int(len(d)), "cover": round(float((r > 0).sum() / max(1, (r != 0).sum())), 3)}
    (ROOT / "output/research/screen7/s6_preseason.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
