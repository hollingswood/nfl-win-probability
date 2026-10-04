"""Dynamic market blend (nfelo-style), NFL. Written 2026-10-04 before running.

Idea (research report, rank 9): lean on the model more for teams where it has recently matched the market's accuracy,
less where it has trailed. For each game, D = mean over the two teams of an exponentially weighted average of
(model margin error^2 - Vegas margin error^2) over that team's EARLIER games (shifted: no leakage).
Blend in log-odds: logit p = w * logit(model) + (1 - w) * logit(Vegas), with w = w0 * exp(-lam * max(D, 0) / 100) and,
when the model has been better (D < 0), w = min(1, w0 * (1 + kappa * (-D) / 100)).
Parameters (w0, lam, kappa, halflife) tuned on 2015-2019 by log loss; scored once on 2020-2025 against Vegas alone and
the best static blend (w = w0, tuned on 2015-19). CLAUDE.md rule: to claim new information the blend must beat Vegas
alone on 2020-2025.
"""
import datetime as dt
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nflpred import model as M  # noqa: E402
from nflpred.pipeline import build  # noqa: E402


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    df = build(refresh=False, today=dt.date.today())
    parts = []
    for s in range(2014, 2026):
        te = df[(df.season == s) & df.home_win.notna()].copy()
        fit = M.fit(df, before_season=s)
        te["p"] = M.predict(fit, te)
        te["mu"] = fit.predict_margin(te)
        parts.append(te)
    d = pd.concat(parts)
    d = d[d.vegas_home_prob.notna() & d.spread_line.notna()].sort_values(["gameday", "game_id"]).reset_index(drop=True)
    d["result_margin"] = d.home_score - d.away_score
    d["e_m"] = (d.mu - d.result_margin) ** 2
    d["e_v"] = (d.spread_line - d.result_margin) ** 2
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    d["lm"], d["lv"] = lg(d.p), lg(d.vegas_home_prob)
    long = pd.concat([d[["game_id", "gameday", "home_team", "e_m", "e_v"]].rename(columns={"home_team": "team"}),
                      d[["game_id", "gameday", "away_team", "e_m", "e_v"]].rename(columns={"away_team": "team"})]).sort_values(["gameday", "game_id"])
    long["diff"] = long.e_m - long.e_v

    def team_D(hl):
        x = long.copy()
        x["ew"] = x.groupby("team")["diff"].transform(lambda s: s.shift(1).ewm(halflife=hl, min_periods=3).mean())
        k = x.set_index(["game_id", "team"]).ew
        hD = k.reindex(list(zip(d.game_id, d.home_team))).to_numpy()
        aD = k.reindex(list(zip(d.game_id, d.away_team))).to_numpy()
        return np.nanmean(np.vstack([hD, aD]), axis=0)

    y = d.home_win.to_numpy()
    val, hold = (d.season.between(2015, 2019)).to_numpy(), (d.season >= 2020).to_numpy()
    sig = lambda z: 1 / (1 + np.exp(-z))
    # static blend
    best_static = min(((ll(y[val], sig(w * d.lm[val] + (1 - w) * d.lv[val])), w) for w in np.arange(0, 0.61, 0.02)))
    w_s = best_static[1]
    best = None
    Ds = {hl: np.nan_to_num(team_D(hl), nan=0.0) for hl in (4, 8, 16)}
    for hl, w0, lam, kap in itertools.product((4, 8, 16), np.arange(0.0, 0.61, 0.05), (0.0, 0.5, 1, 2, 4), (0.0, 0.5, 1, 2)):
        D = Ds[hl]
        w = np.where(D > 0, w0 * np.exp(-lam * D / 100), np.minimum(1, w0 * (1 + kap * (-D) / 100)))
        s = ll(y[val], sig(w[val] * d.lm[val] + (1 - w[val]) * d.lv[val]))
        if best is None or s < best[0]:
            best = (s, hl, w0, lam, kap)
    _, hl, w0, lam, kap = best
    D = Ds[hl]
    w = np.where(D > 0, w0 * np.exp(-lam * D / 100), np.minimum(1, w0 * (1 + kap * (-D) / 100)))
    p_dyn = sig(w * d.lm + (1 - w) * d.lv)
    p_static = sig(w_s * d.lm + (1 - w_s) * d.lv)
    out = {"tuned_on": "2015-2019", "scored_on": "2020-2025", "games_holdout": int(hold.sum()),
           "params": {"halflife_games": hl, "w0": round(float(w0), 3), "lam": lam, "kappa": kap}, "static_w": round(float(w_s), 3),
           "log_loss_holdout": {"vegas": round(ll(y[hold], d.vegas_home_prob[hold]), 5), "model": round(ll(y[hold], d.p[hold]), 5),
                                "static_blend": round(ll(y[hold], p_static[hold]), 5), "dynamic_blend": round(ll(y[hold], p_dyn[hold]), 5)},
           "log_loss_tuning": {"vegas": round(ll(y[val], d.vegas_home_prob[val]), 5), "static_blend": round(best_static[0], 5), "dynamic_blend": round(best[0], 5)},
           "by_season_holdout": {int(s): {"vegas": round(ll(y[m], d.vegas_home_prob[m]), 4), "dynamic": round(ll(y[m], p_dyn[m]), 4)}
                                 for s in range(2020, 2026) for m in [(d.season == s).to_numpy()]}}
    out["beats_vegas_holdout"] = out["log_loss_holdout"]["dynamic_blend"] < out["log_loss_holdout"]["vegas"]
    (ROOT / "output" / "research" / "dynamic_blend.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
