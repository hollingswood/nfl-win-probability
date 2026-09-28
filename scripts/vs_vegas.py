"""Can the model beat Vegas? Writes output/vs_vegas.json.

1. Blend test: fit P = f(model logit, Vegas logit) on 2015-2019, score on 2020-2025. If the blend
   beats Vegas alone out of sample, the model has information the closing line is missing.
2. Betting simulation (2020-2025): bet 1 unit at the actual moneyline whenever expected value
   exceeds a threshold. Uses closing lines, the hardest prices to beat.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from nflpred import model as M
from nflpred.pipeline import build
import datetime as dt

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = build(refresh=False, today=dt.date.today())
    parts = []
    for s in range(2015, 2026):
        te = df[(df.season == s) & df.home_win.notna()].copy()
        te["p"] = M.predict(M.fit(df, before_season=s), te)
        parts.append(te)
    d = pd.concat(parts)
    d = d[d.vegas_home_prob.notna() & d.home_moneyline.notna()]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    d["lm"], d["lv"] = lg(d.p), lg(d.vegas_home_prob)
    val, hold = d[d.season <= 2019], d[d.season >= 2020].copy()
    blend = LogisticRegression(C=1e6).fit(val[["lm", "lv"]], val.home_win)
    hold["pb"] = blend.predict_proba(hold[["lm", "lv"]])[:, 1]
    L = lambda p: float(log_loss(hold.home_win, np.clip(p, 1e-6, 1 - 1e-6)))
    dec = lambda ml: np.where(ml > 0, 1 + ml / 100, 1 + 100 / (-ml))
    dh, da = dec(hold.home_moneyline.values), dec(hold.away_moneyline.values)
    bets = []
    for src, col in (("model", "p"), ("blend", "pb")):
        for edge in (0.0, 0.03, 0.05, 0.08):
            p = hold[col].values
            bh, ba = p * dh - 1 > edge, (1 - p) * da - 1 > edge
            pnl = np.where(bh, np.where(hold.home_win == 1, dh - 1, -1), 0) + \
                np.where(ba, np.where(hold.home_win == 0, da - 1, -1), 0)
            n = int(bh.sum() + ba.sum())
            bets.append({"probs": src, "min_edge": edge, "bets": n,
                         "units": round(float(pnl.sum()), 1), "roi": round(float(pnl.sum() / max(n, 1)), 4)})
    res = {"holdout": "2020-2025", "games": int(len(hold)),
           "log_loss": {"vegas": L(hold.vegas_home_prob), "model": L(hold.p), "blend": L(hold.pb)},
           "blend_weights": {"model": float(blend.coef_[0][0]), "vegas": float(blend.coef_[0][1])},
           "betting": bets}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "vs_vegas.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
