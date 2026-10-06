"""Recency test (2026-10-05): does weighting recent seasons more heavily improve the win-probability model?
Variants fixed before running: training window (all / last 6 / last 4 / last 3 seasons) and exponential season weights
(half-life 1, 2, 4, 8 seasons). Tune = 2015-19, holdout = 2020-25, plus per-season holdout and the 2024-25 subset."""
import sys, json
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nflpred import model as M, features as F
import experiment as E

_, df = E.run("base")
feats = F.FEATURES
def fp(train, test, w=None):
    y = (train.home_score - train.away_score).values
    m = make_pipeline(StandardScaler(), Ridge(alpha=M.RIDGE_ALPHA))
    m.fit(train[feats], y, ridge__sample_weight=w)
    r = y - m.predict(train[feats])
    sig = np.sqrt(np.average(r ** 2, weights=w))
    return M._norm_cdf(m.predict(test[feats]) / sig)
variants = {"all seasons (production)": (None, None)}
for k in (6, 4, 3): variants[f"last {k} seasons only"] = (k, None)
for h in (1, 2, 4, 8): variants[f"half-life {h} seasons"] = (None, h)
out = {}
for name, (win, hl) in variants.items():
    res = {}
    for label, seasons in (("tune 2015-19", range(2015, 2020)), ("holdout 2020-25", range(2020, 2026))):
        ps, ys, per = [], [], {}
        for s in seasons:
            tr = M.train_rows(df, before_season=s); te = df[(df.season == s) & df.home_win.notna()]
            if win: tr = tr[tr.season >= s - win]
            w = (0.5 ** ((s - 1 - tr.season) / hl)).values if hl else None
            p = fp(tr, te, w); ps.append(p); ys.append(te.home_win.values)
            per[s] = round(M.score(te.home_win.values, p)["log_loss"], 4)
        res[label] = round(M.score(np.concatenate(ys), np.concatenate(ps))["log_loss"], 4)
        if label.startswith("holdout"):
            res["by_season"] = per
            idx = [i for i, s in enumerate(seasons) if s >= 2024]
            res["2024-25"] = round(M.score(np.concatenate([ys[i] for i in idx]), np.concatenate([ps[i] for i in idx]))["log_loss"], 4)
    out[name] = res
    print(name, res, flush=True)
json.dump(out, open('output/research/recency_weighting.json', 'w'), indent=1)
