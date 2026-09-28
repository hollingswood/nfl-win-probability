"""Experiment harness: tune on 2015-2019 (validation), confirm on 2020-2025 (holdout)."""
import importlib, sys, time
import numpy as np, pandas as pd
from nflpred import data, features as F, model as M

VAL, HOLD = range(2015, 2020), range(2020, 2026)
G = data.load_schedules(); P = data.load_pbp(range(2012, 2027))
INJ = (data.load_injuries(range(2012, 2027)), data.load_snaps(range(2012, 2027)), data.load_players())


def run(name, feat_cfg=None, model_kind="logistic", features=None, fit_kw=None, inj=True):
    importlib.reload(F); M.FEATURES = F.FEATURES  # reset defaults
    for k, v in (feat_cfg or {}).items(): setattr(F, k, v)
    df = F.build_features(G, P, INJ if inj else None)
    feats = features or F.FEATURES
    out = {"name": name}
    for label, seasons in (("val", VAL), ("hold", HOLD)):
        ps, ys = [], []
        for s in seasons:
            tr = M.train_rows(df, before_season=s); te = df[(df.season == s) & df.home_win.notna()]
            p = M.fit_predict(tr, te, feats, model_kind, **(fit_kw or {}))
            ps.append(p); ys.append(te.home_win.values)
        sc = M.score(np.concatenate(ys), np.concatenate(ps))
        out[f"{label}_ll"] = round(sc["log_loss"], 4); out[f"{label}_acc"] = round(sc["accuracy"], 3)
    return out, df
