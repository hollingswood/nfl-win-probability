"""Opponent-adjusted power ratings, re-solved before every week.

For each (season, week), fit  y_g = hfa*home_g + r_home - r_away  by weighted ridge over all
games played BEFORE that week's first kickoff, weighting older games by a half-life and
discounting across offseasons. y is point margin (points rating) or net EPA/play (EPA rating).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALFLIFE_WEEKS = 10.0
OFFSEASON_PENALTY_WEEKS = 12.0  # an offseason "ages" games by this many extra weeks
RIDGE = 2.0


def _solve(teams_idx, h, a, y, w, home, n):
    k = len(y)
    X = np.zeros((k, n + 1))
    X[np.arange(k), h] = 1
    X[np.arange(k), a] = -1
    X[:, n] = home
    W = w[:, None]
    A = X.T @ (X * W) + RIDGE * np.diag(np.r_[np.ones(n), 0.0])
    b = X.T @ (w * y)
    return np.linalg.solve(A + 1e-9 * np.eye(n + 1), b)


def power_ratings(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """df needs game_id, season, week, gameday, home_team, away_team, completed, location, and `target`."""
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    d = df.sort_values("gameday").reset_index(drop=True)
    # continuous "week clock": weeks since start + offseason penalty per new season
    season0 = d["season"].min()
    clock = (d["gameday"] - d["gameday"].min()).dt.days / 7.0 + \
        (d["season"] - season0) * OFFSEASON_PENALTY_WEEKS
    d["clock"] = clock
    h_all = d["home_team"].map(ix).values
    a_all = d["away_team"].map(ix).values
    home_all = (d["location"] != "Neutral").astype(float).values
    y_all = d[target].values.astype(float)
    done = d["completed"].values & ~np.isnan(y_all)
    out = np.full((len(d), 2), np.nan)
    for (s, wk), grp in d.groupby(["season", "week"], sort=False):
        start = grp["gameday"].min()
        m = done & (d["gameday"] < start).values
        if m.sum() < 50:
            continue
        now = grp["clock"].min()
        w = 0.5 ** ((now - d["clock"].values[m]) / HALFLIFE_WEEKS)
        sol = _solve(ix, h_all[m], a_all[m], y_all[m], w, home_all[m], n)
        r = sol[:n]
        idx = grp.index.values
        out[idx, 0] = r[h_all[idx]]
        out[idx, 1] = r[a_all[idx]]
    res = d[["game_id"]].copy()
    res[f"home_{target}_rating"] = out[:, 0]
    res[f"away_{target}_rating"] = out[:, 1]
    return res
