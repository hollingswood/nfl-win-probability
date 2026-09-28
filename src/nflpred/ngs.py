"""Next Gen Stats features (tracking data, 2016+).

Uses weekly rows only (week 0 rows are season totals that include future games, so they are
dropped). Each stat is an exponentially weighted, attempt-weighted average over games strictly
before kickoff, shrunk toward league average when the sample is small.
"""
from __future__ import annotations

import pandas as pd

TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAC": "JAX"}
DECAY = 0.93   # per game
PRIOR_N = 60   # attempts/targets of shrinkage toward the league mean


def _with_dates(d: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    d = d[d["week"] > 0].copy()
    d["team"] = d["team_abbr"].replace(TEAM_MAP)
    long = pd.concat([
        sched[["season", "week", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
        sched[["season", "week", "gameday", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    return d.merge(long, on=["season", "week", "team"], how="inner")


def _decayed(d: pd.DataFrame, key: str, val: str, w: str, prior: float) -> pd.DataFrame:
    """Post-game shrunk decayed average of `val` (weighted by `w`) per `key`, keyed by date."""
    rows = []
    for k, g in d.sort_values("gameday").groupby(key, sort=False):
        num = den = 0.0
        for gd, v, n in zip(g["gameday"], g[val], g[w]):
            if pd.isna(v) or pd.isna(n) or n <= 0:
                continue
            num = num * DECAY + v * n
            den = den * DECAY + n
            rows.append((k, gd, (num + prior * PRIOR_N) / (den + PRIOR_N)))
    return pd.DataFrame(rows, columns=[key, "gameday", "val"]).sort_values("gameday")


def _asof(keys: pd.DataFrame, table: pd.DataFrame, key: str, default: float) -> pd.Series:
    k = keys.reset_index().sort_values("gameday")
    m = pd.merge_asof(k, table, on="gameday", by=key, allow_exact_matches=False)
    return m.set_index("index")["val"].reindex(keys.index).fillna(default)


def ngs_features(df: pd.DataFrame, passing: pd.DataFrame, rushing: pd.DataFrame,
                 receiving: pd.DataFrame) -> pd.DataFrame:
    """Adds qb_cpoe_diff, qb_ttt_diff, ryoe_diff, sep_diff (positive favors home)."""
    out = df.copy()
    p = _with_dates(passing, out)
    cp_mean = float((p["completion_percentage_above_expectation"] * p["attempts"]).sum() / p["attempts"].sum())
    tt_mean = float((p["avg_time_to_throw"] * p["attempts"]).sum() / p["attempts"].sum())
    cpoe = _decayed(p, "player_gsis_id", "completion_percentage_above_expectation", "attempts", cp_mean)
    ttt = _decayed(p, "player_gsis_id", "avg_time_to_throw", "attempts", tt_mean)

    r = _with_dates(rushing, out)
    r = r.groupby(["team", "gameday"]).apply(
        lambda x: pd.Series({"ryoe": x["rush_yards_over_expected"].sum() / max(x["rush_attempts"].sum(), 1),
                             "att": x["rush_attempts"].sum()}), include_groups=False).reset_index()
    ryoe = _decayed(r, "team", "ryoe", "att", 0.0)

    c = _with_dates(receiving, out)
    c = c.groupby(["team", "gameday"]).apply(
        lambda x: pd.Series({"sep": (x["avg_separation"] * x["targets"]).sum() / max(x["targets"].sum(), 1),
                             "tgt": x["targets"].sum()}), include_groups=False).reset_index()
    sep_mean = float((c["sep"] * c["tgt"]).sum() / c["tgt"].sum())
    sep = _decayed(c, "team", "sep", "tgt", sep_mean)

    vals = {}
    for side in ("home", "away"):
        qk = out[["gameday", f"{side}_qb_id"]].rename(columns={f"{side}_qb_id": "player_gsis_id"})
        tk = out[["gameday", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
        vals[f"{side}_cpoe"] = _asof(qk, cpoe, "player_gsis_id", cp_mean)
        vals[f"{side}_ttt"] = _asof(qk, ttt, "player_gsis_id", tt_mean)
        vals[f"{side}_ryoe"] = _asof(tk, ryoe, "team", 0.0)
        vals[f"{side}_sep"] = _asof(tk, sep, "team", sep_mean)
    pre2016 = out["season"] < 2016  # no tracking data: neutral
    out["qb_cpoe_diff"] = (vals["home_cpoe"] - vals["away_cpoe"]).where(~pre2016, 0.0)
    out["qb_ttt_diff"] = (vals["away_ttt"] - vals["home_ttt"]).where(~pre2016, 0.0)  # quicker is better
    out["ryoe_diff"] = (vals["home_ryoe"] - vals["away_ryoe"]).where(~pre2016, 0.0)
    out["sep_diff"] = (vals["home_sep"] - vals["away_sep"]).where(~pre2016, 0.0)
    return out
