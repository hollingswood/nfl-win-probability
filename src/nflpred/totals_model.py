"""Total-points model (display / benchmark only; NOT a betting track and NOT a model input).

Ridge regression on pre-game information only:
  * each team's EWMA of points scored and allowed over its strictly-earlier games (shift(1)),
  * each team's offensive / defensive / passing / rushing EPA per play (the leak-free `pre_`
    EWMA columns from features.py),
  * the league-wide scoring level (EWMA of game totals over games on EARLIER dates),
  * stadium type (fixed dome / retractable roof) and the final-regular-season-week flag.
Weather is deliberately left out: recorded game-time wind is not what a forecast knows.

Fit protocol matches model.fit(before_date=...): only completed games with gameday < cutoff.

Season-forward results (each season predicted by a model trained only on earlier seasons,
`holdout_report`), mean absolute error in points vs the actual total:
  validation 2015-2019: model 10.89, closing total (nflverse total_line) 10.61  (1,335 games)
  holdout    2020-2025: model 10.62, closing total 10.31  (1,693 games); over/under calls
  vs the closing total hit 50.0%.
The market is better, as expected; this is context for the cards, not an edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIRST_TRAIN_SEASON = 2013   # 2012 warms up the EWMAs
TEAM_HALFLIFE = 8           # games, same as features.TEAM_HALFLIFE
LEAGUE_HALFLIFE = 128       # games (~half a season) for the league scoring level
LEAGUE_PRIOR = 44.0         # league total before any history (only affects 2012 warm-up)
ALPHA = 50.0

TOTAL_FEATURES = [
    "t_home_pf", "t_home_pa", "t_away_pf", "t_away_pa",
    "t_home_off_epa", "t_home_def_epa", "t_away_off_epa", "t_away_def_epa",
    "t_home_pass_epa", "t_away_pass_epa", "t_home_rush_epa", "t_away_rush_epa",
    "t_league_total", "t_dome", "t_retractable", "t_final_week",
]


def _team_points_ewm(df: pd.DataFrame) -> pd.DataFrame:
    """EWMA points for / against over each team's strictly-earlier games."""
    cols = ["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]
    h = df[cols].rename(columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}).drop(columns="away_team")
    a = df[cols].rename(columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}).drop(columns="home_team")
    lg = pd.concat([h, a], ignore_index=True).sort_values(["team", "gameday", "game_id"])
    for c in ("pf", "pa"):
        lg[f"pre_{c}"] = lg.groupby("team")[c].transform(
            lambda x: x.shift(1).ewm(halflife=TEAM_HALFLIFE, ignore_na=True).mean())
    return lg[["game_id", "team", "pre_pf", "pre_pa"]]


def _league_level(df: pd.DataFrame) -> pd.Series:
    """League scoring level as of the START of each game's date (no same-day games used)."""
    done = df[df["home_score"].notna() & df["away_score"].notna()].sort_values(["gameday", "game_id"])
    tot = done["home_score"] + done["away_score"]
    ew = tot.ewm(halflife=LEAGUE_HALFLIFE).mean()
    by_day = pd.DataFrame({"gameday": done["gameday"].values, "lvl": ew.values}).groupby("gameday")["lvl"].last()
    by_day = by_day.reset_index().sort_values("gameday")
    key = df[["gameday"]].reset_index().sort_values("gameday")
    m = pd.merge_asof(key, by_day, on="gameday", allow_exact_matches=False)
    return m.set_index("index")["lvl"].reindex(df.index).fillna(LEAGUE_PRIOR)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the feature frame (features.build_features output) with TOTAL_FEATURES."""
    out = df.copy()
    pts = _team_points_ewm(out)
    for side in ("home", "away"):
        m = pts.rename(columns={"team": f"{side}_team", "pre_pf": f"t_{side}_pf", "pre_pa": f"t_{side}_pa"})
        out = out.merge(m, on=["game_id", f"{side}_team"], how="left")
        for s in ("off_epa", "def_epa", "pass_epa", "rush_epa"):
            out[f"t_{side}_{s}"] = out.get(f"{side}_{s}", pd.Series(np.nan, index=out.index))
    out["t_league_total"] = _league_level(out)
    half = out["t_league_total"] / 2
    for side in ("home", "away"):  # teams with no history yet: league average
        out[f"t_{side}_pf"] = out[f"t_{side}_pf"].fillna(half)
        out[f"t_{side}_pa"] = out[f"t_{side}_pa"].fillna(half)
    roof = out.get("roof", pd.Series("", index=out.index)).fillna("")
    out["t_dome"] = (roof == "dome").astype(float)
    out["t_retractable"] = roof.isin(["closed", "open"]).astype(float)
    out["t_final_week"] = out.get("final_week", pd.Series(0, index=out.index)).fillna(0).astype(float)
    out[TOTAL_FEATURES] = out[TOTAL_FEATURES].astype(float).fillna(0.0)
    return out


def train_rows(df: pd.DataFrame, before_season: int | None = None,
               before_date: pd.Timestamp | None = None) -> pd.DataFrame:
    d = df[(df["season"] >= FIRST_TRAIN_SEASON) & df["home_score"].notna() & df["away_score"].notna()]
    if before_season is not None:
        d = d[d["season"] < before_season]
    if before_date is not None:
        d = d[d["gameday"] < before_date]
    return d


class TotalsModel:
    def __init__(self, alpha: float | None = None):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA if alpha is None else alpha))

    def fit(self, X: pd.DataFrame, total) -> "TotalsModel":
        self.pipe.fit(X[TOTAL_FEATURES], total)
        self.sigma = float(np.sqrt(np.mean((np.asarray(total) - self.pipe.predict(X[TOTAL_FEATURES])) ** 2)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict(X[TOTAL_FEATURES])


def _ensure(df: pd.DataFrame) -> pd.DataFrame:
    return df if set(TOTAL_FEATURES) <= set(df.columns) else add_features(df)


def fit(df: pd.DataFrame, **kw) -> TotalsModel:
    """kw: before_season / before_date, as model.fit."""
    d = train_rows(_ensure(df), **kw)
    return TotalsModel().fit(d, (d["home_score"] + d["away_score"]).values)


def predict(model: TotalsModel, df: pd.DataFrame) -> np.ndarray:
    return model.predict(_ensure(df))


def walk_forward(df: pd.DataFrame, season: int) -> pd.Series:
    """Predicted totals for this season's completed games, each week's model trained only on
    games before that week's first kickoff (as published live). Indexed by game_id."""
    t = _ensure(df)
    done = t[(t["season"] == season) & t["home_score"].notna() & t["away_score"].notna()]
    out = []
    for _, d in done.groupby("week"):
        m = fit(t, before_date=d["gameday"].min())
        out.append(pd.Series(m.predict(d), index=d["game_id"].values))
    return pd.concat(out) if out else pd.Series(dtype=float)


def holdout_report(df: pd.DataFrame, seasons=range(2020, 2026)) -> dict:
    """Season-forward MAE of the model vs the closing total (total_line) on the same games."""
    t = _ensure(df)
    parts = []
    for s in seasons:
        test = t[(t["season"] == s) & t["home_score"].notna() & t["total_line"].notna()].copy()
        if test.empty:
            continue
        test["pred"] = fit(t, before_season=s).predict(test)
        parts.append(test)
    d = pd.concat(parts)
    actual = d["home_score"] + d["away_score"]
    over = actual > d["total_line"]
    push = actual == d["total_line"]
    call_over = d["pred"] > d["total_line"]
    ou = (~push) & (d["pred"] != d["total_line"])
    return {"games": int(len(d)), "seasons": f"{seasons.start}-{seasons.stop - 1}",
            "mae_model": round(float((d["pred"] - actual).abs().mean()), 2),
            "mae_market": round(float((d["total_line"] - actual).abs().mean()), 2),
            "bias_model": round(float((d["pred"] - actual).mean()), 2),
            "corr_model_market": round(float(np.corrcoef(d["pred"], d["total_line"])[0, 1]), 3),
            "ou_vs_close_hit_rate": round(float((call_over[ou] == over[ou]).mean()), 3)}
