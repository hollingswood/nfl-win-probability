"""Model training, season-forward backtesting, and per-game explanations.

Production model ("margin"): ridge regression predicts the home team's point margin; the win
probability is P(margin > 0) under a normal distribution whose spread (sigma) is the model's
training residual standard deviation. Chosen over logistic regression and gradient boosting
because it scored best on 2015-2019 validation and held up on the 2020-2025 holdout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

FIRST_TRAIN_SEASON = 2013  # 2012 is warm-up for rolling stats / Elo
PRODUCTION = "margin"
RIDGE_ALPHA = 50.0


def _norm_cdf(x):
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x) / sqrt(2)))


class MarginModel:
    """Predicts home point margin; exposes win probability via predict_proba."""

    def __init__(self, alpha: float = RIDGE_ALPHA):
        self.pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X, margin):
        self.pipe.fit(X, margin)
        self.sigma = float(np.sqrt(np.mean((margin - self.pipe.predict(X)) ** 2)))
        return self

    def predict_margin(self, X):
        return self.pipe.predict(X[FEATURES])

    def predict_proba(self, X):
        p = _norm_cdf(self.predict_margin(X) / self.sigma)
        return np.column_stack([1 - p, p])


def make_model(kind: str):
    if kind == "margin":
        return MarginModel()
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    if kind == "gbm":
        base = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0)
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    raise ValueError(kind)


def train_rows(df: pd.DataFrame, before_season: int | None = None,
               before_date: pd.Timestamp | None = None) -> pd.DataFrame:
    d = df[(df["season"] >= FIRST_TRAIN_SEASON) & df["home_win"].notna()]
    if before_season is not None:
        d = d[d["season"] < before_season]
    if before_date is not None:
        d = d[d["gameday"] < before_date]
    return d


def fit(df: pd.DataFrame, kind: str = PRODUCTION, **kw):
    d = train_rows(df, **kw)
    m = make_model(kind)
    if kind == "margin":
        return m.fit(d[FEATURES], (d["home_score"] - d["away_score"]).values)
    m.fit(d[FEATURES], d["home_win"].astype(int))
    return m


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[FEATURES])[:, 1] if not isinstance(model, MarginModel) \
        else model.predict_proba(X)[:, 1]


def score(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "accuracy": float(((p > 0.5) == y).mean())}


def backtest(df: pd.DataFrame, seasons, kinds=("margin", "logistic", "gbm")):
    """Expanding-window: for each test season, train only on prior seasons."""
    preds = []
    for s in seasons:
        test = df[(df["season"] == s) & df["home_win"].notna()].copy()
        for k in kinds:
            test[f"p_{k}"] = predict(fit(df, k, before_season=s), test)
        preds.append(test)
    preds = pd.concat(preds)
    home_rate = train_rows(df)["home_win"].mean()
    rows = []
    for s, d in list(preds.groupby("season")) + [("ALL", preds)]:
        for name, col in [(k, f"p_{k}") for k in kinds] + [("vegas", "vegas_home_prob"), ("home_always", None)]:
            p = d[col] if col else np.full(len(d), home_rate)
            rows.append({"season": s, "model": name, **score(d["home_win"], p)})
    return preds, pd.DataFrame(rows)


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


# ---------------------------------------------------------------- explanations
@dataclass
class Explanation:
    factor: str
    points: float  # contribution to predicted home margin; positive favors home


# Correlated features are grouped so explanations don't show offsetting pieces
# (e.g. "offense helps home" and "passing hurts home" from the same underlying signal).
FACTOR_GROUPS = {
    "Team strength (Elo + point diff)": ["elo_diff", "pt_diff_diff"],
    "Starting QB": ["qb_diff", "qb_change_diff"],
    "Offensive efficiency": ["off_epa_diff", "off_sr_diff", "pass_epa_diff", "rush_epa_diff"],
    "Defensive efficiency": ["def_epa_diff", "def_sr_diff"],
    "Injuries": ["inj_off_diff", "inj_def_diff"],
    "Turnover margin": ["to_margin_diff"],
    "Rest advantage": ["rest_diff"],
    "Home field": ["home_field"],
    "Division game": ["div_game"],
}


def contributions(model: MarginModel, X: pd.DataFrame) -> pd.DataFrame:
    """Per-game points contributed by each factor group. Rows sum exactly to predicted margin.

    For standardized ridge, contribution_i = coef_i * z_i. The intercept is the league-wide
    home edge, so it is credited to "Home field" (its net effect is ~0 at neutral sites).
    """
    scaler, rg = model.pipe[0], model.pipe[-1]
    z = scaler.transform(X[FEATURES])
    c = pd.DataFrame(z * rg.coef_, columns=FEATURES, index=X.index)
    c["home_field"] += rg.intercept_
    return pd.DataFrame({g: c[cols].sum(axis=1) for g, cols in FACTOR_GROUPS.items()})


def explain(model: MarginModel, X: pd.DataFrame, top: int = 3) -> list[list[Explanation]]:
    g = contributions(model, X)
    return [[Explanation(f, float(row[f])) for f in row.abs().sort_values(ascending=False).index[:top]]
            for _, row in g.iterrows()]


def coefficients(model: MarginModel) -> dict:
    rg = model.pipe[-1]
    return {f: float(c) for f, c in zip(FEATURES, rg.coef_)} | {
        "intercept": float(rg.intercept_), "sigma": model.sigma}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
