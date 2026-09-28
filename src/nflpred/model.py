"""Model training, season-forward backtesting, and per-game explanations."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_LABELS, FEATURES

FIRST_TRAIN_SEASON = 2013  # 2012 is warm-up for rolling stats / Elo


def make_model(kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
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


def fit(df: pd.DataFrame, kind: str = "logistic", **kw):
    d = train_rows(df, **kw)
    m = make_model(kind)
    m.fit(d[FEATURES], d["home_win"].astype(int))
    return m


def score(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "accuracy": float(((p > 0.5) == y).mean())}


def backtest(df: pd.DataFrame, seasons, kinds=("logistic", "gbm")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window: for each test season, train only on prior seasons."""
    preds = []
    for s in seasons:
        test = df[(df["season"] == s) & df["home_win"].notna()].copy()
        for k in kinds:
            test[f"p_{k}"] = fit(df, k, before_season=s).predict_proba(test[FEATURES])[:, 1]
        preds.append(test)
    preds = pd.concat(preds)
    rows = []
    for s, d in list(preds.groupby("season")) + [("ALL", preds)]:
        for name, col in [(k, f"p_{k}") for k in kinds] + [("vegas", "vegas_home_prob"), ("home_always_55%", None)]:
            p = d[col] if col else np.full(len(d), train_rows(df, before_season=2100)["home_win"].mean())
            rows.append({"season": s, "model": name, **score(d["home_win"], p)})
    return preds, pd.DataFrame(rows)


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    t = d.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


@dataclass
class Explanation:
    feature: str
    label: str
    logodds: float


# Correlated features are grouped so explanations don't show offsetting pieces
# (e.g. "offense helps home" and "passing hurts home" from the same underlying signal).
FACTOR_GROUPS = {
    "Team strength (Elo + point diff)": ["elo_diff", "pt_diff_diff"],
    "Starting QB edge": ["qb_diff"],
    "Offensive efficiency": ["off_epa_diff", "off_sr_diff", "pass_epa_diff", "rush_epa_diff"],
    "Defensive efficiency": ["def_epa_diff", "def_sr_diff"],
    "Turnover margin": ["to_margin_diff"],
    "Rest advantage": ["rest_diff"],
    "Home field": ["home_field"],
    "Division game": ["div_game"],
}


def explain_logistic(model, X: pd.DataFrame, top: int = 3) -> list[list[Explanation]]:
    """Per-game contribution of each factor group, in log-odds, vs. a neutral-site average matchup.

    For standardized logistic regression, contribution_i = coef_i * z_i; together with the
    intercept these sum exactly to the model's log-odds. The intercept is the league-wide
    home edge, so it is credited to "Home field" (for non-neutral games).
    """
    scaler, lr = model[0], model[-1]
    z = scaler.transform(X[FEATURES])
    contrib = pd.DataFrame(z * lr.coef_[0], columns=FEATURES, index=X.index)
    contrib["home_field"] += lr.intercept_[0]  # baseline; ~0 net for neutral sites
    grouped = pd.DataFrame({g: contrib[cols].sum(axis=1) for g, cols in FACTOR_GROUPS.items()})
    out = []
    for _, row in grouped.iterrows():
        order = row.abs().sort_values(ascending=False).index[:top]
        out.append([Explanation(g, g, float(row[g])) for g in order])
    return out


def coefficients(model) -> dict:
    lr = model[-1]
    return {f: float(c) for f, c in zip(FEATURES, lr.coef_[0])} | {"intercept": float(lr.intercept_[0])}


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
