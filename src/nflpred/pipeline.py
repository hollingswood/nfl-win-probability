"""End-to-end pipeline.

    python -m nflpred.pipeline update      # refresh data, retrain, predict upcoming games, grade
    python -m nflpred.pipeline backtest    # season-forward backtest vs Vegas (writes output/backtest.json)
    python -m nflpred.pipeline gate        # fail (exit 1) if backtest log loss regressed vs baseline
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features as F, model as M

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
BASELINE = ROOT / "model_baseline.json"
BACKTEST_SEASONS = range(2018, 2026)
GATE_TOLERANCE = 0.002  # allowed log-loss regression before CI fails


def current_season(today: dt.date) -> int:
    return today.year if today.month >= 8 else today.year - 1


def build(refresh: bool, today: dt.date) -> pd.DataFrame:
    season = current_season(today)
    games = data.load_schedules(refresh=refresh)
    pbp = data.load_pbp(range(2012, season + 1), refresh_current=season if refresh else None)
    return F.build_features(games, pbp)


def _to_prob(logodds: float) -> float:
    return float(1 / (1 + np.exp(-logodds)))


def predict_games(model, games: pd.DataFrame) -> list[dict]:
    p = model.predict_proba(games[F.FEATURES])[:, 1]
    expl = M.explain_logistic(model, games)
    out = []
    for (_, g), ph, ex in zip(games.iterrows(), p, expl):
        fav_home = ph >= 0.5
        out.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "gameday": g["gameday"].date().isoformat(), "gametime": g.get("gametime"),
            "home_team": g["home_team"], "away_team": g["away_team"],
            "neutral_site": bool(g["location"] == "Neutral"),
            "home_qb": g.get("home_qb_name"), "away_qb": g.get("away_qb_name"),
            "home_win_prob": round(float(ph), 4), "away_win_prob": round(float(1 - ph), 4),
            "pick": g["home_team"] if fav_home else g["away_team"],
            "vegas_home_prob": None if pd.isna(g["vegas_home_prob"]) else round(float(g["vegas_home_prob"]), 4),
            "home_win": None if pd.isna(g["home_win"]) else int(g["home_win"]),
            "home_score": None if pd.isna(g["home_score"]) else int(g["home_score"]),
            "away_score": None if pd.isna(g["away_score"]) else int(g["away_score"]),
            # Positive = favors home team. Converted to approx. probability points vs a coin flip.
            "top_factors": [
                {"factor": e.label, "favors": g["home_team"] if e.logodds > 0 else g["away_team"],
                 "logodds": round(e.logodds, 3),
                 "prob_points": round(100 * (_to_prob(abs(e.logodds)) - 0.5), 1)}
                for e in ex
            ],
        })
    return out


def season_to_date(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Walk-forward predictions for completed games this season: each week's model is trained
    only on games before that week's first kickoff (what we would have published live)."""
    done = df[(df["season"] == season) & df["home_win"].notna()]
    parts = []
    for wk, d in done.groupby("week"):
        m = M.fit(df, "logistic", before_date=d["gameday"].min())
        d = d.copy()
        d["p_model"] = m.predict_proba(d[F.FEATURES])[:, 1]
        parts.append(d)
    return pd.concat(parts) if parts else done.assign(p_model=[])


def cmd_backtest(df: pd.DataFrame) -> dict:
    preds, summ = M.backtest(df, BACKTEST_SEASONS)
    allrows = summ[summ["season"] == "ALL"].set_index("model")
    res = {
        "seasons": f"{BACKTEST_SEASONS.start}-{BACKTEST_SEASONS.stop - 1}",
        "overall": allrows.drop(columns="season").to_dict(orient="index"),
        "by_season": summ[summ["season"] != "ALL"].to_dict(orient="records"),
        "calibration_logistic": M.calibration_table(preds["home_win"], preds["p_logistic"]).to_dict(orient="records"),
    }
    OUT.mkdir(exist_ok=True)
    M.save_json(res, OUT / "backtest.json")
    return res


def cmd_gate(res: dict) -> int:
    new = res["overall"]["logistic"]["log_loss"]
    if not BASELINE.exists():
        print(f"No baseline; writing {new:.4f}")
        BASELINE.write_text(json.dumps({"log_loss": new}, indent=2))
        return 0
    base = json.loads(BASELINE.read_text())["log_loss"]
    print(f"backtest log loss: new={new:.4f} baseline={base:.4f} tolerance={GATE_TOLERANCE}")
    if new > base + GATE_TOLERANCE:
        print("FAIL: model regressed")
        return 1
    print("PASS")
    return 0


def cmd_update(df: pd.DataFrame, today: dt.date, horizon_days: int = 9) -> dict:
    season = current_season(today)
    model = M.fit(df, "logistic")
    upcoming = df[(~df["completed"]) & (df["gameday"].dt.date >= today)
                  & (df["gameday"].dt.date <= today + dt.timedelta(days=horizon_days))]
    std = season_to_date(df, season)
    std_score = M.score(std["home_win"], std["p_model"]) if len(std) else {}
    vegas_score = M.score(std["home_win"], std["vegas_home_prob"]) if len(std) else {}
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "upcoming": predict_games(model, upcoming),
        "season_to_date": {"model": std_score, "vegas": vegas_score,
                           "games": predict_games_with_p(std)},
        "coefficients": M.coefficients(model),
        "trained_on_games": int(len(M.train_rows(df))),
    }
    OUT.mkdir(exist_ok=True)
    M.save_json(result, OUT / "predictions.json")
    return result


def predict_games_with_p(d: pd.DataFrame) -> list[dict]:
    return [{"game_id": r.game_id, "week": int(r.week), "home_team": r.home_team, "away_team": r.away_team,
             "home_win_prob": round(float(r.p_model), 4),
             "vegas_home_prob": None if pd.isna(r.vegas_home_prob) else round(float(r.vegas_home_prob), 4),
             "home_score": int(r.home_score), "away_score": int(r.away_score),
             "correct": bool((r.p_model > 0.5) == (r.home_win == 1))}
            for r in d.itertuples()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "backtest", "gate"])
    ap.add_argument("--no-refresh", action="store_true", help="use cached data")
    ap.add_argument("--today", help="override date (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    df = build(refresh=not a.no_refresh, today=today)
    if a.cmd == "update":
        r = cmd_update(df, today)
        for g in r["upcoming"]:
            print(f"{g['gameday']} {g['away_team']:>3} @ {g['home_team']:<3}  home {g['home_win_prob']:.0%}"
                  f"  (Vegas {g['vegas_home_prob'] if g['vegas_home_prob'] is not None else 'n/a'})")
        return 0
    res = cmd_backtest(df)
    print(json.dumps(res["overall"], indent=2))
    return cmd_gate(res) if a.cmd == "gate" else 0


if __name__ == "__main__":
    sys.exit(main())
