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
from . import odds as odds_lib, weather as weather_lib
from . import qb_availability as qba
from . import news as news_lib
from . import bets as bets_lib
from . import spread_bets as spread_lib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
BASELINE = ROOT / "model_baseline.json"
BACKTEST_SEASONS = range(2018, 2026)
GATE_TOLERANCE = 0.002  # allowed log-loss regression before CI fails


def current_season(today: dt.date) -> int:
    return today.year if today.month >= 8 else today.year - 1


LAST_NEWS: dict = {}


def official_reports(official: pd.DataFrame) -> set:
    """(season, week, team) with a published official game-status report."""
    o = official[official["report_status"].notna()]
    team = o["team"].replace({"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"})
    return set(zip(o["season"].astype(int), o["week"].astype(int), team))


def build(refresh: bool, today: dt.date, live_news: bool = False, horizon_days: int = 9,
          news_sources: tuple = ("sleeper", "espn")) -> pd.DataFrame:
    """Load data and build features. With live_news, upcoming games get the latest injury
    statuses and projected starting QBs from Sleeper/ESPN (see news.py)."""
    season = current_season(today)
    games = data.load_schedules(refresh=refresh)
    seasons = range(2012, season + 1)
    cur = season if refresh else None
    pbp = data.load_pbp(seasons, refresh_current=cur)
    injuries, snaps, players = data.load_injuries(seasons, cur), data.load_snaps(seasons, cur), data.load_players(refresh)
    LAST_NEWS.clear()
    official_before_live = injuries
    if live_news:
        gd = pd.to_datetime(games["gameday"]).dt.date
        up = games[games["home_score"].isna() & (gd >= today) & (gd <= today + dt.timedelta(days=horizon_days))]
        live, report = news_lib.fetch_live(up, players, news_sources)
        try:
            report["news_log_new_entries"] = news_lib.log_first_seen(live, ROOT / "history")
        except Exception as e:
            print("news log failed:", e)
        log = []
        if not live.empty:
            rows = news_lib.injury_rows(live, up)
            log += news_lib.status_changes(injuries, rows, up)
            games, starter_log = news_lib.apply_to_schedule(
                games, news_lib.projected_starters(live), live, set(up["game_id"]))
            log += starter_log
            injuries = news_lib.merge_injuries(injuries, rows)
            report["live_injury_rows"] = int(len(rows))
        LAST_NEWS.update({"report": report, "changes": log})
        print("news:", json.dumps(report))
        for c in log:
            print("news:", c["text"])
    df = F.build_features(games, pbp, (injuries, snaps, players))
    # "Injury report out" means the OFFICIAL game-status report (Out/Doubtful/Questionable designations,
    # published Wed-Fri) exists for that team and week. Live feeds and practice-only reports don't count.
    rep = official_reports(official_before_live)
    for side in ("home", "away"):
        keys = list(zip(df["season"], df["week"], df[f"{side}_team"]))
        df[f"{side}_inj_reported"] = [1.0 if k in rep else 0.0 for k in keys]
    return df


def _pts_to_prob_points(pts: float, sigma: float) -> float:
    """Approximate win-probability points (vs. a coin flip) for a margin contribution."""
    return float(100 * (M._norm_cdf(abs(pts) / sigma) - 0.5))


def _s(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def _opt(v, nd=4):
    return None if pd.isna(v) else round(float(v), nd)


def _dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def game_context(g, forecast: dict | None, live: dict | None, p_home: float) -> dict:
    """Informational context shown with each pick. Travel and weather were tested as model
    inputs and did not improve out-of-sample accuracy, so they are displayed, not modeled."""
    ctx = {"travel": {
        "away_km": round(float(g.get("away_km", 0) or 0)), "home_km": round(float(g.get("home_km", 0) or 0)),
        "away_tz_shift": float(g.get("away_tz_shift", 0) or 0), "home_tz_shift": float(g.get("home_tz_shift", 0) or 0),
        "away_body_hour": float(g.get("away_body_hour", 13) or 13), "home_body_hour": float(g.get("home_body_hour", 13) or 13),
    }}
    if g.get("roof") in ("dome", "closed"):
        ctx["weather"] = {"indoors": True}
    elif forecast:
        ctx["weather"] = {"indoors": False, **forecast}
    if live:
        ctx["live_odds"] = dict(live)
        # EV uses the same model + market blend as the paper-bet rules (raw model edges are overconfident).
        w = bets_lib.load_rules()["probability"]
        pb = bets_lib.blend_prob(p_home, live["consensus_home_prob"], w)
        ctx["live_odds"]["blend_home_prob"] = round(pb, 4)
        for side, p in (("home", pb), ("away", 1 - pb)):
            b = live.get(f"best_{side}_ml")
            if b:
                ctx["live_odds"][f"{side}_ev_at_best"] = round(p * _dec(b["price"]) - 1, 4)
    return ctx


def predict_games(model: M.MarginModel, games: pd.DataFrame, forecasts: dict | None = None,
                  live: dict | None = None) -> list[dict]:
    avail = qba.availability(games, qba.load_overrides())
    p_full = M.predict(model, games)
    p = qba.blended_prob(model, games, avail, M.predict)  # == p_full when both QBs are healthy
    margin = model.predict_margin(games)
    expl = M.explain(model, games)
    out = []
    for (_, g), ph, pf, mg, ex, (_, av) in zip(games.iterrows(), p, p_full, margin, expl, avail.iterrows()):
        ctx = game_context(
            g, (forecasts or {}).get(g["game_id"]),
            odds_lib.match(live, g["home_team"], g["away_team"], g["gameday"].date()) if live else None, ph)
        lo = ctx.get("live_odds")
        out.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "gameday": g["gameday"].date().isoformat(), "gametime": g.get("gametime"),
            "kickoff_utc": weather_lib._kickoff_utc(g["gameday"], g.get("gametime")).isoformat(),
            "home_team": g["home_team"], "away_team": g["away_team"],
            "neutral_site": bool(g["location"] == "Neutral"),
            "home_qb": _s(g.get("home_qb_name")), "away_qb": _s(g.get("away_qb_name")),
            "qb_change": {"home": _opt(g.get("home_qb_change"), 3), "away": _opt(g.get("away_qb_change"), 3)},
            "home_win_prob": round(float(ph), 4), "away_win_prob": round(float(1 - ph), 4),
            "qb_status": {side: {"status": _s(g.get(f"{side}_qb_status")), "practice": _s(g.get(f"{side}_qb_practice")),
                                 "play_prob": round(float(av[f"{side}_qb_play_prob"]), 2),
                                 "win_prob_if_starts": round(float(pf if side == "home" else 1 - pf), 4)}
                          for side in ("home", "away")},
            "pick": g["home_team"] if ph >= 0.5 else g["away_team"],
            # Spreads as "home margin": +3 means home favored by 3.
            "model_home_margin": round(float(mg), 1),
            "vegas_home_margin": _opt(g.get("spread_line"), 1),
            "vegas_home_prob": _opt(g["vegas_home_prob"]),
            "injury_report": bool(g.get("home_inj_reported", 0) and g.get("away_inj_reported", 0)),
            "home_win": None if pd.isna(g["home_win"]) else int(g["home_win"]),
            "home_score": None if pd.isna(g["home_score"]) else int(g["home_score"]),
            "away_score": None if pd.isna(g["away_score"]) else int(g["away_score"]),
            # Line when we published: live multi-book consensus if available, else nflverse line.
            "line_at_publish": lo["consensus_home_prob"] if lo else _opt(g["vegas_home_prob"]),
            "context": ctx,
            "top_factors": [
                {"factor": e.factor, "favors": g["home_team"] if e.points > 0 else g["away_team"],
                 "points": round(abs(e.points), 1),
                 "prob_points": round(_pts_to_prob_points(e.points, model.sigma), 1)}
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
        m = M.fit(df, before_date=d["gameday"].min())
        d = d.copy()
        d["p_model"] = M.predict(m, d)
        parts.append(d)
    return pd.concat(parts) if parts else done.assign(p_model=[])


def clv_report(df: pd.DataFrame, history_dir: Path = ROOT / "history") -> dict:
    """Closing line value: did the market move toward the model's side after we published?

    For each finished game, take the EARLIEST saved prediction (usually Tuesday), note the
    Vegas line at that moment, and compare it with the closing line now in the schedule.
    Consistently positive movement is the standard evidence of a real betting edge; it needs
    far fewer games to detect than win/loss results.
    """
    first: dict[str, dict] = {}
    for f in sorted(history_dir.glob("predictions_*.json")):
        for g in json.loads(f.read_text()).get("upcoming", []):
            if g.get("line_at_publish", g.get("vegas_home_prob")) is not None and g["game_id"] not in first:
                first[g["game_id"]] = g
    close = df.set_index("game_id")
    rows = []
    for gid, g in first.items():
        if gid not in close.index or not close.at[gid, "completed"] or pd.isna(close.at[gid, "vegas_home_prob"]):
            continue
        open_p, close_p, model_p = g.get("line_at_publish", g.get("vegas_home_prob")), float(close.at[gid, "vegas_home_prob"]), g["home_win_prob"]
        if abs(model_p - open_p) < 0.02:
            continue  # model agrees with the market; no side
        side = 1 if model_p > open_p else -1
        rows.append({"game_id": gid, "side": g["home_team"] if side > 0 else g["away_team"],
                     "line_at_pick": open_p, "closing": close_p, "model": model_p,
                     "clv_pts": round(100 * side * (close_p - open_p), 2)})
    if not rows:
        return {"games": 0}
    c = np.array([r["clv_pts"] for r in rows])
    return {"games": len(rows), "avg_clv_pts": round(float(c.mean()), 2),
            "pct_toward_model": round(float((c > 0).mean()), 3),
            "pct_away_from_model": round(float((c < 0).mean()), 3), "detail": rows}


def cmd_backtest(df: pd.DataFrame) -> dict:
    preds, summ = M.backtest(df, BACKTEST_SEASONS)
    allrows = summ[summ["season"] == "ALL"].set_index("model")
    res = {
        "seasons": f"{BACKTEST_SEASONS.start}-{BACKTEST_SEASONS.stop - 1}",
        "overall": allrows.drop(columns="season").to_dict(orient="index"),
        "by_season": summ[summ["season"] != "ALL"].to_dict(orient="records"),
        "calibration": M.calibration_table(preds["home_win"], preds[f"p_{M.PRODUCTION}"]).to_dict(orient="records"),
    }
    OUT.mkdir(exist_ok=True)
    M.save_json(res, OUT / "backtest.json")
    return res


def cmd_gate(res: dict) -> int:
    new = res["overall"][M.PRODUCTION]["log_loss"]
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


def cmd_update(df: pd.DataFrame, today: dt.date, horizon_days: int = 9, offline: bool = False) -> dict:
    season = current_season(today)
    model = M.fit(df)
    upcoming = df[(~df["completed"]) & (df["gameday"].dt.date >= today)
                  & (df["gameday"].dt.date <= today + dt.timedelta(days=horizon_days))]
    if today == dt.date.today():  # live run: drop games that have already kicked off
        now = dt.datetime.now(dt.timezone.utc)
        started = [weather_lib._kickoff_utc(g, t) <= now for g, t in zip(upcoming["gameday"], upcoming["gametime"])]
        upcoming = upcoming[[not x for x in started]]
    forecasts = {} if offline else weather_lib.forecasts_for(upcoming)
    live = None if offline else odds_lib.snapshot(ROOT / "history")
    std = season_to_date(df, season)
    std_score = M.score(std["home_win"], std["p_model"]) if len(std) else {}
    vegas_score = M.score(std["home_win"], std["vegas_home_prob"]) if len(std) else {}
    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "upcoming": predict_games(model, upcoming, forecasts, live),
        "live_odds_available": bool(live),
        "news": dict(LAST_NEWS),
        "season_to_date": {"model": std_score, "vegas": vegas_score,
                           "games": predict_games_with_p(std)},
        "clv": clv_report(df),
        "coefficients": M.coefficients(model),
        "trained_on_games": int(len(M.train_rows(df))),
    }
    try:
        result["bets"] = bets_lib.process(result, df, ROOT / "history")
    except Exception as e:  # paper betting must never block predictions
        result["bets"] = {"error": str(e)}
        print("bets: failed:", e)
    try:
        result["spread_bets"] = spread_lib.process(result, df, ROOT / "history")
    except Exception as e:
        result["spread_bets"] = {"error": str(e)}
        print("spread bets: failed:", e)
    try:
        from . import ml_v2
        result["ml_v2_bets"] = ml_v2.process(result, df, ROOT / "history")
    except Exception as e:
        result["ml_v2_bets"] = {"error": str(e)}
        print("moneyline v2 bets: failed:", e)
    try:
        from . import ml_v2 as _v
        result["ml_v3_bets"] = _v.process_v3(result, df, ROOT / "history")
    except Exception as e:
        result["ml_v3_bets"] = {"error": str(e)}
        print("moneyline v3 bets: failed:", e)
    try:
        from . import night_west
        result["night_west_bets"] = night_west.process(result, df, ROOT / "history")
    except Exception as e:
        result["night_west_bets"] = {"error": str(e)}
        print("night west bets: failed:", e)
    try:
        from . import totals as totals_lib
        result["totals_wind_bets"] = totals_lib.process(result, df, ROOT / "history")
    except Exception as e:
        result["totals_wind_bets"] = {"error": str(e)}
        print("totals wind bets: failed:", e)
    OUT.mkdir(exist_ok=True)
    M.save_json(result, OUT / "predictions.json")
    alert = OUT / "alert.md"
    alert.unlink(missing_ok=True)
    lines = []
    b = result.get("bets", {})
    if b.get("mode") == "live":
        lines += [f"- **{x['team']}** moneyline {x['price']:+d} at {x['book']} vs {x['opponent']} "
                  f"({x['gameday']}): edge {x['edge']:+.1%}, stake {x['units']}u" for x in b.get("new", [])]
    sb = result.get("spread_bets", {})
    if sb.get("mode") == "live":
        lines += [f"- **{x['team']} {x['point']:+g}** ({x['price']:+d}) at {x['book']} vs {x['opponent']} "
                  f"({x['gameday']}): edge {x['edge']:+.1%}, stake {x['units']}u" for x in sb.get("new", [])]
    nw = result.get("night_west_bets", {})
    if nw.get("mode") == "live":
        lines += [f"- **{x['team']} {x['point']:+g}** ({x['price']:+d}) at {x['book']} vs {x['opponent']} "
                  f"({x['gameday']}): night-game body clock, stake {x['units']}u" for x in nw.get("new", [])]
    tw = result.get("totals_wind_bets", {})
    if tw.get("mode") == "live":
        lines += [f"- **{x['team']} {x['point']}** ({x['price']:+d}) at {x['book']} ({x['gameday']}): "
                  f"forecast wind {x['forecast_wind_mph']:.0f} mph, stake {x['units']}u" for x in tw.get("new", [])]
    for x in (result.get("ml_v3_bets", {}).get("new", []) if result.get("ml_v3_bets", {}).get("mode") == "live" else []):
        lines.append(f"- **{x['team']}** moneyline {x['price']:+d} at {x['book']} vs {x['opponent']} ({x['gameday']}): "
                     f"{x['edge']:+.1%} vs Pinnacle/sharp fair, stake {x['units']}u")
    v2 = result.get("ml_v2_bets", {})
    if v2.get("mode") == "live":
        lines += [f"- **{x['team']}** moneyline {x['price']:+d} at {x['book']} vs {x['opponent']} "
                  f"({x['gameday']}): {x['edge']:+.1%} vs sharp books, stake {x['units']}u" for x in v2.get("new", [])]
    if lines:
        alert.write_text("New qualifying bets (validated track):\n\n" + "\n".join(lines))
    return result


def predict_games_with_p(d: pd.DataFrame) -> list[dict]:
    return [{"game_id": r.game_id, "week": int(r.week), "home_team": r.home_team, "away_team": r.away_team,
             "home_win_prob": round(float(r.p_model), 4),
             "vegas_home_prob": None if pd.isna(r.vegas_home_prob) else round(float(r.vegas_home_prob), 4),
             "home_score": int(r.home_score), "away_score": int(r.away_score),
             "correct": bool((r.p_model > 0.5) == (r.home_win == 1))}
            for r in d.itertuples()]


def cmd_watch(now: dt.datetime | None = None) -> dict | None:
    """Hourly odds watch (no retraining): refresh live prices on the last published predictions and
    run the tracks whose edge depends on catching prices quickly (moneyline v2, forecast-wind
    unders). The v1 tracks only act on full runs, as pre-registered."""
    from . import ml_v2, night_west, totals as totals_lib
    path = OUT / "predictions.json"
    if not path.exists():
        print("watch: no predictions.json yet")
        return None
    pred = json.loads(path.read_text())
    now = now or dt.datetime.now(dt.timezone.utc)
    if not any(g.get("kickoff_utc") and dt.datetime.fromisoformat(g["kickoff_utc"]) > now
               for g in pred.get("upcoming", [])):
        print("watch: no upcoming games, skipping odds call (saves credits)")
        return None
    live = odds_lib.snapshot(ROOT / "history")
    if not live:
        print("watch: no odds")
        return None
    ups = []
    for g in pred.get("upcoming", []):
        if g.get("kickoff_utc") and dt.datetime.fromisoformat(g["kickoff_utc"]) <= now:
            continue  # already kicked off
        lo = odds_lib.match(live, g["home_team"], g["away_team"], dt.date.fromisoformat(g["gameday"]))
        ctx = g.setdefault("context", {})
        if lo:
            w = bets_lib.load_rules()["probability"]
            lo = dict(lo)
            pb = bets_lib.blend_prob(g["home_win_prob"], lo["consensus_home_prob"], w)
            lo["blend_home_prob"] = round(pb, 4)
            for side, p in (("home", pb), ("away", 1 - pb)):
                if lo.get(f"best_{side}_ml"):
                    lo[f"{side}_ev_at_best"] = round(p * _dec(lo[f"best_{side}_ml"]["price"]) - 1, 4)
            ctx["live_odds"] = lo
        ups.append(g)
    pred["upcoming"] = ups
    pred["odds_checked_at"] = now.isoformat(timespec="minutes")
    no_games = pd.DataFrame(columns=["game_id", "completed"])  # grading happens on full runs
    for key, fn in (("ml_v2_bets", ml_v2.process), ("ml_v3_bets", ml_v2.process_v3), ("totals_wind_bets", totals_lib.process),
                    ("night_west_bets", night_west.process)):
        try:
            res = fn(pred, no_games, ROOT / "history")
            prev = pred.get(key) or {}
            res["recent_graded"] = prev.get("recent_graded", res.get("recent_graded", []))
            pred[key] = res
            for b in res.get("new", []):
                print(f"watch: new {key} paper bet {b['id']} {b['price']:+d} at {b['book']}")
        except Exception as e:
            print(f"watch: {key} failed: {e}")
    M.save_json(pred, path)
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["update", "backtest", "gate", "watch"])
    ap.add_argument("--no-refresh", action="store_true", help="use cached data")
    ap.add_argument("--today", help="override date (YYYY-MM-DD)")
    ap.add_argument("--offline", action="store_true", help="skip live odds, weather and news APIs")
    ap.add_argument("--news-sources", default="sleeper,espn", help="comma list: sleeper,espn")
    a = ap.parse_args(argv)
    if a.cmd == "watch":
        cmd_watch()
        return 0
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    df = build(refresh=not a.no_refresh, today=today, live_news=(a.cmd == "update" and not a.offline),
               news_sources=tuple(x.strip() for x in a.news_sources.split(",") if x.strip()))
    if a.cmd == "update":
        r = cmd_update(df, today, offline=a.offline)
        for g in r["upcoming"]:
            print(f"{g['gameday']} {g['away_team']:>3} @ {g['home_team']:<3}  home {g['home_win_prob']:.0%}"
                  f"  (Vegas {g['vegas_home_prob'] if g['vegas_home_prob'] is not None else 'n/a'})")
        return 0
    res = cmd_backtest(df)
    print(json.dumps(res["overall"], indent=2))
    return cmd_gate(res) if a.cmd == "gate" else 0


if __name__ == "__main__":
    sys.exit(main())
