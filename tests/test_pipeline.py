import numpy as np
import pandas as pd
import pytest

from nflpred import data, features as F, model as M

SEASONS = range(2020, 2025)


@pytest.fixture(scope="session")
def raw():
    g = data.load_schedules()
    g = g[g["season"].between(SEASONS.start, SEASONS.stop - 1)]
    p = data.load_pbp(SEASONS)
    inj = (data.load_injuries(SEASONS), data.load_snaps(SEASONS), data.load_players())
    return g, p, inj


@pytest.fixture(scope="session")
def feats(raw):
    return F.build_features(*raw)


# ---------------------------------------------------------------- leakage
@pytest.mark.parametrize("cutoff", ["2022-10-09", "2023-12-24", "2024-09-08"])
def test_no_future_leakage(raw, feats, cutoff):
    """Features for games on `cutoff` must be identical if every result from `cutoff` onward
    is erased. If a feature changes, it was using information from the game itself or later."""
    g, p, (inj, snaps, players) = raw
    cut = pd.Timestamp(cutoff)
    g2 = g.copy()
    future = pd.to_datetime(g2["gameday"]) >= cut
    g2.loc[future, ["home_score", "away_score", "result", "total"]] = np.nan
    future_ids = set(g2.loc[future, "game_id"])
    p2 = p[~p["game_id"].isin(future_ids)]
    snaps2 = snaps[~snaps["game_id"].isin(future_ids)]
    # Injury reports for the cutoff date's games are pre-game info; later weeks' are not.
    wk = g2.loc[pd.to_datetime(g2["gameday"]) == cut, ["season", "week"]].drop_duplicates()
    later = pd.to_datetime(g2["gameday"]) > cut
    later_keys = set(map(tuple, g2.loc[later, ["season", "week"]].drop_duplicates().values)) \
        - set(map(tuple, wk.values))
    inj2 = inj[~inj[["season", "week"]].apply(tuple, axis=1).isin(later_keys)]
    masked = F.build_features(g2, p2, (inj2, snaps2, players)).set_index("game_id")
    full = feats.set_index("game_id")
    ids = full.index[full["gameday"] == cut]
    assert len(ids) > 0
    pd.testing.assert_frame_equal(full.loc[ids, F.FEATURES], masked.loc[ids, F.FEATURES],
                                  check_exact=False, atol=1e-9)


def test_backtest_trains_only_on_prior_seasons(feats):
    for s in (2023, 2024):
        assert M.train_rows(feats, before_season=s)["season"].max() < s


# ---------------------------------------------------------------- schema / sanity
def test_feature_schema(feats):
    assert set(F.FEATURES) <= set(feats.columns)
    assert feats["game_id"].is_unique
    assert not feats[F.FEATURES].isna().any().any()
    done = feats[feats["home_win"].notna()]
    assert done["home_win"].isin([0, 1]).all()
    assert 0.5 < done["home_win"].mean() < 0.6  # home teams win ~52-57%


def test_features_have_expected_direction(feats):
    d = feats[(feats["season"] >= 2021) & feats["home_win"].notna()]
    for f in ["elo_diff", "qb_diff", "off_epa_diff", "def_epa_diff", "pt_diff_diff",
              "qb_change_diff", "inj_off_diff"]:
        assert d[f].corr(d["home_win"]) > 0.05, f


def test_vegas_prob_is_vig_free():
    h = pd.Series([-150.0, 120.0, -110.0])
    a = pd.Series([130.0, -140.0, -110.0])
    p = F.vegas_prob(h, a)
    assert np.allclose(p + F.vegas_prob(a, h), 1)
    assert p.iloc[2] == pytest.approx(0.5)


def test_explanations_sum_to_predicted_margin(feats):
    m = M.fit(feats[feats["season"] < 2024])
    X = feats[feats["season"] == 2024].head(20)
    total = M.contributions(m, X).sum(axis=1).values
    assert np.allclose(total, m.predict_margin(X), atol=1e-8)


def test_margin_probability_is_consistent(feats):
    m = M.fit(feats[feats["season"] < 2024])
    X = feats[feats["season"] == 2024].head(50)
    p, mg = M.predict(m, X), m.predict_margin(X)
    assert ((p > 0.5) == (mg > 0)).all()
    assert 11 < m.sigma < 15  # NFL margins have ~13-point spread around expectations


def test_model_beats_coin_flip(feats):
    _, summ = M.backtest(feats, [2023, 2024], kinds=("margin",))
    ll = summ.query("season == 'ALL' and model == 'margin'")["log_loss"].item()
    assert ll < 0.67


def test_injury_snap_share_uses_only_prior_games():
    """A Doubtful player who then plays must be valued on PRIOR snaps, not that game's snaps."""
    from nflpred.injuries import team_injury_load
    sched = pd.DataFrame({
        "game_id": ["g1", "g2"], "season": [2024, 2024], "week": [1, 2],
        "gameday": pd.to_datetime(["2024-09-08", "2024-09-15"]),
        "home_team": ["AAA", "AAA"], "away_team": ["BBB", "CCC"]})
    snaps = pd.DataFrame({"game_id": ["g1", "g2"], "pfr_player_id": ["p1", "p1"],
                          "offense_pct": [0.2, 1.0], "defense_pct": [0.0, 0.0]})
    inj = pd.DataFrame({"season": [2024], "week": [2], "team": ["AAA"], "gsis_id": ["x1"],
                        "position": ["WR"], "report_status": ["Doubtful"]})
    players = pd.DataFrame({"gsis_id": ["x1"], "pfr_id": ["p1"]})
    r = team_injury_load(inj, snaps, players, sched).set_index(["game_id", "team"])
    assert r.loc[("g2", "AAA"), "inj_off"] == pytest.approx(0.8 * 0.2)


def test_clv_measures_line_move_toward_model(tmp_path):
    import json
    from nflpred.pipeline import clv_report
    df = pd.DataFrame({"game_id": ["a", "b"], "completed": [True, True],
                       "vegas_home_prob": [0.60, 0.40]})
    snap = {"upcoming": [
        {"game_id": "a", "home_team": "H1", "away_team": "A1", "vegas_home_prob": 0.55, "home_win_prob": 0.65},
        {"game_id": "b", "home_team": "H2", "away_team": "A2", "vegas_home_prob": 0.45, "home_win_prob": 0.50},
    ]}
    (tmp_path / "predictions_2026-01-01.json").write_text(json.dumps(snap))
    r = clv_report(df, tmp_path)
    # a: model liked home, line moved 55->60 toward home: +5. b: model liked home, line moved away: -5.
    assert r["games"] == 2 and r["avg_clv_pts"] == pytest.approx(0.0)
    assert sorted(x["clv_pts"] for x in r["detail"]) == [-5.0, 5.0]


# ---------------------------------------------------------------- context sources (offline)
def test_odds_summary_consensus_and_best_price():
    from nflpred import odds
    ev = [{"home_team": "Chicago Bears", "away_team": "Philadelphia Eagles",
           "commence_time": "2026-09-29T00:15:00Z", "bookmakers": [
               {"title": "BookA", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": 170},
                                               {"name": "Philadelphia Eagles", "price": -205}]},
                   {"key": "spreads", "outcomes": [{"name": "Chicago Bears", "price": -110, "point": 4.5},
                                                   {"name": "Philadelphia Eagles", "price": -110, "point": -4.5}]}]},
               {"title": "BookB", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": 180},
                                               {"name": "Philadelphia Eagles", "price": -215}]}]}]}]
    s = odds.summarize(ev)
    g = s[("CHI", "PHI", "2026-09-29")]
    assert g["books"] == 2 and g["best_home_ml"] == {"price": 180, "book": "BookB"}
    assert g["best_away_ml"]["book"] == "BookA"
    assert 0.33 < g["consensus_home_prob"] < 0.37 and g["consensus_home_margin"] == -4.5
    import datetime as dt
    assert odds.match(s, "CHI", "PHI", dt.date(2026, 9, 28)) is g  # 8:15pm ET = next UTC day


def test_weather_picks_kickoff_hour():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from nflpred import weather
    payload = {"hourly": {"time": ["2026-10-04T16:00", "2026-10-04T17:00", "2026-10-04T18:00"],
                          "temperature_2m": [60, 62, 64], "wind_speed_10m": [5, 18, 9],
                          "wind_gusts_10m": [9, 27, 12], "precipitation_probability": [0, 40, 10]}}
    ko = weather._kickoff_utc(pd.Timestamp("2026-10-04"), "13:00")  # 1pm EDT = 17:00 UTC
    assert ko == datetime(2026, 10, 4, 17, tzinfo=ZoneInfo("UTC"))
    assert weather.parse_forecast(payload, ko) == {"temp_f": 62, "wind_mph": 18, "gust_mph": 27, "precip_pct": 40}


def test_travel_west_coast_team_early_kickoff(feats):
    g = feats[(feats["away_team"] == "SEA") & (feats["gametime"] == "13:00")
              & (feats["home_team"].isin(["NYG", "NYJ", "CAR", "ATL", "WAS", "MIA", "BUF", "PHI"]))]
    assert len(g) > 0
    assert (g["away_tz_shift"] == 3).all() and (g["away_body_hour"] == 10).all() and (g["away_km"] > 3000).all()


def test_qb_availability_blend():
    from nflpred import qb_availability as qba
    assert qba.play_prob("Questionable", "Did Not") == 0.42
    assert qba.play_prob(None, None) == 1.0 and qba.play_prob("Out", "Did Not") == 0.0
    games = pd.DataFrame({"game_id": ["g"], "home_qb_status": ["Questionable"], "home_qb_practice": ["Limited"],
                          "away_qb_status": [None], "away_qb_practice": [None]})
    av = qba.availability(games, {"g": {"away_qb_play_prob": 0.5}})
    assert av.loc[0, "home_qb_play_prob"] == 0.53 and av.loc[0, "away_qb_play_prob"] == 0.5
    # healthy QBs => blend equals the plain prediction; replacement only ever lowers that side
    fake = lambda m, g: 1 / (1 + np.exp(-(g["qb_diff"].values * 10)))
    g = pd.DataFrame({"qb_diff": [0.2], "qb_change_diff": [0.0], "final_week": [0], "fw_qb_diff": [0.0],
                      "home_qb_rating": [0.1], "away_qb_rating": [-0.1]})
    healthy = pd.DataFrame({"home_qb_play_prob": [1.0], "away_qb_play_prob": [1.0]})
    hurt = pd.DataFrame({"home_qb_play_prob": [0.4], "away_qb_play_prob": [1.0]})
    assert qba.blended_prob(None, g, healthy, fake)[0] == pytest.approx(fake(None, g)[0])
    assert qba.blended_prob(None, g, hurt, fake)[0] < fake(None, g)[0]


def test_blowout_cap_limits_single_game_influence():
    assert F.PT_CAP == 21 and F.EPA_CAP == 0.3


# ---------------------------------------------------------------- live news (offline fixtures)
SLEEPER_FIXTURE = {
    "1": {"full_name": "Caleb Williams", "team": "CHI", "position": "QB", "gsis_id": "00-0039918",
          "injury_status": "Out", "practice_participation": "DNP", "depth_chart_position": "QB",
          "depth_chart_order": 1, "news_updated": 1759000000000},
    "2": {"full_name": "Tyson Bagent", "team": "CHI", "position": "QB", "gsis_id": "00-0038416",
          "injury_status": "Questionable", "practice_participation": "DNP", "depth_chart_position": "QB",
          "depth_chart_order": 2},
    "3": {"full_name": "Case Keenum", "team": "CHI", "position": "QB", "gsis_id": "00-0029076",
          "injury_status": None, "depth_chart_position": "QB", "depth_chart_order": 3},
    "4": {"full_name": "Dallas Goedert", "team": "PHI", "position": "TE", "gsis_id": "00-0034272",
          "injury_status": "IR", "depth_chart_position": "TE", "depth_chart_order": 1},
    "5": {"full_name": "Free Agent", "team": None, "position": "WR"},
    "6": {"full_name": "Matthew Stafford", "team": "LAR", "position": "QB", "gsis_id": "00-0026498",
          "injury_status": None, "depth_chart_position": "QB", "depth_chart_order": 1},
}


def test_sleeper_parsing_and_team_codes():
    from nflpred import news
    d = news.parse_sleeper(SLEEPER_FIXTURE)
    assert len(d) == 5  # free agent dropped
    assert d.set_index("full_name").loc["Matthew Stafford", "team"] == "LA"
    assert d.set_index("full_name").loc["Dallas Goedert", "report_status"] == "Out"  # IR -> Out
    assert d.set_index("full_name").loc["Tyson Bagent", "practice_status"] == "Did Not Participate In Practice"


def test_projected_starter_skips_out_qb_and_sets_backup():
    from nflpred import news
    live = news.parse_sleeper(SLEEPER_FIXTURE)
    st = news.projected_starters(live)["CHI"]
    assert st["starter"] == "Tyson Bagent" and st["backup"] == "Case Keenum"
    games = pd.DataFrame({"game_id": ["g"], "season": [2026], "week": [3], "home_team": ["CHI"], "away_team": ["PHI"],
                          "home_qb_id": ["00-0039918"], "home_qb_name": ["Caleb Williams"],
                          "away_qb_id": ["00-0035704"], "away_qb_name": ["Jalen Hurts"], "home_score": [np.nan]})
    g2, log = news.apply_to_schedule(games, news.projected_starters(live), live, {"g"})
    assert g2.at[0, "home_qb_name"] == "Tyson Bagent" and g2.at[0, "home_backup_qb_id"] == "00-0029076"
    assert g2.at[0, "away_qb_name"] == "Jalen Hurts"  # PHI not in fixture depth chart: untouched
    assert len(log) == 1 and "Tyson Bagent projected to start" in log[0]["text"]


def test_live_injuries_replace_nflverse_for_that_week_only():
    from nflpred import news
    live = news.parse_sleeper(SLEEPER_FIXTURE)
    up = pd.DataFrame({"season": [2026], "week": [3], "home_team": ["CHI"], "away_team": ["PHI"]})
    rows = news.injury_rows(live, up)
    old = pd.DataFrame({"season": [2026, 2026], "week": [3, 2], "team": ["CHI", "CHI"],
                        "gsis_id": ["00-0038416", "00-0038416"], "position": ["QB", "QB"],
                        "report_status": ["Questionable", "Questionable"],
                        "practice_status": ["Full Participation in Practice"] * 2})
    merged = news.merge_injuries(old, rows)
    wk3 = merged[(merged.week == 3) & (merged.gsis_id == "00-0038416")]
    assert wk3["practice_status"].tolist() == ["Did Not Participate In Practice"]  # live wins
    assert len(merged[merged.week == 2]) == 1  # older week untouched
    log = news.status_changes(old, rows, up)
    assert any("Caleb Williams" in c["text"] and "Out" in c["text"] for c in log)


def test_espn_summary_parsing():
    from nflpred import news
    summ = {"injuries": [{"team": {"abbreviation": "WSH"}, "injuries": [
        {"status": "Out", "date": "2026-09-27T18:00Z",
         "athlete": {"id": "123", "displayName": "Some Tackle", "position": {"abbreviation": "OT"}}}]}]}
    d = news.parse_espn_summary(summ, {"123": "00-0099999"})
    assert d.iloc[0][["team", "gsis_id", "report_status", "source"]].tolist() == ["WAS", "00-0099999", "Out", "ESPN"]
    both = news.combine(news.parse_sleeper(SLEEPER_FIXTURE), d)
    assert (both["full_name"] == "Some Tackle").sum() == 1


def test_backup_rating_used_when_starter_sits():
    from nflpred import qb_availability as qba
    g = pd.DataFrame({"qb_diff": [0.0], "qb_change_diff": [0.0], "final_week": [0], "fw_qb_diff": [0.0],
                      "home_qb_rating": [0.0], "away_qb_rating": [0.0], "home_backup_qb_rating": [-0.3]})
    assert qba.with_replacement(g, "home").at[0, "qb_diff"] == pytest.approx(-0.3)
    assert qba.with_replacement(g, "away").at[0, "qb_diff"] == pytest.approx(0.1)  # no backup known: -0.10


# ---------------------------------------------------------------- paper betting
def _game(p_home=0.60, cons=0.50, home_ml=+110, away_ml=-120, qb_ok=True, inj=True, gid="2026_05_AAA_BBB"):
    return {"game_id": gid, "season": 2026, "week": 5, "gameday": "2026-10-11",
            "home_team": "BBB", "away_team": "AAA", "home_win_prob": p_home, "injury_report": inj,
            "qb_status": {"home": {"play_prob": 1.0 if qb_ok else 0.53}, "away": {"play_prob": 1.0}},
            "context": {"live_odds": {"consensus_home_prob": cons,
                                      "best_home_ml": {"price": home_ml, "book": "BookA"},
                                      "best_away_ml": {"price": away_ml, "book": "BookB"}}}}


def test_bet_rules_are_frozen_and_complete():
    from nflpred import bets
    r = bets.load_rules()
    assert r["version"] == 1 and r["validation"]["min_bets"] == 200
    assert 0 < r["probability"]["model"] < 1.5 and 0 < r["probability"]["vegas"] < 1.5


def test_bet_qualifies_only_with_edge_and_safety_checks():
    from nflpred import bets
    r = bets.load_rules()
    bet = bets.evaluate(_game(), r, None)
    assert bet and bet["team"] == "BBB" and bet["price"] == 110 and bet["book"] == "BookA"
    assert bet["edge"] >= 0.03 and 0.25 <= bet["units"] <= 2.0
    assert 0.5 < bet["p_blend"] < 0.6  # blend sits between market (0.50) and model (0.60)
    assert bets.evaluate(_game(p_home=0.51), r, None) is None          # no edge
    assert bets.evaluate(_game(qb_ok=False), r, None) is None          # QB questionable
    assert bets.evaluate(_game(inj=False), r, None) is None            # injury report not out
    assert bets.evaluate(_game(), r, first_market=0.56) is None        # line moved 6 pts against BBB


def test_bet_locks_once_and_grades_with_clv(tmp_path):
    from nflpred import bets
    games = pd.DataFrame({"game_id": ["2026_05_AAA_BBB"], "completed": [False], "home_score": [np.nan],
                          "away_score": [np.nan], "vegas_home_prob": [0.55]})
    out1 = bets.process({"upcoming": [_game()]}, games, tmp_path)
    assert len(out1["new"]) == 1 and out1["mode"] == "shadow"
    out2 = bets.process({"upcoming": [_game(home_ml=+150)]}, games, tmp_path)  # better price later
    assert out2["new"] == [] and out2["open"][0]["price"] == 110                 # still the locked bet
    games.loc[0, ["completed", "home_score", "away_score"]] = [True, 24, 17]
    out3 = bets.process({"upcoming": []}, games, tmp_path)
    g = out3["recent_graded"][0]
    assert g["result"] == "win" and g["profit_units"] == pytest.approx(g["units"] * 1.10, abs=1e-3)
    assert g["clv"] == pytest.approx(2.10 * 0.55 - 1, abs=1e-4)  # +110 vs closing fair 55%
    assert out3["record"]["graded"] == 1 and not out3["record"]["passed"]


def test_validation_requires_all_checks():
    from nflpred import bets
    r = bets.load_rules()
    mk = lambda clv, res: {"status": "graded", "rules_version": 1, "units": 1.0, "result": res,
                           "profit_units": 0.95 if res == "win" else -1.0, "clv": clv}
    good = [mk(0.03 + 0.01 * (i % 3), "win" if i % 2 else "loss") for i in range(250)]
    rec = bets.record(good, r)
    assert rec["checks"]["enough_bets"] and rec["checks"]["clv_positive_and_significant"]
    assert not rec["checks"]["roi_positive"] and not rec["passed"]  # 125-125 at -105 loses money
    few = bets.record(good[:50], r)
    assert not few["checks"]["enough_bets"]


def test_outputs_are_strict_json(tmp_path):
    """NaN in predictions.json broke the dashboard once (missing QB name). Never again."""
    import json
    M.save_json({"a": float("nan"), "b": [np.float64("nan"), 1.5], "c": {"d": np.int64(3)}}, tmp_path / "x.json")
    assert json.loads((tmp_path / "x.json").read_text(), parse_constant=lambda c: pytest.fail(c)) == \
        {"a": None, "b": [None, 1.5], "c": {"d": 3}}


def test_weekly_exposure_cap(tmp_path):
    from nflpred import bets
    games = pd.DataFrame({"game_id": [], "completed": [], "home_score": [], "away_score": [], "vegas_home_prob": []})
    ups = [_game(p_home=0.75, cons=0.50, gid=f"2026_05_A{i}_B{i}") for i in range(8)]
    out = bets.process({"upcoming": ups}, games, tmp_path)
    assert sum(b["units"] for b in out["new"]) <= 8.0 and len(out["new"]) < 8
    assert any("weekly exposure cap" in "; ".join(g["bet_check"]) for g in ups)
