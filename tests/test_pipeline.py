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
