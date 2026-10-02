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
    g2.loc[future, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
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


def test_injury_report_flag_uses_official_report_only():
    """Live feeds carry last week's statuses; they must not count as this week's report being out."""
    from nflpred.pipeline import official_reports
    official = pd.DataFrame({"season": [2026, 2026], "week": [4.0, 4.0], "team": ["CHI", "PHI"],
                             "report_status": [None, "Out"]})  # CHI: practice-only row so far
    rep = official_reports(official)
    assert (2026, 4, "PHI") in rep and (2026, 4, "CHI") not in rep


def test_void_bets_are_ignored(tmp_path):
    import json
    from nflpred import bets
    (tmp_path / "paper_bets.json").write_text(json.dumps([{**bets.evaluate(_game(), bets.load_rules(), None),
                                                           "status": "void"}]))
    games = pd.DataFrame({"game_id": [], "completed": [], "home_score": [], "away_score": [], "vegas_home_prob": []})
    out = bets.process({"upcoming": [_game()]}, games, tmp_path)
    assert len(out["new"]) == 1 and out["record"]["graded"] == 0  # void bet doesn't block or count


# ---------------------------------------------------------------- spread track + key numbers
def test_key_numbers_make_3_and_7_likely_and_ties_rare():
    from nflpred import spread_bets as sb, margins as K
    r = sb.load_rules()
    p = K.pmf(np.array([-3.0]), r["margin"]["sigma"], r["_weights"])[0]
    at = lambda k: p[K.KS == k][0] + p[K.KS == -k][0]
    assert at(3) > 1.8 * at(2) and at(7) > at(8) and at(3) > at(4)
    assert p[K.KS == 0][0] < 0.01
    hc, pu, ac = K.cover_probs(np.array([0.0]), 13.0, np.array([-3.0]), r["_weights"])
    assert pu[0] > 0.05 and abs(hc[0] + pu[0] + ac[0] - 1) < 1e-9  # pushes on 3 are common


def test_spread_best_line_across_books_and_grading(tmp_path):
    from nflpred import spread_bets as sb
    r = sb.load_rules()
    g = _game(p_home=0.6)
    g["model_home_margin"] = 14.0  # big disagreement; blend (mostly market) still ~5.8
    g["context"]["live_odds"].update({"consensus_home_margin": 3.0, "spreads_by_book": [
        {"book": "A", "home_point": -3.5, "home_price": -110, "away_point": 3.5, "away_price": -110},
        {"book": "B", "home_point": -3.0, "home_price": -115, "away_point": 3.0, "away_price": -105}]})
    a = sb.analyze(g, r)
    assert a["best"]["side"] == "home" and a["best"]["book"] == "B"   # -3 at -115 beats -3.5 at -110 near a key number
    assert len(a["best"]["buy_options"]) == 2 and a["best"]["buy_options"][0]["point"] == -2.5
    games = pd.DataFrame({"game_id": [g["game_id"]], "completed": [False], "home_score": [np.nan],
                          "away_score": [np.nan], "spread_line": [3.5]})
    out = sb.process({"upcoming": [g]}, games, tmp_path)
    bet = out["new"][0]
    assert bet["point"] == -3.0 and bet["book"] == "B"
    games.loc[0, ["completed", "home_score", "away_score"]] = [True, 20, 17]   # won by exactly 3 -> push
    out2 = sb.process({"upcoming": []}, games, tmp_path)
    graded = out2["recent_graded"][0]
    assert graded["result"] == "push" and graded["profit_units"] == 0.0
    assert graded["clv_points"] == pytest.approx(0.5)  # we got -3, market closed -3.5


def test_buy_point_price_shift():
    from nflpred.spread_bets import _shift_price
    assert _shift_price(-110, 20) == -130 and _shift_price(105, 20) == -115 and _shift_price(150, 10) == 140


def test_odds_summary_keeps_each_books_spread():
    from nflpred import odds
    ev = [{"home_team": "Chicago Bears", "away_team": "Philadelphia Eagles", "commence_time": "2026-09-29T00:15:00Z",
           "bookmakers": [{"title": "BookA", "markets": [
               {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": 170}, {"name": "Philadelphia Eagles", "price": -205}]},
               {"key": "spreads", "outcomes": [{"name": "Chicago Bears", "price": -110, "point": 4.5},
                                               {"name": "Philadelphia Eagles", "price": -110, "point": -4.5}]}]}]}]
    s = odds.summarize(ev)[("CHI", "PHI", "2026-09-29")]
    assert s["spreads_by_book"] == [{"book": "BookA", "home_point": 4.5, "home_price": -110,
                                     "away_point": -4.5, "away_price": -110}]


def test_line_shopping_only_uses_allowed_books():
    from nflpred import odds
    ev = [{"home_team": "Chicago Bears", "away_team": "Philadelphia Eagles", "commence_time": "2026-09-29T00:15:00Z",
           "bookmakers": [
               {"key": "offshore", "title": "Offshore", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": 200}, {"name": "Philadelphia Eagles", "price": -190}]},
                   {"key": "spreads", "outcomes": [{"name": "Chicago Bears", "price": -105, "point": 5.0},
                                                   {"name": "Philadelphia Eagles", "price": -105, "point": -5.0}]}]},
               {"key": "draftkings", "title": "DraftKings", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": 170}, {"name": "Philadelphia Eagles", "price": -205}]},
                   {"key": "spreads", "outcomes": [{"name": "Chicago Bears", "price": -110, "point": 4.5},
                                                   {"name": "Philadelphia Eagles", "price": -110, "point": -4.5}]}]}]}]
    s = odds.summarize(ev, allowed={"draftkings"})[("CHI", "PHI", "2026-09-29")]
    assert s["best_home_ml"]["book"] == "DraftKings" and s["books"] == 2      # consensus still uses both
    assert [b["book"] for b in s["spreads_by_book"]] == ["DraftKings"]


# ---------------------------------------------------------------- grades
def test_grades_reward_confirmation_and_penalize_warning_signs():
    from nflpred import grading as G
    base = G.grade("spread", 0.06, 2.0, None, 3.5)                    # +2 edge, +1 right side of 3
    assert base["grade"] == "A" and base["score"] == 3
    assert G.grade("spread", 0.06, 2.0, 1.0, 3.5)["grade"] == "A+"    # line moved our way
    assert G.grade("spread", 0.06, 9.0, None, 3.5)["score"] == 2      # model far from market
    assert G.grade("spread", 0.06, 2.0, None, 2.5)["score"] == 1      # wrong side of 3
    assert G.grade("spread", 0.06, 2.0, None, 3.5, qb_flag=True)["score"] == 2
    assert G.grade("moneyline", 0.40, 5.0, None)["score"] == 2        # huge edge is capped at +2
    assert G.letter(-3) == "C"


def test_grade_performance_table():
    from nflpred import grading as G
    led = [{"status": "graded", "grade": "A", "units": 1, "result": "win", "profit_units": 0.9, "clv": 0.02},
           {"status": "graded", "grade": "A", "units": 1, "result": "loss", "profit_units": -1.0, "clv": 0.00},
           {"status": "graded", "grade": "C", "units": 1, "result": "loss", "profit_units": -1.0},
           {"status": "open", "grade": "A", "units": 1}]
    t = {r["grade"]: r for r in G.by_grade(led)}
    assert t["A"]["bets"] == 2 and t["A"]["roi"] == pytest.approx(-0.05) and t["A"]["avg_clv"] == pytest.approx(0.01)
    assert t["C"]["avg_clv"] is None


def test_every_paper_bet_carries_a_grade(tmp_path):
    from nflpred import bets
    games = pd.DataFrame({"game_id": [], "completed": [], "home_score": [], "away_score": [], "vegas_home_prob": []})
    g = _game()
    out = bets.process({"upcoming": [g]}, games, tmp_path)
    assert out["new"][0]["grade"] in {"A+", "A", "B+", "B", "C+", "C"}
    assert g["moneyline"]["verdict"] == "bet" and "p_needed" in g["moneyline"]


def test_odds_history_plan_and_backfill(tmp_path):
    """Historical odds: snapshots only when a game is upcoming; rows parsed; resumable; reserve respected."""
    import gzip, csv
    from datetime import datetime, timezone
    from nflpred import odds_history as OH
    games = pd.DataFrame({"season": [2023, 2023], "gameday": pd.to_datetime(["2023-09-10", "2023-09-17"]),
                          "gametime": ["13:00", "20:20"]})
    plan = OH.plan_snapshots(games, 2023)
    assert plan[0] == datetime(2023, 9, 1, 21, 40, tzinfo=timezone.utc)          # first run within 9 days of kickoff
    assert datetime(2023, 9, 8, 21, 40, tzinfo=timezone.utc) in plan               # Friday run
    assert datetime(2023, 9, 10, 15, 45, tzinfo=timezone.utc) in plan              # 75 min before 1pm ET
    assert datetime(2023, 9, 18, 14, 10, tzinfo=timezone.utc) not in plan          # nothing left to play
    payload = {"timestamp": "2023-09-01T14:05:00Z", "data": [{
        "id": "e1", "commence_time": "2023-09-10T17:00:00Z", "home_team": "Washington Football Team",
        "away_team": "Arizona Cardinals", "bookmakers": [{"key": "draftkings", "title": "DraftKings",
        "last_update": "x", "markets": [
            {"key": "h2h", "outcomes": [{"name": "Washington Football Team", "price": -300},
                                        {"name": "Arizona Cardinals", "price": 250}]},
            {"key": "spreads", "outcomes": [{"name": "Washington Football Team", "price": -110, "point": -7},
                                            {"name": "Arizona Cardinals", "price": -110, "point": 7}]}]}]}]}
    calls = []
    def fake(key, t, regions, markets):
        calls.append(t)
        return payload, 100000 - 40 * len(calls)
    res = OH.backfill(games, [2023], "k", out_dir=tmp_path, fetcher=fake)
    assert res["fetched"] == len(plan) and not res["stopped"]
    with gzip.open(tmp_path / "nfl_odds_2023.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["home"] == "WAS" and rows[0]["ml_away"] == "250" and rows[0]["sp_home_point"] == "-7"
    n = len(calls)
    OH.backfill(games, [2023], "k", out_dir=tmp_path, fetcher=fake)                 # resume: nothing refetched
    assert len(calls) == n
    (tmp_path / "done_2023.txt").unlink()
    res = OH.backfill(games, [2023], "k", out_dir=tmp_path, fetcher=lambda *a: (payload, 1030), reserve=1000)
    assert res["fetched"] == 0 or res["stopped"]


def test_spread_clv_uses_closing_prices(tmp_path):
    """Spread CLV comes from our last odds snapshot before kickoff, prices included; a later
    (post-kickoff) snapshot is ignored and +3 at -120 is not treated like +3 at -105."""
    import json as _j
    from nflpred import spread_bets as SB
    r = SB.load_rules()
    def ev(price_home, price_away, point_home=-3):
        return [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos",
                 "commence_time": "2026-10-04T20:25:00Z", "bookmakers": [{"key": "draftkings", "markets": [
                     {"key": "spreads", "outcomes": [
                         {"name": "Kansas City Chiefs", "point": point_home, "price": price_home},
                         {"name": "Denver Broncos", "point": -point_home, "price": price_away}]}]}]}]
    (tmp_path / "odds_2026-10-01T1410.json").write_text(_j.dumps(ev(-110, -110)))
    (tmp_path / "odds_2026-10-04T1910.json").write_text(_j.dumps(ev(-125, 105)))   # the close
    (tmp_path / "odds_2026-10-04T2300.json").write_text(_j.dumps(ev(-300, 250)))   # after kickoff: ignored
    closes = SB.closing_margins(tmp_path, r)
    c = closes[("KC", "DEN")][0]
    assert c["ts"].startswith("2026-10-04T19:10") and c["mu"] > 3.2
    games = pd.DataFrame([{"game_id": "g", "completed": True, "home_score": 24, "away_score": 20,
                           "home_team": "KC", "away_team": "DEN", "gameday": pd.Timestamp("2026-10-04"),
                           "spread_line": 3.0}])
    bet = {"game_id": "g", "side": "away", "point": 3.0, "price": -105, "units": 1.0}
    g = SB.grade(bet, games, r, closes)
    # closing juice made DEN +3 worth less than even: our -105 did NOT beat the close
    assert g["clv"] < 0 and "prices included" in g["clv_source"]
    assert "clv" not in SB.grade(bet, games, r, {})   # no closing snapshot -> no CLV claimed


def test_ml_v2_soft_vs_sharp(tmp_path):
    """Moneyline v2 bets only when an allowed book beats the sharp no-vig price by 2%+ and the model agrees."""
    from nflpred import odds as O, ml_v2
    ev = [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos", "commence_time": "2026-10-04T20:25:00Z",
           "bookmakers": [
               {"key": "lowvig", "title": "LowVig", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "Kansas City Chiefs", "price": -200}, {"name": "Denver Broncos", "price": 190}]}]},
               {"key": "betonlineag", "title": "BetOnline", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "Kansas City Chiefs", "price": -205}, {"name": "Denver Broncos", "price": 185}]}]},
               {"key": "draftkings", "title": "DraftKings", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "Kansas City Chiefs", "price": -240}, {"name": "Denver Broncos", "price": 215}]}]}]}]
    s = O.summarize(ev, {"draftkings"})
    lo = next(iter(s.values()))
    assert lo["sharp_books"] == 2 and 0.64 < lo["sharp_home_prob"] < 0.67
    game = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "KC", "away_team": "DEN",
            "home_win_prob": 0.62, "injury_report": True, "qb_status": {}, "context": {"live_odds": lo}}
    bet = ml_v2.evaluate(game, ml_v2.load_rules())
    assert bet and bet["team"] == "DEN" and bet["price"] == 215 and bet["edge"] >= 0.02   # ~34.5% fair at +215
    game2 = dict(game, home_win_prob=0.75, context={"live_odds": lo})
    assert ml_v2.evaluate(game2, ml_v2.load_rules()) is None and "model disagrees" in game2["ml_v2_check"]
    game3 = dict(game, injury_report=False, context={"live_odds": lo})
    assert ml_v2.evaluate(game3, ml_v2.load_rules()) is None


def test_totals_wind_track_and_closing(tmp_path):
    """Wind-under track: bets the best under only with a >=15 mph outdoor forecast within 6 days;
    CLV from our own closing totals snapshot (prices included)."""
    import gzip, json as _j
    from datetime import datetime, timezone
    from nflpred import totals as T
    d = T.default_dist()
    w, pu = d.probs(44.0, 44.0, "under")
    assert 0.02 < pu < 0.06 and abs(d.implied_mu(44.5, 0.5) - 44.5) < 1.5
    tot = {"consensus_total": 44.5, "sharp_total": 44.4, "median_point": 44.5,
           "totals_by_book": [{"book": "DraftKings", "point": 44.5, "over_price": -110, "under_price": -110},
                              {"book": "FanDuel", "point": 45.5, "over_price": -105, "under_price": -115}]}
    base = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "CHI", "away_team": "GB",
            "kickoff_utc": "2026-10-04T17:00:00+00:00"}
    r = T.load_rules()
    now = datetime(2026, 9, 29, 14, 10, tzinfo=timezone.utc)
    g = dict(base, context={"live_odds": {"totals": tot}, "weather": {"indoors": False, "wind_mph": 18}})
    bet = T.evaluate(g, r, d, now)
    assert bet and bet["point"] == 45.5 and bet["book"] == "FanDuel"            # best under = higher number
    calm = dict(base, context={"live_odds": {"totals": tot}, "weather": {"indoors": False, "wind_mph": 8}})
    assert T.evaluate(calm, r, d, now) is None
    dome = dict(base, context={"live_odds": {"totals": tot}, "weather": {"indoors": True}})
    assert T.evaluate(dome, r, d, now) is None
    early = dict(base, context={"live_odds": {"totals": tot}, "weather": {"indoors": False, "wind_mph": 20}})
    assert T.evaluate(early, r, d, datetime(2026, 9, 26, 14, 10, tzinfo=timezone.utc)) is None   # 8 days out
    ev = [{"home_team": "Chicago Bears", "away_team": "Green Bay Packers", "commence_time": "2026-10-04T17:00:00Z",
           "bookmakers": [{"key": "draftkings", "markets": [{"key": "totals", "outcomes": [
               {"name": "Over", "point": 42.5, "price": -110}, {"name": "Under", "point": 42.5, "price": -110}]}]}]}]
    with gzip.open(tmp_path / "odds_2026-10-04T1545.json.gz", "wt") as f:
        f.write(_j.dumps(ev))
    closes = T.closing_totals(tmp_path, d)
    games = pd.DataFrame([{"game_id": "g", "completed": True, "home_score": 20, "away_score": 17,
                           "home_team": "CHI", "away_team": "GB", "gameday": pd.Timestamp("2026-10-04")}])
    out = T.grade(bet, games, d, closes)
    assert out["result"] == "win" and out["clv"] > 0      # 45.5 under vs a 42.5 close = big CLV


def test_news_first_seen_log(tmp_path):
    from nflpred import news
    from datetime import datetime, timezone
    live = pd.DataFrame([{"gsis_id": "00-1", "full_name": "A QB", "team": "CHI", "position": "QB",
                          "report_status": "Questionable", "practice_status": "Limited", "depth_order": 1, "source": "Sleeper"}])
    t0 = datetime(2026, 10, 1, 14, tzinfo=timezone.utc)
    assert news.log_first_seen(live, tmp_path, t0) == 1
    assert news.log_first_seen(live, tmp_path, t0) == 0                     # unchanged -> nothing new
    live.loc[0, "report_status"] = "Out"
    assert news.log_first_seen(live, tmp_path, t0) == 1
    assert len((tmp_path / "news_log.jsonl").read_text().splitlines()) == 2


def test_summary_pinnacle_exchanges_and_all_book_consensus():
    from nflpred import odds as O
    def bk(key, mh, ma, sh):
        return {"key": key, "title": key, "markets": [
            {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": mh}, {"name": "Denver Broncos", "price": ma}]},
            {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": sh, "price": -110},
                                            {"name": "Denver Broncos", "point": -sh, "price": -110}]}]}
    ev = [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos", "commence_time": "2026-10-04T20:25:00Z",
           "bookmakers": [bk("draftkings", -200, 170, -4.5), bk("pinnacle", -190, 175, -3.5),
                          bk("kalshi", -185, 180, -3.5), bk("lowvig", -190, 172, -3.5)]}]
    lo = next(iter(O.summarize(ev, {"draftkings"}).values()))
    assert lo["pinnacle_home_prob"] is not None and lo["best_exchange_away_ml"]["price"] == 180
    assert lo["best_away_ml"]["book"] == "draftkings"          # best price only from allowed books
    assert lo["consensus_home_margin"] == 3.5                   # consensus spread uses ALL books


def test_night_west_track():
    from datetime import datetime, timezone
    from nflpred import night_west as NW
    r = NW.load_rules()
    lo = {"consensus_home_margin": 3.0, "spreads_by_book": [
        {"book": "DraftKings", "home_point": -3.0, "home_price": -110, "away_point": 3.0, "away_price": -110},
        {"book": "FanDuel", "home_point": -2.5, "home_price": -120, "away_point": 3.5, "away_price": -105}]}
    g = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "NYG", "away_team": "SF",
         "kickoff_utc": "2026-10-05T00:20:00+00:00", "context": {"live_odds": lo}}    # 8:20pm ET, SF (Pacific) at NYG
    bet = NW.evaluate(g, r, datetime(2026, 10, 4, 22, 0, tzinfo=timezone.utc))
    assert bet and bet["team"] == "SF" and bet["point"] == 3.5 and bet["book"] == "FanDuel"
    assert NW.evaluate(dict(g, context={"live_odds": lo}), r, datetime(2026, 10, 4, 14, 0, tzinfo=timezone.utc)) is None  # too early
    day = dict(g, kickoff_utc="2026-10-04T17:00:00+00:00", context={"live_odds": lo})
    assert NW.evaluate(day, r, datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)) is None   # 1pm ET: not a night game
    same = dict(g, away_team="DAL", home_team="KC", context={"live_odds": lo})
    assert NW.evaluate(same, r, datetime(2026, 10, 4, 22, 0, tzinfo=timezone.utc)) is None  # same zone
    rec = NW.record([{"status": "graded", "rules_version": 1, "result": "win", "units": 1, "profit_units": 0.95, "price": -105}], r)
    assert rec["wins"] == 1 and not rec["passed"]


def test_ml_v3_uses_pinnacle_blend(tmp_path):
    from nflpred import odds as O, ml_v2
    def bk(key, mh, ma):
        return {"key": key, "title": key, "markets": [{"key": "h2h", "outcomes": [
            {"name": "Kansas City Chiefs", "price": mh}, {"name": "Denver Broncos", "price": ma}]}]}
    ev = [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos", "commence_time": "2026-10-04T20:25:00Z",
           "bookmakers": [bk("pinnacle", -200, 180), bk("lowvig", -200, 190), bk("betonlineag", -205, 185),
                          bk("draftkings", -240, 215)]}]
    lo = next(iter(O.summarize(ev, {"draftkings"}).values()))
    assert lo["pin_sharp_home_prob"] is not None
    g = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "KC", "away_team": "DEN",
         "home_win_prob": 0.62, "injury_report": True, "qb_status": {}, "context": {"live_odds": lo}}
    out = ml_v2.process_v3({"upcoming": [g]}, pd.DataFrame(columns=["game_id", "completed"]), tmp_path)
    assert out["track"] == "moneyline_v3" and (tmp_path / "paper_bets_ml_v3.json").exists()
    assert g.get("ml_v3") and "ml_v2" not in g


def test_news_llm_scan_dedupes_and_logs(tmp_path):
    from datetime import datetime, timezone
    from nflpred import news_llm as NL
    items = [{"title": "Bears QB Smith ruled out Sunday", "link": "u1", "published": None, "summary": "Smith (ankle) is out."},
             {"title": "Game recap", "link": "u2", "published": None, "summary": "..."}]
    calls = []
    def fake_llm(batch, key):
        calls.append(len(batch))
        return [{"item": 0, "player": "J. Smith", "team": "CHI", "position": "QB", "signal": "out",
                 "is_starting_qb_news": True, "game_week_relevant": True, "certainty": 1.0, "quote": "ruled out"},
                {"item": 1, "player": "x", "team": "XXX"}]              # invalid team dropped
    t = datetime(2026, 10, 1, 16, tzinfo=timezone.utc)
    assert NL.scan(tmp_path, api_key=None)["skipped"]
    r = NL.scan(tmp_path, "k", t, fetcher=lambda u: items, llm=fake_llm)
    assert r["signals"] == 1 and r["qb_signals"] == ["J. Smith (CHI): out"]
    r2 = NL.scan(tmp_path, "k", t, fetcher=lambda u: items, llm=fake_llm)
    assert r2["new_items"] == 0 and len(calls) == 1                     # nothing re-sent to the model


# ---------------------------------------------------------------- totals model (display only)
def test_totals_model_no_future_leakage(raw, feats):
    """Totals features and walk-forward predictions for games on the cutoff date must not change
    when every result from the cutoff onward is erased."""
    from nflpred import totals_model as T
    g, p, (inj, snaps, players) = raw
    cut = pd.Timestamp("2023-12-24")
    g2 = g.copy()
    future = pd.to_datetime(g2["gameday"]) >= cut
    g2.loc[future, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
    future_ids = set(g2.loc[future, "game_id"])
    wk = g2.loc[pd.to_datetime(g2["gameday"]) == cut, ["season", "week"]].drop_duplicates()
    later = set(map(tuple, g2.loc[pd.to_datetime(g2["gameday"]) > cut, ["season", "week"]].drop_duplicates().values)) \
        - set(map(tuple, wk.values))
    inj2 = inj[~inj[["season", "week"]].apply(tuple, axis=1).isin(later)]
    masked = T.add_features(F.build_features(g2, p[~p["game_id"].isin(future_ids)],
                                             (inj2, snaps[~snaps["game_id"].isin(future_ids)], players)))
    full = T.add_features(feats)
    ids = full.loc[full["gameday"] == cut, "game_id"]
    assert len(ids) > 0
    a, b = full.set_index("game_id").loc[ids], masked.set_index("game_id").loc[ids]
    pd.testing.assert_frame_equal(a[T.TOTAL_FEATURES], b[T.TOTAL_FEATURES], check_exact=False, atol=1e-9)
    pa = T.predict(T.fit(full, before_date=cut), a.reset_index())
    pb = T.predict(T.fit(masked, before_date=cut), b.reset_index())
    np.testing.assert_allclose(pa, pb, atol=1e-9)


def test_totals_model_predictions_finite_and_sane(feats):
    from nflpred import totals_model as T
    t = T.add_features(feats)
    m = T.fit(t, before_season=2024)
    assert T.train_rows(t, before_season=2024)["season"].max() < 2024
    test = t[t["season"] == 2024]
    pred = T.predict(m, test)
    assert np.isfinite(pred).all() and 30 < pred.mean() < 55 and pred.std() > 1
    wf = T.walk_forward(t, 2024)
    assert np.isfinite(wf.values).all() and set(wf.index) <= set(test["game_id"])
    rep = T.holdout_report(t, range(2023, 2025))
    assert rep["games"] > 400 and 8 < rep["mae_model"] < 14 and 8 < rep["mae_market"] < 14


# ---------------------------------------------------------------- dashboard
def _load_build_dashboard():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dash_predictions(with_totals: bool) -> dict:
    up = {"game_id": "2026_05_AAA_BBB", "season": 2026, "week": 5, "gameday": "2026-10-04", "gametime": "13:00",
          "home_team": "BBB", "away_team": "AAA", "home_win_prob": 0.6, "away_win_prob": 0.4,
          "model_home_margin": 3.1, "vegas_home_margin": 2.5, "vegas_home_prob": 0.58, "injury_report": True,
          "top_factors": [], "context": {"weather": {"indoors": False, "temp_f": 50, "wind_mph": 18, "gust_mph": 25}}}
    std = {"game_id": "2026_01_AAA_BBB", "week": 1, "home_team": "BBB", "away_team": "AAA",
           "home_win_prob": 0.6, "vegas_home_prob": 0.55, "home_score": 24, "away_score": 20, "correct": True}
    if with_totals:
        up["model_total"] = 44.5
        up["context"]["live_odds"] = {"consensus_home_prob": 0.58, "totals": {
            "consensus_total": 45.0, "sharp_total": 44.5,
            "totals_by_book": [{"book": "FanDuel", "point": 44.5, "over_price": -110, "under_price": -110}]}}
        up["totals_view"] = {"forecast_wind_mph": 18, "consensus_total": 45.0, "verdict": "pass",
                             "best_under": {"book": "FanDuel", "point": 44.5, "price": -110, "ev": 0.01},
                             "reasons": ["injury report not out"]}
        up["ml_v2"] = {"team": "AAA", "price": 150, "book": "FanDuel", "p_sharp": 0.41, "p_needed": 0.4,
                       "ev_sharp": 0.025, "verdict": "lean", "reasons": ["model disagrees"]}
        up["night_west"] = {"team": "AAA", "point": 2.5, "price": -110, "book": "FanDuel", "zones": "AAA P @ BBB E",
                            "verdict": "pending", "reasons": ["bets only in the last 3 h before kickoff"]}
        std.update(model_home_margin=3.0, spread_line=2.5, model_total=44.0, total_line=45.5)
    return {"generated_at": "2026-10-01T12:00:00+00:00", "season": 2026, "upcoming": [up],
            "season_to_date": {"model": {"n": 1, "log_loss": 0.5, "accuracy": 1.0},
                               "vegas": {"n": 1, "log_loss": 0.6, "accuracy": 1.0}, "games": [std]},
            "clv": {"games": 0}, "trained_on_games": 3000, "live_odds_available": with_totals}


@pytest.mark.parametrize("present", [True, False])
def test_dashboard_renders_with_and_without_news_and_totals(tmp_path, present):
    import datetime as dt
    import json
    import re
    bd = _load_build_dashboard()
    now = dt.datetime(2026, 10, 1, 12, tzinfo=dt.timezone.utc)
    news = tmp_path / "news_llm.jsonl"
    rows = [
        {"seen_at": "2026-09-30T10:00+00:00", "team": "AAA", "player": "QB One", "position": "QB", "signal": "out",
         "is_starting_qb_news": True, "game_week_relevant": True, "certainty": 0.9, "source": "cbs",
         "link": "https://example.com/a", "title": "QB One out"},
        {"seen_at": "2026-09-30T11:00+00:00", "team": "BBB", "player": "WR Two", "signal": "questionable",
         "is_starting_qb_news": False, "game_week_relevant": True, "certainty": 0.6, "source": "yahoo",
         "link": "javascript:alert(1)", "title": "</script><script>x</script>"},
        {"seen_at": "2026-09-01T11:00+00:00", "team": "BBB", "player": "Old News", "signal": "out", "certainty": 1.0},
        {"seen_at": "2026-09-30T11:00+00:00", "team": "ZZZ", "player": "Other Team", "signal": "out"},
    ]
    news.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n" if present else "")
    signals = bd.load_news_signals(news if present else tmp_path / "missing.jsonl")
    pred = _dash_predictions(present)
    backtest = {"overall": {"margin": {"accuracy": 0.66, "n": 2000}, "vegas": {"accuracy": 0.67, "n": 2000}},
                "seasons": "2018-2025", "calibration": [], "by_season": []}
    payload = bd.build_payload(pred, backtest, None, signals, now=now)
    if present:
        got = payload["news_ai"]["2026_05_AAA_BBB"]
        assert [s["player"] for s in got] == ["QB One", "WR Two"]  # QB news first; old and other-team news dropped
    else:
        assert payload["news_ai"] == {}
    body = bd.render(payload)
    assert "__DATA__" not in body and "</script><script>x" not in body
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', body, re.S).group(1))
    assert data["predictions"]["upcoming"][0]["game_id"] == "2026_05_AAA_BBB"
    assert bd.write_site(body, tmp_path / "site").exists()


def test_ml_v4_spread_vs_moneyline(tmp_path):
    from datetime import datetime, timezone
    from nflpred import ml_v4
    r = ml_v4.load_rules()
    lo = {"consensus_home_margin": 3.0, "sharp_home_prob": 0.60, "totals": {"median_point": 44.5},
          "sharp_spreads": [[-3.0, -120, 100], [-3.0, -118, -102]],          # sharp spread says home > 3 pts
          "best_home_ml": {"price": -135, "book": "DraftKings"}, "best_away_ml": {"price": 120, "book": "FanDuel"}}
    g = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "KC", "away_team": "DEN",
         "context": {"live_odds": lo}}
    p_sp = ml_v4.spread_implied_home(lo, r)
    assert 0.6 < p_sp < 0.7
    tue = datetime(2026, 9, 29, 14, 15, tzinfo=timezone.utc)
    bet = ml_v4.evaluate(g, r, tue)
    assert bet and bet["team"] == "KC" and bet["edge"] >= 0.02
    wed = datetime(2026, 9, 30, 14, 15, tzinfo=timezone.utc)
    g2 = dict(g, context={"live_odds": lo})
    assert ml_v4.evaluate(g2, r, wed) is None and any("only bets" in x for x in g2["ml_v4_check"])


@pytest.mark.parametrize("live", [True, False])
def test_dashboard_promo_calculator_section(live):
    """The promo & boost calculator renders with full live odds (sharp fair prices, best moneylines,
    per-book spreads/totals) and without any live odds; the section, nav link and disclaimer are present."""
    import json
    import re
    bd = _load_build_dashboard()
    pred = _dash_predictions(False)
    if live:
        pred["live_odds_available"] = True
        pred["upcoming"][0]["context"]["live_odds"] = {
            "commence_time": "2026-10-04T17:00:00Z", "consensus_home_prob": 0.58, "sharp_home_prob": 0.585,
            "pinnacle_home_prob": 0.59, "pin_sharp_home_prob": 0.587, "consensus_home_margin": 2.5,
            "best_home_ml": {"price": -135, "book": "DraftKings"}, "best_away_ml": {"price": 125, "book": "FanDuel"},
            "spreads_by_book": [{"book": "FanDuel", "home_point": -2.5, "home_price": -110, "away_point": 2.5,
                                 "away_price": -110}],
            "totals": {"consensus_total": 45.0, "totals_by_book": [
                {"book": "FanDuel", "point": 44.5, "over_price": -110, "under_price": -110}]}}
    backtest = {"overall": {"margin": {"accuracy": 0.66, "n": 2000}, "vegas": {"accuracy": 0.67, "n": 2000}},
                "seasons": "2018-2025", "calibration": [], "by_season": []}
    body = bd.render(bd.build_payload(pred, backtest, None, []))
    assert '<section id="promos">' in body and 'href="#promos"' in body
    assert "Not financial advice. Promo terms vary by book and state; check the terms." in body
    for el in ("pc-game", "pc-odds", "pcout", "bb-t", "ns-t", "pb-t"):
        assert f'id="{el}"' in body
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', body, re.S).group(1))
    assert ("live_odds" in data["predictions"]["upcoming"][0]["context"]) == live


# ---------------------------------------------------------------- grade v2 (moneyline predicted CLV)
def _g2_event(fd_away=150):
    def bk(key, mh, ma, hp, hpr, apr):
        return {"key": key, "title": key.title(), "markets": [
            {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": mh}, {"name": "Denver Broncos", "price": ma}]},
            {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": hp, "price": hpr},
                                            {"name": "Denver Broncos", "point": -hp, "price": apr}]},
            {"key": "totals", "outcomes": [{"name": "Over", "point": 44.5, "price": -110},
                                           {"name": "Under", "point": 44.5, "price": -110}]}]}
    return [{"home_team": "Kansas City Chiefs", "away_team": "Denver Broncos", "commence_time": "2026-10-04T20:25:00Z",
             "bookmakers": [bk("lowvig", -150, 140, -3.0, -110, -110), bk("betonlineag", -152, 138, -3.0, -112, -108),
                            bk("draftkings", -160, 145, -3.0, -115, -105), bk("fanduel", -165, fd_away, -3.0, -110, -110),
                            bk("kalshi", -140, 160, -3.0, -110, -110)]}]


def _g2_game(lo, **kw):
    g = {"game_id": "g", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": "KC", "away_team": "DEN",
         "kickoff_utc": "2026-10-04T20:25:00+00:00", "home_win_prob": 0.62, "injury_report": True,
         "qb_status": {"home": {"play_prob": 1.0}, "away": {"play_prob": 1.0}}, "context": {"live_odds": lo}}
    g.update(kw)
    return g


def test_grade_v2_features_match_research_definitions():
    import math
    from datetime import datetime, timezone
    from nflpred import odds as O, grade_v2 as GV
    lo = next(iter(O.summarize(_g2_event(), {"draftkings", "fanduel"}).values()))
    assert lo["best_away_ml"] == {"price": 150, "book": "Fanduel", "key": "fanduel"}
    assert "kalshi" not in lo["by_book"]                                      # exchanges left out, as in research
    ref = GV.market_refs(lo)
    nv = lambda h, a: GV._imp(h) / (GV._imp(h) + GV._imp(a))  # noqa: E731
    assert abs(ref["p_sharp"] - (nv(-150, 140) + nv(-152, 138)) / 2) < 1e-9   # median of LowVig/BetOnline
    assert ref["pt_cons"] == -3.0 and 0.001 < ref["p_tie"] < 0.02 and ref["disp"] > 0
    assert 0.55 < ref["p_sp_sharp"] < 0.7
    now = datetime(2026, 10, 1, 20, 25, tzinfo=timezone.utc)                  # 72 h before kickoff
    o = GV.offer_features(_g2_game(lo), "away", 150, "fanduel", ref, now, first_p=0.62, first_m=2.0)
    F, pt = o["features"], ref["p_tie"]
    assert abs(F["ev_ml_sharp"] - ((1 - ref["p_sharp"]) * 2.5 - 1) * (1 - pt)) < 1e-9
    assert abs(F["ev_model"] - (0.38 * 2.5 - 1)) < 1e-9 and F["model_elig"] == 1.0   # no tie factor on ev_model
    assert F["p_imp"] == 0.4 and F["is_dog"] == 1 and F["longshot"] == 0 and F["best_gap"] == 0
    assert F["bk_fanduel"] == 1 and F["bk_betmgm"] == 0 and F["sharp_missing"] == 0
    assert abs(F["log_hours"] - math.log1p(72)) < 1e-9 and F["is_last"] == 0
    assert F["key_pos"] == 1 and F["on3"] == 1                               # DEN +3 = right side of 3
    assert abs(F["move_p"] - (-(lo["consensus_home_prob"] - 0.62))) < 1e-9   # + = toward DEN
    assert abs(F["move_pts"] - (-(lo["consensus_home_margin"] - 2.0))) < 1e-9
    assert o["candidate"]
    no_rep = GV.offer_features(_g2_game(lo, injury_report=False), "away", 150, "fanduel", ref, now)
    assert no_rep["features"]["model_elig"] == 0 and no_rep["features"]["ev_model"] == 0
    assert set(F) == set(json_load_frozen()["feature_order"])


def json_load_frozen():
    import json
    from nflpred import grade_v2 as GV
    return json.loads(GV.MODEL_PATH.read_text())


def test_grade_v2_letters_and_frozen_copy():
    import json
    from pathlib import Path
    from nflpred import grade_v2 as GV
    th = json_load_frozen()["thresholds_pred_clv"]
    assert th == {"A+": 0.025, "A": 0.015, "B": 0.005}
    assert [GV.letter(x, th) for x in (0.03, 0.025, 0.02, 0.015, 0.01, 0.005, 0.0, -0.05)] == \
        ["A+", "A+", "A", "A", "B", "B", "C", "C"]
    assert GV.letter(None, th) is None and GV.letter(float("nan"), th) is None
    research = Path(__file__).resolve().parents[1] / "output" / "research" / "grade_v2_frozen.json"
    if research.exists():   # production model is a verbatim copy of the frozen research model
        assert json.loads(research.read_text())["model"] == json_load_frozen()["model"]


def test_grade_v2_attach_and_bet_label(tmp_path):
    pytest.importorskip("lightgbm")
    from datetime import datetime, timezone
    from nflpred import odds as O, grade_v2 as GV, ml_v2
    lo = next(iter(O.summarize(_g2_event(fd_away=165), {"draftkings", "fanduel"}).values()))
    g = _g2_game(lo, home_win_prob=0.58)
    pred = {"upcoming": [g]}
    s = GV.attach(pred, tmp_path, datetime(2026, 10, 1, 20, 25, tzinfo=timezone.utc))
    assert s["available"] and s["graded_games"] == 1
    G = g["grade_v2"]
    assert G["grade"] in GV.ORDER and G["grade"] == GV.letter(G["predicted_clv"], s["thresholds"])
    assert G["side"] in ("home", "away") and G["price"] == lo[f"best_{G['side']}_ml"]["price"]
    assert all(x["predicted_clv"] <= G["predicted_clv"] for x in g["grade_v2_sides"].values())
    # recorded on a new v2 paper bet as a label; qualification unchanged
    before = ml_v2.evaluate(_g2_game(lo, home_win_prob=0.58), ml_v2.load_rules())
    bet = ml_v2.evaluate(g, ml_v2.load_rules())
    assert before and bet and {k: v for k, v in bet.items() if k != "placed_at"} == \
        {**{k: v for k, v in before.items() if k != "placed_at"}, "grade_v2": bet["grade_v2"],
         "predicted_clv": bet["predicted_clv"]}
    side = g["grade_v2_sides"][bet["side"]]
    assert bet["grade_v2"] == side["grade"] and bet["predicted_clv"] == side["predicted_clv"]
    assert before["grade_v2"] is None                                           # ungraded game: label null
    assert GV.bet_fields(g, "away", 999) == {"grade_v2": None, "predicted_clv": None}   # different price: no label
    rows = GV.by_grade([{"status": "graded", "grade_v2": "A+", "units": 1, "result": "win", "profit_units": 1.5,
                         "clv": 0.03, "predicted_clv": 0.028}])
    assert rows[0]["grade"] == "A+" and rows[0]["avg_clv"] == 0.03 and rows[0]["roi"] == 1.5


def test_grade_v2_fail_safe(tmp_path, monkeypatch):
    import sys
    from nflpred import odds as O, grade_v2 as GV
    lo = next(iter(O.summarize(_g2_event(), {"draftkings", "fanduel"}).values()))
    assert GV.load_model(tmp_path / "missing.json") is None                     # model file missing
    monkeypatch.setitem(sys.modules, "lightgbm", None)                          # lightgbm not installed
    (tmp_path / "m.json").write_text(GV.MODEL_PATH.read_text())
    assert GV.load_model(tmp_path / "m.json") is None
    g = _g2_game(lo)
    s = GV.attach({"upcoming": [g]}, tmp_path, model=False)
    assert s["available"] is False and g["grade_v2"] is None
    class Boom:
        def predict(self, *a, **k):
            raise RuntimeError("boom")
    bad = {"booster": Boom(), "feature_order": json_load_frozen()["feature_order"], "thresholds": {"A+": .025, "A": .015, "B": .005}}
    g2 = _g2_game(lo)
    s = GV.attach({"upcoming": [g2]}, tmp_path, model=bad)                      # any error -> null, no raise
    assert s["errors"] == 1 and g2["grade_v2"] is None
    from nflpred import ml_v2
    ml_v2.evaluate(g2, ml_v2.load_rules())                                       # tracks still run on ungraded games


def test_dashboard_renders_grade_v2(tmp_path):
    import json
    import re
    bd = _load_build_dashboard()
    pred = _dash_predictions(True)
    up = pred["upcoming"][0]
    up["ml_v2"].update(side="away")
    up["grade_v2"] = {"side": "away", "team": "AAA", "price": 150, "book": "FanDuel", "predicted_clv": 0.031,
                      "grade": "A+", "grading_version": 2}
    up["grade_v2_sides"] = {"away": up["grade_v2"]}
    up["moneyline"] = {"side": "away", "team": "AAA", "price": 150, "book": "FanDuel", "p_ours": 0.45, "p_market": 0.41,
                       "p_needed": 0.4, "edge": 0.12, "grade": "B", "why": [], "verdict": "lean", "reasons": []}
    pred["grade_v2"] = {"version": 2, "available": True, "note": "Grade = predicted closing-line value; only A+ has shown an edge (2023-25: +2.3% CLV)",
                        "record": {"all": [{"grade": "A+", "bets": 2, "wins": 1, "losses": 1, "profit_units": 0.5,
                                            "roi": 0.25, "avg_clv": 0.02, "avg_predicted_clv": 0.03}], "by_track": []}}
    pred["ml_v2_bets"] = {"mode": "shadow", "record": {"graded": 0, "min_bets": 150, "wins": 0, "losses": 0,
                          "profit_units": 0.0, "roi": 0.0, "avg_clv": 0.0, "clv_p_value": 1.0,
                          "checks": {"enough_bets": False, "roi_positive": False, "clv_positive_and_significant": False}},
                          "open": [{"game_id": up["game_id"], "gameday": "2026-10-04", "team": "AAA", "opponent": "BBB",
                                    "side": "away", "price": 150, "book": "FanDuel", "edge": 0.03, "units": 1,
                                    "placed_at": "2026-10-01T12:00+00:00", "grade_v2": "A+", "predicted_clv": 0.031}],
                          "recent_graded": [], "by_grade": [], "by_grade_v2": []}
    backtest = {"overall": {"margin": {"accuracy": 0.66, "n": 2000}, "vegas": {"accuracy": 0.67, "n": 2000}},
                "seasons": "2018-2025", "calibration": [], "by_season": []}
    body = bd.render(bd.build_payload(pred, backtest, None, []))
    assert 'id="g2rec"' in body and 'id="g2games"' in body and "Grade v2 record" in body
    assert "only A+ has shown an edge (2023-25: +2.3% CLV)" in body
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', body, re.S).group(1))
    assert data["predictions"]["upcoming"][0]["grade_v2"]["grade"] == "A+"
