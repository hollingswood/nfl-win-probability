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
