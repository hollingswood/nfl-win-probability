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
    return g, p


@pytest.fixture(scope="session")
def feats(raw):
    return F.build_features(*raw)


# ---------------------------------------------------------------- leakage
@pytest.mark.parametrize("cutoff", ["2022-10-09", "2023-12-24", "2024-09-08"])
def test_no_future_leakage(raw, feats, cutoff):
    """Features for games on `cutoff` must be identical if every result from `cutoff` onward
    is erased. If a feature changes, it was using information from the game itself or later."""
    g, p = raw
    cut = pd.Timestamp(cutoff)
    g2 = g.copy()
    future = pd.to_datetime(g2["gameday"]) >= cut
    g2.loc[future, ["home_score", "away_score", "result", "total"]] = np.nan
    future_ids = set(g2.loc[future, "game_id"])
    p2 = p[~p["game_id"].isin(future_ids)]
    masked = F.build_features(g2, p2).set_index("game_id")
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
    for f in ["elo_diff", "qb_diff", "off_epa_diff", "def_epa_diff", "pt_diff_diff"]:
        assert d[f].corr(d["home_win"]) > 0.05, f


def test_vegas_prob_is_vig_free():
    h = pd.Series([-150.0, 120.0, -110.0])
    a = pd.Series([130.0, -140.0, -110.0])
    p = F.vegas_prob(h, a)
    assert np.allclose(p + F.vegas_prob(a, h), 1)
    assert p.iloc[2] == pytest.approx(0.5)


def test_explanations_sum_to_model_logit(feats):
    train = feats[feats["season"] < 2024]
    m = M.fit(train, "logistic")
    X = feats[feats["season"] == 2024].head(20)
    p = m.predict_proba(X[F.FEATURES])[:, 1]
    scaler, lr = m[0], m[-1]
    for i, ex in enumerate(M.explain_logistic(m, X, top=len(M.FACTOR_GROUPS))):
        assert sum(e.logodds for e in ex) == pytest.approx(np.log(p[i] / (1 - p[i])), abs=1e-6)


def test_model_beats_coin_flip(feats):
    _, summ = M.backtest(feats, [2023, 2024], kinds=("logistic",))
    ll = summ.query("season == 'ALL' and model == 'logistic'")["log_loss"].item()
    assert ll < 0.67
