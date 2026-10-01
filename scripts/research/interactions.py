"""Do COMBINATIONS of situational / game factors predict where NFL games finish relative to the closing
market (spread, total), even though the individual factors don't?

Targets (per game, home perspective):  y_side = home margin - spread_line,  y_tot = total points - total_line.

Part A  "all combinations at once": shallow gradient boosting (HistGradientBoosting, depth 2-3) and a ridge
        on ALL pairwise interactions of ~67 pre-game features (Walters factor components, rest, travel,
        QB/coach changes, weather, team style, standings, the closing lines). Tuned by season-grouped
        expanding-window CV on 2003-2019, final fit on 2003-2019, evaluated on 2020-2022 (dev); model,
        bet thresholds and a hash of its 2020-22 predictions are then FROZEN and 2023-2025 is scored once.
        Null benchmark: the same pipeline refit on targets shuffled within season.
Part B  ten hypothesis-driven combinations, written down (HYPOTHESES below) BEFORE any result was computed.
        Dev = 2003-2019; survivors of a Bonferroni bar are frozen and tested ONCE on 2020-2025.
Part C  false-discovery picture: the B scorecard plus an exhaustive sweep of all pairwise AND-combinations
        of ~30 binary flags (dev 2003-19 -> replication 2020-25), to show what 'winning combos' look like
        when nothing is there.

Stages (each writes output/research/interactions.json):
    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/interactions.py dev
    PYTHONPATH=src:scripts python scripts/research/interactions.py freeze      # writes interactions_frozen.json
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/interactions.py holdout
    PYTHONPATH=src:scripts python scripts/research/interactions.py posthoc     # labelled diagnostics after holdout
    PYTHONPATH=src:scripts python scripts/research/interactions.py report      # interactions.md from the json

Leakage notes: every team-level rolling stat uses strictly earlier games (shift(1)); standings use results
with an earlier gameday only. Two inputs are NOT strictly ex-ante: nflverse temp/wind/rain are RECORDED game
conditions (a near-perfect forecast at the close, optimistic for an early-week bet), and the starting QB is
the actual starter (known by the close in nearly all games, not always on Tuesday). For early-line bets the
closing spread/total features are replaced by the early-week consensus line.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]

import walters_factors as W                     # noqa: E402  (reused, not modified)
from nflpred.travel import travel_features      # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "interactions.json"
MD = OUT / "interactions.md"
FROZEN = OUT / "interactions_frozen.json"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/interactions")
TRAIN, DEVA, HOLDA = (2003, 2019), (2020, 2022), (2023, 2025)     # Part A
DEVB, HOLDB = (2003, 2019), (2020, 2025)                          # Part B / C
CV_VAL = list(range(2009, 2020))                                  # expanding-window validation seasons
CLIP = {"y_side": 21.0, "y_tot": 24.0}                            # training-target winsorisation (pre-set)
N_PERM = 60
BE = 0.5238                                                       # break-even at -110

# =====================================================================================================
# PART B PRE-REGISTRATION. Written 2026-10-01 ~01:20 UTC, before any result in this file was computed.
# Each hypothesis: one target, one pre-stated direction, one population. Bar on dev (2003-2019):
# n >= 40 active games, one-sided t of the mean residual in the stated direction >= 2.576
# (p < 0.05/10, Bonferroni over the 10 hypotheses) AND dev cover rate > 52.38%.
# Survivors are frozen and tested ONCE on 2020-2025 (one-sided p < 0.05 / #survivors).
# 'sign' convention: +1 = bet HOME (side) or OVER (total).
# =====================================================================================================
HYPOTHESES = [
    {"id": "H1_wind_passheavy_under", "target": "y_tot",
     "definition": "outdoor/open roof, wind >= 15 mph, and the two teams' mean pre-game neutral-situation pass "
                   "rate >= its 2012-19 median (pbp seasons only) -> UNDER",
     "rationale": "the market lowers totals for wind on average, but wind should hurt pass-dependent offenses more "
                  "than a generic adjustment allows"},
    {"id": "H2_tnf_long_travel_home", "target": "y_side",
     "definition": "Thursday game with both teams on <= 5 days rest and the visitor travelling >= 1600 km "
                   "(~1000 mi) base-to-venue -> HOME",
     "rationale": "short preparation plus a long trip compounds fatigue/logistics for the road team"},
    {"id": "H3_backup_qb_road_weather_home", "target": "y_side",
     "definition": "road team's starter started < 50% of its previous 8 games (>= 4 prior games), outdoor game "
                   "with temp <= 40F or wind >= 15 mph -> HOME",
     "rationale": "the market prices the backup, but an inexperienced QB on the road in bad weather is worse "
                  "than either adjustment alone"},
    {"id": "H4_bye_vs_off_mnf_rested", "target": "y_side",
     "definition": "regular season, one team has >= 13 days rest (off bye) and the other <= 6 days (off MNF) "
                   "-> the RESTED team",
     "rationale": "largest regular-season rest gap; bye and MNF are each priced individually"},
    {"id": "H5_tnf_night_west", "target": "y_side",
     "definition": "Thursday game, kickoff >= 7pm ET, both teams on <= 5 days rest, team base zones differ by "
                   ">= 2 (E vs M/P, or C vs P) -> the more WESTERN team",
     "rationale": "night-game body-clock edge of the western team should be larger with no time to adjust"},
    {"id": "H6_div_big_road_fav_dog", "target": "y_side",
     "definition": "divisional game with the road team favoured by >= 7 at the close -> HOME (the dog)",
     "rationale": "division familiarity compresses margins; big road favourites are popular with the public"},
    {"id": "H7_warm_team_cold_late", "target": "y_side",
     "definition": "road warm-weather team (Walters' definition), outdoor game, temp <= 35F, week >= 13 or "
                   "postseason -> HOME",
     "rationale": "cold-weather effect on warm-climate teams should be largest late in the season"},
    {"id": "H8_low_stakes_late_over", "target": "y_tot",
     "definition": "regular season week >= 15 and BOTH teams eliminated or clinched (conservative mathematical "
                   "rules from prior results only) -> OVER",
     "rationale": "reduced defensive intensity / backups in games with nothing at stake"},
    {"id": "H9_dome_team_outdoor_wind_under", "target": "y_tot",
     "definition": "at least one team with a dome/retractable home plays at an outdoor/open venue with wind "
                   ">= 15 mph -> UNDER",
     "rationale": "dome offenses (built for and used to calm conditions) suffer more in wind"},
    {"id": "H10_early_new_qb_or_coach_fade", "target": "y_side",
     "definition": "weeks 1-3 of the regular season, exactly one team has a new starting QB (not its most frequent "
                   "starter of the previous season) or a new head coach -> FADE that team",
     "rationale": "the market may overrate new-QB/new-coach teams before evidence accumulates"},
]
B_T_BAR = 2.576          # one-sided p < 0.005
B_MIN_N = 40


# ===================================================================================================== utils
def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def rnd(x, k=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


def pval_one(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def pval_two(t):
    return math.erfc(abs(t) / math.sqrt(2))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def per(d, lo_hi):
    return d[d.season.between(*lo_hi)]


# ===================================================================================================== data
def ewm_prev(lg: pd.DataFrame, col: str, hl=8) -> pd.Series:
    """EWMA over the team's strictly-previous games (crosses seasons), NaN-aware."""
    return lg.groupby("team")[col].transform(lambda x: x.shift(1).ewm(halflife=hl, ignore_na=True).mean())


def team_long(s: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "season", "week", "game_type", "gameday"]
    h = s[cols + ["home_team", "away_team", "home_score", "away_score", "home_qb_id", "home_coach"]].set_axis(
        cols + ["team", "opp", "pf", "pa", "qb", "coach"], axis=1).assign(side="home")
    a = s[cols + ["away_team", "home_team", "away_score", "home_score", "away_qb_id", "away_coach"]].set_axis(
        cols + ["team", "opp", "pf", "pa", "qb", "coach"], axis=1).assign(side="away")
    return pd.concat([h, a]).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def qb_coach_flags(L: pd.DataFrame) -> pd.DataFrame:
    L = L.copy()
    L["pf_ewm"] = ewm_prev(L, "pf")
    L["pa_ewm"] = ewm_prev(L, "pa")
    g = L.groupby("team")
    L["prev_qb"] = g.qb.shift(1)
    L["qb_change"] = (L.prev_qb.notna() & (L.qb != L.prev_qb)).astype(float)
    irr = np.zeros(len(L))
    for _, idx in g.groups.items():
        idx = np.asarray(sorted(idx))
        q = L.loc[idx, "qb"].values
        for j in range(len(idx)):
            prev = q[max(0, j - 8):j]
            if len(prev) >= 4:
                irr[idx[j]] = float(np.mean(prev == q[j]) < 0.5)
    L["qb_irreg"] = irr
    # previous season's most frequent starter and last coach
    ts = L.groupby(["team", "season"]).agg(top_qb=("qb", lambda x: x.value_counts().index[0]),
                                           last_coach=("coach", "last")).reset_index()
    ts["season"] = ts.season + 1
    L = L.merge(ts, on=["team", "season"], how="left")
    L["new_qb"] = (L.top_qb.notna() & (L.qb != L.top_qb)).astype(float)
    L["new_coach"] = (L.last_coach.notna() & (L.coach != L.last_coach)).astype(float)
    return L


def standings_flags(s: pd.DataFrame) -> pd.DataFrame:
    """Per (game, team): pre-game win pct, conservative 'eliminated' / 'clinched' from prior results only.
    eliminated: >= k conference rivals already have more wins than my max possible AND a division rival does too.
    clinched:   no division rival can reach my current wins, OR fewer than (k-4) conference rivals can."""
    reg = s[s.game_type == "REG"].copy()
    rows = []
    for season, d in reg.groupby("season"):
        k = 7 if season >= 2020 else 6
        teams = sorted(set(d.home_team) | set(d.away_team))
        afc = {t: t in W.AFC for t in teams}
        parent = {t: t for t in teams}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for r in d[d.div_game == 1].itertuples():
            parent[find(r.home_team)] = find(r.away_team)
        div = {t: find(t) for t in teams}
        ngames = pd.concat([d.home_team, d.away_team]).value_counts()
        days = sorted(d.gameday.unique())
        for day in days:
            pr = d[d.gameday < day]
            w = pd.Series(0.0, index=teams)
            gp = pd.Series(0.0, index=teams)
            if len(pr):
                hw = np.where(pr.result > 0, 1.0, np.where(pr.result == 0, 0.5, 0.0))
                w = w.add(pd.Series(hw, index=pr.home_team.values).groupby(level=0).sum(), fill_value=0)
                w = w.add(pd.Series(1 - hw, index=pr.away_team.values).groupby(level=0).sum(), fill_value=0)
                gp = gp.add(pr.home_team.value_counts(), fill_value=0).add(pr.away_team.value_counts(), fill_value=0)
            mx = w + (ngames.reindex(teams).fillna(0) - gp)
            today = d[d.gameday == day]
            for r in today.itertuples():
                for t in (r.home_team, r.away_team):
                    conf = [o for o in teams if afc[o] == afc[t] and o != t]
                    divr = [o for o in conf if div[o] == div[t]]
                    above = sum(w[o] > mx[t] for o in conf)
                    div_above = any(w[o] > mx[t] for o in divr)
                    elim = (above >= k) and div_above
                    can_reach = sum(mx[o] >= w[t] for o in conf)
                    cl_div = len(divr) > 0 and not any(mx[o] >= w[t] for o in divr)
                    clinch = cl_div or (can_reach < k - 4)
                    rows.append((r.game_id, t, w[t] / gp[t] if gp[t] > 0 else np.nan, float(elim), float(clinch)))
    return pd.DataFrame(rows, columns=["game_id", "team", "wpct", "elim", "clinch"])


def pbp_style() -> pd.DataFrame:
    import totals as T          # scripts/research/totals.py: pace_stats (seconds per snap, neutral situations)
    cache = SCR / "pbp_style.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["game_id", "play_id", "posteam", "defteam", "play_type", "down", "qtr", "score_differential", "epa",
            "game_seconds_remaining", "fixed_drive", "half_seconds_remaining"]
    out = []
    for yr in range(2012, 2026):
        p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{yr}.parquet", columns=cols)
        p = p[p.play_type.isin(["pass", "run"])].copy()
        for c in ("posteam", "defteam"):
            p[c] = p[c].replace(W.F.TEAM_MAP if hasattr(W.F, "TEAM_MAP") else {})
        neu = p[p.down.between(1, 3) & (p.qtr <= 3) & (p.score_differential.abs() <= 10)]
        pr = neu.groupby(["game_id", "posteam"]).play_type.agg(lambda x: (x == "pass").mean()).rename("pass_rate")
        oe = p.groupby(["game_id", "posteam"]).epa.mean().rename("off_epa")
        de = p.groupby(["game_id", "defteam"]).epa.mean().rename("def_epa").rename_axis(["game_id", "posteam"])
        sp = T.pace_stats(p).set_index(["game_id", "team"]).spp.rename_axis(["game_id", "posteam"])
        out.append(pd.concat([pr, oe, de, sp], axis=1).reset_index().rename(columns={"posteam": "team"}))
    st = pd.concat(out, ignore_index=True)
    st.to_parquet(cache)
    return st


def build_table() -> pd.DataFrame:
    cache = SCR / "table.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    SCR.mkdir(parents=True, exist_ok=True)
    fac = W.build_factors()
    s = W.load_schedule()
    tr = travel_features(s)
    ctx = W.team_season_context(s)
    L = qb_coach_flags(team_long(s))
    st = pbp_style()
    L = L.merge(st, on=["game_id", "team"], how="left")
    for c in ("pass_rate", "off_epa", "def_epa", "spp"):
        L[f"{c}_ewm"] = ewm_prev(L, c)
        L.loc[L.season < 2012, f"{c}_ewm"] = np.nan
    L = L.merge(standings_flags(s), on=["game_id", "team"], how="left")
    keep = ["pf_ewm", "pa_ewm", "qb_change", "qb_irreg", "new_qb", "new_coach", "pass_rate_ewm", "off_epa_ewm",
            "def_epa_ewm", "spp_ewm", "wpct", "elim", "clinch"]
    d = s[["game_id", "season", "week", "game_type", "gameday", "weekday", "hour", "home_team", "away_team",
           "home_score", "away_score", "result", "total", "spread_line", "total_line", "location", "div_game",
           "roof", "surface", "temp", "wind", "home_rest", "away_rest"]].copy()
    for side in ("home", "away"):
        m = L[L.side == side][["game_id"] + keep].rename(columns={c: f"{side}_{c}" for c in keep})
        d = d.merge(m, on="game_id", how="left")
        c = ctx.rename(columns={"team": f"{side}_team"})[["season", f"{side}_team", "dome", "warm", "zone"]] \
            .rename(columns={"dome": f"{side}_dome_team", "warm": f"{side}_warm", "zone": f"{side}_zone_b"})
        d = d.merge(c, on=["season", f"{side}_team"], how="left")
    d = d.merge(tr, on="game_id", how="left")
    comp = fac[["game_id"] + W.COMPONENTS]
    d = d.merge(comp, on="game_id", how="left")
    d["rain"] = d.game_id.map(W.rain_levels()).fillna(0.0)
    d = d[d.season.between(2003, 2025) & d.spread_line.notna() & d.total_line.notna() & d.result.notna()].copy()
    d["y_side"] = d.result - d.spread_line
    d["y_tot"] = d.total - d.total_line
    d["outdoor"] = d.roof.isin(["outdoors", "open"]).astype(float)
    d["temp_o"] = np.where(d.outdoor == 1, d.temp, 70.0)          # indoor = 70F, missing outdoor = NaN
    d["wind_o"] = np.where(d.outdoor == 1, d.wind, 0.0)
    d["art_turf"] = (~d.surface.astype(str).str.strip().isin(["grass", "dessograss", ""])).astype(float)
    d["playoff"] = (d.game_type != "REG").astype(float)
    d["neutral"] = (d.location == "Neutral").astype(float)
    d["rest_diff"] = (d.home_rest - d.away_rest).clip(-10, 10)
    d["div_game"] = d.div_game.fillna(0).astype(float)
    d["late"] = ((d.game_type == "REG") & (d.week >= 15)).astype(float)
    for side in ("home", "away"):
        d[f"{side}_low_stakes"] = ((d[f"{side}_elim"] == 1) | (d[f"{side}_clinch"] == 1)).astype(float) * d.late
    d = d.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    d.to_parquet(cache)
    return d


FEATS = (W.COMPONENTS +
         ["home_rest", "away_rest", "rest_diff", "home_km", "away_km", "home_tz_shift", "away_tz_shift",
          "home_body_hour", "away_body_hour", "week", "playoff", "neutral", "div_game",
          "home_qb_change", "away_qb_change", "home_qb_irreg", "away_qb_irreg", "home_new_qb", "away_new_qb",
          "home_new_coach", "away_new_coach", "outdoor", "temp_o", "wind_o", "rain", "art_turf",
          "home_dome_team", "away_dome_team", "home_warm", "away_warm",
          "home_pf_ewm", "home_pa_ewm", "away_pf_ewm", "away_pa_ewm",
          "home_pass_rate_ewm", "away_pass_rate_ewm", "home_spp_ewm", "away_spp_ewm",
          "home_off_epa_ewm", "home_def_epa_ewm", "away_off_epa_ewm", "away_def_epa_ewm",
          "home_wpct", "away_wpct", "home_low_stakes", "away_low_stakes", "home_elim", "away_elim",
          "spread_line", "total_line"])


# ===================================================================================================== Part A
class PairRidge:
    """Standardise -> impute 0 -> all pairwise products (+ main effects + has_pbp) -> standardise -> Ridge."""

    def __init__(self, alpha):
        self.alpha = alpha

    def _base(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.nan_to_num(Z, nan=0.0)
        return Z

    def _expand(self, Z):
        i, j = np.triu_indices(Z.shape[1], k=1)
        return np.hstack([Z, Z[:, i] * Z[:, j]])

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        X = np.asarray(X, float)
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd[~np.isfinite(self.sd) | (self.sd == 0)] = 1.0
        self.mu = np.nan_to_num(self.mu)
        P = self._expand(self._base(X))
        self.pm, self.ps = P.mean(0), P.std(0)
        self.ps[self.ps == 0] = 1.0
        self.m = Ridge(alpha=self.alpha).fit((P - self.pm) / self.ps, y)
        return self

    def predict(self, X):
        P = self._expand(self._base(np.asarray(X, float)))
        return self.m.predict((P - self.pm) / self.ps)


def make_model(spec):
    if spec["kind"] == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        p = spec["params"]
        return HistGradientBoostingRegressor(max_depth=p["max_depth"], learning_rate=p["lr"], max_iter=p["n"],
                                             min_samples_leaf=p["leaf"], l2_regularization=p["l2"],
                                             max_leaf_nodes=2 ** p["max_depth"], early_stopping=False,
                                             random_state=0)
    return PairRidge(spec["params"]["alpha"])


GRID = ([{"kind": "gbm", "params": {"max_depth": d, "lr": 0.02, "n": n, "leaf": lf, "l2": 10.0}}
         for d in (2, 3) for n in (100, 300) for lf in (100, 300)] +
        [{"kind": "ridge2", "params": {"alpha": a}} for a in (3e3, 1e4, 3e4, 1e5, 3e5)])


def fit_spec(spec, tr, target):
    y = tr[target].clip(-CLIP[target], CLIP[target]).values
    return make_model(spec).fit(tr[FEATS].values, y)


def r2_vs0(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum(y ** 2)


def mse_delta(y, p):
    """MSE(model) - MSE(market: residual 0), with SE (paired)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    dd = (y - p) ** 2 - y ** 2
    return {"n": len(y), "mse_market": rnd(np.mean(y ** 2), 2), "mse_model": rnd(np.mean((y - p) ** 2), 2),
            "delta": rnd(dd.mean(), 3), "delta_se": rnd(dd.std(ddof=1) / math.sqrt(len(y)), 3),
            "r2_vs_market": rnd(r2_vs0(y, p), 5), "corr": rnd(np.corrcoef(y, p)[0, 1], 4),
            "pred_sd": rnd(np.std(p), 3)}


def cv(d, spec, target):
    ys, ps = [], []
    for v in CV_VAL:
        tr = d[d.season.between(TRAIN[0], v - 1)]
        va = d[d.season == v]
        m = fit_spec(spec, tr, target)
        ys.append(va[target].values)
        ps.append(m.predict(va[FEATS].values))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {**mse_delta(y, p), "spec": spec}


def ats_sel(y, p, thr, close_at=0):
    """Bet sign(p) where |p| >= thr at -110 against the closing number (y = residual vs that number)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.abs(p) >= thr
    adj = np.sign(p[m]) * y[m]
    n, w, l = int(m.sum()), int((adj > 0).sum()), int((adj < 0).sum())
    if w + l == 0:
        return {"bets": n}
    c = w / (w + l)
    pnl = np.where(adj > 0, 100 / 110, np.where(adj < 0, -1.0, 0.0))
    return {"bets": n, "w": w, "l": l, "push": n - w - l, "cover": rnd(c, 4),
            "cover_se": rnd(math.sqrt(c * (1 - c) / (w + l)), 4),
            "z_vs_524": rnd((c - BE) / math.sqrt(BE * (1 - BE) / (w + l)), 2),
            "roi": rnd(pnl.mean(), 4), "roi_se": rnd(pnl.std(ddof=1) / math.sqrt(n), 4) if n > 1 else None,
            "mean_resid_toward": rnd(adj.mean(), 3)}


def a_eval_period(y, p, thr):
    return {"fit": mse_delta(y, p), "top10": ats_sel(y, p, thr["top10"]), "top5": ats_sel(y, p, thr["top5"]),
            "all": ats_sel(y, p, 0.0)}


def pred_hash(p):
    return hashlib.sha256(np.round(np.asarray(p, float), 6).tobytes()).hexdigest()[:16]


def permuted(d, seed, target):
    rng = np.random.default_rng(seed)
    tr = per(d, TRAIN).copy()
    tr[target] = tr.groupby("season")[target].transform(lambda x: rng.permutation(x.values))
    return tr


# -------------------------------------------------- early lines (2020-25), price-based CLV
def early_spread(hold: bool):
    """(first snapshot per game with early consensus point, best allowed-book offer per side there)."""
    import edge_lab as E
    import line_move_model as LM
    seasons = E.HOLD if hold else E.DEV
    t = E.load(hold)
    s, off = LM.build(t, seasons)
    first = s.sort_values("requested_ts").groupby("game_id").head(1)
    o = off[off.sp_ok].merge(first[["game_id", "requested_ts"]], on=["game_id", "requested_ts"])
    o = o[o.sp_clv.notna()]
    best = o.sort_values("sp_ev_now", ascending=False).groupby(["game_id", "side"]).head(1)
    return first[["game_id", "season", "m_cons", "mu_px_all", "hours_before"]], best


def early_totals(hold: bool):
    """Tuesday snapshot: (game -> early consensus point), best allowed-book offer per side (price-based CLV)."""
    import totals as T
    name = "holdout" if hold else "dev"
    p = Path(f"/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/totals/bets_{name}.parquet")
    mine = SCR / f"tot_bets_{name}.parquet"
    if mine.exists():
        t = pd.read_parquet(mine)
    elif p.exists():
        t = pd.read_parquet(p)
    else:
        g = T.load_games()
        t, _, _ = T.build(T.HOLD if hold else T.DEV, T.TotalDist(g), g)
        t.to_parquet(mine)
    x = t[(t.snap == "tue") & t.tot_clv.notna()]
    best = x.sort_values("ev_ref_now", ascending=False).groupby(["game_id", "side"]).head(1)
    first = x.groupby("game_id").agg(pt_early=("pt_all", "first"), season=("season", "first")).reset_index()
    return first, best


def early_bets(best, sig_by_game: pd.Series, thr: float, market: str) -> dict:
    """Bet the side the (home/over-signed) signal favours where |signal| >= thr, best allowed price at the early
    snapshot. excess = that side's CLV minus the mean of both sides' best-price CLV in the same game."""
    clv_c, pnl_c, pos = ("sp_clv", "sp_pnl", "home") if market == "sp" else ("tot_clv", "tot_pnl", "over")
    b = best.copy()
    b["sig"] = b.game_id.map(sig_by_game)
    b["pair"] = b.groupby("game_id")[clv_c].transform("mean")
    b["n_sides"] = b.groupby("game_id")[clv_c].transform("size")
    pick = (b.sig.abs() >= thr) & (np.where(b.side == pos, 1, -1) * np.sign(b.sig) > 0) & (b.n_sides == 2)
    b = b[pick]
    if len(b) < 3:
        return {"bets": int(len(b))}
    clv, pnl, exc = b[clv_c].values, b[pnl_c].values, (b[clv_c] - b.pair).values
    tt = lambda v: rnd(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))), 2)
    return {"bets": int(len(b)), "clv": rnd(clv.mean()), "clv_t": tt(clv), "excess_clv": rnd(exc.mean()),
            "excess_t": tt(exc), "beat_close": rnd((clv > 0).mean(), 3), "roi": rnd(pnl.mean()),
            "roi_se": rnd(pnl.std(ddof=1) / math.sqrt(len(pnl)))}


def early_baseline(best, market):
    clv_c, pnl_c = ("sp_clv", "sp_pnl") if market == "sp" else ("tot_clv", "tot_pnl")
    return {"bets": int(len(best)), "clv": rnd(best[clv_c].mean()), "roi": rnd(best[pnl_c].mean())}


def early_preds(model, d_games, first, market):
    """Model predictions with the closing line feature(s) replaced by the early consensus."""
    x = d_games.merge(first, on=["game_id", "season"], how="inner").copy()
    if market == "sp":
        x["spread_line"] = x.m_cons
    else:
        x["total_line"] = x.pt_early
    return pd.Series(model.predict(x[FEATS].values), index=x.game_id.values), len(x)


# ===================================================================================================== Part B
def pass_rate_median(d_full: pd.DataFrame) -> float:
    """H1 threshold: median of the two teams' mean pre-game pass rate over 2012-19 games (features only)."""
    x = d_full[d_full.season.between(2012, 2019)]
    return float(((x.home_pass_rate_ewm + x.away_pass_rate_ewm) / 2).median())


def hyp_signals(d: pd.DataFrame, med: float, spread_col="spread_line") -> dict[str, pd.Series]:
    """id -> home/over-signed signal (+1 bet home/over, -1 away/under, 0 inactive), indexed like d."""
    out = {}
    outdoor = d.outdoor == 1
    w15 = outdoor & (d.wind_o >= 15)
    comb = (d.home_pass_rate_ewm + d.away_pass_rate_ewm) / 2
    out["H1_wind_passheavy_under"] = -(w15 & (comb >= med)).astype(float)
    thu = (d.weekday == "Thursday") & (d.home_rest <= 5) & (d.away_rest <= 5)
    out["H2_tnf_long_travel_home"] = (thu & (d.away_km >= 1600)).astype(float)
    bad = outdoor & ((d.temp_o <= 40) | (d.wind_o >= 15))
    out["H3_backup_qb_road_weather_home"] = ((d.away_qb_irreg == 1) & bad).astype(float)
    reg = d.game_type == "REG"
    hb, ab = d.home_rest >= 13, d.away_rest >= 13
    hs, as_ = d.home_rest <= 6, d.away_rest <= 6
    out["H4_bye_vs_off_mnf_rested"] = (reg & hb & as_).astype(float) - (reg & ab & hs).astype(float)
    zn = {"E": 0, "C": 1, "M": 2, "P": 3}
    hz, az = d.home_zone_b.map(zn), d.away_zone_b.map(zn)
    night_thu = thu & (d.hour >= 19)
    out["H5_tnf_night_west"] = np.where(night_thu & ((hz - az).abs() >= 2), np.sign(hz - az), 0.0)
    out["H6_div_big_road_fav_dog"] = ((d.div_game == 1) & (d[spread_col] <= -7)).astype(float)
    late = ((d.game_type == "REG") & (d.week >= 13)) | (d.game_type != "REG")
    out["H7_warm_team_cold_late"] = ((d.away_warm == 1) & outdoor & (d.temp_o <= 35) & late).astype(float)
    out["H8_low_stakes_late_over"] = ((d.home_low_stakes == 1) & (d.away_low_stakes == 1)).astype(float)
    out["H9_dome_team_outdoor_wind_under"] = -(w15 & ((d.home_dome_team == 1) | (d.away_dome_team == 1))).astype(float)
    early = (d.game_type == "REG") & (d.week <= 3)
    hn = ((d.home_new_qb == 1) | (d.home_new_coach == 1)).astype(float)
    an = ((d.away_new_qb == 1) | (d.away_new_coach == 1)).astype(float)
    out["H10_early_new_qb_or_coach_fade"] = np.where(early, an - hn, 0.0)   # fade new => back the other side
    return {k: pd.Series(np.asarray(v, float), index=d.index) for k, v in out.items()}


def cell(y, sig) -> dict:
    m = sig != 0
    v = (np.sign(sig[m]) * y[m]).values
    n = len(v)
    if n < 5:
        return {"n": n}
    se = v.std(ddof=1) / math.sqrt(n)
    w, l = int((v > 0).sum()), int((v < 0).sum())
    c = w / max(w + l, 1)
    pnl = np.where(v > 0, 100 / 110, np.where(v < 0, -1.0, 0.0))
    t = v.mean() / se if se > 0 else 0.0
    return {"n": n, "per_season": None, "resid_toward": rnd(v.mean(), 3), "se": rnd(se, 3), "t": rnd(t, 2),
            "p_one_sided": rnd(pval_one(t), 5), "w": w, "l": l, "push": n - w - l, "cover": rnd(c, 4),
            "cover_se": rnd(math.sqrt(c * (1 - c) / max(w + l, 1)), 4),
            "cover_z_vs_524": rnd((c - BE) / math.sqrt(BE * (1 - BE) / max(w + l, 1)), 2),
            "roi": rnd(pnl.mean(), 4), "roi_se": rnd(pnl.std(ddof=1) / math.sqrt(n), 4)}


# ===================================================================================================== Part C
def sweep_flags(d: pd.DataFrame) -> dict[str, pd.Series]:
    o = d.outdoor == 1
    f = {
        "div": d.div_game == 1, "nonconf": d.conference != 0, "tnf": d.home_tnf != 0, "snf": d.home_snf != 0,
        "mnf": d.home_mnf != 0, "home_off_mnf": d.home_off_mnf != 0, "away_off_mnf": d.away_off_mnf != 0,
        "away_3rd_road_in_4": d.three_away != 0, "home_off_bye": (d.home_rest >= 13) & (d.game_type == "REG"),
        "away_off_bye": (d.away_rest >= 13) & (d.game_type == "REG"), "travel_2000mi": d.travel_2000 != 0,
        "short_trip": d.short_trip != 0, "tz_10am": d.tz_10am != 0, "night_tz": d.night_tz != 0,
        "away_2nd_tz_trip": d.two_tz != 0, "home_bounce": d.bounce_back > 0, "away_bounce": d.bounce_back < 0,
        "wind15": o & (d.wind_o >= 15), "cold35": o & (d.temp_o <= 35), "rain": d.rain > 0,
        "home_qb_change": d.home_qb_change == 1, "away_qb_change": d.away_qb_change == 1,
        "home_qb_irreg": d.home_qb_irreg == 1, "away_qb_irreg": d.away_qb_irreg == 1,
        "home_fav7": d.spread_line >= 7, "away_fav": d.spread_line < 0, "late_wk13": (d.week >= 13) & (d.game_type == "REG"),
        "early_wk3": (d.week <= 3) & (d.game_type == "REG"), "playoff": d.game_type != "REG",
        "home_new_coach": d.home_new_coach == 1, "away_new_coach": d.away_new_coach == 1,
        "art_turf": d.art_turf == 1, "high_total48": d.total_line >= 48, "low_total40": d.total_line <= 40,
    }
    return {k: v.fillna(False).astype(bool) for k, v in f.items()}


def tstat(v):
    v = np.asarray(v, float)
    n = len(v)
    if n < 3 or v.std(ddof=1) == 0:
        return 0.0, n
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(n))), n


def sweep(d: pd.DataFrame, min_n=50) -> pd.DataFrame:
    f = sweep_flags(d)
    rows = []
    names = sorted(f)
    for a, b in itertools.combinations(names, 2):
        m = f[a] & f[b]
        for tgt in ("y_side", "y_tot"):
            y = d.loc[m, tgt]
            t, n = tstat(y)
            # pure interaction: coefficient of a*b in y ~ a + b + a*b
            X = np.column_stack([f[a], f[b], m]).astype(float)
            ti = np.nan
            if n >= min_n and (f[a] & ~f[b]).sum() >= 10 and (f[b] & ~f[a]).sum() >= 10:
                bb, se = W.ols(d[tgt].values, X)
                ti = bb[2] / se[2] if se[2] > 0 else np.nan
            rows.append({"a": a, "b": b, "target": tgt, "n": n, "mean": float(y.mean()) if n else np.nan,
                         "t_cell": t, "t_inter": ti})
    return pd.DataFrame(rows)


# ===================================================================================================== stages
def stage_dev():
    SCR.mkdir(parents=True, exist_ok=True)
    d = build_table()
    res = json.loads(JSON.read_text()) if JSON.exists() else {}
    if "holdout" in res:
        raise SystemExit("holdout already run; dev results are final")
    hyp_hash = hashlib.sha256(json.dumps(HYPOTHESES, sort_keys=True).encode()).hexdigest()[:16]
    R = {"generated": now(), "hypotheses_hash": hyp_hash, "n_games": {}, "feature_coverage": {}}
    for lab, rng in (("train_2003_19", TRAIN), ("dev_2020_22", DEVA), ("hold_2023_25", HOLDA)):
        R["n_games"][lab] = int(len(per(d, rng)))
    tr = per(d, TRAIN)
    R["feature_coverage"] = {c: rnd(tr[c].notna().mean(), 3) for c in FEATS}
    R["target_summary"] = {lab: {t: {"mean": rnd(per(d, rng)[t].mean(), 3), "sd": rnd(per(d, rng)[t].std(), 3)}
                                 for t in ("y_side", "y_tot")}
                           for lab, rng in (("2003-19", TRAIN), ("2020-22", DEVA))}

    # ------------------------------------------------------------------ Part B dev (first, independent of A)
    med = pass_rate_median(d)
    R["H1_pass_rate_median"] = rnd(med, 4)
    sig = hyp_signals(d, med)
    dv = d.season.between(*DEVB)
    B = {}
    for h in HYPOTHESES:
        s = sig[h["id"]]
        c = cell(d.loc[dv, h["target"]], s[dv])
        if c.get("n", 0) >= 5:
            c["per_season"] = rnd(c["n"] / len(set(d.loc[dv & (s != 0), "season"])), 1)
        c["by_era"] = {lab: cell(d.loc[d.season.between(lo, hi), h["target"]], s[d.season.between(lo, hi)])
                       for lab, (lo, hi) in (("2003-2011", (2003, 2011)), ("2012-2019", (2012, 2019)))}
        c["passes_dev_bar"] = bool(c.get("n", 0) >= B_MIN_N and (c.get("t") or 0) >= B_T_BAR
                                   and (c.get("cover") or 0) > BE)
        B[h["id"]] = c
    R["B_dev"] = B
    print("B dev:", {k: (v.get("n"), v.get("t"), v.get("cover"), v["passes_dev_bar"]) for k, v in B.items()})

    # ------------------------------------------------------------------ Part C sweep (dev)
    sw = sweep(d[dv])
    sw.to_parquet(SCR / "sweep_dev.parquet")
    ok = sw[sw.n >= 50]
    m_tests = len(ok)
    R["C_sweep_dev"] = {
        "flags": sorted(sweep_flags(d)), "pairs_x_targets_tested": int(m_tests),
        "nominal_p05_cell": int((ok.t_cell.abs() >= 1.96).sum()),
        "expected_by_chance": rnd(0.05 * m_tests, 1),
        "nominal_p05_interaction": int((ok.t_inter.abs() >= 1.96).sum()),
        "interaction_tested": int(ok.t_inter.notna().sum()),
        "bonferroni_t": rnd(-_ninv(0.025 / m_tests), 2),
        "bonferroni_cell": int((ok.t_cell.abs() >= -_ninv(0.025 / m_tests)).sum()),
        "bonferroni_interaction": int((ok.t_inter.abs() >= -_ninv(0.025 / m_tests)).sum()),
        "top10": ok.reindex(ok.t_cell.abs().sort_values(ascending=False).index).head(10)
                   .round(3).to_dict("records"),
    }
    print("C sweep dev:", {k: v for k, v in R["C_sweep_dev"].items() if k not in ("top10", "flags")})

    # ------------------------------------------------------------------ Part A: CV tuning on 2003-2019
    A = {}
    for target in ("y_side", "y_tot"):
        rows = []
        for spec in GRID:
            r = cv(tr, spec, target)
            rows.append(r)
            print(target, spec, r["delta"], r["r2_vs_market"])
        A[target] = {"cv": rows}
        for kind in ("gbm", "ridge2"):
            best = min((r for r in rows if r["spec"]["kind"] == kind), key=lambda r: r["mse_model"])
            A[target][f"best_{kind}"] = best["spec"]
    # CV baseline: training mean (no features)
    for target in ("y_side", "y_tot"):
        ys, ps = [], []
        for v in CV_VAL:
            ys.append(tr[tr.season == v][target].values)
            ps.append(np.full((tr.season == v).sum(), tr[tr.season < v][target].clip(-CLIP[target], CLIP[target]).mean()))
        A[target]["cv_mean_only"] = mse_delta(np.concatenate(ys), np.concatenate(ps))

    # ------------------------------------------------------------------ Part A: final fit, dev 2020-22
    dev = per(d, DEVA)
    first_sp, best_sp = early_spread(False)
    first_tot, best_tot = early_totals(False)
    for target in ("y_side", "y_tot"):
        for kind in ("gbm", "ridge2"):
            spec = A[target][f"best_{kind}"]
            m = fit_spec(spec, tr, target)
            p = m.predict(dev[FEATS].values)
            thr = {"top10": float(np.quantile(np.abs(p), 0.90)), "top5": float(np.quantile(np.abs(p), 0.95))}
            ent = {"spec": spec, "thresholds": thr, "pred_hash_2020_22": pred_hash(p),
                   "train_in_sample": mse_delta(tr[target].values, m.predict(tr[FEATS].values)),
                   "dev_2020_22": a_eval_period(dev[target].values, p, thr)}
            mk = "sp" if target == "y_side" else "tot"
            first, best = (first_sp, best_sp) if mk == "sp" else (first_tot, best_tot)
            pe, ng = early_preds(m, dev, first, mk)
            ent["early_dev_2020_22"] = {"games_with_early_line": ng,
                                        "baseline_both_sides": early_baseline(best, mk),
                                        "top10": early_bets(best, pe, thr["top10"], mk),
                                        "top5": early_bets(best, pe, thr["top5"], mk),
                                        "all": early_bets(best, pe, 0.0, mk)}
            if kind == "gbm":
                imp = permutation_importance_lite(m, dev, target)
                ent["dev_top_features_by_mse_increase"] = imp
            A[target][kind] = ent
            print(target, kind, ent["dev_2020_22"]["fit"], ent["dev_2020_22"]["top10"])
    R["A"] = A

    # ------------------------------------------------------------------ Part A null on dev (shuffled targets)
    R["A_null_dev"] = null_runs(d, A, periods={"dev_2020_22": DEVA})
    res["dev"] = _jsonable(R)
    OUT.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(res, indent=1))
    print("dev written", JSON)


def _ninv(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


def permutation_importance_lite(m, dev, target, top=8):
    rng = np.random.default_rng(0)
    X = dev[FEATS].values.copy()
    y = dev[target].values
    base = np.mean((y - m.predict(X)) ** 2)
    out = {}
    for j, c in enumerate(FEATS):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        out[c] = float(np.mean((y - m.predict(Xp)) ** 2) - base)
    return {k: rnd(v, 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1])[:top]}


def null_runs(d, A, periods, n_perm=N_PERM):
    """Refit each frozen spec on within-season shuffled 2003-19 targets; thresholds from that model's own 2020-22
    predictions (as for the real model); record R^2 and top-decile/5% cover in each period."""
    out = {}
    dev = per(d, DEVA)
    for target in ("y_side", "y_tot"):
        for kind in ("gbm", "ridge2"):
            spec = A[target][f"best_{kind}"] if f"best_{kind}" in A[target] else A[target][kind]["spec"]
            recs = []
            for k in range(n_perm):
                trp = permuted(d, 1000 + k, target)
                m = fit_spec(spec, trp, target)
                pdv = m.predict(dev[FEATS].values)
                thr = {"top10": np.quantile(np.abs(pdv), 0.90), "top5": np.quantile(np.abs(pdv), 0.95)}
                r = {}
                for lab, rng in periods.items():
                    x = per(d, rng)
                    p = pdv if rng == DEVA else m.predict(x[FEATS].values)
                    y = x[target].values
                    r[lab] = {"r2": r2_vs0(y, p), "c10": ats_sel(y, p, thr["top10"]).get("cover"),
                              "c5": ats_sel(y, p, thr["top5"]).get("cover")}
                recs.append(r)
            summ = {}
            for lab in periods:
                for key in ("r2", "c10", "c5"):
                    v = np.array([r[lab][key] for r in recs if r[lab][key] is not None], float)
                    summ[f"{lab}_{key}"] = {"mean": rnd(v.mean(), 4), "sd": rnd(v.std(ddof=1), 4),
                                            "p05": rnd(np.quantile(v, 0.05), 4), "p95": rnd(np.quantile(v, 0.95), 4),
                                            "max": rnd(v.max(), 4)}
            summ["n_perm"] = n_perm
            summ["raw"] = recs
            out[f"{target}_{kind}"] = summ
            print("null", target, kind, {k: v for k, v in summ.items() if k != "raw"})
    return out


def stage_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; never overwritten")
    res = json.loads(JSON.read_text())
    R = res["dev"]
    surv = [h for h in HYPOTHESES if R["B_dev"][h["id"]]["passes_dev_bar"]]
    sw = pd.read_parquet(SCR / "sweep_dev.parquet")
    ok = sw[sw.n >= 50]
    bt = R["C_sweep_dev"]["bonferroni_t"]
    fz = {
        "frozen_at": now(), "hypotheses_hash": R["hypotheses_hash"],
        "A": {t: {k: {"spec": R["A"][t][k]["spec"], "features": FEATS, "train_seasons": list(TRAIN),
                      "target_clip": CLIP[t], "thresholds_from_2020_22_preds": R["A"][t][k]["thresholds"],
                      "pred_hash_2020_22": R["A"][t][k]["pred_hash_2020_22"]}
                  for k in ("gbm", "ridge2")} for t in ("y_side", "y_tot")},
        "A_holdout_rule": "refit frozen spec on 2003-2019 (deterministic), assert 2020-22 prediction hash, score "
                          "2023-2025 once: R^2 vs market, ATS/O-U at the close for |pred| >= frozen top10/top5 "
                          "thresholds, early-line price-based CLV at the same thresholds",
        "B": {"bar_dev": f"n>={B_MIN_N}, one-sided t>={B_T_BAR}, cover>{BE}",
              "survivors": [h["id"] for h in surv],
              "holdout_bar": "one-sided p < 0.05/#survivors on mean residual toward; ATS cover reported with SE",
              "all_hypotheses": HYPOTHESES},
        "C": {"sweep_dev_nominal_winners": ok[ok.t_cell.abs() >= 1.96][["a", "b", "target", "n", "t_cell"]]
              .round(3).to_dict("records"),
              "sweep_dev_bonferroni_winners": ok[ok.t_cell.abs() >= bt][["a", "b", "target", "n", "t_cell"]]
              .round(3).to_dict("records")},
    }
    FROZEN.write_text(json.dumps(_jsonable(fz), indent=1))
    print(json.dumps({k: v for k, v in fz.items() if k != "C"}, indent=1)[:4000])


def stage_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout locked: freeze first and set EDGE_HOLDOUT")
    res = json.loads(JSON.read_text())
    if "holdout" in res:
        raise SystemExit("holdout already run once")
    fz = json.loads(FROZEN.read_text())
    d = build_table()
    tr, ho, dev = per(d, TRAIN), per(d, HOLDA), per(d, DEVA)
    H = {"run_at": now()}
    # ------------------------------------------------------------------ A holdout 2023-25
    first_sp, best_sp = early_spread(True)
    first_tot, best_tot = early_totals(True)
    A = {}
    for target in ("y_side", "y_tot"):
        A[target] = {}
        for kind in ("gbm", "ridge2"):
            f = fz["A"][target][kind]
            assert f["features"] == FEATS
            m = fit_spec(f["spec"], tr, target)
            assert pred_hash(m.predict(dev[FEATS].values)) == f["pred_hash_2020_22"], "frozen model not reproduced"
            thr = f["thresholds_from_2020_22_preds"]
            p = m.predict(ho[FEATS].values)
            ent = {"hold_2023_25": a_eval_period(ho[target].values, p, thr),
                   "by_season": {int(s): mse_delta(ho[ho.season == s][target].values, p[ho.season.values == s])
                                 for s in range(HOLDA[0], HOLDA[1] + 1)}}
            mk = "sp" if target == "y_side" else "tot"
            first, best = (first_sp, best_sp) if mk == "sp" else (first_tot, best_tot)
            pe, ng = early_preds(m, ho, first, mk)
            ent["early_hold_2023_25"] = {"games_with_early_line": ng, "baseline_both_sides": early_baseline(best, mk),
                                         "top10": early_bets(best, pe, thr["top10"], mk),
                                         "top5": early_bets(best, pe, thr["top5"], mk),
                                         "all": early_bets(best, pe, 0.0, mk)}
            A[target][kind] = ent
            print(target, kind, ent["hold_2023_25"]["fit"], ent["hold_2023_25"]["top10"])
    H["A"] = A
    H["A_null"] = null_runs(d, {t: {k: {"spec": fz["A"][t][k]["spec"]} for k in ("gbm", "ridge2")}
                                for t in ("y_side", "y_tot")},
                            periods={"dev_2020_22": DEVA, "hold_2023_25": HOLDA})
    # ------------------------------------------------------------------ B holdout 2020-25 (all reported; survivors count)
    med = pass_rate_median(d)
    sig = hyp_signals(d, med)
    hv = d.season.between(*HOLDB)
    surv = fz["B"]["survivors"]
    B = {}
    # early lines 2020-25: dev + holdout tables
    fs = pd.concat([early_spread(False)[0], first_sp])
    bs = pd.concat([early_spread(False)[1], best_sp])
    ftd, btd = early_totals(False)
    ft = pd.concat([ftd, first_tot])
    bt = pd.concat([btd, best_tot])
    d_e = d[hv].merge(fs[["game_id", "m_cons"]], on="game_id", how="left")
    sig_early = hyp_signals(d_e.assign(spread_line=d_e.m_cons.fillna(d_e.spread_line)), med, "spread_line")
    for h in HYPOTHESES:
        s = sig[h["id"]]
        c = cell(d.loc[hv, h["target"]], s[hv])
        c["by_period"] = {lab: cell(d.loc[d.season.between(lo, hi), h["target"]], s[d.season.between(lo, hi)])
                          for lab, (lo, hi) in (("2020-2022", DEVA), ("2023-2025", HOLDA))}
        c["survivor"] = h["id"] in surv
        if c["survivor"]:
            c["passes_holdout"] = bool((c.get("t") or 0) > 0 and (c.get("p_one_sided") or 1) < 0.05 / len(surv))
        se = pd.Series(sig_early[h["id"]].values, index=d_e.game_id.values)
        se = se[se != 0]
        if h["target"] == "y_side":
            c["early_clv_2020_25"] = early_bets(bs, se, 0.5, "sp")
        else:
            c["early_clv_2020_25"] = early_bets(bt, se, 0.5, "tot")
        B[h["id"]] = c
    H["B"] = B
    # ------------------------------------------------------------------ C sweep replication 2020-25
    swd = pd.read_parquet(SCR / "sweep_dev.parquet")
    swh = sweep(d[hv], min_n=1)
    m = swd.merge(swh, on=["a", "b", "target"], suffixes=("_dev", "_hold"))
    m = m[m.n_dev >= 50]
    win = m[m.t_cell_dev.abs() >= 1.96]
    rep = win[(np.sign(win.t_cell_hold) == np.sign(win.t_cell_dev)) & (win.t_cell_hold.abs() >= 1.96)]
    same = win[np.sign(win.t_cell_hold) == np.sign(win.t_cell_dev)]
    bt_ = res["dev"]["C_sweep_dev"]["bonferroni_t"]
    bw = m[m.t_cell_dev.abs() >= bt_]
    H["C"] = {"dev_nominal_winners": int(len(win)),
              "winners_with_holdout_n>=20": int((win.n_hold >= 20).sum()),
              "same_sign_in_holdout": int(len(same)), "replicated_p05_same_sign": int(len(rep)),
              "replicated_expected_if_null": rnd(0.025 * len(win), 1),
              "corr_t_dev_vs_hold_all": rnd(np.corrcoef(m.t_cell_dev, m.t_cell_hold.fillna(0))[0, 1], 3),
              "bonferroni_dev_winners": bw[["a", "b", "target", "n_dev", "t_cell_dev", "n_hold", "t_cell_hold"]]
              .round(3).to_dict("records"),
              "replicated": rep[["a", "b", "target", "n_dev", "t_cell_dev", "n_hold", "t_cell_hold", "mean_hold"]]
              .round(3).to_dict("records"),
              "holdout_nominal_any": int((m.t_cell_hold.abs() >= 1.96).sum()), "tests": int(len(m))}
    print("C:", {k: v for k, v in H["C"].items() if k not in ("replicated", "bonferroni_dev_winners")})
    res["holdout"] = _jsonable(H)
    JSON.write_text(json.dumps(res, indent=1))
    print("holdout written")


def stage_posthoc():
    """Diagnostics AFTER the one-shot holdout. Labelled post-hoc; they select nothing and change no verdict."""
    res = json.loads(JSON.read_text())
    if "holdout" not in res:
        raise SystemExit("run the holdout first")
    fz = json.loads(FROZEN.read_text())
    d = build_table()
    tr = per(d, TRAIN)
    P = {"run_at": now()}
    # 1. is the totals-GBM early-line CLV just recorded wind (unknown to a Tuesday bettor)?
    f = fz["A"]["y_tot"]["gbm"]
    m = fit_spec(f["spec"], tr, "y_tot")
    thr = f["thresholds_from_2020_22_preds"]["top10"]
    wd = {}
    for hold, rng in ((False, DEVA), (True, HOLDA)):
        first, best = early_totals(hold)
        x = per(d, rng)
        pe, _ = early_preds(m, x, first, "tot")
        wind = x.set_index("game_id").wind_o.reindex(pe.index).fillna(0)
        sel = pe[pe.abs() >= thr]
        wd["2023-25" if hold else "2020-22"] = {
            "n_top10": int(len(sel)), "share_under": rnd((sel < 0).mean(), 3),
            "share_recorded_wind_ge12": rnd((wind.reindex(sel.index) >= 12).mean(), 3),
            "top10_excluding_wind_ge12": early_bets(best, pe[wind < 12], thr, "tot"),
            "top10_wind_ge12_only": early_bets(best, pe[wind >= 12], thr, "tot")}
    P["tot_gbm_early_by_wind"] = wd
    # 2. sweep: pure-interaction replication and per-target correlation
    hv = d.season.between(*HOLDB)
    swd = pd.read_parquet(SCR / "sweep_dev.parquet")
    swh = sweep(d[hv], min_n=1)
    mm = swd.merge(swh, on=["a", "b", "target"], suffixes=("_dev", "_hold"))
    mm = mm[mm.n_dev >= 50]
    w = mm[mm.t_inter_dev.abs() >= 1.96]
    ok = mm.t_inter_dev.notna() & mm.t_inter_hold.notna()
    P["sweep_interaction_terms"] = {
        "dev_winners": int(len(w)), "same_sign_hold": int((np.sign(w.t_inter_dev) == np.sign(w.t_inter_hold)).sum()),
        "replicated_p05": int(((np.sign(w.t_inter_dev) == np.sign(w.t_inter_hold)) & (w.t_inter_hold.abs() >= 1.96)).sum()),
        "expected_if_null": rnd(0.025 * len(w), 1),
        "corr_t_dev_vs_hold": rnd(np.corrcoef(mm[ok].t_inter_dev, mm[ok].t_inter_hold)[0, 1], 3)}
    P["sweep_cell_corr_by_target"] = {
        t: rnd(np.corrcoef(mm[mm.target == t].t_cell_dev, mm[mm.target == t].t_cell_hold.fillna(0))[0, 1], 3)
        for t in ("y_side", "y_tot")}
    # 3. base rates and the single flags that carry the 'replicated' cells (cover rate = what pays at -110)
    P["over_rate_ex_push"] = {lab: rnd(((y := per(d, r).y_tot) > 0).sum() / (y != 0).sum(), 4)
                              for lab, r in (("2003-19", DEVB), ("2020-25", HOLDB))}
    P["mean_y_tot"] = {lab: rnd(per(d, r).y_tot.mean(), 3) for lab, r in (("2003-19", DEVB), ("2020-25", HOLDB))}
    flags = sweep_flags(d)
    one = pd.Series(1.0, index=d.index)
    P["single_flags"] = {}
    for k, tg in (("home_fav7", "y_side"), ("art_turf", "y_tot"), ("late_wk13", "y_tot")):
        P["single_flags"][f"{k}:{tg}"] = {
            lab: cell(d.loc[flags[k] & d.season.between(*r), tg], one[flags[k] & d.season.between(*r)])
            for lab, r in (("2003-19", DEVB), ("2020-22", DEVA), ("2023-25", HOLDA))}
    res["posthoc"] = _jsonable(P)
    JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["posthoc"], indent=1))


# ===================================================================================================== report
def f_fit(r):
    return (f"R² {r['r2_vs_market']:+.4f}, ΔMSE {r['delta']:+.2f} ± {r['delta_se']:.2f}, corr {r['corr']:+.3f}, "
            f"pred sd {r['pred_sd']:.2f}")


def f_ats(a):
    if "cover" not in a:
        return f"n {a.get('bets', 0)}"
    return (f"{a['bets']} bets, {a['w']}-{a['l']}-{a['push']}, cover {a['cover']:.3f} ± {a['cover_se']:.3f} "
            f"(z vs 52.4% {a['z_vs_524']:+.2f}), ROI {a['roi']:+.3f}")


def f_early(a):
    if "clv" not in a:
        return f"n {a.get('bets', 0)}"
    return (f"{a['bets']} bets, CLV {a['clv']:+.4f} (t {a['clv_t']:+.1f}), excess {a['excess_clv']:+.4f} "
            f"(t {a['excess_t']:+.1f}), ROI {a['roi']:+.3f} ± {a['roi_se']:.3f}")


def f_null(s, lab, key):
    q = s[f"{lab}_{key}"]
    return f"{q['mean']:+.4f} [{q['p05']:+.4f}, {q['p95']:+.4f}]" if key == "r2" else \
        f"{q['mean']:.3f} [{q['p05']:.3f}, {q['p95']:.3f}]"


def f_cell(c):
    if "t" not in c:
        return f"n {c.get('n', 0)}"
    return (f"n {c['n']}, {c['resid_toward']:+.2f} ± {c['se']:.2f} (t {c['t']:+.2f}), cover {c['cover']:.3f} ± "
            f"{c['cover_se']:.3f}")


def stage_report():
    res = json.loads(JSON.read_text())
    R, H = res["dev"], res.get("holdout")
    fz = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    vf = SCR / "verdict.md"
    L = ["# Factor combinations vs the closing market (spread and total)", "",
         "Question: do combinations of situational/game factors predict where games finish relative to the "
         "closing spread (`y_side` = home margin − spread_line) or total (`y_tot` = total − total_line), even though "
         "the individual factors don't?", "",
         f"Code: `scripts/research/interactions.py` (stages dev → freeze → holdout once → report). Numbers: "
         f"`output/research/interactions.json`. Frozen: `output/research/interactions_frozen.json` "
         f"(frozen {fz.get('frozen_at', '–')}; holdout run {H['run_at'] if H else 'not yet'}).", ""]
    if vf.exists():
        L += ["## Verdict", "", vf.read_text().strip(), ""]
    L += ["## Data", "",
          f"Games with closing spread and total: train 2003-19 {R['n_games']['train_2003_19']}, dev 2020-22 "
          f"{R['n_games']['dev_2020_22']}, holdout 2023-25 {R['n_games']['hold_2023_25']} (REG + postseason).",
          f"{len(FEATS)} pre-game features: the 22 Walters factor components (`walters_factors.build_factors`), "
          "rest days, travel km / tz shift / body-clock hour (`travel_features`), QB change / irregular starter / new "
          "QB / new coach flags, outdoor temp/wind/rain, turf, dome/warm team flags, EWMA (strictly prior games) points "
          "for/against and, 2012+, neutral pass rate, sec/snap, off/def EPA, pre-game win pct and conservative "
          "eliminated/clinched flags, plus the closing spread and total.",
          "Not strictly ex-ante: recorded game-time weather and the actual starting QB (see module docstring).", ""]
    # ---------------------------------------------------------------- A
    L += ["## A. All combinations at once", "",
          "Season-grouped expanding-window CV on 2003-19 (validation seasons 2009-19). R² is against the market "
          "(prediction 0 = the closing number is right): R² = 1 − SSE(model)/SSE(0). ΔMSE = model − market per game.",
          "", "| target | model | spec chosen by CV | CV 2009-19 | dev 2020-22 |", "|---|---|---|---|---|"]
    for t in ("y_side", "y_tot"):
        for k in ("gbm", "ridge2"):
            e = R["A"][t][k]
            cvr = min((r for r in R["A"][t]["cv"] if r["spec"] == e["spec"]), key=lambda r: r["mse_model"])
            L.append(f"| {t} | {k} | `{json.dumps(e['spec']['params'])}` | {f_fit(cvr)} | {f_fit(e['dev_2020_22']['fit'])} |")
        mo = R["A"][t]["cv_mean_only"]
        L.append(f"| {t} | training mean only | – | {f_fit(mo)} | |")
    L += ["", "All CV configurations: " + "; ".join(
        f"{t} {r['spec']['kind']} {json.dumps(r['spec']['params'])}: R² {r['r2_vs_market']:+.4f}"
        for t in ("y_side", "y_tot") for r in R["A"][t]["cv"]), ""]
    L += ["Betting only the largest predicted residuals at the close (−110, break-even 52.4%); thresholds = the "
          "90th/95th percentile of |prediction| on 2020-22, frozen for 2023-25. Null = same pipeline refit on "
          f"targets shuffled within season ({R['A_null_dev'][next(iter(R['A_null_dev']))]['n_perm']} permutations): "
          "mean [5%, 95%].", "",
          "| target | model | period | top 10% | top 5% | null R² | null top-10% cover | null top-5% cover |",
          "|---|---|---|---|---|---|---|---|"]
    for t in ("y_side", "y_tot"):
        for k in ("gbm", "ridge2"):
            e = R["A"][t][k]["dev_2020_22"]
            nd = R["A_null_dev"][f"{t}_{k}"]
            L.append(f"| {t} | {k} | dev 2020-22 | {f_ats(e['top10'])} | {f_ats(e['top5'])} | "
                     f"{f_null(nd, 'dev_2020_22', 'r2')} | {f_null(nd, 'dev_2020_22', 'c10')} | {f_null(nd, 'dev_2020_22', 'c5')} |")
            if H:
                h = H["A"][t][k]["hold_2023_25"]
                nh = H["A_null"][f"{t}_{k}"]
                L.append(f"| {t} | {k} | **holdout 2023-25** (R² {h['fit']['r2_vs_market']:+.4f}) | {f_ats(h['top10'])} | "
                         f"{f_ats(h['top5'])} | {f_null(nh, 'hold_2023_25', 'r2')} | {f_null(nh, 'hold_2023_25', 'c10')} | "
                         f"{f_null(nh, 'hold_2023_25', 'c5')} |")
    L += ["", "Early-line bets (first snapshot ≤ 9 days for spreads; Tuesday snapshot for totals) on the predicted "
          "side at the best allowed-book price (`my_books.json`), closing-line features replaced by the early "
          "consensus line. CLV is price-based (spreads: `edge_lab.closing_fair` mu_close_all via `line_move_model`; "
          "totals: sharp-close fair total, `totals.TotalDist`). Excess = CLV minus the mean CLV of both sides' best "
          "prices in that game (strips the vig level; its t is the directional test).", "",
          "| target | model | period | baseline both sides | all games | top 10% | top 5% |", "|---|---|---|---|---|---|---|"]
    for t in ("y_side", "y_tot"):
        for k in ("gbm", "ridge2"):
            rows = [("dev 2020-22", R["A"][t][k]["early_dev_2020_22"])]
            if H:
                rows.append(("**holdout 2023-25**", H["A"][t][k]["early_hold_2023_25"]))
            for lab, e in rows:
                b = e["baseline_both_sides"]
                L.append(f"| {t} | {k} | {lab} | CLV {b['clv']:+.4f}, ROI {b['roi']:+.3f} (n {b['bets']}) | "
                         f"{f_early(e['all'])} | {f_early(e['top10'])} | {f_early(e['top5'])} |")
    L += ["", "GBM dev permutation importance (MSE increase, 2020-22): " + "; ".join(
        f"{t}: " + ", ".join(f"{c} {v:+.2f}" for c, v in R["A"][t]["gbm"]["dev_top_features_by_mse_increase"].items())
        for t in ("y_side", "y_tot")), ""]
    if H:
        L += ["Holdout by season (R² vs market): " + "; ".join(
            f"{t} {k}: " + ", ".join(f"{s} {v['r2_vs_market']:+.4f}" for s, v in H["A"][t][k]["by_season"].items())
            for t in ("y_side", "y_tot") for k in ("gbm", "ridge2")), ""]
    # ---------------------------------------------------------------- B
    L += ["## B. Hypothesis-driven combinations (pre-registered before any result)", "",
          f"Dev bar (2003-19): n ≥ {B_MIN_N}, one-sided t ≥ {B_T_BAR} (p < 0.05/10) on the mean residual in the "
          "stated direction, and cover > 52.4%. Survivors tested once on 2020-25. Residual = margin − spread_line "
          "(side) or total − total_line (totals), signed toward the bet. Weather = recorded conditions.", ""]
    for h in HYPOTHESES:
        L.append(f"- **{h['id']}** ({h['target']}): {h['definition']}. *Why:* {h['rationale']}.")
    L += ["", "| hypothesis | dev 2003-19 | passes dev bar | 2020-25 (all reported) | 2020-25 early-line CLV |",
          "|---|---|---|---|---|"]
    for h in HYPOTHESES:
        dvc = R["B_dev"][h["id"]]
        hc = H["B"][h["id"]] if H else {}
        ho_txt = f_cell(hc) if hc else "–"
        if hc.get("survivor"):
            ho_txt += f" — **survivor, holdout {'PASS' if hc['passes_holdout'] else 'FAIL'}**"
        L.append(f"| {h['id']} | {f_cell(dvc)} | {'yes' if dvc['passes_dev_bar'] else 'no'} | {ho_txt} | "
                 f"{f_early(hc['early_clv_2020_25']) if hc else '–'} |")
    L += ["", "Dev by era (2003-11 / 2012-19): " + "; ".join(
        f"{h['id']}: {R['B_dev'][h['id']]['by_era']['2003-2011'].get('t', '–')} / "
        f"{R['B_dev'][h['id']]['by_era']['2012-2019'].get('t', '–')}" for h in HYPOTHESES) + " (t values)", ""]
    if H:
        L += ["2020-22 / 2023-25 split (t): " + "; ".join(
            f"{h['id']}: {H['B'][h['id']]['by_period']['2020-2022'].get('t', '–')} / "
            f"{H['B'][h['id']]['by_period']['2023-2025'].get('t', '–')}" for h in HYPOTHESES), ""]
    # ---------------------------------------------------------------- C
    c = R["C_sweep_dev"]
    L += ["## C. False-discovery picture", "",
          f"**B scorecard:** {sum(R['B_dev'][h['id']]['passes_dev_bar'] for h in HYPOTHESES)} of 10 pre-registered "
          f"combinations passed the dev bar; "
          + (f"{sum(1 for h in HYPOTHESES if H['B'][h['id']].get('passes_holdout'))} held up on 2020-25." if H else ""),
          f"Nominally 'significant' at p < 0.05 in dev (two-sided t ≥ 1.96 either direction): "
          f"{sum(1 for h in HYPOTHESES if abs(R['B_dev'][h['id']].get('t') or 0) >= 1.96)} of 10.", "",
          f"**Exhaustive sweep** of all pairwise AND-combinations of {len(c['flags'])} binary flags "
          f"({', '.join(c['flags'])}), both targets, cells with n ≥ 50 in 2003-19: {c['pairs_x_targets_tested']} tests.",
          f"- dev 'winners' at |t| ≥ 1.96 (cell mean ≠ 0): {c['nominal_p05_cell']} (expected by chance ≈ "
          f"{c['expected_by_chance']}); pure interaction term (y ~ a + b + a·b) |t| ≥ 1.96: "
          f"{c['nominal_p05_interaction']} of {c['interaction_tested']}.",
          f"- Bonferroni (|t| ≥ {c['bonferroni_t']}): {c['bonferroni_cell']} cells, {c['bonferroni_interaction']} "
          "interaction terms.",
          "- Largest dev cells: " + "; ".join(f"{r['a']}×{r['b']} {r['target']} n {r['n']} mean {r['mean']:+.2f} "
                                               f"t {r['t_cell']:+.2f}" for r in c["top10"][:6])]
    if H:
        hc = H["C"]
        L += [f"- Of the {hc['dev_nominal_winners']} dev winners, {hc['same_sign_in_holdout']} had the same sign in "
              f"2020-25 and {hc['replicated_p05_same_sign']} were again |t| ≥ 1.96 with the same sign (≈ "
              f"{hc['replicated_expected_if_null']} expected if all were noise). Correlation of dev t vs 2020-25 t "
              f"over all {hc['tests']} tests: {hc['corr_t_dev_vs_hold_all']:+.3f}. Holdout cells at |t| ≥ 1.96 "
              f"regardless of dev: {hc['holdout_nominal_any']}.",
              "- Bonferroni dev winners in 2020-25: " + ("; ".join(
                  f"{r['a']}×{r['b']} {r['target']} dev t {r['t_cell_dev']:+.2f} → hold t {r['t_cell_hold']:+.2f} "
                  f"(n {r['n_hold']})" for r in hc["bonferroni_dev_winners"]) or "none"),
              "- Replicated at p < 0.05: " + ("; ".join(
                  f"{r['a']}×{r['b']} {r['target']} dev t {r['t_cell_dev']:+.2f} → hold t {r['t_cell_hold']:+.2f} "
                  f"(n {r['n_hold']}, mean {r['mean_hold']:+.2f})" for r in hc["replicated"]) or "none")]
    P = res.get("posthoc")
    if P:
        L += ["", "## Post-hoc diagnostics (after the one-shot holdout; select nothing, change no verdict)", ""]
        for lab, r in P["tot_gbm_early_by_wind"].items():
            L.append(f"- Totals GBM early-line top decile {lab}: {r['n_top10']} bets, {r['share_under']:.0%} unders, "
                     f"{r['share_recorded_wind_ge12']:.0%} in games with RECORDED wind ≥ 12 mph. Wind ≥ 12 only: "
                     f"{f_early(r['top10_wind_ge12_only'])}. Excluding them: {f_early(r['top10_excluding_wind_ge12'])}.")
        si = P["sweep_interaction_terms"]
        L += [f"- Sweep, pure interaction terms (y ~ a + b + a·b): {si['dev_winners']} dev winners → "
              f"{si['same_sign_hold']} same sign in 2020-25, {si['replicated_p05']} replicated at p < 0.05 "
              f"(≈ {si['expected_if_null']} expected by chance); corr(dev t, 2020-25 t) = {si['corr_t_dev_vs_hold']:+.3f}.",
              f"- Sweep cell-t correlation by target: " + ", ".join(f"{k} {v:+.3f}" for k, v in P["sweep_cell_corr_by_target"].items())
              + f". Totals residuals are right-skewed: mean total − total_line {P['mean_y_tot']} but over rate (ex push) "
              f"{P['over_rate_ex_push']} — mean-residual cell tests for totals pick up skew that does not pay at −110.",
              "- Single flags behind the 'replicated' cells (cover rate is what pays): " + "; ".join(
                  f"{k}: " + ", ".join(f"{lab} {f_cell(c)}" for lab, c in v.items()) for k, v in P["single_flags"].items())]
    L.append("")
    MD.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": stage_dev, "freeze": stage_freeze, "holdout": stage_holdout, "posthoc": stage_posthoc,
     "report": stage_report}[stage]()
