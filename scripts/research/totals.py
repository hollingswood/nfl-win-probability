"""NFL game TOTALS (over/under) research: is there a bettable edge at the lines we could actually bet?

Stages (run in order; each writes into output/research/):
  python scripts/research/totals.py dev       # QA + angle development on 2020-2022 only -> totals.json["qa","dev"]
  python scripts/research/totals.py freeze    # writes totals_frozen.json from FROZEN_RULES (refuses to overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/totals.py holdout
                                              # 2023-2025, frozen rules evaluated ONCE -> totals.json["holdout"]
  python scripts/research/totals.py posthoc   # post-hoc diagnostics on the holdout (labelled; change no verdict)
  output/research/totals.md is written by hand from totals.json.

Definitions
  * Odds: data/historical_odds/totals/ (Tue 14:10 UTC, Fri 21:40 UTC, ~75 min before each kickoff; region us).
    Rows with broken prices / off-market points are dropped (see load_totals). Matched to games like
    replay_early_lines.match_games (home/away/season, commence within 48h).
  * Total-points distribution: Normal(mu, SIGMA) on integers 0..120 times empirical key-number weights
    w(k) = (actual count at k + s) / (normal-expected count at k + s), fit on 2012-2019 finals vs nflverse
    closing totals. Handles pushes on whole-number totals and the value of half points.
  * Fair expected total of a book = mu such that P(T>pt)/(P(T>pt)+P(T<pt)) = its no-vig over probability.
    Snapshot fair = median over books (all books / sharp books = lowvig, betonlineag, circasports, bookmaker).
  * Closing fair (CLV reference) = fair mu at the LAST pre-kickoff snapshot, sharp books if any are present
    there, else all books. clv_all = same with all books.
  * CLV of a bet (side, point, price) = its EV per unit under the closing fair distribution:
        P(win)*decimal + P(push) - 1.
  * PnL graded on the final total at the bet's point/price (push refunds).
  * Bettable = my_books.json allowed books only. One bet per game, locked at the first qualifying snapshot,
    best-scoring book there (edge_lab.pick_one_per_game).

Weather caveat: nflverse temp/wind are RECORDED game conditions (NaN indoors), i.e. a near-perfect forecast.
For a bet 75 minutes before kickoff that is close to what a live forecast would say; for a Tuesday bet it is
optimistic (a Tuesday bettor has a 5-day forecast at best).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import replay_early_lines as R  # noqa: E402
from edge_lab import SHARP, pick_one_per_game, stats  # noqa: E402
from nflpred import features as F, odds as O  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "totals.json"
FROZEN = OUT / "totals_frozen.json"
MD = OUT / "totals.md"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/totals")
DEV, HOLD = (2020, 2021, 2022), (2023, 2024, 2025)
KT = np.arange(0, 121)
MU_GRID = np.round(np.arange(20.0, 75.0001, 0.05), 2)
PT_GRID = np.arange(20.0, 80.001, 0.5)
MODEL_FIRST_TEST = 2015


def dec(p):
    p = np.asarray(p, float)
    return np.where(p > 0, 1 + p / 100, 1 + 100 / -p)


def imp(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p < 0, -p / (-p + 100), 100 / (p + 100))


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2))  # one-sided P(Z > t)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    return x


# ================================================================== games + total-points distribution
def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[g.season >= 2012].copy()
    for c in ("home_team", "away_team"):
        g[c] = F._norm_team(g[c])
    g["gameday"] = pd.to_datetime(g.gameday)
    g["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()], utc=True)
    g["indoor"] = g.roof.isin(["dome", "closed"])
    g["wind_o"] = np.where(g.indoor, np.nan, g.wind)            # recorded wind, outdoor/open games only
    g["temp_o"] = np.where(g.indoor, np.nan, g.temp)
    return g


class TotalDist:
    """Normal(mu, sigma) on integer totals x empirical key-number weights (fit on 2012-2019)."""

    def __init__(self, g: pd.DataFrame, smooth=10.0, cap=(0.2, 3.0)):
        tr = g[g.season.between(2012, 2019) & g.total.notna() & g.total_line.notna()]
        self.sigma = float((tr.total - tr.total_line).std())
        base = self._normal(tr.total_line.values)
        expct = base.sum(axis=0)
        act = np.bincount(np.clip(tr.total.astype(int).values, 0, KT[-1]), minlength=len(KT))
        self.w = np.clip((act + smooth) / (expct + smooth), *cap)
        self.n_train = len(tr)
        P = self.pmf(MU_GRID)                                    # (n_mu, n_k)
        cdf = np.cumsum(P, axis=1)
        # over wins if T > pt; under if T < pt; push if T == pt (whole numbers only)
        self.p_over = np.stack([1 - cdf[:, int(math.floor(pt))] for pt in PT_GRID], axis=1)
        self.p_under = np.stack([(cdf[:, int(math.ceil(pt)) - 1] if pt == int(pt) else cdf[:, int(math.floor(pt))])
                                 for pt in PT_GRID], axis=1)
        self.p_push = 1 - self.p_over - self.p_under
        self.mean_at = (P * KT[None, :]).sum(axis=1)             # E[T] at each grid mu (~ mu)

    def _normal(self, mu):
        mu = np.atleast_1d(np.asarray(mu, float))[:, None]
        z = (KT[None, :] - mu) / self.sigma
        p = np.exp(-0.5 * z * z)
        return p / p.sum(axis=1, keepdims=True)

    def pmf(self, mu):
        p = self._normal(mu) * self.w[None, :]
        return p / p.sum(axis=1, keepdims=True)

    def _idx(self, mu, pt):
        mi = np.clip(np.round((np.asarray(mu, float) - MU_GRID[0]) / 0.05), 0, len(MU_GRID) - 1).astype(int)
        pi = np.clip(np.round((np.asarray(pt, float) - PT_GRID[0]) / 0.5), 0, len(PT_GRID) - 1).astype(int)
        return mi, pi

    def probs(self, mu, pt, side):
        """(P(win), P(push)) for over/under bets at point pt when the expected total E[T] is mu (vectorized)."""
        mu = np.asarray(mu, float)
        bad = ~np.isfinite(mu)
        loc = np.interp(np.nan_to_num(mu, nan=45.0), self.mean_at, MU_GRID)   # E[T] -> location parameter
        mi, pi = self._idx(loc, pt)
        side = np.asarray(side)
        win = np.where(side == "over", self.p_over[mi, pi], self.p_under[mi, pi])
        push = self.p_push[mi, pi]
        win = np.where(bad, np.nan, win)
        return win, np.where(bad, np.nan, push)

    def ev(self, mu, pt, price, side):
        win, push = self.probs(mu, pt, side)
        return win * dec(price) + push - 1

    def implied_mu(self, pt, nv_over):
        """Fair expected total from a point and its no-vig over probability (vectorized over rows)."""
        pt = np.asarray(pt, float)
        nv = np.asarray(nv_over, float)
        out = np.full(len(pt), np.nan)
        _, pi = self._idx(np.full(len(pt), 45.0), pt)
        for j in np.unique(pi):
            m = pi == j
            q = self.p_over[:, j] / np.maximum(self.p_over[:, j] + self.p_under[:, j], 1e-12)  # increasing in mu
            out[m] = np.interp(nv[m], q, MU_GRID)
        return np.interp(out, MU_GRID, self.mean_at)             # report as the expected total E[T]


# ================================================================== odds
def load_totals(seasons) -> pd.DataFrame:
    fr = []
    for s in seasons:
        f = ROOT / "data" / "historical_odds" / "totals" / f"nfl_odds_{s}.csv.gz"
        fr.append(pd.read_csv(f).assign(season=s))
    o = pd.concat(fr, ignore_index=True)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    o["home"] = F._norm_team(o.home)
    o["away"] = F._norm_team(o.away)
    n0 = len(o)
    ok = (o.tot_over_price.between(-250, 200) & o.tot_under_price.between(-250, 200)
          & (o.tot_over_price.abs() >= 100) & (o.tot_under_price.abs() >= 100) & o.tot_point.between(25, 80))
    o = o[ok].copy()
    o["overround"] = imp(o.tot_over_price) + imp(o.tot_under_price)
    o = o[o.overround.between(1.0, 1.12)]
    med = o.groupby(["event_id", "requested_ts"]).tot_point.transform("median")
    o = o[(o.tot_point - med).abs() <= 5]
    o.attrs["dropped"] = n0 - len(o)
    return o


def build(seasons, dist: TotalDist, g: pd.DataFrame, preds: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (bets, snaps, qa). bets = every (allowed book, snapshot, side) we could have bet."""
    o = load_totals(seasons)
    dropped = o.attrs["dropped"]
    gg = g[g.season.isin(seasons)]
    o = R.match_games(o, gg).merge(gg[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick].copy()
    o["hours_before"] = (o.kick - o.requested_ts).dt.total_seconds() / 3600
    nv = imp(o.tot_over_price) / o.overround
    o["mu_b"] = dist.implied_mu(o.tot_point.values, nv.values)
    key = ["game_id", "requested_ts"]
    snaps = o.groupby(key).agg(mu_all=("mu_b", "median"), pt_all=("tot_point", "median"),
                               n_books=("book", "nunique"), hours_before=("hours_before", "first")).reset_index()
    sh = o[o.book.isin(SHARP)].groupby(key).agg(mu_sharp=("mu_b", "median"), pt_sharp=("tot_point", "median"),
                                                n_sharp=("book", "nunique")).reset_index()
    snaps = snaps.merge(sh, on=key, how="left")
    snaps["n_sharp"] = snaps.n_sharp.fillna(0).astype(int)
    snaps["mu_ref"] = snaps.mu_sharp.fillna(snaps.mu_all)
    # snapshot type
    wd = snaps.requested_ts.dt.day_name()
    hm = snaps.requested_ts.dt.strftime("%H:%M")
    last = snaps.requested_ts == snaps.groupby("game_id").requested_ts.transform("max")
    snaps["snap"] = "other"
    snaps.loc[(wd == "Tuesday") & (hm == "14:10") & (snaps.hours_before <= 7 * 24), "snap"] = "tue"
    # Friday snapshot only when it falls in the game's own week (<= 3.5 days out): for a Thursday game the
    # previous Friday precedes both teams' last game, so model features would not have been known yet
    snaps.loc[(wd == "Friday") & (hm == "21:40") & (snaps.hours_before <= 3.5 * 24), "snap"] = "fri"
    snaps.loc[last & (snaps.hours_before <= 3), "snap"] = "close"
    close = snaps[last].rename(columns={"mu_all": "mu_close_all", "mu_sharp": "mu_close_sharp",
                                        "pt_all": "pt_close", "hours_before": "close_hours",
                                        "n_sharp": "close_n_sharp"})[
        ["game_id", "mu_close_all", "mu_close_sharp", "pt_close", "close_hours", "close_n_sharp"]]
    close["mu_close"] = close.mu_close_sharp.fillna(close.mu_close_all)
    tue = snaps[snaps.snap == "tue"].groupby("game_id").agg(mu_tue=("mu_ref", "last"), mu_tue_all=("mu_all", "last"),
                                                            pt_tue=("pt_all", "last"))
    fri = snaps[snaps.snap == "fri"].groupby("game_id").agg(mu_fri=("mu_ref", "last"), pt_fri=("pt_all", "last"))
    gcols = ["game_id", "season", "week", "game_type", "gameday", "weekday", "gametime", "home_team", "away_team",
             "total", "total_line", "over_odds", "under_odds", "roof", "indoor", "wind_o", "temp_o", "div_game",
             "spread_line"]
    games = gg[gcols].merge(close, on="game_id", how="inner").merge(tue, on="game_id", how="left") \
        .merge(fri, on="game_id", how="left")
    if preds is not None:
        games = games.merge(preds[["game_id", "pred_base", "pred_wx", "k_base", "k_wx"]], on="game_id", how="left")
    snaps = snaps.merge(games, on="game_id")

    allowed = (O.load_allowed_books() or set()) & set(o.book.unique())
    b = o[o.book.isin(allowed)][key + ["book", "tot_point", "tot_over_price", "tot_under_price"]]
    rows = []
    for side in ("over", "under"):
        x = b.rename(columns={"tot_point": "point"}).copy()
        x["side"] = side
        x["price"] = x[f"tot_{side}_price"]
        rows.append(x[key + ["book", "side", "point", "price"]])
    t = pd.concat(rows, ignore_index=True).merge(snaps, on=key)
    t = t[t.total.notna() & (t.close_hours <= 3)].copy()     # CLV needs a genuine pre-kickoff close
    side = t.side.values
    t["dec"] = dec(t.price)
    t["tot_clv"] = dist.ev(t.mu_close, t.point, t.price, side)
    t["clv_all"] = dist.ev(t.mu_close_all, t.point, t.price, side)
    t["clv_sharp"] = dist.ev(t.mu_close_sharp, t.point, t.price, side)
    t["ev_sharp_now"] = dist.ev(t.mu_sharp, t.point, t.price, side)
    t["ev_all_now"] = dist.ev(t.mu_all, t.point, t.price, side)
    t["ev_ref_now"] = dist.ev(t.mu_ref, t.point, t.price, side)
    adj = np.where(side == "over", t.total - t.point, t.point - t.total)
    t["tot_pnl"] = np.where(adj > 0, t.dec - 1, np.where(adj < 0, -1.0, 0.0))
    for v in ("base", "wx"):
        if f"pred_{v}" in t:
            mu_m = t.mu_ref + t[f"k_{v}"] * (t[f"pred_{v}"] - t.mu_ref)
            t[f"mu_model_{v}"] = mu_m
            t[f"ev_model_{v}"] = dist.ev(mu_m, t.point, t.price, side)
    t["price_rank"] = t.groupby(["game_id", "requested_ts", "side"]).ev_ref_now.rank(ascending=False, method="first")
    qa = {"rows_dropped_bad_price_or_point": int(dropped), "allowed_books_present": sorted(allowed),
          "sharp_books_present": sorted(SHARP & set(o.book.unique())),
          "books_present": sorted(o.book.unique())}
    return t, snaps, qa


# ================================================================== walk-forward totals model
def pace_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp.play_type.isin(["pass", "run"])].sort_values(["game_id", "play_id"]).copy()
    p["posteam"] = F._norm_team(p.posteam)
    nx = p.groupby("game_id")[["game_seconds_remaining", "fixed_drive", "qtr"]].shift(-1)
    gap = p.game_seconds_remaining - nx.game_seconds_remaining
    ok = ((nx.fixed_drive == p.fixed_drive) & (nx.qtr == p.qtr) & (p.qtr <= 3) & (p.score_differential.abs() <= 10)
          & gap.between(3, 60) & (p.half_seconds_remaining > 120))
    p["gap"] = np.where(ok, gap, np.nan)
    return p.groupby(["game_id", "posteam"]).agg(spp=("gap", "mean"), plays=("play_id", "size")).reset_index() \
        .rename(columns={"posteam": "team"})


def model_features(g: pd.DataFrame, max_season: int) -> pd.DataFrame:
    cache = SCR / f"model_feats_{max_season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["game_id", "play_id", "posteam", "defteam", "pass", "rush", "epa", "success", "qb_kneel", "qb_spike",
            "interception", "fumble_lost", "fumbled_1_team", "wp", "play_type", "game_seconds_remaining",
            "half_seconds_remaining", "fixed_drive", "qtr", "score_differential"]
    tg, pc = [], []
    for s in range(2012, max_season + 1):
        pbp = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet", columns=cols)
        tg.append(F.team_game_stats(pbp))
        pc.append(pace_stats(pbp))
    tgs = pd.concat(tg).merge(pd.concat(pc), on=["game_id", "team"], how="left")
    sched = g[(g.season <= max_season) & g.total.notna()].sort_values(["gameday", "game_id"])
    lg = F.team_long(sched).merge(tgs, on=["game_id", "team"], how="left")
    lg["off_epa"] = lg.off_epa.clip(-0.3, 0.3)
    lg["def_epa"] = lg.def_epa.clip(-0.3, 0.3)
    lg = lg.sort_values(["team", "gameday", "game_id"])
    for c in ("off_epa", "def_epa", "off_sr", "def_sr", "spp", "plays", "pf", "pa"):
        lg[f"pre_{c}"] = F._pre_ewm(lg, c)                        # strictly prior games only
    pre = [c for c in lg.columns if c.startswith("pre_")]
    h = lg[["game_id", "team"] + pre].rename(columns={"team": "home_team", **{c: "h_" + c for c in pre}})
    a = lg[["game_id", "team"] + pre].rename(columns={"team": "away_team", **{c: "a_" + c for c in pre}})
    d = sched.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    X = pd.DataFrame({"game_id": d.game_id})
    X["epa_sum"] = d.h_pre_off_epa + d.h_pre_def_epa + d.a_pre_off_epa + d.a_pre_def_epa
    X["sr_sum"] = d.h_pre_off_sr + d.h_pre_def_sr + d.a_pre_off_sr + d.a_pre_def_sr
    X["spp_sum"] = d.h_pre_spp + d.a_pre_spp
    X["plays_sum"] = d.h_pre_plays + d.a_pre_plays
    X["pts_sum"] = (d.h_pre_pf + d.h_pre_pa + d.a_pre_pf + d.a_pre_pa) / 2
    X["indoor"] = d.indoor.astype(float).values
    X["div"] = d.div_game.astype(float).values
    # league scoring level: mean total of the last 150 games played before the TUESDAY of this game's week
    # (so it is known at the Tuesday snapshot; e.g. a Sunday game does not see that week's Thursday game)
    s2 = sched[["gameday", "total"]].sort_values("gameday").reset_index(drop=True)
    cs = np.concatenate([[0], np.cumsum(s2.total.values)])
    back = (d.gameday.dt.weekday - 1) % 7
    cutoff = d.gameday - pd.to_timedelta(np.where(back == 0, 7, back), unit="D")
    i = np.searchsorted(s2.gameday.values, cutoff.values, side="left")     # games strictly before cutoff
    lo = np.maximum(0, i - 150)
    X["lg_level"] = np.where(i > 0, (cs[i] - cs[lo]) / np.maximum(1, i - lo), np.nan)
    # weather (RECORDED -> optimistic proxy); outdoor games with missing wind get the outdoor median + flag
    wind = d.wind_o.values.astype(float)
    X["wind"] = np.where(d.indoor, 0.0, np.nan_to_num(wind, nan=8.0))
    X["wind15"] = np.where(d.indoor, 0.0, np.clip(np.nan_to_num(wind, nan=8.0) - 10, 0, None))
    X["cold"] = np.where(d.indoor, 0.0, np.clip(40 - np.nan_to_num(d.temp_o.values.astype(float), nan=55), 0, None))
    X = X.merge(sched[["game_id", "season", "total", "total_line"]], on="game_id")
    SCR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(cache)
    return X


BASE = ["epa_sum", "sr_sum", "spp_sum", "plays_sum", "pts_sum", "indoor", "div", "lg_level"]
WX = BASE + ["wind", "wind15", "cold"]


def _ridge(X, y, lam=5.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    L = lam * np.eye(Z1.shape[1])
    L[0, 0] = 0
    beta = np.linalg.solve(Z1.T @ Z1 + L, Z1.T @ y)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ beta


def walk_forward_preds(g: pd.DataFrame, max_season: int) -> pd.DataFrame:
    X = model_features(g, max_season).dropna(subset=BASE).copy()
    X = X[X.season >= 2013]                                      # 2012 = EWMA warm-up
    parts = []
    for s in range(MODEL_FIRST_TEST, max_season + 1):
        tr, te = X[X.season < s], X[X.season == s].copy()
        for v, cols in (("base", BASE), ("wx", WX)):
            f = _ridge(tr[cols].values, tr.total.values)
            te[f"pred_{v}"] = f(te[cols].values)
        parts.append(te)
    P = pd.concat(parts)
    # blend weight k(s): OLS (no intercept) of (total - line) on (pred - line) over OOS seasons < s
    for v in ("base", "wx"):
        ks = {}
        for s in range(MODEL_FIRST_TEST, max_season + 1):
            h = P[(P.season < s) & P.total_line.notna()]
            if len(h) < 200:
                ks[s] = 0.0
                continue
            x, y = (h[f"pred_{v}"] - h.total_line).values, (h.total - h.total_line).values
            ks[s] = float(np.clip((x @ y) / (x @ x), 0, 1))
        P[f"k_{v}"] = P.season.map(ks)
    return P


def model_eval(P: pd.DataFrame, seasons) -> dict:
    out = {}
    for s in list(seasons) + ["all"]:
        h = P if s == "all" else P[P.season == s]
        h = h[h.total_line.notna()]
        rec = {"n": len(h), "rmse_line": float(np.sqrt(((h.total - h.total_line) ** 2).mean()))}
        for v in ("base", "wx"):
            e = h[f"pred_{v}"] - h.total_line
            bl = h.total_line + h[f"k_{v}"] * e
            rec[f"rmse_model_{v}"] = float(np.sqrt(((h.total - h[f"pred_{v}"]) ** 2).mean()))
            rec[f"rmse_blend_{v}"] = float(np.sqrt(((h.total - bl) ** 2).mean()))
            rec[f"corr_{v}_vs_close_resid"] = float(np.corrcoef(e, h.total - h.total_line)[0, 1])
            rec[f"k_{v}"] = float(h[f"k_{v}"].mean())
        out[str(s)] = rec
    return out


# ================================================================== rules
def apply_rule(t: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Candidate bets for a rule, then one per game (first qualifying snapshot, best score)."""
    x = t[t.snap.isin(rule["snaps"])]
    k = rule["kind"]
    if k == "soft_vs_sharp":
        x = x[x.mu_sharp.notna() & (x.ev_sharp_now > rule["min_ev"])]
        if rule.get("side"):
            x = x[x.side == rule["side"]]
        score = "ev_sharp_now"
    elif k == "wind":
        x = x[(~x.indoor) & (x.wind_o >= rule["wind_min"]) & (x.side == rule["side"])]
        if "min_ev" in rule:
            x = x[x.ev_ref_now > rule["min_ev"]]
        score = "ev_ref_now"
    elif k == "model":
        c = f"ev_model_{rule['variant']}"
        x = x[x[c] > rule["min_ev"]]
        if rule.get("side"):
            x = x[x.side == rule["side"]]
        score = c
    elif k == "move":
        # predicted close = current fair + beta[snap] * (walk-forward model - current fair); bet on EV under it
        beta = x.snap.map(rule["beta"]).astype(float)
        mu_pc = x.mu_ref + beta * (x[f"pred_{rule['variant']}"] - x.mu_ref)
        x = x.assign(ev_move=DIST.ev(mu_pc.values, x.point.values, x.price.values, x.side.values))
        x = x[x.ev_move > rule["min_ev"]]
        score = "ev_move"
    elif k == "blind":
        x = x[x.side == rule["side"]]
        if "total_min" in rule:
            x = x[x.pt_all >= rule["total_min"]]
        if "total_max" in rule:
            x = x[x.pt_all <= rule["total_max"]]
        if "min_ev" in rule:
            x = x[x.ev_ref_now > rule["min_ev"]]
        score = "ev_ref_now"
    else:
        raise ValueError(k)
    return pick_one_per_game(x, "first", score)


def rstats(b: pd.DataFrame) -> dict:
    s = stats(b, "tot")
    if s.get("bets", 0) == 0:
        return s
    s["clv_p_one_sided"] = round(pval(s["clv_t"]), 4)
    for c in ("clv_all", "clv_sharp"):
        v = b[c].dropna()
        s[f"{c}_mean"] = round(float(v.mean()), 4) if len(v) else None
    s["mean_price_dec"] = round(float(b.dec.mean()), 3)
    s["pct_over"] = round(float((b.side == "over").mean()), 3)
    s["by_season"] = {int(k): {"n": len(d), "clv": round(float(d.tot_clv.mean()), 4), "roi": round(float(d.tot_pnl.mean()), 4)}
                      for k, d in b.groupby("season")}
    s["by_snap"] = {k: len(d) for k, d in b.groupby("snap")}
    return s


# ================================================================== QA
def qa_block(t: pd.DataFrame, snaps: pd.DataFrame, g: pd.DataFrame, seasons, dist: TotalDist, movement=True) -> dict:
    games = snaps.drop_duplicates("game_id")
    q = {"per_season": {}}
    for s in seasons:
        gs = g[(g.season == s) & g.total.notna()]
        sn = snaps[snaps.season == s]
        gm = games[games.season == s]
        q["per_season"][s] = {
            "schedule_games": len(gs), "matched_games": int(gm.game_id.nunique()),
            "games_with_tue": int(sn[sn.snap == "tue"].game_id.nunique()),
            "games_with_fri": int(sn[sn.snap == "fri"].game_id.nunique()),
            "games_with_close(<=3h)": int(sn[sn.snap == "close"].game_id.nunique()),
            "median_close_minutes_before": float(gm.close_hours.median() * 60),
            "close_has_sharp": float((gm.close_n_sharp > 0).mean()),
            "median_books_per_snapshot": float(sn.n_books.median()),
            "allowed_books": sorted(t[t.season == s].book.unique()),
        }
    gm = games.copy()
    q["close_fair_vs_nflverse_total_line"] = {
        "mean_abs_diff_mu_close_vs_total_line": float((gm.mu_close - gm.total_line).abs().mean()),
        "share_median_close_point_eq_total_line": float((gm.pt_close == gm.total_line).mean()),
        "mean(mu_close - total_line)": float((gm.mu_close - gm.total_line).mean()),
    }
    q["distribution"] = {"sigma": dist.sigma, "n_train_2012_2019": dist.n_train,
                         "key_weights_top": {int(k): round(float(dist.w[k]), 2) for k in (37, 41, 43, 44, 47, 51, 40, 33, 30, 34)}}
    # calibration of the fair distribution on these seasons (closing, sharp-else-all)
    pov, _ = dist.probs(gm.mu_close.values, gm.pt_close.values, np.array(["over"] * len(gm)))
    pun, ppu = dist.probs(gm.mu_close.values, gm.pt_close.values, np.array(["under"] * len(gm)))
    q["calibration_at_close"] = {"n": len(gm), "pred_over": float(np.nanmean(pov)), "act_over": float((gm.total > gm.pt_close).mean()),
                                 "pred_push": float(np.nanmean(ppu)), "act_push": float((gm.total == gm.pt_close).mean()),
                                 "mean(total - mu_close)": float((gm.total - gm.mu_close).mean()),
                                 "se": float((gm.total - gm.mu_close).std() / math.sqrt(len(gm)))}
    if movement:
        mv = {}
        for a, b in (("tue", "close"), ("fri", "close"), ("tue", "fri")):
            ca = f"mu_{a}" if a != "close" else "mu_close"
            cb = f"mu_{b}" if b != "close" else "mu_close"
            pa = f"pt_{a}"
            pb = "pt_close" if b == "close" else f"pt_{b}"
            h = gm[gm[ca].notna() & gm[cb].notna()]
            d = h[cb] - h[ca]
            dp = h[pb] - h[pa]
            mv[f"{a}->{b}"] = {"n": len(h), "fair_mean_move": float(d.mean()), "fair_mean_abs_move": float(d.abs().mean()),
                               "fair_sd_move": float(d.std()),
                               "point_changed_share": float((dp != 0).mean()), "point_moved_ge1": float((dp.abs() >= 1).mean()),
                               "point_moved_ge2": float((dp.abs() >= 2).mean()),
                               "share_up_(fair>+0.25)": float((d > 0.25).mean()), "share_down_(fair<-0.25)": float((d < -0.25).mean()),
                               "corr(move, total - start)": float(np.corrcoef(d, h.total - h[ca])[0, 1])}
        q["movement"] = mv
    return q


# ================================================================== dev
def dev_analysis(t, snaps, P, g, dist) -> dict:
    D = {}
    games = snaps.drop_duplicates("game_id")
    # ---- (d) timing: blind over/under at each snapshot, best allowed price (one bet per game)
    tm = {}
    for sn in ("tue", "fri", "close"):
        for side in ("over", "under"):
            tm[f"{side}@{sn}"] = rstats(apply_rule(t, {"kind": "blind", "snaps": [sn], "side": side}))
    D["d_timing_blind"] = tm
    # line movement by total level / primetime (dev only)
    gm = games[games.mu_tue.notna()].copy()
    gm["move"] = gm.mu_close - gm.mu_tue
    gm["lvl"] = pd.cut(gm.pt_tue, [0, 41, 44, 47, 50, 99]).astype(str)
    D["d_move_by_level"] = gm.groupby("lvl").move.agg(["count", "mean", "std"]).round(3).to_dict("index")
    gm["prime"] = gm.weekday.isin(["Thursday", "Monday"]) | (gm.gametime >= "19:00")
    D["d_move_by_primetime"] = gm.groupby("prime").move.agg(["count", "mean", "std"]).round(3).to_dict("index")
    # ---- (a) soft vs sharp: EV of every allowed-book quote under the sharp fair at that snapshot
    x = t[t.mu_sharp.notna()].copy()
    x["evb"] = pd.cut(x.ev_sharp_now, [-1, -0.02, 0, 0.01, 0.02, 0.03, 1]).astype(str)
    a = {}
    for sn in ("tue", "fri", "close"):
        y = x[x.snap == sn]
        a[sn] = y.groupby("evb").agg(n=("tot_clv", "size"), clv=("tot_clv", "mean"), clv_all=("clv_all", "mean"),
                                     roi=("tot_pnl", "mean")).round(4).to_dict("index")
    D["a_quote_level_by_ev_bucket(not one-per-game)"] = a
    ar = {}
    for sn in (["tue"], ["fri"], ["close"], ["tue", "fri"], ["tue", "fri", "close"]):
        for th in (0.0, 0.01, 0.02, 0.03):
            ar[f"{'+'.join(sn)}_ev>{th}"] = rstats(apply_rule(t, {"kind": "soft_vs_sharp", "snaps": sn, "min_ev": th}))
    D["a_rules"] = ar
    # ---- (b) wind
    w = {}
    hist = g[g.season.between(2012, 2019) & g.total.notna() & ~g.indoor & g.wind_o.notna()].copy()
    hist["wb"] = pd.cut(hist.wind_o, [-1, 5, 10, 15, 20, 60]).astype(str)
    hist["res"] = hist.total - hist.total_line
    w["2012_2019_total_minus_closing_line_by_wind"] = hist.groupby("wb").res.agg(["count", "mean", "std"]).round(3).to_dict("index")
    gw = games[~games.indoor & games.wind_o.notna()].copy()
    gw["wb"] = pd.cut(gw.wind_o, [-1, 5, 10, 15, 20, 60]).astype(str)
    gw["tue_to_close"] = gw.mu_close - gw.mu_tue
    gw["res_close"] = gw.total - gw.mu_close
    gw["res_tue"] = gw.total - gw.mu_tue
    w["dev_by_wind"] = gw.groupby("wb")[["tue_to_close", "res_close", "res_tue"]].agg(["count", "mean"]).round(3) \
        .pipe(lambda d: {k: {f"{a}_{b}": v for (a, b), v in r.items()} for k, r in d.to_dict("index").items()})
    w["dev_outdoor_games_missing_wind"] = int((~games.indoor & games.wind_o.isna()).sum())
    wr = {}
    for sn in (["tue"], ["fri"], ["close"]):
        for wm in (12, 15, 20):
            for side in ("under", "over"):
                wr[f"{side}@{sn[0]}_wind>={wm}"] = rstats(apply_rule(t, {"kind": "wind", "snaps": sn, "wind_min": wm, "side": side}))
    w["rules"] = wr
    D["b_wind"] = w
    # ---- (c) model
    D["c_model_eval_nflverse_close"] = model_eval(P[P.season.between(2015, 2022)], range(2015, 2023))
    gmm = games[games.mu_tue.notna() & games.pred_base.notna()].copy()
    cm = {}
    for v in ("base", "wx"):
        e = gmm[f"pred_{v}"] - gmm.mu_tue
        cm[v] = {"n": len(gmm), "corr(pred-mu_tue, mu_close-mu_tue)": float(np.corrcoef(e, gmm.mu_close - gmm.mu_tue)[0, 1]),
                 "corr(pred-mu_tue, total-mu_tue)": float(np.corrcoef(e, gmm.total - gmm.mu_tue)[0, 1]),
                 "corr(pred-mu_close, total-mu_close)": float(np.corrcoef(gmm[f"pred_{v}"] - gmm.mu_close, gmm.total - gmm.mu_close)[0, 1]),
                 "sd(pred-mu_tue)": float(e.std())}
    D["c_model_vs_odds_dev"] = cm
    mr = {}
    for v in ("base", "wx"):
        for sn in (["tue"], ["fri"], ["close"], ["tue", "fri"]):
            for th in (0.0, 0.02, 0.04):
                mr[f"{v}@{'+'.join(sn)}_ev>{th}"] = rstats(apply_rule(t, {"kind": "model", "variant": v, "snaps": sn, "min_ev": th}))
    D["c_model_rules"] = mr
    # ---- (c2) does the model's disagreement with the early line predict the MOVE to the close?
    mv = {}
    for sn, col in (("tue", "mu_tue"), ("fri", "mu_fri")):
        for v in ("base", "wx"):
            h = games[games[col].notna() & games[f"pred_{v}"].notna()]
            e = (h[f"pred_{v}"] - h[col]).values
            mv[f"{v}@{sn}"] = {"all": _ols((h.mu_close - h[col]).values, e),
                               "result_beyond_close": _ols((h.total - h.mu_close).values, e),
                               **{int(s): _ols((d.mu_close - d[col]).values, (d[f"pred_{v}"] - d[col]).values)
                                  for s, d in h.groupby("season")}}
    D["c2_move_regressions"] = mv
    beta = {"tue": round(mv["base@tue"]["all"]["slope"], 3), "fri": round(mv["base@fri"]["all"]["slope"], 3)}
    D["c2_beta_fit_on_dev"] = beta
    m2 = {}
    for sn in (["tue"], ["fri"], ["tue", "fri"]):
        for th in (0.0, 0.01, 0.02):
            m2[f"move_base@{'+'.join(sn)}_ev>{th}"] = rstats(apply_rule(
                t, {"kind": "move", "variant": "base", "beta": beta, "snaps": sn, "min_ev": th}))
    D["c2_move_rules(in-sample beta)"] = m2
    n_cfg = len(tm) + len(ar) + len(wr) + len(mr) + len(m2)
    D["n_rule_configs_tried"] = n_cfg
    return D


# ================================================================== stages
DIST: TotalDist | None = None


def _ols(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return {"const": float(b[0]), "slope": float(b[1]), "t_slope": float(b[1] / se[1]), "n": len(y)}


def _prep(seasons, max_season):
    global DIST
    g = load_games()
    dist = DIST = TotalDist(g)
    P = walk_forward_preds(g, max_season)
    t, snaps, qa = build(seasons, dist, g, P)
    return g, dist, P, t, snaps, qa


def stage_dev():
    g, dist, P, t, snaps, qa = _prep(DEV, max(DEV))
    t.to_parquet(SCR / "bets_dev.parquet")
    snaps.to_parquet(SCR / "snaps_dev.parquet")
    # coverage-only QA on ALL seasons (no prices/results looked at for 2023+)
    cov = {}
    for s in DEV + HOLD:
        o = load_totals([s])
        cov[s] = {"rows": len(o), "events": int(o.event_id.nunique()), "snapshots": int(o.requested_ts.nunique()),
                  "books": sorted(o.book.unique()), "dropped_bad_rows": o.attrs["dropped"]}
    qa["coverage_all_seasons"] = cov
    qa.update(qa_block(t, snaps, g, DEV, dist))
    D = dev_analysis(t, snaps, P, g, dist)
    res = json.loads(JSON.read_text()) if JSON.exists() else {}
    res["qa"], res["dev"] = _jsonable(qa), _jsonable(D)
    res["generated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(_jsonable({"qa": qa, "dev": D}), indent=1)[:200000])


def stage_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; frozen rules are never overwritten")
    rules = json.loads((SCR / "rules_to_freeze.json").read_text())
    assert 1 <= len(rules["rules"]) <= 3
    rules["frozen_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    FROZEN.write_text(json.dumps(rules, indent=1))
    print(FROZEN.read_text())


def stage_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout locked: freeze first and set EDGE_HOLDOUT")
    res = json.loads(JSON.read_text())
    if "holdout" in res:
        raise SystemExit("holdout already run once; results are in totals.json")
    fz = json.loads(FROZEN.read_text())
    g, dist, P, t, snaps, qa = _prep(HOLD, max(HOLD))
    H = {"qa": qa_block(t, snaps, g, HOLD, dist), "rules": {}}
    for r in fz["rules"]:
        b = apply_rule(t, r)
        s = rstats(b)
        s["passes_bar"] = bool(s.get("bets", 0) > 2 and s["clv"] > 0 and s["clv_p_one_sided"] < 0.05 / 3)
        H["rules"][r["name"]] = s
    # context (NOT tested): the same summaries the dev stage looked at, for the record
    H["context_blind"] = {f"{side}@{sn}": rstats(apply_rule(t, {"kind": "blind", "snaps": [sn], "side": side}))
                          for sn in ("tue", "close") for side in ("over", "under")}
    H["context_model_eval_nflverse_close"] = model_eval(P[P.season.isin(HOLD)], HOLD)
    games = snaps.drop_duplicates("game_id")
    gw = games[~games.indoor & games.wind_o.notna()].copy()
    gw["wb"] = pd.cut(gw.wind_o, [-1, 5, 10, 15, 20, 60]).astype(str)
    H["context_wind"] = gw.assign(tue_to_close=gw.mu_close - gw.mu_tue, res_close=gw.total - gw.mu_close) \
        .groupby("wb")[["tue_to_close", "res_close"]].agg(["count", "mean"]).round(3) \
        .pipe(lambda d: {k: {f"{a}_{b}": v for (a, b), v in r.items()} for k, r in d.to_dict("index").items()})
    t.to_parquet(SCR / "bets_holdout.parquet")
    res["holdout"] = _jsonable(H)
    res["holdout_run_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["holdout"], indent=1))


def stage_posthoc():
    """Diagnostics AFTER the one-shot holdout (labelled post-hoc; they change no verdict)."""
    global DIST
    res = json.loads(JSON.read_text())
    if "holdout" not in res:
        raise SystemExit("run the holdout first")
    g = load_games()
    DIST = TotalDist(g)
    t = pd.read_parquet(SCR / "bets_holdout.parquet")
    games = t.drop_duplicates("game_id")
    X = {}
    for sn, col in (("tue", "mu_tue"), ("fri", "mu_fri")):
        h = games[games[col].notna() & games.pred_base.notna()]
        e = (h.pred_base - h[col]).values
        X[f"move_regression_base@{sn}"] = {"all": _ols((h.mu_close - h[col]).values, e),
                                           **{int(k): _ols((d.mu_close - d[col]).values, (d.pred_base - d[col]).values)
                                              for k, d in h.groupby("season")}}
    X["wind_under_sensitivity"] = {f"under@{sn}_wind>={wm}": rstats(apply_rule(
        t, {"kind": "wind", "snaps": [sn], "wind_min": wm, "side": "under"}))
        for sn in ("tue", "fri", "close") for wm in (12, 15, 20)}
    X["soft_vs_sharp_sensitivity"] = {f"{'+'.join(sn)}_ev>{th}": rstats(apply_rule(
        t, {"kind": "soft_vs_sharp", "snaps": sn, "min_ev": th})) for sn in (["tue"], ["fri"], ["tue", "fri"])
        for th in (0.0, 0.01, 0.02)}
    res["holdout_posthoc"] = _jsonable(X)
    JSON.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["holdout_posthoc"], indent=1))


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": stage_dev, "freeze": stage_freeze, "holdout": stage_holdout, "posthoc": stage_posthoc}[stage]()
