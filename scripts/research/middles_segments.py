"""Middles, arbitrage, line-shopping value and market-segment efficiency on 2020-2025 multi-book NFL lines.

Stages (run in order; outputs go to output/research/):
  python scripts/research/middles_segments.py build     # caches cleaned odds + fair lines (scratchpad)
  python scripts/research/middles_segments.py dev       # Q1-Q3 descriptive (all seasons, no tuning) + Q4 dev 2020-22
  python scripts/research/middles_segments.py freeze    # writes middles_segments_frozen.json from FROZEN_RULES (no overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/middles_segments.py holdout   # Q4 2023-25, ONCE
  python scripts/research/middles_segments.py report    # writes middles_segments.md from the json

Definitions
  * Sides: data/historical_odds/nfl_odds_*.csv.gz (daily 14:10 UTC, Fri 21:40 UTC, ~75 min pre-kick; regions us+us2).
    Totals: data/historical_odds/totals/ (Tue 14:10, Fri 21:40, pre-kick; region us => espnbet and hardrockbet never
    appear; fanatics only in 2025).
  * Cleaning: spread/total prices in [-250, +200] with |price| >= 100, a book's own two sides must have overround in
    [1.00, 1.12] (an overround < 1 inside ONE book is a feed error, not an arb), points within 4 (spreads) / 5 (totals)
    of the snapshot median. Moneylines: |ml| >= 100, within +-5000, own overround in [1.00, 1.15].
  * Margin distribution: nflpred.margins key-number pmf, sigma and weights from spread_rules.json. Fair home margin of a
    book = location mu at which its no-vig spread price is fair; snapshot fair = median over SHARP books present
    (lowvig, betonlineag, circasports, bookmaker) else over all books ("ref"); "all" = median over all books.
    Totals: the key-number total distribution of scripts/research/totals.py (fit 2012-2019), same construction.
  * EV of a bet (per unit staked) = P(win) * decimal + P(push) - 1 under the fair distribution AT THAT SNAPSHOT.
    CLV = the same under the fair distribution at the game's last pre-kick snapshot (<= 3 h before kickoff).
  * Bettable = my_books.json allowed books only. Consensus/sharp references use every book.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]

import replay_early_lines as R  # noqa: E402
from edge_lab import SHARP  # noqa: E402
from nflpred import features as F, margins as K, odds as O, spread_bets as SB  # noqa: E402
import totals as TR  # noqa: E402  (scripts/research/totals.py: TotalDist, load_totals, load_games)

OUT = ROOT / "output" / "research"
JSON = OUT / "middles_segments.json"
FROZEN = OUT / "middles_segments_frozen.json"
MD = OUT / "middles_segments.md"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/middles")
SEASONS = tuple(range(2020, 2026))
DEV, HOLD = (2020, 2021, 2022), (2023, 2024, 2025)
ALLOWED = O.load_allowed_books() or set()
UNIT_USD = 100
GAME_WEEK_H = 156   # <= 6.5 days before kickoff: after the previous week's Sunday games (excludes look-ahead lines)

# ---------------------------------------------------------------- small helpers
def dec(p):
    p = np.asarray(p, float)
    return np.where(p > 0, 1 + p / 100, 1 + 100 / -p)


def imp(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p < 0, -p / (-p + 100), 100 / (p + 100))


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def summ(x, pct=True):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    t = tstat(x)
    return {"n": int(len(x)), "mean": float(x.mean()), "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
            "t": t, "p_one_sided": pval(t) if np.isfinite(t) else None}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    return x


# ---------------------------------------------------------------- margin distribution (vectorised)
RULES = SB.load_rules()
SIG, WTS = RULES["margin"]["sigma"], RULES["_weights"]
MU = np.round(np.arange(-32.0, 32.0001, 0.05), 2)
PM = K.pmf(MU, SIG, WTS)                      # (n_mu, 121) over KS = -60..60
CDF = np.cumsum(PM, axis=1)


def set_pmf(mult: dict | None = None):
    """Rebuild the margin pmf with key-number weights multiplied by `mult` (sensitivity checks only)."""
    global PM, CDF
    w = dict(WTS)
    for k, f in (mult or {}).items():
        w[k] = w.get(k, 1.0) * f
    PM = K.pmf(MU, SIG, w)
    CDF = np.cumsum(PM, axis=1)


def _mi(mu):
    mu = np.nan_to_num(np.asarray(mu, float), nan=0.0)
    return np.clip(np.round((mu - MU[0]) / 0.05), 0, len(MU) - 1).astype(int)


def _cdf(mi, k):
    """P(margin <= k)."""
    k = np.asarray(k, int)
    j = np.clip(k + K.KMAX, -1, 2 * K.KMAX)
    out = CDF[mi, np.clip(j, 0, None)]
    return np.where(j < 0, 0.0, out)


def cover(mu, L):
    """Home line L (home covers if margin + L > 0): (P home covers, P push, P away covers)."""
    mi = _mi(mu)
    t = -np.asarray(L, float)
    hc = 1 - _cdf(mi, np.floor(t).astype(int))
    ac = _cdf(mi, (np.ceil(t) - 1).astype(int))
    return hc, np.clip(1 - hc - ac, 0, 1), ac


def sp_ev(mu, home_line, price, side):
    """EV per unit of a spread bet. home_line = the HOME line of the bet (for an away bet at +a, home_line = -a)."""
    hc, pu, ac = cover(mu, home_line)
    w = np.where(np.asarray(side) == "home", hc, ac)
    out = w * dec(price) + pu - 1
    return np.where(np.isfinite(np.asarray(mu, float)), out, np.nan)


def implied_mu_sp(home_line, q_home_nv):
    L = np.asarray(home_line, float)
    q = np.asarray(q_home_nv, float)
    out = np.full(len(L), np.nan)
    for v in np.unique(L[np.isfinite(L)]):
        m = L == v
        hc, pu, ac = cover(MU, np.full(len(MU), v))
        qq = hc / np.maximum(hc + ac, 1e-12)
        out[m] = np.interp(q[m], qq, MU)
    return out


def p_between(mu, lo, hi):
    """P(lo < margin < hi) for the margin distribution (lo, hi real)."""
    mi = _mi(mu)
    return np.clip(_cdf(mi, (np.ceil(hi) - 1).astype(int)) - _cdf(mi, np.floor(lo).astype(int)), 0, 1)


# ---------------------------------------------------------------- data
def games() -> pd.DataFrame:
    g = TR.load_games()
    g = g[g.season.isin(SEASONS)].copy()
    g["margin"] = g.home_score - g.away_score
    return g


def load_sides(path_glob="nfl_odds_{}.csv.gz", sub="", seasons=SEASONS) -> pd.DataFrame:
    fr = []
    for s in seasons:
        f = ROOT / "data" / "historical_odds" / sub / path_glob.format(s)
        if f.exists():
            fr.append(pd.read_csv(f).assign(season=s))
    o = pd.concat(fr, ignore_index=True)
    for c in ("requested_ts", "snapshot_ts", "last_update"):
        o[c] = pd.to_datetime(o[c], utc=True, errors="coerce")
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    o["home"], o["away"] = F._norm_team(o.home), F._norm_team(o.away)
    return o


def clean_sides(o: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(o)
    sp_ok = (o.sp_home_price.between(-250, 200) & o.sp_away_price.between(-250, 200)
             & (o.sp_home_price.abs() >= 100) & (o.sp_away_price.abs() >= 100)
             & o.sp_home_point.notna() & o.sp_away_point.notna())
    ovr = imp(o.sp_home_price) + imp(o.sp_away_price)
    sp_ok &= (ovr >= 1.0) & (ovr <= 1.12)
    med = o[sp_ok].groupby(["event_id", "requested_ts"]).sp_home_point.median().rename("med_pt")
    o = o.join(med, on=["event_id", "requested_ts"])
    sp_ok &= (o.sp_home_point - o.med_pt).abs() <= 4
    ml_ok = (o.ml_home.abs() >= 100) & (o.ml_away.abs() >= 100) & o.ml_home.between(-5000, 5000) & o.ml_away.between(-5000, 5000)
    mov = imp(o.ml_home) + imp(o.ml_away)
    ml_ok &= (mov >= 1.0) & (mov <= 1.15)
    for c in ("sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"):
        o.loc[~sp_ok, c] = np.nan
    for c in ("ml_home", "ml_away"):
        o.loc[~ml_ok, c] = np.nan
    qa = {"rows": n0, "spread_rows_dropped": int((~sp_ok).sum()), "ml_rows_dropped": int((~ml_ok).sum())}
    o = o[sp_ok | ml_ok].copy()
    return o, qa


def build():
    SCR.mkdir(parents=True, exist_ok=True)
    g = games()
    # ---------------- sides
    o = load_sides()
    o, qa = clean_sides(o)
    o = R.match_games(o, g).merge(g[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick].copy()
    o["hours_before"] = (o.kick - o.requested_ts).dt.total_seconds() / 3600
    o["age_min"] = (o.snapshot_ts - o.last_update).dt.total_seconds() / 60
    o["allowed"] = o.book.isin(ALLOWED)
    o["sharp"] = o.book.isin(SHARP)
    sp = o.sp_home_price.notna()
    nvh = imp(o.sp_home_price) / (imp(o.sp_home_price) + imp(o.sp_away_price))
    o["mu_b"] = np.nan
    o.loc[sp, "mu_b"] = implied_mu_sp(o.loc[sp, "sp_home_point"].values, nvh[sp.values])
    o["p_b"] = imp(o.ml_home) / (imp(o.ml_home) + imp(o.ml_away))
    key = ["game_id", "requested_ts"]
    # a book whose own no-vig ML prob is > 0.10 from the snapshot median (or spread fair > 4 pts away) is a feed
    # error (e.g. swapped teams), not a stale-but-real price: drop that market for that row
    pm = o.groupby(key).p_b.transform("median")
    bad_ml = (o.p_b - pm).abs() > 0.10
    mm = o.groupby(key).mu_b.transform("median")
    bad_sp = (o.mu_b - mm).abs() > 4
    qa["ml_rows_dropped_off_consensus"] = int(bad_ml.sum())
    qa["spread_rows_dropped_off_consensus"] = int(bad_sp.sum())
    o.loc[bad_ml, ["ml_home", "ml_away", "p_b"]] = np.nan
    o.loc[bad_sp, ["sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price", "mu_b"]] = np.nan
    snaps = o.groupby(key).agg(mu_all=("mu_b", "median"), p_all=("p_b", "median"), n_books=("book", "nunique"),
                               hours_before=("hours_before", "first"), season=("season", "first")).reset_index()
    sh = o[o.sharp].groupby(key).agg(mu_sharp=("mu_b", "median"), p_sharp=("p_b", "median"),
                                     n_sharp=("book", "nunique")).reset_index()
    na = o[o.allowed].groupby(key).agg(n_allowed=("book", "nunique")).reset_index()
    snaps = snaps.merge(sh, on=key, how="left").merge(na, on=key, how="left")
    snaps["n_sharp"] = snaps.n_sharp.fillna(0).astype(int)
    snaps["n_allowed"] = snaps.n_allowed.fillna(0).astype(int)
    snaps["mu_ref"] = snaps.mu_sharp.fillna(snaps.mu_all)
    snaps["p_ref"] = snaps.p_sharp.fillna(snaps.p_all)
    last = snaps.requested_ts == snaps.groupby("game_id").requested_ts.transform("max")
    close = snaps[last & (snaps.hours_before <= 3)][["game_id", "mu_ref", "mu_all", "p_ref", "p_all", "requested_ts"]] \
        .rename(columns={"mu_ref": "mu_close", "mu_all": "mu_close_all", "p_ref": "p_close", "p_all": "p_close_all",
                         "requested_ts": "close_ts"})
    snaps = snaps.merge(close, on="game_id", how="left")
    o[o.allowed | o.sharp].to_parquet(SCR / "sides_rows.parquet")
    snaps.to_parquet(SCR / "sides_snaps.parquet")
    # ---------------- totals
    dist = TR.TotalDist(TR.load_games())
    t = TR.load_totals(SEASONS)
    qa["totals_rows_dropped"] = int(t.attrs.get("dropped", 0))
    t["snapshot_ts"] = pd.to_datetime(t.snapshot_ts, utc=True, errors="coerce")
    t["last_update"] = pd.to_datetime(t.last_update, utc=True, errors="coerce")
    t = R.match_games(t, g).merge(g[["game_id", "kick"]], on="game_id")
    t = t[t.requested_ts < t.kick].copy()
    t["hours_before"] = (t.kick - t.requested_ts).dt.total_seconds() / 3600
    t["age_min"] = (t.snapshot_ts - t.last_update).dt.total_seconds() / 60
    t["allowed"] = t.book.isin(ALLOWED)
    t["sharp"] = t.book.isin(SHARP)
    nv = imp(t.tot_over_price) / t.overround
    t["mu_b"] = dist.implied_mu(t.tot_point.values, nv.values)
    ts = t.groupby(key).agg(tmu_all=("mu_b", "median"), n_books=("book", "nunique"),
                            hours_before=("hours_before", "first"), season=("season", "first")).reset_index()
    tsh = t[t.sharp].groupby(key).agg(tmu_sharp=("mu_b", "median"), n_sharp=("book", "nunique")).reset_index()
    tna = t[t.allowed].groupby(key).agg(n_allowed=("book", "nunique")).reset_index()
    ts = ts.merge(tsh, on=key, how="left").merge(tna, on=key, how="left")
    ts["n_sharp"] = ts.n_sharp.fillna(0).astype(int)
    ts["n_allowed"] = ts.n_allowed.fillna(0).astype(int)
    ts["tmu_ref"] = ts.tmu_sharp.fillna(ts.tmu_all)
    last = ts.requested_ts == ts.groupby("game_id").requested_ts.transform("max")
    tclose = ts[last & (ts.hours_before <= 3)][["game_id", "tmu_ref", "tmu_all"]].rename(
        columns={"tmu_ref": "tmu_close", "tmu_all": "tmu_close_all"})
    ts = ts.merge(tclose, on="game_id", how="left")
    t[t.allowed | t.sharp].to_parquet(SCR / "totals_rows.parquet")
    ts.to_parquet(SCR / "totals_snaps.parquet")
    g.to_parquet(SCR / "games.parquet")
    (SCR / "qa.json").write_text(json.dumps(qa))
    print("built", qa)


def load_cached():
    if not (SCR / "games.parquet").exists():
        build()
    return (pd.read_parquet(SCR / "games.parquet"), pd.read_parquet(SCR / "sides_rows.parquet"),
            pd.read_parquet(SCR / "sides_snaps.parquet"), pd.read_parquet(SCR / "totals_rows.parquet"),
            pd.read_parquet(SCR / "totals_snaps.parquet"), json.loads((SCR / "qa.json").read_text()))


_DIST = None


def tdist():
    global _DIST
    if _DIST is None:
        _DIST = TR.TotalDist(TR.load_games())
    return _DIST


def tot_ev(mu, pt, price, side):
    return tdist().ev(mu, pt, price, side)


# ---------------------------------------------------------------- leg tables
def side_legs(rows: pd.DataFrame, snaps: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """One row per (allowed book, snapshot, side) spread leg with EV now / at close and result."""
    a = rows[rows.allowed & rows.sp_home_point.notna()].merge(
        snaps[["game_id", "requested_ts", "mu_ref", "mu_all", "mu_close", "mu_close_all", "n_sharp"]],
        on=["game_id", "requested_ts"]).merge(g[["game_id", "season", "week", "margin"]].rename(columns={"season": "s2"}),
                                              on="game_id")
    legs = []
    for side in ("home", "away"):
        x = pd.DataFrame({"game_id": a.game_id, "season": a.season, "week": a.week, "requested_ts": a.requested_ts,
                          "hours_before": a.hours_before, "book": a.book, "side": side, "age_min": a.age_min,
                          "point": a[f"sp_{side}_point"], "price": a[f"sp_{side}_price"], "margin": a.margin, "mu_b": a.mu_b,
                          "mu_ref": a.mu_ref, "mu_all": a.mu_all, "mu_close": a.mu_close, "mu_close_all": a.mu_close_all})
        legs.append(x)
    L = pd.concat(legs, ignore_index=True)
    L["home_line"] = np.where(L.side == "home", L.point, -L.point)
    for c, m in (("ev", "mu_ref"), ("ev_all", "mu_all"), ("clv", "mu_close"), ("clv_all", "mu_close_all")):
        L[c] = sp_ev(L[m].values, L.home_line.values, L.price.values, L.side.values)
    adj = np.where(L.side == "home", L.margin + L.point, L.point - L.margin)
    L["pnl"] = np.where(adj > 0, dec(L.price) - 1, np.where(adj < 0, -1.0, 0.0))
    L.loc[L.margin.isna(), "pnl"] = np.nan
    return L


def total_legs(trows: pd.DataFrame, tsn: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    a = trows[trows.allowed].merge(tsn[["game_id", "requested_ts", "tmu_ref", "tmu_all", "tmu_close", "tmu_close_all"]],
                                   on=["game_id", "requested_ts"]).merge(g[["game_id", "week", "total"]], on="game_id")
    legs = []
    for side in ("over", "under"):
        legs.append(pd.DataFrame({"game_id": a.game_id, "season": a.season, "week": a.week, "requested_ts": a.requested_ts,
                                  "hours_before": a.hours_before, "book": a.book, "side": side, "age_min": a.age_min,
                                  "point": a.tot_point, "price": a[f"tot_{side}_price"], "total": a.total, "mu_b": a.mu_b,
                                  "mu_ref": a.tmu_ref, "mu_all": a.tmu_all, "mu_close": a.tmu_close,
                                  "mu_close_all": a.tmu_close_all}))
    L = pd.concat(legs, ignore_index=True)
    for c, m in (("ev", "mu_ref"), ("ev_all", "mu_all"), ("clv", "mu_close"), ("clv_all", "mu_close_all")):
        L[c] = tot_ev(L[m].values, L.point.values, L.price.values, L.side.values)
    adj = np.where(L.side == "over", L.total - L.point, L.point - L.total)
    L["pnl"] = np.where(adj > 0, dec(L.price) - 1, np.where(adj < 0, -1.0, 0.0))
    L.loc[L.total.isna(), "pnl"] = np.nan
    return L


def pairs(L: pd.DataFrame, market: str, min_gap: float = 0.0) -> pd.DataFrame:
    """Cross every side-A leg with every side-B leg at the same snapshot. gap = width of the window in which both win
    (spreads: away point + home point; totals: under point - over point)."""
    sa, sb = ("home", "away") if market == "spread" else ("over", "under")
    cols = ["game_id", "season", "week", "requested_ts", "hours_before", "book", "point", "price", "age_min",
            "ev", "ev_all", "clv", "clv_all", "pnl", "mu_ref", "mu_close", "mu_b"]
    A = L[L.side == sa][cols]
    B = L[L.side == sb][["game_id", "requested_ts", "book", "point", "price", "age_min", "ev", "ev_all", "clv", "clv_all", "pnl", "mu_b"]]
    P = A.merge(B, on=["game_id", "requested_ts"], suffixes=("_a", "_b"))
    P["gap"] = (P.point_b + P.point_a) if market == "spread" else (P.point_b - P.point_a)
    P = P[P.gap >= min_gap].copy()
    for c in ("ev", "ev_all", "clv", "clv_all", "pnl"):
        P[f"pkg_{c}"] = P[f"{c}_a"] + P[f"{c}_b"]
    P["imp_sum"] = imp(P.price_a) + imp(P.price_b)
    P["same_book"] = P.book_a == P.book_b
    P["off_market"] = np.maximum((P.mu_b_a - P.mu_ref).abs(), (P.mu_b_b - P.mu_ref).abs())
    return P


def middle_hit_prob(P: pd.DataFrame, market: str, mu_col="mu_ref"):
    if market == "spread":
        # home line h = point_a; both win if -h < margin < a
        return p_between(P[mu_col].values, -P.point_a.values, P.point_b.values)
    d = tdist()
    mu = P[mu_col].values
    loc = np.interp(np.nan_to_num(mu, nan=45.0), d.mean_at, TR.MU_GRID)
    mi = np.clip(np.round((loc - TR.MU_GRID[0]) / 0.05), 0, len(TR.MU_GRID) - 1).astype(int)
    cdf = np.cumsum(d.pmf(TR.MU_GRID), axis=1)
    lo, hi = P.point_a.values, P.point_b.values
    c_hi = cdf[mi, np.clip(np.ceil(hi).astype(int) - 1, 0, 120)]
    c_lo = cdf[mi, np.clip(np.floor(lo).astype(int), 0, 120)]
    return np.clip(c_hi - c_lo, 0, 1)


def realized_hit(P, market, g):
    if market == "spread":
        m = P.game_id.map(g.set_index("game_id").margin)
        return ((m > -P.point_a) & (m < P.point_b)).astype(float).where(m.notna())
    t = P.game_id.map(g.set_index("game_id").total)
    return ((t > P.point_a) & (t < P.point_b)).astype(float).where(t.notna())


# ================================================================== Q1 middles (same snapshot)
def q1_middles(L: pd.DataFrame, market: str, g: pd.DataFrame) -> dict:
    P = pairs(L, market, 0.5)
    P = P[~P.same_book]
    P["p_mid"] = middle_hit_prob(P, market)
    P["hit"] = realized_hit(P, market, g)
    off_lim = 2.0 if market == "spread" else 2.5
    out = {}
    for lab, selP, selL in (
            ("all snapshots", P.hours_before > -1, L.hours_before > -1),
            ("game week (<= 6.5 days)", P.hours_before <= GAME_WEEK_H, L.hours_before <= GAME_WEEK_H),
            (f"game week, no leg > {off_lim} pts off fair", (P.hours_before <= GAME_WEEK_H) & (P.off_market <= off_lim),
             L.hours_before <= GAME_WEEK_H)):
        out[lab] = _middle_stats(P[selP], L[selL])
    return out


def _middle_stats(P: pd.DataFrame, L: pd.DataFrame) -> dict:
    nsnap = L.groupby(["game_id", "requested_ts"]).ngroups
    ngame = L.game_id.nunique()
    best = P.sort_values("pkg_ev", ascending=False).groupby(["game_id", "requested_ts"]).head(1)
    widest = P.sort_values(["gap", "pkg_ev"], ascending=False).groupby(["game_id", "requested_ts"]).head(1)
    true_mid = widest[widest.p_mid > 0]
    out = {"snapshots_with_allowed_lines": nsnap, "games": ngame,
           "share_snapshots_any_different_numbers": len(best) / nsnap,
           "share_snapshots_true_middle(both_can_win)": len(true_mid) / nsnap,
           "share_games_true_middle_some_snapshot": true_mid.game_id.nunique() / ngame,
           "widest_gap_distribution": widest.gap.clip(upper=4).value_counts().sort_index().to_dict()}
    weeks = L.groupby(["season", "week"]).game_id.nunique()
    gw = true_mid.groupby(["season", "week"]).game_id.nunique().reindex(weeks.index).fillna(0)
    out["games_with_true_middle_per_week"] = float(gw.mean())
    pos = best[best.pkg_ev > 0]
    gwp = pos.groupby(["season", "week"]).game_id.nunique().reindex(weeks.index).fillna(0)
    out["games_with_posEV_package_per_week"] = float(gwp.mean())
    out["games_with_posEV_package_per_season"] = {int(k): int(v) for k, v in pos.groupby("season").game_id.nunique().items()}
    out["true_middle_pkg_ev_quantiles_(units_per_2u_staked)"] = {str(k): v for k, v in
                                                                  true_mid.pkg_ev.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).items()}
    out["true_middle_mean_pkg_ev"] = float(true_mid.pkg_ev.mean()) if len(true_mid) else None
    out["true_middle_mean_p_middle"] = float(true_mid.p_mid.mean()) if len(true_mid) else None
    out["share_snapshots_posEV_package"] = len(pos) / nsnap
    out["posEV_both_legs_pos"] = float(((pos.ev_a > 0) & (pos.ev_b > 0)).mean()) if len(pos) else None
    out["posEV_one_leg_carries"] = float(((pos.ev_a > 0) ^ (pos.ev_b > 0)).mean()) if len(pos) else None
    out["posEV_mean_better_leg_ev"] = float(np.maximum(pos.ev_a, pos.ev_b).mean()) if len(pos) else None
    out["posEV_mean_pkg_ev"] = float(pos.pkg_ev.mean()) if len(pos) else None
    out["posEV_max_leg_age_min_median"] = float(np.maximum(pos.age_min_a, pos.age_min_b).median()) if len(pos) else None
    out["posEV_off_market_pts_median"] = float(pos.off_market.median()) if len(pos) else None
    bg = {}
    for gap, d in widest.groupby(widest.gap.clip(upper=3.0)):
        bg[str(gap)] = {"n": len(d), "mean_pkg_ev": float(d.pkg_ev.mean()), "mean_p_middle": float(d.p_mid.mean()),
                        "realized_middle_rate": float(d.hit.mean()), "share_pos": float((d.pkg_ev > 0).mean())}
    out["by_gap_widest_pkg"] = bg
    one = true_mid.sort_values("requested_ts").groupby("game_id").head(1)
    out["calibration_first_true_middle_per_game"] = {"n": len(one), "pred_hits": float(one.p_mid.sum()),
                                                     "realized_hits": int(one.hit.sum())}
    allm = true_mid
    out["calibration_all_true_middle_snapshots"] = {"n": len(allm), "pred_rate": float(allm.p_mid.mean()) if len(allm) else None,
                                                    "realized_rate": float(allm.hit.mean()) if len(allm) else None}
    pol = best[(best.pkg_ev > 0) & (best.pkg_ev_all > 0)].sort_values("requested_ts").groupby("game_id").head(1)
    n = len(pol)
    out["policy_first_posEV_package_per_game"] = {
        "games": n, "per_season": n / len(SEASONS),
        "share_true_middles": float((pol.p_mid > 0).mean()) if n else None,
        "mean_ev_units_per_pkg": float(pol.pkg_ev.mean()) if n else None,
        "mean_clv_units_per_pkg": float(pol.pkg_clv.mean()) if n else None,
        "realized_units_per_pkg": float(pol.pkg_pnl.mean()) if n else None,
        "realized_se": float(pol.pkg_pnl.std() / math.sqrt(n)) if n > 1 else None,
        "expected_units_per_season": float(pol.pkg_ev.sum() / len(SEASONS)) if n else 0.0,
        "clv_units_per_season": float(pol.pkg_clv.sum() / len(SEASONS)) if n else 0.0,
        "realized_units_per_season": float(pol.pkg_pnl.sum() / len(SEASONS)) if n else 0.0,
        "middle_hits": int(pol.hit.sum()) if n else 0, "pred_middle_hits": float(pol.p_mid.sum()) if n else 0.0,
        "by_season": {int(k): {"n": len(d), "ev": float(d.pkg_ev.sum()), "clv": float(d.pkg_clv.sum()),
                               "pnl": float(d.pkg_pnl.sum())} for k, d in pol.groupby("season")},
        "same_games_bet_only_better_leg": {
            "ev_units_per_game": float(np.maximum(pol.ev_a, pol.ev_b).mean()) if n else None,
            "clv_units_per_game": float(np.where(pol.ev_a >= pol.ev_b, pol.clv_a, pol.clv_b).mean()) if n else None,
            "pnl_units_per_game": float(np.where(pol.ev_a >= pol.ev_b, pol.pnl_a, pol.pnl_b).mean()) if n else None},
    }
    anym = true_mid.sort_values("requested_ts").groupby("game_id").head(1)
    out["policy_first_true_middle_any_EV_per_game"] = {
        "games": len(anym), "mean_ev": float(anym.pkg_ev.mean()) if len(anym) else None,
        "mean_clv": float(anym.pkg_clv.mean()) if len(anym) else None,
        "realized": float(anym.pkg_pnl.mean()) if len(anym) else None,
        "realized_se": float(anym.pkg_pnl.std() / math.sqrt(len(anym))) if len(anym) > 1 else None}
    return out


# ================================================================== Q1b sequential middles
def q1_sequential(L: pd.DataFrame, market: str, g: pd.DataFrame) -> dict:
    sa, sb = ("home", "away") if market == "spread" else ("over", "under")
    L = L.copy()
    # early bet: first snapshot of the game with >= 3 allowed books posting, <= 8 days out, best-EV allowed leg per side
    nb = L.groupby(["game_id", "requested_ts"]).book.transform("nunique")
    E = L[(nb >= 3) & (L.hours_before <= GAME_WEEK_H) & (L.hours_before > 3)]
    first_ts = E.groupby("game_id").requested_ts.min().rename("t1")
    E = E.merge(first_ts, left_on=["game_id", "requested_ts"], right_on=["game_id", "t1"])
    early = E.sort_values("ev", ascending=False).groupby(["game_id", "side"]).head(1)
    res = []
    for side, other in ((sa, sb), (sb, sa)):
        e = early[early.side == side][["game_id", "season", "t1", "book", "point", "price", "ev", "clv", "pnl"]]
        lt = L[L.side == other][["game_id", "requested_ts", "book", "point", "price", "ev", "clv", "pnl", "mu_ref",
                                  "hours_before", "age_min"]]
        m = e.merge(lt, on="game_id", suffixes=("_e", "_l"))
        m = m[m.requested_ts > m.t1]
        if market == "spread":
            m["gap"] = m.point_e + m.point_l
            # early leg's home line: if early side home -> point_e, else -point_e
            m["home_line_e"] = m.point_e if side == "home" else -m.point_e
            m["ev_e_now"] = sp_ev(m.mu_ref.values, m.home_line_e.values, m.price_e.values, np.full(len(m), side))
            lo, hi = (-m.point_e, m.point_l) if side == "home" else (-m.point_l, m.point_e)
            m["p_mid"] = p_between(m.mu_ref.values, lo.values, hi.values)
            m["key_in_window"] = [any(lo_ < k < hi_ for k in (3, -3, 7, -7)) for lo_, hi_ in zip(lo.values, hi.values)]
        else:
            m["gap"] = (m.point_l - m.point_e) if side == "over" else (m.point_e - m.point_l)
            m["ev_e_now"] = tot_ev(m.mu_ref.values, m.point_e.values, m.price_e.values, np.full(len(m), side))
            m["p_mid"] = np.nan
            m["key_in_window"] = False
        m["side_e"] = side
        m["pkg_ev_now"] = m.ev_e_now + m.ev_l
        res.append(m)
    M = pd.concat(res, ignore_index=True)
    nearly = len(early)
    out = {"early_bets": nearly, "early_bet_mean_ev": float(early.ev.mean()), "early_bet_mean_clv": float(early.clv.mean())}
    for lab, cond in (("gap>=0.5", M.gap >= 0.5), ("gap>=1", M.gap >= 1), ("gap>=1.5", M.gap >= 1.5),
                      ("window contains 3 or 7", M.key_in_window & (M.gap >= 0.5)),
                      ("middle & pkg EV>0", (M.gap >= 0.5) & (M.pkg_ev_now > 0)),
                      ("middle & hedge leg EV>0", (M.gap >= 0.5) & (M.ev_l > 0))):
        k = M[cond].groupby(["game_id", "side_e"]).size()
        out[f"share_early_bets_later_{lab}"] = len(k) / nearly
    # policy: hedge at the FIRST later snapshot offering a middle (gap>=1) with package EV>0 at that snapshot
    c = M[(M.gap >= 1) & (M.pkg_ev_now > 0)].sort_values("requested_ts").groupby(["game_id", "side_e"]).head(1)
    c = c.sort_values("ev_l", ascending=False).groupby(["game_id", "side_e"]).head(1) if len(c) else c
    out["hedge_policy"] = {
        "hedged_bets": len(c), "share_of_early_bets": len(c) / nearly,
        "early_leg_clv_mean_hedged_subset": float(c.clv_e.mean()) if len(c) else None,
        "hedge_leg_ev_now_mean": float(c.ev_l.mean()) if len(c) else None,
        "hedge_leg_clv_mean": float(c.clv_l.mean()) if len(c) else None,
        "hedge_leg_realized_mean": float(c.pnl_l.mean()) if len(c) else None,
        "hedge_leg_realized_se": float(c.pnl_l.std() / math.sqrt(len(c))) if len(c) > 1 else None,
        "package_realized_mean_units": float((c.pnl_e + c.pnl_l).mean()) if len(c) else None,
        "early_only_realized_mean_units_same_games": float(c.pnl_e.mean()) if len(c) else None,
        "pred_p_middle_mean": float(c.p_mid.mean()) if len(c) and market == "spread" else None,
        "gap_mean": float(c.gap.mean()) if len(c) else None,
        "per_season": len(c) / len(SEASONS)}
    # what if the hedge is only placed when the hedge leg itself is +EV (the only incremental-EV-positive choice)
    c2 = M[(M.gap >= 0.5) & (M.ev_l > 0)].sort_values("requested_ts").groupby(["game_id", "side_e"]).head(1)
    out["hedge_only_if_leg_posEV"] = {"n": len(c2), "per_season": len(c2) / len(SEASONS),
                                      "hedge_ev_mean": float(c2.ev_l.mean()) if len(c2) else None,
                                      "hedge_clv_mean": float(c2.clv_l.mean()) if len(c2) else None,
                                      "hedge_realized_mean": float(c2.pnl_l.mean()) if len(c2) else None}
    return out


# ================================================================== Q2 arbitrage
def q2_arbs(L: pd.DataFrame, market: str, rows: pd.DataFrame | None = None, snaps=None) -> dict:
    P = pairs(L, market, 0.0)
    P = P[~P.same_book & (P.imp_sum < 1)]
    P["arb_return"] = 1 / P.imp_sum - 1           # guaranteed return on total stake with stakes split by implied prob
    P["max_age_min"] = np.maximum(P.age_min_a, P.age_min_b)
    P["off_leg_ev"] = np.maximum(P.ev_a, P.ev_b)
    nsnap = L.groupby(["game_id", "requested_ts"]).ngroups
    out = {}
    for lab, d in (("same_number", P[P.gap == 0]), ("better_number_(middle+arb)", P[P.gap > 0])):
        b = d.sort_values("arb_return", ascending=False).groupby(["game_id", "requested_ts"]).head(1)
        out[lab] = {"snapshots": int(len(b)), "share_snapshots": len(b) / nsnap, "games": int(b.game_id.nunique()),
                    "per_season": float(b.game_id.nunique() / len(SEASONS)),
                    "arb_return_median": float(b.arb_return.median()) if len(b) else None,
                    "arb_return_p90": float(b.arb_return.quantile(0.9)) if len(b) else None,
                    "max_leg_age_min_median": float(b.max_age_min.median()) if len(b) else None,
                    "share_with_leg_older_than_30min": float((b.max_age_min > 30).mean()) if len(b) else None,
                    "off_market_leg_ev_median": float(b.off_leg_ev.median()) if len(b) else None,
                    "pkg_clv_mean_units": float(b.pkg_clv.mean()) if len(b) else None,
                    "book_pairs": (b.book_a + "/" + b.book_b).value_counts().head(5).to_dict() if len(b) else {},
                    "by_hours_before": b.hours_before.describe()[["min", "50%", "max"]].to_dict() if len(b) else {}}
    return out


def ml_arbs(rows: pd.DataFrame, snaps: pd.DataFrame, g: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a = rows[rows.allowed & rows.ml_home.notna()].merge(snaps[["game_id", "requested_ts", "p_ref", "p_close"]],
                                                       on=["game_id", "requested_ts"])
    a["dh"], a["da"] = dec(a.ml_home), dec(a.ml_away)
    key = ["game_id", "requested_ts"]
    bh = a.loc[a.groupby(key).dh.idxmax(), key + ["book", "dh", "age_min", "p_ref", "p_close", "season", "hours_before"]]
    ba = a.loc[a.groupby(key).da.idxmax(), key + ["book", "da", "age_min"]]
    m = bh.merge(ba, on=key, suffixes=("_h", "_a"))
    m["imp_sum"] = 1 / m.dh + 1 / m.da
    arb = m[(m.imp_sum < 1) & (m.book_h != m.book_a)].copy()
    arb["ret"] = 1 / arb.imp_sum - 1
    arb["max_age"] = np.maximum(arb.age_min_h, arb.age_min_a)
    arb["off_leg_ev"] = np.maximum(arb.dh * arb.p_ref - 1, arb.da * (1 - arb.p_ref) - 1)
    out = {"snapshots": int(len(arb)), "share_snapshots": len(arb) / len(m), "games": int(arb.game_id.nunique()),
           "per_season": arb.game_id.nunique() / len(SEASONS),
           "arb_return_median": float(arb.ret.median()) if len(arb) else None,
           "arb_return_p90": float(arb.ret.quantile(0.9)) if len(arb) else None,
           "max_leg_age_min_median": float(arb.max_age.median()) if len(arb) else None,
           "share_with_leg_older_than_30min": float((arb.max_age > 30).mean()) if len(arb) else None,
           "off_market_leg_ev_median": float(arb.off_leg_ev.median()) if len(arb) else None,
           "book_pairs": (arb.book_h + "/" + arb.book_a).value_counts().head(5).to_dict(),
           "by_season": arb.groupby("season").game_id.nunique().to_dict()}
    return out, m


def hourly_arbs() -> dict:
    """Arb persistence on the hourly 2025 sample (Sat 15:00 -> Sun 16:00 UTC)."""
    o = load_sides(sub="hourly", seasons=(2025,))
    o, _ = clean_sides(o)
    o["p_b"] = imp(o.ml_home) / (imp(o.ml_home) + imp(o.ml_away))
    pm = o.groupby(["event_id", "requested_ts"]).p_b.transform("median")
    o.loc[(o.p_b - pm).abs() > 0.10, ["ml_home", "ml_away"]] = np.nan
    o = o[o.book.isin(ALLOWED)]
    o["dh"], o["da"] = dec(o.ml_home), dec(o.ml_away)
    key = ["event_id", "requested_ts"]
    res = {}
    # ML
    x = o[o.ml_home.notna()]
    bh = x.loc[x.groupby(key).dh.idxmax(), key + ["book", "dh"]]
    ba = x.loc[x.groupby(key).da.idxmax(), key + ["book", "da"]]
    m = bh.merge(ba, on=key, suffixes=("_h", "_a"))
    m["arb"] = (1 / m.dh + 1 / m.da < 1) & (m.book_h != m.book_a)
    # spread same number
    s = o[o.sp_home_point.notna()].copy()
    s["dhs"], s["das"] = dec(s.sp_home_price), dec(s.sp_away_price)
    hh = s.groupby(key + ["sp_home_point"]).dhs.max().rename("bh").reset_index()
    aa = s.groupby(key + ["sp_away_point"]).das.max().rename("ba").reset_index()
    aa["sp_home_point"] = -aa.sp_away_point
    sm = hh.merge(aa, on=key + ["sp_home_point"])
    sm["arb"] = 1 / sm.bh + 1 / sm.ba < 1
    sarb = sm.groupby(key).arb.any().reset_index()
    for lab, d in (("moneyline", m[key + ["arb"]]), ("spread_same_number", sarb)):
        d = d.sort_values(key)
        d["next_arb"] = d.groupby("event_id").arb.shift(-1)
        nxt = d[d.arb & d.next_arb.notna()]
        res[lab] = {"event_hours": len(d), "share_with_arb": float(d.arb.mean()),
                    "events_with_any_arb": int(d[d.arb].event_id.nunique()), "events": int(d.event_id.nunique()),
                    "arb_still_there_1h_later": float(nxt.next_arb.astype(bool).mean()) if len(nxt) else None}
    return res


# ================================================================== Q3 line shopping
def best_of(L: pd.DataFrame, books: list[str], col: str) -> pd.Series:
    d = L[L.book.isin(books)]
    return d.groupby(["game_id", "requested_ts", "side"])[col].max()


def q3_shopping(L: pd.DataFrame, market: str, label_cols=("ev", "clv")) -> dict:
    out = {}
    for era, seasons in (("2020-2022", DEV), ("2023-2025", HOLD)):
        D = L[L.season.isin(seasons)]
        books = sorted(D.book.unique())
        res = {"books_present": books}
        for when, sel in (("pre-kick close", D.hours_before <= 3), ("early (>= 3 days out)", D.hours_before >= 72),
                          ("all snapshots", D.hours_before > -1)):
            X = D[sel]
            if X.empty:
                continue
            key = ["game_id", "requested_ts", "side"]
            r = {}
            for c in label_cols:
                single = {b: float(X[X.book == b][c].mean()) for b in books if (X.book == b).any()}
                cover_ = {b: float((X.book == b).groupby([X.game_id, X.requested_ts, X.side]).any().mean()) for b in books}
                best = X.groupby(key)[c].max()
                typical = X.groupby(key)[c].mean()
                # greedy account ordering on this metric; best over the chosen set, bets where >= 1 chosen book posts
                chosen, curve = [], []
                remaining = list(books)
                while remaining:
                    # metric is mean over bets the set can place; to compare fairly we fill missing with the
                    # typical-book value (a bettor without a posting book would bet elsewhere at a typical price)
                    scores = {b: best_of(X, chosen + [b], c).reindex(best.index).fillna(typical).mean() for b in remaining}
                    nb = max(scores, key=scores.get)
                    chosen.append(nb)
                    remaining.remove(nb)
                    curve.append({"k": len(chosen), "add": nb, "mean": float(scores[nb])})
                r[c] = {"single_book_mean": single, "book_coverage": cover_, "typical_book_mean": float(typical.mean()),
                        "best_all_allowed_mean": float(best.mean()), "greedy_curve": curve, "n_bets": int(len(best))}
            # consensus price: all books (allowed + sharp) median price at the modal point per bet
            out_key = when
            res[out_key] = r
        out[era] = res
    return out


def q3_ml(rows: pd.DataFrame, snaps: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    a = rows[rows.allowed & rows.ml_home.notna()].merge(snaps[["game_id", "requested_ts", "p_ref", "p_close"]],
                                                       on=["game_id", "requested_ts"]).merge(g[["game_id", "week", "margin"]], on="game_id")
    legs = []
    for side in ("home", "away"):
        p = a.p_ref if side == "home" else 1 - a.p_ref
        pc = a.p_close if side == "home" else 1 - a.p_close
        d = dec(a[f"ml_{side}"])
        win = (a.margin > 0) if side == "home" else (a.margin < 0)
        legs.append(pd.DataFrame({"game_id": a.game_id, "season": a.season, "week": a.week, "requested_ts": a.requested_ts,
                                  "hours_before": a.hours_before, "book": a.book, "side": side, "price": a[f"ml_{side}"],
                                  "ev": d * p - 1, "clv": d * pc - 1,
                                  "pnl": np.where(a.margin == 0, 0.0, np.where(win, d - 1, -1.0))}))
    return pd.concat(legs, ignore_index=True)


# ================================================================== Q4 segments
DIVS = {"AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
        "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LV", "LAC"],
        "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
        "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"]}
CONF = {t: d[:3] for d, ts in DIVS.items() for t in ts}
DOMESTIC_NEUTRAL = {"State Farm Stadium", "TIAA Bank Stadium", "Ford Field"}


def standings_flags(g: pd.DataFrame) -> pd.DataFrame:
    """Per REG game: eliminated / clinched flags for each team using only results with an EARLIER gameday.
    Sufficient conditions (no tiebreakers): eliminated if >= 7 other conference teams already have more points
    (win=1, tie=0.5) than this team's maximum; clinched a playoff berth if <= 6 other conference teams can still
    reach this team's current points."""
    reg = g[g.game_type == "REG"].copy()
    out = []
    for s, d in reg.groupby("season"):
        teams = sorted(set(d.home_team) | set(d.away_team))
        long = pd.concat([d[["game_id", "gameday", "home_team", "margin"]].rename(columns={"home_team": "team"}),
                          d[["game_id", "gameday", "away_team", "margin"]].rename(columns={"away_team": "team"})
                          .assign(margin=lambda x: -x.margin)])
        long["pts"] = np.where(long.margin > 0, 1.0, np.where(long.margin == 0, 0.5, 0.0))
        for day in sorted(d.gameday.unique()):
            before = long[long.gameday < day]
            cur = before.groupby("team").pts.sum().reindex(teams).fillna(0)
            remaining = long[long.gameday >= day].groupby("team").size().reindex(teams).fillna(0)
            mx = cur + remaining
            flags = {}
            for t in teams:
                others = [u for u in teams if u != t and CONF.get(u) == CONF.get(t)]
                elim = sum(cur[u] > mx[t] for u in others) >= 7
                clin = sum(mx[u] >= cur[t] for u in others) <= 6
                flags[t] = (elim, clin, cur[t], remaining[t])
            for r in d[d.gameday == day].itertuples():
                eh, ch, _, _ = flags[r.home_team]
                ea, ca, _, _ = flags[r.away_team]
                out.append({"game_id": r.game_id, "home_elim": eh, "away_elim": ea, "home_clinch": ch, "away_clinch": ca})
    return pd.DataFrame(out)


def game_segments(g: pd.DataFrame) -> pd.DataFrame:
    x = g.copy()
    x["wk12"] = (x.game_type == "REG") & (x.week <= 2)
    x["thu"] = x.weekday == "Thursday"
    x["mon"] = x.weekday == "Monday"
    x["sat"] = x.weekday == "Saturday"
    x["intl"] = (x.location == "Neutral") & (x.game_type == "REG") & ~x.stadium.isin(DOMESTIC_NEUTRAL)
    x["playoff"] = x.game_type != "REG"
    sf = standings_flags(g)
    x = x.merge(sf, on="game_id", how="left")
    for c in ("home_elim", "away_elim", "home_clinch", "away_clinch"):
        x[c] = x[c].fillna(False).astype(bool)
    x["late"] = (x.game_type == "REG") & (x.week >= 12)
    x["one_elim"] = x.late & (x.home_elim ^ x.away_elim)
    x["one_clinch"] = x.late & (x.home_clinch ^ x.away_clinch)
    # divisional rematch: second REG meeting of the season between the two teams; first-meeting loser = revenge side
    x = x.sort_values(["gameday", "game_id"])
    x["pair"] = [tuple(sorted((h, a))) + (s,) for h, a, s in zip(x.home_team, x.away_team, x.season)]
    first = {}
    rem, revenge_home = [], []
    for r in x.itertuples():
        if r.game_type == "REG" and r.div_game == 1 and r.pair in first:
            fm = first[r.pair]
            rem.append(True)
            # home team lost the first meeting?
            loser = fm["away"] if fm["margin"] > 0 else (fm["home"] if fm["margin"] < 0 else None)
            revenge_home.append(np.nan if loser is None else float(loser == r.home_team))
        else:
            rem.append(False)
            revenge_home.append(np.nan)
        if r.game_type == "REG" and r.pair not in first and pd.notna(r.margin):
            first[r.pair] = {"home": r.home_team, "away": r.away_team, "margin": r.margin}
    x["div_rematch"] = rem
    x["revenge_home"] = revenge_home
    return x.drop(columns=["pair"])


def q4_table(seasons, L: pd.DataFrame, TL: pd.DataFrame, ML: pd.DataFrame, snaps, tsn, g) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per game: first-snapshot bets (best allowed EV per side) on spread, ML and totals, with CLV at the close,
    plus segment flags and move sizes."""
    def first_bets(X, extra_sel=None):
        X = X[X.season.isin(seasons) & (X.hours_before <= GAME_WEEK_H) & (X.hours_before > 3) & X.clv.notna()]
        nb = X.groupby(["game_id", "requested_ts"]).book.transform("nunique")
        X = X[nb >= 3]
        t1 = X.groupby("game_id").requested_ts.transform("min")
        X = X[X.requested_ts == t1]
        return X.sort_values("ev", ascending=False).groupby(["game_id", "side"]).head(1)
    sp, ml, tt = first_bets(L), first_bets(ML), first_bets(TL)
    seg = game_segments(g)
    seg = seg[seg.season.isin(seasons)]
    # moves (fair, ref) first snapshot -> close
    s1 = snaps[snaps.season.isin(seasons) & (snaps.n_allowed >= 3) & (snaps.hours_before <= GAME_WEEK_H) & (snaps.hours_before > 3)]
    s1 = s1.sort_values("requested_ts").groupby("game_id").head(1)[["game_id", "mu_ref", "p_ref", "mu_close", "p_close", "hours_before"]]
    t1 = tsn[tsn.season.isin(seasons) & (tsn.n_allowed >= 3) & (tsn.hours_before <= GAME_WEEK_H) & (tsn.hours_before > 3)]
    t1 = t1.sort_values("requested_ts").groupby("game_id").head(1)[["game_id", "tmu_ref", "tmu_close"]]
    G = seg.merge(s1, on="game_id", how="inner").merge(t1, on="game_id", how="left")
    G["fav_home"] = G.mu_ref > 0
    G["fav_abs"] = G.mu_ref.abs()
    G["big_fav"] = G.fav_abs >= 10
    G["move_home_pts"] = G.mu_close - G.mu_ref
    G["move_fav_pts"] = np.where(G.fav_home, 1, -1) * G.move_home_pts
    G["move_over_pts"] = G.tmu_close - G.tmu_ref
    G["move_home_prob"] = G.p_close - G.p_ref
    # bets long
    B = []
    for mkt, X in (("spread", sp), ("ml", ml), ("total", tt)):
        B.append(X[["game_id", "side", "book", "point" if "point" in X else "price", "price", "ev", "clv", "pnl"]]
                 .loc[:, lambda d: ~d.columns.duplicated()].assign(market=mkt))
    B = pd.concat(B, ignore_index=True)
    B = B.merge(G[["game_id", "fav_home", "revenge_home", "home_elim", "away_elim", "home_clinch", "away_clinch"]], on="game_id")
    B["is_home"] = B.side == "home"
    B["is_fav"] = np.where(B.side == "home", B.fav_home, ~B.fav_home) & B.side.isin(["home", "away"])
    B["is_over"] = B.side == "over"
    B["is_revenge"] = np.where(B.side == "home", B.revenge_home == 1, B.revenge_home == 0) & B.revenge_home.notna()
    B["is_motivated"] = np.where(B.side == "home", B.away_elim & ~B.home_elim, B.home_elim & ~B.away_elim)
    B["is_vs_clinched"] = np.where(B.side == "home", B.away_clinch & ~B.home_clinch, B.home_clinch & ~B.away_clinch)
    return G, B


SEGMENTS = ["all", "wk12", "wk3plus", "thu", "mon", "sat", "intl", "playoff", "one_elim", "one_clinch", "big_fav",
            "div_rematch", "late"]
DIRECTIONS = {"spread": ["is_fav", "not_fav", "is_home", "not_home", "is_revenge", "is_motivated", "is_vs_clinched"],
              "ml": ["is_fav", "not_fav", "is_home", "not_home", "is_revenge", "is_motivated", "is_vs_clinched"],
              "total": ["is_over", "not_over"]}


def seg_mask(G: pd.DataFrame, s: str) -> pd.Series:
    if s == "all":
        return pd.Series(True, index=G.index)
    if s == "wk3plus":
        return ~G.wk12
    return G[s].astype(bool)


def q4_dev_tables(G, B) -> dict:
    out = {"moves": {}, "clv": {}}
    for s in SEGMENTS:
        m = seg_mask(G, s)
        d = G[m]
        out["moves"][s] = {"games": int(len(d)), "abs_move_pts": float(d.move_home_pts.abs().mean()),
                           "move_toward_fav_pts": summ(d.move_fav_pts), "move_toward_home_pts": summ(d.move_home_pts),
                           "move_toward_over_pts": summ(d.move_over_pts),
                           "abs_move_total_pts": float(d.move_over_pts.abs().mean())}
        ids = set(d.game_id)
        for mkt, dirs in DIRECTIONS.items():
            Bm = B[(B.market == mkt) & B.game_id.isin(ids)]
            for dr in dirs:
                col = dr.replace("not_", "is_")
                sel = ~Bm[col] if dr.startswith("not_") else Bm[col]
                if dr.startswith("not_") and col in ("is_fav",):
                    sel = sel & Bm.side.isin(["home", "away"])
                x = Bm[sel]
                if len(x) < 5:
                    continue
                r = summ(x.clv.values)
                r["roi"] = float(x.pnl.mean())
                r["roi_se"] = float(x.pnl.std() / math.sqrt(len(x)))
                r["ev_at_bet"] = float(x.ev.mean())
                out["clv"][f"{s}|{mkt}|{dr}"] = r
    return out


# ---------------------------------------------------------------- frozen rules (filled in after reading the dev output)
FROZEN_RULES = None  # set below after dev; see FROZEN_RULES_SPEC


def rule_bets(B: pd.DataFrame, G: pd.DataFrame, rule: dict) -> pd.DataFrame:
    ids = set(G[seg_mask(G, rule["segment"])].game_id)
    x = B[(B.market == rule["market"]) & B.game_id.isin(ids)]
    dr = rule["direction"]
    col = dr.replace("not_", "is_")
    sel = ~x[col] if dr.startswith("not_") else x[col]
    if dr == "not_fav":
        sel = sel & x.side.isin(["home", "away"])
    return x[sel]


def evaluate_rules(B, G, rules) -> dict:
    out = {}
    for r in rules:
        x = rule_bets(B, G, r)
        s = summ(x.clv.values)
        s["roi"] = float(x.pnl.mean()) if len(x) else None
        s["roi_se"] = float(x.pnl.std() / math.sqrt(len(x))) if len(x) > 1 else None
        s["ev_at_bet"] = float(x.ev.mean()) if len(x) else None
        s["pass"] = bool(len(x) > 2 and s["mean"] > 0 and s["p_one_sided"] is not None and s["p_one_sided"] < 0.05 / 3)
        out[r["id"]] = s
    return out


# ================================================================== stages
def legs_all():
    g, rows, snaps, trows, tsn, qa = load_cached()
    cache = SCR / "legs.parquet"
    if cache.exists():
        L = pd.read_parquet(cache)
        TL = pd.read_parquet(SCR / "tlegs.parquet")
        ML = pd.read_parquet(SCR / "mllegs.parquet")
    else:
        L = side_legs(rows, snaps, g)
        TL = total_legs(trows, tsn, g)
        ML = q3_ml(rows, snaps, g)
        L.to_parquet(cache)
        TL.to_parquet(SCR / "tlegs.parquet")
        ML.to_parquet(SCR / "mllegs.parquet")
    return g, rows, snaps, trows, tsn, qa, L, TL, ML


def calibration(L, g, snaps) -> dict:
    """Check the margin pmf against realized margins, centred on the price-implied close (one row per game)."""
    c = snaps.drop_duplicates("game_id")[["game_id", "mu_close"]].merge(g[["game_id", "margin", "season"]], on="game_id")
    c = c[c.mu_close.notna() & c.margin.notna()]
    mi = _mi(c.mu_close.values)
    out = {"games": len(c), "resid_sd": float((c.margin - c.mu_close).std()), "model_sigma": SIG}
    for k in (1, 2, 3, 4, 6, 7, 10, 14):
        pred = PM[mi, K.KMAX + k].sum() + PM[mi, K.KMAX - k].sum()
        act = int((c.margin.abs() == k).sum())
        out[f"|margin|={k}"] = {"pred": float(pred), "actual": act}
    return out


def refair(rows: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Re-derive fair lines under the CURRENT pmf (sensitivity runs): shift each snapshot's fair mu by the median
    change of the book-implied mu (sharp rows if any, else all cached rows)."""
    r = rows[rows.sp_home_point.notna()].copy()
    nvh = imp(r.sp_home_price) / (imp(r.sp_home_price) + imp(r.sp_away_price))
    r["d"] = implied_mu_sp(r.sp_home_point.values, nvh) - r.mu_b
    key = ["game_id", "requested_ts"]
    d_all = r.groupby(key).d.median().rename("d_all")
    d_sh = r[r.sharp].groupby(key).d.median().rename("d_sh")
    s = snaps.join(d_all, on=key).join(d_sh, on=key)
    dref = s.d_sh.where(s.n_sharp > 0, s.d_all).fillna(0)
    s["mu_ref"] = s.mu_ref + dref
    s["mu_all"] = s.mu_all + s.d_all.fillna(0)
    cl = s[s.requested_ts == s.close_ts].set_index("game_id")
    s["mu_close"] = s.game_id.map(cl.mu_ref).fillna(s.mu_close)
    s["mu_close_all"] = s.game_id.map(cl.mu_all).fillna(s.mu_close_all)
    return s.drop(columns=["d_all", "d_sh"])


def stage_dev():
    g, rows, snaps, trows, tsn, qa, L, TL, ML = legs_all()
    res = {"qa": qa, "allowed_books": sorted(ALLOWED), "calibration_margin": calibration(L, g, snaps)}
    # totals calibration
    d = tdist()
    tc = tsn.drop_duplicates("game_id")[["game_id", "tmu_close"]].merge(g[["game_id", "total"]], on="game_id").dropna()
    loc = np.interp(tc.tmu_close.values, d.mean_at, TR.MU_GRID)
    Pt = d.pmf(loc)
    res["calibration_total"] = {"games": len(tc), "resid_sd": float((tc.total - tc.tmu_close).std()), "sigma": d.sigma,
                                **{f"T={k}": {"pred": float(Pt[:, k].sum()), "actual": int((tc.total == k).sum())}
                                   for k in (37, 41, 43, 44, 47, 51)}}
    print("Q1 middles ...")
    res["q1_middles_spread"] = q1_middles(L, "spread", g)
    res["q1_middles_total"] = q1_middles(TL, "total", g)
    # sensitivity: the pmf under-predicts |margin| = 3 (see calibration_margin). Scale w(3) by the 2020-2022 (dev)
    # actual/predicted ratio and redo the spread middles (descriptive robustness check)
    cd = calibration(L, g[g.season.isin(DEV)], snaps)["|margin|=3"]
    f3 = cd["actual"] / cd["pred"]
    set_pmf({3: f3})
    Ls = side_legs(rows, refair(rows, snaps), g)
    sens = q1_middles(Ls, "spread", g)["game week (<= 6.5 days)"]
    res["q1_middles_spread_sensitivity_w3"] = {"w3_multiplier": f3, **{k: sens[k] for k in (
        "games_with_posEV_package_per_week", "share_snapshots_posEV_package", "true_middle_mean_pkg_ev",
        "policy_first_posEV_package_per_game", "calibration_all_true_middle_snapshots")}}
    set_pmf(None)
    print("Q1 sequential ...")
    res["q1_sequential_spread"] = q1_sequential(L, "spread", g)
    res["q1_sequential_total"] = q1_sequential(TL, "total", g)
    print("Q2 arbs ...")
    res["q2_arbs_spread"] = q2_arbs(L, "spread")
    res["q2_arbs_total"] = q2_arbs(TL, "total")
    res["q2_arbs_ml"], _ = ml_arbs(rows, snaps, g)
    res["q2_hourly_2025"] = hourly_arbs()
    print("Q3 shopping ...")
    res["q3_shopping_spread"] = q3_shopping(L, "spread")
    res["q3_shopping_ml"] = q3_shopping(ML, "ml")
    res["q3_shopping_total"] = q3_shopping(TL, "total")
    # realized ROI of blind bets: typical book vs best allowed, pre-kick close (sanity check; noisy)
    rz = {}
    for nm, X in (("spread", L), ("ml", ML), ("total", TL)):
        X = X[X.hours_before <= 3]
        key = ["game_id", "requested_ts", "side"]
        bi = X.loc[X.groupby(key).ev.idxmax()]
        rz[nm] = {"best_price_roi": float(bi.pnl.mean()), "best_price_roi_se": float(bi.pnl.std() / math.sqrt(len(bi))),
                  "all_books_roi": float(X.pnl.mean()), "n_bets": int(len(bi))}
    res["q3_realized_blind"] = rz
    print("Q4 dev ...")
    G, B = q4_table(DEV, L, TL, ML, snaps, tsn, g)
    res["q4_dev"] = q4_dev_tables(G, B)
    res["q4_dev"]["n_games"] = int(len(G))
    old = json.loads(JSON.read_text()) if JSON.exists() else {}
    for k in ("q4_holdout", "q4_holdout_moves", "q4_dev_frozen_rules"):
        if k in old:
            res[k] = old[k]
    JSON.write_text(json.dumps(jsonable(res), indent=1))
    print("wrote", JSON)


def stage_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; never overwrite a freeze")
    if not FROZEN_RULES_SPEC:
        raise SystemExit("fill FROZEN_RULES_SPEC first")
    FROZEN.write_text(json.dumps({"frozen_on": "2026-09-30", "pass_bar": "mean price-based CLV > 0, one-sided p < 0.05/3",
                                  "bet": "first snapshot <= 6.5 days (156 h) and > 3 h before kickoff with >= 3 allowed books posting; best allowed EV price on "
                                         "the rule's side; CLV = EV under the last pre-kick fair line (sharp if present)",
                                  "rules": FROZEN_RULES_SPEC}, indent=1))
    print("froze", FROZEN)


def stage_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("holdout locked")
    fr = json.loads(FROZEN.read_text())
    res = json.loads(JSON.read_text())
    if "q4_holdout" in res:
        raise SystemExit("holdout already run once; refusing to rerun")
    g, rows, snaps, trows, tsn, qa, L, TL, ML = legs_all()
    G, B = q4_table(HOLD, L, TL, ML, snaps, tsn, g)
    ev = evaluate_rules(B, G, fr["rules"])
    for r in fr["rules"]:
        x = rule_bets(B, G, r).merge(g[["game_id", "season"]], on="game_id")
        ev[r["id"]]["by_season"] = {int(s): summ(d.clv.values) for s, d in x.groupby("season")}
    Gd, Bd = q4_table(DEV, L, TL, ML, snaps, tsn, g)
    res["q4_dev_frozen_rules"] = evaluate_rules(Bd, Gd, fr["rules"])
    res["q4_holdout"] = ev
    res["q4_holdout_moves"] = q4_dev_tables(G, B)["moves"]
    JSON.write_text(json.dumps(jsonable(res), indent=1))
    print(json.dumps(jsonable(ev), indent=1))


# Chosen after reading q4_dev (2020-2022) only. No dev segment had positive CLV with n >= 30 except Monday unders
# (+0.66%, t=0.57). The strongest systematic drift was toward UNDERS (-0.37 pts opener->close, t=-6.7) and toward
# dogs/away, but the drift is smaller than the vig, so every broad rule had negative CLV. Frozen = the three segment
# rules with the best dev CLV among segments with >= 50 dev bets and a stated mechanism; expected to fail.
# ================================================================== report
def _pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:+.{d}f}%"


def _sh(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def stage_report():
    d = json.loads(JSON.read_text())
    fr = json.loads(FROZEN.read_text())
    W = "game week (<= 6.5 days)"
    o = ["# Middles, arbitrage, line shopping and segment efficiency (NFL 2020-2025)", "",
         "Generated by `scripts/research/middles_segments.py` (stages build -> dev -> freeze -> holdout -> report) from "
         "`output/research/middles_segments.json`. Bettable books = `my_books.json` "
         f"({', '.join(d['allowed_books'])}). EV = expected profit per unit staked under the key-number margin/total "
         "distribution centred on the snapshot's fair line (sharp books lowvig/betonlineag/circa/bookmaker if any posted, "
         "else all books), prices included. CLV = the same at the last pre-kick snapshot. 1 unit = $100 in $ figures.", ""]
    o += ["## Verdict", "",
          "* **Middles are not a money-maker at these books.** In game week, a TRUE middle (both bets can win) is on the board "
          "in ~12% (spreads) / ~16% (totals) of snapshots, but the average such 2-bet package is worth about -5% to -6% of one "
          "unit: you pay the vig twice and the middle hits ~3% of the time. Packages with positive EV appear ~0.26 games/week "
          "per market (~5-6 per season each), average +1.6% per 2-unit package, i.e. **~+0.1 unit (~$10) per season per market** "
          "at 1 unit a leg. ~90% of those +EV packages are +EV only because ONE leg is a stale/off-market price; betting that "
          "leg alone has ~3x the EV (+4.5% vs +1.6%). Including look-ahead lines (posted a week+ early, low limits) raises "
          "this to ~+0.3 (spreads) / ~+0.6 (totals) units per season.",
          "* **Sequential middles are a hedge, not an edge.** After a game-week opener bet, about half of bets later see the "
          "other side at a number giving a >= 1-point window, but adding the second bet has its own EV of about -3.7% (spreads) "
          "/ -4.1% (totals); the package only looks +EV because the first bet already gained CLV. Only ~11-13% of early bets "
          "ever get a hedge leg that is +EV on its own, and those hedge legs averaged ~0% CLV.",
          "* **True arbitrage is essentially absent among the allowed books.** Zero spread or total arbs (same or better "
          "number) in 2020-2025 after cleaning; the best cross-book same-number pair summed to exactly 1.000 implied. "
          "Moneyline arbs: ~17 games/season at our snapshot cadence, median +0.55% (p90 +2.1%) on total stake = ~0.1 "
          "unit/season. Legs were fresh by `last_update` (median max age ~4.5 min, none > 30 min); one leg is typically a "
          "soft book ~3% off the sharp price, so these are real but tiny and short-lived.",
          "* **Line shopping is the only robust value here**: the best price across the 8 allowed books is worth about "
          "+1.5 (spreads) / +2.2 (moneylines) / +1.3 (totals) percentage points of stake vs a typical single book. The 2nd "
          "account adds ~0.6-0.8 pp, the 3rd ~0.3-0.4 pp, the 4th ~0.2 pp, each further account <= 0.15 pp.",
          "* **Segments: no exploitable early-line bias.** Early (game-week opener) totals drift toward the UNDER "
          "(dev -0.37 pts, holdout -0.22 pts, both strongly significant) and early lines drift slightly toward dogs, but "
          "the drift is smaller than the vig: early unders at the best allowed price still had CLV ~-1%. All three frozen "
          "rules failed in 2023-2025 (CLV -2.9%, -1.4%, -2.5%).", ""]
    # Q1
    o += ["## 1. Middles", "", "### Same-snapshot middles (allowed books, different numbers)", "",
          "| market | subset | snapshots w/ different numbers | snapshots w/ true middle | games/week w/ true middle | "
          "true-middle pkg EV (mean, per 2u) | P(middle) mean | +EV pkg games/week | +EV pkg EV (mean) | one leg carries |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for mk in ("spread", "total"):
        for sub, x in d[f"q1_middles_{mk}"].items():
            o.append(f"| {mk} | {sub} | {_sh(x['share_snapshots_any_different_numbers'], 1)} | "
                     f"{_sh(x['share_snapshots_true_middle(both_can_win)'], 1)} | {x['games_with_true_middle_per_week']:.1f} | "
                     f"{_pct(x['true_middle_mean_pkg_ev'])} | {x['true_middle_mean_p_middle']:.3f} | "
                     f"{x['games_with_posEV_package_per_week']:.2f} | {_pct(x['posEV_mean_pkg_ev'])} | "
                     f"{_sh(x['posEV_one_leg_carries'], 0)} |")
    o += ["", "EV distribution of true-middle packages (game week; units per package of 1u + 1u):", ""]
    for mk in ("spread", "total"):
        q = d[f"q1_middles_{mk}"][W]["true_middle_pkg_ev_quantiles_(units_per_2u_staked)"]
        o.append(f"* {mk}: " + ", ".join(f"p{int(float(k) * 100)} {float(v):+.3f}" for k, v in q.items()))
    o += ["", "By window width (widest package at each game-week snapshot):", "",
          "| market | gap (pts) | snapshots | mean pkg EV | predicted P(middle) | realized middle rate | share +EV |",
          "|---|---|---|---|---|---|---|"]
    for mk in ("spread", "total"):
        for gp, x in d[f"q1_middles_{mk}"][W]["by_gap_widest_pkg"].items():
            o.append(f"| {mk} | {gp} | {x['n']} | {_pct(x['mean_pkg_ev'])} | {x['mean_p_middle']:.3f} | "
                     f"{x['realized_middle_rate']:.3f} | {_sh(x['share_pos'], 1)} |")
    o += ["", "Realized results, policy = bet 1u each leg at the first snapshot per game where the best package is +EV "
          "under BOTH the sharp and the all-book fair line:", "",
          "| market | subset | packages | per season | EV/pkg | CLV/pkg | realized/pkg (SE) | expected u/season | "
          "realized u/season | middles hit (pred) | better leg alone: EV / realized |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mk in ("spread", "total"):
        for sub, x in d[f"q1_middles_{mk}"].items():
            p = x["policy_first_posEV_package_per_game"]
            b = p["same_games_bet_only_better_leg"]
            if not p["games"]:
                continue
            o.append(f"| {mk} | {sub} | {p['games']} | {p['per_season']:.1f} | {_pct(p['mean_ev_units_per_pkg'])} | "
                     f"{_pct(p['mean_clv_units_per_pkg'])} | {p['realized_units_per_pkg']:+.3f} ({p['realized_se']:.3f}) | "
                     f"{p['expected_units_per_season']:+.2f} | {p['realized_units_per_season']:+.2f} | "
                     f"{p['middle_hits']} ({p['pred_middle_hits']:.1f}) | {_pct(b['ev_units_per_game'])} / {b['pnl_units_per_game']:+.3f} |")
    cm, ct = d["calibration_margin"], d["calibration_total"]
    sw = d["q1_middles_spread_sensitivity_w3"]
    o += ["", "Distribution checks (one row per game, centred on the price-implied close): margin residual SD "
          f"{cm['resid_sd']:.2f} vs model sigma {cm['model_sigma']:.2f}; |margin|=3 predicted {cm['|margin|=3']['pred']:.0f} "
          f"vs actual {cm['|margin|=3']['actual']}, |margin|=7 {cm['|margin|=7']['pred']:.0f} vs {cm['|margin|=7']['actual']}, "
          f"|margin|=10 {cm['|margin|=10']['pred']:.0f} vs {cm['|margin|=10']['actual']}. Totals residual SD "
          f"{ct['resid_sd']:.2f} vs sigma {ct['sigma']:.2f}. Middle-hit calibration (game-week true-middle snapshots): spreads "
          f"pred {d['q1_middles_spread'][W]['calibration_all_true_middle_snapshots']['pred_rate']:.4f} vs realized "
          f"{d['q1_middles_spread'][W]['calibration_all_true_middle_snapshots']['realized_rate']:.4f}; totals pred "
          f"{d['q1_middles_total'][W]['calibration_all_true_middle_snapshots']['pred_rate']:.4f} vs realized "
          f"{d['q1_middles_total'][W]['calibration_all_true_middle_snapshots']['realized_rate']:.4f} (snapshots within a game "
          "are correlated). The pmf under-predicts margins of exactly 3 by ~20% in 2020-2025. Sensitivity with w(3) x "
          f"{sw['w3_multiplier']:.2f} (dev ratio) and fair lines re-derived: +EV spread packages "
          f"{sw['policy_first_posEV_package_per_game']['per_season']:.1f}/season, expected "
          f"{sw['policy_first_posEV_package_per_game']['expected_units_per_season']:+.2f} u/season - same conclusion.", ""]
    o += ["### Sequential middles (bet early, hedge later)", "",
          "Early bet = best allowed EV price on each side at the game-week opener (first snapshot <= 6.5 days out with >= 3 "
          "allowed books). Later = any later snapshot before kickoff. Package EV valued at the later snapshot's fair line.", "",
          "| market | early bets | early CLV | later gap >= 0.5 | >= 1 | >= 1.5 | window holds 3 or 7 | package EV > 0 | "
          "hedge leg itself +EV | hedge policy: n/season, hedge-leg EV, CLV, realized (SE) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mk in ("spread", "total"):
        x = d[f"q1_sequential_{mk}"]
        h = x["hedge_policy"]
        o.append(f"| {mk} | {x['early_bets']} | {_pct(x['early_bet_mean_clv'])} | {_sh(x['share_early_bets_later_gap>=0.5'], 0)} | "
                 f"{_sh(x['share_early_bets_later_gap>=1'], 0)} | {_sh(x['share_early_bets_later_gap>=1.5'], 0)} | "
                 f"{_sh(x['share_early_bets_later_window contains 3 or 7'], 0) if mk == 'spread' else 'n/a'} | "
                 f"{_sh(x['share_early_bets_later_middle & pkg EV>0'], 0)} | {_sh(x['share_early_bets_later_middle & hedge leg EV>0'], 0)} | "
                 f"{h['per_season']:.0f}, {_pct(h['hedge_leg_ev_now_mean'])}, {_pct(h['hedge_leg_clv_mean'])}, "
                 f"{h['hedge_leg_realized_mean']:+.3f} ({h['hedge_leg_realized_se']:.3f}) |")
    o += ["", "Hedge policy = add the other side at the first later snapshot with a >= 1-point window and package EV > 0. "
          "Once the first bet is placed it is sunk: the decision to add the second bet is worth only the second bet's own EV "
          "(negative on average). Middling reduces variance; it does not create EV.", ""]
    # Q2
    o += ["## 2. Arbitrage (allowed books only, same snapshot)", "",
          "| market | arb snapshots | share | games/season | median return | p90 return | median max leg age (min) | "
          "off-market leg EV vs sharp (median) |", "|---|---|---|---|---|---|---|---|"]
    a = d["q2_arbs_ml"]
    o.append(f"| moneyline | {a['snapshots']} | {_sh(a['share_snapshots'])} | {a['per_season']:.1f} | "
             f"{_pct(a['arb_return_median'])} | {_pct(a['arb_return_p90'])} | {a['max_leg_age_min_median']:.1f} | "
             f"{_pct(a['off_market_leg_ev_median'])} |")
    for mk in ("spread", "total"):
        for lab, x in d[f"q2_arbs_{mk}"].items():
            o.append(f"| {mk} {lab} | {x['snapshots']} | {_sh(x['share_snapshots'])} | {x['per_season']:.1f} | - | - | - | - |")
    h = d["q2_hourly_2025"]
    o += ["", f"Moneyline arb book pairs: {a['book_pairs']}. By season (games): {a['by_season']}.",
          f"Hourly 2025 sample (Sat 15:00-Sun 16:00 UTC, 161 events): ML arb in {_sh(h['moneyline']['share_with_arb'])} of "
          f"event-hours ({h['moneyline']['events_with_any_arb']} events); still there 1 h later "
          f"{_sh(h['moneyline']['arb_still_there_1h_later'], 0)} (tiny n). Spread same-number arb: "
          f"{h['spread_same_number']['events_with_any_arb']} event.",
          "Staleness: rows with a book's own two sides summing to < 1 (in-book arbs) and books whose own no-vig ML was "
          "> 10 points from the snapshot median (e.g. swapped teams) were dropped as feed errors; before that filter a few "
          "'arbs' of +30-180% appeared (all garbage). `last_update` only says when the book's event was last refreshed in "
          "the feed, not that the price was still bettable; with a daily snapshot we cannot see how long each arb lived.", ""]
    # Q3
    o += ["## 3. Line-shopping value (blind bets, both sides of every game)", "",
          "EV of a random bet at each snapshot under the sharp fair line. 'Typical' = average over allowed books posting. "
          "Greedy curve = best price over the first k accounts (accounts added in the order that helps most; a bet a set "
          "cannot place is valued at the typical price).", ""]
    for mk in ("spread", "ml", "total"):
        for era in ("2020-2022", "2023-2025"):
            for when in ("pre-kick close", "early (>= 3 days out)"):
                x = d[f"q3_shopping_{mk}"][era].get(when, {}).get("ev")
                if not x:
                    continue
                curve = " -> ".join(f"{c['add']} {100 * c['mean']:+.2f}" for c in x["greedy_curve"])
                o.append(f"* **{mk}, {era}, {when}** (n={x['n_bets']}): typical book {_pct(x['typical_book_mean'])}, "
                         f"best of all allowed {_pct(x['best_all_allowed_mean'])} (+{100 * (x['best_all_allowed_mean'] - x['typical_book_mean']):.2f} pp). "
                         f"Greedy: {curve}")
    rz = d["q3_realized_blind"]
    o += ["", "Realized check at the pre-kick snapshot (both sides of every game): " + "; ".join(
        f"{k}: best price ROI {_pct(v['best_price_roi'], 1)} (SE {100 * v['best_price_roi_se']:.1f}) vs all books "
        f"{_pct(v['all_books_roi'], 1)}" for k, v in rz.items()) + ".",
        "Per extra account (2023-2025, 8 books): 2nd ~ +0.6-0.8 pp, 3rd ~ +0.3-0.4 pp, 4th ~ +0.2 pp, 5th-8th <= 0.15 pp "
        "each. Book coverage matters: fanatics posted only ~1/3 of games in this data (it only appears in 2025), so its "
        "measured marginal value is understated for the future. At $100 x 100 bets the 2nd account is worth ~$60-80, the 3rd "
        "~$30-40, the 4th ~$20. Shopping cuts the vig roughly in half; it does not make blind bets profitable.", ""]
    # Q4
    q = d["q4_dev"]
    o += ["## 4. Market-segment efficiency (CLV-based, no model)", "",
          "Bet = game-week opener (first snapshot <= 6.5 days and > 3 h before kickoff with >= 3 allowed books), best allowed "
          "EV price on the rule's side. CLV = EV under the last pre-kick fair line. Moves are in fair points (sharp-if-present "
          "fair line, prices included), opener -> close. Eliminated/clinched use only earlier results and sufficient "
          "conditions without tiebreakers (eliminated: >= 7 conference teams already ahead of the team's maximum; clinched: "
          "<= 6 conference teams can still reach its current wins), weeks 12+.", "",
          "### Line moves opener -> close (positive = toward favorite / home / over)", "",
          "| segment | dev games | dev |move| | dev toward fav | dev toward home | dev toward over | holdout games | "
          "holdout toward fav | holdout toward home | holdout toward over |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "n/a" if not x.get("n") or x.get("t") is None else f"{x['mean']:+.2f} (t {x['t']:+.1f})"
    for sg, x in q["moves"].items():
        y = d.get("q4_holdout_moves", {}).get(sg, {})
        o.append(f"| {sg} | {x['games']} | {x['abs_move_pts']:.2f} | {f(x['move_toward_fav_pts'])} | {f(x['move_toward_home_pts'])} | "
                 f"{f(x['move_toward_over_pts'])} | {y.get('games', '')} | {f(y.get('move_toward_fav_pts', {}))} | "
                 f"{f(y.get('move_toward_home_pts', {}))} | {f(y.get('move_toward_over_pts', {}))} |")
    o += ["", "### Dev 2020-2022: CLV of segment x side rules (best 12 and the all-games baselines)", "",
          "| rule | n | CLV | t | EV at bet | ROI (SE) |", "|---|---|---|---|---|---|"]
    rows_ = sorted(q["clv"].items(), key=lambda kv: -(kv[1]["t"] or -99))
    pick = rows_[:12] + [kv for kv in rows_ if kv[0].startswith("all|") and kv not in rows_[:12]]
    for k, v in pick:
        o.append(f"| {k} | {v['n']} | {_pct(v['mean'])} | {v['t']:+.2f} | {_pct(v['ev_at_bet'])} | {v['roi']:+.3f} ({v['roi_se']:.3f}) |")
    o += ["", f"### Frozen rules (`{FROZEN.name}`) and the 2023-2025 holdout (run once)", "",
          "Pass bar: mean CLV > 0 with one-sided p < 0.05/3.", "",
          "| rule | dev n | dev CLV | holdout n | holdout CLV | t | p | holdout ROI (SE) | pass |", "|---|---|---|---|---|---|---|---|---|"]
    for r in fr["rules"]:
        dv, hv = d["q4_dev_frozen_rules"][r["id"]], d["q4_holdout"][r["id"]]
        o.append(f"| {r['id']} ({r['segment']}, {r['market']}, {r['direction']}) | {dv['n']} | {_pct(dv['mean'])} | {hv['n']} | "
                 f"{_pct(hv['mean'])} | {hv['t']:+.2f} | {hv['p_one_sided']:.4f} | {hv['roi']:+.3f} ({hv['roi_se']:.3f}) | "
                 f"{'PASS' if hv['pass'] else 'fail'} |")
    o += ["", "By season (holdout CLV): " + "; ".join(
        f"{rid}: " + ", ".join(f"{s_} {_pct(v['mean'])}" for s_, v in x["by_season"].items())
        for rid, x in d["q4_holdout"].items()), "",
          "Reading: the consistent descriptive finding is that game-week opening totals move DOWN to the close (~0.2-0.4 "
          "points on average, in nearly every segment; same with all-book instead of sharp fair lines: -0.39 dev, -0.20 "
          "holdout). That is a move, not an outcome bias: final totals averaged +0.65 over the opener and +0.86 over the "
          "close in 2023-2025 (vs -0.40 / -0.01 in 2020-2022). Lines drift slightly toward underdogs (-0.16 pts dev, t -2.7; "
          "-0.07 holdout, not significant); the home-side drift changed sign between periods. None of these "
          "moves is large enough to beat ~3% of vig even at the best of 8 books, so there is no segment where simply "
          "betting early is +CLV. Small segments (international n=7/16, Saturday, clinch/elimination n=17-78) are too "
          "small to say anything.", ""]
    o += ["## Caveats", "",
          "* Snapshots are daily (+ Friday + ~75 min pre-kick), so short-lived middles/arbs between snapshots are missed "
          "and the ones seen may not have been bettable at size (limits, bet-acceptance delays, palpable-error voids). "
          "Totals come from region `us` only: espnbet and hardrockbet are absent and fanatics appears only in 2025.",
          "* EV depends on the key-number distribution; it reproduces middle-hit rates within sampling error but "
          "under-predicts exact 3-point margins (sensitivity above). Sharp books are present at ~63% of side snapshots; "
          "otherwise the all-book median is the fair line.",
          "* Q1-Q3 are descriptive over all seasons (no parameters were tuned on them). Q4 rules were chosen on 2020-2022 "
          "and the 2023-2025 holdout was run once.", ""]
    MD.write_text("\n".join(o))
    print("wrote", MD)


FROZEN_RULES_SPEC: list[dict] = [
    {"id": "R1_monday_under", "segment": "mon", "market": "total", "direction": "not_over",
     "why": "dev CLV +0.66% (n=57, t=0.57); Monday totals drifted down most (-0.51 pts)"},
    {"id": "R2_week1_2_under", "segment": "wk12", "market": "total", "direction": "not_over",
     "why": "dev CLV -0.38% (n=96); weeks 1-2 totals drifted down -0.48 pts (t=-3.6), early-season scoring priors"},
    {"id": "R3_thursday_away_spread", "segment": "thu", "market": "spread", "direction": "not_home",
     "why": "dev CLV -0.89% (n=51); Thursday lines drifted toward the away side (-0.50 pts)"},
]

if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"build": build, "dev": stage_dev, "freeze": stage_freeze, "holdout": stage_holdout, "report": stage_report}.get(
        st, lambda: print("stages: build dev freeze holdout report"))()
