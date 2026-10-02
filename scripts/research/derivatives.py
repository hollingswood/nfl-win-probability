"""NFL DERIVATIVE markets research: 1st-half spreads, 1st-half totals and team totals vs the full-game market.

Stages (run in order; outputs in output/research/):
  python scripts/research/derivatives.py fit       # truth: 1H / team-points relationships + key-number pmfs,
                                                    #   learned on nflverse 2012-2022 (closing FG lines vs results)
  python scripts/research/derivatives.py dev       # 2023 only: book ratios, deviations, timing, candidate rules
  python scripts/research/derivatives.py freeze    # writes derivatives_frozen.json from FROZEN (refuses overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN python scripts/research/derivatives.py holdout   # 2024-2025, ONCE
  output/research/derivatives.md is written from derivatives.json.

Definitions
  * Derivative odds: data/historical_odds/derivatives/ (two snapshots per game: "early" = Fri 21:40 UTC for Sunday
    games / kickoff-24h otherwise, "close" = kickoff-75min).  spreads_h1: one row per team, price in over_price.
  * Distributions (all fit on 2012-2022 only):
      1H home margin  M1:  pmf(k) ∝ Normal(k; loc, s_m(T)) * w(|k|),   k = -50..50   (ties at 0 are frequent)
      1H total points P1:  pmf(k) ∝ Normal(k; loc, s(loc)) * w(k),      k = 0..80
      team points     X:   pmf(k) ∝ Normal(k; loc, s(loc)) * w(k),      k = 0..80    (includes OT, like books)
    w = key-number weights by iterative raking (normal-expected vs actual counts), loc chosen so the pmf MEAN equals
    the target mean.  Mean models (OLS on closing nflverse lines): E[M1 | S, T], E[P1 | T, |S|], E[X | T, S].
  * Fair of a quote: P(win), P(push) under that pmf.  Implied mean of a quote = the mean that reproduces its
    two-way no-vig probability (pushes excluded).  Snapshot consensus of a derivative = median implied mean
    over all books ("dcons").
  * FG fair at the same snapshot: S = expected home margin (margin_total_model.json, spreads + juice, median over
    books), T = expected total (totals_dist.json, median over books); also sharp versions (lowvig, betonlineag).
    FG-implied derivative fair = mean model applied to (S, T).
  * CLV (the pass test) = EV of the bet under the CLOSE snapshot's consensus distribution of the SAME derivative
    market (all books' no-vig prices -> median implied mean):  P(win)*decimal + P(push) - 1.
  * PnL graded on actual 1H / team scores (push refunds).
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from nflpred import features as F  # noqa: E402
from nflpred import margin_total as MT  # noqa: E402
from nflpred.totals import TotalDist  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "derivatives.json"
FROZEN = OUT / "derivatives_frozen.json"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/deriv")
SCR.mkdir(parents=True, exist_ok=True)
FIT_SEASONS = range(2012, 2023)
DEV, HOLD = (2023,), (2024, 2025)
SHARP_FG = {"lowvig", "betonlineag"}
ALLOWED = set(json.loads((ROOT / "my_books.json").read_text())["allowed_books"])
TEAM_FULL = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}


# ================================================================== small helpers
def dec(p):
    p = np.asarray(p, float)
    return np.where(p > 0, 1 + p / 100, 1 + 100 / -p)


def imp(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p < 0, -p / (-p + 100), 100 / (p + 100))


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def _js(x):
    if isinstance(x, dict):
        return {str(k): _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_json(key, val):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d[key] = _js(val)
    JSON.write_text(json.dumps(d, indent=1))


def ols(X, y, w=None):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    y = np.asarray(y, float)
    W = np.ones(len(y)) if w is None else np.asarray(w, float)
    XtW = X.T * W
    b = np.linalg.solve(XtW @ X, XtW @ y)
    r = y - X @ b
    s2 = (W * r * r).sum() / (W.sum() - X.shape[1])
    cov = s2 * np.linalg.inv(XtW @ X)
    return b, np.sqrt(np.diag(cov)), r


# ================================================================== history: halves + finals + closing FG lines
def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[(g.season >= 2012) & (g.season <= 2025)].copy()
    for c in ("home_team", "away_team"):
        g[c] = F._norm_team(g[c])
    g["gameday"] = pd.to_datetime(g.gameday)
    g["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()], utc=True)
    return g


def history() -> pd.DataFrame:
    """One row per completed game 2012-2025: closing nflverse S/T + 1H and final points per team."""
    p = SCR / "history.parquet"
    if p.exists():
        return pd.read_parquet(p)
    g = load_games()
    fr = []
    for s in range(2012, 2026):
        x = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet",
                            columns=["game_id", "qtr", "total_home_score", "total_away_score"])
        h = x[x.qtr <= 2].groupby("game_id")[["total_home_score", "total_away_score"]].max()
        fr.append(h.rename(columns={"total_home_score": "h1h", "total_away_score": "h1a"}))
    h = pd.concat(fr)
    g = g.merge(h, left_on="game_id", right_index=True, how="inner")
    g = g[g.home_score.notna() & g.spread_line.notna() & g.total_line.notna()].copy()
    g["S"] = g.spread_line.astype(float)          # expected home margin (nflverse: + = home favoured)
    g["TOT"] = g.total_line.astype(float)
    g["m1"] = g.h1h - g.h1a
    g["p1"] = g.h1h + g.h1a
    g = g[["game_id", "season", "week", "game_type", "gameday", "kick", "home_team", "away_team", "S", "TOT",
           "h1h", "h1a", "m1", "p1", "home_score", "away_score", "home_coach", "away_coach", "roof", "div_game"]]
    g = g.reset_index(drop=True)
    g.to_parquet(p)
    return g


# ================================================================== key-number distributions
class KeyDist:
    """Discrete pmf ∝ Normal(k; loc, sigma) * w(k) with the location solved so the pmf MEAN = target mean.

    kind 'margin': support -50..50, weights symmetric in |k|, sigma = s0 + s1*(scale - 44) where scale = FG total.
    kind 'count' : support 0..80,   sigma = s0 + s1*(loc - ref).
    """
    def __init__(self, kind, s0, s1, ref, w):
        self.kind, self.s0, self.s1, self.ref = kind, float(s0), float(s1), float(ref)
        self.w = np.asarray(w, float)
        if kind == "margin":
            self.K = np.arange(-50, 51)
            self.LOC = np.round(np.arange(-30, 30.0001, 0.02), 3)
        else:
            self.K = np.arange(0, 81)
            self.LOC = np.round(np.arange(0.0, 60.0001, 0.02), 3)
        self._tab = {}

    def wvec(self):
        return self.w[np.abs(self.K)] if self.kind == "margin" else self.w

    def sigma_of(self, loc, scale):
        if self.kind == "margin":
            return np.full(np.shape(loc), self.s0 + self.s1 * (np.clip(scale, 34, 58) - 44))
        return np.maximum(self.s0 + self.s1 * (np.asarray(loc) - self.ref), 1.5)

    def raw(self, loc, scale):
        loc = np.atleast_1d(np.asarray(loc, float))
        sig = np.atleast_1d(self.sigma_of(loc, scale))
        P = np.exp(-0.5 * ((self.K[None, :] - loc[:, None]) / sig[:, None]) ** 2) * self.wvec()[None, :]
        return P / P.sum(1, keepdims=True)

    def table(self, scale=44.0):
        key = round(float(scale)) if self.kind == "margin" else 0
        t = self._tab.get(key)
        if t is None:
            P = self.raw(self.LOC, key)
            mean = P @ self.K
            order = np.argsort(mean)
            t = {"P": P[order], "cdf": np.cumsum(P[order], 1), "mean": mean[order]}
            self._tab[key] = t
        return t

    def _rows(self, mean, scale):
        t = self.table(scale)
        i = np.searchsorted(t["mean"], np.asarray(mean, float))
        return t, np.clip(i, 0, len(t["mean"]) - 1)

    def pmf(self, mean, scale=44.0):
        t, i = self._rows(np.atleast_1d(mean), scale)
        return t["P"][i]

    def p_gt_eq(self, mean, x, scale=44.0):
        """P(X > x), P(X == x) for X with the given mean (vectorized over rows; one scale)."""
        t, i = self._rows(np.atleast_1d(mean), scale)
        x = np.atleast_1d(np.asarray(x, float))
        k0 = self.K[0]
        j = np.floor(x + 1e-9).astype(int) - k0                       # index of floor(x)
        cdf = t["cdf"][i, np.clip(j, 0, len(self.K) - 1)]
        cdf = np.where(j < 0, 0.0, np.where(j >= len(self.K), 1.0, cdf))
        isint = np.abs(x - np.round(x)) < 1e-9
        pe = np.where(isint, t["P"][i, np.clip(np.round(x).astype(int) - k0, 0, len(self.K) - 1)], 0.0)
        return 1 - cdf, pe

    def implied_mean(self, x, q_over, scale=44.0):
        """Mean m such that P(X>x)/(P(X>x)+P(X<x)) = q_over."""
        t = self.table(scale)
        x = np.atleast_1d(np.asarray(x, float))
        q_over = np.atleast_1d(np.asarray(q_over, float))
        out = np.full(len(x), np.nan)
        for xv in np.unique(x[np.isfinite(x)]):
            s = x == xv
            gt, eq = self.p_gt_eq(t["mean"], np.full(len(t["mean"]), xv), scale)
            q = np.maximum.accumulate(gt / np.maximum(1 - eq, 1e-12))
            out[s] = np.interp(q_over[s], q, t["mean"])
        return out

    def to_dict(self):
        return {"kind": self.kind, "s0": self.s0, "s1": self.s1, "ref": self.ref, "w": [round(float(v), 5) for v in self.w]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["kind"], d["s0"], d["s1"], d["ref"], d["w"])


def fit_keydist(kind, mean, actual, scale, s0, s1, ref, smooth=8.0, rounds=4):
    """Rake key weights so the summed model pmf matches actual counts (location re-solved each round)."""
    nw = 51 if kind == "margin" else 81
    w = np.ones(nw)
    mean = np.asarray(mean, float)
    actual = np.asarray(actual, int)
    scale = np.asarray(scale, float)
    for _ in range(rounds):
        d = KeyDist(kind, s0, s1, ref, w)
        E = np.zeros(nw)
        A = np.bincount(np.clip(np.abs(actual) if kind == "margin" else actual, 0, nw - 1), minlength=nw).astype(float)
        sc_r = np.round(scale) if kind == "margin" else np.zeros(len(mean))
        for sc in np.unique(sc_r):
            s = sc_r == sc
            P = d.pmf(mean[s], sc)
            col = P.sum(0)
            if kind == "margin":
                np.add.at(E, np.abs(d.K), col)
            else:
                E += col
        f = (A + smooth) / (E + smooth)
        w = np.clip(w * f, 0.05, 8.0)
    return KeyDist(kind, s0, s1, ref, w)


def loglik(d: KeyDist, mean, actual, scale=None):
    mean = np.asarray(mean, float)
    actual = np.asarray(actual, int)
    sc = np.round(np.asarray(scale, float)) if (d.kind == "margin" and scale is not None) else np.full(len(mean), 44.0)
    out = np.zeros(len(mean))
    for v in np.unique(sc):
        s = sc == v
        P = d.pmf(mean[s], v)
        out[s] = np.log(np.maximum(P[np.arange(s.sum()), np.clip(actual[s] - d.K[0], 0, len(d.K) - 1)], 1e-12))
    return out


# ================================================================== the fair-relationship model
class Fair:
    """Mean models + pmfs.  coefs: m1 = a0 + a1*S + a2*S*(T-44)/10 + a3*S^3/100
                                   p1 = b0 + b1*T + b2*|S|
                                   team (own exp margin s = ±S): x = c0 + c1*T + c2*s + c3*|s| """
    def __init__(self, d):
        self.d = d
        self.a = np.asarray(d["coef_m1"], float)
        self.b = np.asarray(d["coef_p1"], float)
        self.c = np.asarray(d["coef_team"], float)
        self.dm1 = KeyDist.from_dict(d["dist_m1"])
        self.dp1 = KeyDist.from_dict(d["dist_p1"])
        self.dx = KeyDist.from_dict(d["dist_team"])

    def m1(self, S, T):
        S, T = np.asarray(S, float), np.asarray(T, float)
        return self.a[0] + self.a[1] * S + self.a[2] * S * (T - 44) / 10 + self.a[3] * S ** 3 / 100

    def p1(self, S, T):
        S, T = np.asarray(S, float), np.asarray(T, float)
        return self.b[0] + self.b[1] * T + self.b[2] * np.abs(S)

    def team(self, s_own, T):
        s, T = np.asarray(s_own, float), np.asarray(T, float)
        return self.c[0] + self.c[1] * T + self.c[2] * s + self.c[3] * np.abs(s)


def fit_stage():
    h = history()
    tr = h[h.season.isin(FIT_SEASONS)].copy()
    te = h[h.season.isin(DEV)].copy()                 # 2023 dev check (closing lines only)
    out = {"n_fit_games": len(tr)}
    # ---- mean models
    S, T = tr.S.values, tr.TOT.values
    a, ase, rm = ols([S, S * (T - 44) / 10, S ** 3 / 100], tr.m1.values)
    b, bse, rp = ols([T, np.abs(S)], tr.p1.values)
    # team rows (home and away stacked)
    st = np.r_[S, -S]
    Tt = np.r_[T, T]
    xt = np.r_[tr.home_score.values, tr.away_score.values]
    c, cse, rx = ols([Tt, st, np.abs(st)], xt)
    # simple ratio views (what "rules of thumb" say)
    simple = {
        "m1_over_S": float((tr.m1 * tr.S).sum() / (tr.S ** 2).sum()),
        "p1_over_T": float(tr.p1.sum() / tr.TOT.sum()),
        "p1_share_by_T": {f"{lo}-{lo + 4}": float(tr[(tr.TOT >= lo) & (tr.TOT < lo + 4)].p1.mean() /
                                                   tr[(tr.TOT >= lo) & (tr.TOT < lo + 4)].TOT.mean())
                          for lo in (36, 40, 44, 48, 52)},
        "m1_share_by_absS": {}, "team_bias_vs_half_formula": float((xt - (Tt + st) / 2).mean()),
        "fg_total_bias": float((tr.home_score + tr.away_score - tr.TOT).mean()),
        "fg_margin_bias": float((tr.home_score - tr.away_score - tr.S).mean()),
        "tie_rate_1h": float((tr.m1 == 0).mean()),
    }
    for lo, hi in ((0, 2.5), (2.5, 4.5), (4.5, 7.5), (7.5, 10.5), (10.5, 30)):
        z = tr[(tr.S.abs() >= lo) & (tr.S.abs() < hi)]
        simple["m1_share_by_absS"][f"{lo}-{hi}"] = float((z.m1 * np.sign(z.S)).mean() / z.S.abs().mean()) if len(z) else None
    out["coef_m1"], out["se_m1"] = a.tolist(), ase.tolist()
    out["coef_p1"], out["se_p1"] = b.tolist(), bse.tolist()
    out["coef_team"], out["se_team"] = c.tolist(), cse.tolist()
    out["simple"] = simple
    # ---- sigma models (residual sd)
    mu_m1 = a[0] + a[1] * S + a[2] * S * (T - 44) / 10 + a[3] * S ** 3 / 100
    g1, _, _ = ols([T - 44], np.abs(rm) * math.sqrt(math.pi / 2))
    mu_p1 = b[0] + b[1] * T + b[2] * np.abs(S)
    g2, _, _ = ols([mu_p1 - 22], np.abs(rp) * math.sqrt(math.pi / 2))
    mu_x = c[0] + c[1] * Tt + c[2] * st + c[3] * np.abs(st)
    g3, _, _ = ols([mu_x - 22], np.abs(rx) * math.sqrt(math.pi / 2))
    dm1 = fit_keydist("margin", mu_m1, tr.m1.values, T, g1[0], g1[1], 44.0)
    dp1 = fit_keydist("count", mu_p1, tr.p1.values, None, g2[0], g2[1], 22.0)
    dx = fit_keydist("count", mu_x, xt, None, g3[0], g3[1], 22.0)
    out["dist_m1"], out["dist_p1"], out["dist_team"] = dm1.to_dict(), dp1.to_dict(), dx.to_dict()
    fair = Fair(out)
    # ---- checks on 2023 (closing nflverse lines): log-lik vs plain normal, calibration of key probabilities
    chk = {}
    Se, Te = te.S.values, te.TOT.values
    for name, d, mu, act, sc in (
            ("m1", fair.dm1, fair.m1(Se, Te), te.m1.values, Te),
            ("p1", fair.dp1, fair.p1(Se, Te), te.p1.values, None),
            ("team", fair.dx, fair.team(np.r_[Se, -Se], np.r_[Te, Te]), np.r_[te.home_score, te.away_score], None)):
        flat = KeyDist(d.kind, d.s0, d.s1, d.ref, np.ones_like(d.w))
        ll = loglik(d, mu, act, sc).mean()
        ll0 = loglik(flat, mu, act, sc).mean()
        chk[name] = {"n": len(act), "ll_keyweights": ll, "ll_plain_normal": ll0, "mean_resid": float((act - mu).mean())}
    # 1H tie rate predicted vs actual (2023)
    pt = fair.dm1.p_gt_eq(fair.m1(Se, Te), np.zeros(len(Se)), 44)[1]
    chk["tie_1h_pred_vs_act_2023"] = [float(pt.mean()), float((te.m1 == 0).mean())]
    out["check_2023_closing"] = chk
    # in-sample key-number table
    out["key_weights_m1_abs0_14"] = dm1.w[:15].round(3).tolist()
    out["team_pts_freq_fit"] = {int(k): float((xt == k).mean()) for k in (0, 3, 7, 10, 13, 14, 17, 20, 21, 23, 24, 27, 28, 30, 31, 34)}
    out["p1_freq_fit"] = {int(k): float((tr.p1 == k).mean()) for k in (0, 3, 7, 10, 13, 14, 17, 20, 21, 23, 24, 27, 28, 31)}
    out["m1_freq_fit"] = {int(k): float((tr.m1.abs() == k).mean()) for k in range(0, 15)}
    team_eff = team_tendencies(h)
    out["team_tendencies"] = team_eff
    save_json("fit", out)
    return out


def team_tendencies(h: pd.DataFrame):
    """Do prior-game 1H residuals (fast starters / slow finishers) predict the next game's 1H residual?
    Uses only games before each game (rolling over the previous 16 team-games), tested on 2012-2022."""
    tr = h[h.season.isin(FIT_SEASONS)].copy()
    S, T = tr.S.values, tr.TOT.values
    a, _, _ = ols([S, S * (T - 44) / 10, S ** 3 / 100], tr.m1.values)
    b, _, _ = ols([T, np.abs(S)], tr.p1.values)
    tr["r_m1"] = tr.m1 - (a[0] + a[1] * S + a[2] * S * (T - 44) / 10 + a[3] * S ** 3 / 100)
    tr["r_p1"] = tr.p1 - (b[0] + b[1] * T + b[2] * np.abs(S))
    # 2H residual relative to remaining expected margin (slow finishers)
    rows = []
    for side, sgn in (("home", 1), ("away", -1)):
        z = tr[["game_id", "gameday", f"{side}_team", "r_m1", "r_p1"]].rename(columns={f"{side}_team": "team"})
        z["r_m1_own"] = sgn * z.r_m1
        rows.append(z)
    tg = pd.concat(rows).sort_values(["team", "gameday"])
    for c in ("r_m1_own", "r_p1"):
        tg[f"prior_{c}"] = tg.groupby("team")[c].transform(lambda s: s.shift(1).rolling(16, min_periods=8).mean())
    res = {}
    for c in ("r_m1_own", "r_p1"):
        z = tg.dropna(subset=[f"prior_{c}"])
        bb, se, _ = ols([z[f"prior_{c}"].values], z[c].values)
        res[c] = {"slope": float(bb[1]), "se": float(se[1]), "n": len(z),
                  "corr": float(np.corrcoef(z[f"prior_{c}"], z[c])[0, 1])}
    return res


def load_fair() -> Fair:
    return Fair(json.loads(JSON.read_text())["fit"])


# ================================================================== odds: full-game fair at each derivative snapshot
def _match(o: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    ev = o.drop_duplicates("event_id")[["event_id", "home", "away", "commence", "season"]]
    m = ev.merge(g[["game_id", "home_team", "away_team", "kick", "season"]],
                 left_on=["home", "away", "season"], right_on=["home_team", "away_team", "season"])
    m["dt"] = (m.commence - m.kick).abs().dt.total_seconds()
    m = m[m.dt < 48 * 3600].sort_values("dt").drop_duplicates("event_id")
    return o.merge(m[["event_id", "game_id"]], on="event_id")


def _read(path, season):
    o = pd.read_csv(path).assign(season=season)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    o["home"] = F._norm_team(o.home)
    o["away"] = F._norm_team(o.away)
    return o


def fg_fair(seasons) -> pd.DataFrame:
    """Per (event, requested_ts) of the full-game files: S (exp. home margin) and T (exp. total), all/sharp."""
    p = SCR / f"fg_{'_'.join(map(str, seasons))}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    td = TotalDist.load()
    mm = MT.load()
    tot, sid = [], []
    for s in seasons:
        t = _read(ROOT / "data" / "historical_odds" / "totals" / f"nfl_odds_{s}.csv.gz", s)
        ok = (t.tot_over_price.between(-250, 200) & t.tot_under_price.between(-250, 200)
              & (t.tot_over_price.abs() >= 100) & (t.tot_under_price.abs() >= 100) & t.tot_point.between(25, 80))
        t = t[ok].copy()
        ov = imp(t.tot_over_price) + imp(t.tot_under_price)
        t = t[(ov > 1.0) & (ov < 1.12)].copy()
        t["nv"] = imp(t.tot_over_price) / (imp(t.tot_over_price) + imp(t.tot_under_price))
        med = t.groupby(["event_id", "requested_ts"]).tot_point.transform("median")
        t = t[(t.tot_point - med).abs() <= 5].copy()
        t["Tb"] = np.nan
        for pt in t.tot_point.unique():
            s_ = t.tot_point == pt
            t.loc[s_, "Tb"] = [td.implied_mu(pt, q) for q in t.loc[s_, "nv"]]
        tot.append(t)
        o = _read(ROOT / "data" / "historical_odds" / f"nfl_odds_{s}.csv.gz", s)
        ok = (o.sp_home_point.notna() & o.sp_home_price.between(-250, 200) & o.sp_away_price.between(-250, 200)
              & (o.sp_home_point == -o.sp_away_point))
        o = o[ok].copy()
        ov = imp(o.sp_home_price) + imp(o.sp_away_price)
        o = o[(ov > 1.0) & (ov < 1.12)].copy()
        med = o.groupby(["event_id", "requested_ts"]).sp_home_point.transform("median")
        o = o[(o.sp_home_point - med).abs() <= 4].copy()
        o["q"] = imp(o.sp_home_price) / (imp(o.sp_home_price) + imp(o.sp_away_price))
        sid.append(o)
    t = pd.concat(tot)
    o = pd.concat(sid)
    key = ["event_id", "requested_ts"]
    T_all = t.groupby(key).Tb.median().rename("T_all")
    T_sh = t[t.book.isin(SHARP_FG)].groupby(key).Tb.median().rename("T_sharp")
    o = o.merge(T_all.reset_index(), on=key, how="left")
    o["Sb"] = mm.implied_mu(o.sp_home_point.values, o.q.values, o.T_all.fillna(44.0).values)
    S_all = o.groupby(key).Sb.median().rename("S_all")
    S_sh = o[o.book.isin(SHARP_FG)].groupby(key).Sb.median().rename("S_sharp")
    nb = o.groupby(key).book.nunique().rename("n_fg_books")
    f = pd.concat([S_all, S_sh, nb, T_all, T_sh], axis=1).reset_index()
    f.to_parquet(p)
    return f


# ================================================================== derivative quotes
def load_derivs(seasons) -> pd.DataFrame:
    """One row per (event, snapshot, book, market, team-or-total) two-way quote, as an over/under on X at x."""
    fr = []
    for s in seasons:
        fr.append(_read(ROOT / "data" / "historical_odds" / "derivatives" /
                        f"spreads_h1+totals_h1+team_totals_{s}.csv.gz", s))
    d = pd.concat(fr, ignore_index=True)
    d["snap"] = np.where((d.commence - d.requested_ts).dt.total_seconds() < 3 * 3600, "close", "early")
    d["team"] = d.player.map(TEAM_FULL)
    # ---- 1H spreads: pair home/away rows
    sp = d[d.market == "spreads_h1"]
    key = ["event_id", "requested_ts", "book"]
    hrow = sp[sp.team == sp.home][key + ["point", "over_price"]].rename(columns={"point": "hp", "over_price": "hpr"})
    arow = sp[sp.team == sp.away][key + ["point", "over_price"]].rename(columns={"point": "ap", "over_price": "apr"})
    s2 = sp.drop_duplicates(key).drop(columns=["point", "over_price", "under_price", "player", "team"]) \
        .merge(hrow, on=key).merge(arow, on=key)
    s2 = s2[s2.hp == -s2.ap].copy()
    # X = 1H home margin; "over" at x = -hp means the home side covers
    s2["x"] = -s2.hp
    s2["over_price"], s2["under_price"] = s2.hpr, s2.apr
    s2["side_team"] = "home"
    s2 = s2.drop(columns=["hp", "ap", "hpr", "apr"])
    tt = d[d.market == "team_totals"].copy()
    tt["side_team"] = np.where(tt.team == tt.home, "home", np.where(tt.team == tt.away, "away", None))
    tt = tt[tt.side_team.notna()]
    th = d[d.market == "totals_h1"].copy()
    th["side_team"] = "game"
    q = pd.concat([s2, tt.rename(columns={"point": "x"}), th.rename(columns={"point": "x"})], ignore_index=True)
    q = q.drop(columns=[c for c in ("player", "team", "point") if c in q.columns])
    n0 = len(q)
    ok = (q.over_price.between(-300, 250) & q.under_price.between(-300, 250)
          & (q.over_price.abs() >= 100) & (q.under_price.abs() >= 100))
    q = q[ok].copy()
    q["overround"] = imp(q.over_price) + imp(q.under_price)
    q = q[(q.overround > 1.0) & (q.overround < 1.15)].copy()
    q["q_over"] = imp(q.over_price) / q.overround
    q["mk"] = q.market + ":" + q.side_team
    med = q.groupby(["event_id", "requested_ts", "mk"]).x.transform("median")
    q = q[(q.x - med).abs() <= 3.5].copy()
    q.attrs["dropped"] = n0 - len(q)
    return q


def attach_fg(q: pd.DataFrame, fg: pd.DataFrame, g: pd.DataFrame, max_gap_h=6.0) -> pd.DataFrame:
    """Attach the most recent full-game fair snapshot at or before (<= +10 min) each derivative snapshot."""
    snaps = q[["event_id", "requested_ts"]].drop_duplicates()
    f = fg[["event_id", "requested_ts", "S_all", "S_sharp", "T_all", "T_sharp", "n_fg_books"]].rename(
        columns={"requested_ts": "fg_ts"})
    m = snaps.merge(f, on="event_id")
    m["gap_h"] = (m.requested_ts - m.fg_ts).dt.total_seconds() / 3600
    m = m[(m.gap_h >= -10 / 60) & (m.gap_h <= max_gap_h)]
    mS = m[m.S_all.notna()].sort_values("gap_h").drop_duplicates(["event_id", "requested_ts"])
    mT = m[m.T_all.notna()].sort_values("gap_h").drop_duplicates(["event_id", "requested_ts"])
    snaps = snaps.merge(mS[["event_id", "requested_ts", "S_all", "S_sharp", "gap_h", "n_fg_books"]],
                        on=["event_id", "requested_ts"], how="left")
    snaps = snaps.merge(mT[["event_id", "requested_ts", "T_all", "T_sharp", "gap_h"]].rename(columns={"gap_h": "gap_T"}),
                        on=["event_id", "requested_ts"], how="left")
    q = q.merge(snaps, on=["event_id", "requested_ts"], how="left")
    return _match(q, g)


def dist_for(fair: Fair, mk: str) -> KeyDist:
    return fair.dm1 if mk.startswith("spreads_h1") else fair.dp1 if mk.startswith("totals_h1") else fair.dx


def fg_mean(fair: Fair, mk, S, T):
    if mk.startswith("spreads_h1"):
        return fair.m1(S, T)
    if mk.startswith("totals_h1"):
        return fair.p1(S, T)
    return fair.team((1.0 if mk.endswith("home") else -1.0) * np.asarray(S, float), T)


def implied(fair: Fair, mk, x, q_over, scale):
    d = dist_for(fair, mk)
    x, q_over = np.asarray(x, float), np.asarray(q_over, float)
    sc = np.round(np.nan_to_num(np.asarray(scale, float), nan=44.0)) if d.kind == "margin" else np.full(len(x), 44.0)
    out = np.full(len(x), np.nan)
    for v in np.unique(sc):
        s = sc == v
        out[s] = d.implied_mean(x[s], q_over[s], v)
    return out


def probs(fair: Fair, mk, mean, x, scale):
    """P(X > x), P(X == x), P(X < x) when the derivative's mean is `mean` (vectorized; one market)."""
    d = dist_for(fair, mk)
    mean, x, scale = (np.asarray(v, float) for v in (mean, x, scale))
    gt = np.full(len(mean), np.nan)
    eq = np.full(len(mean), np.nan)
    sc = np.round(np.nan_to_num(scale, nan=44.0)) if d.kind == "margin" else np.full(len(mean), 44.0)
    ok = np.isfinite(mean)
    for v in np.unique(sc[ok]):
        s = ok & (sc == v)
        gt[s], eq[s] = d.p_gt_eq(mean[s], x[s], v)
    return gt, eq, 1 - gt - eq


def price_rows(q: pd.DataFrame, fair: Fair) -> pd.DataFrame:
    q = q.copy()
    q["scale"] = q.T_all.fillna(44.0).round()
    q["S_ref"] = q.S_sharp.fillna(q.S_all)
    q["T_ref"] = q.T_sharp.fillna(q.T_all)
    for c in ("imean", "fg_mean", "fg_mean_sh"):
        q[c] = np.nan
    for mk in q.mk.unique():
        s = (q.mk == mk).values
        z = q[s]
        q.loc[s, "imean"] = implied(fair, mk, z.x.values, z.q_over.values, z.scale.values)
        q.loc[s, "fg_mean"] = fg_mean(fair, mk, z.S_all.values, z.T_all.values)
        q.loc[s, "fg_mean_sh"] = fg_mean(fair, mk, z.S_ref.values, z.T_ref.values)
    key = ["event_id", "requested_ts", "mk"]
    q["dcons"] = q.groupby(key).imean.transform("median")
    q["n_dbooks"] = q.groupby(key).book.transform("nunique")
    q["dcons_ex"] = np.nan   # consensus excluding the quoting book (leave-one-out median)
    for k, idx in q.groupby(key).indices.items():
        v = q.imean.values[idx]
        if len(v) > 1:
            q.iloc[idx, q.columns.get_loc("dcons_ex")] = [np.median(np.delete(v, i)) for i in range(len(v))]
    bo = q[q.book == "betonlineag"][key + ["imean"]].rename(columns={"imean": "dsharp"})
    return q.merge(bo, on=key, how="left")


def build(seasons, fair: Fair) -> pd.DataFrame:
    """Quotes with implied means, FG fair, close consensus of the same market, and outcomes."""
    p = SCR / f"quotes_{'_'.join(map(str, seasons))}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    g = load_games()
    g = g[g.season.isin(seasons)]
    q = load_derivs(seasons)
    fg = fg_fair(seasons)
    q = attach_fg(q, fg, g)
    q = price_rows(q, fair)
    c = q[q.snap == "close"].groupby(["game_id", "mk"]).agg(
        close_cons=("imean", "median"), close_sharp=("dsharp", "first"), close_scale=("scale", "first"),
        close_n=("book", "nunique"), close_S=("S_all", "first"), close_T=("T_all", "first"),
        close_fg_mean=("fg_mean", "first"), close_fg_mean_sh=("fg_mean_sh", "first")).reset_index()
    q = q.merge(c, on=["game_id", "mk"], how="left")
    h = history()
    q = q.merge(h[["game_id", "m1", "p1", "h1h", "h1a", "home_score", "away_score"]], on="game_id", how="left")
    q["X"] = np.where(q.mk.str.startswith("spreads_h1"), q.m1,
                      np.where(q.mk.str.startswith("totals_h1"), q.p1,
                               np.where(q.mk == "team_totals:home", q.home_score, q.away_score)))
    q.to_parquet(p)
    return q


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if stage == "fit":
        print(json.dumps(_js(fit_stage()), indent=1)[:6000])
