"""Total-aware (and optionally spread-aware) NFL final-margin distribution.

    P(home margin = k | mu, total[, |spread|])  ∝  Normal(k; mu, sigma(total)) * w(|k|; total[, |spread|])

* mu = expected home margin implied by a spread AND its juice (inverted through this same distribution).
* sigma(total) = sigma0 + sigma_slope * (total - 44)   (sigma_slope = 0 -> one sigma for all games).
* w(|k|; z) = key-number weights that depend smoothly on the covariates z. Fit by *kernel raking*: for every node z*
  of a grid (totals 30..62 step 1; optional |spread| nodes), each training game gets kernel weight
  K_i = exp(-0.5 * sum_j ((z_ij - z*_j) / h_j)^2); weights are raked until the kernel-weighted model frequency of
  every |margin| matches the kernel-weighted actual frequency; then shrunk toward the pooled (all-games) weights in
  log space with factor E_k / (E_k + smooth), E_k = kernel-weighted expected count of |margin| = k (rare margins
  shrink more). Weights between nodes are interpolated linearly in log w.
* covariate "total" can be the posted total itself (mode='abs') or the total relative to the mean total of the
  previous 256 games' closing totals (mode='rel'; known before kickoff) -- the latter removes era drift in scoring.

Usage (after a fit has been stored, e.g. by ml_spread_consistency.py part1):
    import margin_by_total as MT
    M = MT.load()                        # reads output/research/ml_spread_consistency.json["part1"]["model"]
    mu = M.implied_mu(home_point, q_home_novig, total)
    pw, pp = M.side_probs(mu, side_is_home, point, total)
    p_home_win, p_tie = M.win_tie(mu, total)
All functions are vectorized over rows; distributions are cached per (total rounded to 0.5[, spread node]).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root (src/nflpred/..)
KS = np.arange(-60, 61)
AK = np.abs(KS)
GRID = np.round(np.arange(-30, 30.0001, 0.02), 4)
TNODES = np.arange(30.0, 62.0001, 1.0)
SNODES = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 7.0, 8.5, 10.5, 14.0])
T_REF = 44.0
STORE = ROOT / "margin_total_model.json"


# ------------------------------------------------------------------------------------------------ core pmf
def normal_rows(mu, sigma):
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    sigma = np.broadcast_to(np.atleast_1d(np.asarray(sigma, float)), mu.shape[:1])[:, None]
    p = np.exp(-0.5 * ((KS[None, :] - mu) / sigma) ** 2)
    return p / p.sum(1, keepdims=True)


def rake(base, kern, act_k, w0, iters=8, floor=(0.03, 5.0)):
    """base: (N,121) normal pmf rows; kern: (N,) kernel weights; act_k: (61,) kernel-weighted actual |k| counts.
    Returns symmetric weights (121,) such that sum_i kern_i * pmf_i(|k|) ~= act_k, plus expected counts."""
    w = w0.copy()
    for _ in range(iters):
        P = base * w[None, :]
        P /= P.sum(1, keepdims=True)
        e = np.zeros(61)
        np.add.at(e, AK, (kern[:, None] * P).sum(0))
        f = np.clip((act_k + 1e-3) / (e + 1e-3), 0.2, 5)
        w = np.clip(w * f[AK], *floor)
    P = base * w[None, :]
    P /= P.sum(1, keepdims=True)
    e = np.zeros(61)
    np.add.at(e, AK, (kern[:, None] * P).sum(0))
    return w, e


class _Dist:
    """One (sigma, w): grid lookups for cover / push / implied mu (same conventions as teasers_v2.Dist)."""

    def __init__(self, sigma, w):
        self.sigma, self.w = float(sigma), np.asarray(w, float)
        P = normal_rows(GRID, self.sigma) * self.w[None, :]
        self.P = P / P.sum(1, keepdims=True)
        self.cdf = np.cumsum(self.P, 1)
        self._q = {}

    def gi(self, mu):
        return np.clip(np.rint((np.asarray(mu, float) + 30) / 0.02).astype(int), 0, len(GRID) - 1)

    def p_gt(self, mu, x):
        gi = self.gi(mu)
        x = np.asarray(x, float)
        j = np.clip(np.floor(x + 1e-9).astype(int) + 60, -1, 120)
        return 1 - np.where(j >= 0, self.cdf[gi, np.clip(j, 0, 120)], 0.0)

    def p_eq(self, mu, x):
        gi = self.gi(mu)
        x = np.asarray(x, float)
        isint = np.abs(x - np.round(x)) < 1e-9
        return np.where(isint, self.P[gi, np.clip(np.round(x).astype(int) + 60, 0, 120)], 0.0)

    def implied_mu(self, home_point, q_home):
        home_point = np.asarray(home_point, float)
        q_home = np.asarray(q_home, float)
        out = np.full(len(home_point), np.nan)
        for L in np.unique(home_point[~np.isnan(home_point)]):
            if L not in self._q:
                hc = self.p_gt(GRID, np.full(len(GRID), -L))
                pu = self.p_eq(GRID, np.full(len(GRID), -L))
                self._q[L] = np.maximum.accumulate(hc / np.maximum(1 - pu, 1e-12))
            s = home_point == L
            out[s] = np.interp(q_home[s], self._q[L], GRID)
        return out


# ------------------------------------------------------------------------------------------------ model
class MarginModel:
    def __init__(self, params: dict):
        self.p = params
        self.mode = params.get("mode", "abs")           # abs | rel | none
        self.use_spread = bool(params.get("use_spread", False))
        self.sigma0 = float(params["sigma0"])
        self.sigma_slope = float(params.get("sigma_slope", 0.0))
        self.tnodes = np.asarray(params.get("tnodes", TNODES), float)
        self.snodes = np.asarray(params.get("snodes", SNODES), float)
        self.logw = np.asarray(params["logw"], float)   # (T, S, 121) or (T, 121) or (121,)
        if self.logw.ndim == 1:
            self.logw = self.logw[None, None, :]
        elif self.logw.ndim == 2:
            self.logw = self.logw[:, None, :]
        self._cache: dict = {}

    # covariates -------------------------------------------------------------------------
    def tcov(self, total, ref_total=None):
        """Covariate for the weight table. mode 'rel' needs ref_total (mean of recent closing totals)."""
        total = np.asarray(total, float)
        if self.mode == "rel":
            ref = np.asarray(ref_total if ref_total is not None else np.full(total.shape, T_REF), float)
            return np.nan_to_num(total - ref + T_REF, nan=T_REF)
        return np.nan_to_num(total, nan=T_REF)

    def sigma(self, total):
        t = np.nan_to_num(np.asarray(total, float), nan=T_REF)
        return self.sigma0 + self.sigma_slope * (np.clip(t, 34, 58) - T_REF)

    def weights_at(self, tc, sp=0.0):
        lw = self.logw
        ti = np.interp(tc, self.tnodes, np.arange(len(self.tnodes))) if lw.shape[0] > 1 else 0.0
        t0 = int(np.floor(ti)); t1 = min(t0 + 1, lw.shape[0] - 1); ft = ti - t0
        if lw.shape[1] > 1:
            si = np.interp(sp, self.snodes, np.arange(len(self.snodes)))
            s0 = int(np.floor(si)); s1 = min(s0 + 1, lw.shape[1] - 1); fs = si - s0
            a = (1 - fs) * lw[:, s0] + fs * lw[:, s1]
        else:
            a = lw[:, 0]
        return np.exp((1 - ft) * a[t0] + ft * a[t1])

    def dist(self, total_eff, sigma_total, sp=0.0) -> _Dist:
        key = (round(float(total_eff) * 2) / 2, round(float(sigma_total) * 2) / 2,
               round(float(sp) * 2) / 2 if self.use_spread else 0.0)
        d = self._cache.get(key)
        if d is None:
            d = _Dist(float(self.sigma(key[1])), self.weights_at(key[0], key[2]))
            self._cache[key] = d
        return d

    def _groups(self, total, ref_total=None, spread_abs=None):
        total = np.atleast_1d(np.asarray(total, float))
        tc = self.tcov(total, ref_total)
        tr = np.round(tc * 2) / 2
        sr = np.round(np.nan_to_num(np.nan_to_num(total, nan=T_REF)) * 2) / 2
        sp = (np.round(np.abs(np.nan_to_num(np.asarray(spread_abs, float))) * 2) / 2
              if (self.use_spread and spread_abs is not None) else np.zeros(len(total)))
        keys = pd.DataFrame({"t": tr, "s": sr, "p": sp})
        for (t, s, p), idx in keys.groupby(["t", "s", "p"]).indices.items():
            yield self.dist(t, s, p), idx

    # vectorized API ------------------------------------------------------------------------
    def implied_mu(self, home_point, q_home, total, ref_total=None, spread_abs=None):
        home_point = np.atleast_1d(np.asarray(home_point, float))
        q_home = np.atleast_1d(np.asarray(q_home, float))
        if spread_abs is None:
            spread_abs = np.abs(home_point)
        out = np.full(len(home_point), np.nan)
        for d, idx in self._groups(total, ref_total, spread_abs):
            out[idx] = d.implied_mu(home_point[idx], q_home[idx])
        return out

    def side_probs(self, mu_home, side_home, point, total, ref_total=None, spread_abs=None):
        """P(win), P(push) of a side getting `point` (+ = getting points). home wins iff m + point > 0."""
        mu_home = np.atleast_1d(np.asarray(mu_home, float))
        point = np.atleast_1d(np.asarray(point, float))
        side_home = np.atleast_1d(np.asarray(side_home, bool))
        pw = np.full(len(mu_home), np.nan)
        pp = np.full(len(mu_home), np.nan)
        for d, idx in self._groups(total, ref_total, spread_abs):
            m, pt, sh = mu_home[idx], point[idx], side_home[idx]
            pwh = d.p_gt(m, -pt)
            pwa = 1 - d.p_gt(m, pt) - d.p_eq(m, pt)
            pw[idx] = np.where(sh, pwh, pwa)
            pp[idx] = d.p_eq(m, np.where(sh, -pt, pt))
        return pw, pp

    def win_tie(self, mu_home, total, ref_total=None, spread_abs=None):
        """(P(home wins), P(tie)). Ties are ~0.2-0.3% after the 2017 OT change; the weights fit them directly."""
        mu_home = np.atleast_1d(np.asarray(mu_home, float))
        pw = np.full(len(mu_home), np.nan)
        pt = np.full(len(mu_home), np.nan)
        for d, idx in self._groups(total, ref_total, spread_abs):
            pw[idx] = d.p_gt(mu_home[idx], np.zeros(len(idx)))
            pt[idx] = d.p_eq(mu_home[idx], np.zeros(len(idx)))
        return pw, pt

    def pmf(self, mu_home, total, ref_total=None, spread_abs=None):
        """(N,121) pmf over KS for each row (exact, not grid-rounded in mu)."""
        mu_home = np.atleast_1d(np.asarray(mu_home, float))
        out = np.zeros((len(mu_home), len(KS)))
        for d, idx in self._groups(total, ref_total, spread_abs):
            P = normal_rows(mu_home[idx], d.sigma) * d.w[None, :]
            out[idx] = P / P.sum(1, keepdims=True)
        return out


# ------------------------------------------------------------------------------------------------ fitting
def fit(mu, m, total, sigma0, sigma_slope=0.0, h=None, smooth=50.0, mode="abs", ref_total=None,
        spread_abs=None, h_spread=None, iters=8) -> dict:
    """Fit the weight table on training games. mu = price-implied expected home margin, m = actual home margin,
    total = closing total. h = kernel bandwidth in total points (None -> no total dependence).
    h_spread = bandwidth in |spread| (None -> no spread dependence)."""
    mu, m, total = (np.asarray(x, float) for x in (mu, m, total))
    shell = MarginModel({"sigma0": sigma0, "sigma_slope": sigma_slope, "mode": mode, "logw": np.zeros(121)})
    tc = shell.tcov(total, ref_total)
    sig = shell.sigma(total)
    base = normal_rows(mu, sig)
    am = np.clip(np.abs(m).astype(int), 0, 60)
    ones = np.ones(len(mu))
    act = np.bincount(am, minlength=61).astype(float)
    w0, _ = rake(base, ones, act, np.ones(121), iters=iters)
    lw0 = np.log(w0)
    params = {"sigma0": sigma0, "sigma_slope": sigma_slope, "mode": mode, "h": h, "smooth": smooth,
              "h_spread": h_spread, "use_spread": h_spread is not None, "tnodes": TNODES.tolist(),
              "snodes": SNODES.tolist(), "n_fit": int(len(mu))}
    if h is None and h_spread is None:
        params["logw"] = lw0.tolist()
        return params
    tn = TNODES if h is not None else np.array([T_REF])
    sn = SNODES if h_spread is not None else np.array([0.0])
    sa = np.abs(np.asarray(spread_abs, float)) if spread_abs is not None else np.zeros(len(mu))
    out = np.zeros((len(tn), len(sn), 121))
    for a, t in enumerate(tn):
        kt = np.exp(-0.5 * ((tc - t) / h) ** 2) if h is not None else ones
        for b, s in enumerate(sn):
            k = kt * (np.exp(-0.5 * ((sa - s) / h_spread) ** 2) if h_spread is not None else 1.0)
            if k.sum() < 1:
                out[a, b] = lw0
                continue
            actk = np.bincount(am, weights=k, minlength=61)
            w, e = rake(base, k, actk, w0, iters=iters)
            lam = e / (e + smooth)
            out[a, b] = lam[AK] * np.log(w) + (1 - lam[AK]) * lw0
    if h is None:
        params["tnodes"] = [T_REF]
    params["logw"] = out.tolist() if h_spread is not None else out[:, 0, :].tolist()
    return params


def loglik(model: MarginModel, mu, m, total, ref_total=None, spread_abs=None) -> np.ndarray:
    P = model.pmf(mu, total, ref_total, spread_abs)
    return np.log(np.maximum(P[np.arange(len(m)), np.clip(np.asarray(m, int), -60, 60) + 60], 1e-12))


def ref_totals(g: pd.DataFrame, n=256) -> pd.Series:
    """Mean closing total of the previous n games (by kickoff order) -- known before each game; for 'rel' mode."""
    s = g.sort_values(["gameday", "game_id"])
    r = s.total_line.shift(1).rolling(n, min_periods=50).mean()
    return r.reindex(g.index).fillna(s.total_line.expanding().mean().shift(1).reindex(g.index)).fillna(T_REF)


def load(path: Path = STORE, key=()) -> MarginModel:
    d = json.loads(Path(path).read_text())
    for k in key:
        d = d[k]
    return MarginModel(d)


def american_to_dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def american_to_imp(a):
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
