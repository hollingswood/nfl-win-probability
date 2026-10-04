"""College key-number distributions for pricing any spread or total from the sharp line.

Margin (home points minus away points) given the expected margin mu:
    P(M = k) ∝ φ((k − mu) / s(mu)) · w(|k|),   s(mu) = a + b·|mu|,  w(0) = 0 (no ties in college)
Total given the expected total mu_t:
    P(T = t) ∝ φ((t − mu_t) / s_t(mu_t)) · v(t)
The weights w (key numbers 3, 7, 10, 14, 17, 21, 24, 28 ...) and v are fit on 2014-2021 closing lines
(scripts/research/cfb/dist_fit.py) and stored in cfb_dist.json. Pricing a quote:
  1. the sharp book's two prices -> no-vig win share q at its number (pushes excluded),
  2. solve for mu so the model reproduces q at that number,
  3. any other book's number/price -> P(win), P(push), P(lose) -> EV per unit staked.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "cfb_dist.json"
KM = np.arange(-120, 121)          # margins
KT = np.arange(0, 181)             # totals


@lru_cache(maxsize=1)
def params() -> dict:
    return json.loads(PARAMS.read_text())


_WCACHE: dict = {}


def _w(P: dict) -> np.ndarray:
    k = ("w", tuple(P["w"].items()))
    if k in _WCACHE:
        return _WCACHE[k]
    w = np.ones(len(KM))
    for k, x in P["w"].items():
        w[np.abs(KM) == int(k)] = x
    w[KM == 0] = 0.0
    _WCACHE[k] = w
    return w


def _v(P: dict) -> np.ndarray:
    k = ("v", tuple(P["v"].items()))
    if k in _WCACHE:
        return _WCACHE[k]
    v = np.ones(len(KT))
    for kk, x in P["v"].items():
        v[KT == int(kk)] = x
    _WCACHE[k] = v
    return v


def margin_pmf(mu: float, P: dict | None = None) -> np.ndarray:
    P = P or params()
    s = P["a"] + P["b"] * abs(mu)
    f = np.exp(-0.5 * ((KM - mu) / s) ** 2) * _w(P)
    return f / f.sum()


def total_pmf(mu: float, P: dict | None = None) -> np.ndarray:
    P = P or params()
    s = P["at"] + P["bt"] * mu
    f = np.exp(-0.5 * ((KT - mu) / s) ** 2) * _v(P)
    return f / f.sum()


def spread_probs(mu: float, home_point: float, side: str = "home", P=None) -> tuple[float, float, float]:
    """(win, push, lose) for a bet on `side` where the HOME line is `home_point` (home -7 -> -7)."""
    pm = margin_pmf(mu, P)
    x = KM + home_point                    # home covers if margin + home_point > 0
    hw, push = pm[x > 0].sum(), pm[x == 0].sum()
    hl = 1 - hw - push
    return (hw, push, hl) if side == "home" else (hl, push, hw)


def total_probs(mu: float, point: float, side: str = "over", P=None) -> tuple[float, float, float]:
    pt = total_pmf(mu, P)
    o, push = pt[KT > point].sum(), pt[KT == point].sum()
    u = 1 - o - push
    return (o, push, u) if side == "over" else (u, push, o)


def _solve(f, target: float, lo: float, hi: float) -> float:
    """f increasing in x; bisection."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def implied(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def decimal(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def mu_from_spread(home_point: float, home_price: float, away_price: float, P=None) -> float:
    """Expected home margin implied by a two-sided spread quote (no-vig, pushes excluded)."""
    ph, pa = implied(home_price), implied(away_price)
    q = ph / (ph + pa)
    def share(mu):
        w, p, l = spread_probs(mu, home_point, "home", P)
        return w / (w + l)
    return _solve(share, q, -80, 80)


def mu_from_total(point: float, over_price: float, under_price: float, P=None) -> float:
    po, pu = implied(over_price), implied(under_price)
    q = po / (po + pu)
    def share(mu):
        o, p, u = total_probs(mu, point, "over", P)
        return o / (o + u)
    return _solve(share, q, 10, 120)


def mu_from_ml(home_price: float, away_price: float, P=None) -> float:
    ph, pa = implied(home_price), implied(away_price)
    q = ph / (ph + pa)
    return _solve(lambda mu: margin_pmf(mu, P)[KM > 0].sum(), q, -80, 80)


def ev(win: float, push: float, lose: float, american: float) -> float:
    return win * (decimal(american) - 1) - lose


if __name__ == "__main__":
    P = params()
    for mu in (0, 3, 7, 14):
        pm = margin_pmf(mu, P)
        print(mu, {k: round(float(pm[KM == k][0]), 3) for k in (3, 7, 10, 14)})
