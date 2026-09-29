"""Key-number-aware distribution of NFL final margins.

A normal curve around the predicted margin treats every point as equally likely. Real NFL margins
pile up on 3, 7, 10, 6, 14, 4 (field goals and touchdowns) and almost never end tied. We keep the
normal curve for the overall shape and multiply each integer margin by a key-number weight
w(|k|) = (how often games actually ended at |k|) / (how often the normal curve said they would),
estimated on training seasons only. Probabilities are then renormalized over integers.

Used for: spread cover probabilities, push probabilities on whole-number lines, the value of
buying half-points, and P(win) including the small chance of a tie.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KMAX = 60
KS = np.arange(-KMAX, KMAX + 1)


def _normal_pmf(mu, sigma):
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    z = (KS[None, :] - mu) / sigma
    p = np.exp(-0.5 * z * z)
    return p / p.sum(axis=1, keepdims=True)


def fit_key_weights(mu, actual, sigma, smooth=5.0, cap=(0.05, 3.0)) -> dict[int, float]:
    """Weights by |margin|, from training games only. `smooth` = pseudo-games pulling toward 1."""
    base = _normal_pmf(mu, sigma)
    expected = np.zeros(KMAX + 1)
    for j, k in enumerate(KS):
        expected[abs(k)] += base[:, j].sum()
    act = np.bincount(np.clip(np.abs(np.asarray(actual, int)), 0, KMAX), minlength=KMAX + 1)
    w = (act + smooth) / (expected + smooth)
    return {int(k): float(np.clip(w[k], *cap)) for k in range(KMAX + 1)}


def pmf(mu, sigma, weights: dict | None):
    p = _normal_pmf(mu, sigma)
    if weights:
        wv = np.array([weights.get(abs(int(k)), 1.0) for k in KS])
        p = p * wv[None, :]
        p = p / p.sum(axis=1, keepdims=True)
    return p


def win_prob(mu, sigma, weights=None, tie_value=0.5):
    """P(home wins) counting a tie as half (moneyline ties are rare and usually refunded)."""
    p = pmf(mu, sigma, weights)
    return p[:, KS > 0].sum(axis=1) + tie_value * p[:, KS == 0].sum(axis=1)


def cover_probs(mu, sigma, home_line, weights=None):
    """For a home spread `home_line` (e.g. -3.5 = home favored by 3.5): returns
    (P(home covers), P(push), P(away covers)). Home covers if margin + line > 0."""
    p = pmf(mu, sigma, weights)
    line = np.atleast_1d(np.asarray(home_line, float))[:, None]
    adj = KS[None, :] + line
    return ((p * (adj > 0)).sum(axis=1), (p * (adj == 0)).sum(axis=1), (p * (adj < 0)).sum(axis=1))


def save(weights: dict, sigma: float, path: Path, meta: dict):
    path.write_text(json.dumps({"sigma": sigma, "weights": weights, **meta}, indent=1))


def load(path: Path):
    d = json.loads(path.read_text())
    return float(d["sigma"]), {int(k): v for k, v in d["weights"].items()}
