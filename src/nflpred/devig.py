"""Vig removal for two-way prices. Multiplicative (proportional) is the simplest and the least accurate on
lopsided moneylines: it hands the underdog too much. Shin and power correct for the favorite-longshot bias.
On Pinnacle closing moneylines (output/research/devig_methods.md) multiplicative had the worst log loss in NFL and CFB."""
from __future__ import annotations

import math


def implied(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def multiplicative(h: float, a: float) -> float:
    x, y = implied(h), implied(a)
    return x / (x + y)


def shin(h: float, a: float) -> float:
    """Home no-vig probability, Shin (1993) with the insider share z solved numerically."""
    return shin_from_implied(implied(h), implied(a))


def shin_from_implied(x: float, y: float) -> float:
    """Shin no-vig probability of the first outcome, from the two implied probabilities."""
    s = x + y
    if s <= 1:
        return x / s
    def p(q, z):
        return (math.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z))
    lo, hi = 0.0, 0.5
    for _ in range(60):
        z = (lo + hi) / 2
        if p(x, z) + p(y, z) > 1:
            lo = z
        else:
            hi = z
    z = (lo + hi) / 2
    ph, pa = p(x, z), p(y, z)
    return ph / (ph + pa)


def power(h: float, a: float) -> float:
    x, y = implied(h), implied(a)
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if x ** k + y ** k > 1:
            lo = k
        else:
            hi = k
    return x ** ((lo + hi) / 2)


METHODS = {"multiplicative": multiplicative, "shin": shin, "power": power}
