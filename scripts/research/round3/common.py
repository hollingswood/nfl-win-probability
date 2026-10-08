"""Round 3 rule screen: shared registry, data loading and grading. PRE-DECLARED 2026-10-07, before any results were run.

REGISTRY (12 betting rules; Bonferroni alpha = 0.05 / 12 = 0.00417, one-sided):
  R1  NFL kickoff-rule lag: OVER the closing total, regular-season weeks 1-4 of 2024 and 2025 (pooled).
  R2  College 2023 clock rule: UNDER the closing total, weeks 1-4 of 2023.
  R3  College 2024 two-minute warning: OVER the closing total, weeks 1-4 of 2024 (extra end-of-half stoppage, totals
      set off the lower 2023 base).
  R4  College service academies (Army, Navy, Air Force): UNDER the closing total, every game involving one, 2014-2025.
  R5  All triple-option teams (academies + Georgia Tech 2014-18, Georgia Southern 2014-20, New Mexico 2014-19,
      Kennesaw State 2015-24): UNDER the closing total, 2014-2025.
  R6  Power team vs Group-of-5 team, non-conference: back the POWER team ATS at the close, 2014-2025.
  R7  Same as R6, only when the power team is the FAVORITE.
  R8  Same as R6, only when the power team is the UNDERDOG.
  R9  FBS vs FCS: back the FBS team ATS at the close, 2014-2025 (two-sided p reported, since either side could win).
  R10 College weeks 0-2 (CFBD weeks 1-2; CFBD files week 0 games under week 1): UNDER when total_close >= 55.
  R11 Symmetric alternative: weeks 0-2, OVER when total_close <= 45.
  R12 College big favorites: back the UNDERDOG when the closing spread is >= 21 points, 2014-2025, all games with a line.

Power = SEC, Big Ten, Big 12, ACC, Pac-12 (2014-2023 only; the 2024-25 two-team Pac-12 is NOT power) + Notre Dame.
Group of 5 = American Athletic, Mountain West, Mid-American, Sun Belt, Conference USA. Other independents excluded.

Grading: closing lines; college prices are not in the CFBD feed, so -110 is assumed (win +0.909u, loss -1u); NFL uses the
nflverse closing over/under price when present, else -110. Pushes are excluded from cover % and count 0 in ROI.
Cover % = W / (W + L). Break-even at -110 = 52.38%.

p-values: one-sided exact binomial on W out of W+L, (a) vs 52.38% (beats the vig = the betting test) and (b) vs 50%
(is there a market bias at all). R9 also gets a two-sided p vs 50%.

Pass (all of): p(a) < 0.00417; cover % > 52.38% in BOTH eras 2014-2019 and 2020-2025 (for single-season rules R1-R3:
in each season pooled, i.e. R1 needs 2024 and 2025 each > 52.38%). R1 is capped at LEAD regardless, because the
hypothesis was formed after seeing 2024-25 scoring; 2026 weeks 1-4 is its only clean out-of-sample data.
Lead: p(a) < 0.05 unadjusted, OR p(b) < 0.00417, with the same sign in both eras.  Fail: everything else.

Extra reporting (not part of pass/fail): by-season tables; CLV for 2021+ college rules re-applied at the OPEN
(CFBD opening number, rule condition evaluated on the open), CLV = points the close moved in the bet's favor.
Declared robustness check for any pass or lead from R4-R12: re-grade 2021-2025 at the Pinnacle close (cfb_pin snapshots,
actual prices). 2026 to date is reported separately as out-of-sample data for every rule.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cfbpred import data as D  # noqa: E402

OUT = ROOT / "output" / "research" / "round3"
OUT.mkdir(parents=True, exist_ok=True)
N_RULES = 12
ALPHA = 0.05 / N_RULES
BE = 110 / 210  # 52.38%

POWER = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
G5 = {"American Athletic", "Mountain West", "Mid-American", "Sun Belt", "Conference USA"}
ACADEMIES = {"Army", "Navy", "Air Force"}
OPTION_EXTRA = {"Georgia Tech": range(2014, 2019), "Georgia Southern": range(2014, 2021),
                "New Mexico": range(2014, 2020), "Kennesaw State": range(2015, 2025)}


def dec(price):
    price = float(price)
    return 1 + price / 100 if price > 0 else 1 + 100 / abs(price)


def grade(res: np.ndarray, price=None) -> dict:
    """res: +1 win, -1 loss, 0 push. price: American odds per bet (None -> -110)."""
    res = np.asarray(res, float)
    if price is None:
        price = np.full(len(res), -110.0)
    price = np.where(pd.isna(price), -110.0, np.asarray(price, float))
    pay = np.array([dec(p) - 1 for p in price])
    units = np.where(res > 0, pay, np.where(res < 0, -1.0, 0.0))
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    out = {"bets": int(len(res)), "w": w, "l": l, "push": p, "cover": w / n if n else float("nan"),
           "roi": units.sum() / len(res) if len(res) else float("nan"), "units": float(units.sum())}
    out["p_be"] = binomtest(w, n, BE, alternative="greater").pvalue if n else float("nan")
    out["p_50"] = binomtest(w, n, 0.5, alternative="greater").pvalue if n else float("nan")
    out["p_two"] = binomtest(w, n, 0.5, alternative="two-sided").pvalue if n else float("nan")
    return out


def era(season: int) -> str:
    return "2014-19" if season <= 2019 else ("2020-25" if season <= 2025 else "2026")


def verdict(full: dict, eras: dict, cap_lead=False) -> str:
    """eras: {label: grade dict} that must all agree."""
    ok_eras = all(e["cover"] > BE for e in eras.values() if e["w"] + e["l"] > 0)
    same_sign = all(e["cover"] > 0.5 for e in eras.values()) or all(e["cover"] < 0.5 for e in eras.values())
    if full["p_be"] < ALPHA and ok_eras and not cap_lead:
        return "pass"
    if (full["p_be"] < 0.05 or full["p_50"] < ALPHA) and same_sign and all(e["cover"] > 0.5 for e in eras.values()):
        return "lead"
    if cap_lead and full["p_be"] < ALPHA and ok_eras:
        return "lead (capped: hypothesis formed on this data)"
    return "fail"


def fmt(g: dict) -> str:
    return (f"{g['bets']} | {g['w']}-{g['l']}-{g['push']} | {100*g['cover']:.1f}% | {100*g['roi']:+.1f}% | "
            f"{g['p_be']:.3g} | {g['p_50']:.3g}")


HDR = "| slice | bets | W-L-P | cover % | ROI | p vs 52.38% | p vs 50% |\n|---|---|---|---|---|---|---|"


def table(rows: list[tuple[str, dict]]) -> str:
    return HDR + "\n" + "\n".join(f"| {k} | {fmt(g)} |" for k, g in rows if g["bets"])


def cfb_frame(years=range(2014, 2027)) -> pd.DataFrame:
    """Completed college games with a closing spread or total (preferred real-book close)."""
    g = D.games(years)
    L = D.lines(years, prefer=D.PREFERRED_PROVIDERS)
    m = g.merge(L, on="game_id", how="inner")
    m = m[m.completed & m.margin.notna()].copy()
    return m


def ou_result(total, line, side: str) -> np.ndarray:
    d = np.asarray(total, float) - np.asarray(line, float)
    r = np.sign(d)
    return r if side == "over" else -r


def save(name: str, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if math.isnan(o) else float(o)
        if isinstance(o, float) and math.isnan(o):
            return None
        raise TypeError(type(o))
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=conv))
