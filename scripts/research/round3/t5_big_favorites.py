"""Test 5: college big favorites (PRE-DECLARED 2026-10-07; R12).

R12 Every completed college game with a closing spread of 21 points or more (|spread_close| >= 21), 2014-2025,
    regular season + bowls, all divisions with a CFBD line: back the UNDERDOG ATS at the close, -110.
Note: the FBS-vs-FBS version of this rule was already tested and failed (factor_screen.py F7, 51.3% on 1,347 bets);
this run adds FBS-vs-FCS games and reports by era, so it is a re-test on a superset, not new information on F7's slice.
Eras: 2014-16, 2017-19, 2020-22, 2023-25 (descriptive), and the pass eras 2014-19 / 2020-25 (both must clear 52.38%).
Descriptive splits: FBS v FBS, FBS v FCS, spread 21-27.5 / 28-34.5 / 35+, dog at home/away/neutral, by season.
2026 to date separate. CLV (2021+): rule applied to the CFBD open (|spread_open| >= 21, dog from the open), CLV = points
the close moved toward the dog.
Bonferroni: 12 rules, one-sided p < 0.00417 vs 52.38% (see common.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402


def main():
    c = C.cfb_frame()
    c = c[c.spread_close.notna()].copy()
    # dog = home when spread_close > 0 (home line, + = home underdog)
    c["dog_home"] = c.spread_close > 0
    c["dog_line"] = c.spread_close.abs()
    c["dog_margin"] = np.where(c.dog_home, c.margin, -c.margin)
    c["res"] = np.sign(c.dog_margin + c.dog_line)
    c["fbs2"] = (c.home_div == "fbs") & (c.away_div == "fbs")
    c["ffcs"] = ((c.home_div == "fbs") & (c.away_div == "fcs")) | ((c.home_div == "fcs") & (c.away_div == "fbs"))
    s = c[c.dog_line >= 21].copy()
    hist = s[s.season <= 2025]
    full = C.grade(hist.res.values)
    eras = {e: C.grade(d.res.values) for e, d in hist.groupby(hist.season.map(C.era))}
    oos = C.grade(s[s.season == 2026].res.values)
    v = C.verdict(full, eras)
    e4 = hist.season.map(lambda y: "2014-16" if y <= 2016 else "2017-19" if y <= 2019 else "2020-22" if y <= 2022 else "2023-25")
    rows = [("2014-2025", full), *eras.items(), ("2026 to date (oos)", oos)]
    rows += [(k, C.grade(d.res.values)) for k, d in hist.groupby(e4)]
    loc = np.where(hist.neutral, "neutral", np.where(hist.dog_home, "home", "away"))
    rows += [("FBS v FBS", C.grade(hist[hist.fbs2].res.values)), ("FBS v FCS", C.grade(hist[hist.ffcs].res.values)),
             ("21-27.5", C.grade(hist[hist.dog_line < 28].res.values)),
             ("28-34.5", C.grade(hist[hist.dog_line.between(28, 34.5)].res.values)),
             ("35+", C.grade(hist[hist.dog_line >= 35].res.values)),
             ("dog at home", C.grade(hist[loc == "home"].res.values)), ("dog away", C.grade(hist[loc == "away"].res.values)),
             ("neutral", C.grade(hist[loc == "neutral"].res.values))]
    rows += [(str(y), C.grade(d.res.values)) for y, d in hist.groupby("season")]
    o = c[c.season.between(2021, 2025) & c.spread_open.notna() & (c.spread_open.abs() >= 21)].copy()
    o["dog_home_o"] = o.spread_open > 0
    o["clv"] = np.where(o.dog_home_o, o.spread_close - o.spread_open, o.spread_open - o.spread_close)  # + = more points for dog
    dm = np.where(o.dog_home_o, o.margin, -o.margin)
    o["res_open"] = np.sign(dm + o.spread_open.abs())
    clv = {"n": len(o), "mean_clv_pts": float(o.clv.mean()), "pct_pos": float((o.clv > 0).mean()),
           "at_open": C.grade(o.res_open.values)}
    out = {"R12": {"full": full, "eras": eras, "2026": oos, "rows": dict(rows), "clv_open_2021_25": clv, "verdict": v,
                   "mean_dog_err": float((hist.dog_margin + hist.dog_line).mean())}}
    md = ["### R12 college underdog when the closing spread >= 21\n", C.table(rows),
          f"\nMean (dog margin + dog line) = {(hist.dog_margin + hist.dog_line).mean():+.2f} pts. Placed at the open 2021-25: "
          f"n={clv['n']}, mean CLV {clv['mean_clv_pts']:+.2f} pts (moved toward dog {100*clv['pct_pos']:.0f}%); graded at the open: "
          f"{C.fmt(clv['at_open'])}", f"\nVerdict R12: **{v}**"]
    C.save("t5_big_favorites", out)
    (C.OUT / "t5_big_favorites.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
