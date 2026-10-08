"""Test 2: service academies and triple-option teams, UNDER the closing total (PRE-DECLARED 2026-10-07; R4, R5).

R4  Every game involving Army, Navy or Air Force with a closing total, 2014-2025 (regular + postseason). One bet per game
    (academy vs academy counted once). Bet UNDER the closing total at -110.
R5  Same, for all triple-option teams: academies (all years) + Georgia Tech 2014-18 (Paul Johnson), Georgia Southern
    2014-20, New Mexico 2014-19 (Bob Davie), Kennesaw State 2015-24 (Brian Bohannon; only games with a line).
Data: cfbpred.data games + lines (preferred real-book close, -110 assumed). Eras: 2014-2019 vs 2020-2025 (both must
clear 52.38% to pass); 2026 to date reported separately as out-of-sample.
Breakdowns (descriptive, not extra rules): by season; academy vs academy vs academy vs others; by team; by total size.
CLV (2021+): the same rule placed at the CFBD OPENING total; CLV = open total - close total (points in the under's favor).
Pass/lead/fail and Bonferroni (12 rules, one-sided p < 0.00417 vs 52.38%): see common.py.
If R4 or R5 is a pass or lead: re-grade 2021-2025 at the Pinnacle close with actual under prices (t2b_pinnacle.py,
written only in that case, with the method fixed here: last Pinnacle snapshot before kickoff with a total).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402


def is_option(team, season, academies_only):
    if team in C.ACADEMIES:
        return True
    if academies_only:
        return False
    yrs = C.OPTION_EXTRA.get(team)
    return yrs is not None and season in yrs


def run_rule(c, academies_only, label, md, out):
    m = c.apply(lambda r: is_option(r.home, r.season, academies_only) or is_option(r.away, r.season, academies_only), axis=1)
    s = c[m].copy()
    both = s.apply(lambda r: is_option(r.home, r.season, academies_only) and is_option(r.away, r.season, academies_only), axis=1)
    s["res"] = C.ou_result(s.total, s.total_close, "under")
    hist = s[s.season <= 2025]
    full = C.grade(hist.res.values)
    eras = {e: C.grade(d.res.values) for e, d in hist.groupby(hist.season.map(C.era))}
    oos = C.grade(s[s.season == 2026].res.values)
    v = C.verdict(full, eras)
    rows = [("2014-2025", full), *eras.items(), ("2026 to date (oos)", oos),
            ("vs each other", C.grade(hist[both.loc[hist.index]].res.values)),
            ("vs others", C.grade(hist[~both.loc[hist.index]].res.values)),
            ("close total < 45", C.grade(hist[hist.total_close < 45].res.values)),
            ("close total >= 45", C.grade(hist[hist.total_close >= 45].res.values))]
    for y, d in hist.groupby("season"):
        rows.append((str(y), C.grade(d.res.values)))
    teams = sorted(set(C.ACADEMIES) | (set() if academies_only else set(C.OPTION_EXTRA)))
    for t in teams:
        d = hist[(hist.home == t) | (hist.away == t)]
        d = d[d.apply(lambda r: is_option(t, r.season, academies_only), axis=1)]
        rows.append((f"team: {t}", C.grade(d.res.values)))
    # CLV at the open, 2021+
    o = s[(s.season >= 2021) & (s.season <= 2025) & s.total_open.notna()].copy()
    o["clv"] = o.total_open - o.total_close
    o["res_open"] = C.ou_result(o.total, o.total_open, "under")
    clv = {"n": len(o), "mean_clv_pts": float(o.clv.mean()), "pct_pos": float((o.clv > 0).mean()),
           "pct_neg": float((o.clv < 0).mean()), "at_open": C.grade(o.res_open.values)}
    out[label] = {"full": full, "eras": eras, "2026": oos, "rows": {k: g for k, g in rows}, "clv_open_2021_25": clv,
                  "mean_close_total": float(hist.total_close.mean()), "mean_actual": float(hist.total.mean()), "verdict": v}
    md.append(f"\n### {label}\n")
    md.append(C.table(rows))
    md.append(f"\nMean closing total {hist.total_close.mean():.1f}, mean actual {hist.total.mean():.1f}. "
              f"Placed at the open 2021-25: n={clv['n']}, mean CLV {clv['mean_clv_pts']:+.2f} pts "
              f"(close below open {100*clv['pct_pos']:.0f}%, above {100*clv['pct_neg']:.0f}%); graded at the open: {C.fmt(clv['at_open'])}")
    md.append(f"\nVerdict {label}: **{v}**")


def main():
    c = C.cfb_frame()
    c = c[c.total_close.notna()].copy()
    md, out = [], {}
    run_rule(c, True, "R4 academies UNDER", md, out)
    run_rule(c, False, "R5 all triple-option UNDER", md, out)
    C.save("t2_option_unders", out)
    (C.OUT / "t2_option_unders.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
