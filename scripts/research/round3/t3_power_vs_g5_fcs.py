"""Test 3: power conference vs Group of 5, and FBS vs FCS, ATS at the close (PRE-DECLARED 2026-10-07; R6-R9).

R6  Non-conference games (conf_game False), one team power, the other Group of 5 (definitions in common.py; conference
    labels are CFBD's per-season labels, so realignment is handled by season): back the POWER team ATS at the closing
    spread, 2014-2025, regular season + bowls, -110.
R7  R6 restricted to games where the power team is the closing favorite.
R8  R6 restricted to games where the power team is the closing underdog.
    (Pick'em closes go to neither R7 nor R8.)
R9  FBS vs FCS (home_div/away_div): back the FBS team ATS at the close, 2014-2025. Two-sided p vs 50% reported; the FCS
    side's record is the mirror image.
Eras 2014-19 / 2020-25 (both must clear 52.38% to pass); 2026 to date separately. Breakdowns by season, by spread size
and home/away/neutral are descriptive only.
CLV (2021+): same rule at the CFBD opening spread (side chosen from the open), CLV = points the close moved toward the bet.
Bonferroni: 12 rules, one-sided p < 0.00417 vs 52.38% (see common.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402


def is_power(conf, team, season):
    if team == "Notre Dame":
        return True
    if conf == "Pac-12" and season >= 2024:
        return False
    return conf in C.POWER


def ats(margin_for_team, team_line):
    """team_line: the team's spread (negative = favored). Result +1 cover / -1 / 0."""
    return np.sign(np.asarray(margin_for_team, float) + np.asarray(team_line, float))


def section(label, s, md, out, two_sided=False, line_col="team_line"):
    hist = s[s.season <= 2025]
    full = C.grade(hist.res.values)
    eras = {e: C.grade(d.res.values) for e, d in hist.groupby(hist.season.map(C.era))}
    oos = C.grade(s[s.season == 2026].res.values)
    v = C.verdict(full, eras)
    rows = [("2014-2025", full), *eras.items(), ("2026 to date (oos)", oos)]
    for lab, mask in [("line within 7", hist[line_col].abs() <= 7), ("7.5-14", hist[line_col].abs().between(7.5, 14)),
                      ("14.5-24", hist[line_col].abs().between(14.5, 24)), ("24.5+", hist[line_col].abs() >= 24.5),
                      ("bet team at home", hist.loc_bet == "home"), ("bet team away", hist.loc_bet == "away"),
                      ("neutral", hist.loc_bet == "neutral")]:
        rows.append((lab, C.grade(hist[mask].res.values)))
    for y, d in hist.groupby("season"):
        rows.append((str(y), C.grade(d.res.values)))
    o = s[(s.season.between(2021, 2025)) & s.open_line.notna()].copy()
    clv = {"n": len(o), "mean_clv_pts": float(o.clv.mean()) if len(o) else None,
           "pct_pos": float((o.clv > 0).mean()) if len(o) else None, "at_open": C.grade(o.res_open.values)}
    out[label] = {"full": full, "eras": eras, "2026": oos, "rows": dict(rows), "clv_open_2021_25": clv, "verdict": v,
                  "mean_err_vs_close": float(hist.err.mean())}
    md.append(f"\n### {label}\n")
    md.append(C.table(rows))
    extra = f" Two-sided p vs 50%: {full['p_two']:.3g}." if two_sided else ""
    md.append(f"\nMean (bet team's margin + its closing line) = {hist.err.mean():+.2f} pts.{extra} "
              f"Placed at the open 2021-25: n={clv['n']}, mean CLV {clv['mean_clv_pts']:+.2f} pts "
              f"(moved toward the bet {100*clv['pct_pos']:.0f}%); graded at the open: {C.fmt(clv['at_open'])}")
    md.append(f"\nVerdict {label}: **{v}**")


def build(c, team_is_home: pd.Series):
    """Rows from the perspective of the team to bet."""
    s = c.copy()
    h = team_is_home.values
    s["team_line"] = np.where(h, s.spread_close, -s.spread_close)
    s["open_line"] = np.where(h, s.spread_open, -s.spread_open)
    s["team_margin"] = np.where(h, s.margin, -s.margin)
    s["res"] = ats(s.team_margin, s.team_line)
    s["err"] = s.team_margin + s.team_line
    s["res_open"] = ats(s.team_margin, s.open_line)
    s["clv"] = s.open_line - s.team_line  # bet got open_line; close moved to team_line; + = we got more points
    s["loc_bet"] = np.where(s.neutral, "neutral", np.where(h, "home", "away"))
    return s


def main():
    c = C.cfb_frame()
    c = c[c.spread_close.notna()].copy()
    md, out = [], {}
    hp = c.apply(lambda r: is_power(r.home_conf, r.home, r.season), axis=1)
    ap = c.apply(lambda r: is_power(r.away_conf, r.away, r.season), axis=1)
    hg = c.home_conf.isin(C.G5) & (c.home_div == "fbs")
    ag = c.away_conf.isin(C.G5) & (c.away_div == "fbs")
    nc = ~c.conf_game
    pg = nc & ((hp & ag) | (ap & hg))
    s = build(c[pg], hp[pg])
    # For CLV, side for R7/R8 chosen from the open would differ; keep side fixed (power team) — it is the same team.
    section("R6 power team vs G5 (all)", s, md, out)
    section("R7 power team as favorite", s[s.team_line < 0], md, out)
    section("R8 power team as underdog", s[s.team_line > 0], md, out)
    ff = ((c.home_div == "fbs") & (c.away_div == "fcs")) | ((c.home_div == "fcs") & (c.away_div == "fbs"))
    s = build(c[ff], (c.home_div == "fbs")[ff])
    section("R9 FBS vs FCS: back FBS", s, md, out, two_sided=True)
    C.save("t3_power_vs_g5_fcs", out)
    (C.OUT / "t3_power_vs_g5_fcs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
