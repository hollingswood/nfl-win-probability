"""Test 1: rule-change scoring lag (PRE-DECLARED 2026-10-07, before running; rules R1-R3 in common.py).

Hypothesis (general rule to pre-register for the future): after a scoring-relevant rule change, posted totals lag the
new scoring level for the first weeks, so bet OVERS (rule raises scoring) or UNDERS (rule lowers scoring) in weeks 1-4
until totals catch up.

Rules (closing totals; NFL priced at the nflverse closing over price, -110 if missing; college at -110):
  R1  NFL: OVER, regular-season weeks 1-4 of 2024 and 2025 (dynamic kickoff 2024, touchback to the 35 in 2025).
      Capped at LEAD: formed after seeing 2024-25 scoring. 2026 weeks 1-4 (to date) = out-of-sample.
  R2  College 2023 (clock keeps running after first downs): UNDER, CFBD weeks 1-4 of 2023.
  R3  College 2024 (two-minute warning added): OVER, CFBD weeks 1-4 of 2024.
Data: data/raw/games.parquet (NFL, closing lines) and cfbpred.data (college, preferred real-book close).
Holdout logic: there is no tuning; each rule is a single fixed slice. Comparison context: the same week-bucket slices in
every other season (NFL 2018-2026, college 2014-2026) are reported so the reader can see whether the rule-change seasons
stand out from ordinary early-season noise. Exploratory (labelled): NFL over rate weeks 1-4 in every season 1999-2026.
Pass criteria: see common.py (p vs 52.38% < 0.00417 and each season in the slice > 52.38%).
Buckets: NFL weeks 1-4 / 5-9 / 10-18 (regular season only); college weeks 1-4 / 5-9 / 10+ (regular season).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402


def nfl():
    g = pd.read_parquet(C.ROOT / "data/raw/games.parquet")
    g = g[(g.game_type == "REG") & g.total_line.notna() & g.home_score.notna()].copy()
    g["actual"] = g.home_score + g.away_score
    g["bucket"] = pd.cut(g.week, [0, 4, 9, 18], labels=["1-4", "5-9", "10-18"])
    g["res"] = C.ou_result(g.actual, g.total_line, "over")
    return g


def bucket_table(df, season_col, price_col=None, side="over"):
    rows = []
    for (s, b), d in df.groupby([season_col, "bucket"], observed=True):
        gr = C.grade(d.res.values if side == "over" else -d.res.values, None if price_col is None else d[price_col].values)
        rows.append({"season": int(s), "bucket": str(b), "games": len(d), "mean_close_total": d.line.mean(),
                     "mean_actual": d.actual.mean(), "mean_diff": (d.actual - d.line).mean(),
                     "over_pct": gr["cover"], "over_roi": gr["roi"]})
    return pd.DataFrame(rows)


def main():
    out, md = {}, []
    g = nfl()
    g["line"] = g.total_line
    bt = bucket_table(g[g.season >= 2018], "season", "over_odds")
    out["nfl_buckets"] = bt.to_dict("records")
    md.append("### NFL over rate by season and week bucket (closing totals, regular season)\n")
    md.append("| season | bucket | games | mean close total | mean actual | actual - close | over % | over ROI |\n|---|---|---|---|---|---|---|---|")
    for r in bt.itertuples():
        md.append(f"| {r.season} | {r.bucket} | {r.games} | {r.mean_close_total:.1f} | {r.mean_actual:.1f} | {r.mean_diff:+.1f} | "
                  f"{100*r.over_pct:.1f}% | {100*r.over_roi:+.1f}% |")

    # R1
    s = g[g.season.isin([2024, 2025]) & (g.week <= 4)]
    r1 = C.grade(s.res.values, s.over_odds.values)
    per = {str(y): C.grade(d.res.values, d.over_odds.values) for y, d in s.groupby("season")}
    s26 = g[(g.season == 2026) & (g.week <= 4)]
    r1_26 = C.grade(s26.res.values, s26.over_odds.values)
    # comparison: weeks 1-4 OVER in 2018-2023 (no rule change) pooled
    base = g[g.season.between(2018, 2023) & (g.week <= 4)]
    r1_base = C.grade(base.res.values, base.over_odds.values)
    v = C.verdict(r1, per, cap_lead=True)
    out["R1"] = {"pooled": r1, "by_season": per, "2026": r1_26, "weeks1_4_2018_2023": r1_base, "verdict": v}
    md.append("\n### R1 NFL OVER weeks 1-4, 2024+2025\n")
    md.append(C.table([("2024+2025 pooled", r1), *[(k, x) for k, x in per.items()], ("2026 wk 1-4 (out of sample)", r1_26),
                       ("context: 2018-2023 wk 1-4", r1_base)]))
    md.append(f"\nVerdict R1: **{v}**")

    # Exploratory: weeks 1-4 over rate every season since 1999
    ex = []
    for y, d in g[g.week <= 4].groupby("season"):
        gr = C.grade(d.res.values, d.over_odds.values)
        ex.append({"season": int(y), "games": len(d), "over_pct": gr["cover"], "mean_diff": float((d.actual - d.line).mean())})
    out["nfl_wk1_4_all_seasons_exploratory"] = ex
    exd = pd.DataFrame(ex)
    md.append("\nExploratory: NFL weeks 1-4 over % by season 1999-2026: " + ", ".join(
        f"{r.season} {100*r.over_pct:.0f}% ({r.mean_diff:+.1f})" for r in exd.itertuples()))
    sd = exd[exd.season < 2024].over_pct.std()
    md.append(f"\nSeason-to-season SD of weeks 1-4 over % 1999-2023: {100*sd:.1f} pts "
              f"(mean {100*exd[exd.season < 2024].over_pct.mean():.1f}%).")

    # College
    c = C.cfb_frame()
    c = c[(c.season_type == "regular") & c.total_close.notna()].copy()
    c["actual"] = c.total
    c["line"] = c.total_close
    c["bucket"] = pd.cut(c.week, [0, 4, 9, 30], labels=["1-4", "5-9", "10+"])
    c["res"] = C.ou_result(c.actual, c.line, "over")
    ct = bucket_table(c, "season")
    out["cfb_buckets"] = ct.to_dict("records")
    md.append("\n### College over rate by season and week bucket (closing totals, regular season, -110)\n")
    md.append("| season | bucket | games | mean close total | mean actual | actual - close | over % | over ROI |\n|---|---|---|---|---|---|---|---|")
    for r in ct.itertuples():
        md.append(f"| {r.season} | {r.bucket} | {r.games} | {r.mean_close_total:.1f} | {r.mean_actual:.1f} | {r.mean_diff:+.1f} | "
                  f"{100*r.over_pct:.1f}% | {100*r.over_roi:+.1f}% |")

    s = c[(c.season == 2023) & (c.week <= 4)]
    r2 = C.grade(-s.res.values)
    v2 = C.verdict(r2, {"2023": r2})
    s = c[(c.season == 2024) & (c.week <= 4)]
    r3 = C.grade(s.res.values)
    v3 = C.verdict(r3, {"2024": r3})
    base_u = c[(c.season.between(2014, 2022)) & (c.week <= 4)]
    out["R2"] = {"g": r2, "verdict": v2, "context_under_wk1_4_2014_2022": C.grade(-base_u.res.values)}
    out["R3"] = {"g": r3, "verdict": v3}
    md.append("\n### R2 college 2023 UNDER weeks 1-4 / R3 college 2024 OVER weeks 1-4\n")
    md.append(C.table([("R2 2023 wk 1-4 under", r2), ("context: under wk 1-4 2014-2022", out["R2"]["context_under_wk1_4_2014_2022"]),
                       ("R3 2024 wk 1-4 over", r3)]))
    md.append(f"\nVerdict R2: **{v2}**; R3: **{v3}**")

    # General rule, pooled over the three instances (descriptive; each bet in its declared direction)
    s1 = g[g.season.isin([2024, 2025]) & (g.week <= 4)]
    pooled = np.concatenate([s1.res.values, -c[(c.season == 2023) & (c.week <= 4)].res.values,
                             c[(c.season == 2024) & (c.week <= 4)].res.values])
    out["general_pooled"] = C.grade(pooled)
    md.append(f"\nGeneral rule pooled over the 3 instances (-110): {C.fmt(out['general_pooled'])}")
    C.save("t1_rule_change_lag", out)
    (C.OUT / "t1_rule_change_lag.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
