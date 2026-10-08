"""Test 4: college early season (PRE-DECLARED 2026-10-07; R10, R11).

Week definition: CFBD files "week 0" games under week 1, so weeks 0-2 = CFBD regular-season weeks 1-2.
Descriptive (not bets): closing-line accuracy by week bucket (1-2, 3-5, 6-9, 10+): mean absolute error of the margin vs
the closing spread and of the total vs the closing total, mean (actual - close total), over %, for 2014-2019 and
2020-2025; also restricted to FBS vs FBS.
R10 Weeks 0-2, total_close >= 55: UNDER at the close (-110), all games with a closing total, 2014-2025.
R11 Symmetric alternative, weeks 0-2, total_close <= 45: OVER at the close (-110).
Eras 2014-19 / 2020-25 (both must clear 52.38% to pass); 2026 to date separate. Descriptive splits: FBS vs FBS only,
by season. CLV (2021+): same rule at the CFBD open total (threshold applied to the open), CLV = points in our favor.
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
    c = c[c.season_type == "regular"].copy()
    c["bucket"] = pd.cut(c.week, [0, 2, 5, 9, 30], labels=["1-2", "3-5", "6-9", "10+"])
    c["fbs2"] = (c.home_div == "fbs") & (c.away_div == "fbs")
    md, out = [], {}
    md.append("### Closing-line accuracy by week bucket (regular season)\n")
    md.append("| era | games | bucket | spread MAE | total MAE | actual - close total | over % |\n|---|---|---|---|---|---|---|")
    acc = []
    for scope, d0 in [("all", c), ("FBS v FBS", c[c.fbs2])]:
        for e, d1 in d0[d0.season <= 2025].groupby(d0.season.map(C.era)):
            for b, d in d1.groupby("bucket", observed=True):
                sp = d[d.spread_close.notna()]
                to = d[d.total_close.notna()]
                ov = C.grade(C.ou_result(to.total, to.total_close, "over"))
                r = {"scope": scope, "era": e, "bucket": str(b), "n_spread": len(sp), "n_total": len(to),
                     "spread_mae": float((sp.margin + sp.spread_close).abs().mean()),
                     "total_mae": float((to.total - to.total_close).abs().mean()),
                     "total_bias": float((to.total - to.total_close).mean()), "over_pct": ov["cover"]}
                acc.append(r)
                md.append(f"| {scope} {e} | {len(sp)}/{len(to)} | {b} | {r['spread_mae']:.2f} | {r['total_mae']:.2f} | "
                          f"{r['total_bias']:+.2f} | {100*r['over_pct']:.1f}% |")
    out["accuracy"] = acc

    for label, side, cond, cond_open in [
            ("R10 weeks 0-2 UNDER, close total >= 55", "under", lambda d: d.total_close >= 55, lambda d: d.total_open >= 55),
            ("R11 weeks 0-2 OVER, close total <= 45", "over", lambda d: d.total_close <= 45, lambda d: d.total_open <= 45)]:
        e2 = c[(c.week <= 2) & c.total_close.notna()]
        s = e2[cond(e2)].copy()
        s["res"] = C.ou_result(s.total, s.total_close, side)
        hist = s[s.season <= 2025]
        full = C.grade(hist.res.values)
        eras = {e: C.grade(d.res.values) for e, d in hist.groupby(hist.season.map(C.era))}
        oos = C.grade(s[s.season == 2026].res.values)
        v = C.verdict(full, eras)
        rows = [("2014-2025", full), *eras.items(), ("2026 to date (oos)", oos),
                ("FBS v FBS only", C.grade(hist[hist.fbs2].res.values)),
                ("FBS v FCS", C.grade(hist[~hist.fbs2].res.values))]
        for y, d in hist.groupby("season"):
            rows.append((str(y), C.grade(d.res.values)))
        o = e2[(e2.season.between(2021, 2025)) & e2.total_open.notna()]
        o = o[cond_open(o)].copy()
        o["clv"] = (o.total_open - o.total_close) * (1 if side == "under" else -1)
        o["res_open"] = C.ou_result(o.total, o.total_open, side)
        clv = {"n": len(o), "mean_clv_pts": float(o.clv.mean()), "pct_pos": float((o.clv > 0).mean()),
               "at_open": C.grade(o.res_open.values)}
        out[label] = {"full": full, "eras": eras, "2026": oos, "rows": dict(rows), "clv_open_2021_25": clv, "verdict": v}
        md.append(f"\n### {label}\n")
        md.append(C.table(rows))
        md.append(f"\nPlaced at the open 2021-25 (threshold on the open): n={clv['n']}, mean CLV {clv['mean_clv_pts']:+.2f} pts "
                  f"(moved our way {100*clv['pct_pos']:.0f}%); graded at the open: {C.fmt(clv['at_open'])}")
        md.append(f"\nVerdict {label}: **{v}**")
    C.save("t4_cfb_early_season", out)
    (C.OUT / "t4_cfb_early_season.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
