"""Re-scores round_robin_more / round_robin_grid with singles on EXACTLY the same weeks as each round robin (the
pre-registered comparison is 'the same picks as singles'; the first pass compared 'all picks' round robins, which only
exist in weeks with >= k picks, against singles over every week). Same pass rule otherwise.

Run: same env as round_robin_more.py -> output/research/round5/round_robin_matched.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path(__file__).parent)]
import round_robin_grid as RG  # noqa: E402
import round_robin_more as RM  # noqa: E402


def matched(x: pd.DataFrame, ns) -> dict:
    out, passes, n_f = {}, [], 0
    for n in ns:
        for k in RG.KS:
            if not (k == "2-6" or n == "all" or k <= n):
                continue
            n_f += 1
            res, ok = {}, True
            for lab, yrs in RG.PERIODS.items():
                xx = x[x.season.isin(yrs)]
                f = pd.DataFrame(RG.week_rows(xx, n, k))
                if not len(f):
                    ok = False; continue
                s = pd.DataFrame(RG.week_rows(xx[xx.slate.isin(set(f.slate))], n, "singles"))
                a, b = RG.summarize(f.to_dict("records")), RG.summarize(s.to_dict("records"))
                res[lab] = {"format": a, "singles_same_weeks": b}
                ok &= bool(a["roi"] > b["roi"] and a["clv_implied_roi"] > b["clv_implied_roi"]
                           and a["max_drawdown_weeks_of_stake"] <= 1.5 * b["max_drawdown_weeks_of_stake"])
            out[f"N={n} by {k}"] = res
            if ok:
                passes.append(f"N={n} by {k}")
    return {"formats": n_f, "passes": passes, "detail": out}


def main():
    L = pd.concat([pd.read_parquet(p) for p in sorted(RM.CACHE.glob("*.parquet"))], ignore_index=True)
    res = {}
    for name, srcs in {**RM.POOLS, "0 College shop vs sharp (first test)": ["college_aplus", "college_shop_A"]}.items():
        res[name] = matched(RM.pool(L, srcs), RG.NS)
        print(name, "passes", res[name]["passes"], "of", res[name]["formats"])
    (RM.OUT / "round_robin_matched.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
