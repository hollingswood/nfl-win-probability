"""POST-HOC sensitivity for cfb_shop_segments.py (written AFTER its results were seen; not a pre-declared test).

Why: the EV >= 10% bets (42 in 2021-25, 79% FanDuel, mostly 2022 totals 3-13 points off Pinnacle) are high-leverage
points for T11 (EV -> CLV slope) and T13 (season trend). This re-runs those two estimates without them, to show
whether the pre-declared verdicts depend on a handful of probable stale/erroneous soft-book quotes.
Output: output/research/round3/cfb_shop_segments_sensitivity.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import cfb_shop_segments as S  # noqa: E402

SB = pd.read_parquet(S.ROOT / "output" / "research" / "cfb" / "shop_bets.parquet")
out = {}
B = pd.concat([SB[SB.rule == r] for r in S.FROZEN.values()])
B = B[B.season.between(*S.DEV)].dropna(subset=["clv"])
for lab, d in (("all", B), ("ev_below_10pct", B[B.ev < 0.10])):
    X = np.hstack([pd.get_dummies(d.market).to_numpy(float), (d.season.to_numpy(float) - 2023)[:, None]])
    b, V = S.ols_cl(d.clv.to_numpy(float), X, d.event_id.to_numpy())
    se = math.sqrt(V[-1, -1])
    out[f"T13_{lab}"] = {"n": int(len(d)), "slope_per_season": round(float(b[-1]), 4), "se": round(se, 4),
                         "p_two_sided": float(2 * norm.sf(abs(b[-1] / se))),
                         "clv_by_season": {int(k): round(float(v), 4) for k, v in d.groupby("season").clv.mean().items()}}
C = pd.concat([SB[SB.rule == r] for r in S.CAL_RULES])
C = C[C.season.between(*S.DEV) & (C.ev < 0.10)].dropna(subset=["clv"])
for mk, g in C.groupby("market"):
    X = np.column_stack([np.ones(len(g)), g.ev.to_numpy(float)])
    b, V = S.ols_cl(g.clv.to_numpy(float), X, g.event_id.to_numpy())
    out[f"T11_ev_below_10pct_{mk}"] = {"n": int(len(g)), "intercept": round(float(b[0]), 4), "slope": round(float(b[1]), 3),
                                       "slope_se": round(math.sqrt(V[1, 1]), 3)}
hi = B[B.ev >= 0.10]
out["ev_10pct_plus_frozen_bets"] = {"n": int(len(hi)), "by_book": hi.book.value_counts().to_dict(),
                                    "by_season": {int(k): int(v) for k, v in hi.season.value_counts().items()}}
(S.OUT / "cfb_shop_segments_sensitivity.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
