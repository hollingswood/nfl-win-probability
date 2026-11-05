"""Round 7: published-picker consensus (pre-registered in output/research/round7/consensus_prereg.json).

Does the majority side of several published pickers cover the CLOSING line more than half the time, and how do
opposing picks change it? One row per (sport, game, class): class 'side' pools spread + moneyline picks by team,
'total' = over/under. Output: output/research/round7/consensus.json + consensus_rows.parquet + consensus.md.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "round6"))
OUT = ROOT / "output" / "research" / "round7"
SPLIT = "2026-10-20"


def binom_p2(w: int, n: int) -> float:
    """Two-sided normal-approx binomial p vs 50%."""
    if n == 0:
        return 1.0
    z = (w - n / 2) / math.sqrt(n / 4)
    return math.erfc(abs(z) / math.sqrt(2))


def closes():
    """game_id -> (home, away, home_score, away_score, home-favored spread, total) at the close."""
    from nflpred import picks_score as PS
    out = {}
    n = PS.nfl_games()
    for r in n.itertuples():
        if pd.notna(r.home_score):
            out[str(r.game_id)] = (r.home_team, r.away_team, float(r.home_score), float(r.away_score),
                                   float(r.spread_line) if pd.notna(r.spread_line) else None,
                                   float(r.total_line) if pd.notna(r.total_line) else None)
    c = PS.cfb_games()
    for r in c.itertuples():
        if bool(r.completed) and pd.notna(r.home_pts):
            out[str(r.game_id)] = (r.home, r.away, float(r.home_pts), float(r.away_pts),
                                   -float(r.spread_close) if pd.notna(r.spread_close) else None,
                                   float(r.total_close) if pd.notna(r.total_close) else None)
    return out


def units(graded: list[dict]) -> pd.DataFrame:
    by = defaultdict(lambda: defaultdict(dict))         # (sport, game, class) -> direction -> picker -> first seen
    meta = {}
    for g in graded:
        cls = "total" if g["market"] == "total" else "side"
        k = (g["sport"], str(g["game_id"]), cls)
        d = by[k][g["side"]]
        if g["picker"] not in d or g["seen_at"] < d[g["picker"]]:
            d[g["picker"]] = g["seen_at"]
        meta[k] = g["kick"]
    rows = []
    for (sport, gid, cls), dirs in by.items():
        # a picker on both directions of one unit (spread one way, ML the other is already removed by the scorer)
        both = set.intersection(*[set(v) for v in dirs.values()]) if len(dirs) == 2 else set()
        cnt = {d: {p: t for p, t in v.items() if p not in both} for d, v in dirs.items()}
        ds = sorted(cnt, key=lambda d: -len(cnt[d]))
        F = len(cnt[ds[0]]); A = len(cnt[ds[1]]) if len(ds) > 1 else 0
        if F == A or F == 0:
            continue
        times = sorted(cnt[ds[0]].values())
        rows.append({"sport": sport, "game_id": gid, "cls": cls, "dir": ds[0], "F": F, "A": A, "kick": meta[(sport, gid, cls)],
                     "t_F": times[-1], "t_first": times[0], "pickers_for": sorted(cnt[ds[0]])})
    return pd.DataFrame(rows)


def grade(df: pd.DataFrame, cl: dict) -> pd.DataFrame:
    res = []
    for r in df.itertuples():
        c = cl.get(r.game_id)
        if not c:
            res.append(None); continue
        home, away, hs, as_, hml, tot = c
        if r.cls == "total":
            if tot is None:
                res.append(None); continue
            d = (hs + as_ - tot) * (1 if r.dir == "over" else -1)
        else:
            if hml is None or r.dir not in (home, away):
                res.append(None); continue
            pm = (hs - as_) if r.dir == home else (as_ - hs)
            d = pm + (-hml if r.dir == home else hml)
        res.append("push" if d == 0 else "win" if d > 0 else "loss")
    df = df.assign(result=res)
    return df[df.result.isin(["win", "loss"])].assign(cover=lambda x: (x.result == "win").astype(int))


def fb(F):
    return "1" if F == 1 else "2" if F == 2 else "3" if F == 3 else "4-5" if F <= 5 else "6+"


def ab(A):
    return "0" if A == 0 else "1" if A == 1 else "2+"


def cell_stats(s: pd.DataFrame) -> dict:
    n, w = len(s), int(s.cover.sum())
    return {"n": n, "w": w, "rate": round(w / n, 4) if n else None, "p": round(binom_p2(w, n), 4),
            "roi_at_110": round((w * 100 / 110 - (n - w)) / n, 4) if n else None}


def main():
    from pick_steam import picks_scored
    graded, _ = picks_scored()
    df = grade(units(graded), closes())
    df["net"] = df.F - df.A
    df["period"] = np.where(df.kick < SPLIT, "dev", "holdout")
    import statsmodels.api as sm
    m = sm.OLS(df.cover, sm.add_constant(df[["net"]])).fit(cov_type="HC1")
    t1 = {"n": len(df), "slope_per_net_picker": round(float(m.params["net"]), 4), "t": round(float(m.tvalues["net"]), 2),
          "p_two_sided": round(float(m.pvalues["net"]), 4), "intercept": round(float(m.params["const"]), 4),
          "overall_majority_cover": cell_stats(df), "pass": bool(m.pvalues["net"] < 0.05)}
    df["Fb"], df["Ab"] = df.F.map(fb), df.A.map(ab)
    cells = []
    for key, s in df.groupby(["sport", "cls", "Fb", "Ab"]):
        c = {"sport": key[0], "class": key[1], "F": key[2], "A": key[3], **cell_stats(s),
             "dev": cell_stats(s[s.period == "dev"]), "holdout": cell_stats(s[s.period == "holdout"])}
        cells.append(c)
    big = [c for c in cells if c["n"] >= 20]
    bonf = 0.05 / max(1, len(big))
    for c in cells:
        rep = (c["dev"]["n"] and c["holdout"]["n"] and c["dev"]["p"] < 0.05 and c["holdout"]["p"] < 0.05
               and (c["dev"]["rate"] - 0.5) * (c["holdout"]["rate"] - 0.5) > 0)
        c["lead"] = bool((c["n"] >= 20 and c["p"] < bonf) or rep)
    # coarser views people ask about (descriptive)
    views = {}
    for name, f in {"unanimous F>=3 A=0": (df.F >= 3) & (df.A == 0), "unanimous F>=5 A=0": (df.F >= 5) & (df.A == 0),
                    "F>=3 any A": df.F >= 3, "net>=3": df.net >= 3, "net>=5": df.net >= 5, "share>=0.8 F>=4": (df.F / (df.F + df.A) >= 0.8) & (df.F >= 4),
                    "contested A>=2": df.A >= 2, "lone pick F=1 A=0": (df.F == 1) & (df.A == 0)}.items():
        views[name] = {sp: cell_stats(df[f & (df.sport == sp)]) for sp in ("nfl", "cfb")} | {"all": cell_stats(df[f])}
    out = {"prereg": "output/research/round7/consensus_prereg.json", "units_graded": len(df),
           "by_sport_class": {f"{a} {b}": cell_stats(s) for (a, b), s in df.groupby(["sport", "cls"])},
           "T1": t1, "bonferroni_alpha": round(bonf, 5), "cells_n20": len(big), "leads": [c for c in cells if c["lead"]],
           "views": views, "cells": cells}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "consensus.json").write_text(json.dumps(out, indent=1))
    df.drop(columns=["pickers_for"]).to_parquet(OUT / "consensus_rows.parquet")
    print(json.dumps({k: out[k] for k in ("units_graded", "by_sport_class", "T1", "bonferroni_alpha", "cells_n20", "leads", "views")}, indent=1))


if __name__ == "__main__":
    main()
