"""Round robin grid (pre-registered: output/research/round5/round_robin_grid_prereg.json, 2026-10-16):
every combination of k = 2..6 legs from each week's top-N picks, edge legs and ordinary legs.

Run: PYTHONPATH=src python scripts/research/round5/round_robin_grid.py -> output/research/round5/round_robin_grid.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]
import parlays as PL  # noqa: E402

OUT = ROOT / "output" / "research" / "round5"
NS = (4, 5, 6, 7, 8, 9, 10, "all")
KS = (2, 3, 4, 5, 6, "2-6")
PERIODS = {"development_2021_22": (2021, 2022), "holdout_2023_25": (2023, 2024, 2025)}


def combo_results(d: np.ndarray, o: np.ndarray, c: np.ndarray, ev: np.ndarray, k: int):
    """All k-combinations of legs from different games -> (profit per parlay, CLV-implied return per parlay)."""
    idx = np.array(list(itertools.combinations(range(len(d)), k)), dtype=np.int32)
    if not len(idx):
        return np.array([]), np.array([])
    e = ev[idx]
    keep = np.all(np.diff(np.sort(e, axis=1), axis=1) != 0, axis=1)        # never two legs from the same game
    idx = idx[keep]
    if not len(idx):
        return np.array([]), np.array([])
    O, Dd = o[idx], d[idx]
    lost = (O < 0).any(axis=1)
    pay = np.prod(np.where(O > 0, Dd, 1.0), axis=1)
    prof = np.where(lost, -1.0, pay - 1.0)
    clv = np.prod(1 + c[idx], axis=1) - 1
    return prof, clv


def week_rows(legs: pd.DataFrame, n, k) -> list[dict]:
    rows = []
    for s, g in legs.groupby("slate"):
        g = g.sort_values("rank")
        top = g if n == "all" else g.drop_duplicates("ev_id").head(n)
        if n != "all" and len(top) < n:
            continue
        d, o, c = top.d.values, top.o.values, top.clv.values
        ev = pd.factorize(top.ev_id)[0]
        if k == "singles":
            prof, clv = np.where(o > 0, d - 1, np.where(o < 0, -1.0, 0.0)), c
        else:
            ks = range(2, 7) if k == "2-6" else [k]
            P, C = [], []
            for kk in ks:
                if kk <= len(top):
                    p_, c_ = combo_results(d, o, c, ev, kk); P.append(p_); C.append(c_)
            prof, clv = (np.concatenate(P), np.concatenate(C)) if P else (np.array([]), np.array([]))
        if len(prof):
            rows.append({"slate": s, "season": int(g.season.iloc[0]), "ret": float(prof.mean()), "clv_ret": float(clv.mean()),
                         "n_bets": int(len(prof)), "hit": float((prof > 0).mean())})
    return rows


def summarize(rows: list[dict]) -> dict:
    w = pd.DataFrame(rows)
    if not len(w):
        return {}
    cum = w.ret.cumsum()
    return {"weeks": int(len(w)), "bets_per_week": round(float(w.n_bets.mean()), 1), "roi": round(float(w.ret.mean()), 4),
            "clv_implied_roi": round(float(w.clv_ret.mean()), 4), "parlays_cashed": round(float(w.hit.mean()), 3),
            "losing_weeks": round(float((w.ret < 0).mean()), 3), "worst_week": round(float(w.ret.min()), 3),
            "max_drawdown_weeks_of_stake": round(float((cum.cummax() - cum).max()), 2)}


def grid(legs: pd.DataFrame, ns) -> dict:
    out = {}
    for lab, yrs in PERIODS.items():
        x = legs[legs.season.isin(yrs)]
        res = {}
        for n in ns:
            res[f"N={n}"] = {"singles": summarize(week_rows(x, n, "singles"))}
            for k in KS:
                if k == "2-6" or n == "all" or k <= n:
                    res[f"N={n}"][f"by {k}"] = summarize(week_rows(x, n, k))
        out[lab] = res
    verdict = {}
    for n in out["holdout_2023_25"]:
        for k, h in out["holdout_2023_25"][n].items():
            if k == "singles" or not h:
                continue
            ok = True
            for lab in PERIODS:
                a, s = out[lab][n].get(k) or {}, out[lab][n]["singles"]
                ok &= bool(a and s and a["roi"] > s["roi"] and a["clv_implied_roi"] > s["clv_implied_roi"]
                           and a["max_drawdown_weeks_of_stake"] <= 1.5 * s["max_drawdown_weeks_of_stake"])
            verdict[f"{n} {k}"] = ok
    out["passes"] = [k for k, v in verdict.items() if v]
    out["formats_tested"] = len(verdict)
    return out


def edge() -> pd.DataFrame:
    d = PL.edge_legs()
    d["rank"] = -d.ev
    d["ev_id"] = d.event_id
    return d


def ordinary(max_n: int = 10) -> pd.DataFrame:
    c = PL.closing_legs()["college"].copy()
    rng = np.random.default_rng(7)
    c["o"] = np.sign(c.res.values * (rng.integers(0, 2, len(c)) * 2 - 1)).astype(int)
    c["d"], c["clv"] = PL.DEC110, -0.0455          # a -110 leg at a fair 50% close: CLV-implied -4.55%
    c["rank"] = rng.random(len(c))                  # random N games per week
    c["slate"] = c.season.astype(str) + "-" + c.wk.astype(str)
    c["ev_id"] = c.game_id
    return c


def main():
    E = edge()
    out = {"prereg": "output/research/round5/round_robin_grid_prereg.json", "edge_legs": grid(E, NS),
           "ordinary_legs": grid(ordinary(), (4, 6, 8, 10))}
    (OUT / "round_robin_grid.json").write_text(json.dumps(out, indent=1, default=str))
    for which in ("edge_legs", "ordinary_legs"):
        print("=====", which, "passes:", out[which]["passes"], "of", out[which]["formats_tested"])
        for n, r in out[which]["holdout_2023_25"].items():
            print(n, " | ".join(f"{k}: {v.get('roi', 0):+.1%} roi, {v.get('clv_implied_roi', 0):+.1%} clv, dd {v.get('max_drawdown_weeks_of_stake')}, lose {v.get('losing_weeks')}, {v.get('bets_per_week')}/wk"
                              for k, v in r.items() if v))


if __name__ == "__main__":
    main()
