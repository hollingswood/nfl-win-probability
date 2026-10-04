"""Fit college key-number distributions (cfbpred.dist) on 2014-2021 closing lines; check calibration on 2022-2025.
Closing lines: CFBD consensus close (output/research/cfb/preds.parquet; spread_close = home line, + = home underdog)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cfbpred import dist as DI  # noqa: E402

OUT = ROOT / "output" / "research" / "cfb"


def nll_margin(P, mu, m):
    tot = 0.0
    for u, k in zip(mu, m):
        pm = DI.margin_pmf(u, P)
        tot -= np.log(max(pm[DI.KM == k][0], 1e-12))
    return tot


def fit(d, t):
    P = {"a": 15.0, "b": 0.0, "w": {}, "at": 16.0, "bt": 0.0, "v": {}}
    mu, m = (-d.spread_close).to_numpy(), d.margin.astype(int).to_numpy()
    mut, tt = t.total_close.to_numpy(), t.total.astype(int).to_numpy()
    for it in range(6):
        # scale: grid search a, b
        best = None
        for a in np.arange(13.0, 18.01, 0.5):
            for b in (0.0, 0.02, 0.04, 0.06, 0.08, 0.1):
                Q = dict(P, a=float(a), b=float(b))
                s = nll_margin(Q, mu[::3], m[::3])
                if best is None or s < best[0]:
                    best = (s, a, b)
        P["a"], P["b"] = float(best[1]), float(best[2])
        # key weights: observed / expected count of |margin| = k (k <= 45; beyond: 1)
        exp = np.zeros(len(DI.KM))
        for u in mu:
            exp += DI.margin_pmf(u, P)
        w = {}
        for k in range(1, 46):
            o = int((np.abs(m) == k).sum())
            e = exp[np.abs(DI.KM) == k].sum()
            w[str(k)] = float(P["w"].get(str(k), 1.0) * (o + 2) / (e + 2))
        P["w"] = w
        # totals
        best = None
        for a in np.arange(8.0, 20.01, 1.0):
            for b in (0.0, 0.05, 0.1, 0.15, 0.2):
                Q = dict(P, at=float(a), bt=float(b))
                s = 0.0
                for u, k in zip(mut[::3], tt[::3]):
                    pt = DI.total_pmf(u, Q)
                    s -= np.log(max(pt[DI.KT == k][0], 1e-12))
                if best is None or s < best[0]:
                    best = (s, a, b)
        P["at"], P["bt"] = float(best[1]), float(best[2])
        expt = np.zeros(len(DI.KT))
        for u in mut:
            expt += DI.total_pmf(u, P)
        v = {}
        for k in range(20, 121):
            o = int((tt == k).sum())
            v[str(k)] = float(P["v"].get(str(k), 1.0) * (o + 3) / (expt[DI.KT == k].sum() + 3))
        P["v"] = v
        print(it, "a,b", P["a"], P["b"], "at,bt", P["at"], P["bt"], {k: round(w[k], 2) for k in ("3", "7", "10", "14", "17", "21")}, flush=True)
    return P


def calib(P, d, t):
    """Holdout: predicted vs actual cover rate at the close +/- n points, and exact key-number frequencies."""
    out = {}
    mu, m = (-d.spread_close).to_numpy(), d.margin.to_numpy()
    for shift in (-7, -3, -1, 0, 1, 3, 7):
        pr, ac = [], []
        for u, k, L in zip(mu, m, d.spread_close.to_numpy()):
            hp = L + shift
            w, p, l = DI.spread_probs(u, hp, "home", P)
            pr.append(w); ac.append(float(k + hp > 0))
        out[f"home covers at close{shift:+d}"] = {"pred": round(float(np.mean(pr)), 4), "actual": round(float(np.mean(ac)), 4)}
    for k in (3, 7, 10, 14):
        pr = np.mean([DI.margin_pmf(u, P)[np.abs(DI.KM) == k].sum() for u in mu])
        out[f"|margin| = {k}"] = {"pred": round(float(pr), 4), "actual": round(float((np.abs(m) == k).mean()), 4)}
    mut, tt = t.total_close.to_numpy(), t.total.to_numpy()
    for shift in (-7, -3, 0, 3, 7):
        pr = np.mean([DI.total_probs(u, u + shift, "over", P)[0] for u in mut])
        out[f"over at close{shift:+d}"] = {"pred": round(float(pr), 4), "actual": round(float((tt > mut + shift).mean()), 4)}
    # normal-only benchmark log loss vs the key-number model on exact margins
    s_k = -np.mean([np.log(DI.margin_pmf(u, P)[DI.KM == int(k)][0]) for u, k in zip(mu, m)])
    Pn = dict(P, w={})
    s_n = -np.mean([np.log(DI.margin_pmf(u, Pn)[DI.KM == int(k)][0]) for u, k in zip(mu, m)])
    out["exact-margin log loss (key-number vs plain normal)"] = {"key": round(float(s_k), 4), "normal": round(float(s_n), 4)}
    return out


def main():
    p = pd.read_parquet(OUT / "preds.parquet")
    p = p[(p.home_div == "fbs") | (p.away_div == "fbs")]
    d = p.dropna(subset=["spread_close", "margin"])
    t = p.dropna(subset=["total_close", "total"])
    P = fit(d[d.season <= 2021], t[t.season <= 2021])
    P["fit_on"] = "CFBD consensus closing lines 2014-2021, FBS games"
    (ROOT / "cfb_dist.json").write_text(json.dumps(P, indent=1))
    DI.params.cache_clear()
    c = calib(P, d[d.season.between(2022, 2025)], t[t.season.between(2022, 2025)])
    (OUT / "dist_calibration.json").write_text(json.dumps(c, indent=1))
    print(json.dumps(c, indent=1))


if __name__ == "__main__":
    main()
