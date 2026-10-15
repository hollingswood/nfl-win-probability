"""College totals from points per play x plays (pre-registered: output/research/round5/ppp_totals_prereg.json, 2026-10-15).

Run: PYTHONPATH=src python scripts/research/round5/ppp_totals.py -> output/research/round5/ppp_totals.json
"""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cfbpred import data as D  # noqa: E402

OUT = ROOT / "output" / "research" / "round5"
KS = (0, 3, 5, 7, 10)
DEV, HOLD = range(2014, 2021), range(2021, 2026)


def plays() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data" / "cfb" / "raw").glob("stats_game_advanced_*.json.gz")):
        for r in json.load(gzip.open(f)):
            p = (r.get("offense") or {}).get("plays")
            if p:
                rows.append({"game_id": r["gameId"], "team": r["team"], "plays": float(p)})
    return pd.DataFrame(rows).drop_duplicates(["game_id", "team"])


def team_games(G: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    h = G.rename(columns={"home": "team", "away": "opp", "home_pts": "pf", "away_pts": "pa"})
    a = G.rename(columns={"away": "team", "home": "opp", "away_pts": "pf", "home_pts": "pa"})
    T = pd.concat([h, a])[["game_id", "season", "start", "team", "opp", "pf", "pa"]]
    T = T.merge(P, on=["game_id", "team"], how="left")
    T = T.merge(P.rename(columns={"team": "opp", "plays": "opp_plays"}), on=["game_id", "opp"], how="left")
    return T.sort_values(["team", "start"]).reset_index(drop=True)


def season_to_date(T: pd.DataFrame) -> pd.DataFrame:
    ok = T.pf.notna() & T.plays.notna() & T.opp_plays.notna()
    X = T.assign(pf_=T.pf.where(ok, 0), pa_=T.pa.where(ok, 0), pl_=T.plays.where(ok, 0), op_=T.opp_plays.where(ok, 0), n_=ok.astype(int))
    g = X.groupby(["team", "season"])
    cs = g[["pf_", "pa_", "pl_", "op_", "n_"]].cumsum() - X[["pf_", "pa_", "pl_", "op_", "n_"]]   # strictly earlier games
    out = T[["game_id", "team"]].copy()
    out["n"] = cs.n_
    out["oppp"] = cs.pf_ / cs.pl_.replace(0, np.nan)
    out["dppp"] = cs.pa_ / cs.op_.replace(0, np.nan)
    out["avg_plays"] = cs.pl_ / cs.n_.replace(0, np.nan)
    return out


def p_binom_gt(w: int, n: int, p0: float) -> float:
    if not n:
        return 1.0
    z = (w - 0.5 - n * p0) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def ols_t(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones_like(x), x]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; s2 = e @ e / (len(y) - 2); se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return {"b": round(float(b[1]), 4), "t": round(float(b[1] / se), 2), "n": int(len(y))}


def bets(D_: pd.DataFrame, line: str, k: float) -> dict:
    d = D_.dropna(subset=[line])
    e = d.proj - d[line]
    pick = np.where(e >= k, 1, np.where(e <= -k, -1, 0)) if k > 0 else np.where(e > 0, 1, np.where(e < 0, -1, 0))
    res = np.sign((d.total - d[line]).values) * pick
    w, l = int((res > 0).sum()), int((res < 0).sum())
    ov = pick == 1
    return {"k": k, "bets": w + l, "wins": w, "losses": l, "win_rate": round(w / (w + l), 4) if w + l else None,
            "p_vs_52.38": round(p_binom_gt(w, w + l, 0.5238), 4),
            "overs": int(((res != 0) & ov).sum()), "over_wins": int(((res > 0) & ov).sum()),
            "unders": int(((res != 0) & (pick == -1)).sum()), "under_wins": int(((res > 0) & (pick == -1)).sum())}


def main():
    G = D.games(range(2014, 2027))
    G = G[G.completed & G.home_pts.notna()]
    T = team_games(G, plays())
    S = season_to_date(T)
    H = S.rename(columns={c: f"h_{c}" for c in ("n", "oppp", "dppp", "avg_plays")}).rename(columns={"team": "home"})
    A = S.rename(columns={c: f"a_{c}" for c in ("n", "oppp", "dppp", "avg_plays")}).rename(columns={"team": "away"})
    X = G.merge(H, on=["game_id", "home"]).merge(A, on=["game_id", "away"])
    L = D.lines(range(2014, 2027), prefer=D.PREFERRED_PROVIDERS)
    X = X.merge(L[["game_id", "total_close", "total_open"]], on="game_id", how="left")
    X = X[(X.home_div == "fbs") & (X.away_div == "fbs") & (X.h_n >= 3) & (X.a_n >= 3) & X.total_close.notna()].copy()
    X["home_proj"] = (X.h_oppp * X.h_avg_plays + X.a_dppp * X.h_avg_plays) / 2
    X["away_proj"] = (X.a_oppp * X.a_avg_plays + X.h_dppp * X.a_avg_plays) / 2
    X["proj"] = X.home_proj + X.away_proj
    X = X.dropna(subset=["proj"])
    out = {"prereg": "output/research/round5/ppp_totals_prereg.json", "games": int(len(X))}
    per = {}
    for lab, yrs in (("development_2014_20", DEV), ("holdout_2021_25", HOLD)):
        d = X[X.season.isin(yrs)]
        per[lab] = {"games": int(len(d)), "mae_projection": round(float((d.proj - d.total).abs().mean()), 2),
                    "mae_close": round(float((d.total_close - d.total).abs().mean()), 2),
                    "avg_projection_minus_close": round(float((d.proj - d.total_close).mean()), 2),
                    "regression_resid_on_gap": ols_t((d.total - d.total_close).values, (d.proj - d.total_close).values),
                    "bets_vs_close": [bets(d, "total_close", k) for k in KS],
                    "bets_vs_open": [bets(d, "total_open", k) for k in KS]}
    out["periods"] = per
    dv, ho = per["development_2014_20"], per["holdout_2021_25"]
    sign = 1 if dv["regression_resid_on_gap"]["b"] >= 0 else -1
    t = ho["regression_resid_on_gap"]["t"] * sign
    out["T2"] = {"dev_sign": sign, "holdout_t": t, "p_one_sided": round(0.5 * math.erfc(t / math.sqrt(2)), 4)}
    out["T2"]["pass"] = out["T2"]["p_one_sided"] < 0.05
    cand = [b for b in dv["bets_vs_close"] if b["bets"] >= 200]
    kbest = max(cand, key=lambda b: b["win_rate"])["k"]
    hb = next(b for b in ho["bets_vs_close"] if b["k"] == kbest)
    out["T3"] = {"k_from_dev": kbest, "holdout": hb, "pass": bool(hb["win_rate"] and hb["win_rate"] > 0.5238 and hb["p_vs_52.38"] < 0.05)}
    # this season (2026, live check of the same rule; not part of the test)
    d26 = X[X.season == 2026]
    out["season_2026_info"] = {"games": int(len(d26)), "bets_vs_close": [bets(d26, "total_close", k) for k in KS]} if len(d26) else None
    print(json.dumps(out, indent=1))
    (OUT / "ppp_totals.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
