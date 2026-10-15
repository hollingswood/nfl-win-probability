"""College per-play totals v2 (pre-registered: output/research/round5/ppp_totals_v2_prereg.json, 2026-10-15):
A opponent-adjusted, B last-season blend, C both. Same tests as ppp_totals.py.

Run: PYTHONPATH=src python scripts/research/round5/ppp_totals_v2.py -> output/research/round5/ppp_totals_v2.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).parent)]
from cfbpred import data as D  # noqa: E402
import ppp_totals as V1  # noqa: E402

OUT = ROOT / "output" / "research" / "round5"
LAMBDAS, WS = (100, 300, 1000), (2, 4, 8)
ALPHA = 0.05 / 3


def obs(T: pd.DataFrame) -> pd.DataFrame:
    o = T[T.pf.notna() & T.plays.notna() & (T.plays > 0)].copy()
    o["date"] = o.start.dt.tz_convert("America/New_York").dt.date
    o["ppp"] = o.pf / o.plays
    return o


class Ridge:
    """y = mu + off[t] + def[o]; weighted LS, ridge toward prior theta0 (mu unpenalized). Accumulates normal equations."""

    def __init__(self, teams: list[str], lam: float, theta0: dict | None):
        self.ix = {t: i for i, t in enumerate(teams)}
        self.k = 1 + 2 * len(teams)
        self.A = np.zeros((self.k, self.k)); self.b = np.zeros(self.k)
        self.lam = lam
        self.t0 = np.zeros(self.k)
        if theta0:
            for t, (o, d) in theta0.items():
                if t in self.ix:
                    self.t0[1 + self.ix[t]] = o; self.t0[1 + len(teams) + self.ix[t]] = d

    def add(self, off: list[str], de: list[str], y: np.ndarray, w: np.ndarray) -> None:
        n = len(self.ix)
        io = np.array([1 + self.ix[t] for t in off]); idf = np.array([1 + n + self.ix[t] for t in de])
        i0 = np.zeros_like(io)
        for a, c in ((i0, i0), (i0, io), (i0, idf), (io, io), (idf, idf), (io, idf)):
            np.add.at(self.A, (a, c), w)
            if a is not c:
                np.add.at(self.A, (c, a), w)
        for a in (i0, io, idf):
            np.add.at(self.b, a, w * y)

    def solve(self) -> np.ndarray:
        L = np.full(self.k, float(self.lam)); L[0] = 1e-6
        return np.linalg.solve(self.A + np.diag(L), self.b + L * self.t0)

    def rating(self, th: np.ndarray, t: str) -> tuple[float, float]:
        n = len(self.ix); i = self.ix[t]
        return th[1 + i], th[1 + n + i]


def full_season_ratings(o: pd.DataFrame, lam: float) -> dict:
    """Last-season ratings for version C: the same model fit on a whole season."""
    out = {}
    for s, d in o.groupby("season"):
        teams = sorted(set(d.team) | set(d.opp))
        rp, rl = Ridge(teams, lam, None), Ridge(teams, lam, None)
        rp.add(list(d.team), list(d.opp), d.ppp.values, d.plays.values)
        rl.add(list(d.team), list(d.opp), d.plays.values, np.ones(len(d)))
        tp, tl = rp.solve(), rl.solve()
        out[s] = {"ppp": {t: rp.rating(tp, t) for t in teams}, "plays": {t: rl.rating(tl, t) for t in teams}}
    return out


def adjusted(G: pd.DataFrame, o: pd.DataFrame, lam: float, prior: dict | None) -> pd.DataFrame:
    rows = []
    G = G.assign(date=G.start.dt.tz_convert("America/New_York").dt.date)
    for s, gs in G.groupby("season"):
        d = o[o.season == s]
        teams = sorted(set(d.team) | set(d.opp) | set(gs.home) | set(gs.away))
        p0 = (prior or {}).get(s - 1) or {}
        rp, rl = Ridge(teams, lam, p0.get("ppp")), Ridge(teams, lam, p0.get("plays"))
        mu_p0 = mu_l0 = None
        for day, gd in gs.groupby("date"):
            tp, tl = rp.solve(), rl.solve()
            if rp.A[0, 0] == 0:          # nothing played yet this season: league averages from last season
                last = o[o.season == s - 1]
                if not len(last):
                    tp = tl = None
                else:
                    mu_p0 = mu_p0 or float((last.pf.sum()) / last.plays.sum()); mu_l0 = mu_l0 or float(last.plays.mean())
                    tp, tl = rp.t0.copy(), rl.t0.copy(); tp[0], tl[0] = mu_p0, mu_l0
            if tp is not None:
                for g in gd.itertuples():
                    (oh, dh), (oa, da) = rp.rating(tp, g.home), rp.rating(tp, g.away)
                    (ph, qh), (pa, qa) = rl.rating(tl, g.home), rl.rating(tl, g.away)
                    hp = (tp[0] + oh + da) * (tl[0] + ph + qa)
                    ap = (tp[0] + oa + dh) * (tl[0] + pa + qh)
                    rows.append({"game_id": g.game_id, "proj": hp + ap})
            x = d[d.date == day]
            if len(x):
                rp.add(list(x.team), list(x.opp), x.ppp.values, x.plays.values)
                rl.add(list(x.team), list(x.opp), x.plays.values, np.ones(len(x)))
    return pd.DataFrame(rows)


def full_season_rates(T: pd.DataFrame) -> pd.DataFrame:
    ok = T.pf.notna() & T.plays.notna() & T.opp_plays.notna()
    g = T[ok].groupby(["team", "season"])
    r = pd.DataFrame({"oppp": g.pf.sum() / g.plays.sum(), "dppp": g.pa.sum() / g.opp_plays.sum(), "avg_plays": g.plays.mean()}).reset_index()
    r["season"] += 1          # becomes "last season" for the next season
    return r.rename(columns={c: f"p_{c}" for c in ("oppp", "dppp", "avg_plays")})


def evaluate(X: pd.DataFrame) -> dict:
    per = {}
    for lab, yrs in (("development_2014_20", V1.DEV), ("holdout_2021_25", V1.HOLD)):
        d = X[X.season.isin(yrs)]
        per[lab] = {"games": int(len(d)), "mae_projection": round(float((d.proj - d.total).abs().mean()), 3),
                    "mae_close": round(float((d.total_close - d.total).abs().mean()), 3),
                    "avg_projection_minus_close": round(float((d.proj - d.total_close).mean()), 2),
                    "regression_resid_on_gap": V1.ols_t((d.total - d.total_close).values, (d.proj - d.total_close).values),
                    "bets_vs_close": [V1.bets(d, "total_close", k) for k in V1.KS],
                    "bets_vs_open": [V1.bets(d, "total_open", k) for k in V1.KS]}
    dv, ho = per["development_2014_20"], per["holdout_2021_25"]
    sign = 1 if dv["regression_resid_on_gap"]["b"] >= 0 else -1
    t = ho["regression_resid_on_gap"]["t"] * sign
    T2 = {"dev_sign": sign, "holdout_t": t, "p_one_sided": round(0.5 * math.erfc(t / math.sqrt(2)), 4)}
    T2["pass"] = T2["p_one_sided"] < ALPHA
    cand = [b for b in dv["bets_vs_close"] if b["bets"] >= 200]
    kb = max(cand, key=lambda b: b["win_rate"])["k"] if cand else 0
    hb = next(b for b in ho["bets_vs_close"] if b["k"] == kb)
    T3 = {"k_from_dev": kb, "holdout": hb, "pass": bool(hb["win_rate"] and hb["win_rate"] > 0.5238 and hb["p_vs_52.38"] < ALPHA)}
    d26 = X[X.season == 2026]
    return {"periods": per, "T2": T2, "T3": T3,
            "season_2026_info": {"games": int(len(d26)), "bets_vs_close": [V1.bets(d26, "total_close", k) for k in V1.KS]}}


def main():
    G = D.games(range(2013, 2027))
    G = G[G.completed & G.home_pts.notna()]
    T = V1.team_games(G, V1.plays())
    S = V1.season_to_date(T)
    L = D.lines(range(2014, 2027), prefer=D.PREFERRED_PROVIDERS)
    base = G[(G.home_div == "fbs") & (G.away_div == "fbs") & (G.season >= 2014)].merge(
        L[["game_id", "total_close", "total_open"]], on="game_id").dropna(subset=["total_close"])
    cnt = S[["game_id", "team", "n"]]
    base = base.merge(cnt.rename(columns={"team": "home", "n": "h_n"}), on=["game_id", "home"], how="left") \
               .merge(cnt.rename(columns={"team": "away", "n": "a_n"}), on=["game_id", "away"], how="left")
    o = obs(T)
    FR = full_season_rates(T)
    has_prior = set(zip(FR.team, FR.season))
    base["h_prior"] = [(h, s) in has_prior for h, s in zip(base.home, base.season)]
    base["a_prior"] = [(a, s) in has_prior for a, s in zip(base.away, base.season)]
    elig_A = (base.h_n >= 3) & (base.a_n >= 3)
    elig_B = ((base.h_n >= 3) | base.h_prior) & ((base.a_n >= 3) | base.a_prior)
    out = {"prereg": "output/research/round5/ppp_totals_v2_prereg.json", "alpha_each_version": round(ALPHA, 4), "versions": {}}

    def pick(cands):   # (param, frame) -> param with best development MAE
        sc = {p: float((X.proj - X.total).abs()[X.season.isin(V1.DEV)].mean()) for p, X in cands}
        return min(sc, key=sc.get), {str(k): round(v, 3) for k, v in sc.items()}

    # A: opponent-adjusted
    cA = []
    for lam in LAMBDAS:
        P = adjusted(G, o, lam, None)
        cA.append((lam, base[elig_A].merge(P, on="game_id").dropna(subset=["proj"])))
    lamA, scA = pick(cA)
    XA = dict(cA)[lamA]
    out["versions"]["A_opponent_adjusted"] = {"lambda": lamA, "dev_mae_by_lambda": scA, **evaluate(XA)}
    # B: last-season blend
    cB = []
    Sx = S.merge(T[["game_id", "team", "season"]], on=["game_id", "team"]).merge(FR, on=["team", "season"], how="left")
    for w in WS:
        Z = Sx.copy()
        for c in ("oppp", "dppp", "avg_plays"):
            cur, pr, n = Z[c], Z[f"p_{c}"], Z.n
            Z[f"b_{c}"] = np.where(pr.notna(), (n * cur.fillna(0) + w * pr) / (n + w), np.where(n >= 3, cur, np.nan))
        H = Z[["game_id", "team", "b_oppp", "b_dppp", "b_avg_plays"]]
        X = base[elig_B].merge(H.rename(columns={"team": "home", "b_oppp": "h_o", "b_dppp": "h_d", "b_avg_plays": "h_p"}), on=["game_id", "home"]) \
                        .merge(H.rename(columns={"team": "away", "b_oppp": "a_o", "b_dppp": "a_d", "b_avg_plays": "a_p"}), on=["game_id", "away"])
        X["proj"] = (X.h_o * X.h_p + X.a_d * X.h_p) / 2 + (X.a_o * X.a_p + X.h_d * X.a_p) / 2
        cB.append((w, X.dropna(subset=["proj"])))
    wB, scB = pick(cB)
    out["versions"]["B_last_season_blend"] = {"w": wB, "dev_mae_by_w": scB, **evaluate(dict(cB)[wB])}
    # C: both
    cC = []
    for lam in LAMBDAS:
        prior = full_season_ratings(o, lam)
        P = adjusted(G, o, lam, prior)
        cC.append((lam, base[elig_B].merge(P, on="game_id").dropna(subset=["proj"])))
    lamC, scC = pick(cC)
    out["versions"]["C_both"] = {"lambda": lamC, "dev_mae_by_lambda": scC, **evaluate(dict(cC)[lamC])}
    print(json.dumps({k: {kk: (vv if kk not in ("periods", "season_2026_info") else None) for kk, vv in v.items()} for k, v in out["versions"].items()}, indent=1))
    for k, v in out["versions"].items():
        for lab, p in v["periods"].items():
            print(k, lab, p["games"], "MAE", p["mae_projection"], "close", p["mae_close"], "bias", p["avg_projection_minus_close"], "reg", p["regression_resid_on_gap"],
                  "| k0", p["bets_vs_close"][0]["win_rate"], "k5", p["bets_vs_close"][2]["win_rate"], p["bets_vs_close"][2]["bets"], "k10", p["bets_vs_close"][4]["win_rate"], p["bets_vs_close"][4]["bets"],
                  "| open k0", p["bets_vs_open"][0]["win_rate"], "k5", p["bets_vs_open"][2]["win_rate"], p["bets_vs_open"][2]["bets"])
    (OUT / "ppp_totals_v2.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
