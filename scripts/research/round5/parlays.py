"""Parlays, straight and round robin (pre-registered: output/research/round5/parlays_prereg.json, 2026-10-15).

Run: PYTHONPATH=src python scripts/research/round5/parlays.py -> output/research/round5/parlays.json
"""
from __future__ import annotations

import itertools
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
RNG = np.random.default_rng(7)
DEC110 = 1 + 100 / 110


def dec(a: float) -> float:
    return 1 + (a / 100 if a > 0 else 100 / -a)


def parlay(legs: list[tuple[float, int]]) -> float:
    """legs = [(decimal, outcome +1 win / 0 push / -1 loss)] -> profit per 1 unit staked."""
    if any(o < 0 for _, o in legs):
        return -1.0
    return float(np.prod([d for d, o in legs if o > 0])) - 1.0 if any(o > 0 for _, o in legs) else 0.0


# ---------------------------------------------------------------- P1: no-edge legs at the closing spread, -110
def closing_legs() -> dict[str, pd.DataFrame]:
    G = D.games(range(2021, 2026))
    L = D.lines(range(2021, 2026), prefer=D.PREFERRED_PROVIDERS)
    c = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.completed].merge(L[["game_id", "spread_close"]], on="game_id").dropna(subset=["spread_close"])
    c["res"] = c.margin + c.spread_close                 # home cover margin (home line: + = home underdog)
    c = c.rename(columns={"week": "wk"})[["season", "wk", "game_id", "res"]]
    from nflpred import data as ND
    s = ND.load_schedules()
    n = s[s.season.between(2021, 2025) & s.home_score.notna() & s.spread_line.notna()].copy()
    n["res"] = (n.home_score - n.away_score) - n.spread_line   # nflverse spread_line: + = home favored
    n = n.rename(columns={"week": "wk"})[["season", "wk", "game_id", "res"]]
    return {"college": c, "nfl": n}


def p1(df: pd.DataFrame, n_sims: int = 20000) -> dict:
    side = RNG.integers(0, 2, len(df)) * 2 - 1         # random side per game: +1 home, -1 away
    out_ = np.sign(df.res.values * side).astype(int)
    df = df.assign(o=out_)
    p = float((df.o > 0).sum() / (df.o != 0).sum())
    weeks = [g.o.values for _, g in df.groupby(["season", "wk"]) if len(g) >= 6]
    res = {"legs": int(len(df)), "leg_cover_rate": round(p, 4), "singles_roi": round(p * DEC110 - 1, 4)}
    for k in range(2, 7):
        prof = []
        for _ in range(n_sims):
            w = weeks[RNG.integers(len(weeks))]
            legs = RNG.choice(w, k, replace=False)
            prof.append(parlay([(DEC110, int(o)) for o in legs]))
        res[f"parlay_{k}"] = {"expected_roi_if_50pct": round((0.5 * DEC110) ** k - 1, 4), "expected_roi_at_leg_rate": round((p * DEC110) ** k - 1, 4),
                              "realized_roi": round(float(np.mean(prof)), 4), "hit_rate": round(float(np.mean(np.array(prof) > 0)), 4)}
    for n_, k in ((4, 2), (5, 2), (5, 3), (6, 3)):
        prof = []
        for _ in range(n_sims // 4):
            w = weeks[RNG.integers(len(weeks))]
            legs = RNG.choice(w, n_, replace=False)
            cb = list(itertools.combinations(legs, k))
            prof.append(np.mean([parlay([(DEC110, int(o)) for o in c]) for c in cb]))
        res[f"round_robin_{n_}_by_{k}"] = {"realized_roi": round(float(np.mean(prof)), 4)}
    return res


# ---------------------------------------------------------------- P2: edge legs (college shop vs sharp at your books)
def edge_legs() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "output" / "research" / "cfb" / "shop_bets.parquet")
    d = d[d.rule.str.startswith("MINE") & d.season.between(2021, 2025)]
    d = d[((d.market == "spread") & (d.ev >= 0.04)) | (d.market.isin(["total", "ml"]) & (d.ev >= 0.02))]
    d = d[(d.price >= -200) & (d.price <= 200)]
    d = d.sort_values("t").drop_duplicates(["event_id", "market"])          # one per game per market, first sight
    d["o"] = np.sign(d.pnl).astype(int)
    d["d"] = d.price.map(dec)
    ko = d.ko.dt.tz_convert("America/New_York")
    d["slate"] = (ko - pd.to_timedelta(ko.dt.weekday, unit="D")).dt.date     # Monday of the kickoff week
    return d


def combos(g: pd.DataFrame, k: int):
    rows = list(g.itertuples())
    for c in itertools.combinations(rows, k):
        if len({r.event_id for r in c}) == k:      # never two legs from the same game
            yield c


def weekly(d: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """One row per slate: profit and CLV-implied return per 1 unit of total weekly stake."""
    rows = []
    for s, g in d.groupby("slate"):
        g = g.sort_values("ev", ascending=False)
        if fmt == "singles":
            P = [(r.pnl, r.clv) for r in g.itertuples()]
        elif fmt.startswith("straight_"):
            k = int(fmt.split("_")[1])
            P = [(parlay([(r.d, r.o) for r in c]), float(np.prod([1 + r.clv for r in c]) - 1)) for c in combos(g, k)]
        else:      # rr_N_by_k: top N legs (different games) by EV
            n_, k = map(int, fmt.split("_")[1::2])
            top = g.drop_duplicates("event_id").head(n_)
            if len(top) < n_:
                continue
            if fmt.endswith("singles"):
                P = [(r.pnl, r.clv) for r in top.itertuples()]
            else:
                P = [(parlay([(r.d, r.o) for r in c]), float(np.prod([1 + r.clv for r in c]) - 1)) for c in itertools.combinations(list(top.itertuples()), k)]
        if P:
            rows.append({"slate": s, "season": int(g.season.iloc[0]), "ret": float(np.mean([p for p, _ in P])), "clv_ret": float(np.mean([c for _, c in P])), "n": len(P)})
    return pd.DataFrame(rows)


def summarize(w: pd.DataFrame) -> dict:
    if not len(w):
        return {}
    cum = w.ret.cumsum()
    return {"weeks": int(len(w)), "roi": round(float(w.ret.mean()), 4), "clv_implied_roi": round(float(w.clv_ret.mean()), 4),
            "losing_weeks": round(float((w.ret < 0).mean()), 3), "worst_week": round(float(w.ret.min()), 3),
            "max_drawdown_weeks_of_stake": round(float((cum.cummax() - cum).max()), 2), "sd_week": round(float(w.ret.std()), 3)}


def p2() -> dict:
    d = edge_legs()
    out = {"legs": int(len(d)), "legs_per_slate_median": float(d.groupby("slate").size().median())}
    fmts = ["singles", "straight_2", "straight_3", "straight_4"]
    rr = [(4, 2), (5, 2), (5, 3), (6, 2), (6, 3)]
    for lab, yrs in (("development_2021_22", (2021, 2022)), ("holdout_2023_25", (2023, 2024, 2025))):
        x = d[d.season.isin(yrs)]
        res = {f: summarize(weekly(x, f)) for f in fmts}
        for n_, k in rr:
            res[f"rr_{n_}_by_{k}"] = summarize(weekly(x, f"rr_{n_}_by_{k}"))
            res[f"rr_{n_}_singles"] = summarize(weekly(x, f"rr_{n_}_by_1_singles"))
        out[lab] = res
    h = out["holdout_2023_25"]
    verdict = {}
    for f in ["straight_2", "straight_3", "straight_4"] + [f"rr_{n_}_by_{k}" for n_, k in rr]:
        base = h["singles"] if f.startswith("straight") else h[f"rr_{f.split('_')[1]}_singles"]
        x = h[f]
        verdict[f] = bool(x and x["clv_implied_roi"] > base["clv_implied_roi"] and x["roi"] > base["roi"]
                          and x["max_drawdown_weeks_of_stake"] <= 1.5 * base["max_drawdown_weeks_of_stake"])
    out["P2_beats_singles"] = verdict
    return out


# ---------------------------------------------------------------- P3: same-game correlation (information)
def p3() -> dict:
    G = D.games(range(2014, 2026))
    L = D.lines(range(2014, 2026), prefer=D.PREFERRED_PROVIDERS)
    c = G[(G.home_div == "fbs") & (G.away_div == "fbs") & G.completed].merge(L[["game_id", "spread_close", "total_close"]], on="game_id").dropna(subset=["spread_close", "total_close"])
    c["fav_cover"] = np.sign(np.where(c.spread_close < 0, c.margin + c.spread_close, -(c.margin + c.spread_close)))
    c["over"] = np.sign(c.total - c.total_close)
    c["ratio"] = c.spread_close.abs() / c.total_close
    from nflpred import data as ND
    s = ND.load_schedules()
    n = s[s.season.between(2013, 2025) & s.home_score.notna() & s.spread_line.notna() & s.total_line.notna()].copy()
    m = n.home_score - n.away_score
    n["fav_cover"] = np.sign(np.where(n.spread_line > 0, m - n.spread_line, n.spread_line - m))
    n["over"] = np.sign(n.home_score + n.away_score - n.total_line)
    n["ratio"] = n.spread_line.abs() / n.total_line
    out = {}
    for lg, df, dev in (("college", c, range(2014, 2021)), ("nfl", n, range(2013, 2020))):
        df = df[(df.fav_cover != 0) & (df.over != 0) & (df.ratio > 0)].copy()   # pick'em and pushes out
        df["bucket"] = pd.cut(df.ratio, [0, 0.15, 0.30, 9], labels=["<0.15", "0.15-0.30", ">=0.30"], right=False)
        tab = {}
        for per, yrs in (("dev", dev), ("holdout", range(2021, 2026))):
            x = df[df.season.isin(yrs)]
            for b, xb in x.groupby("bucket", observed=True):
                for pair, (f, o) in {"fav+over": (1, 1), "fav+under": (1, -1), "dog+over": (-1, 1), "dog+under": (-1, -1)}.items():
                    hit = ((xb.fav_cover == f) & (xb.over == o)).mean()
                    tab.setdefault(per, {}).setdefault(str(b), {})[pair] = {"n": int(len(xb)), "hit": round(float(hit), 4)}
        best = max(((b, p, v["hit"]) for b, ps in tab["dev"].items() for p, v in ps.items()), key=lambda t: t[2])
        hv = tab["holdout"][best[0]][best[1]]
        k = round(hv["hit"] * hv["n"])
        z = (k - 0.5 - hv["n"] * 0.2747) / math.sqrt(hv["n"] * 0.2747 * 0.7253)
        out[lg] = {"table": tab, "best_on_dev": {"bucket": best[0], "pair": best[1], "dev_hit": best[2]},
                   "holdout": hv, "holdout_roi_at_plus264": round(hv["hit"] * 3.64 - 1, 4), "p_vs_27.5pct": round(0.5 * math.erfc(z / math.sqrt(2)), 4)}
    return out


def main():
    out = {"prereg": "output/research/round5/parlays_prereg.json",
           "P1_no_edge": {lg: p1(df) for lg, df in closing_legs().items()},
           "P2_edge_legs_college_shop": p2(), "P3_same_game_correlation": p3()}
    (OUT / "parlays.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "P3_same_game_correlation"}, indent=1, default=str)[:9000])
    for lg, v in out["P3_same_game_correlation"].items():
        print(lg, v["best_on_dev"], v["holdout"], v["holdout_roi_at_plus264"], v["p_vs_27.5pct"])


if __name__ == "__main__":
    main()
