"""NFL shop-vs-sharp screen, all markets. PRE-DECLARED 2026-10-04 (committed before any result was seen), mirroring the
college screen (scripts/research/cfb/shop_screen.py) that passed.

Pricing: NFL key-number margin/total distributions (same form as cfbpred.dist), fit on nflverse closing lines 2015-2021,
saved to nfl_dist.json. Odds: hourly snapshots 2022-25 (data/historical_odds/dense) and Pinnacle (dense_pin), which is
present only at some request times; a quote is priced only at snapshots where Pinnacle is present.

For each snapshot 1 h - 7 days before kickoff, each book in the set, each market (spread at any number, total at any number,
moneyline): EV vs Pinnacle's no-vig line at the same snapshot. Qualify at EV >= threshold, price -200..+200.
One bet per game per market: the first qualifying snapshot, best EV at that snapshot.
Book sets: MINE (DraftKings, FanDuel, ESPN Bet/Barstool) and AZ (MINE + BetMGM, Caesars, BetRivers, Fanatics, Hard Rock, Bally).
Thresholds 2% and 4% -> 12 rules. Grading: CLV = EV under Pinnacle's last pre-kickoff quote of that market; ROI vs results.
Pass: >= 100 bets, CLV > 0 at one-sided p < 0.05/12, ROI >= 0, CLV positive in >= 3 of 4 seasons.
Model-free check reported: points our number beat Pinnacle's closing number by.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "cfb"))
from cfbpred import dist as DI  # noqa: E402
from nflpred import data as ND  # noqa: E402

OUT = ROOT / "output" / "research"
MINE = ["draftkings", "fanduel", "espnbet", "barstool"]
AZ = MINE + ["betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]
THRESH = (0.02, 0.04)
N_RULES = 12


def fit_dist(sched: pd.DataFrame) -> dict:
    d = sched[sched.season.between(2015, 2021) & sched.result.notna() & sched.spread_line.notna() & (sched.result != 0)]
    P = {"a": 13.5, "b": 0.0, "w": {}, "at": 10.0, "bt": 0.0, "v": {}}
    mu, m = d.spread_line.to_numpy(float), d.result.astype(int).to_numpy()
    t = d.dropna(subset=["total_line"])
    mut, tt = t.total_line.to_numpy(float), t.total.astype(int).to_numpy()
    for it in range(6):
        best = None
        for a in np.arange(11.0, 16.01, 0.25):
            for b in (0.0, 0.02, 0.04, 0.06, 0.08):
                Q = dict(P, a=float(a), b=float(b))
                s = -sum(np.log(max(DI.margin_pmf(u, Q)[DI.KM == k][0], 1e-12)) for u, k in zip(mu[::2], m[::2]))
                if best is None or s < best[0]:
                    best = (s, a, b)
        P["a"], P["b"] = float(best[1]), float(best[2])
        exp = np.zeros(len(DI.KM))
        for u in mu:
            exp += DI.margin_pmf(u, P)
        P["w"] = {str(k): float(P["w"].get(str(k), 1.0) * ((np.abs(m) == k).sum() + 2) / (exp[np.abs(DI.KM) == k].sum() + 2)) for k in range(1, 46)}
        best = None
        for a in np.arange(7.0, 16.01, 0.5):
            for b in (0.0, 0.05, 0.1, 0.15):
                Q = dict(P, at=float(a), bt=float(b))
                s = -sum(np.log(max(DI.total_pmf(u, Q)[DI.KT == k][0], 1e-12)) for u, k in zip(mut[::2], tt[::2]))
                if best is None or s < best[0]:
                    best = (s, a, b)
        P["at"], P["bt"] = float(best[1]), float(best[2])
        expt = np.zeros(len(DI.KT))
        for u in mut:
            expt += DI.total_pmf(u, P)
        P["v"] = {str(k): float(P["v"].get(str(k), 1.0) * ((tt == k).sum() + 3) / (expt[DI.KT == k].sum() + 3)) for k in range(10, 91)}
        print(it, P["a"], P["b"], P["at"], P["bt"], {k: round(P["w"][k], 2) for k in ("3", "7", "10", "14")}, flush=True)
    P["fit_on"] = "nflverse closing lines 2015-2021 (no ties)"
    return P


def load(sub):
    fs = sorted(glob.glob(str(ROOT / f"data/historical_odds/{sub}/nfl_odds_*.csv.gz")))
    o = pd.concat([pd.read_csv(f).assign(season=int(Path(f).name[9:13])) for f in fs], ignore_index=True)
    o["t"] = pd.to_datetime(o.requested_ts, utc=True)
    o["ko"] = pd.to_datetime(o.commence_time, utc=True)
    return o[o.t < o.ko]


def results(O, sched):
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    s = sched[sched.result.notna()][["season", "home_team", "away_team", "gameday", "result", "total", "game_id"]]
    m = ev.merge(s, left_on=["season", "home", "away"], right_on=["season", "home_team", "away_team"])
    m = m[(pd.to_datetime(m.gameday).dt.tz_localize("UTC") - m.ko.dt.floor("D")).abs() <= pd.Timedelta(days=2)]
    return m.set_index("event_id")[["result", "total"]].rename(columns={"result": "margin"}).assign(p4_game=True)


def main():
    import shop_screen as SS                        # college screen machinery (pricing, grading, stats)
    sched = ND.load_schedules()
    path = ROOT / "nfl_dist.json"
    if not path.exists():
        path.write_text(json.dumps(fit_dist(sched), indent=1))
    DI.PARAMS = path
    DI.params.cache_clear()
    DI._WCACHE.clear()
    O, Pn = load("dense"), load("dense_pin")
    Pn = Pn[Pn.book == "pinnacle"]
    R = results(O, sched)
    print("events with results", len(R), flush=True)
    pin = SS.mus(Pn.drop_duplicates(["event_id", "t"]))
    pin.to_parquet(OUT / "nfl_pin_mus.parquet")
    close = pin.sort_values("t").groupby("event_id")[["mu_m", "mu_t", "q_ml"]].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    close.columns = ["c_mu_m", "c_mu_t", "c_q_ml"]
    lastpts = Pn.sort_values("t").groupby("event_id")[["sp_home_point", "tot_point"]].last()
    out, allbets = {}, []
    for setname, books in (("MINE", MINE), ("AZ", AZ)):
        mon = O.ko.dt.strftime("%Y-%m")
        C = pd.concat([SS.candidates(O[mon == m], pin, books) for m in sorted(mon.unique())], ignore_index=True)   # by month: memory
        for th in THRESH:
            for mk in ("spread", "total", "ml"):
                q = C[(C.market == mk) & (C.ev >= th)]
                first_t = q.groupby("event_id").t.min()
                q = q[q.t == q.event_id.map(first_t)].sort_values("ev", ascending=False).groupby("event_id").head(1)
                B = SS.grade(q.set_index("event_id"), close, R).assign(rule=f"{setname} {mk} EV>={th:.0%}")
                allbets.append(B)
                st = SS.stats(B)
                per = B.groupby("season").clv.mean()
                st["seasons_clv_positive"] = f"{int((per > 0).sum())}/{len(per)}"
                st["by_season"] = {int(s): SS.stats(g) for s, g in B.groupby("season")}
                st["pass"] = bool(st.get("bets", 0) >= 100 and st.get("p", 1) < 0.05 / N_RULES and st.get("roi", -1) >= 0 and int((per > 0).sum()) >= 3)
                hb = (B.ko - B.t).dt.total_seconds() / 3600
                st["by_hours_before"] = {lab: SS.stats(B[(hb > lo) & (hb <= hi)]) for lab, lo, hi in (("1-6h", 1, 6), ("6-24h", 6, 24), ("1-3d", 24, 72), ("3-7d", 72, 170))}
                st["by_book"] = {b: SS.stats(g) for b, g in B.groupby("book")}
                if mk != "ml" and len(B):
                    L = B.join(lastpts, on="event_id")
                    if mk == "spread":
                        hp = np.where(L.side == "home", L.point, -L.point)
                        gain = np.where(L.side == "home", hp - L.sp_home_point, L.sp_home_point - hp)
                    else:
                        gain = np.where(L.side == "over", L.tot_point - L.point, L.point - L.tot_point)
                    gain = gain[~np.isnan(gain)]
                    st["points_beaten_vs_close"] = {"mean": round(float(gain.mean()), 2), "se": round(float(gain.std() / math.sqrt(len(gain))), 2)} if len(gain) else None
                out[f"{setname} {mk} EV>={th:.0%}"] = st
                print(f"{setname} {mk} >= {th:.0%}:", {k: st[k] for k in ("bets", "mean_ev_at_bet", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass") if k in st},
                      st.get("points_beaten_vs_close"), flush=True)
    pd.concat(allbets).to_parquet(OUT / "nfl_shop_bets.parquet")
    (OUT / "nfl_shop.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
