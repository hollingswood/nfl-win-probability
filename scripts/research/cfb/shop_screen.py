"""College shop-vs-sharp screen, all markets. PRE-DECLARED 2026-10-04 (written before any result was seen).

Idea (how retail pros make money): price every quote at your books against the sharpest line available at the
same moment, and bet only the quotes that are +EV against it. Sharp line = Pinnacle (data/historical_odds/cfb_pin),
priced at ANY number with the key-number distribution in cfbpred.dist (fit on 2014-21 closes, cfb_dist.json).

For each snapshot 1 h - 7 days before kickoff, each book in the set, each market:
  spreads: P(win/push/lose) of either side at the book's number from Pinnacle's no-vig spread-implied mean margin
  totals:  same from Pinnacle's no-vig total-implied mean total
  ML:      Pinnacle no-vig win probability
  EV = P(win)·(decimal − 1) − P(lose); qualify if EV >= threshold and price within −200..+200.
One bet per game per market: the first qualifying snapshot, best EV at that snapshot.
Book sets: MINE (DraftKings, FanDuel, ESPN Bet/theScore) and AZ (MINE + BetMGM, Caesars, BetRivers, Fanatics,
Hard Rock, Bally). Thresholds: 2% and 4%. -> 3 markets x 2 sets x 2 thresholds = 12 rules.
Grading: CLV = EV of our bet under Pinnacle's LAST pre-kickoff quote of that market (same pricing); ROI vs results.
Pass: >= 100 bets 2021-25, mean CLV > 0 at one-sided p < 0.05/12, ROI >= 0, CLV positive in >= 4 of 5 seasons.
2026 reported separately (out of sample). Reported but not pass criteria: Power-4 vs other games, hours before
kickoff, book, stale-quote check (book last_update older than 3 h at the snapshot).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS  # noqa: E402
from cfbpred import dist as DI  # noqa: E402

MINE = ["draftkings", "fanduel", "espnbet"]
AZ = MINE + ["betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]
P4 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"}
THRESH = (0.02, 0.04)
N_RULES = 12


def results_full(O):
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    out = []
    for s, e in ev.groupby("season"):
        g = PS.D.games([s])
        g = g[g.completed & g.margin.notna()]
        m = PS.team_matcher(sorted(set(g.home) | set(g.away)))
        idx = {(r.home, r.away): r for r in g.itertuples()}
        for r in e.itertuples():
            gg = idx.get((m(r.home), m(r.away)))
            if gg is not None and abs((gg.start - r.ko).total_seconds()) < 36 * 3600:
                out.append({"event_id": r.event_id, "margin": float(gg.margin), "total": float(gg.total),
                            "p4_game": bool(gg.home_conf in P4 or gg.away_conf in P4),
                            "p4_both": bool(gg.home_conf in P4 and gg.away_conf in P4)})
    return pd.DataFrame(out).set_index("event_id")


def mus(pin):
    """Pinnacle rows -> mu_m (spread-implied mean margin), mu_t (total), q_ml (no-vig home win)."""
    pin = pin.copy()
    mm, mt, qm = [], [], []
    for r in pin.itertuples():
        mm.append(DI.mu_from_spread(r.sp_home_point, r.sp_home_price, r.sp_away_price)
                  if not (math.isnan(r.sp_home_point) or math.isnan(r.sp_home_price) or math.isnan(r.sp_away_price)) else np.nan)
        mt.append(DI.mu_from_total(r.tot_point, r.tot_over_price, r.tot_under_price)
                  if not (math.isnan(r.tot_point) or math.isnan(r.tot_over_price) or math.isnan(r.tot_under_price)) else np.nan)
        if not (math.isnan(r.ml_home) or math.isnan(r.ml_away)):
            a, b = DI.implied(r.ml_home), DI.implied(r.ml_away)
            qm.append(a / (a + b))
        else:
            qm.append(np.nan)
    pin["mu_m"], pin["mu_t"], pin["q_ml"] = mm, mt, qm
    return pin


def cdf_tables(mu, kind):
    """rows of P(X <= k) and P(X = k) for each mu (cached by rounded mu)."""
    K = DI.KM if kind == "m" else DI.KT
    cache = {}
    pmf = np.zeros((len(mu), len(K)))
    for i, u in enumerate(mu):
        if np.isnan(u):
            pmf[i] = np.nan; continue
        key = round(float(u), 2)
        if key not in cache:
            cache[key] = DI.margin_pmf(key) if kind == "m" else DI.total_pmf(key)
        pmf[i] = cache[key]
    return np.cumsum(pmf, axis=1), pmf, K


def probs(cdf, pmf, K, x, kind_side):
    """vectorized (win, push, lose) at threshold numbers x. kind_side:
    'home': win if M > -x (x = home point); 'away': win if M < x (x = away point);
    'over': win if T > x; 'under': win if T < x."""
    n = len(x)
    rows = np.arange(n)
    def at(arr, k):     # arr[row, index of integer k], clipped
        idx = np.clip(np.searchsorted(K, k), 0, len(K) - 1)
        return arr[rows, idx]
    if kind_side in ("home", "over"):
        thr = -x if kind_side == "home" else x                      # win if X > thr
        fl = np.floor(thr)
        integer = np.isclose(thr, fl)
        le = at(cdf, fl)                                             # P(X <= floor(thr))
        push = np.where(integer, at(pmf, fl), 0.0)
        win = 1 - le
        lose = 1 - win - push
    else:
        thr = x                                                      # win if X < thr
        ce = np.ceil(thr)
        integer = np.isclose(thr, ce)
        win = at(cdf, ce - 1)                                        # P(X <= ceil(thr) - 1)
        push = np.where(integer, at(pmf, ce), 0.0)
        lose = 1 - win - push
    return win, push, lose


def candidates(O, pinmu, books):
    """All (event, t, book, market, side) quotes with EV vs Pinnacle at the same snapshot."""
    key = ["event_id", "t"]
    Q = O[O.book.isin(books)].merge(pinmu[key + ["mu_m", "mu_t", "q_ml"]], on=key, how="inner")
    Q = Q[((Q.ko - Q.t) >= pd.Timedelta(hours=1)) & ((Q.ko - Q.t) <= pd.Timedelta(days=7))]
    out = []
    # spreads
    S = Q.dropna(subset=["mu_m", "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"]).reset_index(drop=True)
    cdf, pmf, K = cdf_tables(S.mu_m.to_numpy(), "m")
    for side, pt, pr in (("home", "sp_home_point", "sp_home_price"), ("away", "sp_away_point", "sp_away_price")):
        w, p, l = probs(cdf, pmf, K, S[pt].to_numpy(float), side)
        d = PS.dec(S[pr].to_numpy(float))
        out.append(S[key + ["ko", "season", "book", "last_update"]].assign(market="spread", side=side, point=S[pt], price=S[pr],
                                                                           ev=w * (d - 1) - l, pwin=w))
    T = Q.dropna(subset=["mu_t", "tot_point", "tot_over_price", "tot_under_price"]).reset_index(drop=True)
    cdf, pmf, K = cdf_tables(T.mu_t.to_numpy(), "t")
    for side, pr in (("over", "tot_over_price"), ("under", "tot_under_price")):
        w, p, l = probs(cdf, pmf, K, T.tot_point.to_numpy(float), side)
        d = PS.dec(T[pr].to_numpy(float))
        out.append(T[key + ["ko", "season", "book", "last_update"]].assign(market="total", side=side, point=T.tot_point, price=T[pr],
                                                                           ev=w * (d - 1) - l, pwin=w))
    M = Q.dropna(subset=["q_ml", "ml_home", "ml_away"]).reset_index(drop=True)
    for side, pr in (("home", "ml_home"), ("away", "ml_away")):
        q = M.q_ml.to_numpy() if side == "home" else 1 - M.q_ml.to_numpy()
        d = PS.dec(M[pr].to_numpy(float))
        out.append(M[key + ["ko", "season", "book", "last_update"]].assign(market="ml", side=side, point=np.nan, price=M[pr],
                                                                          ev=q * d - 1, pwin=q))
    C = pd.concat(out, ignore_index=True)
    return C[C.price.between(-200, 200)]


def grade(B, close, R):
    """CLV under Pinnacle's last pre-kickoff quote of the same market, and realized result."""
    B = B.join(close, on="event_id", how="inner").join(R, on="event_id", how="inner").reset_index(drop=True)
    clv = np.full(len(B), np.nan)
    res = np.full(len(B), np.nan)
    d = PS.dec(B.price.to_numpy(float))
    for mk, kind, mucol in (("spread", "m", "c_mu_m"), ("total", "t", "c_mu_t")):
        idx = np.where((B.market == mk).to_numpy() & B[mucol].notna().to_numpy())[0]
        if not len(idx):
            continue
        sub = B.iloc[idx]
        cdf, pmf, K = cdf_tables(sub[mucol].to_numpy(), kind)
        for side in (("home", "away") if mk == "spread" else ("over", "under")):
            j = np.where((sub.side == side).to_numpy())[0]
            if not len(j):
                continue
            w, p, l = probs(cdf[j], pmf[j], K, sub.point.to_numpy(float)[j], side)
            clv[idx[j]] = w * (d[idx[j]] - 1) - l
    idx = np.where((B.market == "ml").to_numpy() & B.c_q_ml.notna().to_numpy())[0]
    q = np.where(B.side.to_numpy()[idx] == "home", B.c_q_ml.to_numpy()[idx], 1 - B.c_q_ml.to_numpy()[idx])
    clv[idx] = q * d[idx] - 1
    m, t, pt = B.margin.to_numpy(), B.total.to_numpy(), B.point.to_numpy(float)
    side = B.side.to_numpy()
    mk = B.market.to_numpy()
    res = np.select([(mk == "spread") & (side == "home"), (mk == "spread") & (side == "away"),
                     (mk == "total") & (side == "over"), (mk == "total") & (side == "under"),
                     (mk == "ml") & (side == "home"), (mk == "ml") & (side == "away")],
                    [m + pt, -m + pt, t - pt, pt - t, m, -m], np.nan)
    B["clv"], B["res"] = clv, res
    B["pnl"] = np.where(res > 0, d - 1, np.where(res < 0, -1.0, 0.0))
    return B


def stats(x):
    x = x.dropna(subset=["clv"])
    if not len(x):
        return {"bets": 0}
    m, se = x.clv.mean(), x.clv.std() / math.sqrt(len(x)) if len(x) > 1 else 1.0
    return {"bets": int(len(x)), "mean_ev_at_bet": round(float(x.ev.mean()), 4), "mean_clv": round(float(m), 4),
            "se": round(float(se), 4), "p": float(0.5 * math.erfc((m / se) / math.sqrt(2))) if se > 0 else 1.0,
            "roi": round(float(x.pnl.mean()), 4), "roi_se": round(float(x.pnl.std() / math.sqrt(len(x))), 4) if len(x) > 1 else None,
            "win_rate": round(float((x.res > 0).sum() / max(1, (x.res != 0).sum())), 3)}


def main():
    O, Pn = PS.load("cfb"), PS.load("cfb_pin")
    R = results_full(O)
    print("events with results", len(R), flush=True)
    pin = mus(Pn.drop_duplicates(["event_id", "t"]))
    pin.to_parquet(PS.OUT / "pin_mus.parquet")
    close = pin.sort_values("t").groupby("event_id").agg(c_mu_m=("mu_m", "last"), c_mu_t=("mu_t", "last"), c_q_ml=("q_ml", "last"))
    # last NON-null per market
    close = pin.sort_values("t").groupby("event_id")[["mu_m", "mu_t", "q_ml"]].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    close.columns = ["c_mu_m", "c_mu_t", "c_q_ml"]
    out, allbets = {}, []
    for setname, books in (("MINE", MINE), ("AZ", AZ)):
        C = candidates(O, pin, books)
        C["stale"] = (C.t - pd.to_datetime(C.last_update, utc=True)) > pd.Timedelta(hours=3)
        for th in THRESH:
            for mk in ("spread", "total", "ml"):
                q = C[(C.market == mk) & (C.ev >= th)]
                first_t = q.groupby("event_id").t.min()
                q = q[q.t == q.event_id.map(first_t)].sort_values("ev", ascending=False).groupby("event_id").head(1)
                B = grade(q.set_index("event_id"), close, R).assign(rule=f"{setname} {mk} EV>={th:.0%}")
                allbets.append(B)
                dev = B[B.season <= 2025]
                st = stats(dev)
                per = dev.groupby("season").clv.mean()
                st["seasons_clv_positive"] = f"{int((per > 0).sum())}/{len(per)}"
                st["by_season"] = {int(s): {**stats(g), "p": None} for s, g in dev.groupby("season")}
                st["pass"] = bool(st.get("bets", 0) >= 100 and st.get("p", 1) < 0.05 / N_RULES and st.get("roi", -1) >= 0
                                  and int((per > 0).sum()) >= 4)
                st["2026_out_of_sample"] = stats(B[B.season == 2026])
                st["power4_game"] = stats(dev[dev.p4_game])
                st["non_power4"] = stats(dev[~dev.p4_game])
                st["stale_quotes"] = stats(dev[dev.stale])
                st["fresh_quotes"] = stats(dev[~dev.stale])
                hb = ((dev.ko - dev.t).dt.total_seconds() / 3600)
                st["by_hours_before"] = {lab: stats(dev[(hb > lo) & (hb <= hi)]) for lab, lo, hi in
                                         (("1-6h", 1, 6), ("6-24h", 6, 24), ("1-3d", 24, 72), ("3-7d", 72, 170))}
                st["by_book"] = {b: stats(g) for b, g in dev.groupby("book")}
                out[f"{setname} {mk} EV>={th:.0%}"] = st
                print(f"{setname} {mk} >= {th:.0%}:", {k: st[k] for k in ("bets", "mean_ev_at_bet", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass") if k in st},
                      "| 2026", {k: st["2026_out_of_sample"].get(k) for k in ("bets", "mean_clv", "roi")}, flush=True)
    pd.concat(allbets).to_parquet(PS.OUT / "shop_bets.parquet")
    (PS.OUT / "shop_screen.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
