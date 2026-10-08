"""Stale soft book after a sharp (Pinnacle) move + the Kambi-cluster hypothesis.
PRE-DECLARED 2026-10-07, written before any result of this test was computed. Nothing below is tuned on the confirm years.

Question. When Pinnacle's line moves (sharp information), are some Arizona-legal books slow to copy it, so that taking
the lagging book's stale price at the next hourly snapshot beats Pinnacle's close? This conditions on a recent Pinnacle
MOVE; the earlier static shop screens (scripts/research/nfl_shop.py: NFL failed; scripts/research/cfb/shop_screen.py:
college passed) did not.

Data (local only). NFL: hourly Odds API snapshots (requested at :10 every hour) 2022-25, all US books in
data/historical_odds/dense, Pinnacle in dense_pin. There are NO hourly NFL snapshots for 2020-21, so the requested
explore 2020-2022 / confirm 2023-2025 split becomes explore = 2022 (one season), confirm = 2023-2025 (kept as asked).
College: data/historical_odds/cfb + cfb_pin 2021-26, NOT hourly: snapshots at 16:10 and 23:10 UTC daily plus
Sat 13:10/19:10 and Sun 01:10/03:10 (gaps 2-17 h). Explore = 2021-2022, confirm = 2023-2025, 2026 (partial) reported
separately as extra out-of-sample data, never part of the verdict.

Pricing (same as the shop screens, so results are comparable): spreads with the key-number margin distribution
(cfbpred.dist; NFL parameters nfl_dist.json fit on 2015-21 closes, college cfb_dist.json); Pinnacle's no-vig
spread-implied mean margin mu (shop_screen.mus). Moneyline fair = Pinnacle Shin no-vig (src/nflpred/devig.py; the
repo's moneyline tracks use Shin since 2026-10-04; the old shop screens used multiplicative).

Pinnacle move (per event, between consecutive Pinnacle snapshots t0 -> t1 of that event):
  NFL: t1 - t0 = exactly 1 h.  College: consecutive snapshots with t1 - t0 <= 18 h (the data has no hourly college).
  ML move:     |q_shin(t1) - q_shin(t0)| >= 0.03 (home win prob); move side = the side whose probability rose.
  Spread move: |mu(t1) - mu(t0)| >= 1.0 point (spread-implied mean home margin, so point OR price moves count);
               move side = home if mu rose, else away.
  t1 must be 1 h - 7 days before kickoff.
Rules (bet at t1, the first snapshot after the move; book set AZ = draftkings, fanduel, espnbet, barstool, betmgm,
williamhill_us, betrivers, fanatics, hardrockbet, ballybet; barstool added to the suggested list because it was the
Arizona-legal predecessor of ESPN Bet and is in the AZ set of nfl_shop.py):
  R1 NFL moneyline, R2 NFL spread, R3 college moneyline, R4 college spread.
  Candidate = an AZ book's quote at t1 on the MOVE SIDE (the stale side) with EV >= 2% vs Pinnacle's t1 fair
  (ML: q*dec - 1; spread: P(win)*(dec-1) - P(lose) at the book's own number under mu(t1)), price -200..+200.
  One bet per game per rule: the first t1 with a candidate, the best-EV candidate at that t1 (1 unit flat).
Grading: CLV = EV of the bet under Pinnacle's LAST pre-kickoff snapshot of that market (Shin ML / mu at the bet's
  number); bets whose last Pinnacle snapshot is not strictly after t1 are dropped (CLV would be trivially = EV).
  ROI vs final results (nflverse schedules / CFBD games). Latency check: the same book and side filled one snapshot
  LATER (NFL: t1 + 1 h; college: the book's next snapshot), whatever its price then, CLV vs the same close.
Pass (each rule, CONFIRM seasons only, Bonferroni over 4 rules -- 8 after the amendment below): >= 100 bets;
  mean CLV > 0 at one-sided p < 0.05/4 = 0.0125 (0.00625 after the amendment); ROI >= 0; mean CLV > 0 in >= 2 of 3 confirm seasons; NFL only: delayed (+1 h) fill mean CLV > 0
  (live checks are hourly, so the edge must survive hour-scale latency; for college the next snapshot is 2-17 h later,
  a much harsher test than the live hourly check, so it is reported, not required).
  LEAD = not a pass, but confirm mean CLV > 0 at unadjusted one-sided p < 0.05.  FAIL = otherwise.
Informational only (no verdict): explore-season results; by season / book / hours before kickoff; bets where the book's
  quote was unchanged since t0; how much of Pinnacle's move survives to its close; a no-move baseline (same book set,
  threshold and pricing at snapshots WITHOUT a qualifying move, any side, first per game) to see whether conditioning on
  a move adds anything over plain shopping.
Lag profile (all books): for each Pinnacle move (ML >= 3 pts prob, spread >= 1 pt; NFL t1 >= 3 h before kickoff) and each
  book quoting at t0 and t1: the book "matched" at time tk if its own no-vig prob (Shin) / own spread-implied mu moved
  >= half of Pinnacle's move in the same direction since t0. Moves where the book already sat >= half a move toward
  Pinnacle's new line at t0 ("book led") are counted separately and excluded from the lag shares. Reported: share
  unmatched at t1 (0-1 h after the move), t1+1 h, t1+2 h (college: t1 and the next snapshot).
Kambi cluster (descriptive): per season, share of co-quoted snapshots where two books post identical ML (both prices)
  and identical spreads (point + both prices), all pairs; and co-lag P(B unmatched | A unmatched) at t1 on NFL ML moves.
  Pre-declared reading: betrivers + ballybet are "one feed" if they post identical ML quotes on >= 80% of co-quoted
  snapshots in the confirm seasons AND P(ballybet unmatched | betrivers unmatched) >= 0.8 on NFL ML moves.

AMENDMENT 2026-10-07, before any confirm-season (2023+) data was loaded. A code-debugging pass on the EXPLORE seasons
only (EXPLORE_ONLY=1 restricts loading to them) showed the primary rules are too rare to ever reach the 100-bet bar:
NFL 2022 gave 4 (R1) and 7 (R2) bets from 59 ML / 106 spread moves; college 2021-22 gave 22 (R3) and 40 (R4). The
explore CLVs of those handfuls were also printed (meaningless at that size and not used). Added, and frozen now:
  R1b/R2b/R3b/R4b ("loose"): same as R1-R4 but ML move >= 2 pts prob, spread move >= 0.5 point, EV >= 1%.
Bonferroni is now over 8 rules: one-sided p < 0.05/8 = 0.00625. The lag profile keeps the primary 3 pts / 1 pt moves.
The no-move baseline of each rule uses that rule's own EV bar and excludes that rule's own move snapshots.

Usage: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/research/round3/stale_after_move.py nfl|cfb
Output: output/research/round3/stale_after_move_{nfl,cfb}.json and *_bets.parquet.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "research" / "cfb"), str(ROOT / "scripts" / "research")]
from cfbpred import dist as DI  # noqa: E402
from nflpred import devig  # noqa: E402

import price_screen as PS  # noqa: E402
import shop_screen as SS  # noqa: E402

OUT = ROOT / "output" / "research" / "round3"
AZ = ["draftkings", "fanduel", "espnbet", "barstool", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]
KAMBI = ["betrivers", "sugarhouse", "unibet_us", "unibet", "twinspires", "betparx", "barstool", "ballybet"]
ML_MOVE, SP_MOVE, EV_MIN, PLO, PHI = 0.03, 1.0, 0.02, -200, 200
RULES = {"nfl": [("R1 NFL moneyline", "ml", 0.03, 0.02), ("R2 NFL spread", "sp", 1.0, 0.02),
                 ("R1b NFL moneyline loose", "ml", 0.02, 0.01), ("R2b NFL spread loose", "sp", 0.5, 0.01)],
         "cfb": [("R3 college moneyline", "ml", 0.03, 0.02), ("R4 college spread", "sp", 1.0, 0.02),
                 ("R3b college moneyline loose", "ml", 0.02, 0.01), ("R4b college spread loose", "sp", 0.5, 0.01)]}
N_RULES = 8
ALPHA = 0.05 / N_RULES
EXPLORE = {"nfl": [2022], "cfb": [2021, 2022]}
CONFIRM = {"nfl": [2023, 2024, 2025], "cfb": [2023, 2024, 2025]}
H = pd.Timedelta(hours=1)
COLS = ["requested_ts", "event_id", "commence_time", "home", "away", "book", "ml_home", "ml_away",
        "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"]


# ---------------------------------------------------------------- helpers
def shin_vec(h, a):
    """Vectorized devig.shin (home no-vig prob); checked against devig.shin in main()."""
    h, a = np.asarray(h, float), np.asarray(a, float)
    x, y = PS.imp(h), PS.imp(a)
    s = x + y
    lo, hi = np.zeros_like(s), np.full_like(s, 0.5)

    def p(q, z):
        return (np.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z))
    for _ in range(60):
        z = (lo + hi) / 2
        over = p(x, z) + p(y, z) > 1
        lo, hi = np.where(over, z, lo), np.where(over, hi, z)
    z = (lo + hi) / 2
    ph, pa = p(x, z), p(y, z)
    out = ph / (ph + pa)
    return np.where(s <= 1, x / s, out)


def read(files, season_of):
    parts = []
    for f in files:
        d = pd.read_csv(f, usecols=COLS)
        d["t"] = pd.to_datetime(d.pop("requested_ts"), utc=True)
        d["ko"] = pd.to_datetime(d.pop("commence_time"), utc=True)
        d = d[(d.t < d.ko) & (d.ko - d.t <= pd.Timedelta(days=8))]
        d["season"] = season_of(f)
        parts.append(d)
    o = pd.concat(parts, ignore_index=True)
    o = o.drop_duplicates(["event_id", "t", "book"])
    o["book"] = o.book.astype("category")
    return o


def load(league):
    if league == "nfl":
        DI.PARAMS = ROOT / "nfl_dist.json"
        DI.params.cache_clear()
        DI._WCACHE.clear()
        so = lambda f: int(Path(f).name[9:13])  # noqa: E731
        O = read(sorted(glob.glob(str(ROOT / "data/historical_odds/dense/nfl_odds_*.csv.gz"))), so)
        Pn = read(sorted(glob.glob(str(ROOT / "data/historical_odds/dense_pin/nfl_odds_*.csv.gz"))), so)
    else:
        so = lambda f: int(re.findall(r"(\d{4})", f)[-1])  # noqa: E731
        O = read(sorted(glob.glob(str(ROOT / "data/historical_odds/cfb/cfb_odds_*.csv.gz"))), so)
        Pn = read(sorted(glob.glob(str(ROOT / "data/historical_odds/cfb_pin/cfb_odds_*.csv.gz"))), so)
    Pn = Pn[Pn.book == "pinnacle"]
    O = O[O.book != "pinnacle"]
    if os.environ.get("EXPLORE_ONLY"):          # code debugging pass: explore seasons only, confirm years unseen
        O, Pn = O[O.season.isin(EXPLORE[league])], Pn[Pn.season.isin(EXPLORE[league])]
    O["book"] = O.book.cat.remove_unused_categories()
    return O, Pn


def results(league, O):
    if league == "nfl":
        import nfl_shop
        from nflpred import data as ND
        return nfl_shop.results(O.drop_duplicates("event_id"), ND.load_schedules())[["margin", "total"]]
    return SS.results_full(O.drop_duplicates("event_id"))[["margin", "total"]]


def pin_table(Pn):
    pin = SS.mus(Pn.drop_duplicates(["event_id", "t"]).reset_index(drop=True).assign(tot_point=np.nan, tot_over_price=np.nan, tot_under_price=np.nan))
    ok = pin.ml_home.notna() & pin.ml_away.notna()
    pin["q"] = np.where(ok, shin_vec(pin.ml_home.fillna(100), pin.ml_away.fillna(100)), np.nan)
    return pin.sort_values(["event_id", "t"]).reset_index(drop=True)


def closes(pin):
    out = {}
    for col, name in (("q", "c_q_ml"), ("mu_m", "c_mu_m")):
        x = pin.dropna(subset=[col]).groupby("event_id").last()
        out[name] = x[col]
        out[name + "_t"] = x["t"]
    c = pd.DataFrame(out)
    c["c_mu_t"] = np.nan
    return c


def moves(pin, league, ml_thr=ML_MOVE, sp_thr=SP_MOVE):
    P = pin.copy()
    g = P.groupby("event_id")
    P["t0"], P["q0"], P["mu0"] = g.t.shift(), g.q.shift(), g.mu_m.shift()
    gap = P.t - P.t0
    okgap = (gap == H) if league == "nfl" else (gap <= pd.Timedelta(hours=18))
    hb = P.ko - P.t
    ok = okgap & (hb >= H) & (hb <= pd.Timedelta(days=7))
    P["dq"], P["dmu"] = P.q - P.q0, P.mu_m - P.mu0
    ml = P[ok & (P.dq.abs() >= ml_thr)].copy()
    ml["side"] = np.where(ml.dq > 0, "home", "away")
    sp = P[ok & (P.dmu.abs() >= sp_thr)].copy()
    sp["side"] = np.where(sp.dmu > 0, "home", "away")
    return P, ml, sp


def with_next(Oaz, league):
    """Each AZ book row plus that book's quote at the next snapshot (NFL: exactly t+1h)."""
    O = Oaz.sort_values(["event_id", "book", "t"]).copy()
    g = O.groupby(["event_id", "book"], observed=True)
    for c in ("t", "ml_home", "ml_away", "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"):
        O["n_" + c] = g[c].shift(-1)
    if league == "nfl":
        bad = O.n_t != O.t + H
        for c in [c for c in O.columns if c.startswith("n_")]:
            O.loc[bad, c] = np.nan if c != "n_t" else pd.NaT
    return O


def price_cands(mv, Oaz, market, prev):
    """Quotes at t1 on the move side, with EV vs Pinnacle at t1, plus delayed-fill quote and t0 quote."""
    keep = ["event_id", "t", "t0", "ko", "season", "side", "q", "q0", "mu_m", "mu0"]
    C = mv[keep].merge(Oaz.drop(columns=["ko", "season", "home", "away"]), on=["event_id", "t"], how="inner")
    C = C.merge(prev, left_on=["event_id", "t0", "book"], right_on=["event_id", "t", "book"], how="left", suffixes=("", "_drop"))
    C = C.drop(columns=[c for c in C.columns if c.endswith("_drop")]).reset_index(drop=True)
    home = (C.side == "home").to_numpy()
    if market == "ml":
        C["price"] = np.where(home, C.ml_home, C.ml_away)
        C["n_price"] = np.where(home, C.n_ml_home, C.n_ml_away)
        C["p0_price"] = np.where(home, C.p0_ml_home, C.p0_ml_away)
        C["point"] = np.nan
        C["n_point"] = np.nan
        C["unchanged"] = C.price == C.p0_price
        qs = np.where(home, C.q, 1 - C.q)
        C["ev"] = qs * PS.dec(C.price) - 1
    else:
        C["price"] = np.where(home, C.sp_home_price, C.sp_away_price)
        C["point"] = np.where(home, C.sp_home_point, C.sp_away_point)
        C["n_price"] = np.where(home, C.n_sp_home_price, C.n_sp_away_price)
        C["n_point"] = np.where(home, C.n_sp_home_point, C.n_sp_away_point)
        p0pt = np.where(home, C.p0_sp_home_point, C.p0_sp_away_point)
        p0pr = np.where(home, C.p0_sp_home_price, C.p0_sp_away_price)
        C["unchanged"] = (C.point == p0pt) & (C.price == p0pr)
        C = C.dropna(subset=["mu_m", "point", "price"]).reset_index(drop=True)
        C["ev"] = np.nan
        for side in ("home", "away"):
            j = np.where((C.side == side).to_numpy())[0]
            if not len(j):
                continue
            cdf, pmf, K = SS.cdf_tables(C.mu_m.to_numpy()[j], "m")
            w, p, l = SS.probs(cdf, pmf, K, C.point.to_numpy(float)[j], side)
            C.loc[j, "ev"] = w * (PS.dec(C.price.to_numpy(float)[j]) - 1) - l
    C["market"] = "spread" if market == "sp" else "ml"
    return C[C.price.between(PLO, PHI)]


def pick(C, ev_min):
    q = C[C.ev >= ev_min]
    first = q.groupby("event_id").t.min()
    q = q[q.t == q.event_id.map(first)].sort_values("ev", ascending=False).groupby("event_id").head(1)
    return q


def grade(q, close, R):
    ctc = "c_q_ml_t" if (q.market == "ml").all() else "c_mu_m_t"
    q = q.join(close, on="event_id")
    q = q[q[ctc] > q.t]
    B = SS.grade(q.drop(columns=[c for c in close.columns if c in q.columns]).set_index("event_id"), close, R)
    # delayed fill (+1 snapshot), same side, the book's price/number then
    D = B[["event_id", "market", "side", "n_point", "n_price"]].rename(columns={"n_point": "point", "n_price": "price"})
    D = D.dropna(subset=["price"] + (["point"] if (B.market == "spread").all() else []))
    if len(D):
        Dg = SS.grade(D.set_index("event_id"), close, R)
        B = B.merge(Dg[["event_id", "clv", "pnl"]].rename(columns={"clv": "clv_delay", "pnl": "pnl_delay"}), on="event_id", how="left")
    else:
        B["clv_delay"], B["pnl_delay"] = np.nan, np.nan
    return B


def stats(x):
    s = SS.stats(x)
    if s.get("bets"):
        s["p"] = round(s["p"], 5)
    return s


def rule_report(B, league, pinmove):
    out = {}
    for lab, seasons in (("explore", EXPLORE[league]), ("confirm", CONFIRM[league]), ("2026_oos", [2026])):
        x = B[B.season.isin(seasons)]
        if not len(x):
            continue
        st = stats(x)
        st["by_season"] = {int(s): stats(g) for s, g in x.groupby("season")}
        d = x.dropna(subset=["clv_delay"])
        st["delayed_fill"] = {"bets": int(len(d)), "mean_clv": round(float(d.clv_delay.mean()), 4) if len(d) else None,
                              "se": round(float(d.clv_delay.std() / math.sqrt(len(d))), 4) if len(d) > 1 else None,
                              "roi": round(float(d.pnl_delay.mean()), 4) if len(d) else None,
                              "missing_next_quote": int(x.clv_delay.isna().sum())}
        st["unchanged_since_t0"] = stats(x[x.unchanged == True])  # noqa: E712
        st["moved_since_t0"] = stats(x[x.unchanged != True])  # noqa: E712
        st["by_book"] = {b: stats(g) for b, g in x.groupby("book", observed=True)}
        hb = (x.ko - x.t).dt.total_seconds() / 3600
        st["by_hours_before"] = {lab2: stats(x[(hb > lo) & (hb <= hi)]) for lab2, lo, hi in
                                 (("1-6h", 1, 6), ("6-24h", 6, 24), ("1-3d", 24, 72), ("3-7d", 72, 170))}
        st["move_retained_at_close"] = pinmove(x)
        if lab == "confirm":
            per = x.groupby("season").clv.mean()
            n_pos = int((per > 0).sum())
            st["seasons_clv_positive"] = f"{n_pos}/{len(per)}"
            dl = st["delayed_fill"]["mean_clv"]
            crit = {"bets>=100": st.get("bets", 0) >= 100, "p<0.00625": st.get("p", 1) < ALPHA and st.get("mean_clv", 0) > 0,
                    "roi>=0": st.get("roi", -1) >= 0, "clv+ in >=2/3 seasons": n_pos >= 2}
            if league == "nfl":
                crit["delayed_fill_clv>0"] = dl is not None and dl > 0
            st["criteria"] = crit
            st["verdict"] = "PASS" if all(crit.values()) else ("LEAD" if st.get("p", 1) < 0.05 and st.get("mean_clv", 0) > 0 else "FAIL")
        out[lab] = st
    return out


def retained(close, col, ccol, base0, base1):
    def f(x):
        j = x
        mv = j[base1] - j[base0]
        r = (j[ccol] - j[base0]) / mv
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
        return {"median_share_of_move_kept_at_close": round(float(r.median()), 3) if len(r) else None,
                "share_fully_reverted(<=0)": round(float((r <= 0).mean()), 3) if len(r) else None,
                "share_kept_>=1": round(float((r >= 1).mean()), 3) if len(r) else None}
    return f


# ---------------------------------------------------------------- lag profile + Kambi
@lru_cache(maxsize=None)
def mu_sp(pt, hp, ap):
    return DI.mu_from_spread(pt, hp, ap)


def book_state(O, market):
    O = O.copy()
    if market == "ml":
        ok = O.ml_home.notna() & O.ml_away.notna()
        O = O[ok]
        O["v"] = shin_vec(O.ml_home, O.ml_away)
    else:
        O = O.dropna(subset=["sp_home_point", "sp_home_price", "sp_away_price"])
        O["v"] = [mu_sp(float(a), float(b), float(c)) for a, b, c in zip(O.sp_home_point, O.sp_home_price, O.sp_away_price)]
    return O[["event_id", "t", "book", "v"]]


def lag_profile(O, Pall, league, market):
    thr = ML_MOVE if market == "ml" else SP_MOVE
    dcol, vcol, v0col = ("dq", "q", "q0") if market == "ml" else ("dmu", "mu_m", "mu0")
    gap = Pall.t - Pall.t0
    okgap = (gap == H) if league == "nfl" else (gap <= pd.Timedelta(hours=18))
    hb = Pall.ko - Pall.t
    M = Pall[okgap & (Pall[dcol].abs() >= thr) & (hb >= (3 * H if league == "nfl" else H)) & (hb <= pd.Timedelta(days=7))]
    M = M[["event_id", "t0", "t", "season", dcol, v0col, vcol]].rename(columns={"t": "t1"}).reset_index(drop=True)
    M["mid"] = np.arange(len(M))
    if league == "nfl":
        offs = {"t0": M.t0, "t1": M.t1, "t1+1h": M.t1 + H, "t1+2h": M.t1 + 2 * H}
    else:
        nxt = O[["event_id", "t"]].drop_duplicates().sort_values(["event_id", "t"])
        nxt["tn"] = nxt.groupby("event_id").t.shift(-1)
        M = M.merge(nxt.rename(columns={"t": "t1"}), on=["event_id", "t1"], how="left")
        offs = {"t0": M.t0, "t1": M.t1, "next": M.tn}
    keys = set()
    for v in offs.values():
        keys |= set(zip(M.event_id, v))
    kk = pd.DataFrame(list(keys), columns=["event_id", "t"]).dropna()
    S = book_state(O.merge(kk, on=["event_id", "t"]), market)
    base = M[["mid", "event_id", "season", dcol, v0col]]
    L = None
    for lab, v in offs.items():
        x = pd.DataFrame({"mid": M.mid, "event_id": M.event_id, "t": v}).dropna(subset=["t"]).merge(S, on=["event_id", "t"])
        x = x[["mid", "book", "v"]].rename(columns={"v": "v_" + lab})
        L = base.merge(x, on="mid") if L is None else L.merge(x, on=["mid", "book"], how="left")
    sgn = np.sign(L[dcol])
    half = 0.5 * L[dcol].abs()
    L["led"] = sgn * (L["v_t0"] - L[v0col]) >= half
    labs = [k for k in offs if k != "t0"]
    for lab in labs:
        L["un_" + lab] = np.where(L["v_" + lab].notna(), (sgn * (L["v_" + lab] - L["v_t0"]) < half).astype(float), np.nan)
    rep = {}
    for lab_s, seasons in (("explore", EXPLORE[league]), ("confirm", CONFIRM[league])):
        x = L[L.season.isin(seasons) & L.v_t1.notna()]
        r = {}
        for b, g in x.groupby("book", observed=True):
            nl = g[~g.led]
            if len(g) < 30:
                continue
            r[b] = {"moves": int(len(g)), "book_led_share": round(float(g.led.mean()), 3),
                    **{"unmatched_" + lab: round(float(nl["un_" + lab].mean()), 3) for lab in labs}}
        rep[lab_s] = dict(sorted(r.items(), key=lambda kv: -kv[1]["unmatched_t1"]))
    rep["n_moves"] = {lab_s: int(M.season.isin(s).sum()) for lab_s, s in (("explore", EXPLORE[league]), ("confirm", CONFIRM[league]))}
    return rep, L


def colag(L, league):
    x = L[~L.led & L.un_t1.notna() & L.season.isin(CONFIRM[league])]
    W = x.pivot_table(index="mid", columns="book", values="un_t1", observed=True)
    out = {}
    books = [b for b in W.columns if W[b].notna().sum() >= 50]
    for a in books:
        for b in books:
            if a >= b:
                continue
            m = W[[a, b]].dropna()
            if len(m) < 50:
                continue
            A, Bb = m[a].to_numpy(), m[b].to_numpy()
            pa, pb = A.mean(), Bb.mean()
            den = math.sqrt(pa * (1 - pa) * pb * (1 - pb))
            out[f"{a}|{b}"] = {"n": int(len(m)), "unmatched_a": round(float(pa), 3), "unmatched_b": round(float(pb), 3),
                               "p_b_un_given_a_un": round(float(Bb[A == 1].mean()), 3) if (A == 1).any() else None,
                               "p_a_un_given_b_un": round(float(A[Bb == 1].mean()), 3) if (Bb == 1).any() else None,
                               "phi": round(float(((A * Bb).mean() - pa * pb) / den), 3) if den > 0 else None}
    return out


def identity(O):
    out = {}
    for s, d in O.groupby("season"):
        d = d.copy()
        d["k_ml"] = pd.factorize(d.ml_home.astype(str) + "/" + d.ml_away.astype(str))[0]
        d.loc[d.ml_home.isna() | d.ml_away.isna(), "k_ml"] = -1
        d["k_sp"] = pd.factorize(d.sp_home_point.astype(str) + "/" + d.sp_home_price.astype(str) + "/" + d.sp_away_price.astype(str))[0]
        d.loc[d.sp_home_point.isna() | d.sp_home_price.isna(), "k_sp"] = -1
        res = {}
        for mk in ("k_ml", "k_sp"):
            W = d.pivot_table(index=["event_id", "t"], columns="book", values=mk, aggfunc="first", observed=True)
            books = [b for b in W.columns if (W[b] >= 0).sum() >= 2000]
            for i, a in enumerate(books):
                for b in books[i + 1:]:
                    A, Bb = W[a].to_numpy(), W[b].to_numpy()
                    both = (A >= 0) & (Bb >= 0) & ~np.isnan(A) & ~np.isnan(Bb)
                    if both.sum() < 2000:
                        continue
                    key = "|".join(sorted((a, b)))
                    res.setdefault(key, {})["n" if mk == "k_ml" else "n_sp"] = int(both.sum())
                    res[key][mk[2:] + "_identical"] = round(float((A[both] == Bb[both]).mean()), 3)
        out[int(s)] = dict(sorted(res.items(), key=lambda kv: -kv[1].get("ml_identical", 0)))
    return out


def base_pool(O, pin):
    """All AZ quotes (ML + spread, any side) priced vs Pinnacle at the same snapshot, EV >= 1% (lowest bar used)."""
    pm = pin[["event_id", "t", "mu_m", "mu_t", "q"]].rename(columns={"q": "q_ml"})
    mon = O.ko.dt.strftime("%Y-%m")
    parts = []
    for m in sorted(mon.unique()):
        Om = O[(mon == m) & O.book.isin(AZ)].assign(last_update=pd.NaT, tot_point=np.nan, tot_over_price=np.nan, tot_under_price=np.nan)
        C = SS.candidates(Om, pm, AZ)
        parts.append(C[C.market.isin(["ml", "spread"]) & (C.ev >= 0.01)])
    return pd.concat(parts, ignore_index=True)


def baseline(pool, movers, close, R, league, market, ev_min):
    """No-move shop baseline (informational): AZ quotes at snapshots WITHOUT a qualifying move, any side."""
    C = pool[pool.market == ("ml" if market == "ml" else "spread")]
    mv = set(zip(movers.event_id, movers.t))
    C = C[[k not in mv for k in zip(C.event_id, C.t)]]
    q = pick(C, ev_min)
    ctc = "c_q_ml_t" if market == "ml" else "c_mu_m_t"
    q = q.join(close[[ctc]], on="event_id")
    q = q[q[ctc] > q.t].drop(columns=[ctc])
    B = SS.grade(q.set_index("event_id"), close, R)
    return {lab: stats(B[B.season.isin(s)]) for lab, s in (("explore", EXPLORE[league]), ("confirm", CONFIRM[league]))}


# ---------------------------------------------------------------- main
def main(league):
    OUT.mkdir(parents=True, exist_ok=True)
    chk = [(-150, 130), (-300, 250), (110, -130), (-105, -115)]
    for h, a in chk:
        assert abs(shin_vec([h], [a])[0] - devig.shin(h, a)) < 1e-9
    O, Pn = load(league)
    print(league, "rows", len(O), "pin rows", len(Pn), "events", O.event_id.nunique(), flush=True)
    R = results(league, O)
    print("events with results", len(R), flush=True)
    pin = pin_table(Pn)
    close = closes(pin)
    Pall, ml_mv, sp_mv = moves(pin, league)
    print("ML moves", ml_mv.groupby("season").size().to_dict(), "spread moves", sp_mv.groupby("season").size().to_dict(), flush=True)
    pool = base_pool(O, pin)
    Oaz = with_next(O[O.book.isin(AZ)], league)
    prev = Oaz[["event_id", "t", "book", "ml_home", "ml_away", "sp_home_point", "sp_home_price", "sp_away_point", "sp_away_price"]]
    prev = prev.rename(columns={c: "p0_" + c for c in prev.columns if c not in ("event_id", "t", "book")})
    report = {"league": league, "explore": EXPLORE[league], "confirm": CONFIRM[league], "rules": {}}
    allb = []
    for name, market, mthr, ev_min in RULES[league]:
        mv = moves(pin, league, ml_thr=mthr, sp_thr=mthr)[1 if market == "ml" else 2]
        C = price_cands(mv, Oaz, market, prev)
        q = pick(C, ev_min)
        B = grade(q, close, R).assign(rule=name)
        allb.append(B)
        f = retained(close, None, "c_q_ml" if market == "ml" else "c_mu_m", "q0" if market == "ml" else "mu0", "q" if market == "ml" else "mu_m")
        rep = rule_report(B, league, f)
        rep["thresholds"] = {"move": mthr, "ev_min": ev_min}
        rep["candidates_at_t1"] = {"quotes": int(len(C)), "qualifying_quotes": int((C.ev >= ev_min).sum()),
                                   "moves_with_a_candidate": int(C[C.ev >= ev_min][["event_id", "t"]].drop_duplicates().shape[0]),
                                   "moves": int(len(mv))}
        # Kambi: how often betrivers and ballybet both qualify at the same move
        qq = C[C.ev >= ev_min]
        bm = qq.assign(book=qq.book.astype(str)).groupby(["event_id", "t"]).book.apply(lambda s: set(s)).tolist()
        rep["co_qualify"] = {f"{a}&{b}": {"a": int(sum(a in s for s in bm)), "b": int(sum(b in s for s in bm)),
                                          "both": int(sum(a in s and b in s for s in bm))}
                             for a, b in (("betrivers", "ballybet"), ("draftkings", "fanduel"), ("betrivers", "draftkings"))}
        rep["no_move_baseline"] = baseline(pool, mv, close, R, league, market, ev_min)
        report["rules"][name] = rep
        c = rep.get("confirm", {})
        print(name, {k: c.get(k) for k in ("bets", "mean_ev_at_bet", "mean_clv", "se", "p", "roi", "roi_se", "seasons_clv_positive", "verdict")},
              "delayed", c.get("delayed_fill"), "| explore", {k: rep.get("explore", {}).get(k) for k in ("bets", "mean_clv", "p", "roi")},
              "| baseline", rep["no_move_baseline"]["confirm"], flush=True)
    if not os.environ.get("EXPLORE_ONLY"):
        pd.concat(allb).to_parquet(OUT / f"stale_after_move_{league}_bets.parquet")
    lag = {}
    Ls = {}
    for market in ("ml", "sp"):
        lag[market], Ls[market] = lag_profile(O, Pall, league, market)
        print("lag", market, json.dumps(lag[market]["confirm"])[:1500], flush=True)
    report["lag_profile"] = lag
    if league == "nfl":
        report["colag_ml_t1"] = colag(Ls["ml"], league)
        report["colag_sp_t1"] = colag(Ls["sp"], league)
    report["identical_quotes"] = identity(O)
    if os.environ.get("EXPLORE_ONLY"):
        Path(os.environ["EXPLORE_ONLY"]).write_text(json.dumps(report, indent=1, default=str))
        return
    (OUT / f"stale_after_move_{league}.json").write_text(json.dumps(report, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
