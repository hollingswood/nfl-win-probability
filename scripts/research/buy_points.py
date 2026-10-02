"""Buying points at REAL book prices: alternate-spread price curves, fair value of each half point, and whether any
systematic buy (or sell) is +EV.

Stages (outputs in output/research/buy_points.{json,md}, buy_points_frozen.json):
  python scripts/research/buy_points.py curve     # price per half point by book / crossed number / fav-dog (2023-2025,
                                                  # descriptive: book pricing behaviour, no rule selection)
  python scripts/research/buy_points.py dev       # 2023 ONLY: EV of every alt line, marginal value vs price, rule grid
  python scripts/research/buy_points.py freeze    # writes buy_points_frozen.json (<=3 rules; refuses overwrite)
  BUYPTS_HOLDOUT=I_HAVE_FROZEN python scripts/research/buy_points.py holdout   # 2024-2025, ONCE
  python scripts/research/buy_points.py dash      # descriptive: dashboard-qualifying spread bets, real alt prices
  python scripts/research/buy_points.py report    # prints the json layout (buy_points.md is written from it by hand)
Optional env BUYPTS_CACHE=<dir> caches the per-season ladder tables (parquet).

Data / model
  * Alternate spreads (data/historical_odds/alternates/, per-event endpoint): every (book, team, point, price) at two
    snapshots per game: "early" (Fri 21:40 UTC for Sunday games, kickoff-24h otherwise) and "close" (~75 min pre-kick).
    Most books' alt ladders EXCLUDE the main line, so each book's main line + price is taken from the main-market file
    (data/historical_odds/nfl_odds_{season}.csv.gz) at the nearest snapshot of the same game (<= 90 min apart) and
    spliced into the ladder; snapshots with no main-market snapshot within 90 min are dropped (counted).
  * Fair expected home margin at a snapshot: price-implied mu of every book's MAIN spread + juice under the corrected
    key-number distribution of teasers_v2.py (Normal(mu, 13.4) x w(|margin|), raked to 2012-2019 margin frequencies,
    fit on 2012-2019 only). fair_sharp = median over lowvig/betonlineag (fallback: all books), fair_cons = median
    over all books. CLV = EV of the bet under the CLOSE snapshot's fair_sharp. For close-snapshot bets CLV ~= EV by
    construction (same moment), so their only independent test is realized ROI.
  * EV per unit = P(cover) * decimal + P(push) - 1. Lines kept: allowed books (my_books.json), |point - that book's
    main point| <= 3.5 (consensus main point if the book has no main line).
  * Cents: American odds on a continuous scale c(a) = a + 100 (a < 0) or a - 100 (a > 0); -110 -> -130 = 20 cents,
    +105 -> -115 = 20 cents (same convention as spread_bets._shift_price).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))
sys.path.insert(0, str(ROOT / "src"))
import teasers_v2 as T2  # noqa: E402
from nflpred import odds as O, spread_bets as SB  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "buy_points.json"
FROZEN = OUT / "buy_points_frozen.json"
MD = OUT / "buy_points.md"
DEV = (2023,)
HOLD = (2024, 2025)
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
CACHE = os.environ.get("BUYPTS_CACHE")
MAXDIST = 3.5
TOL = pd.Timedelta("90min")
ASSUMED = {"other": 10, "3": 20, "7": 15}
KEYS = (3, 7)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def imp(a):
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def cents(a):
    a = np.asarray(a, float)
    return np.where(a < 0, a + 100, a - 100)


def am_from_dec(d):
    d = np.asarray(d, float)
    return np.where(d >= 2, (d - 1) * 100, -100 / np.maximum(d - 1, 1e-9))


def r4(x):
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else round(float(x), 4)


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return str(o)
    return str(o)


def store(key, res):
    J = json.loads(JSON.read_text()) if JSON.exists() else {}
    J[key] = res
    JSON.write_text(json.dumps(J, indent=1, default=_jd))


def mse(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None, None
    return r4(x.mean()), r4(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None


# ======================================================================================= tables
def old_dist() -> T2.Dist:
    r = SB.load_rules()
    w = np.array([r["_weights"].get(abs(int(k)), 1.0) for k in T2.KS])
    return T2.Dist(r["margin"]["sigma"], w)


def main_market(season, g, key):
    o = T2.match(T2.load_odds([season]), g)
    o = o.merge(g[["game_id", "kick", "m"]], on="game_id")
    o = o[(o.requested_ts < o.kick) & o.sp_home_point.notna() & o.sp_home_price.notna() & o.sp_away_price.notna()]
    o = o[(o.sp_home_point + o.sp_away_point).abs() < 1e-9]
    o = o[o.sp_home_price.between(-145, 125) & o.sp_away_price.between(-145, 125) & (o.sp_home_point.abs() <= 30)]
    o = o[(o.sp_home_point - o.groupby(["event_id", "requested_ts"]).sp_home_point.transform("median")).abs() <= 2.5]
    o = o[(o.sp_home_point * 2) % 1 == 0]
    o = o.sort_values("last_update").drop_duplicates(["game_id", "requested_ts", "book"], keep="last")
    ih, ia = imp(o.sp_home_price.values), imp(o.sp_away_price.values)
    o["q_home"] = ih / (ih + ia)
    o["overround"] = ih + ia - 1
    o = o[o.overround.between(-0.01, 0.12)].copy()
    o["mu_book"] = key.implied_mu(o.sp_home_point.values, o.q_home.values)
    k2 = ["game_id", "requested_ts"]
    cons = o.groupby(k2).agg(mu_cons=("mu_book", "median"), n_books=("book", "nunique"),
                             pt_cons=("sp_home_point", "median"))
    sh = o[o.book.isin(SHARP)].groupby(k2).agg(mu_sharp=("mu_book", "median"), n_sharp=("book", "nunique"))
    snap = cons.join(sh).reset_index()
    snap["sharp_fallback"] = snap.mu_sharp.isna()
    snap["mu_sharp"] = snap.mu_sharp.fillna(snap.mu_cons)
    last = snap.sort_values("requested_ts").groupby("game_id").tail(1)
    snap = snap.merge(last[["game_id", "mu_sharp", "mu_cons", "requested_ts"]].rename(
        columns={"mu_sharp": "mu_close_sharp", "mu_cons": "mu_close_cons", "requested_ts": "close_ts"}), on="game_id")
    return o, snap


def load_alt(season, g):
    a = pd.read_csv(ROOT / "data" / "historical_odds" / "alternates" / f"alternate_spreads_{season}.csv.gz")
    a["season"] = season
    a["requested_ts"] = pd.to_datetime(a.requested_ts, utc=True)
    a["commence"] = pd.to_datetime(a.commence_time, utc=True)
    a["abbr"] = a.player.map(O.TEAM_ABBR)
    a["side"] = np.where(a.abbr == a.home, "home", np.where(a.abbr == a.away, "away", None))
    a = a[a.side.notna() & a.over_price.notna() & a.point.notna()]
    a = T2.match(a, g).merge(g[["game_id", "kick", "m"]], on="game_id")
    a = a[a.requested_ts < a.kick]
    a = a.sort_values("snapshot_ts").drop_duplicates(["game_id", "requested_ts", "book", "side", "point"], keep="last")
    return a.rename(columns={"over_price": "price"})[
        ["game_id", "season", "requested_ts", "kick", "m", "book", "side", "point", "price"]]


def build(season) -> tuple[pd.DataFrame, dict]:
    """Ladder rows: one per (game, alt snapshot, allowed book, side, point) incl. the book's main line (src=main)."""
    if CACHE:
        p = Path(CACHE) / f"ladder_{season}.parquet"
        pj = Path(CACHE) / f"ladder_{season}.json"
        if p.exists() and pj.exists():
            return pd.read_parquet(p), json.loads(pj.read_text())
    key, _, _ = T2.load_dist()
    old = old_dist()
    allowed = O.load_allowed_books()
    g = T2.games([season])
    o, snap = main_market(season, g, key)
    a = load_alt(season, g)
    meta = {"season": season, "alt_rows": int(len(a)), "alt_books": sorted(a.book.unique().tolist())}
    # alt snapshot -> nearest main-market snapshot of the same game
    K = a[["game_id", "requested_ts"]].drop_duplicates().sort_values("requested_ts")
    M = snap[["game_id", "requested_ts"]].rename(columns={"requested_ts": "main_ts"}).sort_values("main_ts")
    K = pd.merge_asof(K, M, left_on="requested_ts", right_on="main_ts", by="game_id", direction="nearest",
                      tolerance=TOL)
    meta["alt_snapshots"] = int(len(K))
    meta["alt_snapshots_matched"] = int(K.main_ts.notna().sum())
    K = K[K.main_ts.notna()]
    a = a[a.book.isin(allowed)].merge(K, on=["game_id", "requested_ts"])
    # each allowed book's main line at the matched main snapshot
    mb = o[o.book.isin(allowed)]
    mains = []
    for side in ("home", "away"):
        x = mb[["game_id", "requested_ts", "book", f"sp_{side}_point", f"sp_{side}_price", "overround"]].rename(
            columns={"requested_ts": "main_ts", f"sp_{side}_point": "point", f"sp_{side}_price": "price"})
        x["side"] = side
        mains.append(x)
    mains = pd.concat(mains, ignore_index=True)
    mm = K.merge(mains, on=["game_id", "main_ts"]).merge(
        a[["game_id", "requested_ts", "book"]].drop_duplicates(), on=["game_id", "requested_ts", "book"])
    mm = mm.merge(g[["game_id", "kick", "m", "season"]], on="game_id")
    # alt price at the book's own main point vs the main-market price (does the alt ladder include the main line?)
    chk = a.merge(mm[["game_id", "requested_ts", "book", "side", "point", "price"]].rename(columns={"price": "mprice"}),
                  on=["game_id", "requested_ts", "book", "side", "point"])
    meta["alt_includes_main_point_share_by_book"] = {
        b: r4(len(chk[chk.book == b].drop_duplicates(["game_id", "requested_ts", "side"])) /
              max(1, len(mm[mm.book == b].drop_duplicates(["game_id", "requested_ts", "side"]))))
        for b in sorted(mm.book.unique())}
    meta["alt_vs_main_price_at_main_point_cents_by_book"] = {
        b: r4((cents(d.mprice) - cents(d.price)).mean()) for b, d in chk.groupby("book")}
    a = a.assign(src="alt")
    mm = mm.assign(src="main")
    lad = pd.concat([mm.drop(columns=["overround"]), a], ignore_index=True)
    lad = lad.drop_duplicates(["game_id", "requested_ts", "book", "side", "point"], keep="first")   # main wins
    mainpt = mm[["game_id", "requested_ts", "book", "side", "point", "price", "overround"]].rename(
        columns={"point": "main_point", "price": "main_price", "overround": "main_overround"})
    lad = lad.merge(mainpt, on=["game_id", "requested_ts", "book", "side"], how="left")
    lad = lad.merge(snap, left_on=["game_id", "main_ts"], right_on=["game_id", "requested_ts"],
                    suffixes=("", "_snap")).drop(columns=["requested_ts_snap"])
    hs = (lad.side == "home").values
    cons_side_pt = np.where(hs, lad.pt_cons, -lad.pt_cons)
    lad["has_main"] = lad.main_point.notna()
    lad["ref_point"] = np.where(lad.has_main, lad.main_point, cons_side_pt)
    lad["dist"] = lad.point - lad.ref_point                      # + = bought points, - = sold points
    lad = lad[lad.dist.abs() <= MAXDIST + 1e-9].copy()
    hs = (lad.side == "home").values
    lad["hours_before"] = (lad.kick - lad.requested_ts).dt.total_seconds() / 3600
    lad["window"] = np.where(lad.hours_before < 3, "close", "early")
    lad["dec"] = dec(lad.price.values)
    for c in ("sharp", "cons", "close_sharp", "close_cons"):
        pw, pp = key.side_probs(lad[f"mu_{c}"].values, hs, lad.point.values)
        lad[f"pw_{c}"], lad[f"pp_{c}"] = pw, pp
        lad[f"ev_{c}"] = pw * lad.dec.values + pp - 1
    pw, pp = old.side_probs(lad.mu_sharp.values, hs, lad.point.values)
    lad["ev_sharp_oldw"] = pw * lad.dec.values + pp - 1
    sm = np.where(hs, lad.m, -lad.m)
    lad["side_margin"] = sm
    lad["y"] = np.sign(sm + lad.point.values).astype(int)
    lad["profit"] = np.where(lad.y > 0, lad.dec - 1, np.where(lad.y == 0, 0.0, -1.0))
    lad["fav"] = lad.ref_point < 0
    lo, hi = np.minimum(lad.point, lad.ref_point), np.maximum(lad.point, lad.ref_point)
    # integer side margins whose outcome changes between the main point and this point: k in [-hi, -lo]
    for kn in KEYS:
        lad[f"cross{kn}"] = ((-kn >= -hi) & (-kn <= -lo)) | ((kn >= -hi) & (kn <= -lo))
    lad["cross_key"] = lad.cross3 | lad.cross7
    meta["ladder_rows"] = int(len(lad))
    if CACHE:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        lad.to_parquet(Path(CACHE) / f"ladder_{season}.parquet")
        (Path(CACHE) / f"ladder_{season}.json").write_text(json.dumps(meta, default=_jd))
    return lad, meta


def ladders(seasons):
    Ls, metas = [], []
    for s in seasons:
        L, m = build(s)
        Ls.append(L)
        metas.append(m)
    return pd.concat(Ls, ignore_index=True), metas


def steps(L: pd.DataFrame) -> pd.DataFrame:
    """Consecutive half-point steps p -> p+0.5 (buying half a point) within one (game, snapshot, book, side) ladder."""
    k = ["game_id", "requested_ts", "book", "side"]
    L = L.sort_values(k + ["point"])
    nx = L.groupby(k).shift(-1)
    S = L[k + ["season", "window", "point", "price", "src", "ref_point", "dist", "has_main", "pw_sharp", "pp_sharp",
               "ev_sharp", "main_overround"]].copy()
    S["to_point"], S["to_price"], S["to_src"] = nx.point, nx.price, nx.src
    S["to_pw"], S["to_pp"], S["to_ev"] = nx.pw_sharp, nx.pp_sharp, nx.ev_sharp
    S = S[(S.to_point - S.point).sub(0.5).abs() < 1e-9].copy()
    isint_to = (S.to_point % 1 == 0)
    S["num"] = np.where(isint_to, S.to_point, S.point).astype(int)     # the whole number involved
    S["kind"] = np.where(isint_to, "onto", "off")
    S["absn"] = S.num.abs()
    S["cat"] = np.where(S.absn == 3, "3", np.where(S.absn == 7, "7", "other"))
    S["fav"] = (S.point + 0.25) < 0
    S["cents"] = cents(S.price.values) - cents(S.to_price.values)
    S["prob_cost"] = imp(S.to_price.values) - imp(S.price.values)
    # value of the half point: P(side margin lands on the number) (no-vig prob terms: onto = loss->push,
    # off = push->win); breakeven cents = price change that leaves EV unchanged
    S["p_land"] = np.where(isint_to, S.to_pp, S.pp_sharp)
    S["d_ev"] = S.to_ev - S.ev_sharp
    d_be = (S.ev_sharp + 1 - S.to_pp) / np.maximum(S.to_pw, 1e-9)
    S["be_cents"] = cents(S.price.values) - cents(am_from_dec(d_be))
    S["from_main"] = S.has_main & ((S.point - S.ref_point).abs() < 1e-9)
    S["to_main"] = S.has_main & ((S.to_point - S.ref_point).abs() < 1e-9)
    return S


# ======================================================================================= curve
def curve_tables(S: pd.DataFrame) -> dict:
    out = {}
    X = S[S.price.between(-400, 400) & S.to_price.between(-400, 400)]

    def agg(d):
        return pd.Series({"n": len(d), "cents_med": d.cents.median(), "cents_mean": d.cents.mean(),
                          "be_cents_mean": d.be_cents.mean(), "prob_cost": d.prob_cost.mean(),
                          "p_land": d.p_land.mean(), "d_ev": d.d_ev.mean(),
                          "share_dev_pos": (d.d_ev > 0).mean()})

    def tbl(d, by):
        t = d.groupby(by).apply(agg, include_groups=False).reset_index()
        t = t[t.n >= 20]
        return [{k: (r4(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                for r in t.to_dict("records")]

    # (1) first half point bought off the main line (what the dashboard estimate represents), and 2nd half point
    F = X[X.from_main].copy()
    out["first_buy_from_main"] = tbl(F, ["book", "cat", "kind", "fav"])
    out["first_buy_from_main_allbooks"] = tbl(F, ["cat", "kind", "fav"])
    out["first_buy_by_book"] = tbl(F, ["book", "cat"])
    out["first_sell_to_main"] = tbl(X[X.to_main], ["book", "cat"])
    # second half point (main+0.5 -> main+1)
    sec = X[X.has_main & ((X.point - X.ref_point - 0.5).abs() < 1e-9)]
    out["second_buy"] = tbl(sec, ["book", "cat"])
    # (2) every step within +-3.5 of main, alt-to-alt only (pure alt ladder slope)
    A = X[(X.src == "alt") & (X.to_src == "alt")]
    out["alt_ladder_steps"] = tbl(A, ["book", "cat", "kind", "fav"])
    out["alt_ladder_steps_allbooks"] = tbl(A, ["cat", "kind", "fav"])
    out["alt_ladder_by_season"] = tbl(A, ["season", "book", "cat"])
    out["alt_ladder_by_window"] = tbl(A, ["window", "cat"])
    # assumed vs measured summary per book: first buy off main (cents), by crossed number
    summ = []
    for (b, c), d in F.groupby(["book", "cat"]):
        summ.append({"book": b, "cat": c, "n": len(d), "assumed": ASSUMED[c], "measured_med": r4(d.cents.median()),
                     "measured_mean": r4(d.cents.mean()), "breakeven_mean": r4(d.be_cents.mean())})
    out["assumed_vs_measured_first_buy"] = summ
    return out


def run_curve():
    L, metas = ladders(DEV + HOLD)
    S = steps(L)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV + HOLD),
           "meta": metas, "steps": int(len(S)), **curve_tables(S)}
    store("curve", res)
    print(json.dumps({k: res[k] for k in ("meta", "assumed_vs_measured_first_buy", "first_buy_from_main_allbooks",
                                          "alt_ladder_steps_allbooks")}, indent=1, default=_jd)[:12000])


# ======================================================================================= EV analysis
def ev_tables(L: pd.DataFrame) -> dict:
    out = {}
    A = L[(L.src == "alt") & L.price.between(-400, 400)].copy()
    Mn = L[L.src == "main"]
    out["main_ev_by_book"] = [{"book": b, "n": len(d), "ev_sharp": r4(d.ev_sharp.mean()),
                               "share_pos": r4((d.ev_sharp > 0).mean())} for b, d in Mn.groupby("book")]
    A["dist_b"] = A.dist.round(1)
    out["alt_ev_by_dist"] = [{"dist": k, "n": len(d), "ev_sharp": r4(d.ev_sharp.mean()), "ev_cons": r4(d.ev_cons.mean()),
                              "share_pos": r4((d.ev_sharp > 0).mean()), "ev_oldw": r4(d.ev_sharp_oldw.mean())}
                             for k, d in A.groupby("dist_b")]
    out["alt_ev_by_book_window"] = [
        {"book": b, "window": w, "n": len(d), "ev_sharp": r4(d.ev_sharp.mean()), "share_pos": r4((d.ev_sharp > 0).mean()),
         "share_ge_2pct": r4((d.ev_sharp >= 0.02).mean()), "max": r4(d.ev_sharp.max())}
        for (b, w), d in A.groupby(["book", "window"])]
    out["alt_ev_by_cross"] = [
        {"cross3": c3, "cross7": c7, "bought": bool(bo), "n": len(d), "ev_sharp": r4(d.ev_sharp.mean()),
         "share_pos": r4((d.ev_sharp > 0).mean())}
        for (c3, c7, bo), d in A.groupby(["cross3", "cross7", A.dist > 0])]
    # per-line overall (one row per (game, window, side, point): best allowed-book price) -> what a shopper sees
    B = A.sort_values("ev_sharp").groupby(["game_id", "window", "side", "point"]).tail(1)
    out["best_book_alt_ev_by_dist"] = [{"dist": k, "n": len(d), "ev_sharp": r4(d.ev_sharp.mean()),
                                        "share_pos": r4((d.ev_sharp > 0).mean())} for k, d in B.groupby("dist_b")]
    return out


def calib_exact(L: pd.DataFrame) -> list[dict]:
    """Predicted vs actual P(|home margin| == 3 / 7) at the close (sanity check of the half-point value)."""
    key, _, _ = T2.load_dist()
    G = L.drop_duplicates("game_id")[["game_id", "season", "m", "mu_close_sharp"]]
    rows = []
    for s, d in G.groupby("season"):
        rec = {"season": int(s), "games": len(d)}
        for kn in KEYS:
            pe = key.p_eq(d.mu_close_sharp.values, np.full(len(d), kn)) + key.p_eq(d.mu_close_sharp.values,
                                                                                  np.full(len(d), -kn))
            rec[f"pred{kn}"] = r4(pe.sum())
            rec[f"act{kn}"] = int((d.m.abs() == kn).sum())
        rows.append(rec)
    return rows


# ======================================================================================= rules
def candidates_pool(L: pd.DataFrame) -> pd.DataFrame:
    """Bettable rows: alt lines (and main lines, flagged) at allowed books with sane prices."""
    return L[L.price.between(-300, 300)].copy()


def apply_rule(L: pd.DataFrame, r: dict) -> pd.DataFrame:
    X = L[(L.src == "alt") & L.price.between(-300, 300) & L.has_main]
    if r.get("window"):
        X = X[X.window == r["window"]]
    if r.get("books"):
        X = X[X.book.isin(r["books"])]
    trs = r.get("transitions") or ([r["transition"]] if r.get("transition") else None)
    if trs:
        ok = np.zeros(len(X), bool)
        for f, t in trs:
            ok |= (((X.ref_point - f).abs() < 1e-9) & ((X.point - t).abs() < 1e-9)).values
        X = X[ok]
    if r.get("dist") is not None:
        X = X[(X.dist - r["dist"]).abs() < 1e-9]
    if r.get("bought") is True:
        X = X[X.dist > 0]
    if r.get("bought") is False:
        X = X[X.dist < 0]
    if r.get("cross_key"):
        X = X[X.cross_key]
    if r.get("min_ev") is not None:
        X = X[X.ev_sharp >= r["min_ev"]]
    if r.get("max_abs_main"):
        X = X[X.ref_point.abs() <= r["max_abs_main"]]
    # one bet per game per window: the highest fair-EV line
    X = X.sort_values("ev_sharp").groupby(["game_id", "window"]).tail(1)
    if not r.get("window"):           # if both windows allowed, the first (early) bet only
        X = X.sort_values("requested_ts").groupby("game_id").head(1)
    return X


def summarize(X: pd.DataFrame) -> dict:
    n = len(X)
    if n == 0:
        return {"bets": 0}
    roi, roi_se = mse(X.profit)
    clv, clv_se = mse(X.ev_close_sharp)
    ev, _ = mse(X.ev_sharp)
    nopush = X[X.y != 0]
    return {"bets": n, "hit_rate": r4((nopush.y > 0).mean()) if len(nopush) else None, "pushes": int((X.y == 0).sum()),
            "avg_price": r4(np.median(X.price)), "model_ev": ev, "roi": roi, "roi_se": roi_se,
            "clv": clv, "clv_se": clv_se, "clv_t": r4(clv / clv_se) if clv_se else None,
            "share_clv_pos": r4((X.ev_close_sharp > 0).mean()),
            "exp_roi_close": clv, "books": X.book.value_counts().to_dict()}


TRANSITIONS = [(2.5, 3.0), (3.0, 3.5), (-3.5, -3.0), (-7.5, -7.0), (-3.0, -2.5), (6.5, 7.0), (7.0, 7.5),
               (-7.0, -6.5), (2.5, 3.5), (-3.5, -2.5), (1.5, 3.0), (-4.0, -3.0), (6.0, 7.0), (-8.0, -7.0),
               # sells
               (3.0, 2.5), (3.5, 3.0), (-3.0, -3.5), (-2.5, -3.0), (7.0, 6.5), (-7.0, -7.5)]


def rule_grid(books):
    rules = []
    for w in ("early", "close"):
        for f, t in TRANSITIONS:
            rules.append({"id": f"T {f:+g}->{t:+g} {w} all", "window": w, "transition": [f, t]})
            for b in books:
                rules.append({"id": f"T {f:+g}->{t:+g} {w} {b}", "window": w, "transition": [f, t], "books": [b]})
        for th in (0.0, 0.01, 0.02, 0.03):
            rules.append({"id": f"EV>={th:.2f} {w} any", "window": w, "min_ev": th})
            rules.append({"id": f"EV>={th:.2f} {w} bought", "window": w, "min_ev": th, "bought": True})
            rules.append({"id": f"EV>={th:.2f} {w} sold", "window": w, "min_ev": th, "bought": False})
            rules.append({"id": f"EV>={th:.2f} {w} key-cross", "window": w, "min_ev": th, "cross_key": True})
            for b in books:
                rules.append({"id": f"EV>={th:.2f} {w} {b}", "window": w, "min_ev": th, "books": [b]})
    return rules


def run_dev():
    L, metas = ladders(DEV)
    S = steps(L)
    allowed = sorted(L.book.unique())
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV), "meta": metas,
           "ev": ev_tables(L), "calib_exact": calib_exact(L)}
    # marginal value of each half point vs price paid (2023)
    X = S[S.price.between(-400, 400) & S.to_price.between(-400, 400)]
    res["marginal"] = [
        {"book": b, "cat": c, "kind": k, "fav": bool(f), "n": len(d), "p_land": r4(d.p_land.mean()),
         "prob_cost": r4(d.prob_cost.mean()), "cents": r4(d.cents.mean()), "be_cents": r4(d.be_cents.mean()),
         "d_ev": r4(d.d_ev.mean()), "share_dev_pos": r4((d.d_ev > 0).mean())}
        for (b, c, k, f), d in X.groupby(["book", "cat", "kind", "fav"]) if len(d) >= 20]
    rows = []
    for r in rule_grid(allowed):
        e = summarize(apply_rule(L, r))
        if e["bets"] >= 10:
            rows.append({"id": r["id"], **{k: v for k, v in e.items() if k != "books"}})
    res["grid"] = rows
    store("dev", res)
    G = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(json.dumps(res["ev"], default=_jd)[:4000])
    print(pd.DataFrame(res["marginal"]).to_string())
    print(G.sort_values("model_ev", ascending=False).head(40).to_string())
    print(G[G.id.str.startswith("T ") & G.id.str.endswith(" all")].to_string())


# Frozen after reading the 2023 dev grid (see buy_points.md section 3 for the reasoning).
# Every fixed buy/sell transition had 2023 model EV -2.8% .. -5.5% (worse than or equal to the -4.4% main line), so
# no fixed transition is a candidate on its own; B3 is the canonical "buy onto 3/7" as a pre-registered control.
CANDIDATES: list[dict] = [
    {"id": "B1_alt_ev1_close", "window": "close", "min_ev": 0.01,
     "desc": "Any alt line (bought OR sold, within 3.5 pts of that book's main line, price -300..+300) at an allowed "
             "book at the ~75-min close snapshot with fair_sharp EV >= +1%; one bet per game (highest EV).",
     "dev_2023": {"bets": 56, "model_ev": 0.0227, "roi": 0.1015, "roi_se": 0.1616},
     "edge_if": "ROI - 1.64*SE > 0 (CLV == model EV at the same snapshot, so not an independent test)"},
    {"id": "B2_alt_ev1_early", "window": "early", "min_ev": 0.01,
     "desc": "Same as B1 at the early snapshot (Fri 21:40 UTC / kickoff-24h); judged on CLV vs the close fair_sharp.",
     "dev_2023": {"bets": 44, "model_ev": 0.0211, "clv": 0.0219, "clv_se": 0.0091, "roi": -0.1476, "roi_se": 0.1669},
     "edge_if": "CLV t > 2 AND ROI > 0"},
    {"id": "B3_buy_onto_3_7_close", "window": "close",
     "transitions": [[2.5, 3.0], [-3.5, -3.0], [6.5, 7.0], [-7.5, -7.0]],
     "desc": "Systematic buy ONTO 3 or 7 at the close: book main +2.5 -> +3, -3.5 -> -3, +6.5 -> +7, -7.5 -> -7 at "
             "that same book's alt price; best allowed book by fair EV; no EV filter; one per game.",
     "dev_2023": {"bets": 108, "model_ev": -0.044, "roi": -0.0598, "roi_se": 0.0826},
     "edge_if": "ROI - 1.64*SE > 0 (expected about -4%)"},
]


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen rules are final)")
    J = json.loads(JSON.read_text())
    if "dev" not in J:
        raise SystemExit("run dev first")
    if not CANDIDATES or len(CANDIDATES) > 3:
        raise SystemExit("define 1-3 CANDIDATES")
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
          "developed_on": "2023 alternate-spread snapshots only; distribution fit 2012-2019 (teasers_v2.json)",
          "dev_generated": J["dev"]["generated"], "holdout": list(HOLD), "evaluate_once": True,
          "fair": "fair_sharp at the bet's snapshot (lowvig/betonlineag main spreads + juice, fallback all books)",
          "one_bet_per": "game per window (highest fair EV); if no window given, the early bet only",
          "candidates": CANDIDATES}
    FROZEN.write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))


def run_holdout():
    if os.environ.get("BUYPTS_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked: set BUYPTS_HOLDOUT=I_HAVE_FROZEN after freezing")
    if not FROZEN.exists():
        raise SystemExit("freeze first")
    J = json.loads(JSON.read_text())
    if "holdout" in J:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    L, metas = ladders(HOLD)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"],
           "seasons": list(HOLD), "meta": metas, "rules": {}, "bets": {}}
    for r in fz["candidates"]:
        X = apply_rule(L, r)
        res["rules"][r["id"]] = {**summarize(X), "by_season": {int(s): summarize(d) for s, d in X.groupby("season")}}
        res["bets"][r["id"]] = X[["game_id", "window", "book", "side", "ref_point", "point", "price", "ev_sharp",
                                  "ev_close_sharp", "y", "profit"]].to_dict("records")
    res["ev"] = ev_tables(L)                      # descriptive
    res["calib_exact"] = calib_exact(L)
    store("holdout", res)
    print(json.dumps({k: v for k, v in res.items() if k != "bets"}, indent=1, default=_jd)[:6000])


# ======================================================================================= dashboard question
def run_dash():
    """Descriptive, 2023-2025: at each alt snapshot, the dashboard's own spread pick (blend of walk-forward model
    margin and consensus market margin, old key weights, best main line at an allowed book, EV >= 3%, price in
    [-200, 200]; QB/injury gates not reproducible). Would buying at the same book (real alt price) raise EV?"""
    r = SB.load_rules()
    key, _, _ = T2.load_dist()
    old = old_dist()
    L, _ = ladders(DEV + HOLD)
    wf = pd.read_parquet(ROOT / "data" / "replay_predictions.parquet")[["game_id", "mu_model"]]
    L = L.merge(wf, on="game_id")
    L["mu_dash"] = SB.expected_margin(L.mu_model, -L.pt_cons, r)
    hs = (L.side == "home").values
    pw, pp = old.side_probs(L.mu_dash.values, hs, L.point.values)
    L["ev_dash"] = pw * L.dec + pp - 1
    pw, pp = key.side_probs(L.mu_dash.values, hs, L.point.values)
    L["ev_dash_neww"] = pw * L.dec + pp - 1
    k = ["game_id", "requested_ts"]
    Mn = L[(L.src == "main") & L.price.between(-200, 200)]
    best = Mn.sort_values("ev_dash").groupby(k).tail(1)
    q = best[best.ev_dash >= r["qualify"]["min_edge_at_best_price"]]
    rows = []
    for row in q.itertuples():
        same = L[(L.game_id == row.game_id) & (L.requested_ts == row.requested_ts) & (L.side == row.side)]
        rec = {"game_id": row.game_id, "season": int(row.season), "window": row.window, "book": row.book,
               "side": row.side, "point": row.point, "price": row.price, "ev_dash": row.ev_dash,
               "ev_dash_neww": row.ev_dash_neww, "ev_fair": row.ev_sharp}
        for h in (0.5, 1.0):
            est = SB.buy_point_options(row.mu_dash, row.point, int(row.price), row.side, r)[int(h * 2) - 1]
            alt = same[(same.book == row.book) & ((same.point - row.point - h).abs() < 1e-9)]
            rec[f"est_price_{h}"] = est["price"] if "price" in est else est["est_price"]
            rec[f"est_ev_{h}"] = est["ev"]
            if len(alt):
                a = alt.iloc[0]
                rec[f"act_price_{h}"], rec[f"act_ev_dash_{h}"] = a.price, a.ev_dash
                rec[f"act_ev_fair_{h}"], rec[f"act_ev_neww_{h}"] = a.ev_sharp, a.ev_dash_neww
        bb = same[(same.point > row.point) & (same.src == "alt") & same.price.between(-300, 300)]
        if len(bb):
            top = bb.sort_values("ev_dash").iloc[-1]
            rec["best_buy_ev_dash"], rec["best_buy_ev_fair"] = top.ev_dash, top.ev_sharp
            rec["best_buy_pts"], rec["best_buy_book"] = top.point - row.point, top.book
        rows.append(rec)
    D = pd.DataFrame(rows)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV + HOLD),
           "qualifying_snapshots": len(D), "games": int(D.game_id.nunique()) if len(D) else 0}
    if len(D):
        for h in (0.5, 1.0):
            d = D[D[f"act_price_{h}"].notna()]
            res[f"buy_{h}"] = {
                "n_with_alt": len(d),
                "est_price_med": r4(d[f"est_price_{h}"].median()), "act_price_med": r4(d[f"act_price_{h}"].median()),
                "est_minus_act_cents_mean": r4((cents(d[f"est_price_{h}"]) - cents(d[f"act_price_{h}"])).mean()),
                "dash_ev_main": r4(d.ev_dash.mean()), "dash_ev_est_buy": r4(d[f"est_ev_{h}"].mean()),
                "dash_ev_actual_buy": r4(d[f"act_ev_dash_{h}"].mean()),
                "share_buy_improves_dash_ev": r4((d[f"act_ev_dash_{h}"] > d.ev_dash).mean()),
                "share_est_says_improves": r4((d[f"est_ev_{h}"] > d.ev_dash).mean()),
                "neww_ev_main": r4(d.ev_dash_neww.mean()), "neww_ev_buy": r4(d[f"act_ev_neww_{h}"].mean()),
                "share_buy_improves_neww_ev": r4((d[f"act_ev_neww_{h}"] > d.ev_dash_neww).mean()),
                "fair_ev_main": r4(d.ev_fair.mean()), "fair_ev_buy": r4(d[f"act_ev_fair_{h}"].mean()),
                "share_buy_improves_fair_ev": r4((d[f"act_ev_fair_{h}"] > d.ev_fair).mean())}
        d = D[D.best_buy_ev_dash.notna()]
        res["best_buy_any_book"] = {"n": len(d), "share_improves_dash_ev": r4((d.best_buy_ev_dash > d.ev_dash).mean()),
                                    "mean_gain_dash": r4((d.best_buy_ev_dash - d.ev_dash).mean()),
                                    "mean_gain_fair": r4((d.best_buy_ev_fair - d.ev_fair).mean())}
    # (post-holdout diagnostic) is the key-number model calibrated on lines far from the main line? close snapshot,
    # one row per (game, side, point); predicted P(win) under the close fair_sharp vs realized
    C = L[L.window == "close"].drop_duplicates(["game_id", "side", "point"])
    C = C.assign(adist=C.dist.abs().clip(upper=3.5).round(0), direction=np.where(C.dist > 0, "bought",
                                                                                  np.where(C.dist < 0, "sold", "main")))
    cal = []
    for (dr, ad, s), d in C.groupby(["direction", "adist", "season"]):
        se = math.sqrt((d.pw_close_sharp * (1 - d.pw_close_sharp)).sum()) / len(d)
        cal.append({"direction": dr, "abs_dist": ad, "season": int(s), "n": len(d), "pred_win": r4(d.pw_close_sharp.mean()),
                    "act_win": r4((d.y > 0).mean()), "z": r4(((d.y > 0).mean() - d.pw_close_sharp.mean()) / se)})
    res["alt_line_calibration"] = cal
    # the frozen rules' bets: expected vs realized wins (dev + holdout)
    fz = json.loads(FROZEN.read_text())
    chk = {}
    for r_ in fz["candidates"]:
        for lab, seas in (("dev_2023", DEV), ("holdout_2024_25", HOLD)):
            X = apply_rule(L[L.season.isin(seas)], r_)
            pwc = X.pw_close_sharp
            se = math.sqrt((pwc * (1 - pwc)).sum())
            chk[f"{r_['id']} {lab}"] = {"bets": len(X), "exp_wins_close": r4(pwc.sum()), "act_wins": int((X.y > 0).sum()),
                                        "z": r4(((X.y > 0).sum() - pwc.sum()) / se) if se else None}
    res["frozen_expected_vs_actual_wins"] = chk
    store("dash", res)
    print(json.dumps(res, indent=1, default=_jd))


# ======================================================================================= report
def run_report():
    J = json.loads(JSON.read_text())
    print("report: see buy_points.md (written by hand from buy_points.json)")
    print(json.dumps({k: list(v.keys()) if isinstance(v, dict) else v for k, v in J.items()}, indent=1)[:3000])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "curve"
    {"curve": run_curve, "dev": run_dev, "freeze": run_freeze, "holdout": run_holdout, "dash": run_dash,
     "report": run_report}[stage]()
