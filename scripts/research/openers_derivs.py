"""DERIVATIVES + ALTERNATE SPREADS AT THE TUESDAY OPEN (14:10 UTC snapshot).

Builds on (imports, does not modify):
  scripts/research/derivatives.py / derivatives_v2.py   1H spreads, 1H totals, team totals: fair from the sharp
                                                         full-game S,T (2012-22 mean models + key-number pmfs),
                                                         same-point cross-book refs, CLV / grading
  scripts/research/buy_points.py                         alternate-spread ladders (allowed books, main line spliced)
  src/nflpred/margin_total.py                            total-aware final-margin distribution (alt-spread fair)

Snapshots: open = Tuesday 14:10 UTC (new), early = Fri 21:40 UTC / kick-24h, close = kick-75min.

Stages (outputs output/research/openers_derivs.{json,md}, openers_derivs_frozen.json):
  dev       2023 ONLY: descriptives (softness at open vs later, convergence) + rule grids for both tests
  freeze    writes openers_derivs_frozen.json from FROZEN (<= 3 rules; refuses to overwrite)
  holdout   OPENDERIV_HOLDOUT=I_HAVE_FROZEN: 2024-2025 once (refuses if already present) + descriptives
  (openers_derivs.md written by hand from the json)

Test 1 (derivatives): CLV = EV under the CLOSE of the same derivative: median no-vig of all books quoting the
  same point at the close; else the close consensus implied mean through the fitted pmf (derivatives_v2.grade).
Test 2 (alternate spreads): fair = price-implied expected home margin of the sharp books' (lowvig/betonlineag/
  circa/bookmaker; fallback all) MAIN spread + juice under margin_total (total-aware, covariate = median posted
  total at the same snapshot). CLV = EV under the close of the SAME alt line: median two-way no-vig of all books
  quoting that (side, point) at the close (pairs home p / away -p within a book); else close fair under the model.
SEs are clustered by game (several bets can share a game).
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]

import derivatives as D  # noqa: E402
import derivatives_v2 as V2  # noqa: E402
import buy_points as BP  # noqa: E402
import teasers_v2 as T2  # noqa: E402
from nflpred import margin_total as MT  # noqa: E402

SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/openderiv")
SCR.mkdir(parents=True, exist_ok=True)
# redirect the imported modules' caches (their old caches predate the Tuesday snapshot)
D.SCR = SCR
V2.SCR = SCR
BP.MAXDIST = 7.0          # alt ladder kept within 7 pts of the book's main line (prior study: 3.5)

OUT = ROOT / "output" / "research"
JSON = OUT / "openers_derivs.json"
FROZEN_PATH = OUT / "openers_derivs_frozen.json"
DEV, HOLD = (2023,), (2024, 2025)
ALLOWED = D.ALLOWED
SNAPS = ("open", "early", "close")


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        return str(x)
    return x


def save(key, val):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d[key] = js(val)
    JSON.write_text(json.dumps(d, indent=1))


def load(key):
    return json.loads(JSON.read_text()).get(key) if JSON.exists() else None


def is_open(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    return ((ts.dt.dayofweek == 1) & (ts.dt.hour == 14) & (ts.dt.minute == 10)).values


def cl_se(x, cl):
    """Mean and game-clustered SE."""
    x = pd.Series(np.asarray(x, float))
    cl = pd.Series(np.asarray(cl))
    ok = x.notna().values
    x, cl = x[ok], cl[ok]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else None), None
    r = (x - x.mean()).groupby(cl.values).sum()
    G = len(r)
    se = math.sqrt((r ** 2).sum()) / n * math.sqrt(G / max(G - 1, 1))
    return float(x.mean()), se


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2)) if t is not None and np.isfinite(t) else None


def summ(b: pd.DataFrame, n_weeks: int, clv="clv", pnl="pnl") -> dict:
    if len(b) == 0:
        return {"n": 0}
    c, cse = cl_se(b[clv], b.game_id)
    r, rse = cl_se(b[pnl], b.game_id)
    t = c / cse if cse else None
    return {"n": len(b), "games": int(b.game_id.nunique()), "bets_per_week": len(b) / max(n_weeks, 1),
            "mean_ev": float(b.ev.mean()), "clv": c, "clv_se": cse, "clv_t": t, "clv_p1": pval(t),
            "clv_pos_rate": float((b[clv] > 0).mean()), "roi": r, "roi_se": rse,
            "roi_t": r / rse if rse else None, "avg_dec": float(b.dec.mean()),
            "by_book": {k: int(v) for k, v in b.book.value_counts().items()}}


# ====================================================================================== TEST 1: derivatives
def quotes(seasons) -> pd.DataFrame:
    p = SCR / f"q_{'_'.join(map(str, seasons))}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    q = V2.quotes(seasons)
    q["snap"] = np.where(is_open(q.requested_ts), "open", q.snap)
    q.to_parquet(p)
    return q


def n_weeks(q, snap="open"):
    return q[q.snap == snap][["season", "week"]].drop_duplicates().shape[0]


def describe_derivs(q: pd.DataFrame, o: pd.DataFrame) -> dict:
    out = {}
    q = q.copy()
    q["d_fg"] = q.imean - q.fair_fg
    q["d_anch"] = q.imean - q.fair_anch
    q["d_cons"] = q.imean - q.dcons_ex
    al = q[q.book.isin(ALLOWED | {"betonlineag"})]
    # (a) deviation of each quote's implied mean from fair (sharp FG at the same snapshot) and from other books
    r = {}
    for (f_, sn), z in al.groupby(["fam", "snap"]):
        r[f"{f_}|{sn}"] = {"n": len(z), "books": int(z.book.nunique()),
                           "mad_vs_fg": z.d_fg.abs().mean(), "sd_vs_fg": z.d_fg.std(),
                           "mean_vs_fg": z.d_fg.mean(),
                           "mad_vs_anch": z.d_anch.abs().mean(), "share_vs_anch_ge1": (z.d_anch.abs() >= 1).mean(),
                           "mad_vs_otherbooks": z.d_cons.abs().mean(),
                           "share_vs_otherbooks_ge1": (z.d_cons.abs() >= 1).mean()}
    out["deviation_by_snap"] = r
    # same books only (book present at all three snapshots of a game-market) to remove composition effects
    k = ["game_id", "mk", "book"]
    pv = al.pivot_table(index=k, columns="snap", values=["d_anch", "d_cons"], aggfunc="first")
    pv.columns = [a + "_" + b for a, b in pv.columns]
    need = [c for c in ("d_anch_open", "d_anch_early", "d_anch_close") if c in pv.columns]
    pv = pv.dropna(subset=need).reset_index()
    pv["fam"] = pv.mk.map(V2.fam)
    out["deviation_same_book_triplets"] = {
        f_: {"n": len(z), **{f"mad_anch_{s}": z[f"d_anch_{s}"].abs().mean() for s in SNAPS},
             **{f"mad_otherbooks_{s}": z[f"d_cons_{s}"].abs().mean() for s in SNAPS}}
        for f_, z in pv.groupby("fam")}
    # (b) cross-book dispersion at the same point: range of no-vig q_over among books (>= 3 books)
    kk = ["event_id", "requested_ts", "mk", "x"]
    g = q.groupby(kk).agg(n=("book", "nunique"), rng=("q_over", lambda v: v.max() - v.min()),
                          sd=("q_over", "std"), snap=("snap", "first"), fam=("fam", "first")).reset_index()
    g = g[g.n >= 3]
    out["same_point_dispersion"] = {f"{f_}|{sn}": {"groups": len(z), "med_range": z.rng.median(),
                                                    "mean_sd": z.sd.mean(), "share_range_ge_5pct": (z.rng >= 0.05).mean()}
                                    for (f_, sn), z in g.groupby(["fam", "snap"])}
    # (c) best allowed-book EV vs same-point other-book consensus per game-market (shopping value) by snap
    z = o[o.ev_consq.notna() & (o.src_consq == "same_point")]
    best = z.sort_values("ev_consq").groupby(["game_id", "mk", "snap"]).tail(1)
    out["best_shop_ev_consq"] = {f"{f_}|{sn}": {"n": len(v), "mean": v.ev_consq.mean(),
                                                "share_ge_1pct": (v.ev_consq >= 0.01).mean(),
                                                "share_ge_3pct": (v.ev_consq >= 0.03).mean()}
                                 for (f_, sn), v in best.groupby(["fam", "snap"])}
    # (d) convergence: do open deviations from the FG-implied fair close by Friday / kickoff?
    pv = q.pivot_table(index=k, columns="snap", values=["imean", "fair_fg", "fair_anch"], aggfunc="first")
    pv.columns = [a + "_" + b for a, b in pv.columns]
    pv = pv.reset_index()
    pv["fam"] = pv.mk.map(V2.fam)
    r = {}
    for f_, z in pv.groupby("fam"):
        for tgt in ("early", "close"):
            zz = z.dropna(subset=["imean_open", f"imean_{tgt}", "fair_fg_open", f"fair_fg_{tgt}", "fair_anch_open"])
            if len(zz) < 40:
                continue
            dd = zz[f"imean_{tgt}"] - zz.imean_open
            gap = zz.fair_anch_open - zz.imean_open
            dfg = zz[f"fair_fg_{tgt}"] - zz.fair_fg_open
            b_, se_, res = D.ols([gap, dfg], dd)
            r[f"{f_}|open->{tgt}"] = {"n": len(zz), "slope_on_open_gap": b_[1], "se_gap": se_[1],
                                      "slope_on_fg_move": b_[2], "se_fg": se_[2], "sd_deriv_move": dd.std(),
                                      "sd_open_gap": gap.std(), "sd_fg_move": dfg.std(),
                                      "r2": 1 - res.var() / dd.var()}
    out["convergence_open"] = r
    # (e) how good is each fair at predicting the outcome, by snapshot (one row per game-market)
    one = q.groupby(["game_id", "mk", "snap"]).agg(X=("X", "first"), cons=("dcons", "first"),
                                                     fg=("fair_fg", "first"), anch=("fair_anch", "first")).reset_index()
    one["fam"] = one.mk.map(V2.fam)
    out["fair_mse_vs_outcome"] = {
        f"{f_}|{sn}": {c: {"n": int(z[c].notna().sum()), "mse": float(((z.X - z[c]) ** 2).mean()),
                           "bias": float((z.X - z[c]).mean())} for c in ("cons", "fg", "anch")}
        for (f_, sn), z in one.groupby(["fam", "snap"])}
    # (f) team totals vs the same book's own full-game lines, by snapshot
    tt = al[(al.fam == "team_totals")].dropna(subset=["b_sp", "b_tot", "fg_own"]).copy()
    sgn = np.where(tt.side_team == "home", 1, -1)
    tt["d_arith"] = tt.x - (tt.b_tot - sgn * tt.b_sp) / 2
    tt["d_own"] = tt.imean - tt.fg_own
    out["tt_vs_own_fg"] = {f"{bk}|{sn}": {"n": len(z), "line_off_ge1": (z.d_arith.abs() >= 1).mean(),
                                          "imean_vs_own_ge1": (z.d_own.abs() >= 1).mean(),
                                          "mad_imean_vs_own": z.d_own.abs().mean()}
                           for (bk, sn), z in tt.groupby(["book", "snap"])}
    # (g) calibration of open offers: EV bins vs CLV and ROI
    cal = {}
    for fc in ("anch", "fg", "consq"):
        z = o[(o.snap == "open") & o[f"ev_{fc}"].notna()].copy()
        z["ev"] = z[f"ev_{fc}"]
        z = z[z.ev > -0.04]
        gr = V2.grade(z, q)
        bins = pd.cut(gr.ev, [-0.04, 0, 0.02, 0.04, 0.07, 0.12, 1])
        cal[fc] = {str(kk_): {"n": len(v), "ev": v.ev.mean(), "clv": v.clv.mean(), "roi": v.pnl.mean()}
                   for kk_, v in gr.groupby(bins, observed=True)}
    out["open_calibration"] = cal
    return out


def deriv_rules():
    R = []
    mk = {"spreads_h1": ["spreads_h1"], "totals_h1": ["totals_h1"], "team_totals": ["team_totals"],
          "all": ["spreads_h1", "totals_h1", "team_totals"]}
    for fc in ("fair_fg", "fair_anch", "fair_cons", "fair_consq"):
        for mn, ms in mk.items():
            for thr in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
                R.append({"fair": fc, "snap": "open", "markets": ms, "ev_min": thr, "mname": mn})
    for mn, ms in mk.items():
        for thr in (0.0, 0.01, 0.02, 0.03):
            for nref in (2, 3):
                R.append({"fair": "fair_consq", "snap": "open", "markets": ms, "ev_min": thr, "min_ref_books": nref,
                          "mname": mn})
            R.append({"fair": "fair_consq", "snap": "open", "markets": ms, "ev_min": thr, "agree": "anch",
                      "mname": mn})
        for thr in (0.02, 0.04, 0.06):
            R.append({"fair": "fair_anch", "snap": "open", "markets": ms, "ev_min": thr, "agree": "consq",
                      "mname": mn})
    for mn in ("team_totals", "all"):
        for inc in (0.5, 1.0, 1.5):
            for thr in (0.0, 0.02):
                R.append({"fair": "fair_anch", "snap": "open", "markets": mk[mn], "ev_min": thr,
                          "own_incons_min": inc, "mname": mn})
    # reference: same rules at early / close for comparison (is open softer?)
    for sn in ("early", "close"):
        for fc in ("fair_anch", "fair_consq"):
            for thr in (0.02, 0.04):
                R.append({"fair": fc, "snap": sn, "markets": mk["all"], "ev_min": thr, "mname": "all", "ref": True})
    return R


def rid(r):
    s = f"D|{r['fair'][5:]}|{r['snap']}|{r['mname']}|ev>={r['ev_min']}"
    for k in ("own_incons_min", "agree", "min_ref_books"):
        if r.get(k) is not None:
            s += f"|{k}={r[k]}"
    return s


def deriv_apply(o, q, r):
    b = V2.apply_rule(o, q, r)
    return b


# ====================================================================================== TEST 2: alternate spreads
def totals_by_snap(season, g) -> pd.DataFrame:
    t = T2.match(T2.load_totals([season]), g)
    t = t[t.tot_point.between(25, 80) & t.tot_over_price.between(-250, 200) & t.tot_under_price.between(-250, 200)]
    return t.groupby(["game_id", "requested_ts"]).tot_point.median().rename("tot").reset_index()


def alt_ladder(season) -> pd.DataFrame:
    p = SCR / f"alt_{season}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    L, meta = BP.build(season)
    key, _, _ = T2.load_dist()
    g = T2.games([season])
    o, snap = BP.main_market(season, g, key)
    mm = MT.load()
    # total at each main snapshot (nearest totals snapshot <= 6h; fallback nflverse closing total)
    tb = totals_by_snap(season, g).sort_values("requested_ts")
    S = snap[["game_id", "requested_ts"]].drop_duplicates().sort_values("requested_ts")
    S = pd.merge_asof(S, tb.rename(columns={"requested_ts": "tts"}), left_on="requested_ts", right_on="tts",
                      by="game_id", direction="nearest", tolerance=pd.Timedelta("6h"))
    S = S.merge(g[["game_id", "total_line"]], on="game_id", how="left")
    S["tot"] = S.tot.fillna(S.total_line)
    o = o.merge(S[["game_id", "requested_ts", "tot"]], on=["game_id", "requested_ts"], how="left")
    o["mu_mt"] = mm.implied_mu(o.sp_home_point.values, o.q_home.values, o.tot.fillna(44.0).values)
    k2 = ["game_id", "requested_ts"]
    cons = o.groupby(k2).agg(mt_cons=("mu_mt", "median"), tot=("tot", "first"))
    sh = o[o.book.isin(BP.SHARP)].groupby(k2).mu_mt.median().rename("mt_sharp")
    sn = cons.join(sh).reset_index()
    sn["mt_sharp"] = sn.mt_sharp.fillna(sn.mt_cons)
    last = sn.sort_values("requested_ts").groupby("game_id").tail(1)
    sn = sn.merge(last[["game_id", "mt_sharp", "tot"]].rename(columns={"mt_sharp": "mt_close", "tot": "tot_close"}),
                  on="game_id")
    L = L.merge(sn.rename(columns={"requested_ts": "main_ts"}), on=["game_id", "main_ts"], how="left")
    L = L[L.mt_sharp.notna()].copy()
    L["window"] = np.where(is_open(L.requested_ts), "open", L.window)
    hs = (L.side == "home").values
    pw, pp = mm.side_probs(L.mt_sharp.values, hs, L.point.values, L.tot.values)
    L["pw_mt"], L["pp_mt"] = pw, pp
    L["ev_mt"] = pw * L.dec.values + pp - 1
    pw, pp = mm.side_probs(L.mt_close.values, hs, L.point.values, L.tot_close.values)
    L["pwc_mt"], L["ppc_mt"] = pw, pp
    L["clv_mt"] = pw * L.dec.values + pp - 1
    # same-line two-way no-vig from ALL books (alt ladders + main lines) at every snapshot
    a = BP.load_alt(season, g)
    ma = []
    for side in ("home", "away"):
        x = o[["game_id", "requested_ts", "book", f"sp_{side}_point", f"sp_{side}_price"]].rename(
            columns={"requested_ts": "main_ts", f"sp_{side}_point": "point", f"sp_{side}_price": "price"})
        x["side"] = side
        ma.append(x)
    ma = pd.concat(ma)
    K = L[["game_id", "requested_ts", "main_ts"]].drop_duplicates()
    ma = K.merge(ma, on=["game_id", "main_ts"]).drop(columns="main_ts")
    allq = pd.concat([ma.assign(src="main"), a[["game_id", "requested_ts", "book", "side", "point", "price"]]
                      .assign(src="alt")], ignore_index=True)
    allq = allq.drop_duplicates(["game_id", "requested_ts", "book", "side", "point"], keep="first")
    allq = allq[allq.price.between(-1000, 1000)]
    h = allq[allq.side == "home"].rename(columns={"price": "ph", "src": "srch"})
    w = allq[allq.side == "away"].rename(columns={"price": "pa"}).drop(columns=["src"])
    w["point"] = -w.point
    pr = h.drop(columns="side").merge(w.drop(columns="side"), on=["game_id", "requested_ts", "book", "point"])
    ih, ia = BP.imp(pr.ph.values), BP.imp(pr.pa.values)
    pr["ovr"] = ih + ia
    pr["ih"] = ih
    pr = pr[(pr.ovr > 1.0) & (pr.ovr < 1.20)].copy()
    pr["qh"] = pr.ih / pr.ovr        # no-vig P(home covers home point | no push); multiplicative
    pr.to_parquet(SCR / f"altpairs_{season}.parquet")
    # LOO median of other books at the same snapshot & line; close median of all books
    kk = ["game_id", "requested_ts", "point"]
    loo = np.full(len(pr), np.nan)
    nref = np.zeros(len(pr), int)
    qv = pr.qh.values
    for _, idx in pr.groupby(kk).indices.items():
        if len(idx) < 2:
            continue
        for j, i in enumerate(idx):
            loo[i] = np.median(np.delete(qv[idx], j))
            nref[i] = len(idx) - 1
    pr["qh_loo"], pr["n_ref"] = loo, nref
    # map to ladder rows: home side point p -> qh at p; away side point p -> 1 - qh at -p
    L["hp"] = np.where(L.side == "home", L.point, -L.point)
    L = L.merge(pr[["game_id", "requested_ts", "book", "point", "qh_loo", "n_ref"]].rename(columns={"point": "hp"}),
                on=["game_id", "requested_ts", "book", "hp"], how="left")
    # rows where this book posts only one side: LOO = median of all books' pairs at that line
    allmed = pr.groupby(kk).agg(qh_all=("qh", "median"), n_all=("book", "nunique")).reset_index().rename(
        columns={"point": "hp"})
    L = L.merge(allmed, on=["game_id", "requested_ts", "hp"], how="left")
    own = L.qh_loo.isna() & L.qh_all.notna()
    L.loc[own, "qh_loo"] = L.loc[own, "qh_all"]
    L.loc[own, "n_ref"] = L.loc[own, "n_all"]
    L["n_ref"] = L.n_ref.fillna(0).astype(int)
    qs = np.where(L.side == "home", L.qh_loo, 1 - L.qh_loo)
    L["ev_xq"] = qs * (1 - L.pp_mt) * L.dec + L.pp_mt - 1
    # close: last snapshot per game, median over all books at that line
    cts = pr.groupby("game_id").requested_ts.max().rename("cts").reset_index()
    pc = pr.merge(cts, on="game_id")
    pc = pc[pc.requested_ts == pc.cts].groupby(["game_id", "point"]).qh.median().rename("qh_close").reset_index()
    L = L.merge(pc.rename(columns={"point": "hp"}), on=["game_id", "hp"], how="left")
    qc = np.where(L.side == "home", L.qh_close, 1 - L.qh_close)
    L["clv_mkt"] = qc * (1 - L.ppc_mt) * L.dec + L.ppc_mt - 1
    L["clv"] = np.where(L.qh_close.notna(), L.clv_mkt, L.clv_mt)
    L["clv_src"] = np.where(L.qh_close.notna(), "same_line", "model")
    L["dog"] = L.ref_point > 0
    L = L.drop(columns=["qh_all", "n_all"])
    L.to_parquet(p)
    return L


def alt_ladders(seasons):
    return pd.concat([alt_ladder(s) for s in seasons], ignore_index=True)


def describe_alts(L: pd.DataFrame) -> dict:
    out = {}
    A = L[(L.src == "alt") & L.price.between(-400, 400)].copy()
    A["adist"] = A.dist.abs().clip(upper=7).round(0)
    A["dirn"] = np.where(A.dist > 0, "bought", "sold")
    out["ev_by_window_dist"] = {
        f"{w}|{d}|{int(ad)}": {"n": len(z), "ev_mt": z.ev_mt.mean(), "share_pos": (z.ev_mt > 0).mean(),
                               "ev_xq": z.ev_xq.mean(), "clv": z.clv.mean(), "clv_mt": z.clv_mt.mean()}
        for (w, d, ad), z in A.groupby(["window", "dirn", "adist"])}
    out["ev_by_window_book"] = {
        f"{w}|{b}": {"n": len(z), "ev_mt": z.ev_mt.mean(), "share_pos": (z.ev_mt > 0).mean(),
                     "share_ge_2pct": (z.ev_mt >= 0.02).mean(), "ev_xq_mean": z.ev_xq.mean(),
                     "share_xq_ge_2pct": (z.ev_xq >= 0.02).mean()}
        for (w, b), z in A.groupby(["window", "book"])}
    # best allowed-book line per (game, window, side, point): what a shopper sees
    B = A.sort_values("ev_mt").groupby(["game_id", "window", "side", "point"]).tail(1)
    out["best_book_by_window"] = {w: {"n": len(z), "ev_mt": z.ev_mt.mean(), "share_pos": (z.ev_mt > 0).mean(),
                                      "share_ge_2pct": (z.ev_mt >= 0.02).mean(), "ev_xq": z.ev_xq.mean()}
                                  for w, z in B.groupby("window")}
    # alt dog +7 / +10 region specifically
    dg = A[A.point.isin([6.5, 7.0, 7.5, 9.5, 10.0, 10.5]) & (A.dist > 0)]
    out["dog_7_10"] = {f"{w}|{pt}": {"n": len(z), "ev_mt": z.ev_mt.mean(), "share_pos": (z.ev_mt > 0).mean(),
                                     "clv": z.clv.mean(), "roi": z.profit.mean()}
                       for (w, pt), z in dg.groupby(["window", "point"])}
    # cross-book same-line dispersion of no-vig (pairs) by window
    pr = pd.concat([pd.read_parquet(SCR / f"altpairs_{s}.parquet").assign(season=s) for s in L.season.unique()])
    pr["window"] = np.where(is_open(pr.requested_ts), "open", "x")
    kt = L[["game_id", "requested_ts", "window"]].drop_duplicates()
    pr = pr.drop(columns="window").merge(kt, on=["game_id", "requested_ts"])
    gg = pr.groupby(["game_id", "requested_ts", "point"]).agg(n=("book", "nunique"), rng=("qh", lambda v: v.max() - v.min()),
                                                              w=("window", "first"), ovr=("ovr", "mean")).reset_index()
    gg = gg[gg.n >= 3]
    out["same_line_dispersion"] = {w: {"groups": len(z), "med_range": z.rng.median(), "mean_range": z.rng.mean(),
                                       "share_ge_5pct": (z.rng >= 0.05).mean(), "mean_overround": z.ovr.mean()}
                                   for w, z in gg.groupby("w")}
    # convergence: the same (book, side, point) priced at open and at close -> did the price move toward fair?
    k = ["game_id", "book", "side", "point"]
    pv = A.pivot_table(index=k, columns="window", values=["ev_mt", "price"], aggfunc="first")
    pv.columns = [a + "_" + b for a, b in pv.columns]
    pv = pv.dropna(subset=["ev_mt_open", "ev_mt_close"]).reset_index()
    pv["evb"] = pd.cut(pv.ev_mt_open, [-1, -0.1, -0.05, -0.02, 0, 0.02, 1])
    out["open_ev_vs_close_ev_same_line"] = {str(b): {"n": len(z), "ev_open": z.ev_mt_open.mean(),
                                                     "ev_close": z.ev_mt_close.mean()}
                                            for b, z in pv.groupby("evb", observed=True)}
    # model calibration on alt lines (win rate vs model at close), by direction and distance
    C = L[L.window == "close"].drop_duplicates(["game_id", "side", "point"]).copy()
    C["adist"] = C.dist.abs().clip(upper=7).round(0)
    C["dirn"] = np.where(C.dist > 0, "bought", np.where(C.dist < 0, "sold", "main"))
    cal = {}
    for (d, ad), z in C.groupby(["dirn", "adist"]):
        se = math.sqrt((z.pwc_mt * (1 - z.pwc_mt)).sum()) / len(z)
        cal[f"{d}|{int(ad)}"] = {"n": len(z), "pred_win": z.pwc_mt.mean(), "act_win": (z.y > 0).mean(),
                                 "z": ((z.y > 0).mean() - z.pwc_mt.mean()) / se if se else None}
    out["mt_calibration_close"] = cal
    return out


def alt_apply(L: pd.DataFrame, r: dict) -> pd.DataFrame:
    X = L[(L.src == "alt") & L.price.between(r.get("pmin", -300), r.get("pmax", 300)) & L.has_main
          & (L.window == r["window"])].copy()
    X["ev"] = X[r["ev_col"]]
    X = X[X.ev.notna() & (X.ev >= r["ev_min"])]
    if r.get("min_ref"):
        X = X[X.n_ref >= r["min_ref"]]
    if r.get("agree_mt") is not None:
        X = X[X.ev_mt >= r["agree_mt"]]
    if r.get("dir") == "bought":
        X = X[X.dist > 0]
    elif r.get("dir") == "sold":
        X = X[X.dist < 0]
    if r.get("dog") is True:
        X = X[X.point > 0]
    if r.get("points"):
        X = X[X.point.isin(r["points"])]
    if r.get("max_adist") is not None:
        X = X[X.dist.abs() <= r["max_adist"]]
    X = X.sort_values("ev").groupby("game_id").tail(1)
    X["pnl"] = X.profit
    return X


def alt_rules():
    R = []
    for w in ("open", "early", "close"):
        for thr in (0.0, 0.01, 0.02, 0.03, 0.05):
            for d in (None, "bought", "sold"):
                for mx in (3.5, 7.0):
                    R.append({"window": w, "ev_col": "ev_mt", "ev_min": thr, "dir": d, "max_adist": mx})
        for thr in (0.0, 0.01, 0.02, 0.03):
            for nref in (1, 2, 3):
                R.append({"window": w, "ev_col": "ev_xq", "ev_min": thr, "min_ref": nref, "max_adist": 7.0})
            R.append({"window": w, "ev_col": "ev_xq", "ev_min": thr, "min_ref": 2, "agree_mt": 0.0, "max_adist": 7.0})
        for pts in ([7.0, 7.5], [10.0, 10.5], [7.0, 7.5, 10.0, 10.5], [3.0, 3.5]):
            for thr in (-1.0, -0.02, 0.0):
                R.append({"window": w, "ev_col": "ev_mt", "ev_min": thr, "dog": True, "dir": "bought", "points": pts,
                          "max_adist": 7.0})
    return R


def arid(r):
    s = f"A|{r['window']}|{r['ev_col']}>={r['ev_min']}"
    for k in ("dir", "dog", "points", "min_ref", "agree_mt", "max_adist"):
        if r.get(k) is not None:
            s += f"|{k}={r[k]}"
    return s


# ====================================================================================== DEV
def dev_stage():
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds")}
    # ---- test 1
    q = quotes(DEV)
    o = V2.all_offers(q)
    o.to_parquet(SCR / "offers_2023.parquet")
    res["deriv_counts"] = {f"{f_}|{sn}": {"quotes": len(z), "books": sorted(z.book.unique().tolist()),
                                          "games": int(z.game_id.nunique())} for (f_, sn), z in q.groupby(["fam", "snap"])}
    res["deriv_describe"] = describe_derivs(q, o)
    nw = n_weeks(q)
    rows = []
    for r in deriv_rules():
        b = deriv_apply(o, q, r)
        s = summ(b, nw)
        rows.append({"rule": rid(r), **{k: s.get(k) for k in ("n", "bets_per_week", "mean_ev", "clv", "clv_se", "clv_t",
                                                                "clv_pos_rate", "roi", "roi_se")}})
    t1 = pd.DataFrame(rows)
    t1.to_csv(SCR / "dev_deriv_rules.csv", index=False)
    res["deriv_rules"] = t1.to_dict("records")
    # ---- test 2
    L = alt_ladders(DEV)
    res["alt_counts"] = {w: {"rows": len(z), "alt_rows": int((z.src == "alt").sum()), "books": sorted(z.book.unique().tolist()),
                             "games": int(z.game_id.nunique())} for w, z in L.groupby("window")}
    res["alt_describe"] = describe_alts(L)
    nwa = L[L.window == "open"][["season", "game_id"]].merge(
        D.load_games()[["game_id", "week"]], on="game_id")[["season", "week"]].drop_duplicates().shape[0]
    rows = []
    for r in alt_rules():
        X = alt_apply(L, r)
        s = summ(X, nwa)
        rows.append({"rule": arid(r), **{k: s.get(k) for k in ("n", "bets_per_week", "mean_ev", "clv", "clv_se", "clv_t",
                                                                 "clv_pos_rate", "roi", "roi_se")},
                     "clv_mt": float(X.clv_mt.mean()) if len(X) else None,
                     "same_line_share": float((X.clv_src == "same_line").mean()) if len(X) else None})
    t2 = pd.DataFrame(rows)
    t2.to_csv(SCR / "dev_alt_rules.csv", index=False)
    res["alt_rules"] = t2.to_dict("records")
    save("dev_2023", res)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(t1.sort_values("clv_t", ascending=False).round(4).head(50).to_string())
    print(t2.sort_values("clv_t", ascending=False).round(4).head(50).to_string())


# ====================================================================================== FREEZE / HOLDOUT
ALLM = ["spreads_h1", "totals_h1", "team_totals"]
FROZEN_RULES: list[dict] = [   # chosen after reading the 2023 dev output only
    {"id": "O1_h1tot_model_open", "test": "deriv", "fair": "fair_anch", "snap": "open", "markets": ["totals_h1"],
     "ev_min": 0.04, "mname": "totals_h1",
     "dev_2023": {"n": 17, "clv": 0.0318, "clv_se": 0.0175, "roi": 0.24},
     "desc": "Tuesday 14:10 UTC: allowed-book 1H total (over/under) with EV >= 4% vs Fair(sharp FG S,T at the same "
             "snapshot) + rolling market level offset; one bet per game. Motivation: 1H totals close 69% of the "
             "open gap to the FG-implied fair by kickoff (largest of the three markets)."},
    {"id": "O2_xbook_open", "test": "deriv", "fair": "fair_consq", "snap": "open", "markets": ALLM, "ev_min": 0.01,
     "min_ref_books": 3, "mname": "all",
     "dev_2023": {"n": 11, "clv": 0.0266, "clv_se": 0.0223, "roi": 0.077},
     "desc": "Tuesday 14:10 UTC: derivatives_v2 R1 moved to the open -- best allowed-book price on a 1H spread / 1H "
             "total / team total with EV >= 1% vs the leave-one-out median no-vig of >= 3 other books quoting the "
             "SAME point; one bet per game-market."},
    {"id": "O3_alt_model_open", "test": "alt", "window": "open", "ev_col": "ev_mt", "ev_min": 0.03, "max_adist": 7.0,
     "dev_2023": {"n": 56, "clv": 0.0189, "clv_se": 0.0147, "clv_model": 0.0408, "roi": -0.154},
     "desc": "Tuesday 14:10 UTC: any alternate spread (bought or sold, within 7 pts of that book's main line, price "
             "-300..+300) at an allowed book with EV >= 3% under margin_total at the sharp main spread+juice and the "
             "median posted total of the same snapshot; one bet per game (highest EV)."},
]


def freeze_stage():
    if FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} exists; refusing to overwrite")
    if not 1 <= len(FROZEN_RULES) <= 3:
        raise SystemExit("define 1-3 FROZEN_RULES")
    FROZEN_PATH.write_text(json.dumps({
        "frozen_on": dt.datetime.now().isoformat(timespec="seconds"), "dev_seasons": list(DEV),
        "holdout_seasons": list(HOLD), "allowed_books": sorted(ALLOWED), "k": len(FROZEN_RULES),
        "pass_bar": "pooled CLV > 0 with one-sided p < 0.05/k (game-clustered SE) AND CLV > 0 in 2024 and in 2025",
        "rules": FROZEN_RULES}, indent=1))
    print(FROZEN_PATH.read_text())


def holdout_stage():
    if os.environ.get("OPENDERIV_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("set OPENDERIV_HOLDOUT=I_HAVE_FROZEN")
    if load("holdout") is not None:
        raise SystemExit("holdout already run once")
    fz = json.loads(FROZEN_PATH.read_text())
    k = fz["k"]
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "rules": {}}
    q = o = L = None
    for r in fz["rules"]:
        if r["test"] == "deriv":
            if q is None:
                q = quotes(HOLD)
                o = V2.all_offers(q)
            b = deriv_apply(o, q, r)
            nw = {s: n_weeks(q[q.season == s]) for s in HOLD}
        else:
            if L is None:
                L = alt_ladders(HOLD)
            b = alt_apply(L, r)
            gw = D.load_games()[["game_id", "week"]]
            nw = {s: L[(L.window == r["window"]) & (L.season == s)][["game_id"]].merge(gw, on="game_id").week.nunique()
                  for s in HOLD}
            b["clv_src_sl"] = b.clv_src
        pooled = summ(b, sum(nw.values()))
        per = {s: summ(b[b.season == s], nw[s]) for s in HOLD}
        ok = (pooled.get("clv") or -1) > 0 and (pooled.get("clv_p1") or 1) < 0.05 / k and \
            all((per[s].get("clv") or -1) > 0 for s in HOLD)
        res["rules"][r["id"]] = {"rule": r, "pooled": pooled, "by_season": per, "pass": bool(ok),
                                 "clv_same_point_share": float(b.clv_src.isin(["same_point", "same_line"]).mean())
                                 if len(b) else None}
        b.to_csv(SCR / f"holdout_bets_{r['id']}.csv", index=False)
    # descriptives on the holdout seasons (not used for selection)
    if q is None:
        q = quotes(HOLD)
        o = V2.all_offers(q)
    res["deriv_describe"] = describe_derivs(q, o)
    if L is None:
        L = alt_ladders(HOLD)
    res["alt_describe"] = describe_alts(L)
    save("holdout", res)
    print(json.dumps(js(res["rules"]), indent=1))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": dev_stage, "freeze": freeze_stage, "holdout": holdout_stage}[st]()
