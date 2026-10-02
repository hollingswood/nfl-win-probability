"""NFL DERIVATIVES v2: 1st-half spreads / 1st-half totals / team totals priced off the SHARP full-game line.

Builds on scripts/research/derivatives.py (imported, not modified): its 'fit' stage (2012-2022 nflverse closing
lines vs results) gives the mean models E[M1|S,T], E[P1|T,|S|], E[team pts|T,s] and key-number pmfs.

Stages (outputs in output/research/derivatives_v2.json, report output/research/derivatives.md):
  style     walk-forward team-style features from pbp (1H vs 2H EPA, scripted-drive EPA, 1H pace, prior 1H residuals);
            do they predict the 1H residual vs the line-implied mean?  fit 2012-2022, check 2023.
  dev       2023 ONLY: descriptive consistency/staleness stats + rule search (EV vs several fair values)
  freeze    writes derivatives_frozen.json from FROZEN below (refuses to overwrite)
  holdout   EDGE_HOLDOUT=I_HAVE_FROZEN: 2024-2025, ONCE (refuses if already present)
  (output/research/derivatives.md is written by hand from derivatives_v2.json)

Fair value of an offer (all at the SAME snapshot):
  fg    : Fair mean model applied to the sharp full-game S,T (lowvig/betonlineag spreads+juice -> margin_total model,
          totals -> totals_dist), + style adjustment if it passed the 'style' test.
  anch  : fg + rolling level offset (median of sharp-derivative implied mean - fg over the previous 80 game-markets,
          market data only, strictly earlier snapshots) -> keeps the model's cross-sectional structure, takes the
          level from the market (guards against the 2012-22 ratio drifting).
  bol   : betonlineag's own no-vig implied mean of the same derivative at that snapshot (cross-book angle).
  cons  : leave-one-out median implied mean of all other books' quotes of the same derivative.
EV = P(win)*decimal + P(push) - 1 under the fair pmf.  One bet per (game, market, snapshot): the best-EV side/book
among allowed books.  CLV: close snapshot, all books quoting the SAME point -> median no-vig; else the close
consensus implied mean through the fitted pmf.  PnL on actual 1H / final scores (push = 0).
"""
from __future__ import annotations

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
from nflpred import features as F  # noqa: E402
from nflpred import margin_total as MT  # noqa: E402
from nflpred.totals import TotalDist  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "derivatives_v2.json"
FROZEN_PATH = OUT / "derivatives_frozen.json"
MD = OUT / "derivatives.md"
SCR = D.SCR
ALLOWED = D.ALLOWED
FIT_SEASONS = D.FIT_SEASONS


def save(key, val):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d[key] = D._js(val)
    JSON.write_text(json.dumps(d, indent=1))


def load(key):
    return json.loads(JSON.read_text()).get(key) if JSON.exists() else None


def fam(mk):
    return mk.split(":")[0]


# ================================================================== STYLE (walk-forward)
def team_games() -> pd.DataFrame:
    p = SCR / "v2_teamgames.parquet"
    if p.exists():
        return pd.read_parquet(p)
    fr = []
    for s in range(2012, 2026):
        x = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet",
                            columns=["game_id", "posteam", "defteam", "game_half", "play_type", "epa", "play_id",
                                     "game_seconds_remaining"])
        x = x[x.play_type.isin(["pass", "run"]) & x.epa.notna() & x.posteam.notna()].copy()
        x["posteam"] = F._norm_team(x.posteam)
        x["defteam"] = F._norm_team(x.defteam)
        x = x.sort_values(["game_id", "play_id"])
        k = ["game_id", "posteam"]
        h1 = x[x.game_half == "Half1"].groupby(k).epa.agg(["mean", "count"]).rename(columns={"mean": "off_h1", "count": "n_h1"})
        h2 = x[x.game_half == "Half2"].groupby(k).epa.mean().rename("off_h2")
        sc = x.groupby(k).head(15).groupby(k).epa.mean().rename("off_scr")
        t = pd.concat([h1, h2, sc], axis=1).reset_index()
        fr.append(t)
    t = pd.concat(fr, ignore_index=True).rename(columns={"posteam": "team"})
    # defence = what the opponent's offence did
    opp = t.rename(columns={"team": "opp", "off_h1": "def_h1", "off_h2": "def_h2", "off_scr": "def_scr", "n_h1": "dn_h1"})
    g = D.load_games()[["game_id", "gameday", "home_team", "away_team"]]
    rows = []
    for side, other in (("home_team", "away_team"), ("away_team", "home_team")):
        z = g.rename(columns={side: "team", other: "opp"})[["game_id", "gameday", "team", "opp"]]
        rows.append(z)
    tg = pd.concat(rows).merge(t, on=["game_id", "team"]).merge(opp, on=["game_id", "opp"])
    tg["plays_h1"] = tg.n_h1 + tg.dn_h1
    tg.to_parquet(p)
    return tg


def style_frame(fair: D.Fair, window=16, minp=6) -> pd.DataFrame:
    """History rows + prior-only (shift(1)) rolling team-style features for home and away."""
    h = D.history().copy()
    tg = team_games().sort_values(["team", "gameday", "game_id"]).copy()
    # 1H residuals vs line-implied mean, attributed to teams (prior-only)
    h["r_m1"] = h.m1 - fair.m1(h.S, h.TOT)
    h["r_p1"] = h.p1 - fair.p1(h.S, h.TOT)
    rr = pd.concat([h[["game_id", "home_team", "r_m1", "r_p1"]].rename(columns={"home_team": "team"}),
                    h[["game_id", "away_team", "r_m1", "r_p1"]].rename(columns={"away_team": "team"}).assign(r_m1=lambda d: -d.r_m1)])
    tg = tg.merge(rr, on=["game_id", "team"], how="left")
    tg["d_off"] = tg.off_h1 - tg.off_h2
    tg["d_def"] = tg.def_h1 - tg.def_h2
    feats = ["off_scr", "def_scr", "d_off", "d_def", "plays_h1", "off_h1", "def_h1", "r_m1", "r_p1"]
    for c in feats:
        tg[f"p_{c}"] = tg.groupby("team")[c].transform(lambda s: s.shift(1).rolling(window, min_periods=minp).mean())
    P = tg[["game_id", "team"] + [f"p_{c}" for c in feats]]
    h = h.merge(P.rename(columns={"team": "home_team", **{f"p_{c}": f"h_{c}" for c in feats}}), on=["game_id", "home_team"], how="left")
    h = h.merge(P.rename(columns={"team": "away_team", **{f"p_{c}": f"a_{c}" for c in feats}}), on=["game_id", "away_team"], how="left")
    # composite predictors
    h["x_scr_m"] = (h.h_off_scr - h.h_def_scr) - (h.a_off_scr - h.a_def_scr)        # scripted net EPA edge (home)
    h["x_half_m"] = (h.h_d_off - h.h_d_def) - (h.a_d_off - h.a_d_def)              # fast-starter net edge (home)
    h["x_h1_m"] = (h.h_off_h1 - h.h_def_h1) - (h.a_off_h1 - h.a_def_h1)            # 1H net EPA edge (home)
    h["x_res_m"] = h.h_r_m1 - h.a_r_m1                                              # prior 1H margin residuals
    h["x_scr_p"] = h.h_off_scr + h.h_def_scr + h.a_off_scr + h.a_def_scr           # scripted EPA environment
    h["x_half_p"] = h.h_d_off + h.h_d_def + h.a_d_off + h.a_d_def                  # 1H-minus-2H EPA environment
    h["x_pace_p"] = (h.h_plays_h1 + h.a_plays_h1) / 2 - 63.0                       # 1H plays per game (both teams)
    h["x_res_p"] = (h.h_r_p1 + h.a_r_p1) / 2
    return h


def style_stage():
    fair = D.load_fair()
    h = style_frame(fair)
    out = {}
    specs = {"r_m1": ["x_scr_m", "x_half_m", "x_h1_m", "x_res_m"], "r_p1": ["x_scr_p", "x_half_p", "x_pace_p", "x_res_p"]}
    for y, xs in specs.items():
        tr = h[h.season.isin(FIT_SEASONS)].dropna(subset=xs + [y])
        te = h[h.season == 2023].dropna(subset=xs + [y])
        res = {"n_fit": len(tr), "n_2023": len(te), "single": {}, "joint": {}}
        for x in xs:
            b, se, _ = D.ols([tr[x]], tr[y])
            b3, se3, _ = D.ols([te[x]], te[y])
            res["single"][x] = {"slope": b[1], "t": b[1] / se[1], "slope_2023": b3[1], "t_2023": b3[1] / se3[1],
                                "sd_x": float(tr[x].std())}
        b, se, r = D.ols([tr[x] for x in xs], tr[y])
        pred_te = b[0] + sum(b[i + 1] * te[x] for i, x in enumerate(xs))
        res["joint"] = {"coef": dict(zip(["const"] + xs, b)), "t": dict(zip(["const"] + xs, b / se)),
                        "r2_fit": 1 - r.var() / tr[y].var(),
                        "mse_2023_base": float(((te[y] - te[y].mean()) ** 2).mean()),
                        "mse_2023_model": float(((te[y] - pred_te) ** 2).mean()),
                        "sd_pred_fit": float((tr[y] - r).std())}
        out[y] = res
    # adoption rule (decided BEFORE looking at 2023 bets): a single feature is used only if |t|>=3 in 2012-22 AND the
    # 2023 slope has the same sign; adjustment = slope * feature (centered on its 2012-22 mean)
    adopt = {}
    for y, res in out.items():
        for x, v in res["single"].items():
            if abs(v["t"]) >= 3 and np.sign(v["slope"]) == np.sign(v["slope_2023"]):
                tr = h[h.season.isin(FIT_SEASONS)].dropna(subset=[x, y])
                adopt.setdefault(y, {})[x] = {"slope": v["slope"], "center": float(tr[x].mean())}
    out["adopted"] = adopt
    save("style", out)
    print(json.dumps(D._js(out), indent=1))
    return out


def style_adjust(q: pd.DataFrame, fair: D.Fair) -> pd.Series:
    """Additive adjustment of the 1H fair mean (home margin for spreads_h1, total for totals_h1)."""
    st = load("style") or {}
    adopt = st.get("adopted", {})
    adj = pd.Series(0.0, index=q.index)
    if not adopt:
        return adj
    h = style_frame(fair).set_index("game_id")
    for y, mk in (("r_m1", "spreads_h1:home"), ("r_p1", "totals_h1:game")):
        for x, v in adopt.get(y, {}).items():
            s = q.mk == mk
            val = q.loc[s, "game_id"].map(h[x])
            adj.loc[s] += (v["slope"] * (val - v["center"])).fillna(0.0).values
    return adj


# ================================================================== per-book full-game lines
def fg_books(seasons) -> pd.DataFrame:
    """(event, requested_ts, book): spread point/q, total point/nv, S_b, T_b."""
    p = SCR / f"v2_fgbooks_{'_'.join(map(str, seasons))}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    td = TotalDist.load()
    mm = MT.load()
    fr = []
    for s in seasons:
        o = D._read(ROOT / "data" / "historical_odds" / f"nfl_odds_{s}.csv.gz", s)
        o = o[o.sp_home_point.notna() & (o.sp_home_point == -o.sp_away_point)
              & o.sp_home_price.between(-250, 200) & o.sp_away_price.between(-250, 200)].copy()
        o["q"] = D.imp(o.sp_home_price) / (D.imp(o.sp_home_price) + D.imp(o.sp_away_price))
        t = D._read(ROOT / "data" / "historical_odds" / "totals" / f"nfl_odds_{s}.csv.gz", s)
        t = t[t.tot_point.between(25, 80) & t.tot_over_price.between(-250, 200) & t.tot_under_price.between(-250, 200)].copy()
        t["nv"] = D.imp(t.tot_over_price) / (D.imp(t.tot_over_price) + D.imp(t.tot_under_price))
        k = ["event_id", "requested_ts", "book"]
        m = o[k + ["sp_home_point", "q"]].drop_duplicates(k).merge(
            t[k + ["tot_point", "nv"]].drop_duplicates(k), on=k, how="outer")
        fr.append(m)
    m = pd.concat(fr, ignore_index=True)
    m["T_b"] = np.nan
    s_ = m.tot_point.notna()
    m.loc[s_, "T_b"] = [td.implied_mu(a, b) for a, b in zip(m.loc[s_, "tot_point"], m.loc[s_, "nv"])]
    m["S_b"] = np.nan
    s_ = m.sp_home_point.notna()
    m.loc[s_, "S_b"] = mm.implied_mu(m.loc[s_, "sp_home_point"].values, m.loc[s_, "q"].values,
                                     m.loc[s_, "T_b"].fillna(44.0).values)
    m.to_parquet(p)
    return m


# ================================================================== quotes with fair values
def quotes(seasons) -> pd.DataFrame:
    fair = D.load_fair()
    q = D.build(tuple(seasons), fair).copy()
    q = q[q.game_id.notna() & q.X.notna()].copy()
    g = D.load_games()[["game_id", "week", "game_type"]]
    q = q.merge(g, on="game_id", how="left")
    q["fam"] = q.mk.map(fam)
    # style adjustment (1H only; zero unless adopted)
    sadj = style_adjust(q, fair)
    q["fair_fg"] = q.fg_mean_sh + sadj
    # anchored: rolling level offset from the sharp derivative (fallback: all-book consensus), strictly prior snapshots
    one = q.groupby(["game_id", "requested_ts", "mk"]).agg(
        ref=("dsharp", "first"), cons=("dcons", "first"), fg=("fair_fg", "first"), fam=("fam", "first")).reset_index()
    one["ref"] = one.ref.fillna(one.cons)
    one["gap"] = one.ref - one.fg
    one = one.sort_values("requested_ts")
    offs = []
    for f_, z in one.groupby("fam"):
        z = z.copy()
        gap = z.gap.values
        ts = z.requested_ts.values
        off = np.full(len(z), np.nan)
        for i in range(len(z)):
            prior = gap[(ts < ts[i]) & np.isfinite(gap)]
            if len(prior) >= 20:
                off[i] = np.median(prior[-80:])
        z["offset"] = off
        offs.append(z[["game_id", "requested_ts", "mk", "offset"]])
    q = q.merge(pd.concat(offs), on=["game_id", "requested_ts", "mk"], how="left")
    q["fair_anch"] = q.fair_fg + q.offset.fillna(0.0)
    q["fair_bol"] = q.dsharp
    q["fair_cons"] = q.dcons_ex
    # previous full-game snapshot (staleness): latest FG fair 12-120 h before the derivative snapshot
    fg = D.fg_fair(tuple(seasons))
    snaps = q[["event_id", "requested_ts"]].drop_duplicates()
    m = snaps.merge(fg[["event_id", "requested_ts", "S_sharp", "T_sharp", "S_all", "T_all"]].rename(
        columns={"requested_ts": "pts", "S_sharp": "S_prev", "T_sharp": "T_prev", "S_all": "Sa_prev", "T_all": "Ta_prev"}),
        on="event_id")
    m["lag_h"] = (m.requested_ts - m.pts).dt.total_seconds() / 3600
    m = m[(m.lag_h >= 12) & (m.lag_h <= 120) & m.S_prev.fillna(m.Sa_prev).notna()
          & m.T_prev.fillna(m.Ta_prev).notna()].sort_values("lag_h").drop_duplicates(["event_id", "requested_ts"])
    m["S_prev"] = m.S_prev.fillna(m.Sa_prev)
    m["T_prev"] = m.T_prev.fillna(m.Ta_prev)
    q = q.merge(m[["event_id", "requested_ts", "S_prev", "T_prev", "lag_h"]], on=["event_id", "requested_ts"], how="left")
    q["fg_prev"] = np.nan
    for mk in q.mk.unique():
        s = q.mk == mk
        q.loc[s, "fg_prev"] = D.fg_mean(fair, mk, q.loc[s, "S_prev"].values, q.loc[s, "T_prev"].values)
    # this book's own full-game lines at the same snapshot
    fb = fg_books(tuple(seasons))
    q = q.merge(fb.rename(columns={"sp_home_point": "b_sp", "tot_point": "b_tot", "q": "b_q", "nv": "b_nv"}),
                on=["event_id", "requested_ts", "book"], how="left")
    q["fg_own"] = np.nan
    for mk in q.mk.unique():
        s = q.mk == mk
        q.loc[s, "fg_own"] = D.fg_mean(fair, mk, q.loc[s, "S_b"].values, q.loc[s, "T_b"].values)
    return q


# ================================================================== EV / CLV / grading
def side_probs(fair, mk, mean, x, scale):
    gt, eq, lt = D.probs(fair, mk, mean, x, scale)
    return gt, eq, lt


def offers(q: pd.DataFrame, fair_col: str) -> pd.DataFrame:
    """Two rows per quote (over/under): EV under fair_col, closing CLV, result.  Allowed books only."""
    fair = D.load_fair()
    z = q[q.book.isin(ALLOWED) & q[fair_col].notna()].copy()
    z["p_gt"] = np.nan
    z["p_eq"] = np.nan
    for mk in z.mk.unique():
        s = (z.mk == mk).values
        gt, eq, _ = side_probs(fair, mk, z.loc[s, fair_col].values, z.loc[s, "x"].values, z.loc[s, "scale"].values)
        z.loc[s, "p_gt"], z.loc[s, "p_eq"] = gt, eq
    ov = z.assign(side="over", price=z.over_price, p_win=z.p_gt)
    un = z.assign(side="under", price=z.under_price, p_win=1 - z.p_gt - z.p_eq)
    o = pd.concat([ov, un], ignore_index=True)
    o["dec"] = D.dec(o.price)
    o["ev"] = o.p_win * o.dec + o.p_eq - 1
    o["fair_used"] = o[fair_col]
    return o


def close_lookup(q: pd.DataFrame) -> pd.DataFrame:
    c = q[q.snap == "close"].groupby(["game_id", "mk", "x"]).q_over.median().rename("cq_over").reset_index()
    return c


def grade(b: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """CLV (same-point close no-vig, else close consensus pmf) and PnL on the actual result."""
    fair = D.load_fair()
    b = b.merge(close_lookup(q), on=["game_id", "mk", "x"], how="left")
    b["c_gt"] = np.nan
    b["c_eq"] = np.nan
    for mk in b.mk.unique():
        s = (b.mk == mk).values
        gt, eq, _ = side_probs(fair, mk, b.loc[s, "close_cons"].values, b.loc[s, "x"].values,
                               b.loc[s, "close_scale"].fillna(44).values)
        b.loc[s, "c_gt"], b.loc[s, "c_eq"] = gt, eq
    qo = np.where(b.cq_over.notna(), b.cq_over, b.c_gt / np.maximum(1 - b.c_eq, 1e-9))
    qs = np.where(b.side == "over", qo, 1 - qo)
    b["clv"] = qs * (1 - b.c_eq) * b.dec + b.c_eq - 1
    b["clv_src"] = np.where(b.cq_over.notna(), "same_point", "pmf")
    win = np.where(b.side == "over", b.X > b.x, b.X < b.x)
    push = b.X == b.x
    b["pnl"] = np.where(push, 0.0, np.where(win, b.dec - 1, -1.0))
    b["win"] = np.where(push, np.nan, win.astype(float))
    return b


def pick(o: pd.DataFrame, rule: dict) -> pd.DataFrame:
    z = o[(o.snap == rule["snap"]) & o.fam.isin(rule["markets"]) & (o.ev >= rule["ev_min"])]
    if rule.get("ev_max") is not None:
        z = z[z.ev <= rule["ev_max"]]
    if rule.get("stale_min") is not None:      # full-game fair moved >= stale_min since the previous FG snapshot,
        mv = (o.loc[z.index, rule["fair"]] - o.loc[z.index, "fg_prev"])  # and the bet is in the direction of the move
        dirn = np.where(z.side == "over", 1, -1)
        z = z[(mv.abs() >= rule["stale_min"]) & (np.sign(mv) * dirn > 0)]
    if rule.get("own_incons_min") is not None:  # book's derivative vs its OWN full-game lines
        d = z.imean - z.fg_own
        dirn = np.where(z.side == "over", -1, 1)  # over when the book's derivative is set below its own FG implied
        z = z[(d.abs() >= rule["own_incons_min"]) & (np.sign(d) * dirn > 0)]
    if rule.get("agree") is not None:            # second fair value must also give EV >= 0
        z = z[z[f"ev_{rule['agree']}"] >= 0]
    z = z.sort_values("ev", ascending=False).drop_duplicates(["game_id", "mk", "snap"])
    return z


def summarize(b: pd.DataFrame, n_weeks=None) -> dict:
    if len(b) == 0:
        return {"n": 0}
    n = len(b)
    wk = n_weeks or b[["season", "week"]].drop_duplicates().shape[0]
    roi = b.pnl.mean()
    se = b.pnl.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    clv = b.clv.mean()
    cse = b.clv.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "bets_per_week": n / max(wk, 1), "mean_ev": b.ev.mean(), "clv": clv, "clv_se": cse,
            "clv_t": clv / cse if cse and cse > 0 else None, "clv_pos_rate": float((b.clv > 0).mean()),
            "roi": roi, "roi_se": se, "roi_t": roi / se if se and se > 0 else None,
            "hit": float(np.nanmean(b.win)), "push_rate": float((b.pnl == 0).mean()) if "pnl" in b else None,
            "avg_dec": b.dec.mean(), "by_market": {f: int(c) for f, c in b.fam.value_counts().items()},
            "by_book": {f: int(c) for f, c in b.book.value_counts().items()}}


def same_point_refs(q: pd.DataFrame) -> pd.DataFrame:
    """Per quote: leave-one-out median no-vig q_over of all OTHER books at the same (snapshot, market, point), its
    count, and betonlineag's q_over at the same point.  Model-free (no pmf translation across points)."""
    k = ["event_id", "requested_ts", "mk", "x"]
    out_med = np.full(len(q), np.nan)
    out_n = np.zeros(len(q), int)
    qo = q.q_over.values
    for _, idx in q.groupby(k).indices.items():
        if len(idx) < 2:
            continue
        v = qo[idx]
        for j, i in enumerate(idx):
            out_med[i] = np.median(np.delete(v, j))
            out_n[i] = len(v) - 1
    r = pd.DataFrame({"sp_cons_q": out_med, "sp_cons_n": out_n}, index=q.index)
    bo = q[q.book == "betonlineag"][k + ["q_over"]].drop_duplicates(k).rename(columns={"q_over": "sp_bol_q"})
    r = pd.concat([q[k], r], axis=1).merge(bo, on=k, how="left")
    r.index = q.index
    return r[["sp_cons_q", "sp_cons_n", "sp_bol_q"]]


def all_offers(q):
    """Offers with EV under every fair value as columns; 'ev' column set per rule's fair.
    ev_consq / ev_bolq: SAME-POINT no-vig of other books / betonlineag (pmf only for the push probability);
    when no other book quotes that point, fall back to the pmf translation (ev_cons / ev_bol)."""
    q = q.copy()
    q[["sp_cons_q", "sp_cons_n", "sp_bol_q"]] = same_point_refs(q)
    base = None
    for fc in ("fair_fg", "fair_anch", "fair_bol", "fair_cons"):
        o = offers(q, fc)
        o = o.set_index(["event_id", "requested_ts", "book", "mk", "x", "side"])
        if base is None:
            base = o.rename(columns={"ev": "ev_" + fc[5:]})
        else:
            base["ev_" + fc[5:]] = o.ev.reindex(base.index)
            base["peq_" + fc[5:]] = o.p_eq.reindex(base.index)
    b = base.reset_index()
    for name, qc, fb in (("consq", "sp_cons_q", "cons"), ("bolq", "sp_bol_q", "bol")):
        qs = np.where(b.side == "over", b[qc], 1 - b[qc])
        peq = b[f"peq_{fb}"].fillna(0.0)
        ev = qs * (1 - peq) * b.dec + peq - 1
        b[f"ev_{name}"] = np.where(b[qc].notna(), ev, b[f"ev_{fb}"])
        b[f"src_{name}"] = np.where(b[qc].notna(), "same_point", "pmf")
    return b


def apply_rule(o_all, q, rule):
    o = o_all.copy()
    o["ev"] = o[f"ev_{rule['fair'][5:]}"]
    o = o[o.ev.notna()]
    if rule.get("min_ref_books"):
        o = o[o.sp_cons_n >= rule["min_ref_books"]]
    b = pick(o, rule)
    return grade(b, q)


# ================================================================== descriptive: consistency + staleness
def describe(q: pd.DataFrame) -> dict:
    out = {}
    al = q[q.book.isin(ALLOWED | {"betonlineag"})].copy()
    # (1) team totals vs the same book's posted full-game arithmetic: TT ~ (total -/+ home spread point)/2
    tt = al[al.fam == "team_totals"].dropna(subset=["b_sp", "b_tot"]).copy()
    sgn = np.where(tt.side_team == "home", 1, -1)
    tt["arith"] = (tt.b_tot - sgn * tt.b_sp) / 2
    tt["d_arith"] = tt.x - tt.arith
    tt["d_mean_own"] = tt.imean - tt.fg_own
    tt["d_mean_sharp"] = tt.imean - tt.fg_mean_sh
    r = {}
    for bk, z in tt.groupby("book"):
        r[bk] = {"n": len(z), "line_vs_own_arith_ge0.5": float((z.d_arith.abs() >= 0.5).mean()),
                 "line_vs_own_arith_ge1": float((z.d_arith.abs() >= 1).mean()),
                 "mean_line_minus_arith": float(z.d_arith.mean()),
                 "imean_vs_own_fair_ge0.5": float((z.d_mean_own.abs() >= 0.5).mean()),
                 "imean_vs_own_fair_ge1": float((z.d_mean_own.abs() >= 1).mean()),
                 "imean_vs_sharp_fair_ge0.5": float((z.d_mean_sharp.abs() >= 0.5).mean()),
                 "imean_vs_sharp_fair_ge1": float((z.d_mean_sharp.abs() >= 1).mean()),
                 "mean_imean_minus_sharp_fair": float(z.d_mean_sharp.mean())}
    out["team_totals_consistency"] = r
    # sum of the two team-total implied means vs the book's own total-implied mean
    w = tt.pivot_table(index=["event_id", "requested_ts", "book"], columns="side_team", values="imean", aggfunc="first").dropna()
    w = w.join(tt.groupby(["event_id", "requested_ts", "book"]).T_b.first())
    w["d"] = w.home + w.away - w.T_b
    out["tt_sum_minus_own_total"] = {bk: {"n": len(z), "mean": float(z.d.mean()), "sd": float(z.d.std()),
                                          "abs_ge1": float((z.d.abs() >= 1).mean())}
                                     for bk, z in w.reset_index().groupby("book")}
    # (2) 1H markets vs the book's own and the sharp full-game implied means
    r = {}
    for (f_, bk), z in al[al.fam != "team_totals"].dropna(subset=["fg_own"]).groupby(["fam", "book"]):
        d = z.imean - z.fg_own
        ds = z.imean - z.fg_mean_sh
        r[f"{f_}|{bk}"] = {"n": len(z), "vs_own_ge0.5": float((d.abs() >= 0.5).mean()),
                           "vs_own_ge1": float((d.abs() >= 1).mean()), "mean_vs_own": float(d.mean()),
                           "vs_sharp_ge0.5": float((ds.abs() >= 0.5).mean()), "vs_sharp_ge1": float((ds.abs() >= 1).mean()),
                           "sd_vs_sharp": float(ds.std())}
    out["h1_consistency"] = r
    # (3) staleness at the early snapshot: derivative implied mean ~ a + b_now*fg_now + b_prev*fg_prev
    st = {}
    e = q[(q.snap == "early")].dropna(subset=["imean", "fair_fg", "fg_prev"])
    for (f_, bk), z in e.groupby(["fam", "book"]):
        if len(z) < 60:
            continue
        mv = z.fair_fg - z.fg_prev
        big = z[mv.abs() >= 0.5]
        b_, se_, _ = D.ols([z.fair_fg, z.fg_prev], z.imean)
        # among big moves: is the book closer to the previous FG-implied than to the current?
        closer_prev = float(((big.imean - big.fg_prev).abs() < (big.imean - big.fair_fg).abs()).mean()) if len(big) else None
        st[f"{f_}|{bk}"] = {"n": len(z), "b_now": b_[1], "b_prev": b_[2], "se_prev": se_[2],
                            "stale_weight": b_[2] / (b_[1] + b_[2]) if (b_[1] + b_[2]) else None,
                            "n_fg_move_ge0.5": len(big), "share_closer_to_prev": closer_prev}
    out["staleness_early"] = st
    # (4) early -> close: does the derivative close toward the early full-game-implied fair?
    k = ["game_id", "mk", "book"]
    ec = q.pivot_table(index=k, columns="snap", values=["imean", "fair_fg"], aggfunc="first").dropna()
    ec.columns = [a + "_" + b for a, b in ec.columns]
    ec = ec.reset_index()
    ec["fam"] = ec.mk.map(fam)
    r = {}
    for f_, z in ec.groupby("fam"):
        dd = z.imean_close - z.imean_early
        gap = z.fair_fg_early - z.imean_early
        dfg = z.fair_fg_close - z.fair_fg_early
        b_, se_, _ = D.ols([gap, dfg], dd)
        r[f_] = {"n": len(z), "slope_on_early_gap": b_[1], "se": se_[1], "slope_on_fg_move": b_[2], "se2": se_[2],
                 "sd_deriv_move": float(dd.std())}
    out["early_to_close"] = r
    # (5) outcome check of fair values at close (mean residual / mse), one row per game-market
    one = q[q.snap == "close"].groupby(["game_id", "mk"]).agg(
        X=("X", "first"), cons=("dcons", "first"), fg=("fair_fg", "first"), anch=("fair_anch", "first"),
        bol=("dsharp", "first")).reset_index()
    one["fam"] = one.mk.map(fam)
    r = {}
    for f_, z in one.groupby("fam"):
        r[f_] = {c: {"n": int(z[c].notna().sum()), "mean_resid": float((z.X - z[c]).mean()),
                     "mse": float(((z.X - z[c]) ** 2).mean())} for c in ("cons", "fg", "anch", "bol")}
    out["close_fair_vs_outcome"] = r
    return out


# ================================================================== DEV (2023)
def candidate_rules():
    R = []
    mk_sets = {"spreads_h1": ["spreads_h1"], "totals_h1": ["totals_h1"], "team_totals": ["team_totals"],
               "all": ["spreads_h1", "totals_h1", "team_totals"]}
    for fair_c in ("fair_fg", "fair_anch", "fair_bol", "fair_cons", "fair_consq", "fair_bolq"):
        for snap in ("early", "close"):
            for mname, mks in mk_sets.items():
                for thr in (0.0, 0.02, 0.04, 0.06, 0.08):
                    R.append({"fair": fair_c, "snap": snap, "markets": mks, "ev_min": thr, "mname": mname})
    for snap in ("early", "close"):
        for mname, mks in mk_sets.items():
            for thr in (0.0, 0.01, 0.02, 0.03):
                R.append({"fair": "fair_consq", "snap": snap, "markets": mks, "ev_min": thr, "min_ref_books": 3,
                          "mname": mname})
                R.append({"fair": "fair_consq", "snap": snap, "markets": mks, "ev_min": thr, "agree": "anch",
                          "mname": mname})
    for snap in ("early",):
        for mname, mks in mk_sets.items():
            for sm in (0.5, 1.0):
                for thr in (0.0, 0.03):
                    R.append({"fair": "fair_anch", "snap": snap, "markets": mks, "ev_min": thr, "stale_min": sm,
                              "mname": mname})
    for snap in ("early", "close"):
        for mname in ("team_totals", "all"):
            for inc in (0.5, 1.0):
                R.append({"fair": "fair_anch", "snap": snap, "markets": mk_sets[mname], "ev_min": 0.0,
                          "own_incons_min": inc, "mname": mname})
    for snap in ("early", "close"):
        for mname, mks in mk_sets.items():
            for thr in (0.02, 0.04):
                R.append({"fair": "fair_anch", "snap": snap, "markets": mks, "ev_min": thr, "agree": "cons",
                          "mname": mname})
    return R


def rname(r):
    s = f"{r['fair'][5:]}|{r['snap']}|{r['mname']}|ev>={r['ev_min']}"
    for k in ("stale_min", "own_incons_min", "agree", "min_ref_books"):
        if r.get(k) is not None:
            s += f"|{k}={r[k]}"
    return s


def dev_stage():
    q = quotes((2023,))
    o = all_offers(q)
    res = {"n_quotes": len(q), "books_by_market": {f: sorted(z.book.unique().tolist()) for f, z in q.groupby("fam")},
           "describe": describe(q)}
    nweeks = q[["season", "week"]].drop_duplicates().shape[0]
    rows = []
    for r in candidate_rules():
        b = apply_rule(o, q, r)
        s = summarize(b, nweeks)
        rows.append({"rule": rname(r), **{k: s.get(k) for k in ("n", "bets_per_week", "mean_ev", "clv", "clv_t",
                                                                "clv_pos_rate", "roi", "roi_se", "roi_t", "hit")}})
    tab = pd.DataFrame(rows)
    tab.to_csv(SCR / "v2_dev_rules.csv", index=False)
    res["n_candidates"] = len(tab)
    res["rules"] = tab.to_dict("records")
    # calibration: does EV under each fair predict CLV and PnL (all allowed offers with EV > 0, early + close)?
    cal = {}
    for fc in ("fg", "anch", "bol", "cons", "consq", "bolq"):
        for snap in ("early", "close"):
            z = o[(o.snap == snap) & o[f"ev_{fc}"].notna()].copy()
            z["ev"] = z[f"ev_{fc}"]
            z = z[z.ev > -0.03]
            g = grade(z, q)
            bins = pd.cut(g.ev, [-0.03, 0, 0.02, 0.04, 0.07, 0.12, 1])
            cal[f"{fc}|{snap}"] = {str(k): {"n": len(v), "ev": v.ev.mean(), "clv": v.clv.mean(), "roi": v.pnl.mean()}
                                   for k, v in g.groupby(bins, observed=True)}
    res["calibration"] = cal
    save("dev_2023", res)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(tab.sort_values("clv_t", ascending=False).head(60).round(4).to_string())
    return res


# ================================================================== FREEZE / HOLDOUT
ALLM = ["spreads_h1", "totals_h1", "team_totals"]
FROZEN_RULES = [   # chosen on 2023 only (see derivatives.md "Dev 2023"); written once by `freeze`
    {"id": "R1_consq_close", "fair": "fair_consq", "snap": "close", "markets": ALLM, "ev_min": 0.01,
     "min_ref_books": 3, "mname": "all",
     "desc": "75 min pre-kick: best allowed-book price with EV >= 1% vs the leave-one-out median no-vig of >= 3 other "
             "books quoting the SAME point (any derivative market); one bet per game-market"},
    {"id": "R2_consq_early_agree", "fair": "fair_consq", "snap": "early", "markets": ALLM, "ev_min": 0.01,
     "agree": "anch", "mname": "all",
     "desc": "early snapshot (Fri 21:40 UTC / kick-24h): EV >= 1% vs same-point other-book no-vig (pmf fallback when "
             "no other book has that point) AND EV >= 0 vs the sharp-full-game-implied anchored fair"},
    {"id": "R3_model_early", "fair": "fair_anch", "snap": "early", "markets": ALLM, "ev_min": 0.08, "mname": "all",
     "desc": "hypothesis test of 'derivatives priced off the sharp full-game line': early snapshot, EV >= 8% vs "
             "Fair(sharp S,T) + rolling market level offset (key-number pmfs fit 2012-22)"},
]


def freeze_stage(rules: list[dict], note: str):
    if FROZEN_PATH.exists():
        raise SystemExit(f"{FROZEN_PATH} exists; refusing to overwrite")
    assert 1 <= len(rules) <= 3
    FROZEN_PATH.write_text(json.dumps({"frozen_on": pd.Timestamp.utcnow().isoformat(), "dev_seasons": [2023],
                                       "holdout_seasons": [2024, 2025], "allowed_books": sorted(ALLOWED),
                                       "note": note, "rules": rules}, indent=1))
    print(FROZEN_PATH.read_text())


def holdout_stage():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("set EDGE_HOLDOUT=I_HAVE_FROZEN")
    if load("holdout") is not None:
        raise SystemExit("holdout already run once")
    fz = json.loads(FROZEN_PATH.read_text())
    q = quotes((2024, 2025))
    o = all_offers(q)
    res = {"n_quotes": len(q), "books_by_market": {f: sorted(z.book.unique().tolist()) for f, z in q.groupby("fam")},
           "describe": describe(q), "rules": {}}
    for r in fz["rules"]:
        b = apply_rule(o, q, r)
        per = {}
        for s in (2024, 2025):
            z = b[b.season == s]
            per[s] = summarize(z, q[q.season == s][["season", "week"]].drop_duplicates().shape[0])
        nweeks = q[["season", "week"]].drop_duplicates().shape[0]
        res["rules"][r["id"]] = {"rule": r, "pooled": summarize(b, nweeks), "by_season": per,
                                 "clv_same_point_share": float((b.clv_src == "same_point").mean()) if len(b) else None}
        b.to_csv(SCR / f"v2_holdout_bets_{r['id']}.csv", index=False)
    save("holdout", res)
    print(json.dumps(D._js(res["rules"]), indent=1))


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "style"
    if st == "style":
        style_stage()
    elif st == "dev":
        dev_stage()
    elif st == "freeze":
        freeze_stage(FROZEN_RULES, "Selected on 2023 only. R1/R2 = soft-book outliers vs the derivative consensus "
                                   "(R2 also needs the full-game model to agree); R3 = pure full-game-model pricing, "
                                   "kept as an out-of-sample test of the core hypothesis although its 2023 CLV was ~0.")
    elif st == "holdout":
        holdout_stage()
