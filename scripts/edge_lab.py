"""Research harness for finding betting angles on historical early-week lines.

DISCIPLINE (non-negotiable):
  * Development = 2020-2022 seasons only. `load(holdout=False)` refuses to return 2023+.
  * Candidate rules are frozen in `edge_candidates.json` and committed BEFORE the holdout run.
  * Holdout (2023-2025) is run ONCE per frozen candidate list (`--holdout`), and every candidate's
    result is reported, not just the winners.
  * Judge by closing-line value (CLV) first — far less noisy than win/loss — then ROI.

Table built here: one row per (game, snapshot, book side) = every bet we could have placed, with
the model's view, the market consensus, a "sharp" consensus, the closing line and the result.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import replay_early_lines as R
from nflpred import bets as ML, spread_bets as SB, margins as K, odds as O
from nflpred.weather import _kickoff_utc

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "edge_lab_{}.parquet"
DEV, HOLD = (2020, 2021, 2022), (2023, 2024, 2025)
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}


def _nv(h, a):
    ih = np.where(h < 0, -h / (-h + 100), 100 / (h + 100))
    ia = np.where(a < 0, -a / (-a + 100), 100 / (a + 100))
    return ih / (ih + ia)


def _dec(p):
    p = np.asarray(p, float)
    return np.where(p > 0, 1 + p / 100, 1 + 100 / -p)


def build(seasons) -> pd.DataFrame:
    wf = R.walk_forward()
    wf = wf[wf.season.isin(seasons) & wf.home_score.notna()].copy()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in wf.itertuples()], utc=True)
    o = R.load_odds()
    o = o[o.season.isin(seasons)]
    o = R.match_games(o, wf)
    o = o.merge(wf[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick]
    o["nv_home"] = np.where(o.ml_home.notna() & o.ml_away.notna(), _nv(o.ml_home.fillna(100), o.ml_away.fillna(100)), np.nan)
    o["home_margin"] = -o.sp_home_point
    key = ["game_id", "requested_ts"]
    cons = o.groupby(key).agg(p_cons=("nv_home", "median"), m_cons=("home_margin", "median"),
                              n_books=("book", "nunique"))
    sh = o[o.book.isin(SHARP)].groupby(key).agg(p_sharp=("nv_home", "median"), m_sharp=("home_margin", "median"))
    snaps = cons.join(sh).reset_index().merge(
        wf[["game_id", "season", "week", "weekday", "kick", "home_team", "away_team", "home_score", "away_score",
            "spread_line", "vegas_home_prob", "mu_model", "p_model", "home_qb_change", "away_qb_change"]], on="game_id")
    snaps["hours_before"] = (snaps.kick - snaps.requested_ts).dt.total_seconds() / 3600
    # live-rule eligibility for MODEL-based bets (injury report out; QB confirmed), as in the replay
    qd, qout = R.qb_flags(wf)
    el = {}
    for r in wf.itertuples():
        st = R.eligible_from(r.kick, r.weekday)
        if r.game_id in qd:
            st = max(st, r.kick - pd.Timedelta(minutes=90))
        elif r.game_id in qout:
            st = st + pd.Timedelta(days=1)
        el[r.game_id] = st
    snaps["eligible"] = snaps.requested_ts >= snaps.game_id.map(el)
    # first line seen within 9 days (what the live system would call "first seen")
    s9 = snaps[snaps.hours_before <= 9 * 24].sort_values("requested_ts")
    first = s9.groupby("game_id").agg(m_first=("m_cons", "first"), p_first=("p_cons", "first"))
    snaps = snaps.merge(first, on="game_id", how="left")
    allowed = O.load_allowed_books()
    b = o[o.book.isin(allowed)][key + ["book", "ml_home", "ml_away", "sp_home_point", "sp_home_price",
                                        "sp_away_point", "sp_away_price"]]
    rows = []
    for side in ("home", "away"):
        x = b.copy()
        x["side"] = side
        x["ml"] = x[f"ml_{side}"]
        x["point"] = x[f"sp_{side}_point"]
        x["sp_price"] = x[f"sp_{side}_price"]
        rows.append(x[key + ["book", "side", "ml", "point", "sp_price"]])
    t = pd.concat(rows).merge(snaps, on=key)
    sgn = np.where(t.side == "home", 1, -1)
    t["result_margin"] = sgn * (t.home_score - t.away_score)          # our side's margin
    t["is_fav_ml"] = t.ml < 0
    t["is_dog_sp"] = t.point > 0
    # ---- moneyline: fair prob of our side from several views
    for c in ("p_cons", "p_sharp", "p_model", "p_first"):
        t[f"{c}_side"] = np.where(t.side == "home", t[c], 1 - t[c])
    t["p_close_side"] = np.where(t.side == "home", t.vegas_home_prob, 1 - t.vegas_home_prob)
    t["ml_dec"] = _dec(t.ml.fillna(100))
    t.loc[t.ml.isna(), "ml_dec"] = np.nan
    t["ml_win"] = (t.result_margin > 0).astype(float)
    t.loc[t.result_margin == 0, "ml_win"] = np.nan
    t["ml_clv"] = t.ml_dec * t.p_close_side - 1
    t["ml_pnl"] = np.where(t.result_margin > 0, t.ml_dec - 1, np.where(t.result_margin < 0, -1, 0))
    # ---- spread: cover probability of our point/price under several expected margins
    w, sig = SB.load_rules()["_weights"], SB.load_rules()["margin"]["sigma"]
    ok = t.point.notna() & t.sp_price.notna()
    t["sp_dec"] = np.where(ok, _dec(t.sp_price.fillna(-110)), np.nan)

    def ev_under(mu_home):
        mu_side = np.where(t.side == "home", mu_home, -mu_home)       # our side's expected margin
        hc, pu, _ = K.cover_probs(np.nan_to_num(mu_side), sig, np.nan_to_num(t.point.values), w)
        return hc * t.sp_dec + pu - 1

    t["sp_ev_cons"] = ev_under(t.m_cons.values)
    t["sp_ev_sharp"] = ev_under(t.m_sharp.values)
    t["sp_clv"] = ev_under(t.spread_line.values)
    r = SB.load_rules()
    t["mu_blend"] = SB.expected_margin(t.mu_model, t.m_cons, r)
    t["sp_ev_model"] = ev_under(t.mu_blend.values)
    adj = t.result_margin + t.point
    t["sp_pnl"] = np.where(adj > 0, t.sp_dec - 1, np.where(adj < 0, -1, 0))
    t["model_minus_mkt_side"] = np.where(t.side == "home", 1, -1) * (t.mu_model - t.m_cons)
    t["line_move_side"] = np.where(t.side == "home", 1, -1) * (t.m_cons - t.m_first)   # + = market moved toward our side
    t["close_move_side"] = np.where(t.side == "home", 1, -1) * (t.spread_line - t.m_cons)
    t["qb_flag"] = (t.home_qb_change.abs() > 0.05) | (t.away_qb_change.abs() > 0.05)
    return t.drop(columns=["home_qb_change", "away_qb_change"])


def load(holdout: bool = False) -> pd.DataFrame:
    seasons = HOLD if holdout else DEV
    if holdout and os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("holdout is locked: freeze and commit edge_candidates.json first")
    p = Path(str(CACHE).format("holdout" if holdout else "dev"))
    if p.exists():
        return pd.read_parquet(p)
    t = build(seasons)
    t.to_parquet(p)
    return t


def pick_one_per_game(bets: pd.DataFrame, when: str = "first", score: str | None = None) -> pd.DataFrame:
    """Live rule: one bet per game, locked at the first qualifying snapshot, best-scoring book/side there."""
    if bets.empty:
        return bets
    b = bets.sort_values(["game_id", "requested_ts"] + ([score] if score else []),
                         ascending=[True, True] + ([False] if score else []))
    return b.groupby("game_id").head(1) if when == "first" else b.groupby("game_id").tail(1)


def stats(b: pd.DataFrame, market: str) -> dict:
    if b is None or len(b) == 0:
        return {"bets": 0}
    pnl, clv = b[f"{market}_pnl"].values, b[f"{market}_clv"].dropna().values
    n = len(pnl)
    z = clv.mean() / (clv.std(ddof=1) / math.sqrt(len(clv))) if len(clv) > 2 and clv.std() > 0 else 0.0
    se = pnl.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"bets": n, "per_season": round(n / b.season.nunique(), 1), "roi": round(float(pnl.mean()), 4),
            "roi_se": round(float(se), 4), "clv": round(float(clv.mean()), 4),
            "clv_t": round(float(z), 2), "beat_close": round(float((clv > 0).mean()), 3)}


# ---------------------------------------------------------------- honest closing line (prices, not just the number)
def closing_fair(seasons=None) -> pd.DataFrame:
    """Per game, from the LAST pre-kickoff snapshot (~75 min before kickoff):
    mu_close_sharp / mu_close_all = expected home margin implied by spreads AND prices (key-number model);
    p_close_sharp / p_close_all = no-vig home win probability. Use these for CLV; the nflverse
    spread_line ignores the closing juice (that flaw made two spread angles look like winners)."""
    p = ROOT / "data" / "closing_fair.parquet"
    if p.exists():
        c = pd.read_parquet(p)
        return c if seasons is None else c[c.season.isin(seasons)]
    r = SB.load_rules()
    w, sig = r["_weights"], r["margin"]["sigma"]
    wf = R.walk_forward()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(x.gameday, x.gametime) for x in wf.itertuples()], utc=True)
    o = R.match_games(R.load_odds(), wf).merge(wf[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick]
    last = o[o.requested_ts == o.groupby("game_id").requested_ts.transform("max")]
    rows = []
    for gid, d in last.groupby("game_id"):
        rec = {"game_id": gid, "season": int(d.season.iloc[0])}
        for lab, dd in (("sharp", d[d.book.isin(SHARP)]), ("all", d)):
            rec[f"mu_close_{lab}"] = K.market_mu(zip(dd.sp_home_point, dd.sp_home_price, dd.sp_away_point,
                                                     dd.sp_away_price), sig, w)
            m = dd[dd.ml_home.notna() & dd.ml_away.notna()]
            rec[f"p_close_{lab}"] = float(np.median(_nv(m.ml_home.values, m.ml_away.values))) if len(m) else np.nan
        rows.append(rec)
    c = pd.DataFrame(rows)
    c.to_parquet(p)
    return c if seasons is None else c[c.season.isin(seasons)]


def spread_clv_price(mu_close_home, point, price, side) -> float:
    """CLV of a spread bet valued at the price-implied closing expected margin."""
    if mu_close_home is None or pd.isna(mu_close_home):
        return np.nan
    return SB.side_ev(float(mu_close_home), float(point), int(price), side, SB.load_rules())[0]


# ---------------------------------------------------------------- frozen candidates (edge_candidates.json)
def _prep(t: pd.DataFrame):
    side_cons_pt = np.where(t.side == "home", -t.m_cons, t.m_cons)
    t = t.assign(pt_off=t.point - side_cons_pt,
                 ml_off=np.where(t.ml.notna(), 1 / t.ml_dec - t.p_cons_side, np.nan))
    sp = t[t.point.notna() & (t.pt_off.abs() <= 2.5) & t.sp_price.between(-200, 200) & t.m_sharp.notna()].copy()
    ml = t[t.ml.notna() & (t.ml_off.abs() <= 0.12) & t.ml.between(-1000, 1000) & t.p_sharp.notna()].copy()
    ml["ev_sharp"] = ml.ml_dec * ml.p_sharp_side - 1
    ml["ev_model"] = ml.ml_dec * ml.p_model_side - 1
    return sp, ml


def _gate(b: pd.DataFrame, mkt: str, n: int = 20, min_hist: int = 10) -> pd.DataFrame:
    b = b.sort_values("requested_ts").reset_index(drop=True)
    keep = []
    for r in b.itertuples():
        past = b[b.kick < r.requested_ts].tail(n)
        keep.append(len(past) < min_hist or past[f"{mkt}_clv"].mean() > 0)
    return b[keep]


def candidates(t: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, str]]:
    sp, ml = _prep(t)
    P = pick_one_per_game
    return {
        "C1_spread_sharp_dog": (P(sp[(sp.sp_ev_sharp >= 0.03) & (sp.point > 0)], score="sp_ev_sharp"), "sp"),
        "C2_ml_sharp_dog": (P(ml[(ml.ev_sharp >= 0.03) & ml.ml.between(100, 400)], score="ev_sharp"), "ml"),
        "C3_spread_sharp_and_model": (P(sp[sp.eligible & (sp.sp_ev_sharp >= 0.02) & (sp.sp_ev_model >= 0.02)],
                                        score="sp_ev_sharp"), "sp"),
        "C4_ml_sharp_and_model": (P(ml[ml.eligible & (ml.ev_sharp >= 0.02) & (ml.ev_model >= 0)], score="ev_sharp"), "ml"),
        "C5_spread_model_dog": (P(sp[sp.eligible & (sp.point > 0) & (sp.sp_ev_model >= 0.03)], score="sp_ev_model"), "sp"),
        "C6_ml_longshot_model": (P(ml[ml.eligible & ml.ml.between(150, 300) & ((ml.p_model_side - ml.p_sharp_side) >= 0.12)],
                                   score="ev_model"), "ml"),
        "C7_spread_sharp_gated": (_gate(P(sp[sp.sp_ev_sharp >= 0.02], score="sp_ev_sharp"), "sp"), "sp"),
    }


def evaluate(holdout: bool) -> dict:
    t = load(holdout)
    out = {}
    for cid, (b, mkt) in candidates(t).items():
        s = stats(b, mkt)
        n = s.get("bets", 0)
        p = 0.5 * math.erfc(s.get("clv_t", 0) / math.sqrt(2)) if n > 2 else 1.0
        s["clv_p"] = round(p, 5)
        s["pass"] = bool(s.get("clv", 0) > 0 and p < 0.05 / 7)
        s["by_season"] = {int(y): stats(g, mkt) for y, g in b.groupby("season")}
        out[cid] = s
    return out


if __name__ == "__main__":
    import json, sys
    hold = "--holdout" in sys.argv
    res = evaluate(hold)
    name = "edge_holdout_2023_2025.json" if hold else "edge_dev_2020_2022.json"
    (ROOT / "output" / name).write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:28s} n={v['bets']:4d} roi={v.get('roi',0):+.3f}±{v.get('roi_se',0):.3f} clv={v.get('clv',0):+.4f} "
              f"p={v['clv_p']:.4f} beat={v.get('beat_close',0):.2f} PASS={v['pass']}")
