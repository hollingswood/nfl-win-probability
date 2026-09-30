"""Openers and check frequency for soft-vs-sharp moneyline/spread line shopping (research only).

Question A (openers): are the Sunday-night opening lines for next week (Sun 23:30 UTC and Mon 04:00 UTC
snapshots, data/historical_odds/openers/) softer or more exploitable than the lines our live runs see?
Question B (check frequency): with HOURLY snapshots Sat 15:00 -> Sun 16:00 UTC (2025 weeks 3-12, Sunday
games, data/historical_odds/hourly/), how many extra C2/C4-style opportunities appear vs our existing run
times, how long do they last, and what is their CLV?

DISCIPLINE
  * A is developed on 2020-2022 only (`dev`). At most 2 opener rules are frozen in
    output/research/openers_frozen.json (`freeze`, never overwritten), then 2023-2025 is run ONCE (`holdout`).
    Pass = mean price-based CLV (vs edge_lab.closing_fair() p_close_all for moneyline / mu_close_all for spreads)
    > 0 with one-sided p < 0.025 (Bonferroni over 2 rules). nflverse close and the sharp-only close are reported too.
  * B uses 2025 only, which is part of the 2023-2025 holdout already used once for C2/C4 -> descriptive only.
  * Model-based rules at the opener use data/replay_predictions.parquet, whose QB/injury inputs are the FINAL
    report versions. At the Sunday-night opener the injury report is not out, so the live system would have
    had older inputs: model-based opener results are optimistic (look-ahead), not just stale.
  * Books: the openers/hourly files are region "us" only. For like-for-like comparisons, later snapshots from
    data/historical_odds/nfl_odds_*.csv.gz are restricted to the books present in the opener/hourly file of the
    same season (so ESPN Bet / Fanatics(pre-2025) / Hard Rock etc. are excluded everywhere). "Sharp" in these
    files is effectively lowvig + betonlineag (one company); circasports/bookmaker appear only in 2020/2022.

    PYTHONPATH=src:scripts python scripts/research/openers_hourly.py dev
    PYTHONPATH=src:scripts python scripts/research/openers_hourly.py freeze
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/openers_hourly.py holdout
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/openers_hourly.py hourly
    PYTHONPATH=src:scripts python scripts/research/openers_hourly.py report
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import edge_lab as E  # noqa: E402
import replay_early_lines as R  # noqa: E402
from nflpred import margins as K, odds as O, spread_bets as SB  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "openers_hourly.json"
MD = OUT / "openers_hourly.md"
FROZEN = OUT / "openers_frozen.json"
HIST = ROOT / "data" / "historical_odds"
DEV, HOLD = (2020, 2021, 2022), (2023, 2024, 2025)
ALPHA = 0.05 / 2
OPEN_HB = (72, 216)  # an opener snapshot counts for games 3-9 days out (next week's slate, not look-ahead lines)
SPR = SB.load_rules()
W, SIG = SPR["_weights"], SPR["margin"]["sigma"]


# ============================================================================ loading
def games() -> pd.DataFrame:
    wf = R.walk_forward().copy()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in wf.itertuples()], utc=True)
    wf = wf[wf.home_score.notna()].reset_index(drop=True)
    cf = E.closing_fair()[["game_id", "p_close_all", "p_close_sharp", "mu_close_all", "mu_close_sharp"]]
    wf = wf.merge(cf, on="game_id", how="left")
    # eligibility for model-based rules (same as edge_lab.build / the replay)
    qd, qout = R.qb_flags(wf)
    el = []
    for r in wf.itertuples():
        st = R.eligible_from(r.kick, r.weekday)
        if r.game_id in qd:
            st = max(st, r.kick - pd.Timedelta(minutes=90))
        elif r.game_id in qout:
            st = st + pd.Timedelta(days=1)
        el.append(st)
    wf["eligible_from"] = el
    wf["qb_questionable"] = wf.game_id.isin(qd)
    # "clean" = no QB on any final report and no QB change: removes the largest look-ahead in p_model/mu_model
    # (final-report QB/injury inputs) when the model is used at an opener
    wf["qb_clean"] = ~wf.game_id.isin(qd | qout) & (wf.home_qb_change.abs().fillna(0) <= 0.05) & \
        (wf.away_qb_change.abs().fillna(0) <= 0.05)
    # kickoff of each team's previous game: at an opener the live model would NOT know results of games still
    # being played (late Sunday, SNF, MNF), but replay_predictions' features include them
    long = pd.concat([wf[["game_id", "season", "kick", "home_team"]].rename(columns={"home_team": "team"}),
                      wf[["game_id", "season", "kick", "away_team"]].rename(columns={"away_team": "team"})])
    long = long.sort_values("kick")
    long["prev_kick"] = long.groupby(["season", "team"]).kick.shift(1)
    pk = long.groupby("game_id").prev_kick.max()
    wf["prev_kick_max"] = wf.game_id.map(pk)
    return wf


def read_odds(kind: str, seasons) -> pd.DataFrame:
    d = HIST / kind if kind else HIST
    fr = []
    for s in seasons:
        p = d / f"nfl_odds_{s}.csv.gz"
        if p.exists():
            fr.append(pd.read_csv(p).assign(season=s))
    o = pd.concat(fr, ignore_index=True)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    return o


def _ev_spread(mu_home, side, point, dec):
    """Vectorized SB.side_ev(...)[0]: EV of `side` at its own `point`, decimal price `dec`, true mean home margin mu."""
    mu_home = np.asarray(mu_home, float)
    ok = ~np.isnan(mu_home) & ~np.isnan(point) & ~np.isnan(dec)
    home_line = np.where(side == "home", point, -point)
    hc, pu, ac = K.cover_probs(np.nan_to_num(mu_home), SIG, np.nan_to_num(home_line), W)
    pw = np.where(side == "home", hc, ac)
    return np.where(ok, pw * dec + pu - 1, np.nan)


def table(o: pd.DataFrame, wf: pd.DataFrame, books: dict[int, set] | None = None) -> pd.DataFrame:
    """One row per (game, snapshot, soft book, side). `books[season]` restricts ALL books (consensus, sharp,
    soft) to that set, for like-for-like comparisons across files."""
    o = R.match_games(o, wf).merge(wf[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick]
    if books is not None:
        o = o[[b in books.get(s, set()) for b, s in zip(o.book, o.season)]]
    o = o.copy()
    o["nv_home"] = np.where(o.ml_home.notna() & o.ml_away.notna(),
                            E._nv(o.ml_home.fillna(100), o.ml_away.fillna(100)), np.nan)
    o["home_margin"] = -o.sp_home_point
    key = ["game_id", "requested_ts"]
    cons = o.groupby(key).agg(p_cons=("nv_home", "median"), m_cons=("home_margin", "median"), n_books=("book", "nunique"))
    sh = o[o.book.isin(E.SHARP)].groupby(key).agg(p_sharp=("nv_home", "median"), m_sharp=("home_margin", "median"),
                                                  n_sharp=("book", "nunique"))
    snaps = cons.join(sh).reset_index()
    soft = O.load_allowed_books()
    b = o[o.book.isin(soft)]
    parts = []
    for side in ("home", "away"):
        x = b[key + ["book"]].copy()
        x["side"] = side
        x["ml"] = b[f"ml_{side}"].values
        x["point"] = b[f"sp_{side}_point"].values
        x["sp_price"] = b[f"sp_{side}_price"].values
        parts.append(x)
    t = pd.concat(parts, ignore_index=True).merge(snaps, on=key).merge(
        wf[["game_id", "season", "week", "weekday", "kick", "home_score", "away_score", "spread_line", "vegas_home_prob",
            "mu_model", "p_model", "p_close_all", "p_close_sharp", "mu_close_all", "mu_close_sharp", "eligible_from", "qb_clean", "prev_kick_max"]],
        on="game_id")
    t["hours_before"] = (t.kick - t.requested_ts).dt.total_seconds() / 3600
    t["eligible"] = t.requested_ts >= t.eligible_from
    # prior games of both teams finished (kick + 4h) before this snapshot; "clean" = that AND no QB news
    t["prev_done"] = t.prev_kick_max.isna() | (t.requested_ts >= t.prev_kick_max + pd.Timedelta(hours=4))
    t["clean"] = t.qb_clean & t.prev_done
    home = (t.side == "home").values
    sgn = np.where(home, 1, -1)
    t["result_margin"] = sgn * (t.home_score - t.away_score)
    for c in ("p_cons", "p_sharp", "p_model", "vegas_home_prob", "p_close_all", "p_close_sharp"):
        t[f"{c}_side"] = np.where(home, t[c], 1 - t[c])
    t["ml_dec"] = np.where(t.ml.notna(), E._dec(t.ml.fillna(100)), np.nan)
    t["ev_sharp"] = t.ml_dec * t.p_sharp_side - 1
    t["ev_model"] = t.ml_dec * t.p_model_side - 1
    t["ml_off"] = 1 / t.ml_dec - t.p_cons_side
    t["ml_clv_nfl"] = t.ml_dec * t.vegas_home_prob_side - 1
    t["ml_clv_all"] = t.ml_dec * t.p_close_all_side - 1
    t["ml_clv_sharp"] = t.ml_dec * t.p_close_sharp_side - 1
    t["ml_pnl"] = np.where(t.result_margin > 0, t.ml_dec - 1, np.where(t.result_margin < 0, -1.0, 0.0))
    side = t.side.values
    pt = t.point.values.astype(float)
    t["sp_dec"] = np.where(t.sp_price.notna(), E._dec(t.sp_price.fillna(-110)), np.nan)
    dec = t.sp_dec.values
    t["pt_off"] = t.point - np.where(home, -t.m_cons, t.m_cons)
    t["sp_ev_sharp"] = _ev_spread(t.m_sharp.values, side, pt, dec)
    t["sp_ev_model"] = _ev_spread(SB.expected_margin(t.mu_model, t.m_cons, SPR).values, side, pt, dec)
    t["sp_clv_all"] = _ev_spread(t.mu_close_all.values, side, pt, dec)
    t["sp_clv_sharp"] = _ev_spread(t.mu_close_sharp.values, side, pt, dec)
    t["sp_clv_nfl"] = _ev_spread(t.spread_line.values, side, pt, dec)
    adj = t.result_margin + t.point
    t["sp_pnl"] = np.where(adj > 0, t.sp_dec - 1, np.where(adj < 0, -1.0, 0.0))
    return t


def sane_ml(t):
    return t[t.ml.notna() & (t.ml_off.abs() <= 0.12) & t.ml.between(-1000, 1000) & t.p_sharp.notna()]


def sane_sp(t):
    return t[t.point.notna() & (t.pt_off.abs() <= 2.5) & t.sp_price.between(-200, 200) & t.m_sharp.notna()]


# ============================================================================ rules
# every rule: (market, filter on sane rows -> mask, score column). "model" rules flagged: look-ahead at openers.
RULES = {
    "ml_dog_sharp3": ("ml", lambda x: (x.ev_sharp >= 0.03) & x.ml.between(100, 400), "ev_sharp", False),
    "ml_dog_sharp2": ("ml", lambda x: (x.ev_sharp >= 0.02) & x.ml.between(100, 400), "ev_sharp", False),
    "ml_dog_sharp5": ("ml", lambda x: (x.ev_sharp >= 0.05) & x.ml.between(100, 400), "ev_sharp", False),
    "ml_any_sharp2": ("ml", lambda x: x.ev_sharp >= 0.02, "ev_sharp", False),
    "ml_any_sharp3": ("ml", lambda x: x.ev_sharp >= 0.03, "ev_sharp", False),
    "ml_fav_sharp2": ("ml", lambda x: (x.ev_sharp >= 0.02) & (x.ml < 100), "ev_sharp", False),
    "ml_sharp2_model0": ("ml", lambda x: (x.ev_sharp >= 0.02) & (x.ev_model >= 0), "ev_sharp", True),
    "ml_model5": ("ml", lambda x: (x.p_model_side - x.p_cons_side >= 0.05) & (x.ev_model >= 0.03), "ev_model", True),
    "ml_model8": ("ml", lambda x: (x.p_model_side - x.p_cons_side >= 0.08) & (x.ev_model >= 0.05), "ev_model", True),
    "sp_dog_sharp3": ("sp", lambda x: (x.sp_ev_sharp >= 0.03) & (x.point > 0), "sp_ev_sharp", False),
    "sp_any_sharp2": ("sp", lambda x: x.sp_ev_sharp >= 0.02, "sp_ev_sharp", False),
    "sp_any_sharp3": ("sp", lambda x: x.sp_ev_sharp >= 0.03, "sp_ev_sharp", False),
    "sp_sharp2_model2": ("sp", lambda x: (x.sp_ev_sharp >= 0.02) & (x.sp_ev_model >= 0.02), "sp_ev_sharp", True),
    "sp_model3": ("sp", lambda x: x.sp_ev_model >= 0.03, "sp_ev_model", True),
    # same model rules on "clean" rows only: no QB on any report / no QB change, and both teams' previous games
    # finished before the snapshot (less look-ahead when the model is used at an opener)
    "ml_model5_clean": ("ml", lambda x: x.clean & (x.p_model_side - x.p_cons_side >= 0.05) & (x.ev_model >= 0.03),
                        "ev_model", True),
    "sp_model3_clean": ("sp", lambda x: x.clean & (x.sp_ev_model >= 0.03), "sp_ev_model", True),
    "ml_sharp2_model0_clean": ("ml", lambda x: x.clean & (x.ev_sharp >= 0.02) & (x.ev_model >= 0), "ev_sharp", True),
}


def bets(t: pd.DataFrame, rule: str, eligible_only: bool = False) -> pd.DataFrame:
    mkt, f, score, _ = RULES[rule]
    x = sane_ml(t) if mkt == "ml" else sane_sp(t)
    if eligible_only:
        x = x[x.eligible]
    return E.pick_one_per_game(x[f(x)], score=score)


def _p(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def summ(b: pd.DataFrame, mkt: str) -> dict:
    if b is None or len(b) == 0:
        return {"bets": 0}
    out = {"bets": int(len(b)), "per_season": round(len(b) / b.season.nunique(), 1),
           "avg_price_or_point": round(float(b.ml.mean() if mkt == "ml" else b.point.mean()), 2),
           "avg_ev_sharp": round(float(b[f"{'ev_sharp' if mkt == 'ml' else 'sp_ev_sharp'}"].mean()), 4),
           "median_hours_before": round(float(b.hours_before.median()), 1)}
    for c in ("all", "sharp", "nfl"):
        v = b[f"{mkt}_clv_{c}"].dropna().values
        if len(v) > 2 and v.std() > 0:
            z = v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))
            out[f"clv_{c}"] = round(float(v.mean()), 4)
            out[f"clv_{c}_p"] = round(_p(z), 5)
            out[f"beat_{c}"] = round(float((v > 0).mean()), 3)
    pnl = b[f"{mkt}_pnl"].values
    out["roi"] = round(float(pnl.mean()), 4)
    out["roi_se"] = round(float(pnl.std(ddof=1) / math.sqrt(len(pnl))), 4) if len(pnl) > 1 else None
    return out


# ============================================================================ descriptive analyses
def bucket(hb: pd.Series, is_close: pd.Series) -> pd.Series:
    lab = np.select([is_close, hb < 24, hb < 72, hb < 120], ["close (last pre-kick)", "gameday/eve (<24h)",
                                                            "1-3 days", "3-5 days"], "5+ days (main)")
    return pd.Series(lab, index=hb.index)


def gaps_by_bucket(to: pd.DataFrame, tm: pd.DataFrame) -> dict:
    """Soft-vs-sharp gap per (game, snapshot): max over sides/books of EV vs sharp no-vig (ML) and spread."""
    out = {}
    tm = tm.copy()
    last = tm.groupby("game_id").requested_ts.transform("max")
    tm["bucket"] = bucket(tm.hours_before, tm.requested_ts == last)
    to = to.assign(bucket="opener (Sun 23:30/Mon 04:00)")
    for lab, d in pd.concat([to, tm]).groupby("bucket"):
        ml = sane_ml(d).groupby(["game_id", "requested_ts"]).ev_sharp.max()
        ml_dog = sane_ml(d[d.ml.between(100, 400)]).groupby(["game_id", "requested_ts"]).ev_sharp.max()
        sp = sane_sp(d).groupby(["game_id", "requested_ts"]).sp_ev_sharp.max()
        nb = d.groupby(["game_id", "requested_ts"]).n_books.first()
        out[lab] = {"snapshots": int(len(ml)), "games": int(d.game_id.nunique()),
                    "books_per_snapshot": round(float(nb.mean()), 1),
                    "ml_best_ev_vs_sharp_mean": round(float(ml.mean()), 4),
                    "ml_share_snap_ge2pct": round(float((ml >= 0.02).mean()), 3),
                    "ml_share_snap_ge3pct": round(float((ml >= 0.03).mean()), 3),
                    "ml_dog_share_snap_ge3pct": round(float((ml_dog >= 0.03).reindex(ml.index).fillna(False).mean()), 3),
                    "sp_best_ev_vs_sharp_mean": round(float(sp.mean()), 4),
                    "sp_share_snap_ge2pct": round(float((sp >= 0.02).mean()), 3),
                    "sp_share_snap_ge3pct": round(float((sp >= 0.03).mean()), 3)}
    return out


def movement(to: pd.DataFrame, wf: pd.DataFrame) -> dict:
    """How far lines move from the opener (first opener snapshot) to the close."""
    s = to.groupby(["game_id", "requested_ts"]).agg(p_cons=("p_cons", "first"), m_cons=("m_cons", "first"),
                                                     p_sharp=("p_sharp", "first")).reset_index()
    s = s.sort_values("requested_ts").groupby("game_id").first().reset_index().merge(wf, on="game_id")
    dp_all = (s.p_close_all - s.p_cons).abs()
    dp_nfl = (s.vegas_home_prob - s.p_cons).abs()
    dm_nfl = (s.spread_line - s.m_cons).abs()  # both = expected home margin (nflverse spread_line > 0: home favored)
    dmu = (s.mu_close_all - s.m_cons).abs()
    y = (s.home_score > s.away_score).astype(float)
    ok = s.home_score != s.away_score

    def ll(p):
        p = np.clip(p[ok], 1e-4, 1 - 1e-4)
        return round(float(-np.mean(y[ok] * np.log(p) + (1 - y[ok]) * np.log(1 - p))), 4)

    return {"games": int(len(s)),
            "abs_winprob_move_vs_price_close_mean": round(float(dp_all.mean()), 4),
            "abs_winprob_move_vs_nflverse_close_mean": round(float(dp_nfl.mean()), 4),
            "abs_spread_move_pts_vs_nflverse_mean": round(float(dm_nfl.mean()), 2),
            "abs_spread_move_pts_vs_nflverse_median": round(float(dm_nfl.median()), 2),
            "share_spread_moved_ge_1pt": round(float((dm_nfl >= 1).mean()), 3),
            "share_spread_moved_ge_2pt": round(float((dm_nfl >= 2).mean()), 3),
            "share_spread_crossed_3_or_7": round(float(_crossed(s.m_cons, s.spread_line).mean()), 3),
            "abs_mu_move_vs_price_close_mean": round(float(dmu.mean()), 2),
            "logloss_opener_consensus": ll(s.p_cons.values), "logloss_opener_sharp": ll(s.p_sharp.fillna(s.p_cons).values),
            "logloss_close_price": ll(s.p_close_all.values), "logloss_close_nflverse": ll(s.vegas_home_prob.values),
            "logloss_model": ll(s.p_model.values)}


def _crossed(a, b):
    lo, hi = np.minimum(a.abs(), b.abs()), np.maximum(a.abs(), b.abs())
    return ((lo < 3) & (hi > 3)) | ((lo < 7) & (hi > 7)) | (np.sign(a) != np.sign(b))


def model_vs_market(to: pd.DataFrame, tm: pd.DataFrame, wf: pd.DataFrame) -> dict:
    """Does the model know something the line doesn't? At each timing bucket, regress the move to the close
    (p_close_all - p_cons) on the model's disagreement (p_model - p_cons), one row per game (earliest
    snapshot in the bucket). slope>0 = the market moves toward the model afterwards."""
    tm = tm.copy()
    last = tm.groupby("game_id").requested_ts.transform("max")
    tm["bucket"] = bucket(tm.hours_before, tm.requested_ts == last)
    to = to.assign(bucket="opener (Sun 23:30/Mon 04:00)")
    out = {}
    for lab, d in pd.concat([to, tm]).groupby("bucket"):
        s = d.sort_values("requested_ts").groupby("game_id").first().reset_index()
        s = s[s.p_cons.notna() & s.p_close_all.notna()]
        x = (s.p_model - s.p_cons).values
        y = (s.p_close_all - s.p_cons).values
        xc = x - x.mean()
        slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
        res = y - y.mean() - slope * xc
        se = float(np.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))
        big = np.abs(x) >= 0.05
        mv = np.sign(x[big]) * y[big]
        cl = s.clean.values.astype(bool)
        xq, yq = x[cl] - x[cl].mean(), y[cl] - y[cl].mean()
        sq = float((xq * yq).sum() / (xq ** 2).sum())
        seq = float(np.sqrt(((yq - sq * xq) ** 2).sum() / (cl.sum() - 2) / (xq ** 2).sum()))
        out[lab] = {"games": int(len(s)), "mean_abs_model_minus_market": round(float(np.abs(x).mean()), 4),
                    "slope_clean_rows": round(sq, 3), "clean_rows": int(cl.sum()), "slope_t_clean": round(sq / seq, 2) if seq > 1e-9 else None,
                    "slope_close_move_on_model_gap": round(slope, 3), "slope_t": round(slope / se, 2) if se > 1e-9 else None,
                    "games_gap_ge5pct": int(big.sum()),
                    "avg_move_toward_model_when_gap_ge5pct": round(float(mv.mean()), 4) if big.any() else None}
    return out


# ============================================================================ runs
def _store(key, val):
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = val
    OUT.mkdir(parents=True, exist_ok=True)
    JSON.write_text(json.dumps(cur, indent=2, default=str))


def build(seasons):
    wf = games()
    op = read_odds("openers", seasons)
    books = {s: set(op[op.season == s].book.unique()) for s in seasons}
    to = table(op, wf)
    to = to[to.hours_before.between(*OPEN_HB)]
    tm = table(read_odds("", seasons), wf, books)
    return wf, to, tm


def compare_rules(to, tm, names) -> dict:
    """Same rule at the opener, at the later (main) snapshots, and 'opener first, else later'."""
    out = {}
    for r in names:
        mkt, _, _, is_model = RULES[r]
        bo, bm = bets(to, r), bets(tm, r)
        both = E.pick_one_per_game(pd.concat([bo, bm]))  # earliest qualifying snapshot across both
        extra = bo[~bo.game_id.isin(bm.game_id)]
        rec = {"uses_model": is_model, "opener": summ(bo, mkt), "later_main_snapshots": summ(bm, mkt),
               "opener_then_later": summ(both, mkt), "opener_bets_on_games_later_never_qualify": summ(extra, mkt)}
        if is_model:
            rec["later_main_eligible_only"] = summ(bets(tm, r, eligible_only=True), mkt)
        out[r] = rec
    return out


def run_dev():
    wf, to, tm = build(DEV)
    wd = wf[wf.season.isin(DEV)]
    res = {"seasons": list(DEV), "opener_games": int(to.game_id.nunique()), "games": int(len(wd)),
           "gaps_by_bucket": gaps_by_bucket(to, tm), "movement_opener_to_close": movement(to, wd),
           "model_vs_market_by_bucket": model_vs_market(to, tm, wd),
           "rules": compare_rules(to, tm, list(RULES))}
    res["rules_by_season_opener"] = {r: {int(s): summ(g, RULES[r][0]) for s, g in bets(to, r).groupby("season")}
                                     for r in RULES}
    _store("dev", res)
    print(json.dumps(res, indent=1, default=str))


def freeze(rules: list[str], why: str):
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists; frozen rules are never overwritten")
    assert 1 <= len(rules) <= 2 and all(r in RULES for r in rules)
    FROZEN.write_text(json.dumps({
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "developed_on": list(DEV), "holdout": list(HOLD),
        "evaluate_once": True, "rules": rules, "why": why,
        "definition": {r: {"market": RULES[r][0], "score": RULES[r][2], "uses_model": RULES[r][3]} for r in rules},
        "bet_placement": "opener snapshots only (Sun 23:30 and Mon 04:00 UTC, games 72-216h away); one bet per game at the "
                         "first qualifying opener snapshot, best-scoring book/side; books = my_books.json present in the "
                         "region-us opener file; sanity filters as edge_lab._prep",
        "pass_bar": "mean price-based CLV (ML: p_close_all; spread: mu_close_all via key-number pricing) > 0 with "
                    "one-sided p < 0.025 (0.05/2)"}, indent=2))
    print("frozen", FROZEN)


def run_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout locked: freeze first and set EDGE_HOLDOUT")
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    if "holdout" in cur:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    wf, to, tm = build(HOLD)
    wh = wf[wf.season.isin(HOLD)]
    res = {"seasons": list(HOLD), "frozen": fz["rules"], "results": {}}
    for r in fz["rules"]:
        mkt = RULES[r][0]
        b = bets(to, r)
        s = summ(b, mkt)
        s["pass"] = bool(s.get("clv_all", 0) > 0 and s.get("clv_all_p", 1) < ALPHA)
        s["by_season"] = {int(y): summ(g, mkt) for y, g in b.groupby("season")}
        res["results"][r] = s
    # descriptive (after the freeze): same tables as dev, plus every rule side by side
    res["opener_games"] = int(to.game_id.nunique())
    res["gaps_by_bucket"] = gaps_by_bucket(to, tm)
    res["movement_opener_to_close"] = movement(to, wh)
    res["model_vs_market_by_bucket"] = model_vs_market(to, tm, wh)
    res["all_rules_descriptive_not_a_test"] = compare_rules(to, tm, list(RULES))
    _store("holdout", res)
    print(json.dumps(res, indent=1, default=str))


# ============================================================================ B: hourly checks (2025, descriptive)
def _runs(ts_sorted: list) -> list[int]:
    """Lengths of runs of consecutive hourly timestamps."""
    runs, n = [], 1
    for a, b in zip(ts_sorted, ts_sorted[1:]):
        if b - a == pd.Timedelta(hours=1):
            n += 1
        else:
            runs.append(n)
            n = 1
    runs.append(n)
    return runs


def run_hourly():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("2025 is holdout data: set EDGE_HOLDOUT (descriptive use only)")
    wf = games()
    hr = read_odds("hourly", [2025])
    hbooks = {2025: set(hr.book.unique())}
    th = table(hr, wf)
    main = read_odds("", [2025])
    tm = table(main, wf, hbooks)  # existing run times, same (region-us) book set
    # close time = our last pre-kickoff main snapshot (what closing_fair uses); bets must be strictly before it
    mo = R.match_games(main, wf).merge(wf[["game_id", "kick"]], on="game_id")
    close_ts = mo[mo.requested_ts < mo.kick].groupby("game_id").requested_ts.max()
    th = th[th.requested_ts < th.game_id.map(close_ts)]
    # only that weekend's Sunday games (the file also carries MNF / next week's look-ahead lines)
    th = th[(th.weekday == "Sunday") & (th.hours_before <= 48)]
    gids = set(th.game_id)
    g = wf[wf.game_id.isin(gids)]
    assert (g.weekday == "Sunday").all()
    tm = tm[tm.game_id.isin(gids)]
    w0, w1 = th.requested_ts.min(), th.requested_ts.max()
    # per-game window: that weekend's Sat 15:00 -> Sun 16:00
    win_lo = th.groupby("game_id").requested_ts.min()
    win_hi = th.groupby("game_id").requested_ts.max()
    tm_in = tm[(tm.requested_ts >= tm.game_id.map(win_lo)) & (tm.requested_ts <= tm.game_id.map(win_hi))]
    th = th.assign(src="hourly")
    tm = tm.assign(src="existing")
    weeks = int(g.week.nunique())
    res = {"note": "2025 weeks 3-12, Sunday games only; 2025 is part of the 2023-25 holdout already used for C2/C4 -> "
                   "descriptive, not a validation. Region us only (no ESPN Bet / Hard Rock etc.).",
           "games": int(len(g)), "weeks": weeks, "hourly_snapshots": int(th.requested_ts.nunique()),
           "window_utc": [str(w0), str(w1)],
           "existing_snapshots_inside_window_per_game": round(float(tm_in.groupby("game_id").requested_ts.nunique().mean()), 2),
           "existing_run_times_inside_window": sorted(tm_in.requested_ts.dt.strftime("%a %H:%M").unique().tolist()),
           "rules": {}}
    RB = {"C2_ml_sharp_dog": ("ml", lambda x: (x.ev_sharp >= 0.03) & x.ml.between(100, 400), "ev_sharp", False),
          "C4_ml_sharp_and_model": ("ml", lambda x: (x.ev_sharp >= 0.02) & (x.ev_model >= 0), "ev_sharp", True),
          "C1_spread_sharp_dog": ("sp", lambda x: (x.sp_ev_sharp >= 0.03) & (x.point > 0), "sp_ev_sharp", False)}
    for rid, (mkt, f, score, elig) in RB.items():
        def qual(t):
            x = sane_ml(t) if mkt == "ml" else sane_sp(t)
            if elig:
                x = x[x.eligible]
            return x[f(x)]
        qh, qm, qmi = qual(th), qual(tm), qual(tm_in)
        P = lambda b: E.pick_one_per_game(b, score=score)
        b_exist = P(qm)                                   # existing schedule, whole week
        b_all = P(pd.concat([qm, qh]))                    # existing + hourly Sat-Sun window
        b_win_exist, b_win_hourly = P(qmi), P(qh)          # inside the window only
        new_games = b_all[~b_all.game_id.isin(b_exist.game_id)]
        changed = b_all[b_all.game_id.isin(b_exist.game_id) & (b_all.src == "hourly")]
        # persistence: (game, side) opportunities across consecutive hourly snapshots
        runs, runs_book, first_rows = [], [], []
        for (gid, sd), d in qh.groupby(["game_id", "side"]):
            ts = sorted(d.requested_ts.unique())
            rr = _runs(ts)
            runs += rr
            # first row of each run (the moment an hourly checker would see it)
            starts = [ts[0]] + [b for a, b in zip(ts, ts[1:]) if b - a != pd.Timedelta(hours=1)]
            for s0 in starts:
                first_rows.append(d[d.requested_ts == s0].sort_values(score, ascending=False).iloc[0])
            for bk, db in d.groupby("book"):
                runs_book += _runs(sorted(db.requested_ts.unique()))
        fr = pd.DataFrame(first_rows)
        # was the opportunity visible at an existing run time inside the window (same game & side)?
        seen_exist = set(zip(qmi.game_id, qmi.side))
        runs = np.array(runs) if runs else np.array([0])
        runs_book = np.array(runs_book) if runs_book else np.array([0])
        res["rules"][rid] = {
            "uses_model": elig, "market": mkt,
            "bets_existing_schedule_full_week": summ(b_exist, mkt),
            "bets_existing_plus_hourly": summ(b_all, mkt),
            "extra_games_only_with_hourly": summ(new_games, mkt),
            "extra_bets_per_week": round(len(new_games) / weeks, 2),
            "games_where_hourly_locks_a_different_earlier_or_better_bet": int(len(changed)),
            "inside_window_existing_runs_only": summ(b_win_exist, mkt),
            "inside_window_hourly": summ(b_win_hourly, mkt),
            "hourly_opportunities_game_side": int(len(fr)),
            "hourly_opps_not_visible_at_existing_runs_in_window": int(sum((a, b) not in seen_exist
                                                                          for a, b in zip(fr.game_id, fr.side))) if len(fr) else 0,
            "all_hourly_opportunity_starts_clv": summ(fr, mkt) if len(fr) else {"bets": 0},
            "persistence_hours_game_side": {"runs": int(len(runs)), "mean": round(float(runs.mean()), 2),
                                            "median": float(np.median(runs)), "share_1h_only": round(float((runs == 1).mean()), 3),
                                            "share_ge3h": round(float((runs >= 3).mean()), 3),
                                            "share_ge6h": round(float((runs >= 6).mean()), 3), "max": int(runs.max())},
            "persistence_hours_same_book": {"runs": int(len(runs_book)), "mean": round(float(runs_book.mean()), 2),
                                            "median": float(np.median(runs_book)),
                                            "share_1h_only": round(float((runs_book == 1).mean()), 3)},
            "hourly_qualifying_share_of_snapshots": round(float(qh.groupby(["game_id", "requested_ts"]).ngroups /
                                                                max(th.groupby(["game_id", "requested_ts"]).ngroups, 1)), 4),
        }
    # hour-of-day profile of the ML gap (soft best vs sharp) in the hourly window
    x = sane_ml(th).groupby(["game_id", "requested_ts"]).ev_sharp.max().reset_index()
    x["hour"] = x.requested_ts.dt.strftime("%a %H")
    prof = x.groupby("hour", sort=False).ev_sharp.agg(["mean", lambda v: (v >= 0.02).mean(), "count"])
    prof.columns = ["mean_best_ev_vs_sharp", "share_ge2pct", "n"]
    res["gap_by_hour"] = {k: {c: round(float(v), 4) for c, v in r.items()} for k, r in prof.iterrows()}
    _store("hourly", res)
    print(json.dumps(res, indent=1, default=str))


# ============================================================================ report
def _fmt(s: dict, pre="clv_all") -> str:
    if not s or not s.get("bets"):
        return "0 bets"
    parts = [f"n={s['bets']}"]
    for c in ("all", "sharp", "nfl"):
        if f"clv_{c}" in s:
            parts.append(f"CLV[{c}] {s[f'clv_{c}']:+.2%} (p={s[f'clv_{c}_p']:.3f})")
    parts.append(f"ROI {s['roi']:+.1%}±{(s.get('roi_se') or 0):.1%}")
    return ", ".join(parts)


def report():
    J = json.loads(JSON.read_text())
    L = ["# Openers and check frequency (soft-vs-sharp line shopping)", "",
         "Generated by `scripts/research/openers_hourly.py`. CLV[all] = vs our last pre-kickoff snapshot's all-book "
         "no-vig price (edge_lab.closing_fair; spreads priced with key numbers), CLV[sharp] = sharp books only at "
         "that snapshot, CLV[nfl] = nflverse close. One bet per game at the first qualifying snapshot, flat 1 unit.", ""]
    for k in ("dev", "holdout"):
        if k not in J:
            continue
        d = J[k]
        L += [f"## A. Openers - {'development 2020-2022' if k == 'dev' else 'holdout 2023-2025 (run once)'}", ""]
        if k == "holdout":
            L += ["### Frozen rules (the test)", "", "| rule | result | pass |", "|---|---|---|"]
            for r, s in d["results"].items():
                L.append(f"| {r} | {_fmt(s)} | {'PASS' if s['pass'] else 'fail'} |")
            L += [""]
            for r, s in d["results"].items():
                L.append(f"- {r} by season: " + "; ".join(f"{y}: {_fmt(v)}" for y, v in s["by_season"].items()))
            L += [""]
        L += ["### Soft-vs-sharp gap by timing (region-us books only everywhere)", "",
              "| timing | snapshots | books/snap | ML best EV vs sharp (mean) | ML snaps >=2% | ML dog snaps >=3% | spread best EV (mean) | spread snaps >=3% |",
              "|---|---|---|---|---|---|---|---|"]
        for b, v in d["gaps_by_bucket"].items():
            L.append(f"| {b} | {v['snapshots']} | {v['books_per_snapshot']} | {v['ml_best_ev_vs_sharp_mean']:+.2%} | "
                     f"{v['ml_share_snap_ge2pct']:.1%} | {v['ml_dog_share_snap_ge3pct']:.1%} | {v['sp_best_ev_vs_sharp_mean']:+.2%} | "
                     f"{v['sp_share_snap_ge3pct']:.1%} |")
        m = d["movement_opener_to_close"]
        L += ["", "### Opener -> close movement", "", "```", json.dumps(m, indent=1), "```", "",
              "### Does the model predict the move from each snapshot to the close? (slope of close-move on model-minus-market)", "",
              "| timing | games | mean abs model-market | slope | t | slope clean rows (no QB news, prior games final) | t | avg move toward model when gap>=5% |",
              "|---|---|---|---|---|---|---|---|"]
        for b, v in d["model_vs_market_by_bucket"].items():
            L.append(f"| {b} | {v['games']} | {v['mean_abs_model_minus_market']:.3f} | {v['slope_close_move_on_model_gap']:+.3f} | "
                     f"{v['slope_t'] if v['slope_t'] is None else format(v['slope_t'], '+.2f')} | "
                     f"{v['slope_clean_rows']:+.3f} (n={v['clean_rows']}) | {v['slope_t_clean']} | "
                     f"{v['avg_move_toward_model_when_gap_ge5pct']} |")
        rules = d.get("rules") or d.get("all_rules_descriptive_not_a_test")
        L += ["", "### Rules: bets at the opener vs the same rule at later snapshots" +
              (" (descriptive, not a test)" if k == "holdout" else ""), "",
              "| rule | model? | at opener | at later snapshots | opener bets on games later never qualify |", "|---|---|---|---|---|"]
        for r, v in rules.items():
            L.append(f"| {r} | {'yes (look-ahead at opener)' if v['uses_model'] else 'no'} | {_fmt(v['opener'])} | "
                     f"{_fmt(v['later_main_snapshots'])} | {_fmt(v['opener_bets_on_games_later_never_qualify'])} |")
        L += [""]
    if FROZEN.exists():
        fz = json.loads(FROZEN.read_text())
        L += ["## Frozen opener rules", "", f"`{FROZEN.relative_to(ROOT)}`: {', '.join(fz['rules'])} - {fz['why']}", ""]
    if "hourly" in J:
        h = J["hourly"]
        L += ["## B. Hourly checks Sat 15:00 -> Sun 16:00 UTC (2025 wk 3-12 Sunday games; descriptive only)", "",
              f"{h['games']} games, {h['weeks']} weekends, {h['hourly_snapshots']} hourly snapshots. Existing run times inside "
              f"the window: {', '.join(h['existing_run_times_inside_window'])} "
              f"({h['existing_snapshots_inside_window_per_game']} per game on average).", "",
              "| rule | existing schedule (full week) | existing + hourly | extra games only w/ hourly | extra/week | "
              "opp. runs (game-side) | median hours | share 1h only |", "|---|---|---|---|---|---|---|---|"]
        for r, v in h["rules"].items():
            p = v["persistence_hours_game_side"]
            L.append(f"| {r} | {_fmt(v['bets_existing_schedule_full_week'])} | {_fmt(v['bets_existing_plus_hourly'])} | "
                     f"{_fmt(v['extra_games_only_with_hourly'])} | {v['extra_bets_per_week']} | {v['hourly_opportunities_game_side']} | "
                     f"{p['median']} | {p['share_1h_only']:.0%} |")
        L += ["", "Inside the window only:", ""]
        for r, v in h["rules"].items():
            L.append(f"- {r}: existing runs {_fmt(v['inside_window_existing_runs_only'])}; hourly {_fmt(v['inside_window_hourly'])}; "
                     f"every hourly opportunity start {_fmt(v['all_hourly_opportunity_starts_clv'])}; "
                     f"{v['hourly_opps_not_visible_at_existing_runs_in_window']} of {v['hourly_opportunities_game_side']} never "
                     f"visible at an existing run in the window")
        L += ["", "Best soft ML price vs sharp no-vig, by hour (mean, share of games >= 2%):", "",
              "| hour (UTC) | mean | >=2% | n |", "|---|---|---|---|"]
        for hh, v in h["gap_by_hour"].items():
            L.append(f"| {hh} | {v['mean_best_ev_vs_sharp']:+.2%} | {v['share_ge2pct']:.1%} | {int(v['n'])} |")
        L += [""]
    if "conclusions" in J:
        L += ["## Conclusions", ""] + [f"- {c}" for c in J["conclusions"]]
    MD.write_text("\n".join(L) + "\n")
    print(MD)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "dev":
        run_dev()
    elif cmd == "freeze":
        freeze(sys.argv[2].split(","), sys.argv[3])
    elif cmd == "holdout":
        run_holdout()
    elif cmd == "hourly":
        run_hourly()
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
