"""GRADE v2 for SPREAD offers: the predicted honest closing-line value (CLV) of a spread offer, learned from data.

Spread analogue of scripts/research/grade_v2.py (moneyline). Stages (outputs output/research/grade_v2_spread.{json,md},
grade_v2_spread_frozen.json):

  python scripts/research/grade_v2_spread.py dev        # 2020-22 only: offer table, season-grouped CV, model choice,
                                                        # band scan, dev results by grade / strategies / grade v1
  python scripts/research/grade_v2_spread.py freeze     # final fit on 2020-22 -> grade_v2_spread_frozen.json
                                                        # (refuses overwrite): model, features, bands, pass bar
  GRADE2SP_HOLDOUT=I_HAVE_FROZEN python scripts/research/grade_v2_spread.py holdout   # ONCE on 2023-25
  python scripts/research/grade_v2_spread.py report     # writes the .md

Candidate bets: one row per (game, snapshot <= 7 days and >= 10 min before kick, allowed book, side) MAIN-LINE spread
offer at the historical snapshot times (= our live run times: daily 14:10 UTC, Fri 21:40, ~75 min pre-kick), using
the research filters of ml_spread_consistency.build_ml (price -145..+125 both sides, half-point numbers, overround
-1%..12%, number within 2.5 of the snapshot median), EV >= 0 vs at least one fair reference (sharp spread-implied
margin, all-book spread-implied margin, sharp ML-implied margin, walk-forward model blend at eligible snapshots).

ONE margin model everywhere: the total-aware key-number distribution src/nflpred/margin_total.py with
margin_total_model.json (fit 2012-2019, production model of ml v4). Every expected margin (sharp / consensus spread +
juice, sharp / consensus ML, model blend, close) is turned into P(cover), P(push) of OUR point under that model at the
snapshot's (or closing) consensus total; EV = P(cover) x decimal + P(push) - 1.

Target: honest spread CLV = EV of our point + price under the PRICE-IMPLIED closing expected margin: median over the
sharp books (LowVig / BetOnline; fallback all books) of the total-aware mu implied by each book's closing spread AND
juice, at the last pre-kick snapshot (~75 min), valued at the closing total. Never the nflverse spread_line alone.
Robustness: the same bets valued at edge_lab.closing_fair (mu_close_sharp -> mu_close_all; old single-weight
key-number model of spread_rules.json, = edge_lab.spread_clv_price), and points CLV vs the closing consensus number.

Model selection (stated before looking, as grade v2): leave-one-season-out (2020/2021/2022) game-weighted MSE of CLV;
ridge kept unless a GBM beats the best ridge by > 0.5% relative MSE.
Env GRADE2SP_CACHE=<dir> caches tables (also passed to build_ml as MLSP_CACHE).
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
CACHE = os.environ.get("GRADE2SP_CACHE")
if CACHE and not os.environ.get("MLSP_CACHE"):
    os.environ["MLSP_CACHE"] = CACHE
import ml_spread_consistency as MS  # noqa: E402
from nflpred import grading as G1  # noqa: E402
from nflpred import margin_total as MTP  # noqa: E402
from nflpred import margins as K  # noqa: E402
from nflpred import odds as O  # noqa: E402
from nflpred import spread_bets as SB  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "grade_v2_spread.json"
FROZEN = OUT / "grade_v2_spread_frozen.json"
MD = OUT / "grade_v2_spread.md"
DEV = (2020, 2021, 2022)
HOLD = (2023, 2024, 2025)
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet"]
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
EV_CLIP = (-0.15, 0.25)
r4 = MS.r4
KEYS = (-7.0, -3.0, 3.0, 7.0)

FEATURES = {
    "ev_sp_sharp": "EV of our point+price under the sharp (LowVig/BetOnline median) spread+juice implied margin, "
                   "total-aware model at the snapshot consensus total; fallback ev_sp_cons; clipped",
    "ev_sp_cons": "EV under the all-book median spread+juice implied margin (total-aware); clipped",
    "ev_mlimp": "EV under the margin implied by the sharp no-vig MONEYLINE (inverted through the total-aware model); "
                "fallback all-book ML; else ev_sp_cons; clipped",
    "ml_sp_gap": "our side's sign x (sharp ML-implied margin - sharp spread-implied margin), points (fallback "
                 "consensus); + = moneyline rates our side higher than the spread does; clipped +-4",
    "ev_model": "EV under the walk-forward model blend (spread_rules weights: model mu_model + consensus margin) at "
                "eligible snapshots (injury report out, QBs confirmed), else 0; clipped",
    "model_elig": "1 if the snapshot is model-eligible",
    "sharp_missing": "1 if no LowVig/BetOnline spread at the snapshot",
    "pt_off_cons": "our point minus the all-book median point for our side (+ = we get more points); clipped +-2.5",
    "pt_off_sharp": "our point minus the sharp median point for our side (fallback consensus); clipped +-2.5",
    "key_pos": "grading.key_number_side(our point): +1 right / -1 wrong side of 3/7",
    "key_pos_cons": "grading.key_number_side(consensus point for our side, to 0.5)",
    "on3": "1 if |our point| in 2.5..3.5",
    "on7": "1 if |our point| in 6.5..7.5",
    "key_cross": "signed count of key margins (+-3, +-7) between the consensus number and ours (+ = ours better)",
    "p_imp": "1/decimal price of the offer (juice)",
    "overround": "the offering book's two-way spread overround",
    "log_hours": "log(1 + hours before kickoff)",
    "is_last": "1 at the game's last snapshot (~75 min pre-kick)",
    "move_pts": "consensus number move since first sight (<= 9 days), points toward our side (edge_lab m_first)",
    "move_mu": "consensus price-implied margin move since the first snapshot <= 7 days, toward our side",
    "disp": "std across books of the price-implied margin at the snapshot (points); clipped 0..3",
    "best_gap": "best allowed-book EV-vs-consensus for this side at the snapshot minus this offer's (0 = best)",
    "is_dog": "1 if our point > 0",
    "abs_pt": "|consensus spread|",
    "week": "NFL week",
    **{f"bk_{b}": f"1 if book == {b}" for b in BOOKS[1:]},
}
FEATS = list(FEATURES)
MONO = {"ev_sp_sharp": 1, "ev_sp_cons": 1, "ev_mlimp": 1}

_M = None


def MT():
    global _M
    if _M is None:
        _M = MTP.load()          # margin_total_model.json (== ml_spread_consistency part1b total_aware_2012)
    return _M


# ------------------------------------------------------------------------------------------------ pricing helpers
def ev_spread(mu_home, side_home, point, dec, total):
    """EV per unit of a spread bet (our side's point, decimal price) under expected home margin mu_home."""
    mu_home = np.asarray(mu_home, float)
    out = np.full(len(mu_home), np.nan)
    ok = np.isfinite(mu_home) & np.isfinite(np.asarray(point, float))
    if ok.any():
        tot = np.nan_to_num(np.asarray(total, float), nan=44.0)
        pw, pp = MT().side_probs(mu_home[ok], np.asarray(side_home)[ok], np.asarray(point, float)[ok], tot[ok])
        out[ok] = pw * np.asarray(dec, float)[ok] + pp - 1
    return out


def mu_from_pwin(p_home, total):
    """Expected home margin whose P(home win | no tie) equals the no-vig ML probability (total-aware model)."""
    M = MT()
    p_home = np.asarray(p_home, float)
    tr = np.round(np.nan_to_num(np.asarray(total, float), nan=44.0) * 2) / 2
    out = np.full(len(p_home), np.nan)
    G = MTP.GRID
    for t in np.unique(tr):
        s = (tr == t) & np.isfinite(p_home)
        if not s.any():
            continue
        pw, pt = M.win_tie(G, np.full(len(G), t))
        q = np.maximum.accumulate(pw / np.maximum(1 - pt, 1e-12))
        out[s] = np.interp(p_home[s], q, G)
    return out


def ev_old(mu_home, side_home, point, dec):
    """edge_lab / spread_bets valuation (single-weight key-number model of spread_rules.json)."""
    r = SB.load_rules()
    mu_home = np.asarray(mu_home, float)
    mu_side = np.where(side_home, mu_home, -mu_home)
    hc, pu, _ = K.cover_probs(np.nan_to_num(mu_side), r["margin"]["sigma"], np.nan_to_num(np.asarray(point, float)),
                              r["_weights"])
    return np.where(np.isfinite(mu_home), hc * dec + pu - 1, np.nan)


def key_cross(x, c):
    """signed count of key margins between consensus side point c and our side point x (+ = ours better)."""
    x, c = np.asarray(x, float), np.asarray(c, float)
    lo, hi = np.minimum(-x, -c), np.maximum(-x, -c)
    n = np.zeros(len(x))
    for k in KEYS:
        n += ((k >= lo) & (k <= hi) & (np.abs(x - c) > 1e-9)).astype(float)
    return np.sign(x - c) * n


# ------------------------------------------------------------------------------------------------ table
def edge_lab_snaps(seasons) -> pd.DataFrame:
    p = ROOT / "data" / ("edge_lab_dev.parquet" if set(seasons) <= set(DEV) else "edge_lab_holdout.parquet")
    e = pd.read_parquet(p, columns=["game_id", "requested_ts", "season", "m_cons", "m_first", "eligible"])
    e = e[e.season.isin(seasons)].drop_duplicates(["game_id", "requested_ts"]).drop(columns="season")
    return e.rename(columns={"m_cons": "el_m_cons"})


def table(seasons) -> pd.DataFrame:
    seasons = tuple(seasons)
    if set(seasons) & set(HOLD) and os.environ.get("GRADE2SP_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked")
    tag = f"spoffers_{min(seasons)}_{max(seasons)}.parquet"
    if CACHE and (Path(CACHE) / tag).exists():
        return pd.read_parquet(Path(CACHE) / tag)
    rows, snaps = MS.build_ml(seasons, MS.get_model("chosen"))
    k2 = ["game_id", "requested_ts"]
    R = rows[rows.sp_ok].copy()
    ex = R.groupby(k2).agg(disp=("mu_sp", "std"), n_sp=("book", "nunique"))
    ex = ex.join(R[R.book.isin(SHARP)].groupby(k2).agg(pt_sharp=("sp_home_point", "median")))
    S = snaps.merge(ex.reset_index(), on=k2, how="left")
    S["mu_ml_sharp"] = mu_from_pwin(S.p_ml_sharp.values, S.tot.values)
    S["mu_ml_cons"] = mu_from_pwin(S.p_ml_cons.values, S.tot.values)
    # closing fair (last snapshot in our files, ~75 min pre-kick): sharp price-implied margin, fallback all books
    S = S.sort_values("requested_ts")
    last = S.groupby("game_id").tail(1)
    cl = last[["game_id", "mu_sharp", "mu_cons", "tot", "pt_cons", "requested_ts"]].rename(columns={
        "mu_sharp": "mu_close_sharp_ta", "mu_cons": "mu_close_cons_ta", "tot": "tot_close", "pt_cons": "pt_close",
        "requested_ts": "close_ts"})
    cl["mu_close"] = cl.mu_close_sharp_ta.fillna(cl.mu_close_cons_ta)
    S = S.merge(cl, on="game_id", how="left")
    first = S.groupby("game_id").mu_cons.first().rename("mu_first")
    S = S.merge(first.reset_index(), on="game_id", how="left")
    S["is_last"] = S.requested_ts == S.close_ts
    import edge_lab as EL
    cf = EL.closing_fair(list(seasons))[["game_id", "mu_close_sharp", "mu_close_all"]]
    S = S.merge(cf, on="game_id", how="left")
    S["mu_close_old"] = S.mu_close_sharp.fillna(S.mu_close_all)
    rp = pd.read_parquet(ROOT / "data" / "replay_predictions.parquet", columns=["game_id", "mu_model"])
    S = S.merge(rp, on="game_id", how="left")
    S = S.merge(edge_lab_snaps(seasons), on=k2, how="left")
    allowed = O.load_allowed_books()
    B = R[R.book.isin(allowed)][k2 + ["book", "sp_home_point", "sp_home_price", "sp_away_price", "sp_overround",
                                       "mu_sp"]].merge(S, on=k2)
    out = []
    for side in ("home", "away"):
        x = B.copy()
        h = side == "home"
        x["side"] = side
        x["point"] = x.sp_home_point if h else -x.sp_home_point
        x["price"] = x.sp_home_price if h else x.sp_away_price
        out.append(x)
    X = pd.concat(out, ignore_index=True)
    sh = (X.side == "home").values
    sg = np.where(sh, 1.0, -1.0)
    X["dec"] = MTP.american_to_dec(X.price.values)
    pt, dec, tot = X.point.values, X.dec.values, X.tot.values
    X["ev_sharp_raw"] = ev_spread(X.mu_sharp.values, sh, pt, dec, tot)
    X["ev_cons_raw"] = ev_spread(X.mu_cons.values, sh, pt, dec, tot)
    mu_ml = X.mu_ml_sharp.fillna(X.mu_ml_cons).values
    X["ev_mlimp_raw"] = ev_spread(mu_ml, sh, pt, dec, tot)
    X["ml_sp_gap_raw"] = sg * (mu_ml - X.mu_sharp.fillna(X.mu_cons).values)
    r = SB.load_rules()
    X["mu_blend"] = SB.expected_margin(X.mu_model, -X.pt_cons, r)
    X["eligible"] = X.eligible.fillna(False).astype(bool)
    evm = ev_spread(X.mu_blend.values, sh, pt, dec, tot)
    X["ev_model_raw"] = np.where(X.eligible & X.mu_model.notna(), evm, np.nan)
    X["v1_edge"] = ev_old(X.mu_blend.values, sh, pt, dec)          # production grade v1 / spread track edge
    # targets and outcomes
    X["clv"] = ev_spread(X.mu_close.values, sh, pt, dec, X.tot_close.values)
    X["clv_old"] = ev_old(X.mu_close_old.values, sh, pt, dec)
    X["clv_pts"] = pt - sg * X.pt_close.values                       # our point - closing consensus side point
    X["side_margin"] = sg * X.m
    adj = X.side_margin + X.point
    X["pnl"] = np.where(adj > 0, X.dec - 1, np.where(adj < 0, -1.0, 0.0))
    X["cons_pt_side"] = sg * X.pt_cons
    X["sharp_pt_side"] = sg * X.pt_sharp
    X["move_pts"] = sg * (X.el_m_cons - X.m_first)
    X["move_mu"] = sg * (X.mu_cons - X.mu_first)
    best = X.groupby(k2 + ["side"]).ev_cons_raw.transform("max")
    X["best_gap"] = best - X.ev_cons_raw
    X["hours_before"] = (X.kick - X.requested_ts).dt.total_seconds() / 3600
    keep = [c for c in X.columns if c not in ("sp_home_point", "sp_home_price", "sp_away_price")]
    X = X[keep]
    if CACHE:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        X.to_parquet(Path(CACHE) / tag)
    return X


def featurize(X: pd.DataFrame) -> pd.DataFrame:
    c = lambda s: np.clip(s, *EV_CLIP)  # noqa: E731
    F = pd.DataFrame(index=X.index)
    F["ev_sp_sharp"] = c(X.ev_sharp_raw.fillna(X.ev_cons_raw).fillna(0.0))
    F["ev_sp_cons"] = c(X.ev_cons_raw.fillna(0.0))
    F["ev_mlimp"] = c(X.ev_mlimp_raw.fillna(X.ev_cons_raw).fillna(0.0))
    F["ml_sp_gap"] = np.clip(X.ml_sp_gap_raw.fillna(0.0), -4, 4)
    F["ev_model"] = c(X.ev_model_raw.fillna(0.0))
    F["model_elig"] = X.ev_model_raw.notna().astype(float)
    F["sharp_missing"] = X.mu_sharp.isna().astype(float)
    F["pt_off_cons"] = np.clip((X.point - X.cons_pt_side).fillna(0.0), -2.5, 2.5)
    F["pt_off_sharp"] = np.clip((X.point - X.sharp_pt_side.fillna(X.cons_pt_side)).fillna(0.0), -2.5, 2.5)
    F["key_pos"] = [float(G1.key_number_side(float(p))) for p in X.point.values]
    cp = (np.round(X.cons_pt_side.fillna(0) * 2) / 2).values
    F["key_pos_cons"] = [float(G1.key_number_side(float(p))) for p in cp]
    ap = X.point.abs()
    F["on3"] = ((ap >= 2.5) & (ap <= 3.5)).astype(float)
    F["on7"] = ((ap >= 6.5) & (ap <= 7.5)).astype(float)
    F["key_cross"] = key_cross(X.point.values, X.cons_pt_side.fillna(X.point).values)
    F["p_imp"] = 1 / X.dec
    F["overround"] = X.sp_overround.clip(-0.01, 0.12)
    F["log_hours"] = np.log1p(X.hours_before.clip(lower=0))
    F["is_last"] = X.is_last.astype(float)
    F["move_pts"] = np.clip(X.move_pts.fillna(0.0), -7, 7)
    F["move_mu"] = np.clip(X.move_mu.fillna(0.0), -7, 7)
    F["disp"] = X.disp.fillna(0.3).clip(0, 3)
    F["best_gap"] = X.best_gap.fillna(0.0).clip(0, 0.1)
    F["is_dog"] = (X.point > 0).astype(float)
    F["abs_pt"] = X.pt_cons.abs().fillna(3.0)
    F["week"] = X.week.astype(float)
    for b in BOOKS[1:]:
        F[f"bk_{b}"] = (X.book == b).astype(float)
    return F[FEATS].astype(float)


def candidates(X: pd.DataFrame) -> pd.DataFrame:
    ok = X.clv.notna() & X.price.between(-145, 125)
    anypos = ((X.ev_sharp_raw >= 0) | (X.ev_cons_raw >= 0) | (X.ev_mlimp_raw >= 0) | (X.ev_model_raw >= 0))
    C = X[ok & anypos].reset_index(drop=True)
    C["w"] = 1.0 / C.groupby("game_id").game_id.transform("size")
    return C


# ------------------------------------------------------------------------------------------------ models
# Feature sets (dev decision after the first LOSO run showed the full set overfits: the no-fit EV baseline beat
# every full-set model). Complexity tiers for the choice rule below.
FSETS = {
    "core": ["ev_sp_sharp", "ev_mlimp"],
    "market": ["ev_sp_sharp", "ev_mlimp", "ev_sp_cons", "ev_model", "model_elig"],
    "small": ["ev_sp_sharp", "ev_sp_cons", "ev_mlimp", "pt_off_cons", "on3", "key_pos", "best_gap", "move_mu",
              "log_hours", "is_dog", "disp"],
    "all": FEATS,
}


class Ridge:
    kind = "ridge"

    def __init__(self, alpha, cols=None):
        self.alpha = alpha
        self.cols = list(cols or FEATS)

    def fit(self, F, y, w):
        F = F[self.cols]
        self.mu = F.mean().values
        self.sd = F.std().replace(0, 1).values
        Z = (F.values - self.mu) / self.sd
        W = w / w.sum()
        ym = float((W * y).sum())
        Zm = (W[:, None] * Z).sum(0)
        Zc, yc = Z - Zm, y - ym
        A = (Zc * W[:, None]).T @ Zc + self.alpha * np.eye(Z.shape[1]) / len(y)
        self.coef = np.linalg.solve(A, (Zc * W[:, None]).T @ yc)
        self.b0 = ym - Zm @ self.coef
        return self

    def predict(self, F):
        return self.b0 + ((F[self.cols].values - self.mu) / self.sd) @ self.coef

    def to_json(self):
        return {"kind": "ridge", "alpha": self.alpha, "intercept": float(self.b0), "feature_order": self.cols,
                "features": {f: {"mean": float(m), "sd": float(s), "coef_std": float(c), "coef_raw": float(c / s)}
                             for f, m, s, c in zip(self.cols, self.mu, self.sd, self.coef)},
                "formula": "pred_clv = intercept + sum_f coef_std[f] * (x_f - mean[f]) / sd[f]"}


class Base:
    """No fit: predicted CLV = EV vs the sharp spread-implied margin."""
    kind = "ev_sp_sharp"
    cols = ["ev_sp_sharp"]

    def fit(self, F, y, w):
        return self

    def predict(self, F):
        return F.ev_sp_sharp.values


class GBM:
    kind = "lgbm"

    def __init__(self, cols=None, **kw):
        self.kw = kw
        self.cols = list(cols or FEATS)

    def fit(self, F, y, w):
        import lightgbm as lgb
        mono = [MONO.get(f, 0) for f in self.cols]
        p = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=400, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, monotone_constraints=mono, verbose=-1, seed=7,
                 num_threads=4)
        p.update(self.kw)
        n = p.pop("n_trees", 300)
        self.m = lgb.train(p, lgb.Dataset(F[self.cols].values, y, weight=w, feature_name=self.cols), n)
        return self

    def predict(self, F):
        return self.m.predict(F[self.cols].values)


TIERS = {"base": 0, "ridge_core": 1, "ridge_market": 2, "ridge_small": 3, "ridge_all": 4, "lgbm_small": 5,
         "lgbm_all": 6}


def tier(name):
    return next(v for k, v in TIERS.items() if name.startswith(k))


def model_grid():
    g = [("base_ev_sp_sharp", lambda: Base())]
    for fs in ("core", "market", "small", "all"):
        for a in (1.0, 300.0, 3000.0):
            g.append((f"ridge_{fs}_a{a:g}", lambda a=a, fs=fs: Ridge(a, FSETS[fs])))
    for fs in ("small", "all"):
        for nt, lv in ((100, 4), (200, 4)):
            g.append((f"lgbm_{fs}_{nt}x{lv}", lambda nt=nt, lv=lv, fs=fs: GBM(FSETS[fs], n_trees=nt, num_leaves=lv)))
    return g


def wmse(y, p, w):
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def loso(C, F):
    y, w = C.clv.values, C.w.values
    res, oof = {}, {}
    var0 = wmse(y, np.full(len(y), np.average(y, weights=w)), w)
    for name, mk in model_grid():
        pred = np.full(len(C), np.nan)
        for s in DEV:
            te = (C.season == s).values
            m = mk().fit(F[~te], y[~te], w[~te])
            pred[te] = m.predict(F[te])
        oof[name] = pred
        res[name] = {"wmse": wmse(y, pred, w), "r2_vs_const": r4(1 - wmse(y, pred, w) / var0)}
        print(name, res[name], flush=True)
    return res, oof


# ------------------------------------------------------------------------------------------------ grades / bets
ORDER = ["A+", "A", "B", "C"]


def letter(pred, th):
    return np.where(pred >= th["A+"], "A+", np.where(pred >= th["A"], "A", np.where(pred >= th["B"], "B", "C")))


def best_offers(C, pred, th):
    """Best predicted-CLV offer per (game, snapshot) -- what the dashboard would show as that game's spread bet."""
    D = C.assign(pred=pred)
    D = D.sort_values("pred", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    D["grade"] = letter(D.pred.values, th)
    return D.sort_values(["game_id", "requested_ts"])


def first_with(D, mask):
    return D[mask].sort_values("requested_ts").drop_duplicates("game_id")


def boot_se(x, n=2000, seed=11):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    return float(np.std([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], ddof=1))


def summ(B, nseas):
    if B is None or len(B) == 0:
        return {"bets": 0}
    clv, pnl = B.clv.values, B.pnl.values
    n = len(B)
    cm = float(clv.mean())
    cse = float(clv.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    t = cm / cse if cse else None
    co = B.clv_old.values
    co = co[~np.isnan(co)]
    return {"bets": int(n), "per_season": round(n / nseas, 1),
            "pred_clv": r4(B.pred.mean()) if "pred" in B and B.pred.notna().any() else None,
            "clv": r4(cm), "clv_se": r4(cse), "clv_t": r4(t),
            "clv_p_one_sided": r4(0.5 * math.erfc(t / math.sqrt(2))) if t is not None else None,
            "beat_close": r4((clv > 0).mean()), "roi": r4(pnl.mean()),
            "roi_se": r4(pnl.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
            "cover_rate": r4((pnl > 0).sum() / max((pnl != 0).sum(), 1)),
            "clv_old": r4(co.mean()) if len(co) else None,
            "clv_old_se": r4(co.std(ddof=1) / math.sqrt(len(co))) if len(co) > 1 else None,
            "clv_pts": r4(np.nanmean(B.clv_pts)), "dog_share": r4((B.point > 0).mean()),
            "avg_price": int(np.median(B.price))}


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return None, None
    r = spearmanr(x[ok], y[ok])
    return r4(r.statistic), r4(r.pvalue)


def top_minus_bottom(Btop, Bbot, n=2000, seed=3):
    """mean CLV(top) - mean CLV(bottom), SE by game-cluster bootstrap over the union of games."""
    if len(Btop) < 3 or len(Bbot) < 3:
        return {"diff": None}
    games = np.union1d(Btop.game_id.unique(), Bbot.game_id.unique())
    gi = {g: i for i, g in enumerate(games)}
    ti = Btop.game_id.map(gi).values
    bi = Bbot.game_id.map(gi).values
    tc, bc = Btop.clv.values, Bbot.clv.values
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
        wt, wb = cnt[ti], cnt[bi]
        if wt.sum() == 0 or wb.sum() == 0:
            continue
        ds.append(np.sum(wt * tc) / wt.sum() - np.sum(wb * bc) / wb.sum())
    d = float(tc.mean() - bc.mean())
    se = float(np.std(ds, ddof=1))
    return {"diff": r4(d), "se_game_boot": r4(se), "z": r4(d / se) if se else None}


def evaluate(C, pred, th, strategies, nseas) -> dict:
    D = best_offers(C, pred, th)
    out = {"games": int(C.game_id.nunique()), "candidate_rows": int(len(C)), "snapshot_best_offers": int(len(D))}
    bg, bets = {}, {}
    for g in ORDER:
        bets[g] = first_with(D, D.grade == g)
        bg[g] = summ(bets[g], nseas)
    out["by_grade_first_appearance"] = bg
    rho, p = spearman([4, 3, 2, 1], [bg[g].get("clv") if bg[g]["bets"] else np.nan for g in ORDER])
    out["letters_spearman"] = {"rho": rho, "p": p}
    out["top_minus_bottom"] = top_minus_bottom(bets["A+"], bets["C"])
    allb = pd.concat([bets[g] for g in ORDER])
    out["bet_level_spearman_pred_vs_clv"] = dict(zip(("rho", "p"), spearman(allb.pred, allb.clv)))
    sl = {}
    for g in ORDER:
        e = D[D.grade == g]
        if len(e):
            gm = e.groupby("game_id").clv.mean()
            sl[g] = {"offers": int(len(e)), "games": int(len(gm)), "clv": r4(e.clv.mean()),
                     "clv_se_game_boot": r4(boot_se(gm.values)), "pred": r4(e.pred.mean())}
    out["by_grade_snapshot_best_offers"] = sl
    D["dec_bin"] = pd.qcut(D.pred.rank(method="first"), 10, labels=False)
    dd = D.groupby("dec_bin").agg(pred=("pred", "mean"), clv=("clv", "mean"), n=("clv", "size")).reset_index()
    out["deciles"] = [{k: (r4(v) if isinstance(v, float) else int(v)) for k, v in r.items()} for r in dd.to_dict("records")]
    rho, p = spearman(dd.pred, dd.clv)
    out["deciles_spearman"] = {"rho": rho, "p": p}
    from scipy.stats import spearmanr
    out["offer_spearman_pred_vs_clv"] = r4(spearmanr(pred, C.clv.values).statistic)
    W = C.w.values
    pc = pred - np.average(pred, weights=W)
    out["calibration_slope"] = r4(float(np.sum(W * pc * (C.clv.values - np.average(C.clv.values, weights=W))) /
                                        np.sum(W * pc ** 2)))
    st = {}
    for s in strategies:
        B = first_with(D, D.pred >= th[s["min_grade"]])
        st[s["id"]] = summ(B, nseas)
        st[s["id"]]["by_season"] = {int(k): summ(d, 1).get("clv") for k, d in B.groupby("season")}
    out["strategies"] = st
    return out, bets


# ------------------------------------------------------------------------------------------------ grade v1
V1_ORDER = ["A+", "A", "B+", "B", "C+", "C"]


def v1_score(X):
    """grading.grade('spread', ...) vectorized, as spread_bets.evaluate calls it: edge = EV under the blend margin
    (old model); disagreement = mu_model - consensus margin; moved = number move toward our side since first seen;
    key number of our point. qb_flag = False (the historical flag uses the ACTUAL starter = leakage)."""
    e = X.v1_edge.values
    pts = np.where(e >= 0.05, 2, np.where(e >= 0.03, 1, 0))
    dis = (X.mu_model - (-X.pt_cons)).abs().values
    pts = pts - (dis >= 7).astype(int)
    mv = X.move_pts.values
    pts = pts + np.where(np.isnan(mv), 0, np.where(mv >= 0.5, 1, np.where(mv <= -0.5, -1, 0)))
    pts = pts + np.array([G1.key_number_side(float(p)) for p in X.point.values])
    return pts


def v1_letter(score):
    return np.array([G1.letter(int(s)) for s in score])


def v1_eval(X_all, C, nseas, bets_v2) -> dict:
    """(a) grade v1 as production uses it: each snapshot's best-EV (blend) allowed-book offer gets the v1 letter;
    one bet per (game, letter) at its first appearance. (b) the grade v2 by-grade bets re-labelled with v1."""
    X = X_all[X_all.clv.notna() & X_all.price.between(-145, 125)].copy()
    X = X.sort_values("v1_edge", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    X["v1"] = v1_letter(v1_score(X))
    X["pred"] = np.nan
    a = {g: summ(first_with(X, X.v1 == g), nseas) for g in V1_ORDER}
    rho, p = spearman(list(range(6, 0, -1)), [a[g].get("clv") if a[g]["bets"] else np.nan for g in V1_ORDER])
    out = {"production_v1_by_letter": a, "v1_letters_spearman": {"rho": rho, "p": p},
           "v1_top_minus_bottom": top_minus_bottom(first_with(X, X.v1 == "A+"), first_with(X, X.v1 == "C"))}
    B = pd.concat([b.assign(v2=g) for g, b in bets_v2.items()], ignore_index=True)
    B["v1_score"] = v1_score(B)
    B["v1"] = v1_letter(B.v1_score)
    out["same_bets_n"] = int(len(B))
    out["same_bets_spearman_v1score_vs_clv"] = dict(zip(("rho", "p"), spearman(B.v1_score, B.clv)))
    out["same_bets_spearman_v2pred_vs_clv"] = dict(zip(("rho", "p"), spearman(B.pred, B.clv)))
    out["same_bets_by_v1"] = {g: summ(d, nseas) for g, d in B.groupby("v1")}
    out["same_bets_crosstab"] = {f"{a_}|{b_}": int(n) for (a_, b_), n in B.groupby(["v2", "v1"]).size().items()}
    # also on all candidate offers (offer-level rank correlation with CLV)
    out["offer_spearman_v1score_vs_clv"] = spearman(v1_score(C), C.clv.values)[0]
    return out


def spread_rule_v1(X_all, nseas) -> dict:
    """The live spread track rule (spread_rules.json v1) on this table: best blend-EV offer >= 3%, -200..+200,
    eligible snapshot, number not moved >= 1.5 against; first qualifying snapshot."""
    X = X_all[X_all.clv.notna()]
    X = X.sort_values("v1_edge", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    q = X[(X.v1_edge >= 0.03) & X.price.between(-200, 200) & X.eligible & ~(X.move_pts <= -1.5)]
    q = q.sort_values("requested_ts").drop_duplicates("game_id")
    return summ(q.assign(pred=np.nan), nseas)


# ------------------------------------------------------------------------------------------------ stages
def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=MS._jd))


# Chosen on dev (2020-22) only -- see run_dev output; set here BEFORE freeze, never after.
# Chosen model ridge_core (pred CLV is calibrated on dev: slope ~0.96). Dev OOF scan (first bet per game with pred >= cut):
# 2.5% -> 7/season (too few to test, the moneyline A+ cut does not transfer: spread pred CLV is compressed);
# 1.0% -> 28/season, CLV +2.6% +- 1.0 (~ the moneyline A+ volume); 0.0% -> 79/season, +0.7% +- 0.6; -1% -> 169/season,
# -0.4%. Bands: A+ = pred >= +1% (the bet letter), A = predicted +CLV (0..1%), B = within 1% of break-even, C below.
THRESHOLDS = {"A+": 0.010, "A": 0.0, "B": -0.010}
STRATEGIES = [
    {"id": "S1_Aplus_flat", "min_grade": "A+",
     "desc": "One bet per game at the first snapshot where the best spread offer is A+; 1 unit."},
    {"id": "S2_A_and_up_flat", "min_grade": "A",
     "desc": "One bet per game at the first snapshot where the best spread offer is A or A+; 1 unit."},
]
CHOICE_RULE = ("complexity tiers base < ridge core < ridge market < ridge small < ridge all < lgbm small < lgbm all; "
               "start from the base (raw EV vs sharp spread); move to the best model of a higher tier only if its LOSO "
               "game-weighted MSE beats the current choice by > 0.5% relative (moneyline grade v2 rule, generalized)")


def choose(cv):
    cur = "base_ev_sp_sharp"
    for t in sorted(set(TIERS.values()))[1:]:
        ks = [k for k in cv if tier(k) == t]
        if not ks:
            continue
        b = min(ks, key=lambda k: cv[k]["wmse"])
        if cv[b]["wmse"] < cv[cur]["wmse"] * 0.995:
            cur = b
    return cur


def make(name):
    return dict(model_grid())[name]()


def importance(m):
    if isinstance(m, Ridge):
        return {f: r4(c) for f, c in zip(m.cols, m.coef)}
    if isinstance(m, Base):
        return {"ev_sp_sharp": 1.0}
    imp = m.m.feature_importance("gain")
    return {f: r4(g / imp.sum()) for f, g in zip(m.cols, imp)}


def run_dev():
    X = table(DEV)
    C = candidates(X)
    F = featurize(C)
    print("offers", len(X), "candidates", len(C), "games", C.game_id.nunique(), "mean clv all offers",
          r4(X.clv.mean()), "candidates", r4(np.average(C.clv, weights=C.w)), flush=True)
    cv, oof = loso(C, F)
    best = choose(cv)
    print("chosen", best)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV),
           "rows_all_offers": int(len(X)), "candidate_rows": int(len(C)), "games": int(C.game_id.nunique()),
           "clv_all_offers_mean": r4(X.clv.mean()), "clv_candidates_mean": r4(np.average(C.clv, weights=C.w)),
           "corr_clv_vs_clv_old": r4(np.corrcoef(C.clv, C.clv_old.fillna(C.clv))[0, 1]),
           "cv": cv, "chosen_model": best, "choice_rule": CHOICE_RULE}
    p = oof[best]
    res["oof_pred_quantiles"] = {str(q): r4(np.quantile(p, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)}
    res["oof_eval_chosen"], bets = evaluate(C, p, THRESHOLDS, STRATEGIES, len(DEV))
    res["oof_eval_base"], _ = evaluate(C, oof["base_ev_sp_sharp"], THRESHOLDS, STRATEGIES, len(DEV))
    D = C.assign(pred=p).sort_values("pred", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    scan = []
    for cut in (-0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        s = summ(first_with(D, D.pred >= cut), len(DEV))
        scan.append({"cut": cut, **{k: s.get(k) for k in ("bets", "per_season", "pred_clv", "clv", "clv_se", "roi",
                                                          "roi_se", "beat_close", "clv_old")}})
        print(scan[-1])
    res["oof_threshold_scan"] = scan
    m = make(best).fit(F, C.clv.values, C.w.values)
    res["fit_all_dev_importance"] = importance(m)
    res["fit_all_dev_model"] = m.to_json() if isinstance(m, Ridge) else {"kind": getattr(m, "kind", "?")}
    ra = min((k for k in cv if k.startswith("ridge_all")), key=lambda k: cv[k]["wmse"])
    res["fit_all_dev_ridge_all"] = make(ra).fit(F, C.clv.values, C.w.values).to_json()
    lg = min((k for k in cv if k.startswith("lgbm_all")), key=lambda k: cv[k]["wmse"])
    res["fit_all_dev_lgbm_all_gain"] = importance(make(lg).fit(F, C.clv.values, C.w.values))
    res["grade_v1_dev"] = v1_eval(X, C, len(DEV), bets)
    res["spread_rule_v1_dev"] = spread_rule_v1(X, len(DEV))
    # univariate descriptive: offer-level Spearman of each feature with CLV (dev, candidates)
    res["feature_spearman_dev"] = {f: spearman(F[f], C.clv)[0] for f in FEATS if F[f].std() > 0}
    store("dev", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cv", "fit_all_dev_ridge_all")}, indent=1,
                     default=MS._jd)[:20000])


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen spread grade v2 is final)")
    J = json.loads(JSON.read_text())
    if "dev" not in J:
        raise SystemExit("run dev first")
    best = J["dev"]["chosen_model"]
    X = table(DEV)
    C = candidates(X)
    F = featurize(C)
    m = make(best).fit(F, C.clv.values, C.w.values)
    if isinstance(m, Ridge):
        model = m.to_json()
    elif isinstance(m, Base):
        model = {"kind": "base", "feature_order": ["ev_sp_sharp"], "formula": "pred_clv = ev_sp_sharp (clipped)"}
    else:
        model = {"kind": "lgbm", "model_string": m.m.model_to_string(), "feature_order": m.cols}
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "grading_version": 2, "market": "spread",
          "what": "SPREAD GRADE v2 = predicted honest spread CLV of an offer: EV of our point+price under the "
                  "closing sharp (LowVig/BetOnline; fallback all books) spread+juice implied margin, total-aware "
                  "key-number model (margin_total_model.json), at the closing consensus total.",
          "margin_model": "margin_total_model.json (src/nflpred/margin_total.py MarginModel; mode abs, no ref total)",
          "fit_seasons": list(DEV), "holdout": list(HOLD), "model_choice": {"chosen": best, "rule": CHOICE_RULE},
          "model": model, "feature_order": model["feature_order"],
          "feature_definitions": {f: FEATURES[f] for f in model["feature_order"]}, "ev_clip": list(EV_CLIP),
          "fair_reference_definitions": {
              "sharp_books": sorted(SHARP), "consensus": "median over ALL books in the feed (incl. exchanges) of each "
              "book's spread+juice implied margin (books passing the main-line filter)",
              "implied_margin": "MarginModel.implied_mu(home_point, no-vig home cover share, snapshot median total)",
              "ml_implied_margin": "mu with P(home win)/(1-P(tie)) = no-vig ML home prob (total-aware model)",
              "model_blend": "spread_rules.json margin weights: model*mu_model + market*(-consensus home point) + "
                             "intercept; used only at eligible snapshots"},
          "candidate_filter": "allowed book; main line (price -145..+125 both sides, half-point number, overround "
                              "-1%..12%, number within 2.5 of the snapshot median); EV >= 0 vs at least one of sharp "
                              "spread / consensus spread / sharp ML-implied / model blend (eligible)",
          "thresholds_pred_clv": THRESHOLDS, "letters": "A+ >= A+ cut; A >= A cut; B >= B cut; else C",
          "one_bet_per_game": "first snapshot where the game's best predicted-CLV spread offer reaches the grade; "
                              "that offer (book, side, point, price) is the bet",
          "strategies": STRATEGIES,
          "pass_bar": {"monotone": "mean realized CLV of first-appearance bets increases with grade: Spearman rho across "
                                   "the 4 letters >= 0.8 and A+ highest",
                       "top_grade": "A+ realized CLV > 0 with one-sided p < 0.05 (t on bet-level CLV)",
                       "adopt_if": "both hold"},
          "evaluate_once": True}
    FROZEN.write_text(json.dumps(fz, indent=1, default=MS._jd))
    print(json.dumps({k: v for k, v in fz.items() if k != "model"}, indent=1, default=MS._jd))


def load_frozen_model(fz):
    md = fz["model"]
    cols = md["feature_order"]
    if md["kind"] == "base":
        return Base()
    if md["kind"] == "ridge":
        r = Ridge(md["alpha"], cols)
        r.mu = np.array([md["features"][f]["mean"] for f in cols])
        r.sd = np.array([md["features"][f]["sd"] for f in cols])
        r.coef = np.array([md["features"][f]["coef_std"] for f in cols])
        r.b0 = md["intercept"]
        return r
    import lightgbm as lgb
    g = GBM(cols)
    g.m = lgb.Booster(model_str=md["model_string"])
    return g


def run_holdout():
    if os.environ.get("GRADE2SP_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked: set GRADE2SP_HOLDOUT=I_HAVE_FROZEN after freezing")
    if not FROZEN.exists():
        raise SystemExit("freeze first")
    J = json.loads(JSON.read_text())
    if "holdout" in J:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    m = load_frozen_model(fz)
    th, strat = fz["thresholds_pred_clv"], fz["strategies"]
    X = table(HOLD)
    C = candidates(X)
    F = featurize(C)
    pred = m.predict(F)
    ev, bets = evaluate(C, pred, th, strat, len(HOLD))
    eb, _ = evaluate(C, C.ev_sharp_raw.fillna(C.ev_cons_raw).values, th, strat, len(HOLD))
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"],
           "seasons": list(HOLD), "rows_all_offers": int(len(X)), "clv_all_offers_mean": r4(X.clv.mean()),
           "eval": ev, "eval_base_ev_sp_sharp": eb, "grade_v1": v1_eval(X, C, len(HOLD), bets),
           "spread_rule_v1": spread_rule_v1(X, len(HOLD))}
    res["by_grade_by_season"] = {g: {int(s): summ(d, 1) for s, d in bets[g].groupby("season")} for g in ORDER}
    bg = ev["by_grade_first_appearance"]
    rho = ev["letters_spearman"]["rho"]
    top = bg["A+"]
    res["pass"] = {"monotone": bool(rho is not None and rho >= 0.8 and bg["A+"].get("clv", -1) ==
                                    max(bg[g].get("clv", -9) for g in ORDER if bg[g]["bets"])),
                   "top_grade": bool(top.get("bets", 0) > 2 and top["clv"] > 0 and top["clv_p_one_sided"] < 0.05)}
    res["pass"]["adopt"] = res["pass"]["monotone"] and res["pass"]["top_grade"]
    store("holdout", res)
    print(json.dumps(res, indent=1, default=MS._jd)[:25000])


def run_posthoc():
    """NOT pre-registered (after the one-shot holdout; descriptive only): where did the holdout A+ bets go wrong, and
    which raw signals carried CLV in each period (offer-level Spearman on candidates)."""
    if os.environ.get("GRADE2SP_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("set GRADE2SP_HOLDOUT=I_HAVE_FROZEN")
    J = json.loads(JSON.read_text())
    if "holdout" not in J:
        raise SystemExit("run holdout first")
    fz = json.loads(FROZEN.read_text())
    m, th = load_frozen_model(fz), fz["thresholds_pred_clv"]
    res = {"note": "post-hoc, descriptive; not part of the pre-registered test"}
    for lab, seas in (("dev_2020_22_in_sample_fit", DEV), ("holdout_2023_25", HOLD)):
        X = table(seas)
        C = candidates(X)
        C["pred"] = m.predict(featurize(C))
        D = best_offers(C, C.pred.values, th)
        A = first_with(D, D.grade == "A+")
        hb = pd.cut(A.hours_before, [0, 3, 30, 80, 200], labels=["<3h", "3-30h", "30-80h", ">80h"])
        g = lambda d: summ(d, len(seas))  # noqa: E731
        res[lab] = {
            "Aplus_by_hours_before": {str(k): g(d) for k, d in A.groupby(hb, observed=True)},
            "Aplus_sharp_spread_missing": {str(bool(k)): g(d) for k, d in A.groupby(A.mu_sharp.isna())},
            "Aplus_by_book": {k: g(d) for k, d in A.groupby("book")},
            "offer_spearman_with_clv": {
                "ev_sp_sharp": spearman(C.ev_sharp_raw, C.clv)[0], "ev_mlimp": spearman(C.ev_mlimp_raw, C.clv)[0],
                "ev_model_eligible": spearman(C.ev_model_raw, C.clv)[0], "v1_edge_blend": spearman(C.v1_edge, C.clv)[0],
                "move_mu": spearman(C.move_mu, C.clv)[0], "pt_off_cons": spearman(C.point - C.cons_pt_side, C.clv)[0]}}
    store("posthoc", res)
    print(json.dumps(res, indent=1, default=MS._jd)[:8000])


# ------------------------------------------------------------------------------------------------ report
def _pct(x, se=None):
    if x is None:
        return "n/a"
    return f"{100 * x:+.2f}%" + (f" ± {100 * se:.2f}" if se is not None else "")


HEAD = ("| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE | CLV, old-model close | "
        "points vs close |\n|---|---|---|---|---|---|---|---|---|")


def _row(name, s):
    if not s or not s.get("bets"):
        return f"| {name} | 0 | | | | | | | |"
    return (f"| {name} | {s['bets']} ({s['per_season']}/season) | {_pct(s.get('pred_clv'))} | "
            f"{_pct(s['clv'], s['clv_se'])} | {s.get('clv_p_one_sided')} | {100 * s['beat_close']:.0f}% | "
            f"{_pct(s['roi'], s['roi_se'])} | {_pct(s.get('clv_old'), s.get('clv_old_se'))} | "
            f"{s.get('clv_pts'):+.2f} |")


INTERP = """* The pre-registered bar is NOT met: spread grade v2 fails on 2023-25. A+ (pred >= +1%) realized CLV -2.0% ± 1.3
  (80 bets; dev OOF +2.6% ± 1.0), letters rho 0.2, A+ minus C +0.4% ± 1.2. A and B were ~0 (+0.1%), C -2.4%.
  Strategies: A+ only -2.0% ± 1.3; A+ and A -1.0% ± 0.6. No band is +CLV. (A+ ROI +19% ± 10 is luck: CLV says no.)
* What still holds: in the bulk the prediction is calibrated and ranks (decile rho 0.99, slope 0.94, offer-level rank
  corr 0.44, raw EV vs the sharp spread 0.47). It separates 'bad' (C, -2.4%) from 'about break-even' (A/B, ~0%), but it
  cannot find POSITIVE CLV: the top tail is mostly early-week (> 80 h before kick) soft-book numbers that look off
  the sharp line and the market then moves to the soft number (post-hoc: 67 of 80 A+ bets, -2.4%; the 20 with no
  LowVig/BetOnline spread -4.0%; FanDuel A+ -7.8%). Dev had no such pattern (> 80 h A+ +1.3%).
* Features: only the two price references matter out of sample -- EV vs the sharp spread+juice margin (coef 0.66 per
  unit EV) and EV vs the sharp-ML-implied margin (0.11). Off-market number, key-number position (3/7, crossing),
  juice, line move, book dispersion, hours, book, week and the model EV had univariate correlation but added nothing
  beyond the two EVs (full-feature ridge / LightGBM had LOWER LOSO R²: 0.064 / 0.053-0.063 vs 0.073 for the 2-feature
  ridge); the moneyline-vs-spread gap enters only through ev_mlimp.
* Grade v1 (grading.py, never fit to data) ranked spread CLV in BOTH periods: holdout rho 1.0 over six letters, A+
  +2.2% ± 1.3 (49 bets, p 0.05), A+ minus C +4.2% ± 1.4. Its A+ needs blend-model edge >= 5% + line moved our way +
  right side of 3/7, and post-hoc the walk-forward model EV correlated with spread CLV on 2023-25 (rank 0.11) but not
  on 2020-22 (0.04), which is why a model fit on 2020-22 could not learn it. On the same bets as v2, v1 A+ +3.9% ± 2.1
  (28). The live spread track rule (blend EV >= 3%) is -2.1% ± 0.3 CLV on 2023-25: the v1 grade is better than the
  rule it labels, but one borderline holdout does not make v1 A+ a bet signal.
* Recommendation: do not add spread grade v2 to production (do not copy the frozen file to the repo root). Keep v1 as
  the spread label. If anything, a new pre-registered spread study should start from v1's A+ conjunction and the
  model-blend EV, judged on 2026 live CLV; and do not take early-week off-market soft-book spread numbers on EV vs the
  sharp line alone.
* Caveats: 2023-25 was already the holdout of earlier studies; the dev model choice was extended (feature-set tiers)
  after the first dev CV run showed the full set overfit -- all before the freeze, none on 2023-25.
"""


def run_report():
    J = json.loads(JSON.read_text())
    fz = json.loads(FROZEN.read_text())
    Dv, H = J["dev"], J.get("holdout")
    L = ["# Grade v2 for SPREAD offers: predicted honest spread CLV", "",
         "Code `scripts/research/grade_v2_spread.py` (dev -> freeze -> holdout once -> report). Numbers "
         "`grade_v2_spread.json`; frozen model, bands and strategies `grade_v2_spread_frozen.json` (frozen "
         f"{fz['frozen_at']}, before the 2023-25 run). Mirrors the moneyline grade v2 (`grade_v2.md`). Caveat: 2023-25 "
         "was the holdout of earlier studies too; nothing here was tuned on it.", "",
         "**Target**: EV of our point + price under the closing sharp (LowVig/BetOnline, fallback all books) "
         "spread+juice implied margin, valued with the total-aware key-number model (`margin_total_model.json`) at the "
         "closing total -- the same model prices every entry reference. Robustness column: the old single-weight "
         "model at `edge_lab.closing_fair` (= `edge_lab.spread_clv_price`).", ""]
    if H:
        L += [f"**Verdict: {'ADOPT' if H['pass']['adopt'] else 'DO NOT ADOPT'} spread grade v2** (pre-registered bar: "
              f"realized CLV rising with grade [{H['pass']['monotone']}] and A+ CLV > 0 at one-sided p < 0.05 "
              f"[{H['pass']['top_grade']}]).", ""]
    L += ["## Model", "",
          f"Candidates: {Dv['candidate_rows']} offer rows / {Dv['games']} games (dev; all main-line allowed-book offers "
          f"{Dv['rows_all_offers']}, mean CLV {_pct(Dv['clv_all_offers_mean'])}). LOSO (2020/21/22) game-weighted MSE, "
          "R² vs constant: " + ", ".join(f"{k} {v['r2_vs_const']}" for k, v in Dv["cv"].items()) +
          f". Chosen: **{Dv['chosen_model']}** ({CHOICE_RULE}).", ""]
    imp = Dv["fit_all_dev_importance"]
    if fz["model"]["kind"] == "lgbm":
        L += ["Feature importance (share of split gain, fit on 2020-22): " + ", ".join(
            f"{f} {100 * g:.1f}%" for f, g in sorted(imp.items(), key=lambda kv: -kv[1]) if g and g > 0.001) +
            "; others ~0.", ""]
    else:
        L += ["Ridge coefficients (CLV per 1 SD): " + ", ".join(
            f"{f} {100 * g:+.3f}%" for f, g in sorted(imp.items(), key=lambda kv: -abs(kv[1]))[:12]), ""]
    L += ["Univariate offer-level Spearman with CLV (dev candidates): " + ", ".join(
        f"{f} {v:+.3f}" for f, v in sorted(Dv["feature_spearman_dev"].items(), key=lambda kv: -abs(kv[1] or 0))
        if v is not None and abs(v) >= 0.02), "",
        f"Bands (pred CLV, chosen on dev): A+ >= {_pct(fz['thresholds_pred_clv']['A+'])}, A >= "
        f"{_pct(fz['thresholds_pred_clv']['A'])}, B >= {_pct(fz['thresholds_pred_clv']['B'])}, else C.", "",
        "Dev band scan (OOF, first bet per game with pred >= cut): " + "; ".join(
            f"{s['cut']:+.3f}: {s['bets']} bets, CLV {_pct(s['clv'], s['clv_se'])}" for s in Dv["oof_threshold_scan"]), ""]
    blocks = [("Dev 2020-22 (out-of-fold predictions)", Dv["oof_eval_chosen"], Dv["grade_v1_dev"],
               Dv["spread_rule_v1_dev"])]
    if H:
        blocks.append(("HOLDOUT 2023-25 (frozen model, run once)", H["eval"], H["grade_v1"], H["spread_rule_v1"]))
    for lab, E, V1, rule in blocks:
        tb = E["top_minus_bottom"]
        L += [f"## {lab}", "", "By grade (one bet per game per grade, first snapshot where the game's best spread "
              "offer has that grade):", "", HEAD]
        L += [_row(g, E["by_grade_first_appearance"][g]) for g in ORDER]
        L += ["", f"Spearman across letters: rho {E['letters_spearman']['rho']}; A+ minus C CLV "
              f"{_pct(tb.get('diff'), tb.get('se_game_boot'))} (game bootstrap, z {tb.get('z')}); bet-level rank corr "
              f"{E['bet_level_spearman_pred_vs_clv']['rho']}; deciles rho {E['deciles_spearman']['rho']}; offer-level "
              f"rank corr {E['offer_spearman_pred_vs_clv']}; calibration slope {E['calibration_slope']}.", "",
              "Deciles (pred -> realized CLV): " + ", ".join(f"{_pct(d['pred'])}->{_pct(d['clv'])}"
                                                           for d in E["deciles"]), "",
              "Strategies and the live spread rule (spread_rules.json v1):", "", HEAD]
        L += [_row(k, s) for k, s in E["strategies"].items()] + [_row("rule spread v1", rule), ""]
        L += ["Grade v1 (`grading.py`, production use: each snapshot's best blend-EV offer):", "", HEAD]
        L += [_row(f"v1 {g}", V1["production_v1_by_letter"][g]) for g in V1_ORDER]
        t1 = V1["v1_top_minus_bottom"]
        L += ["", f"v1 Spearman across 6 letters: rho {V1['v1_letters_spearman']['rho']}; v1 A+ minus C "
              f"{_pct(t1.get('diff'), t1.get('se_game_boot'))}; offer-level rank corr of v1 score with CLV "
              f"{V1['offer_spearman_v1score_vs_clv']}.", "",
              f"Same bets (the {V1['same_bets_n']} grade-v2 by-grade bets above), relabelled with v1: rank corr with "
              f"CLV v1 score {V1['same_bets_spearman_v1score_vs_clv']['rho']} vs v2 prediction "
              f"{V1['same_bets_spearman_v2pred_vs_clv']['rho']}.", "", HEAD]
        L += [_row(f"same bets, v1 {g}", V1["same_bets_by_v1"][g]) for g in V1_ORDER if g in V1["same_bets_by_v1"]]
        L.append("")
    if H:
        E0 = H["eval_base_ev_sp_sharp"]
        L += ["Holdout baseline -- same bands on raw EV vs the sharp spread-implied margin (no model):", "", HEAD]
        L += [_row(g, E0["by_grade_first_appearance"][g]) for g in ORDER]
        L += ["", "Holdout by grade and season (CLV, bets):", ""]
        for g in ORDER:
            L.append(f"* {g}: " + ", ".join(f"{s}: {_pct(v.get('clv'))} ({v.get('bets')})"
                                            for s, v in H["by_grade_by_season"][g].items()))
        P = J.get("posthoc")
        if P:
            q = lambda d: f"{d['bets']} bets, CLV {_pct(d['clv'], d['clv_se'])}"  # noqa: E731
            L += ["", "## Post-hoc (after the one-shot holdout; descriptive, not part of the test)", ""]
            for lab in ("dev_2020_22_in_sample_fit", "holdout_2023_25"):
                Q = P[lab]
                L += [f"* {lab}: A+ by hours before kick: " + "; ".join(f"{k}: {q(d)}" for k, d in
                                                                          Q["Aplus_by_hours_before"].items()),
                      f"  A+ with no sharp spread: {q(Q['Aplus_sharp_spread_missing'].get('True', {'bets': 0, 'clv': None, 'clv_se': None}))}"
                      f"; by book: " + "; ".join(f"{k}: {q(d)}" for k, d in Q["Aplus_by_book"].items()),
                      "  offer-level Spearman with CLV: " + ", ".join(f"{k} {v}" for k, v in Q["offer_spearman_with_clv"].items())]
        L += ["", "## Reading", "", INTERP]
    MD.write_text("\n".join(L) + "\n")
    print(MD.read_text())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": run_dev, "freeze": run_freeze, "holdout": run_holdout, "posthoc": run_posthoc,
     "report": run_report}[stage]()
