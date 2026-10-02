"""GRADE v2: the predicted closing-line value (CLV) of a candidate MONEYLINE bet, learned from data.

Replaces the hand-made A+..C score of src/nflpred/grading.py (which did not rank bets on 2020-25 early lines) with a
regression of realized CLV on bet-time features. Stages (outputs output/research/grade_v2.{json,md}, grade_v2_frozen.json):

  python scripts/research/grade_v2.py dev        # 2020-22 only: candidate table, season-grouped CV, model choice,
                                                 # thresholds, dev results of grades / strategies / v2-v4 rules
  python scripts/research/grade_v2.py freeze     # final fit on 2020-22 -> grade_v2_frozen.json (refuses overwrite):
                                                 # features, scaler, coefficients, thresholds, strategies, pass bar
  GRADEV2_HOLDOUT=I_HAVE_FROZEN python scripts/research/grade_v2.py holdout   # ONCE on 2023-25 (refuses re-run)
  python scripts/research/grade_v2.py report     # writes the .md

Candidate bets: one row per (game, snapshot <= 7 days and >= 10 min before kick, allowed book, side) moneyline offer at
the historical snapshot times (daily 14:10 UTC, Fri 21:40, ~75 min pre-kick), price -1000..+1000, within 12 pp of the
consensus no-vig (stale/bad quotes out), and EV >= 0 vs at least one fair reference (sharp no-vig ML, v3 sharp median
incl. Pinnacle, sharp spread-implied P(win), all-book consensus, walk-forward model at eligible snapshots).

Target: CLV = dec x closing sharp no-vig P(side) - 1 (edge_lab.closing_fair p_close_sharp, fallback all books), the
definition used by moneyline v1-v4. Robustness: vs closing sharp spread-implied P(win), vs last Pinnacle ML (2024-25).

Model selection (stated before looking): leave-one-season-out (2020 / 2021 / 2022) game-weighted MSE of CLV
(each game's rows weigh 1 in total); the simpler model (ridge) is kept unless a GBM beats it by > 0.5% relative MSE.
Env GRADEV2_CACHE=<dir> caches intermediate tables (default: build_ml's MLSP_CACHE if set).
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
if os.environ.get("GRADEV2_CACHE") and not os.environ.get("MLSP_CACHE"):
    os.environ["MLSP_CACHE"] = os.environ["GRADEV2_CACHE"]
import ml_spread_consistency as MS  # noqa: E402
from nflpred import grading as G1  # noqa: E402
from nflpred import odds as O  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "grade_v2.json"
FROZEN = OUT / "grade_v2_frozen.json"
MD = OUT / "grade_v2.md"
DEV = (2020, 2021, 2022)
HOLD = (2023, 2024, 2025)
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet"]
EV_CLIP = (-0.15, 0.25)
r4 = MS.r4

# ------------------------------------------------------------------------------------------------ features
# name -> definition (kept in the frozen file so production computes the same thing)
FEATURES = {
    "ev_ml_sharp": "EV vs sharp no-vig ML (median LowVig/BetOnline; tie refund); fallback ev_cons; clipped",
    "ev_sp_sharp": "EV vs sharp SPREAD-implied P(win) (LowVig/BetOnline spread+juice -> total-aware margin model); fallback consensus spread-implied; clipped",
    "ev_cons": "EV vs all-book median no-vig ML; clipped",
    "ev_model": "EV vs walk-forward model p_model at eligible snapshots (injury report out, QBs confirmed), else 0; clipped",
    "model_elig": "1 if the snapshot is model-eligible",
    "sharp_missing": "1 if no LowVig/BetOnline ML at the snapshot",
    "n_signals": "count of v2 (ev_ml_sharp>=2% & ev_model>=0 & eligible), v3 (ev_v3>=2% & ev_model>=0 & eligible), "
                 "v4 (Tue/Fri/Sun window & ev_sp_sharp>=2% & ev_ml_sharp>=0 & ML<=+400) conditions met",
    "move_p": "consensus no-vig P(side) now minus at first sight (<= 9 days before kick); + = toward our side",
    "move_pts": "consensus spread move since first sight, points toward our side",
    "log_hours": "log(1 + hours before kickoff)",
    "is_last": "1 at the game's last snapshot (~75 min pre-kick)",
    "p_imp": "1/decimal price (implied probability of the offer)",
    "is_dog": "1 if price > 0",
    "longshot": "1 if price >= +250",
    "big_fav": "1 if price <= -200",
    "disp": "std across all books of no-vig home P (moneyline dispersion) at the snapshot",
    "best_gap": "this offer's implied prob minus the best allowed-book implied prob for the side (0 = best price)",
    "key_pos": "grading.key_number_side(consensus spread of our side, to 0.5): +1 right / -1 wrong side of 3/7",
    "on3": "1 if |consensus spread| in 2.5..3.5",
    **{f"bk_{b}": f"1 if book == {b}" for b in BOOKS[1:]},
}
FEATS = list(FEATURES)
# qb_flag is NOT a model input: the historical flag (edge_lab) uses the ACTUAL starter, known only after the fact.
# ev_v3 (Pinnacle/LowVig/BetOnline median) is NOT a model input: before 2024 it equals ev_ml_sharp, so its Pinnacle
# part could never be learned on 2020-22; it enters only through n_signals (the v3 condition) and the rule comparison.
MONO = {"ev_ml_sharp": 1, "ev_sp_sharp": 1, "ev_cons": 1, "n_signals": 1}


def edge_lab_snaps(seasons) -> pd.DataFrame:
    p = ROOT / "data" / ("edge_lab_dev.parquet" if set(seasons) <= set(DEV) else "edge_lab_holdout.parquet")
    e = pd.read_parquet(p, columns=["game_id", "requested_ts", "season", "p_cons", "m_cons", "p_first", "m_first",
                                    "eligible", "p_model", "qb_flag"])
    e = e[e.season.isin(seasons)].drop_duplicates(["game_id", "requested_ts"]).drop(columns="season")
    return e.rename(columns={"p_cons": "el_p_cons", "m_cons": "el_m_cons"})


def table(seasons) -> pd.DataFrame:
    seasons = tuple(seasons)
    if set(seasons) & set(HOLD) and os.environ.get("GRADEV2_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked")
    M = MS.get_model("chosen")
    rows, snaps = MS.build_ml(seasons, M, MS.get_model("teasers_v2_single"))
    allowed = O.load_allowed_books()
    # per-snapshot extras from all books: dispersion, LowVig / BetOnline no-vig for the v3 reference
    k2 = ["game_id", "requested_ts"]
    ex = rows.groupby(k2).p_ml.std().rename("disp").to_frame()
    for b in ("lowvig", "betonlineag"):
        ex = ex.join(rows[rows.book == b].groupby(k2).p_ml.median().rename(f"p_{b}"))
    snaps = snaps.merge(ex.reset_index(), on=k2, how="left")
    snaps["p_v3"] = snaps[["p_pin", "p_lowvig", "p_betonlineag"]].median(axis=1)
    X = MS.windows(MS.bets_table(rows, snaps, allowed))
    h = (X.side == "home").values
    X["p_v3_s"] = np.where(h, X.p_v3, 1 - X.p_v3)
    X = X.merge(edge_lab_snaps(seasons), on=k2, how="left")
    pt = X.p_tie_cons.fillna(0.003)
    sg = np.where(h, 1, -1)
    X["ev_cons_raw"] = (X.p_ml_cons_s * X.dec - 1) * (1 - pt)
    X["ev_v3_raw"] = (X.p_v3_s * X.dec - 1) * (1 - pt)
    X["p_model_s"] = np.where(h, X.p_model, 1 - X.p_model)
    X["eligible"] = X.eligible.fillna(False).astype(bool)
    X["ev_model_raw"] = np.where(X.eligible & X.p_model_s.notna(), X.dec * X.p_model_s - 1, np.nan)
    X["move_p"] = sg * (X.el_p_cons - X.p_first)
    X["move_pts"] = sg * (X.el_m_cons - X.m_first)
    best = X.groupby(k2 + ["side"]).dec.transform("max")
    X["best_gap"] = 1 / X.dec - 1 / best
    X["ml_gap_cons"] = (1 / X.dec - X.p_ml_cons_s).abs()
    return X


def featurize(X: pd.DataFrame) -> pd.DataFrame:
    c = lambda s: np.clip(s, *EV_CLIP)  # noqa: E731
    F = pd.DataFrame(index=X.index)
    F["ev_cons"] = c(X.ev_cons_raw.fillna(0.0))
    F["ev_ml_sharp"] = c(X.ev_p_ml_sharp.fillna(X.ev_cons_raw).fillna(0.0))
    F["ev_sp_sharp"] = c(X.ev_p_sp_sharp.fillna(X.ev_p_sp_cons).fillna(X.ev_cons_raw).fillna(0.0))
    F["ev_model"] = c(X.ev_model_raw.fillna(0.0))
    F["model_elig"] = X.ev_model_raw.notna().astype(float)
    F["sharp_missing"] = X.ev_p_ml_sharp.isna().astype(float)
    em = X.ev_model_raw
    v2 = (X.ev_p_ml_sharp >= 0.02) & (em >= 0)
    v3 = (X.ev_v3_raw >= 0.02) & (em >= 0)
    v4 = X.win.isin(["tue", "fri", "sun"]) & (X.ev_p_sp_sharp >= 0.02) & (X.ev_p_ml_sharp >= 0) & (X.ml <= 400)
    F["n_signals"] = v2.astype(float) + v3.astype(float) + v4.astype(float)
    F["move_p"] = np.clip(X.move_p.fillna(0.0), -0.2, 0.2)
    F["move_pts"] = np.clip(X.move_pts.fillna(0.0), -7, 7)
    F["log_hours"] = np.log1p(X.hours_before.clip(lower=0))
    F["is_last"] = X.is_last.astype(float)
    F["p_imp"] = 1 / X.dec
    F["is_dog"] = (X.ml > 0).astype(float)
    F["longshot"] = (X.ml >= 250).astype(float)
    F["big_fav"] = (X.ml <= -200).astype(float)
    F["disp"] = X.disp.fillna(X.disp.median() if X.disp.notna().any() else 0.01).clip(0, 0.1)
    F["best_gap"] = X.best_gap.clip(0, 0.1)
    pts = (np.round(X.cons_pt_side.fillna(0) * 2) / 2).values
    F["key_pos"] = [float(G1.key_number_side(float(p))) for p in pts]
    F["on3"] = ((np.abs(pts) >= 2.5) & (np.abs(pts) <= 3.5)).astype(float)
    for b in BOOKS[1:]:
        F[f"bk_{b}"] = (X.book == b).astype(float)
    return F[FEATS].astype(float)


def candidates(X: pd.DataFrame) -> pd.DataFrame:
    ok = X.ml.between(-1000, 1000) & (X.ml_gap_cons <= 0.12) & X.clv.notna()
    anypos = ((X.ev_p_ml_sharp >= 0) | (X.ev_v3_raw >= 0) | (X.ev_p_sp_sharp >= 0) | (X.ev_cons_raw >= 0) |
              (X.ev_model_raw >= 0))
    C = X[ok & anypos].reset_index(drop=True)
    C["w"] = 1.0 / C.groupby("game_id").game_id.transform("size")
    return C


# ------------------------------------------------------------------------------------------------ models
class Ridge:
    kind = "ridge"

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, F, y, w):
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
        return self.b0 + ((F.values - self.mu) / self.sd) @ self.coef

    def to_json(self):
        return {"kind": "ridge", "alpha": self.alpha, "intercept": float(self.b0),
                "features": {f: {"mean": float(m), "sd": float(s), "coef_std": float(c), "coef_raw": float(c / s)}
                             for f, m, s, c in zip(FEATS, self.mu, self.sd, self.coef)},
                "formula": "pred_clv = intercept + sum_f coef_std[f] * (x_f - mean[f]) / sd[f]"}


class Base:
    """No fit: predicted CLV = EV vs sharp no-vig ML (the v2 signal)."""
    kind = "ev_ml_sharp"

    def fit(self, F, y, w):
        return self

    def predict(self, F):
        return F.ev_ml_sharp.values


class GBM:
    kind = "lgbm"

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, F, y, w):
        import lightgbm as lgb
        mono = [MONO.get(f, 0) for f in FEATS]
        p = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=400, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, monotone_constraints=mono, verbose=-1, seed=7)
        p.update(self.kw)
        n = p.pop("n_trees", 300)
        self.m = lgb.train(p, lgb.Dataset(F.values, y, weight=w, feature_name=FEATS), n)
        return self

    def predict(self, F):
        return self.m.predict(F.values)


def model_grid():
    g = [("base_ev_ml_sharp", lambda: Base())]
    for a in (1.0, 30.0, 300.0, 3000.0, 30000.0):
        g.append((f"ridge_a{a:g}", lambda a=a: Ridge(a)))
    for nt, lv in ((200, 4), (400, 7)):
        g.append((f"lgbm_mono_{nt}x{lv}", lambda nt=nt, lv=lv: GBM(n_trees=nt, num_leaves=lv)))
    return g


def wmse(y, p, w):
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def loso(C, F):
    y, w = C.clv.values, C.w.values
    res, oof = {}, {}
    var0 = None
    for name, mk in model_grid():
        pred = np.full(len(C), np.nan)
        for s in DEV:
            te = (C.season == s).values
            m = mk().fit(F[~te], y[~te], w[~te])
            pred[te] = m.predict(F[te])
        oof[name] = pred
        res[name] = {"wmse": wmse(y, pred, w)}
        if var0 is None:
            var0 = wmse(y, np.full(len(y), np.average(y, weights=w)), w)
        res[name]["r2_vs_const"] = r4(1 - res[name]["wmse"] / var0)
        print(name, res[name], flush=True)
    return res, oof


# ------------------------------------------------------------------------------------------------ grades / bets
def letter(pred, th):
    return np.where(pred >= th["A+"], "A+", np.where(pred >= th["A"], "A", np.where(pred >= th["B"], "B", "C")))


ORDER = ["A+", "A", "B", "C"]


def best_offers(C, pred, th):
    """Best predicted-CLV offer per (game, snapshot) -- what the dashboard would show as that game's bet."""
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


def summ(B, nseas, units=None):
    if B is None or len(B) == 0:
        return {"bets": 0}
    u = np.ones(len(B)) if units is None else np.asarray(units, float)
    clv = B.clv.values
    pnl = B.pnl.values
    roi = float((pnl * u).sum() / u.sum())
    rng = np.random.default_rng(5)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(B), len(B))
        bs.append((pnl[i] * u[i]).sum() / u[i].sum())
    cm = float(np.average(clv, weights=u))
    cse = float(np.std(clv, ddof=1) / math.sqrt(len(clv))) if len(clv) > 1 else None
    if units is not None:
        cse = boot_se(clv) if cse else None
    t = cm / cse if cse else None
    out = {"bets": int(len(B)), "per_season": round(len(B) / nseas, 1), "units": round(float(u.sum()), 1),
           "pred_clv": r4(np.average(B.pred, weights=u)) if "pred" in B else None,
           "clv": r4(cm), "clv_se": r4(cse), "clv_t": r4(t),
           "clv_p_one_sided": r4(0.5 * math.erfc(t / math.sqrt(2))) if t is not None else None,
           "beat_close": r4((clv > 0).mean()), "roi": r4(roi), "roi_se_boot": r4(np.std(bs, ddof=1)),
           "avg_ml": int(np.median(B.ml)), "dog_share": r4((B.ml > 0).mean()),
           "clv_spread_close": r4(np.nanmean(B.clv_sp)) if B.clv_sp.notna().any() else None,
           "clv_pinnacle": r4(np.nanmean(B.clv_pin)) if B.clv_pin.notna().any() else None,
           "clv_pinnacle_se": r4(np.nanstd(B.clv_pin, ddof=1) / math.sqrt(B.clv_pin.notna().sum()))
           if B.clv_pin.notna().sum() > 2 else None,
           "n_pin": int(B.clv_pin.notna().sum())}
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return None, None
    r = spearmanr(x[ok], y[ok])
    return r4(r.statistic), r4(r.pvalue)


def evaluate(C, pred, th, strategies, nseas) -> dict:
    D = best_offers(C, pred, th)
    out = {"games": int(C.game_id.nunique()), "candidate_rows": int(len(C)), "snapshot_best_offers": int(len(D))}
    # (1) by grade: one bet per (game, grade) at the first snapshot where the game's best offer carries that grade
    bg = {}
    for g in ORDER:
        bg[g] = summ(first_with(D, D.grade == g), nseas)
    out["by_grade_first_appearance"] = bg
    rho, p = spearman([4, 3, 2, 1], [bg[g].get("clv") if bg[g]["bets"] else np.nan for g in ORDER])
    out["letters_spearman"] = {"rho": rho, "p": p}
    # (2) snapshot-level best offers by grade, game-cluster bootstrap SE
    sl = {}
    for g in ORDER:
        e = D[D.grade == g]
        if len(e):
            gm = e.groupby("game_id").clv.mean()
            sl[g] = {"offers": int(len(e)), "games": int(len(gm)), "clv": r4(e.clv.mean()),
                     "clv_se_game_boot": r4(boot_se(gm.values)), "pred": r4(e.pred.mean())}
    out["by_grade_snapshot_best_offers"] = sl
    # (3) deciles of predicted CLV (snapshot best offers) vs realized
    D["dec_bin"] = pd.qcut(D.pred.rank(method="first"), 10, labels=False)
    dd = D.groupby("dec_bin").agg(pred=("pred", "mean"), clv=("clv", "mean"), n=("clv", "size")).reset_index()
    out["deciles"] = [{k: (r4(v) if isinstance(v, float) else int(v)) for k, v in r.items()} for r in dd.to_dict("records")]
    rho, p = spearman(dd.pred, dd.clv)
    out["deciles_spearman"] = {"rho": rho, "p": p}
    # also offer-level rank correlation (all candidate rows), and calibration slope
    from scipy.stats import spearmanr
    out["offer_spearman_pred_vs_clv"] = r4(spearmanr(pred, C.clv.values).statistic)
    W = C.w.values
    pc = pred - np.average(pred, weights=W)
    out["calibration_slope"] = r4(float(np.sum(W * pc * (C.clv.values - np.average(C.clv.values, weights=W))) /
                                        np.sum(W * pc ** 2)))
    # (4) strategies: one bet per game at the first snapshot where the best offer reaches the minimum grade
    st = {}
    for s in strategies:
        B = first_with(D, D.pred >= th[s["min_grade"]])
        units = None
        if s["stake"] == "prop_pred":
            units = np.clip(B.pred.values / s["unit_clv"], s["min_units"], s["max_units"])
        st[s["id"]] = summ(B, nseas, units)
        st[s["id"]]["by_season"] = {int(k): summ(d, 1, None if units is None else units[(B.season == k).values]).get("clv")
                                   for k, d in B.groupby("season")}
    out["strategies"] = st
    return out


# ------------------------------------------------------------------------------------------------ existing rules
def existing_rules(C_all: pd.DataFrame, nseas) -> dict:
    """v2 / v3 / v4 as written in moneyline_v{2,3,4}_rules.json, on this table (all offers, not just candidates).
    v2/v3: eligible snapshot, ev_vs_ref >= 2% and ev_model >= 0, -1000..+1000, gap <= 12 pp; first qualifying
    snapshot, best EV. v4: first Tue/Fri/Sun window snapshot with ev_sp_sharp >= 2% & ev_ml_sharp >= 0, ML <= +400."""
    X = C_all
    base = X.ml.between(-1000, 1000) & (X.ml_gap_cons <= 0.12) & X.clv.notna()
    out = {}
    defs = {
        "v2": (base & (X.ev_p_ml_sharp >= 0.02) & (X.ev_model_raw >= 0), "ev_p_ml_sharp"),
        "v3": (base & (X.ev_v3_raw >= 0.02) & (X.ev_model_raw >= 0), "ev_v3_raw"),
        "v4": (base & X.win.isin(["tue", "fri", "sun"]) & (X.ev_p_sp_sharp >= 0.02) & (X.ev_p_ml_sharp >= 0)
               & (X.ml <= 400), "ev_p_sp_sharp"),
    }
    for k, (m, sc) in defs.items():
        Y = X[m].sort_values(sc, ascending=False).drop_duplicates(["game_id", "requested_ts"])
        Y = Y.sort_values("requested_ts").drop_duplicates("game_id")
        out[k] = summ(Y.assign(pred=np.nan), nseas)
        out[k]["by_season"] = {int(s): summ(d, 1).get("clv") for s, d in Y.groupby("season")}
    return out


# ------------------------------------------------------------------------------------------------ stages
def _jd(o):
    return MS._jd(o)


def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=_jd))


# Chosen on dev (2020-22) only -- see run_dev output; edited here BEFORE freeze, never after.
# Dev OOF scan (first bet per game with pred >= cut): 1.5% -> 86/season, CLV +2.7%; 2.5% -> 29/season, +3.6%;
# 3% -> 14/season (too few to test). A+ cut at 2.5% keeps ~90 holdout bets for the top-grade test.
THRESHOLDS = {"A+": 0.025, "A": 0.015, "B": 0.005}
STRATEGIES = [
    {"id": "S1_Aplus_flat", "min_grade": "A+", "stake": "flat",
     "desc": "One bet per game at the first snapshot where the best offer is A+ (pred CLV >= A+ cut); 1 unit."},
    {"id": "S2_A_and_up_flat", "min_grade": "A", "stake": "flat",
     "desc": "One bet per game at the first snapshot where the best offer is A or A+; 1 unit."},
    {"id": "S3_B_and_up_prop", "min_grade": "B", "stake": "prop_pred", "unit_clv": 0.02, "min_units": 0.25,
     "max_units": 2.0,
     "desc": "One bet per game at the first snapshot where the best offer is B or better; units = pred CLV / 2% clipped to 0.25..2."},
]
CHOICE_RULE = "lowest LOSO game-weighted MSE; ridge kept unless a GBM beats the best ridge by > 0.5% relative MSE"


def choose(cv):
    ridge = min((k for k in cv if k.startswith("ridge")), key=lambda k: cv[k]["wmse"])
    gbm = min((k for k in cv if k.startswith("lgbm")), key=lambda k: cv[k]["wmse"])
    best = gbm if cv[gbm]["wmse"] < cv[ridge]["wmse"] * 0.995 else ridge
    return best


def make(name):
    return dict(model_grid())[name]()


def run_dev():
    X = table(DEV)
    C = candidates(X)
    F = featurize(C)
    print("rows", len(X), "candidates", len(C), "games", C.game_id.nunique(), flush=True)
    cv, oof = loso(C, F)
    best = choose(cv)
    print("chosen", best)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV),
           "rows_all_offers": int(len(X)), "candidate_rows": int(len(C)), "games": int(C.game_id.nunique()),
           "clv_candidates_mean": r4(np.average(C.clv, weights=C.w)), "cv": cv, "chosen_model": best,
           "choice_rule": CHOICE_RULE}
    p = oof[best]
    res["oof_pred_quantiles"] = {str(q): r4(np.quantile(p, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)}
    res["oof_eval_chosen"] = evaluate(C, p, THRESHOLDS, STRATEGIES, len(DEV))
    res["oof_eval_base"] = evaluate(C, oof["base_ev_ml_sharp"], THRESHOLDS, STRATEGIES, len(DEV))
    if best != min((k for k in cv if k.startswith("ridge")), key=lambda k: cv[k]["wmse"]):
        rb = min((k for k in cv if k.startswith("ridge")), key=lambda k: cv[k]["wmse"])
        res["oof_eval_best_ridge"] = evaluate(C, oof[rb], THRESHOLDS, STRATEGIES, len(DEV))
    # threshold scan on OOF (dev): bets/season and realized CLV for first-appearance >= cut
    D = C.assign(pred=p).sort_values("pred", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    scan = []
    for cut in (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05):
        B = first_with(D, D.pred >= cut)
        s = summ(B, len(DEV))
        scan.append({"cut": cut, **{k: s.get(k) for k in ("bets", "per_season", "pred_clv", "clv", "clv_se", "roi",
                                                          "roi_se_boot", "beat_close")}})
    res["oof_threshold_scan"] = scan
    m = make(best).fit(F, C.clv.values, C.w.values)
    if isinstance(m, Ridge):
        res["fit_all_dev"] = m.to_json()
    else:
        imp = m.m.feature_importance("gain")
        res["fit_all_dev"] = {"gain": {f: r4(g / imp.sum()) for f, g in zip(FEATS, imp)}}
        rb = min((k for k in cv if k.startswith("ridge")), key=lambda k: cv[k]["wmse"])
        res["fit_all_dev_best_ridge"] = make(rb).fit(F, C.clv.values, C.w.values).to_json()
    res["existing_rules_dev"] = existing_rules(X.assign(**{c: X[c] for c in ()}), len(DEV))
    store("dev", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("cv",)}, indent=1, default=_jd)[:15000])


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen grade v2 is final)")
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
    else:
        model = {"kind": "lgbm", "model_string": m.m.model_to_string(), "feature_order": FEATS}
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "grading_version": 2,
          "what": "GRADE v2 = predicted CLV of a moneyline offer (dec x closing sharp no-vig P(side) - 1).",
          "fit_seasons": list(DEV), "holdout": list(HOLD), "model_choice": {"chosen": best, "rule": CHOICE_RULE},
          "model": model, "feature_order": FEATS, "feature_definitions": FEATURES, "ev_clip": list(EV_CLIP),
          "candidate_filter": "price -1000..+1000; |1/dec - consensus no-vig P(side)| <= 0.12; EV >= 0 vs at least one "
                              "of sharp ML / v3 median / sharp spread-implied / consensus / model (eligible)",
          "thresholds_pred_clv": THRESHOLDS, "letters": "A+ >= A+ cut; A >= A cut; B >= B cut; else C",
          "one_bet_per_game": "first snapshot where the game's best predicted-CLV offer reaches the strategy's grade; "
                              "that offer (book, side, price) is the bet",
          "strategies": STRATEGIES,
          "pass_bar": {"monotone": "mean realized CLV of first-appearance bets increases with grade: Spearman rho across "
                                   "the 4 letters >= 0.8 (at most one adjacent swap, and A+ must be highest)",
                       "top_grade": "A+ realized CLV > 0 with one-sided p < 0.05 (t on bet-level CLV)",
                       "adopt_if": "both hold"},
          "evaluate_once": True}
    FROZEN.write_text(json.dumps(fz, indent=1, default=_jd))
    print(json.dumps({k: v for k, v in fz.items() if k != "model"}, indent=1, default=_jd))
    print(json.dumps(model, default=_jd)[:3000])


def load_frozen_model(fz):
    md = fz["model"]
    if md["kind"] != "ridge" and "gain" in Dv.get("fit_all_dev", {}):
        L += ["Feature importance (share of total split gain, fit on 2020-22): " + ", ".join(
            f"{f} {100 * g:.1f}%" for f, g in sorted(Dv["fit_all_dev"]["gain"].items(), key=lambda kv: -kv[1]) if g > 0)
            + "; all other features 0. Production: `lightgbm.Booster(model_str=frozen['model']['model_string'])`, "
              "features in `feature_order`.", ""]
    if md["kind"] == "ridge":
        r = Ridge(md["alpha"])
        r.mu = np.array([md["features"][f]["mean"] for f in FEATS])
        r.sd = np.array([md["features"][f]["sd"] for f in FEATS])
        r.coef = np.array([md["features"][f]["coef_std"] for f in FEATS])
        r.b0 = md["intercept"]
        return r
    import lightgbm as lgb
    g = GBM()
    g.m = lgb.Booster(model_str=md["model_string"])
    return g


def run_holdout():
    if os.environ.get("GRADEV2_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked: set GRADEV2_HOLDOUT=I_HAVE_FROZEN after freezing")
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
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"],
           "seasons": list(HOLD), "eval": evaluate(C, pred, th, strat, len(HOLD)),
           "eval_base_ev_ml_sharp": evaluate(C, C.ev_p_ml_sharp.fillna(C.ev_cons_raw).values, th, strat, len(HOLD)),
           "existing_rules": existing_rules(X, len(HOLD))}
    # robustness: same bets, CLV vs Pinnacle close / spread-implied close; per-season by grade
    D = best_offers(C, pred, th)
    rob = {}
    for g in ORDER:
        B = first_with(D, D.grade == g)
        rob[g] = {"by_season": {int(s): summ(d, 1) for s, d in B.groupby("season")}}
    res["by_grade_by_season"] = rob
    ev = res["eval"]
    bg = ev["by_grade_first_appearance"]
    rho = ev["letters_spearman"]["rho"]
    top = bg["A+"]
    res["pass"] = {"monotone": bool(rho is not None and rho >= 0.8 and bg["A+"].get("clv", -1) ==
                                    max(bg[g].get("clv", -9) for g in ORDER if bg[g]["bets"])),
                   "top_grade": bool(top.get("bets", 0) > 2 and top["clv"] > 0 and top["clv_p_one_sided"] < 0.05)}
    res["pass"]["adopt"] = res["pass"]["monotone"] and res["pass"]["top_grade"]
    store("holdout", res)
    print(json.dumps(res, indent=1, default=_jd)[:20000])


def run_posthoc():
    """NOT pre-registered (run after the one-shot holdout, descriptive only): does A+ add anything beyond the live v2
    rule? Overlap of S1 (A+) and v2 bets on 2023-25, and the grade of each v2 bet's own offer when v2 placed it."""
    if os.environ.get("GRADEV2_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("set GRADEV2_HOLDOUT=I_HAVE_FROZEN")
    J = json.loads(JSON.read_text())
    if "holdout" not in J:
        raise SystemExit("run holdout first")
    fz = json.loads(FROZEN.read_text())
    m, th = load_frozen_model(fz), fz["thresholds_pred_clv"]
    X = table(HOLD)
    C = candidates(X)
    C["pred"] = m.predict(featurize(C))
    D = best_offers(C, C.pred.values, th)
    A = first_with(D, D.pred >= th["A+"])
    base = X.ml.between(-1000, 1000) & (X.ml_gap_cons <= 0.12) & X.clv.notna()
    V = X[base & (X.ev_p_ml_sharp >= 0.02) & (X.ev_model_raw >= 0)].sort_values("ev_p_ml_sharp", ascending=False)
    V = V.drop_duplicates(["game_id", "requested_ts"]).sort_values("requested_ts").drop_duplicates("game_id")
    both = set(A.game_id) & set(V.game_id)
    q = lambda d: summ(d.assign(pred=d.get("pred", np.nan)), len(HOLD))  # noqa: E731
    res = {"note": "post-hoc, descriptive; not part of the pre-registered test",
           "Aplus_and_v2_games": len(both), "Aplus_not_v2": q(A[~A.game_id.isin(both)]),
           "Aplus_also_v2": q(A[A.game_id.isin(both)]), "v2_not_Aplus": q(V[~V.game_id.isin(both)])}
    Vp = V.merge(C[["game_id", "requested_ts", "book", "side", "pred"]], on=["game_id", "requested_ts", "book", "side"],
                 how="left")
    Vp["g"] = np.where(Vp.pred.isna(), "not_candidate", letter(Vp.pred.fillna(-1).values, th))
    res["v2_bets_by_grade_of_their_offer"] = {g: q(d) for g, d in Vp.groupby("g")}
    store("posthoc", res)
    print(json.dumps(res, indent=1, default=_jd)[:6000])


# ------------------------------------------------------------------------------------------------ report
def _pct(x, se=None):
    if x is None:
        return "n/a"
    return f"{100 * x:+.2f}%" + (f" ± {100 * se:.2f}" if se is not None else "")


def _row(name, s):
    if not s or not s.get("bets"):
        return f"| {name} | 0 | | | | | | |"
    return (f"| {name} | {s['bets']} ({s['per_season']}/season) | {_pct(s.get('pred_clv'))} | {_pct(s['clv'], s['clv_se'])} | "
            f"{s.get('clv_p_one_sided')} | {100 * s['beat_close']:.0f}% | {_pct(s['roi'], s['roi_se_boot'])} | "
            f"{_pct(s.get('clv_pinnacle'), s.get('clv_pinnacle_se'))} (n={s.get('n_pin', 0)}) |")


HEAD = ("| | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± boot SE | CLV vs Pinnacle close |\n"
        "|---|---|---|---|---|---|---|---|")


INTERP = """* The pre-registered bar is met: realized holdout CLV rises with the letter (A+ > A > B > C, rho 1.0; decile rho
  0.94) and A+ CLV is positive at p < 0.01, confirmed by the Pinnacle close on 2024-25. Grade v1 never ranked bets;
  grade v2 does, so it should replace v1 as the DISPLAYED grade (GRADING_VERSION 2).
* But the separation is at the ends only: A and B realized ~0% CLV on 2023-25 (dev: +1.5% / +0.6%). Only A+ is a
  bet; A and B are 'lean / no bet'. 'Grade-scaled stakes' (S3) and 'A and up' (S2) dilute A+ with ~zero-CLV bets.
* A+ is not a better bet selector than the live v2 rule: S1 = CLV +2.3% on 54/season vs v2 +2.5% on 48/season
  (same holdout, v2 SE smaller). Post-hoc, half the A+ bets are v2 bets and those carry the edge; A+ bets outside v2
  were +1.3% ± 1.8. The model's top feature is the same EV-vs-sharp-ML that defines v2 (gain 52%), then price level
  (18%) and the sharp-spread EV of v4 (12%). Line move, hours-to-kick, model EV, best-price gap add a little;
  key number, dispersion, book and the signal count add nothing.
* ROI by grade is noise at these sample sizes (A+ ROI +22% ± 15); judge by CLV. Do not add a new bet track from this
  and do not use the grade to veto v2/v3/v4 bets: post-hoc, v2 bets on games A+ never flagged still had positive
  CLV (+2.1% ± 1.1; the C-graded ones were +650 longshots, +2.8% ± 1.9). Use grade v2 as the label / confidence
  display, and A+ as the only 'bet' letter if it is ever used for selection.
* Caveats: 2023-25 was the holdout of earlier studies; Pinnacle could not be a model input (no 2020-22 data); the
  historical QB-change flag uses the actual starter and was dropped as leakage; spread offers were not graded.
"""


def run_report():
    J = json.loads(JSON.read_text())
    fz = json.loads(FROZEN.read_text())
    Dv, H = J["dev"], J["holdout"]
    md = fz["model"]
    L = ["# Grade v2: predicted CLV as the bet grade (moneyline)", "",
         "Code `scripts/research/grade_v2.py` (dev -> freeze -> holdout once -> report). Numbers `grade_v2.json`; frozen "
         f"model, thresholds and strategies `grade_v2_frozen.json` (frozen {fz['frozen_at']}, before the 2023-25 run). "
         "Caveat: 2023-25 was already the holdout of earlier studies (edge_lab, ml_spread_consistency -> v4); nothing "
         "here was tuned on it, but the v4 feature itself was proposed after earlier dev work on 2020-22.", "",
         f"**Verdict: {'ADOPT' if H['pass']['adopt'] else 'DO NOT ADOPT'} grade v2** (pre-registered bar: realized CLV "
         f"rising with grade [{H['pass']['monotone']}] and A+ CLV > 0 at one-sided p < 0.05 [{H['pass']['top_grade']}]).", "",
         "## Model", "",
         f"Candidates: {Dv['candidate_rows']} offer rows / {Dv['games']} games (dev). LOSO (2020/21/22) game-weighted "
         f"MSE, R² vs constant: " + ", ".join(f"{k} {v['r2_vs_const']}" for k, v in Dv["cv"].items()) +
         f". Chosen: **{Dv['chosen_model']}** ({CHOICE_RULE}).", ""]
    if md["kind"] != "ridge" and "gain" in Dv.get("fit_all_dev", {}):
        L += ["Feature importance (share of total split gain, fit on 2020-22): " + ", ".join(
            f"{f} {100 * g:.1f}%" for f, g in sorted(Dv["fit_all_dev"]["gain"].items(), key=lambda kv: -kv[1]) if g > 0)
            + "; all other features 0. Production: `lightgbm.Booster(model_str=frozen['model']['model_string'])`, "
              "features in `feature_order`.", ""]
    if md["kind"] == "ridge":
        L += ["| feature | coef per 1 SD (CLV) | definition |", "|---|---|---|"]
        for f, d in sorted(md["features"].items(), key=lambda kv: -abs(kv[1]["coef_std"])):
            L.append(f"| {f} | {100 * d['coef_std']:+.3f}% | {FEATURES[f]} |")
    L += ["", f"Thresholds (pred CLV): A+ >= {_pct(fz['thresholds_pred_clv']['A+'])}, A >= "
          f"{_pct(fz['thresholds_pred_clv']['A'])}, B >= {_pct(fz['thresholds_pred_clv']['B'])}, else C.", ""]
    for lab, E, rules in (("Dev 2020-22 (out-of-fold predictions)", Dv["oof_eval_chosen"], Dv["existing_rules_dev"]),
                          ("HOLDOUT 2023-25 (frozen model, run once)", H["eval"], H["existing_rules"])):
        L += [f"## {lab}", "", "By grade (one bet per game per grade, first snapshot where the game's best offer has that grade):", "", HEAD]
        for g in ORDER:
            L.append(_row(g, E["by_grade_first_appearance"][g]))
        L += ["", f"Spearman across letters: rho {E['letters_spearman']['rho']}; across pred-CLV deciles (snapshot best "
              f"offers): rho {E['deciles_spearman']['rho']} (p {E['deciles_spearman']['p']}); offer-level rank corr "
              f"{E['offer_spearman_pred_vs_clv']}; calibration slope {E['calibration_slope']}.", "",
              "Deciles (pred -> realized CLV): " + ", ".join(f"{_pct(d['pred'])}->{_pct(d['clv'])}" for d in E["deciles"]), "",
              "Strategies and existing rules:", "", HEAD]
        for k, s in E["strategies"].items():
            L.append(_row(k, s))
        for k, s in rules.items():
            L.append(_row(f"rule {k}", s))
        L.append("")
    E0 = H["eval_base_ev_ml_sharp"]
    L += ["Holdout baseline -- same grade cuts applied to raw EV vs sharp no-vig ML (no model):", "", HEAD]
    for g in ORDER:
        L.append(_row(g, E0["by_grade_first_appearance"][g]))
    L += ["", "Holdout by grade and season (CLV):", ""]
    for g in ORDER:
        L.append(f"* {g}: " + ", ".join(f"{s}: {_pct(v.get('clv'))} ({v.get('bets')})"
                                        for s, v in H["by_grade_by_season"][g]["by_season"].items()))
    L += ["", "Strategies (frozen):", ""] + [f"* **{s['id']}** -- {s['desc']}" for s in fz["strategies"]]
    P = J.get("posthoc")
    if P:
        q = lambda d: f"{d['bets']} bets, CLV {_pct(d['clv'], d['clv_se'])}, ROI {_pct(d['roi'], d['roi_se_boot'])}"  # noqa: E731
        L += ["", "## Post-hoc (after the one-shot holdout; descriptive, not part of the test): A+ vs the live v2 rule", "",
              f"* A+ bets that v2 also took ({P['Aplus_and_v2_games']} games): {q(P['Aplus_also_v2'])}",
              f"* A+ bets v2 did not take: {q(P['Aplus_not_v2'])}",
              f"* v2 bets that were not A+ bets: {q(P['v2_not_Aplus'])}",
              "* v2 bets by the grade of their own offer at bet time: " + "; ".join(
                  f"{g}: {q(d)}" for g, d in P["v2_bets_by_grade_of_their_offer"].items())]
    L += ["", "## Reading", "", INTERP]
    MD.write_text("\n".join(L) + "\n")
    print(MD.read_text())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": run_dev, "freeze": run_freeze, "holdout": run_holdout, "posthoc": run_posthoc,
     "report": run_report}[stage]()
