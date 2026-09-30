"""Line-movement model: predict where the market will CLOSE relative to an early snapshot, bet early
on the side it is predicted to move toward (best allowed-book price), judge by price-based CLV.

Discipline (same as scripts/edge_lab.py):
  * develop on 2020-2022 only (leave-one-season-out CV + a forward-only check);
  * freeze <= 3 strategies (features, coefficients, thresholds) into output/research/line_move_frozen.json;
  * run 2023-2025 exactly once (`--holdout`, needs EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES) and report all.

Targets (per game snapshot, home perspective):
  y_sp     = mu_close_all - mu_px_all   close vs NOW, both price-implied (key-number model, incl. juice)
  y_sp_raw = mu_close_all - m_cons      as literally specified (mixes in the juice lean of the snapshot)
  y_ml     = logit(p_close_all) - logit(p_cons)
CLV of a bet = EV of that exact point/price valued at the honest close (E.spread_clv_price logic,
vectorised; moneyline: dec * p_close_side - 1 with p_close_all).

    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/line_move_model.py            # dev
    PYTHONPATH=src:scripts python scripts/research/line_move_model.py --freeze
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/line_move_model.py --holdout
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

import edge_lab as E
import replay_early_lines as R
from nflpred import margins as K, spread_bets as SB
from nflpred.weather import _kickoff_utc

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "research"
FROZEN = OUT / "line_move_frozen.json"
RESULTS = OUT / "line_move_model.json"
SCRATCH = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/linemove")
RULES = SB.load_rules()
SIG, W = RULES["margin"]["sigma"], RULES["_weights"]
PUBLIC = {"DAL", "KC", "GB", "PIT", "NE", "SF", "PHI"}
MIN_H, MAX_H = 3.0, 216.0            # snapshots strictly before the close, within 9 days of kickoff

# ---- model feature sets (home-signed: + = home expected margin / home prob rises) ----
SP_LIN = ["x_sharp", "x_sharp_pt", "x_moved", "x_model", "x_skew", "x_juice", "x_fav", "x_key3", "x_key7",
          "x_pub", "x_div_fav", "x_sharp_h", "x_moved_h", "x_model_h", "x_skew_h", "x_early"]
ML_LIN = ["z_sharp", "z_moved", "z_model", "z_fav", "z_pub", "z_sharp_h", "z_moved_h", "z_model_h",
          "x_sharp", "x_skew", "x_early"]
CTX = ["hours", "absm", "disp", "week", "div", "slot", "n_books", "eligible_f"]

# Sparse, robust candidates chosen after the dev grid (wide ridge/GBM overfit the fat-tailed target):
# ridge alpha=1 on a winsorised target, bet when EV of the best allowed-book offer valued at the
# PREDICTED close (snapshot price-implied consensus + predicted move) >= thr; first qualifying snapshot.
CANDIDATES = {
    "LM1_spread_sharp_lead": {"market": "sp", "target": "y_sp", "clip": 3.0, "alpha": 1.0, "model": "ridge",
                              "features": ["x_sharp"], "rule": "ev", "thr": 0.02, "min_h": MIN_H, "max_h": MAX_H},
    "LM2_ml_sharp_fav": {"market": "ml", "target": "y_ml", "clip": 0.3, "alpha": 1.0, "model": "ridge",
                         "features": ["z_sharp", "z_fav", "x_sharp"], "rule": "ev", "thr": 0.03,
                         "min_h": MIN_H, "max_h": MAX_H},
    "LM3_spread_sharp_model": {"market": "sp", "target": "y_sp", "clip": 3.0, "alpha": 1.0, "model": "ridge",
                               "features": ["x_sharp", "x_model"], "rule": "ev", "thr": 0.01,
                               "min_h": MIN_H, "max_h": MAX_H},
}


# ================================================================== data
def _q_table(points):
    grid = K._GRID
    tab = {}
    for p in np.unique(points[~np.isnan(points)]):
        hc, pu, ac = K.cover_probs(grid, SIG, np.full_like(grid, float(p)), W)
        tab[p] = hc / np.maximum(hc + ac, 1e-12)
    return grid, tab


def implied_mu_vec(home_point, home_price, away_price):
    """Vectorised margins.implied_mu over rows (NaN where unusable)."""
    hp, hpr, apr = (np.asarray(x, float) for x in (home_point, home_price, away_price))
    ih = np.where(hpr < 0, -hpr / (-hpr + 100), 100 / (hpr + 100))
    ia = np.where(apr < 0, -apr / (-apr + 100), 100 / (apr + 100))
    q = ih / (ih + ia)
    out = np.full(len(hp), np.nan)
    ok = ~(np.isnan(hp) | np.isnan(q))
    grid, tab = _q_table(hp[ok])
    idx = np.where(ok)[0]
    for p, curve in tab.items():
        m = idx[hp[idx] == p]
        out[m] = np.interp(q[m], curve, grid)
    return out


def side_ev_vec(mu_home, point, price, side):
    """Vectorised SB.side_ev: EV per unit of `side` at its own `point` and American `price`."""
    mu_home, point, price = (np.asarray(x, float) for x in (mu_home, point, price))
    home = np.asarray(side) == "home"
    home_line = np.where(home, point, -point)
    out = np.full(len(point), np.nan)
    ok = ~(np.isnan(mu_home) | np.isnan(point) | np.isnan(price))
    if ok.any():
        hc, pu, ac = K.cover_probs(mu_home[ok], SIG, home_line[ok], W)
        pw = np.where(home[ok], hc, ac)
        dec = np.where(price[ok] > 0, 1 + price[ok] / 100, 1 + 100 / -price[ok])
        out[ok] = pw * dec + pu - 1
    return out


def snapshot_features(seasons) -> pd.DataFrame:
    """Per (game_id, requested_ts): price-implied consensus/sharp margins and cross-book shape."""
    cache = SCRATCH / f"snapfeat_{min(seasons)}_{max(seasons)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    wf = R.walk_forward()
    wf = wf[wf.season.isin(seasons)].copy()
    wf["gameday"] = pd.to_datetime(wf.gameday)
    wf["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in wf.itertuples()], utc=True)
    o = R.load_odds()
    o = R.match_games(o[o.season.isin(seasons)], wf).merge(wf[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick].copy()
    o["mu_px"] = implied_mu_vec(o.sp_home_point, o.sp_home_price, o.sp_away_price)
    o["hm"] = -o.sp_home_point
    key = ["game_id", "requested_ts"]
    a = o.groupby(key).agg(mu_px_all=("mu_px", "median"), m_mean=("hm", "mean"), disp=("hm", "std"))
    s = o[o.book.isin(E.SHARP)].groupby(key).agg(mu_px_sharp=("mu_px", "median"))
    f = a.join(s).reset_index()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    f.to_parquet(cache)
    return f


def build(t: pd.DataFrame, seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (snapshot table with features/targets, offer table of allowed-book rows)."""
    close = E.closing_fair(seasons)[["game_id", "mu_close_all", "p_close_all", "mu_close_sharp", "p_close_sharp"]]
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")[["game_id", "div_game"]]
    sc = ["game_id", "requested_ts", "season", "week", "weekday", "kick", "home_team", "away_team", "m_cons",
          "m_sharp", "p_cons", "p_sharp", "m_first", "p_first", "hours_before", "eligible", "mu_model", "p_model",
          "n_books"]
    s = t.drop_duplicates(["game_id", "requested_ts"])[sc].merge(snapshot_features(seasons), on=["game_id", "requested_ts"])
    s = s.merge(close, on="game_id").merge(g, on="game_id", how="left")
    s = s[(s.hours_before >= MIN_H) & (s.hours_before <= MAX_H)].copy()
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    # targets
    s["y_sp"] = s.mu_close_all - s.mu_px_all
    s["y_sp_raw"] = s.mu_close_all - s.m_cons
    s["y_ml"] = lg(s.p_close_all) - lg(s.p_cons)
    # features: spread (points, home-signed)
    h = (s.hours_before / 100.0).clip(upper=2.16)
    s["hours"], s["eligible_f"] = s.hours_before, s.eligible.astype(float)
    s["x_sharp"] = (s.mu_px_sharp - s.mu_px_all).fillna(0.0)
    s["x_sharp_pt"] = (s.m_sharp - s.m_cons).fillna(0.0)
    s["x_moved"] = (s.m_cons - s.m_first).fillna(0.0)
    s["x_model"] = np.where(s.eligible, s.mu_model - s.m_cons, 0.0)
    s["x_skew"] = (s.m_mean - s.m_cons).fillna(0.0)
    s["x_juice"] = s.mu_px_all - s.m_cons
    s["x_fav"] = s.m_cons
    am = s.m_cons.abs()
    s["absm"] = am
    s["x_key3"] = np.where(am.between(1.5, 4.5), np.sign(s.m_cons) * (3 - am), 0.0)
    s["x_key7"] = np.where(am.between(5.5, 8.5), np.sign(s.m_cons) * (7 - am), 0.0)
    s["x_pub"] = s.home_team.isin(PUBLIC).astype(float) - s.away_team.isin(PUBLIC).astype(float)
    s["div"] = s.div_game.fillna(0).astype(float)
    s["x_div_fav"] = s["div"] * s.m_cons
    for c in ("sharp", "moved", "model", "skew"):
        s[f"x_{c}_h"] = s[f"x_{c}"] * h
    s["x_early"] = h                      # home drift by time left (intercept handles the constant)
    s["slot"] = s.weekday.map({"Thursday": 0, "Sunday": 1, "Monday": 2, "Saturday": 3}).fillna(4).astype(float)
    s["disp"] = s.disp.fillna(0.0)
    # features: moneyline (logit, home-signed)
    s["z_sharp"] = (lg(s.p_sharp) - lg(s.p_cons)).fillna(0.0)
    s["z_moved"] = (lg(s.p_cons) - lg(s.p_first)).fillna(0.0)
    s["z_model"] = np.where(s.eligible, lg(s.p_model) - lg(s.p_cons), 0.0)
    s["z_fav"] = lg(s.p_cons)
    s["z_pub"] = s.x_pub
    for c in ("sharp", "moved", "model"):
        s[f"z_{c}_h"] = s[f"z_{c}"] * h
    s = s[s.y_sp.notna() & s.y_ml.notna() & s.mu_px_all.notna()].reset_index(drop=True)
    # ---- offers at allowed books (sanity filter as in edge_lab._prep: no stale / off-market lines)
    oc = ["game_id", "requested_ts", "book", "side", "point", "sp_price", "ml", "ml_dec", "result_margin",
          "sp_pnl", "ml_pnl", "p_cons_side"]
    off = t[oc].merge(s[["game_id", "requested_ts", "season", "hours_before", "m_cons", "mu_px_all", "p_cons",
                          "mu_close_all", "p_close_all", "eligible"]], on=["game_id", "requested_ts"])
    sgn = np.where(off.side == "home", 1.0, -1.0)
    off["pt_off"] = off.point - np.where(off.side == "home", -off.m_cons, off.m_cons)
    off["sp_ok"] = off.point.notna() & off.sp_price.between(-200, 200) & (off.pt_off.abs() <= 2.5)
    off["ml_ok"] = off.ml.notna() & off.ml.between(-1000, 1000) & ((1 / off.ml_dec - off.p_cons_side).abs() <= 0.12)
    off["sp_clv"] = side_ev_vec(off.mu_close_all, off.point, off.sp_price, off.side)
    off["sp_ev_now"] = side_ev_vec(off.mu_px_all, off.point, off.sp_price, off.side)
    off["p_close_side"] = np.where(off.side == "home", off.p_close_all, 1 - off.p_close_all)
    off["ml_clv"] = off.ml_dec * off.p_close_side - 1
    off["ml_ev_now"] = off.ml_dec * off.p_cons_side - 1
    off["sgn"] = sgn
    return s, off


# ================================================================== models
def make_model(kind):
    if kind == "ridge":
        return Ridge(alpha=10.0)
    return HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.04, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)


def feats(kind, market):
    lin = SP_LIN if market == "sp" else ML_LIN
    return lin if kind == "ridge" else lin + CTX


def fit(s, kind, market, target):
    f = feats(kind, market)
    m = make_model(kind).fit(s[f].values, s[target].values)
    return m, f


def oof(s, kind, market, target, forward_only=False):
    pred = np.full(len(s), np.nan)
    for yr in sorted(s.season.unique()):
        tr = s[(s.season < yr)] if forward_only else s[s.season != yr]
        if len(tr) == 0:
            continue
        m, f = fit(tr, kind, market, target)
        te = s.season == yr
        pred[te.values] = m.predict(s.loc[te, f].values)
    return pred


def fit_spec(tr, c):
    y = tr[c["target"]].clip(-c["clip"], c["clip"]) if c.get("clip") else tr[c["target"]]
    return Ridge(alpha=c["alpha"]).fit(tr[c["features"]].values, y.values)


def oof_spec(s, c, forward_only=False):
    pred = np.full(len(s), np.nan)
    for yr in sorted(s.season.unique()):
        tr = s[s.season < yr] if forward_only else s[s.season != yr]
        if len(tr):
            te = (s.season == yr).values
            pred[te] = fit_spec(tr, c).predict(s.loc[te, c["features"]].values)
    return pred


def fit_stats(y, p, games):
    ok = ~np.isnan(p)
    y, p, games = y[ok], p[ok], games[ok]
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r = np.corrcoef(y, p)[0, 1]
    # game-clustered correlation SE via per-game mean products is overkill: report n games instead
    return {"r2": round(float(r2), 4), "corr": round(float(r), 4), "n_snaps": int(ok.sum()),
            "n_games": int(pd.Series(games).nunique())}


# ================================================================== strategies
def select(s, off, pred, market, rule, thr, min_h=MIN_H, max_h=MAX_H, ev_floor=None):
    """rule 'move': predicted move toward a side >= thr (points or logit); bet that side at best allowed price
    rule 'ev'  : predicted EV of the offer valued at the predicted close >= thr (move + price together).
    One bet per game at the first qualifying snapshot."""
    p = s[["game_id", "requested_ts"]].assign(yhat=pred)
    o = off[off[f"{market}_ok"] & off.hours_before.between(min_h, max_h)].merge(p, on=["game_id", "requested_ts"])
    o = o[o.yhat.notna()]
    if market == "sp":
        o["mv"] = o.sgn * o.yhat
        o["pev"] = side_ev_vec(o.mu_px_all + o.yhat, o.point, o.sp_price, o.side)
    else:
        lg = np.log(o.p_cons / (1 - o.p_cons)) + o.yhat
        pc = 1 / (1 + np.exp(-lg))
        o["mv"] = o.sgn * o.yhat
        o["pev"] = o.ml_dec * np.where(o.side == "home", pc, 1 - pc) - 1
    q = o[o.mv >= thr] if rule == "move" else o[o.pev >= thr]
    if ev_floor is not None:
        q = q[q[f"{market}_ev_now"] >= ev_floor]
    q = q.sort_values(["game_id", "requested_ts", "pev"], ascending=[True, True, False])
    return q.groupby("game_id").head(1)


def bstats(b, market):
    if len(b) < 3:
        return {"bets": int(len(b))}
    clv, pnl = b[f"{market}_clv"].values, b[f"{market}_pnl"].values
    t = clv.mean() / (clv.std(ddof=1) / math.sqrt(len(clv)))
    return {"bets": int(len(b)), "per_season": round(len(b) / b.season.nunique(), 1),
            "clv": round(float(clv.mean()), 4), "clv_t": round(float(t), 2),
            "clv_p": round(float(0.5 * math.erfc(t / math.sqrt(2))), 5),
            "beat_close": round(float((clv > 0).mean()), 3),
            "roi": round(float(pnl.mean()), 4), "roi_se": round(float(pnl.std(ddof=1) / math.sqrt(len(pnl))), 4),
            "median_hours": round(float(b.hours_before.median()), 1)}


def by_season(b, market):
    return {int(y): bstats(g, market) for y, g in b.groupby("season")}


def baselines(off):
    """Reference points: every eligible offer, the best-priced side at the first early snapshot, sharp-follow."""
    out = {}
    for mk in ("sp", "ml"):
        o = off[off[f"{mk}_ok"]]
        out[f"{mk}_all_offers_mean_clv"] = round(float(o[f"{mk}_clv"].mean()), 4)
    return out


# ================================================================== main modes
def dev():
    t = E.load(False)
    s, off = build(t, E.DEV)
    res = {"n_snaps": len(s), "n_games": int(s.game_id.nunique()), "fit": {}, "coef": {}, "strategies": {}}
    print(f"snapshots {len(s)}  games {s.game_id.nunique()}")
    print("target sd: y_sp %.3f  y_sp_raw %.3f  y_ml %.4f" % (s.y_sp.std(), s.y_sp_raw.std(), s.y_ml.std()))
    # univariate correlations (home-signed)
    uni = {}
    for tgt, cols in (("y_sp", SP_LIN), ("y_sp_raw", SP_LIN), ("y_ml", ML_LIN)):
        uni[tgt] = {c: round(float(np.corrcoef(s[c], s[tgt])[0, 1]), 4) for c in cols}
    res["univariate_corr"] = uni
    print(json.dumps(uni, indent=0))
    preds = {}
    for market, tgt in (("sp", "y_sp"), ("sp", "y_sp_raw"), ("ml", "y_ml")):
        for kind in ("ridge", "gbm"):
            for fwd in (False, True):
                p = oof(s, kind, market, tgt, forward_only=fwd)
                k = f"{tgt}|{kind}|{'fwd' if fwd else 'loso'}"
                res["fit"][k] = fit_stats(s[tgt].values, p, s.game_id.values)
                # by-horizon R^2 (loso only)
                if not fwd:
                    preds[(tgt, kind)] = p
                    hz = pd.cut(s.hours_before, [3, 24, 72, 216])
                    res["fit"][k]["by_hours"] = {str(b): fit_stats(s[tgt].values[(hz == b).values], p[(hz == b).values],
                                                                   s.game_id.values[(hz == b).values]) for b in hz.cat.categories}
                print(k, res["fit"][k] if fwd else {kk: v for kk, v in res["fit"][k].items() if kk != "by_hours"})
    for market, tgt in (("sp", "y_sp"), ("ml", "y_ml")):
        m, f = fit(s, "ridge", market, tgt)
        res["coef"][tgt] = {"intercept": round(float(m.intercept_), 4), **{c: round(float(v), 4) for c, v in zip(f, m.coef_)}}
        print(tgt, res["coef"][tgt])
    # strategy grid on out-of-fold predictions
    res["baselines"] = baselines(off)
    grid = []
    for (tgt, kind), p in preds.items():
        market = "ml" if tgt == "y_ml" else "sp"
        if tgt == "y_sp_raw":
            continue
        thr_move = [0.25, 0.5, 0.75, 1.0] if market == "sp" else [0.02, 0.04, 0.06, 0.08]
        thr_ev = [0.0, 0.01, 0.02, 0.03]
        for hwin in ((MIN_H, MAX_H), (24, MAX_H), (48, MAX_H)):
            for rule, ths in (("move", thr_move), ("ev", thr_ev)):
                for th in ths:
                    b = select(s, off, p, market, rule, th, *hwin)
                    st = bstats(b, market)
                    st.update(target=tgt, model=kind, rule=rule, thr=th, min_h=hwin[0])
                    grid.append(st)
    g = pd.DataFrame(grid)
    res["strategies"] = grid
    pd.set_option("display.width", 200)
    print(g.sort_values("clv_t", ascending=False).to_string())
    # sparse candidates (LOSO as selected + forward-only check) and a zero-move line-shopping control
    res["candidates"] = {}
    for cid, c in CANDIDATES.items():
        m = fit_spec(s, c)
        d = {"coef_all_dev": dict(zip(c["features"], np.round(m.coef_, 4).tolist())), "intercept": round(float(m.intercept_), 4)}
        for lab, fwd in (("loso", False), ("fwd", True)):
            p = oof_spec(s, c, fwd)
            b = select(s, off, p, c["market"], c["rule"], c["thr"], c["min_h"], c["max_h"])
            d[lab] = {"fit": fit_stats(s[c["target"]].values, p, s.game_id.values), **bstats(b, c["market"]),
                      "by_season": by_season(b, c["market"])}
            b0 = select(s, off, np.zeros(len(s)), c["market"], c["rule"], c["thr"], c["min_h"], c["max_h"])
            d["zero_move_control"] = bstats(b0, c["market"])
        res["candidates"][cid] = d
        print(cid, json.dumps(d))
    RESULTS.write_text(json.dumps({"dev": res}, indent=1, default=str))
    return s, off, preds


def _pred_hash(p):
    return hashlib.sha256(np.round(p, 6).tobytes()).hexdigest()[:16]


def freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists; frozen candidates are not edited")
    spec = CANDIDATES
    t = E.load(False)
    s, off = build(t, E.DEV)
    out = {"frozen_at": pd.Timestamp.now("UTC").isoformat(), "trained_on": list(E.DEV),
           "pass_bar": "mean price-based CLV > 0 and one-sided p < 0.05/3", "strategies": {}}
    for cid, c in spec.items():
        m = fit_spec(s, c)
        entry = dict(c)
        entry["intercept"] = float(m.intercept_)
        entry["coef"] = {k: float(v) for k, v in zip(c["features"], m.coef_)}
        b = select(s, off, oof_spec(s, c), c["market"], c["rule"], c["thr"], c["min_h"], c.get("max_h", MAX_H))
        entry["dev_oof"] = bstats(b, c["market"])
        out["strategies"][cid] = entry
    FROZEN.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def _frozen_predict(entry, s_dev, s_hold):
    """Frozen linear model applied as stored (no refit). s_dev is used only to verify the coefficients."""
    f = entry["features"]
    chk = fit_spec(s_dev, entry)
    if not np.allclose(chk.coef_, [entry["coef"][k] for k in f], atol=1e-8):
        raise SystemExit("dev refit does not reproduce the frozen coefficients")
    return entry["intercept"] + s_hold[f].values @ np.array([entry["coef"][k] for k in f])


def holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout locked: freeze first")
    prev = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if "holdout" in prev:
        raise SystemExit("holdout already run once; results are in " + str(RESULTS))
    fz = json.loads(FROZEN.read_text())
    s_dev, _ = build(E.load(False), E.DEV)
    s, off = build(E.load(True), E.HOLD)
    res = {"frozen_file": str(FROZEN), "n_snaps": len(s), "n_games": int(s.game_id.nunique()), "strategies": {}, "fit": {}}
    for cid, c in fz["strategies"].items():
        p = _frozen_predict(c, s_dev, s)
        res["fit"][cid] = fit_stats(s[c["target"]].values, p, s.game_id.values)
        b = select(s, off, p, c["market"], c["rule"], c["thr"], c["min_h"], c.get("max_h", MAX_H))
        st = bstats(b, c["market"])
        st["pass"] = bool(st.get("clv", 0) > 0 and st.get("clv_p", 1) < 0.05 / 3)
        st["by_season"] = by_season(b, c["market"])
        res["strategies"][cid] = st
        print(cid, res["fit"][cid], json.dumps(st))
        b0 = select(s, off, np.zeros(len(s)), c["market"], c["rule"], c["thr"], c["min_h"], c.get("max_h", MAX_H))
        res["strategies"][cid]["zero_move_control"] = bstats(b0, c["market"])
    res["baselines"] = baselines(off)
    prev["holdout"] = res
    RESULTS.write_text(json.dumps(prev, indent=1, default=str))


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    elif "--holdout" in sys.argv:
        holdout()
    else:
        dev()
