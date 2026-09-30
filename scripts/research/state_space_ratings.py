"""Research: state-space (Kalman filter) team ratings.

Model (per team i, latent strength theta_i in points):
    within a season   theta_t = theta_{t-1} + eta,        eta ~ N(0, q_w * weeks_elapsed)
    offseason         theta   <- rho * theta,  P <- rho^2 P + q_off * I
    observation       y_g = theta_home - theta_away + hfa*home_field [+ beta*(qb_home - qb_away)] + eps

y_g is the (optionally capped) point margin; variant "epa" instead observes a composite of the
margin and the game's net EPA/play margin (scaled to points).  Two noisy observations of the same
linear combination h'theta with correlated errors are exactly equivalent to one GLS-weighted
composite observation, so the bivariate model is parameterized as z = w*margin + (1-w)*k*epa.
Variant "qb" removes the starting QB's (pre-game, leak-free) rating from the team rating and adds
the current starter's rating back at prediction time.

All teams share one full 32x32 covariance (ratings are identified only up to differences; the
filter keeps that structure).  Pre-game ratings for every game on a date are read BEFORE any game
on that date is used to update, so a game's rating uses only games with an earlier `gameday`.

Hyperparameters are fit by minimizing one-step-ahead win log loss (p = Phi(mu / sqrt(h'Ph + s_p^2)))
on 2012-2019 games only.  Evaluation: validation 2015-2019, holdout 2020-2025.

    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/state_space_ratings.py
Writes output/research/state_space_ratings.{md,json}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from nflpred import data, features as F, model as M  # noqa: E402

HIST_START = 2002           # schedules go back to 1999; 32-team era from 2002 (warm-up only)
TUNE = (2012, 2019)         # hyperparameters use only these seasons
VAL, HOLD = range(2015, 2020), range(2020, 2026)
SIGMA_U = 13.0              # margin observation noise (fixes the overall variance scale)
P0 = 25.0                   # initial rating variance (2002; irrelevant by 2012)
OUT = ROOT / "output" / "research"


# ---------------------------------------------------------------- inputs
def game_table(games: pd.DataFrame, pbp: pd.DataFrame | None) -> pd.DataFrame:
    """Schedule from HIST_START with margin, net-EPA margin and pre-game starter QB ratings.
    Everything here is either a game's own result (used only to update AFTER its date) or a
    pre-game quantity computed by the production (leak-tested) QB code."""
    g = games[games["season"] >= HIST_START].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(F.TEAM_MAP)
    g["gameday"] = pd.to_datetime(g["gameday"])
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    g["margin"] = (g["home_score"] - g["away_score"]).astype(float)
    g["home_field"] = (g["location"] != "Neutral").astype(float)
    g["epa_margin"] = np.nan
    g["qb_d"] = 0.0
    if pbp is not None:
        tgs = F.team_game_stats(pbp)
        net = tgs.assign(net=tgs["off_epa"] - tgs["def_epa"])[["game_id", "team", "net"]]
        h = g[["game_id", "home_team"]].merge(net.rename(columns={"team": "home_team"}), how="left")["net"]
        a = g[["game_id", "away_team"]].merge(net.rename(columns={"team": "away_team"}), how="left")["net"]
        g["epa_margin"] = (h.values - a.values)
        sched = F.prepare_schedule(games)
        q = F.attach_qb(sched, F.qb_ratings(pbp, sched))[["game_id", "home_qb_rating", "away_qb_rating"]]
        g = g.drop(columns="qb_d").merge(q, on="game_id", how="left")
        g["qb_d"] = (g["home_qb_rating"] - g["away_qb_rating"]).fillna(0.0)
    return g


# ---------------------------------------------------------------- filter
PARAMS = {  # name: (transform, init)
    "q_w": ("log", 0.3), "q_off": ("log", 8.0), "rho": ("logit", 0.75), "hfa": ("id", 2.0),
    "s_p": ("log", 13.0), "w": ("logit", 0.5), "k": ("log", 30.0), "hfa_e": ("id", 2.0),
    "s_z": ("log", 10.0), "beta": ("id", 30.0),
}
VARIANT_PARAMS = {
    "margin": ["q_w", "q_off", "rho", "hfa", "s_p"],
    "epa": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "k", "hfa_e", "s_z"],
    "qb": ["q_w", "q_off", "rho", "hfa", "s_p", "beta"],
    "epa_qb": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "k", "hfa_e", "s_z", "beta"],
}


def _fwd(name, v):
    t = PARAMS[name][0]
    return np.log(v) if t == "log" else np.log(v / (1 - v)) if t == "logit" else v


def _inv(name, x):
    t = PARAMS[name][0]
    return float(np.exp(x)) if t == "log" else float(1 / (1 + np.exp(-x))) if t == "logit" else float(x)


def kalman(g: pd.DataFrame, p: dict, variant: str, cap: float | None) -> pd.DataFrame:
    """Pre-game ratings for every row of g (a game_table). Returns home/away team rating, the
    predicted margin mu (incl. HFA and QB term) and its predictive variance."""
    use_epa, use_qb = "epa" in variant, "qb" in variant
    beta = p.get("beta", 0.0) if use_qb else 0.0
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    m, P = np.zeros(n), np.eye(n) * P0
    H, A = g["home_team"].map(ix).values, g["away_team"].map(ix).values
    hf, qbd = g["home_field"].values, g["qb_d"].values * (1.0 if use_qb else 0.0)
    y = g["margin"].values.copy()
    if cap is not None:
        y = np.clip(y, -cap, cap)
    ep = g["epa_margin"].values
    done = g["completed"].values & ~np.isnan(y)
    seas, days = g["season"].values, g["gameday"].values.astype("datetime64[D]").astype(np.int64)
    out = np.zeros((len(g), 4))
    cur_season, last_day = None, None
    # rows grouped by date (g is sorted by gameday)
    bounds = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True])
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        if seas[s0] != cur_season:
            if cur_season is not None:
                m = p["rho"] * (m - m.mean())
                P = p["rho"] ** 2 * P + p["q_off"] * np.eye(n)
            cur_season, last_day = seas[s0], days[s0]
        else:
            P = P + p["q_w"] * (days[s0] - last_day) / 7.0 * np.eye(n)
            last_day = days[s0]
        idx = np.arange(s0, s1)
        h, a = H[idx], A[idx]
        out[idx, 0], out[idx, 1] = m[h], m[a]
        out[idx, 2] = m[h] - m[a] + p["hfa"] * hf[idx] + beta * qbd[idx]
        out[idx, 3] = P[h, h] + P[a, a] - 2 * P[h, a] + p["s_p"] ** 2
        for i in idx:  # sequential scalar updates == joint update (independent game noises)
            if not done[i]:
                continue
            hi, ai = H[i], A[i]
            base = m[hi] - m[ai] + beta * qbd[i]
            innov = y[i] - base - p["hfa"] * hf[i]
            r2 = SIGMA_U ** 2
            if use_epa and not np.isnan(ep[i]):
                innov_e = p["k"] * ep[i] - base - p["hfa_e"] * hf[i]
                innov = p["w"] * innov + (1 - p["w"]) * innov_e
                r2 = p["s_z"] ** 2
            Ph = P[:, hi] - P[:, ai]
            S = Ph[hi] - Ph[ai] + r2
            m = m + Ph * (innov / S)
            P = P - np.outer(Ph, Ph) / S
    res = g[["game_id"]].copy()
    res["ss_home"], res["ss_away"], res["ss_mu"], res["ss_var"] = out.T
    res["ss_diff"] = res["ss_home"] - res["ss_away"]          # team part only
    res["ss_diff_qb"] = res["ss_diff"] + beta * qbd            # + current starter's QB term
    return res


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def objective(x, names, g, variant, cap, mask, y):
    p = {k: _inv(k, v) for k, v in zip(names, x)}
    if p["q_w"] > 50 or p["q_off"] > 400 or ("s_z" in p and p["s_z"] > 100):
        return 10.0
    r = kalman(g, p, variant, cap)
    return _ll(y, norm.cdf(r["ss_mu"].values[mask] / np.sqrt(r["ss_var"].values[mask])))


def tune(g, variant, cap):
    names = VARIANT_PARAMS[variant]
    x0 = np.array([_fwd(k, PARAMS[k][1]) for k in names])
    win = np.where(g["completed"] & (g["margin"] != 0), (g["margin"] > 0).astype(float), np.nan)
    mask = g["season"].between(*TUNE).values & ~np.isnan(win)
    args = (names, g, variant, cap, mask, win[mask])
    r = minimize(objective, x0, args=args, method="Nelder-Mead",
                 options={"maxiter": 250 * len(names), "xatol": 1e-3, "fatol": 1e-6})
    return {k: _inv(k, v) for k, v in zip(names, r.x)}, float(r.fun)


# ---------------------------------------------------------------- leakage check
def leakage_check(games, pbp, params, variant, cap, cutoffs=("2019-11-10", "2023-10-01")):
    """Erase every result on/after `cutoff` (scores + that pbp); ratings for all games dated
    <= cutoff must not change.  Mirrors tests/test_pipeline.py::test_no_future_leakage."""
    full = kalman(game_table(games, pbp), params, variant, cap).set_index("game_id")
    out = {}
    for c in cutoffs:
        cut = pd.Timestamp(c)
        g2 = games.copy()
        fut = pd.to_datetime(g2["gameday"]) >= cut
        g2.loc[fut, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
        p2 = pbp[~pbp["game_id"].isin(set(g2.loc[fut, "game_id"]))]
        masked = kalman(game_table(g2, p2), params, variant, cap).set_index("game_id")
        gd = pd.to_datetime(games.set_index("game_id")["gameday"])
        ids = full.index[gd.reindex(full.index) <= cut]
        on_cut = int((gd.reindex(full.index) == cut).sum())
        cols = ["ss_home", "ss_away", "ss_mu", "ss_var", "ss_diff_qb"]
        diff = float(np.abs(full.loc[ids, cols].values - masked.loc[ids, cols].values).max())
        # sanity: ratings AFTER the cutoff must differ (the check has power)
        later = full.index[gd.reindex(full.index) > cut + pd.Timedelta(days=7)]
        later = later[:200]
        moved = float(np.abs(full.loc[later, "ss_mu"].values - masked.loc[later, "ss_mu"].values).max())
        assert diff < 1e-9, f"LEAK: {variant} ratings changed before {c} (max diff {diff})"
        assert moved > 1e-6, "leak check has no power"
        out[c] = {"games_checked": int(len(ids)), "games_on_cutoff": on_cut, "max_abs_diff": diff,
                  "max_change_after_cutoff": moved}
    return out


# ---------------------------------------------------------------- evaluation
def score(y, p):
    s = M.score(y, p)
    return {"ll": round(s["log_loss"], 4), "brier": round(s["brier"], 4), "n": s["n"]}


def walk_forward(df, feats):
    """Identical to scripts/experiment.py::run: for each test season, fit on prior seasons."""
    pr = pd.Series(np.nan, index=df.index)
    for s in list(VAL) + list(HOLD):
        tr = M.train_rows(df, before_season=s)
        te = df[(df.season == s) & df.home_win.notna()]
        pr.loc[te.index] = M.fit_predict(tr, te, feats, "margin")
    return pr


def rating_only_prob(df, mu_col):
    """P = Phi(mu / sigma), sigma fit by log loss on 2015-2019 only."""
    v = df[df.season.isin(VAL) & df.home_win.notna()]
    f = lambda s: _ll(v.home_win.values, norm.cdf(v[mu_col].values / s))
    sig = minimize_scalar(f, bounds=(5, 30), method="bounded").x
    return pd.Series(norm.cdf(df[mu_col].values / sig), index=df.index), float(sig)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(d, cols):
    """Logistic blend on logits, fit on 2015-2019, applied to all rows."""
    X = np.column_stack([logit(d[c]) for c in cols])
    fit_rows = d.season.isin(VAL).values
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(X[fit_rows], d.home_win.values[fit_rows].astype(int))
    return lr.predict_proba(X)[:, 1], [round(float(c), 3) for c in lr.coef_[0]]


def main():
    t0 = time.time()
    import experiment as E  # loads schedules, pbp 2012-2026, injuries, NGS
    prod, df = E.run("production")
    games, pbp = E.G, E.P
    g = game_table(games, pbp)
    print(f"production (harness): {prod}  [{time.time() - t0:.0f}s]")

    # ---- tune each variant on 2012-2019 (cap grid for the margin variant only)
    tuned = {}
    for cap in (None, 28.0, 21.0, 14.0):
        prm, ll = tune(g, "margin", cap)
        tuned[f"margin_cap{cap}"] = {"variant": "margin", "cap": cap, "params": prm, "tune_ll": ll}
        print("tuned margin cap", cap, round(ll, 4), {k: round(v, 3) for k, v in prm.items()})
    best_cap = min((v for v in tuned.values()), key=lambda v: v["tune_ll"])["cap"]
    for variant in ("epa", "qb", "epa_qb"):
        prm, ll = tune(g, variant, best_cap)
        tuned[variant] = {"variant": variant, "cap": best_cap, "params": prm, "tune_ll": ll}
        print("tuned", variant, round(ll, 4), {k: round(v, 3) for k, v in prm.items()},
              f"[{time.time() - t0:.0f}s]")
    tuned["margin"] = tuned[f"margin_cap{best_cap}"]
    main_variants = ["margin", "epa", "qb", "epa_qb"]

    # ---- leakage checks
    leak = {v: leakage_check(games, pbp, tuned[v]["params"], tuned[v]["variant"], tuned[v]["cap"])
            for v in main_variants}
    print("leakage checks passed", json.dumps(leak))

    # ---- attach ratings and evaluate
    d = df.copy()
    for v in main_variants:
        r = kalman(g, tuned[v]["params"], tuned[v]["variant"], tuned[v]["cap"])
        r = r.rename(columns={c: f"{c}_{v}" for c in r.columns if c != "game_id"})
        d = d.merge(r, on="game_id", how="left")
    ev = d[d.season.isin(list(VAL) + list(HOLD)) & d.home_win.notna()].copy()
    assert ev[[f"ss_mu_{v}" for v in main_variants]].notna().all().all()

    rows, preds = [], {}

    def add(name, p, kind):
        preds[name] = pd.Series(np.asarray(p, float), index=ev.index)
        r = {"name": name, "kind": kind}
        for lab, ss in (("val", VAL), ("hold", HOLD)):
            m = ev.season.isin(ss).values
            r[lab] = score(ev.home_win.values[m], preds[name].values[m])
        rows.append(r)

    add("Production (ridge, FEATURES)", walk_forward(d, F.FEATURES).loc[ev.index], "production")
    add("Vegas closing (no-vig)", ev.vegas_home_prob, "market")
    sig = {}
    for v in main_variants:
        p, sig[v] = rating_only_prob(ev, f"ss_mu_{v}")
        add(f"KF rating only [{v}]", p, "rating")
        add(f"KF rating only [{v}] native var", norm.cdf(ev[f"ss_mu_{v}"] / np.sqrt(ev[f"ss_var_{v}"])), "rating")
    feat_tests = {
        "Prod + ss_diff[margin]": F.FEATURES + ["ss_diff_margin"],
        "Prod + ss_diff[epa]": F.FEATURES + ["ss_diff_epa"],
        "Prod + ss_diff[qb] (team part)": F.FEATURES + ["ss_diff_qb"],
        "Prod + ss_diff_qb[qb] (team+QB)": F.FEATURES + ["ss_diff_qb_qb"],
        "Prod + ss_diff[epa_qb] (team part)": F.FEATURES + ["ss_diff_epa_qb"],
        "Prod - Elo/pt_diff + ss_diff[epa]": [f for f in F.FEATURES if f not in ("elo_diff", "pt_diff_diff", "fw_elo_diff", "fw_pt_diff_diff")] + ["ss_diff_epa"],
    }
    for name, feats in feat_tests.items():
        add(name, walk_forward(d, feats).loc[ev.index], "production+rating")

    # ---- Vegas blends (games with a moneyline; fit 2015-2019, score 2020-2025)
    bl = ev[ev.vegas_home_prob.notna()].copy()
    for k, s in preds.items():
        bl[k] = s.loc[bl.index]
    best_feat = min(feat_tests, key=lambda k: next(r["val"]["ll"] for r in rows if r["name"] == k))
    best_rating = min((f"KF rating only [{v}]" for v in main_variants),
                      key=lambda k: next(r["val"]["ll"] for r in rows if r["name"] == k))
    blend_specs = {
        "Vegas alone (recalibrated)": ["Vegas closing (no-vig)"],
        "Production + Vegas": ["Production (ridge, FEATURES)", "Vegas closing (no-vig)"],
        f"{best_rating} + Vegas": [best_rating, "Vegas closing (no-vig)"],
        f"{best_feat} + Vegas": [best_feat, "Vegas closing (no-vig)"],
        f"Production + {best_rating} + Vegas": ["Production (ridge, FEATURES)", best_rating, "Vegas closing (no-vig)"],
    }
    brows = []
    for name, cols in blend_specs.items():
        p, coef = blend(bl, cols)
        r = {"name": name, "weights": dict(zip(cols, coef))}
        for lab, ss in (("val", VAL), ("hold", HOLD)):
            m = bl.season.isin(ss).values
            r[lab] = score(bl.home_win.values[m], p[m])
        brows.append(r)
    raw = {"name": "Vegas closing raw", "weights": {}}
    for lab, ss in (("val", VAL), ("hold", HOLD)):
        m = bl.season.isin(ss).values
        raw[lab] = score(bl.home_win.values[m], bl["Vegas closing (no-vig)"].values[m])
    brows.insert(0, raw)

    # correlation of ratings with Elo / Vegas spread (diagnostic)
    diag = {v: {"corr_with_elo_diff": round(float(ev[f"ss_diff_{v}"].corr(ev.elo_diff)), 3),
                "corr_mu_with_minus_spread_line": round(float(ev[f"ss_mu_{v}"].corr(ev.spread_line)), 3)}
            for v in main_variants}

    base = rows[0]
    verdict = []
    for r in rows[2:]:
        if r["kind"] == "production+rating":
            ok = r["val"]["ll"] < base["val"]["ll"] and r["hold"]["ll"] < base["hold"]["ll"]
            verdict.append({"name": r["name"], "d_val": round(r["val"]["ll"] - base["val"]["ll"], 4),
                            "d_hold": round(r["hold"]["ll"] - base["hold"]["ll"], 4), "adopt_rule_met": ok})
    res = {"tuned": tuned, "best_cap": best_cap, "sigma_rating_only": sig, "leakage": leak,
           "results": rows, "blends": brows, "diagnostics": diag, "verdict": verdict,
           "harness_production": prod}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_space_ratings.json").write_text(json.dumps(M.clean_json(res), indent=2, default=str))
    print(json.dumps(M.clean_json({"results": rows, "blends": brows, "verdict": verdict, "diag": diag}), indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
