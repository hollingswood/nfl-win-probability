"""Research: state-space (Kalman filter) team ratings.

Model (per team i, latent net strength theta_i in points):
    within a season   theta_t = theta_{t-1} + eta,        eta ~ N(0, q_w * weeks_elapsed)
    offseason         theta   <- rho * (theta - mean),    P <- rho^2 P + q_off * I
    observation       y_g = theta_home - theta_away + hfa*home_field [+ beta*(qb_home - qb_away)] + eps

Variants
  margin   y = point margin (optionally capped), eps ~ N(0, 13^2)
  epa      y = composite z = w*margin + (1-w)*k*epa_margin (k fixed = OLS slope of margin on net
           EPA/play margin, 2012-2019).  Two noisy measurements of the same h'theta with correlated
           errors reduce exactly to one GLS-weighted composite, so w and its noise s_z are tuned.
           Before 2012 (no pbp) the plain margin is used.
  qb       margin, but ratings are QB-neutral: the pre-game starter QB rating edge (production,
           leak-tested qb ratings) enters the observation with coefficient beta, and is added back
           for the actual starter at prediction time.
  epa_qb   both.
Separate offense/defense ratings were not fit: with iid score noise and symmetric O/D priors the
margin predictive of an O/D model equals the net-rating model exactly (the total only informs
off-def, which is orthogonal to the margin).

All teams share one full 32x32 covariance.  Pre-game ratings for every game on a date are read
BEFORE any game on that date updates the filter, so a game's rating uses only earlier `gameday`s.

Hyperparameters: minimize one-step-ahead win log loss, p = Phi(mu / sqrt(h'Ph + s_p^2)), on
2012-2019 games only (variant margin_gauss instead maximizes the Gaussian predictive likelihood of
the margin).  Evaluation: validation 2015-2019, holdout 2020-2025, walk-forward as in
scripts/experiment.py.

    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/state_space_ratings.py
Writes output/research/state_space_ratings.json (the .md report is written from it by hand).
    --reuse-tuned  reuse hyperparameters already stored in that json (only fits missing variants).
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
from nflpred import features as F, model as M  # noqa: E402

HIST_START = 2002           # 32-team era; 2002-2011 is filter warm-up only
TUNE = (2012, 2019)         # hyperparameters use only these seasons
VAL, HOLD = range(2015, 2020), range(2020, 2026)
SIGMA_U = 13.0              # margin observation noise (fixes the variance scale of the ratings)
P0 = 25.0
OUT = ROOT / "output" / "research"
MIN_GAIN = 0.001            # adoption rule: val AND holdout log loss both improve by >= this


# ---------------------------------------------------------------- inputs
def game_table(games: pd.DataFrame, pbp: pd.DataFrame | None) -> pd.DataFrame:
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
        g["epa_margin"] = h.values - a.values
        sched = F.prepare_schedule(games)
        q = F.attach_qb(sched, F.qb_ratings(pbp, sched))[["game_id", "home_qb_rating", "away_qb_rating"]]
        g = g.drop(columns="qb_d").merge(q, on="game_id", how="left")
        g["qb_d"] = (g["home_qb_rating"] - g["away_qb_rating"]).fillna(0.0)
    return g


def epa_scale(g: pd.DataFrame) -> float:
    """OLS slope of point margin on net-EPA/play margin, 2012-2019 only (a fixed hyperparameter)."""
    d = g[g["season"].between(*TUNE) & g["completed"] & g["epa_margin"].notna()]
    return float(np.polyfit(d["epa_margin"], d["margin"], 1)[0])


# ---------------------------------------------------------------- filter
PARAMS = {  # name: (transform, init)
    "q_w": ("log", 0.8), "q_off": ("log", 8.0), "rho": ("logit", 0.55), "hfa": ("id", 1.9),
    "s_p": ("log", 8.5), "w": ("logit", 0.5), "hfa_z": ("id", 1.5), "s_z": ("log", 9.0),
    "beta": ("id", 20.0),
}
VARIANT_PARAMS = {
    "margin": ["q_w", "q_off", "rho", "hfa", "s_p"],
    "margin_gauss": ["q_w", "q_off", "rho", "hfa", "s_p"],
    "epa": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "hfa_z", "s_z"],
    "qb": ["q_w", "q_off", "rho", "hfa", "s_p", "beta"],
    "epa_qb": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "hfa_z", "s_z", "beta"],
    # constrained EPA fits: composite HFA tied to the margin HFA, s_z >= 3, q_w, q_off >= 0.05
    "epa_tied": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "s_z"],
    "epa_qb_tied": ["q_w", "q_off", "rho", "hfa", "s_p", "w", "s_z", "beta"],
}


def _fwd(name, v):
    t = PARAMS[name][0]
    return np.log(v) if t == "log" else np.log(v / (1 - v)) if t == "logit" else v


def _inv(name, x):
    t = PARAMS[name][0]
    return float(np.exp(x)) if t == "log" else float(1 / (1 + np.exp(-x))) if t == "logit" else float(x)


def kalman(g: pd.DataFrame, p: dict, variant: str, cap: float | None, k: float = 0.0) -> pd.DataFrame:
    """Pre-game ratings for every row of g. Returns team ratings, predicted margin mu (incl. HFA and
    QB term) and its predictive variance var = h'Ph + s_p^2 (ss_pvar = h'Ph alone)."""
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
        out[idx, 3] = P[h, h] + P[a, a] - 2 * P[h, a]
        for i in idx:  # sequential scalar updates == joint update (independent game noises)
            if not done[i]:
                continue
            hi, ai = H[i], A[i]
            base = m[hi] - m[ai] + beta * qbd[i]
            if use_epa and not np.isnan(ep[i]):
                z = p["w"] * y[i] + (1 - p["w"]) * k * ep[i]
                innov, r2 = z - base - p.get("hfa_z", p["hfa"]) * hf[i], p["s_z"] ** 2
            else:
                innov, r2 = y[i] - base - p["hfa"] * hf[i], SIGMA_U ** 2
            Ph = P[:, hi] - P[:, ai]
            S = Ph[hi] - Ph[ai] + r2
            m = m + Ph * (innov / S)
            P = P - np.outer(Ph, Ph) / S
    res = g[["game_id"]].copy()
    res["ss_home"], res["ss_away"], res["ss_mu"], res["ss_pvar"] = out.T
    res["ss_var"] = res["ss_pvar"] + p["s_p"] ** 2
    res["ss_diff"] = res["ss_home"] - res["ss_away"]          # team part only (QB-neutral for qb variants)
    res["ss_diff_qb"] = res["ss_diff"] + beta * qbd            # + current starter's QB term
    return res


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def objective(x, names, g, variant, cap, k, mask, y, margin):
    p = {kk: _inv(kk, v) for kk, v in zip(names, x)}
    if p["q_w"] > 50 or p["q_off"] > 400 or p.get("s_z", 5) > 100:
        return 10.0
    if variant.endswith("_tied") and (p["s_z"] < 3 or p["q_w"] < 0.05 or p["q_off"] < 0.05):
        return 10.0
    r = kalman(g, p, variant, cap, k)
    mu, var = r["ss_mu"].values[mask], r["ss_var"].values[mask]
    if variant == "margin_gauss":   # Gaussian predictive NLL of the (capped) margin
        return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (margin - mu) ** 2 / var))
    return _ll(y, norm.cdf(mu / np.sqrt(var)))


def tune(g, variant, cap, k, init=None):
    names = VARIANT_PARAMS[variant]
    init = {**{kk: PARAMS[kk][1] for kk in names}, **(init or {})}
    x0 = np.array([_fwd(kk, init[kk]) for kk in names])
    comp = g["completed"].values
    win = np.where(comp & (g["margin"] != 0), (g["margin"] > 0).astype(float), np.nan)
    tune_rows = g["season"].between(*TUNE).values
    if variant == "margin_gauss":
        mask = tune_rows & comp
        mg = g["margin"].values if cap is None else np.clip(g["margin"].values, -cap, cap)
        args = (names, g, variant, cap, k, mask, None, mg[mask])
    else:
        mask = tune_rows & ~np.isnan(win)
        args = (names, g, variant, cap, k, mask, win[mask], None)
    r = minimize(objective, x0, args=args, method="Nelder-Mead",
                 options={"maxiter": 300 * len(names), "xatol": 1e-3, "fatol": 1e-6})
    return {kk: _inv(kk, v) for kk, v in zip(names, r.x)}, float(r.fun), bool(r.success)


# ---------------------------------------------------------------- leakage check
def leakage_check(games, pbp, cfg, cutoffs=("2019-11-10", "2023-10-01")):
    """Erase every result on/after `cutoff` (scores + that pbp); ratings for all games dated
    <= cutoff must not change.  Mirrors tests/test_pipeline.py::test_no_future_leakage."""
    args = (cfg["params"], cfg["variant"], cfg["cap"], cfg["k"])
    full = kalman(game_table(games, pbp), *args).set_index("game_id")
    gd = pd.to_datetime(games.set_index("game_id")["gameday"]).reindex(full.index)
    out = {}
    for c in cutoffs:
        cut = pd.Timestamp(c)
        g2 = games.copy()
        fut = pd.to_datetime(g2["gameday"]) >= cut
        g2.loc[fut, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
        p2 = pbp[~pbp["game_id"].isin(set(g2.loc[fut, "game_id"]))]
        masked = kalman(game_table(g2, p2), *args).set_index("game_id")
        ids = full.index[gd <= cut]
        cols = ["ss_home", "ss_away", "ss_mu", "ss_var", "ss_diff_qb"]
        diff = float(np.abs(full.loc[ids, cols].values - masked.loc[ids, cols].values).max())
        later = full.index[gd > cut + pd.Timedelta(days=7)][:200]
        moved = float(np.abs(full.loc[later, "ss_mu"].values - masked.loc[later, "ss_mu"].values).max())
        assert diff < 1e-9, f"LEAK: {cfg['variant']} ratings changed before {c} (max diff {diff})"
        assert moved > 1e-6, "leak check has no power"
        out[c] = {"games_checked": int(len(ids)), "games_on_cutoff": int((gd == cut).sum()),
                  "max_abs_diff": diff, "max_change_after_cutoff": moved}
    return out


# ---------------------------------------------------------------- evaluation
def score(y, p):
    s = M.score(y, p)
    return {"ll": round(s["log_loss"], 4), "brier": round(s["brier"], 4), "n": s["n"]}


def walk_forward(df, feats):
    """Identical to scripts/experiment.py::run: for each test season, fit ridge on prior seasons."""
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


def hetero(p, var, season, y):
    """Uncertainty-aware conversion: z = Phi^-1(p); p' = Phi(z / sqrt(1 + c*(var - vbar)/169)),
    c and vbar fit on 2015-2019 only."""
    z = norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    fit = np.isin(season, list(VAL))
    vbar = float(np.mean(var[fit]))
    f = lambda c: _ll(y[fit], norm.cdf(z[fit] / np.sqrt(np.maximum(0.25, 1 + c * (var[fit] - vbar) / 169))))
    c = float(minimize_scalar(f, bounds=(-5, 5), method="bounded").x)
    return norm.cdf(z / np.sqrt(np.maximum(0.25, 1 + c * (var - vbar) / 169))), c


def paired_se(y, p1, p2):
    """Mean and SE of per-game log-loss difference (p1 - p2)."""
    l = lambda p: -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    d = l(p1) - l(p2)
    return round(float(d.mean()), 4), round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)


def main():
    t0 = time.time()
    import experiment as E  # loads schedules, pbp 2012-2026, injuries, NGS
    prod, df = E.run("production")
    games, pbp = E.G, E.P
    g = game_table(games, pbp)
    k = epa_scale(g)
    print(f"production (harness): {prod}; EPA->points k={k:.2f}  [{time.time() - t0:.0f}s]", flush=True)

    # ---- tune on 2012-2019 only
    tuned = {}
    prev = OUT / "state_space_ratings.json"
    if "--reuse-tuned" in sys.argv and prev.exists():   # skip re-tuning variants already fit
        tuned = {kk: v for kk, v in json.loads(prev.read_text())["tuned"].items()}
        print("reusing tuned:", list(tuned), flush=True)
    for cap in (None, 28.0, 21.0, 14.0):
        if f"margin_cap{cap}" in tuned:
            continue
        prm, ll, ok = tune(g, "margin", cap, k)
        tuned[f"margin_cap{cap}"] = {"variant": "margin", "cap": cap, "k": k, "params": prm, "tune_obj": ll, "converged": ok}
        print("tuned margin cap", cap, round(ll, 4), ok, {kk: round(v, 3) for kk, v in prm.items()}, flush=True)
    best_cap = min(tuned.values(), key=lambda v: v["tune_obj"])["cap"]
    tuned["margin"] = tuned[f"margin_cap{best_cap}"]
    mp = tuned["margin"]["params"]
    inits = {"margin_gauss": {**mp, "s_p": 13.0}, "epa": {**mp, "hfa_z": mp["hfa"]},
             "qb": mp, "epa_qb": {**mp, "hfa_z": mp["hfa"]}}
    inits["epa_tied"] = {**mp, "w": 0.5, "s_z": 9.0}
    for variant in ("margin_gauss", "epa", "qb", "epa_qb", "epa_tied", "epa_qb_tied"):
        if variant in tuned:
            continue
        if variant == "epa_qb":
            inits[variant] = {**tuned["epa"]["params"], "beta": tuned["qb"]["params"]["beta"]}
        if variant == "epa_qb_tied":
            inits[variant] = {**tuned["epa_tied"]["params"], "beta": tuned["qb"]["params"]["beta"]}
        prm, ll, ok = tune(g, variant, best_cap, k, inits[variant])
        tuned[variant] = {"variant": variant, "cap": best_cap, "k": k, "params": prm, "tune_obj": ll, "converged": ok}
        print("tuned", variant, round(ll, 4), ok, {kk: round(v, 3) for kk, v in prm.items()},
              f"[{time.time() - t0:.0f}s]", flush=True)
    variants = ["margin", "margin_gauss", "epa", "qb", "epa_qb", "epa_tied", "epa_qb_tied"]

    # ---- leakage checks
    leak = {v: leakage_check(games, pbp, tuned[v]) for v in variants}
    print("leakage checks passed", json.dumps(leak), flush=True)

    # ---- attach ratings
    d = df.copy()
    for v in variants:
        c = tuned[v]
        r = kalman(g, c["params"], c["variant"], c["cap"], c["k"])
        r = r.rename(columns={cc: f"{cc}_{v}" for cc in r.columns if cc != "game_id"})
        d = d.merge(r, on="game_id", how="left")
        d[f"fw_ss_diff_{v}"] = d["final_week"] * d[f"ss_diff_{v}"]
        d[f"early_ss_diff_{v}"] = (d["week"] <= 4) * d[f"ss_diff_{v}"]
    ev = d[d.season.isin(list(VAL) + list(HOLD)) & d.home_win.notna()].copy()
    assert ev[[f"ss_mu_{v}" for v in variants]].notna().all().all()

    rows, preds = [], {}

    def add(name, p, kind):
        preds[name] = pd.Series(np.asarray(p, float), index=ev.index)
        r = {"name": name, "kind": kind}
        for lab, ss in (("val", VAL), ("hold", HOLD)):
            m = ev.season.isin(ss).values
            r[lab] = score(ev.home_win.values[m], preds[name].values[m])
        rows.append(r)
        print(r["name"], r["val"]["ll"], r["hold"]["ll"], flush=True)

    PROD = "Production (ridge, FEATURES)"
    add(PROD, walk_forward(d, F.FEATURES).loc[ev.index], "production")
    add("Vegas closing (no-vig)", ev.vegas_home_prob, "market")
    sig = {}
    for v in variants:
        p, sig[v] = rating_only_prob(ev, f"ss_mu_{v}")
        add(f"KF alone [{v}]", p, "rating")
        add(f"KF alone [{v}] native var", norm.cdf(ev[f"ss_mu_{v}"] / np.sqrt(ev[f"ss_var_{v}"])), "rating_uncert")

    no_elo = [f for f in F.FEATURES if f not in ("elo_diff", "fw_elo_diff")]
    no_elo_pd = [f for f in no_elo if f not in ("pt_diff_diff", "fw_pt_diff_diff")]
    feat_tests = {}
    for v in variants:
        team = f"ss_diff_{v}"
        feat_tests[f"Prod + {team}"] = ("add", F.FEATURES + [team])
        feat_tests[f"Prod - Elo + {team}"] = ("replace", no_elo + [team, f"fw_ss_diff_{v}"])
        if "qb" in v:
            feat_tests[f"Prod + ss_diff_qb_{v} (team+QB)"] = ("add", F.FEATURES + [f"ss_diff_qb_{v}"])
    for v in ("margin", "epa_tied"):
        feat_tests[f"Prod - Elo - pt_diff + ss_diff_{v}"] = ("replace", no_elo_pd + [f"ss_diff_{v}", f"fw_ss_diff_{v}"])
        feat_tests[f"Prod + ss_diff_{v} + early x ss_diff"] = ("uncert", F.FEATURES + [f"ss_diff_{v}", f"early_ss_diff_{v}"])
        feat_tests[f"Prod + ss_diff_{v} + ss_pvar"] = ("uncert", F.FEATURES + [f"ss_diff_{v}", f"ss_pvar_{v}"])
    for name, (kind, feats) in feat_tests.items():
        add(name, walk_forward(d, feats).loc[ev.index], f"production+rating:{kind}")

    # ---- uncertainty-aware probability conversion (does KF variance sharpen/flatten correctly?)
    y_all, s_all = ev.home_win.values.astype(float), ev.season.values
    unc = {}
    for v in ("margin", "epa_tied"):
        for base in (PROD, f"KF alone [{v}]"):
            p2, c = hetero(preds[base].values, ev[f"ss_pvar_{v}"].values, s_all, y_all)
            nm = f"{base} x KF-variance [{v}]"
            unc[nm] = c
            add(nm, p2, "uncert_conversion")

    # ---- early vs late season breakdown
    base = next(r for r in rows if r["name"] == PROD)
    cands = [r for r in rows if r["kind"].startswith("production+rating") or r["kind"] == "uncert_conversion"]
    cands = [r for r in cands if not r["name"].startswith("KF alone")]
    best_cand = min(cands, key=lambda r: r["val"]["ll"])["name"]
    best_rating = min((f"KF alone [{v}]" for v in variants),
                      key=lambda kk: next(r["val"]["ll"] for r in rows if r["name"] == kk))
    early = {}
    for nm in (PROD, best_cand, best_rating, "Vegas closing (no-vig)"):
        early[nm] = {}
        for lab, ss in (("val", VAL), ("hold", HOLD)):
            for wl, wm in (("wk1-4", ev.week <= 4), ("wk5+", ev.week > 4)):
                m = (ev.season.isin(ss) & wm & ev.vegas_home_prob.notna()).values
                early[nm][f"{lab}_{wl}"] = score(ev.home_win.values[m], preds[nm].values[m])["ll"]

    # ---- verdict vs production (paired SE on holdout)
    verdict = []
    mh_ev = ev.season.isin(HOLD).values
    for r in cands:
        dv, dh = round(r["val"]["ll"] - base["val"]["ll"], 4), round(r["hold"]["ll"] - base["hold"]["ll"], 4)
        _, se_h = paired_se(y_all[mh_ev], preds[r["name"]].values[mh_ev], preds[PROD].values[mh_ev])
        verdict.append({"name": r["name"], "d_val": dv, "d_hold": dh, "d_hold_se": se_h,
                        "adopt_rule_met": dv <= -MIN_GAIN and dh <= -MIN_GAIN})

    # ---- Vegas blends (games with a closing moneyline; fit 2015-2019, score 2020-2025)
    bl = ev[ev.vegas_home_prob.notna()].copy()
    for kk, s in preds.items():
        bl[kk] = s.loc[bl.index]
    V = "Vegas closing (no-vig)"
    blend_specs = {
        "Vegas alone (recalibrated)": [V],
        "Production + Vegas": [PROD, V],
        f"{best_rating} + Vegas": [best_rating, V],
        f"{best_cand} + Vegas": [best_cand, V],
        f"Production + {best_rating} + Vegas": [PROD, best_rating, V],
    }
    brows = []
    raw = {"name": "Vegas closing raw", "weights": {}}
    for lab, ss in (("val", VAL), ("hold", HOLD)):
        m = bl.season.isin(ss).values
        raw[lab] = score(bl.home_win.values[m], bl[V].values[m])
    brows.append(raw)
    yb, mh = bl.home_win.values.astype(float), bl.season.isin(HOLD).values
    for name, cols in blend_specs.items():
        p, coef = blend(bl, cols)
        r = {"name": name, "weights": dict(zip(cols, coef))}
        for lab, ss in (("val", VAL), ("hold", HOLD)):
            m = bl.season.isin(ss).values
            r[lab] = score(bl.home_win.values[m], p[m])
        r["hold_minus_vegas_raw"], r["hold_minus_vegas_raw_se"] = paired_se(yb[mh], p[mh], bl[V].values[mh])
        brows.append(r)

    diag = {v: {"corr_ss_diff_with_elo_diff": round(float(ev[f"ss_diff_{v}"].corr(ev.elo_diff)), 3),
                "corr_mu_with_spread_line": round(float(ev[f"ss_mu_{v}"].corr(ev.spread_line)), 3),
                "mean_rating_sd_wk1": round(float(np.sqrt(ev.loc[ev.week == 1, f"ss_pvar_{v}"]).mean()), 2),
                "mean_rating_sd_wk10": round(float(np.sqrt(ev.loc[ev.week == 10, f"ss_pvar_{v}"]).mean()), 2)}
            for v in variants}

    res = {"epa_k": k, "tuned": tuned, "best_cap": best_cap, "sigma_rating_only": sig, "leakage": leak,
           "results": rows, "blends": brows, "uncert_conversion_c": unc, "early_late": early,
           "diagnostics": diag, "verdict": verdict, "harness_production": prod,
           "best_candidate": best_cand, "best_rating_only": best_rating}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "state_space_ratings.json").write_text(json.dumps(M.clean_json(res), indent=2, default=str))
    print(json.dumps(M.clean_json({"blends": brows, "verdict": verdict, "early": early, "diag": diag,
                                   "unc": unc}), indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
