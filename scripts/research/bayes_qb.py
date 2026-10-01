"""Hierarchical / empirical-Bayes QB projection with uncertainty (research; NOT a production input).

Model (per QB, a 4-dim latent "true talent" state, per dropback):
    theta = (EPA, success rate, CPOE, sack rate)
  * Prior at the QB's first appearance in the data (2012+):  theta0 ~ N(B x, T0), with
      x = [1, unknown-history vet (rookie_season < 2012), log draft pick (UDFA = 300), years in league
           at debut (cap 10), age at entry]
    B and T0 estimated by empirical Bayes (iterated WLS + method-of-moments between-QB covariance) on
    each QB's first-season observations.
  * Observation per QB-game: component means over n dropbacks (CPOE over n_att attempts), noise R =
    Sigma (x) N/n, Sigma = within-QB per-play covariance (pairwise; CPOE pairs on attempts only).
    The off-diagonal terms of T are what make fast-stabilising stats (success, CPOE, sacks) inform EPA.
  * Dynamics (Kalman random walk): every game P += a*Tall; every season boundary m += d(bucket), P += b*Tall,
    Tall = between-QB-season talent covariance; d = development mean shift by years-in-league bucket x
    career-backup flag (< 300 career dropbacks entering the season).  a, b by max one-step-ahead predictive
    log-likelihood of game EPA/dropback; d by weighted mean of season-mean minus pre-season estimate.
  * Hyperparameters are WALK-FORWARD: H_s is fit only on seasons < s for s = 2015..2019; H_2020 (fit on
    2012-2019) is frozen for 2020+.  Warm-up seasons 2012-14 (training rows only, never scored) use H_2015.
  * Pre-game output for each team's ACTUAL starter (nflverse *_qb_id; hindsight on who starts, same as the
    production model): posterior mean bq and SD bq_sd of EPA/dropback, using games with an earlier gameday.

Evaluation:
  (a) production walk-forward ridge (tune 2015-19, holdout 2020-25) with qb_diff / qb_change_diff replaced or
      augmented, + an uncertainty-widened sigma variant; overall and on QB-change / <300-dropback subsets.
  (b) ATS residual vs nflverse closing spread on QB-change games (2013-2025); early-week line move and
      price-based CLV on 2020-22 (dev) -> frozen rules -> 2023-25 (holdout, run once).
  (c) leakage masking test (results erased from a cutoff; features on that date must not change).

Stages:
    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/bayes_qb.py dev
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/bayes_qb.py holdout
    PYTHONPATH=src:scripts python scripts/research/bayes_qb.py report        # bayes_qb.md from the json
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
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/bayesqb")
SCR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("QBT_CACHE", str(SCR))

from nflpred import features as F, model as M  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "research"
JSON = OUT / "bayes_qb.json"
MD = OUT / "bayes_qb.md"
COMP = ["epa", "sr", "cpoe", "sack"]
VAL, HOLD = range(2015, 2020), range(2020, 2026)
FIT_SEASONS = [2015, 2016, 2017, 2018, 2019, 2020]
GRID_A = [0.0, 0.005, 0.01, 0.02, 0.04]
GRID_B = [0.0, 0.1, 0.25, 0.5, 1.0]
LOW_DB = 300
DB_PER_GAME = 38
TM = F.TEAM_MAP


def _bucket(k: int, backup: bool) -> str:
    kb = "y2" if k <= 1 else "y3" if k == 2 else "y4" if k == 3 else "y5-7" if k <= 6 else "y8-12" if k <= 11 else "y13+"
    return kb + ("_bk" if backup else "")


# ============================================================================ data
def load_plays(max_gameday=None, pbp_seasons=range(2012, 2027)) -> pd.DataFrame:
    cols = ["game_id", "season", "posteam", "qb_dropback", "qb_epa", "id", "cpoe", "success", "sack"]
    fr = []
    for s in pbp_seasons:
        p = RAW / f"pbp_{s}.parquet"
        if p.exists():
            x = pd.read_parquet(p, columns=cols)
            fr.append(x[x.qb_dropback.eq(1) & x.qb_epa.notna() & x.id.notna()])
    p = pd.concat(fr, ignore_index=True)
    p["cp"] = p.cpoe / 100.0
    p["sr"] = p.success.astype(float)
    p["sack"] = p.sack.fillna(0).astype(float)
    p["epa"] = p.qb_epa.astype(float)
    g = pd.read_parquet(RAW / "games.parquet", columns=["game_id", "gameday"])
    p = p.merge(g.assign(gameday=pd.to_datetime(g.gameday)), on="game_id")
    if max_gameday is not None:
        p = p[p.gameday < max_gameday]
    return p[["game_id", "season", "gameday", "id", "epa", "sr", "cp", "sack"]]


def qb_games(p: pd.DataFrame) -> pd.DataFrame:
    a = p.groupby(["id", "game_id", "season", "gameday"]).agg(
        n=("epa", "size"), epa=("epa", "mean"), sr=("sr", "mean"), sack=("sack", "mean"),
        ncp=("cp", "count"), cpoe=("cp", "mean")).reset_index()
    return a.sort_values(["id", "gameday", "game_id"]).reset_index(drop=True)


def play_cov_by_season(p: pd.DataFrame) -> dict:
    """Within-QB-season per-play covariance pieces (sums), so pooled Sigma for any season range is cheap."""
    q = p.copy()
    key = [q.id, q.season]
    for c in ("epa", "sr", "sack"):
        q[c + "_d"] = q[c] - q.groupby(key)[c].transform("mean")
    q["cp_d"] = q.cp - q.groupby(key).cp.transform("mean")
    out = {}
    for s, d in q.groupby("season"):
        X = d[["epa_d", "sr_d", "cp_d", "sack_d"]].to_numpy()
        hascp = ~np.isnan(X[:, 2])
        S = np.zeros((4, 4)); N = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                ok = hascp if (i == 2 or j == 2) else np.ones(len(X), bool)
                S[i, j] = np.nansum(X[ok, i] * X[ok, j]); N[i, j] = ok.sum()
        out[int(s)] = (S, N)
    return out


def qb_meta() -> pd.DataFrame:
    pl = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "position", "rookie_season", "birth_date", "draft_pick"])
    pl = pl.drop_duplicates("gsis_id").set_index("gsis_id")
    bd = pd.to_datetime(pl.birth_date, errors="coerce")
    by = bd.dt.year + (bd.dt.dayofyear - 1) / 365.25
    pl["age_entry"] = (pl.rookie_season + 0.67 - by).fillna(22.8)
    pl["lpick"] = np.log(pl.draft_pick.fillna(300).clip(1, 300))
    return pl


def covariates(meta: pd.DataFrame, qb: str, season: int) -> np.ndarray:
    if qb in meta.index:
        r = meta.loc[qb]
        rs = r.rookie_season if pd.notna(r.rookie_season) else season
        lp, age = r.lpick, r.age_entry
    else:
        rs, lp, age = season, math.log(300), 22.8
    unk = float(rs < 2012)
    return np.array([1.0, unk, lp - math.log(64), (1 - unk) * (min(season - rs, 10) - 1.0), age - 22.8])


X_NAMES = ["intercept", "unknown_history_vet", "log_pick_minus_log64", "years_in_league_minus1(non-vet)", "age_entry_minus22.8"]


def _R(Sig, n, ncp):
    R = Sig / max(n, 1)
    if ncp > 0:
        R = R.copy(); R[2, 2] = Sig[2, 2] / ncp
    return R


# ============================================================================ hyperparameters
class Hyper:
    def __init__(self, Sig, Tall, B, T0, a, b, D):
        self.Sig, self.Tall, self.B, self.T0, self.a, self.b, self.D = Sig, Tall, B, T0, a, b, D

    def to_json(self):
        return {"Sigma_per_play": self.Sig.round(5).tolist(), "T_between_qb_season": self.Tall.round(6).tolist(),
                "prior_B": {n: dict(zip(COMP, self.B[i].round(4).tolist())) for i, n in enumerate(X_NAMES)},
                "prior_T0_new": self.T0.round(6).tolist(), "prior_T0_vet": self.T0_vet.round(6).tolist(), "a_game_drift": self.a, "b_season_drift": self.b,
                "development_d": {k: dict(zip(COMP, np.round(v, 4).tolist())) for k, v in self.D.items()}}


def _psd(M_, floor):
    w, V = np.linalg.eigh((M_ + M_.T) / 2)
    return V @ np.diag(np.maximum(w, floor)) @ V.T


def fit_hyper(qg: pd.DataFrame, covs: dict, meta: pd.DataFrame, upto: int, log=print) -> Hyper:
    """Estimate hyperparameters from seasons < upto only."""
    S = sum(covs[s][0] for s in covs if s < upto); N = sum(covs[s][1] for s in covs if s < upto)
    Sig = S / N
    d = qg[qg.season < upto]
    # ---- between-QB-season covariance of talent
    w = d.assign(**{c: d[c] * d.n for c in ("epa", "sr", "sack")}, cpw=d.cpoe.fillna(0) * d.ncp)
    qs = w.groupby(["id", "season"]).agg(n=("n", "sum"), ncp=("ncp", "sum"), epa=("epa", "sum"), sr=("sr", "sum"),
                                          sack=("sack", "sum"), cpoe=("cpw", "sum")).reset_index()
    for c in ("epa", "sr", "sack"):
        qs[c] /= qs.n
    qs["cpoe"] /= qs.ncp.replace(0, np.nan)
    big = qs[(qs.n >= 150) & (qs.ncp > 0)]
    Y = big[COMP].to_numpy()
    Rbar = np.mean([_R(Sig, n, c) for n, c in zip(big.n, big.ncp)], axis=0)
    Tall = _psd(np.cov(Y.T) - Rbar, 1e-3 * np.diag(np.cov(Y.T)).min())
    # ---- debut prior: first season in data, QBs only
    qs = qs.merge(meta[["position"]], left_on="id", right_index=True, how="left")
    first = qs.sort_values("season").drop_duplicates("id")
    first = first[(first.position == "QB") & (first.ncp > 0)]
    X = np.array([covariates(meta, q, s) for q, s in zip(first.id, first.season)])
    Y = first[COMP].to_numpy()
    Rs = np.array([_R(Sig, n, c) for n, c in zip(first.n, first.ncp)])
    # prior covariance = lambda_g * Tall (shape from the well-estimated Tall), lambda by moments per group
    unk = X[:, 1] > 0.5
    lam = {True: 1.0, False: 1.0}
    B = np.zeros((X.shape[1], 4))
    for _ in range(8):
        T0v = np.where(unk, lam[True], lam[False])[:, None] * np.diag(Tall)[None, :]
        for c in range(4):
            wt = 1.0 / (T0v[:, c] + Rs[:, c, c])
            Xw = X * wt[:, None]
            B[:, c] = np.linalg.solve(X.T @ Xw + 1e-6 * np.eye(X.shape[1]), Xw.T @ Y[:, c])
        res = Y - X @ B
        for gflag in (True, False):
            k = (unk == gflag) & (first.n.to_numpy() >= 100)
            if k.sum() >= 8:
                ex = (res[k] ** 2).mean(0) - Rs[k][:, range(4), range(4)].mean(0)
                lam[gflag] = float(np.clip(np.mean(ex / np.diag(Tall)), 0.1, 3.0))
    log(f"  H_{upto}: n_debut={len(first)} (vets {unk.sum()})  lambda_vet={lam[True]:.2f} lambda_new={lam[False]:.2f} "
        f"sd_Tall_epa={math.sqrt(Tall[0, 0]):.3f}")
    H = Hyper(Sig, Tall, B, lam[False] * Tall, 0.01, 0.25, {})
    H.T0_vet = lam[True] * Tall
    hmap = {s: H for s in range(2012, upto)}
    obs = d
    best = None
    for rnd in range(2):
        for a in GRID_A:
            for b in GRID_B:
                H.a, H.b = a, b
                ll = run_filter(obs, meta, hmap, score=True)["ll"]
                if best is None or ll > best[0]:
                    best = (ll, a, b)
        H.a, H.b = best[1], best[2]
        if rnd == 0:
            H.D = estimate_development(obs, meta, hmap)
            best = None
    log(f"  H_{upto}: a={H.a} b={H.b} ll/game={best[0]:.4f}  D_epa=" +
        ", ".join(f"{k}:{v[0]:+.3f}" for k, v in sorted(H.D.items())))
    return H


# ============================================================================ Kalman filter
def _prior(meta, qb, season, H):
    x = covariates(meta, qb, season)
    return H.B.T @ x, (H.T0_vet if x[1] > 0.5 else H.T0).copy()


def _rookie(meta, qb, season):
    if qb in meta.index and pd.notna(meta.at[qb, "rookie_season"]):
        return int(meta.at[qb, "rookie_season"])
    return season


def _propagate(m, P, last_season, season, career, rookie, hmap, use_d=True):
    m, P = m.copy(), P.copy()
    for ss in range(last_season + 1, season + 1):
        Hs = hmap[ss]
        if use_d:
            m = m + Hs.D.get(_bucket(ss - rookie, career < LOW_DB), 0.0)
        P = P + Hs.b * Hs.Tall
    Hc = hmap[season]
    return m, P + Hc.a * Hc.Tall


def diag_hmap(hmap):
    """Ablation: same hyperparameters with all cross-component covariances zeroed (EPA learns only from EPA)."""
    import copy
    cache, out = {}, {}
    for s, H in hmap.items():
        if id(H) not in cache:
            H2 = copy.copy(H)
            for nm in ("Sig", "Tall", "T0", "T0_vet"):
                setattr(H2, nm, np.diag(np.diag(getattr(H, nm))))
            cache[id(H)] = H2
        out[s] = cache[id(H)]
    return out


def run_filter(obs, meta, hmap, score=False, use_d=True, keep=False):
    """Sequential filter over QB-games. Returns predictive LL (EPA component) and optional per-game states."""
    ll = 0.0; ng = 0
    pre_rows, post = [], {}
    for qb, d in obs.groupby("id", sort=False):
        rookie = None; m = P = None; last = None; career = 0
        hist = []
        for r in d.itertuples(index=False):
            s = r.season
            if m is None:
                rookie = _rookie(meta, qb, s)
                m, P = _prior(meta, qb, s, hmap[s])
                P = P + hmap[s].a * hmap[s].Tall
            else:
                m, P = _propagate(m, P, last, s, career, rookie, hmap, use_d)
            R = _R(hmap[s].Sig, r.n, r.ncp)
            y = np.array([r.epa, r.sr, r.cpoe if r.ncp > 0 else np.nan, r.sack])
            Se = P[0, 0] + R[0, 0]
            v = r.epa - m[0]
            if score:
                ll += -0.5 * (math.log(2 * math.pi * Se) + v * v / Se); ng += 1
            if keep:
                pre_rows.append((qb, r.game_id, r.gameday, s, r.n, r.epa, m[0], math.sqrt(P[0, 0]), Se, career))
            idx = [0, 1, 3] if r.ncp == 0 else [0, 1, 2, 3]
            Si = P[np.ix_(idx, idx)] + R[np.ix_(idx, idx)]
            K = P[:, idx] @ np.linalg.inv(Si)
            m = m + K @ (y[idx] - m[idx])
            P = P - K @ P[idx, :]
            P = (P + P.T) / 2
            last = s; career += r.n
            if keep:
                hist.append((r.gameday, s, m.copy(), P.copy(), career))
        if keep:
            post[qb] = (rookie, hist)
    out = {"ll": ll / max(ng, 1), "n": ng}
    if keep:
        out["pre"] = pd.DataFrame(pre_rows, columns=["qb_id", "game_id", "gameday", "season", "n", "epa_obs",
                                                     "bq", "bq_sd", "pred_var", "career_db"])
        out["post"] = post
    return out


def estimate_development(obs, meta, hmap) -> dict:
    """Mean change of talent at a season boundary, by years-in-league bucket x career-backup flag."""
    f = run_filter(obs, meta, hmap, keep=True, use_d=False)
    rows = []
    for qb, (rookie, hist) in f["post"].items():
        last_by_season = {}
        for gd, s, m, P, car in hist:
            last_by_season[s] = (m, P, car)
        seasons = sorted(last_by_season)
        for s0, s1 in zip(seasons[:-1], seasons[1:]):
            if s1 != s0 + 1:
                continue
            m, P, car = last_by_season[s0]
            dd = obs[(obs.id == qb) & (obs.season == s1)]
            n = dd.n.sum(); ncp = dd.ncp.sum()
            if n < 50 or ncp == 0:
                continue
            ybar = np.array([np.average(dd.epa, weights=dd.n), np.average(dd.sr, weights=dd.n),
                             np.average(dd.cpoe.fillna(0), weights=dd.ncp.clip(lower=1e-9)), np.average(dd.sack, weights=dd.n)])
            Rr = _R(hmap[s1].Sig, n, ncp)
            var = np.diag(P) + hmap[s1].b * np.diag(hmap[s1].Tall) + np.diag(Rr)
            rows.append((_bucket(s1 - rookie, car < LOW_DB), *(ybar - m), *(1 / var)))
    t = pd.DataFrame(rows, columns=["bucket"] + COMP + [c + "_w" for c in COMP])
    D = {}
    for bkt, g in t.groupby("bucket"):
        if len(g) >= 8:
            D[bkt] = np.array([np.average(g[c], weights=g[c + "_w"]) for c in COMP])
    return D


def build_hmap(qg, covs, meta, log=print, fit_seasons=FIT_SEASONS) -> dict:
    Hs = {s: fit_hyper(qg, covs, meta, s, log) for s in fit_seasons}
    hmap = {}
    for s in range(2012, 2027):
        hmap[s] = Hs[min(max(s, fit_seasons[0]), fit_seasons[-1])]
    return hmap, Hs


def pregame(f_post, meta, hmap, qb, season, gameday):
    """Posterior (mean, sd of EPA; full m) for qb before a game on `gameday` (strictly earlier games only)."""
    if qb is None or (isinstance(qb, float) and np.isnan(qb)):
        return np.nan, np.nan, 0
    if qb not in f_post:
        m, P = _prior(meta, qb, season, hmap[season])
        return m[0], math.sqrt(P[0, 0] + hmap[season].a * hmap[season].Tall[0, 0]), 0
    rookie, hist = f_post[qb]
    k = None
    for i in range(len(hist) - 1, -1, -1):
        if hist[i][0] < gameday:
            k = i; break
    if k is None:
        m, P = _prior(meta, qb, season, hmap[season])
        return m[0], math.sqrt(P[0, 0] + hmap[season].a * hmap[season].Tall[0, 0]), 0
    gd, s, m, P, car = hist[k]
    m2, P2 = _propagate(m, P, s, season, car, rookie, hmap)
    return m2[0], math.sqrt(P2[0, 0]), car


def bayes_features(df: pd.DataFrame, f_post, meta, hmap) -> pd.DataFrame:
    out = df[["game_id", "season", "gameday", "completed", "home_team", "away_team", "home_qb_id", "away_qb_id"]].copy()
    for side in ("home", "away"):
        vals = [pregame(f_post, meta, hmap, q, int(s), gd) for q, s, gd in
                zip(out[f"{side}_qb_id"], out.season, out.gameday)]
        out[f"{side}_bq"] = [v[0] for v in vals]
        out[f"{side}_bq_sd"] = [v[1] for v in vals]
        out[f"{side}_career_db"] = [v[2] for v in vals]
    # change vs the team's recent starters (same construction as features.attach_qb_change)
    long = pd.concat([
        out[["game_id", "gameday", "home_team", "home_bq", "completed"]].set_axis(["game_id", "gameday", "team", "q", "completed"], axis=1),
        out[["game_id", "gameday", "away_team", "away_bq", "completed"]].set_axis(["game_id", "gameday", "team", "q", "completed"], axis=1),
    ]).sort_values(["team", "gameday", "game_id"])
    played = long.q.where(long.completed)
    long["base"] = played.groupby(long.team).transform(lambda x: x.shift(1).ewm(halflife=F.TEAM_HALFLIFE, ignore_na=True).mean())
    long["chg"] = (long.q - long.base).fillna(0.0)
    for side in ("home", "away"):
        mm = long[["game_id", "team", "chg"]].rename(columns={"team": f"{side}_team", "chg": f"{side}_bq_change"})
        out = out.merge(mm, on=["game_id", f"{side}_team"], how="left")
    out["bq_diff"] = out.home_bq - out.away_bq
    out["bq_change_diff"] = out.home_bq_change - out.away_bq_change
    out["bq_var_sum"] = out.home_bq_sd ** 2 + out.away_bq_sd ** 2
    return out.drop(columns=["season", "gameday", "completed", "home_team", "away_team", "home_qb_id", "away_qb_id"])


def compute_all(df_base, max_gameday=None, log=print):
    p = load_plays(max_gameday)
    qg = qb_games(p)
    covs = play_cov_by_season(p)
    meta = qb_meta()
    hmap, Hs = build_hmap(qg, covs, meta, log)
    f = run_filter(qg, meta, hmap, keep=True, score=True)
    bf = bayes_features(df_base, f["post"], meta, hmap)
    return bf, f, Hs, meta, hmap, qg


def ablation_pre(qg, meta, hmap):
    f = run_filter(qg, meta, diag_hmap(hmap), keep=True)
    return f["pre"][["qb_id", "game_id", "bq"]].rename(columns={"bq": "bq_epa_only"})


# ============================================================================ evaluation helpers
def base_features() -> pd.DataFrame:
    p = SCR / "base_df.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import experiment as E
    _, df = E.run("baseline")
    df.to_parquet(p)
    return df


def walk_forward(df, feats, seasons, alpha=M.RIDGE_ALPHA):
    """Same as experiment.run / model.fit_predict, but returns margin, sigma and points-per-unit coefficients."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    parts = []
    for s in seasons:
        tr = M.train_rows(df, before_season=s); te = df[(df.season == s) & df.home_win.notna()].copy()
        y = (tr.home_score - tr.away_score).values
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(tr[feats], y)
        te["mu"] = m.predict(te[feats])
        te["sigma"] = float(np.sqrt(np.mean((y - m.predict(tr[feats])) ** 2)))
        coef = dict(zip(feats, m[-1].coef_ / m[0].scale_))
        te["k_qb"] = sum(coef.get(c, 0.0) for c in ("qb_diff", "qb_change_diff", "bq_diff", "bq_change_diff"))
        if "bq_var_sum" in tr:
            te["vbar_train"] = float(tr.bq_var_sum.mean())
        parts.append(te)
    return pd.concat(parts)


def probs(wf, c=0.0):
    sig2 = wf.sigma ** 2 + c * (wf.k_qb ** 2) * (wf.bq_var_sum - wf.vbar_train)
    return M._norm_cdf(wf.mu / np.sqrt(np.maximum(sig2, (0.8 * wf.sigma) ** 2)))


def ll_vec(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 5:
        return float("nan")
    idx = rng.integers(0, len(d), (n, len(d)))
    return float(d[idx].mean(1).std())


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return {"n": int(len(x))}
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi       # HC0
    se = math.sqrt(V[1, 1])
    return {"n": int(len(x)), "slope": round(float(b[1]), 4), "se": round(se, 4), "t": round(float(b[1] / se), 2)}


# ============================================================================ stages
def stage_dev():
    import qb_timing as QT
    log = print
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds")}
    df = base_features()
    log("fitting hyperparameters + filter ...")
    bf, f, Hs, meta, hmap, qg = compute_all(df, log=log)
    bf.to_parquet(SCR / "bayes_feats.parquet")
    res["hyper"] = {str(s): H.to_json() for s, H in Hs.items()}
    res["priors_examples"] = prior_examples(Hs[2020], meta)

    # ---------- direct check: next-game EPA/dropback prediction (Bayes vs production rating)
    pre = f["pre"]
    qbr = F.qb_ratings(_pbp_for_prod(), F.prepare_schedule(pd.read_parquet(RAW / "games.parquet")))
    k = pre[["gameday", "qb_id"]].reset_index().sort_values("gameday")
    mm = pd.merge_asof(k, qbr, on="gameday", by="qb_id", allow_exact_matches=False)
    pre["prod"] = mm.set_index("index").qb_rating.reindex(pre.index).fillna(F.QB_PRIOR)
    pre = pre.merge(ablation_pre(qg, meta, hmap), on=["qb_id", "game_id"], how="left")
    res["next_game_epa"] = next_game_table(pre)

    # ---------- (a) walk-forward ridge
    d = df.merge(bf, on="game_id", how="left")
    qt = QT.team_games(2025)
    chg = qt.groupby("game_id").qb_change.any()
    new = qt.assign(nw=qt.kind.eq("new")).groupby("game_id").nw.any()
    d["qb_change_game"] = d.game_id.map(chg).fillna(False).astype(bool)
    d["new_qb_game"] = d.game_id.map(new).fillna(False).astype(bool)
    d["low_db_game"] = (d.home_career_db < LOW_DB) | (d.away_career_db < LOW_DB)
    d["fw_bq_diff"] = d.final_week * d.bq_diff.fillna(0)
    for c in ("bq_diff", "bq_change_diff", "fw_bq_diff"):
        d[c] = d[c].fillna(0.0)
    base = list(F.FEATURES)
    rep = {"qb_diff": "bq_diff", "fw_qb_diff": "fw_bq_diff"}
    variants = {
        "P0_production": base,
        "B1_replace_qb_diff": [rep.get(c, c) for c in base],
        "B2_replace_qb_diff_and_change": [rep.get(c, c) if c != "qb_change_diff" else "bq_change_diff" for c in base],
        "B3_add_bq_diff": base + ["bq_diff"],
        "B4_add_bq_diff_and_change": base + ["bq_diff", "bq_change_diff"],
    }
    wfs = {k: walk_forward(d, v, list(VAL) + list(HOLD)) for k, v in variants.items()}
    preds = {}
    for k, wf in wfs.items():
        preds[k] = probs(wf.assign(bq_var_sum=0.0, vbar_train=0.0), 0.0)
    # uncertainty-widened sigma, c tuned on validation (on the best mean-variant by validation LL)
    y = wfs["P0_production"].home_win.values
    val_mask = wfs["P0_production"].season.isin(VAL).values
    vll = {k: float(ll_vec(y[val_mask], p[val_mask]).mean()) for k, p in preds.items()}
    best_mean = min([k for k in vll if k != "P0_production"], key=vll.get)
    cgrid = {}
    for c in (0.0, 0.5, 1.0, 2.0, 4.0):
        p = probs(wfs[best_mean], c)
        cgrid[c] = float(ll_vec(y[val_mask], p[val_mask]).mean())
    cbest = min(cgrid, key=cgrid.get)
    preds[f"B5_{best_mean.split('_')[0]}+sd_widen(c={cbest})"] = probs(wfs[best_mean], cbest)
    res["ridge"] = {"variants": {k: v for k, v in variants.items() if k != "P0_production"},
                    "val_ll": {k: round(v, 5) for k, v in vll.items()}, "best_mean_variant_by_val": best_mean,
                    "sd_widen_val_grid": cgrid, "c_chosen": cbest,
                    "table": ll_table(wfs["P0_production"], preds)}
    # coefficient check (points per 0.1 EPA/dropback) in the last holdout fit
    res["ridge"]["k_qb_points_per_0.1epa"] = {k: round(float(w.k_qb.iloc[-1]) * 0.1, 2) for k, w in wfs.items()}

    # ---------- (b1) ATS residual at the close (2013-2025)
    d["ats_resid"] = (d.home_score - d.away_score) - d.spread_line
    d["bq_rev_pts"] = DB_PER_GAME * ((d.home_bq - d.home_qb_rating) - (d.away_bq - d.away_qb_rating))
    mu_b, mu_p = wfs[best_mean].set_index("game_id").mu, wfs["P0_production"].set_index("game_id").mu
    d["mu_bayes"], d["mu_prod"] = d.game_id.map(mu_b), d.game_id.map(mu_p)
    d["mu_b1"] = d.game_id.map(wfs["B1_replace_qb_diff"].set_index("game_id").mu)
    sc = d[d.season.between(2013, 2025) & d.home_win.notna()]
    res["corr_bq_diff_vs_qb_diff"] = {"all": round(float(sc.bq_diff.corr(sc.qb_diff)), 4),
                                      "qb_change_games": round(float(sc[sc.qb_change_game].bq_diff.corr(sc[sc.qb_change_game].qb_diff)), 4),
                                      "sd_bq_rev_pts_qb_change": round(float(sc[sc.qb_change_game].bq_rev_pts.std()), 3),
                                      "sd_mu_bayes_minus_mu_prod": round(float((sc.mu_bayes - sc.mu_prod).std()), 3),
                                      "sd_mu_b1_minus_mu_prod": round(float((sc.mu_b1 - sc.mu_prod).std()), 3)}
    ats = {}
    for lab, seas in (("2013-2019", range(2013, 2020)), ("2020-2025", range(2020, 2026)), ("2013-2025", range(2013, 2026))):
        x = d[d.season.isin(seas) & d.home_win.notna() & d.spread_line.notna()]
        ats[lab] = {"qb_change_games: resid ~ bayes revision vs production (pts)": ols(x[x.qb_change_game].bq_rev_pts, x[x.qb_change_game].ats_resid),
                    "all_games: resid ~ bayes revision (pts)": ols(x.bq_rev_pts, x.ats_resid),
                    "low_db_games: resid ~ bayes revision (pts)": ols(x[x.low_db_game].bq_rev_pts, x[x.low_db_game].ats_resid)}
        x2 = x[x.mu_bayes.notna()]
        xc = x2[x2.qb_change_game]
        ats[lab]["qb_change_games: resid ~ (mu_bayes - spread)"] = ols(xc.mu_bayes - xc.spread_line, xc.ats_resid)
        ats[lab]["qb_change_games: resid ~ (mu_prod - spread)"] = ols(xc.mu_prod - xc.spread_line, xc.ats_resid)
        ats[lab]["qb_change_games: resid ~ (mu_bayes - mu_prod)"] = ols(xc.mu_bayes - xc.mu_prod, xc.ats_resid)
    res["ats_close"] = ats
    d[["game_id", "season", "mu_bayes", "mu_prod", "mu_b1", "qb_change_game", "new_qb_game", "low_db_game",
       "home_bq", "away_bq", "home_bq_sd", "away_bq_sd", "bq_rev_pts"]].to_parquet(SCR / "game_table.parquet")

    # ---------- (b2) early week, 2020-22 dev
    res["early_dev_2020_2022"] = early(d, qt, holdout=False)
    res["frozen_rules"] = FROZEN_RULES | {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
                                          "dev_result_known": True}
    # ---------- (c) leakage
    res["leakage"] = leakage_test(df, bf)
    JSON.write_text(json.dumps(M.clean_json(res), indent=2, default=str))
    log("wrote", JSON)


FROZEN_RULES = {
    "note": ("Thresholds fixed a priori (not tuned). E1/M1 were written before any result; after the first dev run showed "
             "the validation-chosen variant (B3) barely moves the margin (E1: 0 bets), M2/M3/E3/E4 were added using "
             "a-priori thresholds and the whole list is frozen here, before the 2023-25 holdout is run once. "
             "All CLV is price-based vs closing_fair().mu_close_all at the first early-week snapshot "
             "(>= 4h after both teams' previous kickoffs, <= 9 days out), best allowed book. Hindsight caveat: "
             "starters are the ACTUAL starters, which the early-week market may not know yet."),
    "E1": "QB-change games: bet side of mu_bayes - mu_prod (best-val variant B3) when |rev| >= 0.5 pt",
    "E2": "QB-change games: bet side with |mu_bayes - m_early| >= 1.5 pts (reference E2p: same with mu_prod)",
    "E3": "QB-change games: bet side of bq_rev_pts (38 x [Bayes - production QB rating diff]) when |rev| >= 1.0 pt",
    "E4": "QB-change games: bet side of mu_b1 - mu_prod (B1: Bayes replaces qb_diff) when |rev| >= 0.5 pt",
    "M1": "QB-change games: early->close consensus move (home pts) ~ mu_bayes - mu_prod, slope > 0",
    "M2": "QB-change games: move ~ bq_rev_pts, slope > 0",
    "M3": "QB-change games: move ~ mu_b1 - mu_prod, slope > 0",
    "success_criterion": "one-sided p < 0.05/7 (Bonferroni over E1-E4, M1-M3); E2 must also beat E2p to credit the Bayes rating",
}


def _pbp_for_prod():
    fr = []
    for s in range(2012, 2027):
        x = pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=["game_id", "qb_dropback", "qb_epa", "id"])
        fr.append(x[x.qb_dropback.eq(1)])
    return pd.concat(fr, ignore_index=True)


def next_game_table(pre: pd.DataFrame) -> dict:
    """Dropback-weighted MSE and Gaussian LL of next-game EPA/dropback: production EWMA vs Bayes posterior."""
    out = {}
    pre = pre[pre.n >= 10].copy()
    pre["sev"] = np.where(pre.career_db < LOW_DB, "career<300db", "career>=300db")
    for lab, seas in (("val 2015-19", VAL), ("hold 2020-25", HOLD)):
        x = pre[pre.season.isin(seas)]
        for grp, g in [("all", x)] + list(x.groupby("sev")):
            w = g.n.values
            mse_b = np.average((g.epa_obs - g.bq) ** 2, weights=w)
            mse_p = np.average((g.epa_obs - g["prod"]) ** 2, weights=w)
            mse_a = np.average((g.epa_obs - g.bq_epa_only) ** 2, weights=w)
            # noise floor: per-play variance / n
            out[f"{lab} | {grp}"] = {"qb_games": int(len(g)), "wmse_bayes": round(float(mse_b), 5),
                                     "wmse_prod": round(float(mse_p), 5),
                                     "wmse_bayes_epa_only(ablation)": round(float(mse_a), 5),
                                     "corr_bayes": round(float(np.corrcoef(g.epa_obs, g.bq)[0, 1]), 4),
                                     "corr_prod": round(float(np.corrcoef(g.epa_obs, g["prod"])[0, 1]), 4),
                                     "mean_err_bayes": round(float(np.average(g.epa_obs - g.bq, weights=w)), 4),
                                     "mean_err_prod": round(float(np.average(g.epa_obs - g["prod"], weights=w)), 4),
                                     "z_sd_bayes(1=calibrated)": round(float(np.sqrt(np.average(
                                         (g.epa_obs - g.bq) ** 2 / g.pred_var, weights=None))), 3)}
    return out


def ll_table(wf0, preds):
    y = wf0.home_win.values
    masks = {"all": np.ones(len(wf0), bool), "qb_change_game": wf0.qb_change_game.values,
             "new_qb_game": wf0.new_qb_game.values, "starter_<300_career_db": wf0.low_db_game.values}
    tab = {}
    for per, seas in (("val 2015-19", VAL), ("hold 2020-25", HOLD)):
        pm = wf0.season.isin(seas).values
        for mk, mm in masks.items():
            k = pm & mm
            l0 = ll_vec(y[k], preds["P0_production"][k])
            row = {"n": int(k.sum()), "P0_production": round(float(l0.mean()), 4),
                   "vegas": round(float(np.nanmean(ll_vec(y[k], wf0.vegas_home_prob.values[k]))), 4)}
            for name, p in preds.items():
                if name == "P0_production":
                    continue
                l1 = ll_vec(y[k], p[k])
                row[name] = round(float(l1.mean()), 4)
                row[name + " diff±se"] = f"{l1.mean() - l0.mean():+.4f}±{boot_diff(l1, l0):.4f}"
            tab[f"{per} | {mk}"] = row
    return tab


def prior_examples(H, meta):
    ex = {}
    cases = [("1st-round top-10 rookie (pick 5, age 22)", 5, 0, 22.0, False),
             ("late 1st-round rookie (pick 25, age 22.5)", 25, 0, 22.5, False),
             ("3rd-round rookie (pick 80)", 80, 0, 22.8, False),
             ("6th-round rookie (pick 190)", 190, 0, 23.0, False),
             ("undrafted rookie", 300, 0, 23.0, False),
             ("late-round backup, first snaps in year 4 (pick 200)", 200, 3, 23.0, False),
             ("undrafted backup, first snaps in year 5", 300, 4, 23.0, False)]
    for lab, pick, yrs, age, unk in cases:
        x = np.array([1.0, float(unk), math.log(pick) - math.log(64), min(yrs, 10) - 1.0, age - 22.8])
        m = H.B.T @ x
        ex[lab] = {"epa_mean": round(float(m[0]), 3), "epa_sd": round(math.sqrt(H.T0[0, 0]), 3),
                   "sr": round(float(m[1]), 3), "cpoe": round(float(m[2]) * 100, 1), "sack": round(float(m[3]), 3)}
    return ex


def early(d, qt, holdout: bool) -> dict:
    import qb_timing as QT
    from nflpred import spread_bets as SB
    r = SB.load_rules()
    sig, w = r["margin"]["sigma"], r["_weights"]
    snap = QT.snapshots(holdout)
    seasons = range(2023, 2026) if holdout else range(2020, 2023)
    tg = qt[qt.season.isin(seasons)]
    home = tg[tg.side == "home"].set_index("game_id"); away = tg[tg.side == "away"].set_index("game_id")
    ready = pd.concat([home.prev_kick, away.prev_kick], axis=1).max(axis=1) + pd.Timedelta(hours=4)
    snap["ready"] = snap.game_id.map(ready)
    snap = snap[(snap.ready.isna() | (snap.requested_ts >= snap.ready)) & (snap.hours_before <= 9 * 24)]
    first_ts = snap.groupby("game_id").requested_ts.transform("min")
    s0 = snap[snap.requested_ts == first_ts].copy()
    s0["ev_cons"] = QT._ev(s0.m_cons, s0.side.values, s0.point, s0.sp_price, sig, w)
    s0["clv"] = QT._ev(s0.mu_close_all, s0.side.values, s0.point, s0.sp_price, sig, w)
    s0["clv_sharp"] = QT._ev(s0.mu_close_sharp, s0.side.values, s0.point, s0.sp_price, sig, w)
    g = s0.drop_duplicates("game_id").set_index("game_id")[["m_cons", "mu_close_all", "hours_before"]]
    g = g.join(d.set_index("game_id")[["mu_bayes", "mu_prod", "mu_b1", "bq_rev_pts", "qb_change_game", "new_qb_game",
                                       "low_db_game"]], how="inner")
    g["move_home"] = g.mu_close_all - g.m_cons
    g["rev"] = g.mu_bayes - g.mu_prod
    out = {"n_games": int(len(g)), "median_hours_before_early": float(g.hours_before.median())}
    c = g[g.qb_change_game]
    out["M1 move ~ (mu_bayes - mu_prod), qb_change games"] = ols(c.rev, c.move_home)
    out["move ~ (mu_bayes - m_early), qb_change games"] = ols(c.mu_bayes - c.m_cons, c.move_home)
    out["move ~ (mu_prod - m_early), qb_change games"] = ols(c.mu_prod - c.m_cons, c.move_home)
    out["M1 all games"] = ols(g.rev, g.move_home)
    out["M2 move ~ bq_rev_pts (Bayes minus production QB rating, pts), qb_change games"] = ols(c.bq_rev_pts, c.move_home)
    out["M3 move ~ (mu_b1 - mu_prod), qb_change games"] = ols(c.mu_b1 - c.mu_prod, c.move_home)
    for k in [k for k in out if k.startswith(("M1 move", "M2", "M3"))]:
        if "t" in out[k]:
            out[k]["p_one_sided"] = round(0.5 * math.erfc(out[k]["t"] / math.sqrt(2)), 4)

    def bet(sel: pd.Series, side_home: pd.Series):
        gs = sel[sel].index
        want = pd.Series(np.where(side_home.loc[gs], "home", "away"), index=gs)
        b = s0[s0.game_id.isin(gs)].copy()
        b = b[b.side.values == b.game_id.map(want).values]
        b = b.sort_values(["game_id", "ev_cons"], ascending=[True, False]).groupby("game_id").head(1)
        clv = b.clv.dropna()
        if len(clv) < 3:
            return {"bets": int(len(clv))}
        se = clv.std(ddof=1) / math.sqrt(len(clv))
        return {"bets": int(len(clv)), "clv": round(float(clv.mean()), 4), "se": round(float(se), 4),
                "t": round(float(clv.mean() / se), 2), "p_one_sided": round(0.5 * math.erfc(clv.mean() / se / math.sqrt(2)), 4),
                "clv_sharp": round(float(b.clv_sharp.mean()), 4), "beat_close": round(float((clv > 0).mean()), 3)}

    qc = g.qb_change_game
    out["E1 bayes revision side, |rev|>=0.5"] = bet(qc & (g.rev.abs() >= 0.5), g.rev > 0)
    out["E2 mu_bayes vs early line >= 1.5"] = bet(qc & ((g.mu_bayes - g.m_cons).abs() >= 1.5), g.mu_bayes > g.m_cons)
    out["E2p mu_prod vs early line >= 1.5 (reference)"] = bet(qc & ((g.mu_prod - g.m_cons).abs() >= 1.5), g.mu_prod > g.m_cons)
    out["E3 side of bq_rev_pts, |rev|>=1.0 pt"] = bet(qc & (g.bq_rev_pts.abs() >= 1.0), g.bq_rev_pts > 0)
    out["E4 side of mu_b1 - mu_prod, |rev|>=0.5 pt"] = bet(qc & ((g.mu_b1 - g.mu_prod).abs() >= 0.5), g.mu_b1 > g.mu_prod)
    out["N0 all qb_change games, both sides (vig reference)"] = {
        "home": bet(qc, pd.Series(True, index=g.index)), "away": bet(qc, pd.Series(False, index=g.index))}
    return out


def leakage_test(df, bf_full, cutoffs=("2017-10-08", "2022-10-09", "2023-12-24", "2024-09-08")) -> dict:
    """Erase every result (pbp + scores) from `cutoff` on, refit hyperparameters + filter, compare features."""
    cols = ["bq_diff", "bq_change_diff", "home_bq", "away_bq", "home_bq_sd", "away_bq_sd"]
    out = {}
    for c in cutoffs:
        cut = pd.Timestamp(c)
        dm = df.copy()
        dm.loc[dm.gameday >= cut, "completed"] = False
        bf, *_ = compute_all(dm, max_gameday=cut, log=lambda *a: None)
        ids = df.loc[df.gameday == cut, "game_id"]
        a = bf_full.set_index("game_id").loc[ids, cols]; b = bf.set_index("game_id").loc[ids, cols]
        diff = float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))
        out[c] = {"games": int(len(ids)), "max_abs_diff": diff, "pass": bool(diff < 1e-9)}
        print("leakage", c, out[c])
    return out


def stage_holdout():
    import qb_timing as QT
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("holdout locked: set EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES after the dev stage froze the rules")
    res = json.loads(JSON.read_text())
    if "frozen_rules" not in res:
        raise SystemExit("run dev first (freezes rules)")
    if "early_holdout_2023_2025" in res:
        raise SystemExit("holdout already run once; not re-running")
    d = pd.read_parquet(SCR / "game_table.parquet")
    qt = QT.team_games(2025)
    res["early_holdout_2023_2025"] = early(d, qt, holdout=True)
    JSON.write_text(json.dumps(M.clean_json(res), indent=2, default=str))
    print(json.dumps(res["early_holdout_2023_2025"], indent=1))


VERDICT = ("No gain; do not adopt. The Bayes posterior is a sound QB estimate (unbiased where production runs about +0.02 "
           "EPA/db low, nearly calibrated SD with z-SD 1.04-1.12, priors that make sense), but it is 0.94-0.95 correlated with the production EWMA. Swapping it in or adding it does not improve walk-forward "
           "log loss overall or on QB-change / <300-dropback games. Every difference is within about 2 SE, and the subset signs flip "
           "between validation and holdout. The best holdout cell, B4 on new-QB games at -0.0021±0.0010, was +0.0011 in validation. Using the posterior SD to widen σ is rejected on validation (c = 0). The "
           "fast-stabilising components add nothing over an EPA-only filter. Against the market, the Bayes-minus-production QB "
           "revision predicts the early-week to close line move on QB-change games (dev t = 3.7, holdout t = 2.3, p = 0.011). "
           "That does not clear the pre-registered Bonferroni bar, and the bettable version (E3) has −0.3% CLV in the holdout. "
           "E2 clears the bar nominally (+3.1% CLV, p = 0.007). It is hindsight on the actual starter, though, and production "
           "gets the same (E2p +2.9%), so the Bayes rating adds nothing to it. M1 flipped sign between dev and holdout. "
           "The closing line already has this information.")


def _md_table(rows: list[dict], cols: list[str]) -> str:
    esc = lambda v: str(v).replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _bet_row(name, b):
    if not b or b.get("bets", 0) < 3:
        return {"rule": name, "bets": (b or {}).get("bets", 0)}
    return {"rule": name, "bets": b["bets"], "CLV": f"{b['clv']*100:+.1f}% ±{b['se']*100:.1f}",
            "p (1-sided)": b["p_one_sided"], "sharp-close CLV": f"{b['clv_sharp']*100:+.1f}%", "beat close": b["beat_close"]}


def _ols_row(name, o):
    if "slope" not in o:
        return {"test": name, "n": o.get("n", 0)}
    return {"test": name, "n": o["n"], "slope": f"{o['slope']:+.3f} ±{o['se']:.3f}", "t": o["t"],
            "p (1-sided)": o.get("p_one_sided", "")}


def write_report():
    r = json.loads(JSON.read_text())
    H = r["hyper"]["2020"]
    L = [f"# Bayesian QB projection (hierarchical / empirical-Bayes Kalman)", "",
         "Script: `scripts/research/bayes_qb.py` (stages `dev`, `holdout`, `report`). Output: `output/research/bayes_qb.json`.",
         "", f"**Verdict:** {VERDICT}", "", "## Method", "",
         "- Latent talent per QB = (EPA/dropback, success rate, CPOE, sack rate). Prior at first appearance: N(B·x, λ·T), "
         "x = draft pick (log, UDFA = 300), years in league at debut, age at entry, pre-2012-veteran flag.",
         "- Observation per QB-game: component means with per-play noise covariance / n. The cross-covariances between "
         "talent components let faster-stabilising stats (success, CPOE, sacks) move the EPA estimate.",
         "- Dynamics: random walk with drift a·T per game and b·T per season, plus a season-boundary development shift d "
         "by years in league × career backup (< 300 career dropbacks).",
         "- Hyperparameters are walk-forward. H_s is fit on seasons < s (2015-19); H_2020 is fit on 2012-19 and frozen for 2020+.",
         "- Pre-game posterior mean (bq) and SD (bq_sd) for the **actual** starter (nflverse `*_qb_id`, the same hindsight as production).",
         "", "## Priors learned (H_2020, fit on 2012-2019)", "",
         f"Drift: a = {H['a_game_drift']} per game, b = {H['b_season_drift']} per season "
         f"(× between-QB talent covariance, SD of EPA talent = {math.sqrt(H['T_between_qb_season'][0][0]):.3f}). "
         f"Prior SD of a debut QB's EPA/db = {math.sqrt(H['prior_T0_new'][0][0]):.3f}, equivalent to about "
         f"{H['Sigma_per_play'][0][0] / H['prior_T0_new'][0][0]:.0f} dropbacks of data. For comparison, production uses a "
         "fixed −0.10 worth 150 dropbacks.", "",
         _md_table([{"QB type": k, "EPA/db mean": v["epa_mean"], "SD": v["epa_sd"], "success": v["sr"], "CPOE": v["cpoe"],
                     "sack%": round(v["sack"] * 100, 1)} for k, v in r["priors_examples"].items()],
                   ["QB type", "EPA/db mean", "SD", "success", "CPOE", "sack%"]), "",
         "Season-boundary development shift d (EPA/db; `_bk` = under 300 career dropbacks entering the season):", "",
         _md_table([{"bucket": k, "d_epa": f"{v['epa']:+.3f}"} for k, v in sorted(H["development_d"].items())], ["bucket", "d_epa"]),
         "", "## Next-game EPA/dropback prediction (QB-games with 10+ dropbacks, dropback-weighted)", "",
         _md_table([{"period | group": k, "n": v["qb_games"], "MSE Bayes": v["wmse_bayes"],
                     "MSE Bayes EPA-only": v["wmse_bayes_epa_only(ablation)"], "MSE production": v["wmse_prod"],
                     "bias Bayes": v["mean_err_bayes"], "bias prod": v["mean_err_prod"], "z-SD Bayes": v["z_sd_bayes(1=calibrated)"]}
                    for k, v in r["next_game_epa"].items()],
                   ["period | group", "n", "MSE Bayes", "MSE Bayes EPA-only", "MSE production", "bias Bayes", "bias prod", "z-SD Bayes"]),
         "", f"Correlation of bq_diff with production qb_diff: {r['corr_bq_diff_vs_qb_diff']}", "",
         "## (a) Walk-forward ridge log loss (tune 2015-19, holdout 2020-25)", "",
         "Variants: B1 = bq_diff replaces qb_diff (and fw_qb_diff); B2 = B1 plus the Bayes qb_change; B3 = add bq_diff; "
         "B4 = add bq_diff + Bayes change; B5 = best mean variant plus σ widened by QB posterior variance "
         f"(c chosen on validation = {r['ridge']['c_chosen']}). Best by validation: **{r['ridge']['best_mean_variant_by_val']}**. "
         "Cells are the log-loss difference versus production (negative = better) ± game-bootstrap SE.", ""]
    cols = ["period | subset", "n", "vegas", "P0_production"]
    names = [k for k in next(iter(r["ridge"]["table"].values())) if k.startswith("B") and not k.endswith(" diff±se")]
    rows = []
    for k, v in r["ridge"]["table"].items():
        row = {"period | subset": k, "n": v["n"], "vegas": v["vegas"], "P0_production": v["P0_production"]}
        for nm in names:
            row[nm.split("_")[0]] = v[nm + " diff±se"]
        rows.append(row)
    L += [_md_table(rows, cols + [nm.split("_")[0] for nm in names]), "",
          "## (b) Market tests", "", "### ATS residual vs the nflverse closing spread (home margin − spread_line)", "",
          "bq_rev_pts = 38 × [(Bayes − production) home QB − (Bayes − production) away QB]. mu = walk-forward ridge margin (2015+).", ""]
    rows = []
    for per, d in r["ats_close"].items():
        for k, o in d.items():
            rows.append(_ols_row(f"{per}: {k}", o))
    L += [_md_table(rows, ["test", "n", "slope", "t"]), ""]
    for lab, key in (("Early week, dev 2020-22", "early_dev_2020_2022"), ("Early week, HOLDOUT 2023-25 (run once)", "early_holdout_2023_2025")):
        if key not in r:
            L += [f"### {lab}: not run", ""]
            continue
        e = r[key]
        L += [f"### {lab}: {e['n_games']} games, early snapshot median {e['median_hours_before_early']:.0f}h before kickoff", "",
              _md_table([_ols_row(k, o) for k, o in e.items() if isinstance(o, dict) and "slope" in o],
                        ["test", "n", "slope", "t", "p (1-sided)"]), "",
              _md_table([_bet_row(k, b) for k, b in e.items() if k.startswith("E")] +
                        [_bet_row("N0 null: home side", e["N0 all qb_change games, both sides (vig reference)"]["home"]),
                         _bet_row("N0 null: away side", e["N0 all qb_change games, both sides (vig reference)"]["away"])],
                        ["rule", "bets", "CLV", "p (1-sided)", "sharp-close CLV", "beat close"]), ""]
    L += ["Frozen rules (before the holdout):", "", "```", json.dumps(r["frozen_rules"], indent=1), "```", "",
          "## (c) Leakage masking test", "",
          "All results from the cutoff on are erased (pbp and completion), hyperparameters and the filter are refit, "
          "and features for games on the cutoff date are compared.", "",
          _md_table([{"cutoff": k, **v} for k, v in r["leakage"].items()], ["cutoff", "games", "max_abs_diff", "pass"]), "",
          "## Caveats", "",
          "- Starters are the actual starters (nflverse), the same as production. Early-week CLV tests therefore assume the QB "
          "news is known at the early snapshot (hindsight; see qb_timing.md).",
          "- Career dropbacks count only 2012+ pbp. Pre-2012 veterans carry a separate prior and their own variance.",
          "- Warm-up seasons 2012-14 (training rows only, never scored) use H_2015, which is fit on 2012-14.",
          "- Opponent adjustment and the league-wide EPA level are not modelled. Both mostly cancel in a home-minus-away difference."]
    MD.write_text("\n".join(L) + "\n")
    print("wrote", MD)


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": stage_dev, "holdout": stage_holdout, "report": write_report}[st]()
