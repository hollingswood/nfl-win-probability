"""College football walk-forward ratings -> predicted home margin and total for every game.

Strictly walk-forward: games in a (season, week slot) are predicted only from games played in earlier
slots. Ratings are ridge regressions shrunk toward a preseason prior instead of toward zero.

  margin model : margin (capped) = HFA*(not neutral) + R_home - R_away
  points model : pts_for = mu + HFA_p + O_team - D_opp   (per team-game, two rows per game) -> total
  ppa model    : off PPA/play = mu + h + O_team + D_opp (CFBD advanced stats, garbage time excluded)
Preseason prior for season s: a*final_{s-1} + b*talent_z + c*final_{s-1}*(returning_pct - mean) (+ FCS mean),
coefficients fit on 2015-2020 only.

Output: output/research/cfb/preds.parquet (one row per game, FBS and FCS opponents).
Run:  PYTHONPATH=src python scripts/research/cfb/model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cfbpred import data as D  # noqa: E402

OUT = ROOT / "output" / "research" / "cfb"
YEARS = range(2014, 2027)
CAP = 28.0       # margin cap (blowouts carry little extra information)
LAM = 4.0        # ridge strength toward the prior, in "games"
LAM_P = 4.0
LAM_PPA = 6.0


def slot_order(g: pd.DataFrame) -> pd.Series:
    """Ordinal of the week slot inside the season: regular weeks, then postseason after them."""
    return g["week"] + np.where(g["season_type"] == "postseason", 30, 0)


def ridge_toward(teams: list[str], rows: list[tuple], y: np.ndarray, prior: np.ndarray, lam: float,
                 extra_cols: int, w: np.ndarray | None = None):
    """rows: list of (dict team_index->coef, extra feature vector). Solves for team effects (shrunk
    toward `prior`) and unpenalized extra coefficients."""
    n, k = len(rows), len(teams)
    X = np.zeros((n, k + extra_cols))
    for i, (tc, ex) in enumerate(rows):
        for j, c in tc.items():
            X[i, j] = c
        X[i, k:] = ex
    w = np.ones(n) if w is None else w
    base = np.concatenate([prior, np.zeros(extra_cols)])
    r = y - X @ base
    A = (X * w[:, None]).T @ X
    A[np.arange(k), np.arange(k)] += lam
    A[np.arange(k, k + extra_cols), np.arange(k, k + extra_cols)] += 1e-6
    b = (X * w[:, None]).T @ r
    sol = np.linalg.solve(A, b) + base
    return sol[:k], sol[k:]


def build():
    g = D.games(YEARS)
    g = g[g["home"].notna() & g["away"].notna()].copy()
    g["slot"] = slot_order(g)
    adv = D.adv_stats(YEARS)
    tal = D.yearly("talent", YEARS)[["year", "team", "talent"]].rename(columns={"year": "season"})
    ret = D.yearly("player_returning", YEARS)[["season", "team", "percentPPA"]]
    lines = D.lines(YEARS)
    return g, adv, tal, ret, lines


def run():
    g, adv, tal, ret, lines = build()
    teams = sorted(set(g["home"]) | set(g["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    div = {}
    for side in ("home", "away"):
        for t, d in zip(g[side], g[f"{side}_div"]):
            if d:
                div[t] = d
    adv_idx = adv.set_index(["game_id", "team"])
    final_m, final_o, final_d, final_po, final_pd = {}, {}, {}, {}, {}
    preds = []
    prior_rows = []  # for fitting prior coefficients
    coef = {"a": 0.62, "b": 0.0, "c": 0.0}  # refit after enough seasons (dev only)
    for season in YEARS:
        S = g[g["season"] == season].sort_values("start")
        if S.empty:
            continue
        # ---------------- preseason priors
        tz = tal[tal.season == season].set_index("team")["talent"]
        tz = (tz - tz.mean()) / tz.std() if len(tz) else tz
        rp = ret[ret.season == season].set_index("team")["percentPPA"]
        rp_mean = rp.mean() if len(rp) else 0.5
        last_m = final_m.get(season - 1, {})
        fcs_mean = np.mean([v for t, v in last_m.items() if div.get(t) == "fcs"]) if last_m else -20.0
        prior_m = np.zeros(len(teams))
        for t, i in ix.items():
            lm = last_m.get(t, fcs_mean if div.get(t) == "fcs" else 0.0)
            prior_m[i] = coef["a"] * lm + coef["b"] * float(tz.get(t, 0.0) if len(tz) else 0.0) \
                + coef["c"] * lm * (float(rp.get(t, rp_mean)) - rp_mean if len(rp) else 0.0)
            if div.get(t) == "fcs" and t not in last_m:
                prior_m[i] = fcs_mean * coef["a"] + (1 - coef["a"]) * fcs_mean
        lo, ld = final_o.get(season - 1, {}), final_d.get(season - 1, {})
        prior_o = np.array([0.6 * lo.get(t, 0.0) for t in teams])
        prior_d = np.array([0.6 * ld.get(t, 0.0) for t in teams])
        lpo, lpd = final_po.get(season - 1, {}), final_pd.get(season - 1, {})
        prior_po = np.array([0.6 * lpo.get(t, 0.0) for t in teams])
        prior_pd = np.array([0.6 * lpd.get(t, 0.0) for t in teams])

        slots = sorted(S["slot"].unique())
        played = S.iloc[0:0]
        Rm = prior_m.copy(); hfa = 2.5
        Ro, Rd = prior_o.copy(), prior_d.copy(); mu_p, hfa_p = 28.0, 1.2
        Po, Pd = prior_po.copy(), prior_pd.copy(); mu_ppa, h_ppa = 0.0, 0.0
        for sl in slots + [999]:
            cur = S[S["slot"] == sl]
            # predict this slot with ratings fit on earlier slots
            for r in cur.itertuples():
                h, a = ix[r.home], ix[r.away]
                nh = 0.0 if r.neutral else 1.0
                preds.append({"game_id": r.game_id, "season": season, "week": r.week, "season_type": r.season_type,
                              "start": r.start, "home": r.home, "away": r.away, "neutral": r.neutral,
                              "home_div": r.home_div, "away_div": r.away_div, "home_conf": r.home_conf,
                              "away_conf": r.away_conf, "conf_game": r.conf_game, "margin": r.margin, "total": r.total,
                              "games_played_h": int(((played.home == r.home) | (played.away == r.home)).sum()),
                              "games_played_a": int(((played.home == r.away) | (played.away == r.away)).sum()),
                              "m_pred": nh * hfa + Rm[h] - Rm[a],
                              "t_pred": 2 * mu_p + nh * hfa_p + (Ro[h] - Rd[a]) + (Ro[a] - Rd[h]),
                              "p_pred_h": mu_p + nh * hfa_p + Ro[h] - Rd[a], "p_pred_a": mu_p + Ro[a] - Rd[h],
                              "ppa_net": (Po[h] + Pd[a]) - (Po[a] + Pd[h]) + nh * h_ppa,
                              "prior_h": prior_m[h], "prior_a": prior_m[a], "elo_h": r.home_elo, "elo_a": r.away_elo})
            if sl == 999:
                break
            played = pd.concat([played, cur[cur["completed"] & cur["margin"].notna()]])
            if played.empty:
                continue
            # ---- margin ratings
            rows = [({ix[r.home]: 1.0, ix[r.away]: -1.0}, [0.0 if r.neutral else 1.0]) for r in played.itertuples()]
            y = played["margin"].clip(-CAP, CAP).to_numpy(float)
            Rm, ex = ridge_toward(teams, rows, y, prior_m, LAM, 1)
            hfa = float(ex[0])
            # ---- points ratings (offense O minus opponent defense D)
            k = len(teams)
            prow, py = [], []
            for r in played.itertuples():
                nh = 0.0 if r.neutral else 1.0
                prow.append(({ix[r.home]: 1.0, k + ix[r.away]: -1.0}, [1.0, nh])); py.append(r.home_pts)
                prow.append(({ix[r.away]: 1.0, k + ix[r.home]: -1.0}, [1.0, 0.0])); py.append(r.away_pts)
            both, exp_ = ridge_toward(teams + teams, prow, np.array(py, float),
                                      np.concatenate([prior_o, prior_d]) , LAM_P, 2)
            Ro, Rd = both[:k], both[k:]
            mu_p, hfa_p = float(exp_[0]), float(exp_[1])
            # ---- PPA ratings (offense vs opponent defense, per play)
            arow, ay = [], []
            for r in played.itertuples():
                for tm, op, home in ((r.home, r.away, not r.neutral), (r.away, r.home, False)):
                    try:
                        v = adv_idx.loc[(r.game_id, tm), "off_ppa"]
                    except KeyError:
                        continue
                    if v is None or pd.isna(v):
                        continue
                    arow.append(({ix[tm]: 1.0, k + ix[op]: 1.0}, [1.0, 1.0 if home else 0.0])); ay.append(float(v))
            if len(arow) > 20:
                bo, e2 = ridge_toward(teams + teams, arow, np.array(ay), np.concatenate([prior_po, prior_pd]), LAM_PPA, 2)
                Po, Pd = bo[:k], bo[k:]
                mu_ppa, h_ppa = float(e2[0]), float(e2[1]) * 2
        final_m[season] = {t: Rm[i] for t, i in ix.items()}
        final_o[season] = {t: Ro[i] for t, i in ix.items()}
        final_d[season] = {t: Rd[i] for t, i in ix.items()}
        final_po[season] = {t: Po[i] for t, i in ix.items()}
        final_pd[season] = {t: Pd[i] for t, i in ix.items()}
        # prior-fit rows (FBS only) and refit on dev seasons 2015-2020 only
        if season - 1 in final_m and season <= 2020:
            for t in teams:
                if div.get(t) == "fbs" and t in final_m[season - 1]:
                    lm = final_m[season - 1][t]
                    prior_rows.append({"season": season, "y": final_m[season][t], "last": lm,
                                       "tz": float(tz.get(t, np.nan)) if len(tz) else np.nan,
                                       "ret": (float(rp.get(t, np.nan)) - rp_mean) * lm if len(rp) else np.nan})
            P = pd.DataFrame(prior_rows).dropna()
            if len(P) > 200:
                X = P[["last", "tz", "ret"]].to_numpy(); yv = P["y"].to_numpy()
                c = np.linalg.lstsq(X, yv, rcond=None)[0]
                coef = {"a": float(c[0]), "b": float(c[1]), "c": float(c[2])}
                print(f"after {season}: prior coef {coef}")
        print(season, "done", len(preds))
    out = pd.DataFrame(preds).merge(lines, on="game_id", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "preds.parquet")
    print("prior coefficients used for 2021+:", coef)
    return out


if __name__ == "__main__":
    run()
