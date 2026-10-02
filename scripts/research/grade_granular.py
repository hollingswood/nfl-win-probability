"""Grade granularity (how many letters can the data support?) and a TOTALS grade v2 (predicted CLV of over/under offers).

Stages (outputs output/research/grade_granular.{json,md}, grade_totals_frozen.json):

  python scripts/research/grade_granular.py gran            # Q1: 9-band letters for moneyline grade v2 (bands chosen on
                                                            # 2020-22 dev OOF), DESCRIPTIVE look at 2023-25 (that holdout
                                                            # was already used for the 4-band test), resolution limit,
                                                            # sizing; spread grade v1 per letter / per score
  python scripts/research/grade_granular.py totals_dev      # Q2 dev 2020-22: offer table, LOSO CV, model choice, band scan
  python scripts/research/grade_granular.py totals_freeze   # final fit on 2020-22 -> grade_totals_frozen.json (no overwrite)
  GRADETOT_HOLDOUT=I_HAVE_FROZEN python scripts/research/grade_granular.py totals_holdout   # ONCE on 2023-25
  GRADETOT_HOLDOUT=I_HAVE_FROZEN python scripts/research/grade_granular.py totals_posthoc   # descriptive, after holdout
  python scripts/research/grade_granular.py report          # writes the .md

TOTALS target (honest CLV): EV per unit of our (side, point, price) under the CLOSING fair expected total = median over
the sharp books (lowvig / betonlineag / circasports / bookmaker; else all books) of each book's implied E[T] from its
point + no-vig over price, at the game's last pre-kickoff snapshot (~75 min, must be <= 3 h), valued with the
production key-number distribution src/nflpred/totals.py TotalDist.load() (totals_dist.json, fit 2012-19).
Candidate bets: allowed-book (my_books.json) over/under quotes at snapshots 10 min..7 days before kick and strictly
BEFORE the closing snapshot (a bet at the reference snapshot would be graded against itself), research filters of
scripts/research/totals.load_totals (all quotes; see TCAND_MIN_EV).
No recorded weather (it leaks); stadium type (fixed dome / retractable) only.
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
SCR = Path(os.environ.get("GRADEGRAN_SCRATCH", "/tmp/claude-0/-home-claude-nfl-win-probability/"
                                               "9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad"))
# caches of the earlier grade v2 studies (read only; identical tables)
os.environ.setdefault("MLSP_CACHE", str(SCR / "gradev2" / "cache"))
os.environ.setdefault("GRADE2SP_CACHE", str(SCR / "g2spread" / "cache"))
CACHE = SCR / "gradegran" / "cache"

OUT = ROOT / "output" / "research"
JSON = OUT / "grade_granular.json"
MD = OUT / "grade_granular.md"
FROZEN = OUT / "grade_totals_frozen.json"
DEV = (2020, 2021, 2022)
HOLD = (2023, 2024, 2025)
BOOKS = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet", "fanatics", "hardrockbet"]
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
EV_CLIP = (-0.15, 0.25)
Z = 1.96


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=_jd))


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return None, None
    r = spearmanr(x[ok], y[ok])
    return r4(r.statistic), r4(r.pvalue)


def first_with(D, mask):
    return D[mask].sort_values("requested_ts").drop_duplicates("game_id")


def basic(B, pnl="pnl"):
    """n, mean CLV, SE, one-sided p, beat-close, ROI +- SE."""
    if B is None or len(B) == 0:
        return {"bets": 0}
    c = B.clv.values
    n = len(c)
    m = float(c.mean())
    se = float(c.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    t = m / se if se else None
    out = {"bets": int(n), "pred_clv": r4(B.pred.mean()) if "pred" in B and B.pred.notna().any() else None,
           "clv": r4(m), "clv_se": r4(se), "clv_sd": r4(c.std(ddof=1)) if n > 1 else None,
           "clv_p_one_sided": r4(0.5 * math.erfc(t / math.sqrt(2))) if t is not None else None,
           "beat_close": r4((c > 0).mean())}
    if pnl in B:
        p = B[pnl].values
        out["roi"] = r4(p.mean())
        out["roi_se"] = r4(p.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    return out


def diff_boot(Ba, Bb, n=2000, seed=3):
    """mean CLV(a) - mean CLV(b), SE by game-cluster bootstrap over the union of games (bets of different bands can
    share a game)."""
    if len(Ba) < 3 or len(Bb) < 3:
        return {"diff": None}
    games = np.union1d(Ba.game_id.unique(), Bb.game_id.unique())
    gi = {g: i for i, g in enumerate(games)}
    ai, bi = Ba.game_id.map(gi).values, Bb.game_id.map(gi).values
    ac, bc = Ba.clv.values, Bb.clv.values
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games))
        wa, wb = cnt[ai], cnt[bi]
        if wa.sum() == 0 or wb.sum() == 0:
            continue
        ds.append(np.sum(wa * ac) / wa.sum() - np.sum(wb * bc) / wb.sum())
    d = float(ac.mean() - bc.mean())
    se = float(np.std(ds, ddof=1))
    return {"diff": r4(d), "se": r4(se), "z": r4(d / se) if se else None}


# ================================================================================================ Q1 granularity
NINE = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]
FOUR = ["A+", "A", "B", "C"]
# Chosen on 2020-22 dev OOF only (snapshot-best predicted CLV: median -1.1%, 90th pct +1.0%, 98th +2.6%; first-
# appearance bets with pred >= 3.0% / 3.5%: 54 / 20 over 3 dev seasons -> 3.5% is too thin for a band). NESTED refines
# the production 4 bands (A+ >= 2.5, A >= 1.5, B >= 0.5): old A+ -> A+/A, old A -> A-/B+, old B -> B/B-,
# old C -> C+/C/C-. EQUAL9 = 9 equal-count bands of the dev snapshot-best predictions (cuts set from dev quantiles).
ML_NESTED = {"A+": 0.030, "A": 0.025, "A-": 0.020, "B+": 0.015, "B": 0.010, "B-": 0.005, "C+": 0.0, "C": -0.010}
ML_FOUR = {"A+": 0.025, "A": 0.015, "B": 0.005}


def band_letter(pred, cuts: dict, order):
    out = np.full(len(pred), order[-1], dtype=object)
    for g in reversed(order[:-1]):          # lowest cut first, higher cuts overwrite
        out[pred >= cuts[g]] = g
    return out


def best_per_snapshot(C, pred):
    D = C.assign(pred=pred)
    return D.sort_values("pred", ascending=False).drop_duplicates(["game_id", "requested_ts"]) \
        .sort_values(["game_id", "requested_ts"])


def eval_bands(D, cuts, order, nseas, pnl="pnl"):
    D = D.assign(band=band_letter(D.pred.values, cuts, order))
    bets = {g: first_with(D, D.band == g) for g in order}
    by = {g: basic(bets[g], pnl) for g in order}
    for g in order:
        if by[g]["bets"]:
            by[g]["per_season"] = round(by[g]["bets"] / nseas, 1)
    vals = [by[g].get("clv") if by[g]["bets"] else np.nan for g in order]
    rho, p = spearman(list(range(len(order), 0, -1)), vals)
    adj = {}
    for a, b in zip(order[:-1], order[1:]):
        adj[f"{a} vs {b}"] = diff_boot(bets[a], bets[b])
        adj[f"{a} vs {b}"]["pred_gap"] = r4((by[a].get("pred_clv") or np.nan) - (by[b].get("pred_clv") or np.nan)) \
            if by[a]["bets"] and by[b]["bets"] else None
    n_sig = sum(1 for v in adj.values() if v.get("z") is not None and v["z"] >= Z)
    inversions = sum(1 for a, b in zip(vals[:-1], vals[1:]) if np.isfinite(a) and np.isfinite(b) and b > a)
    allb = pd.concat([bets[g] for g in order])
    return {"cuts": cuts, "by_band": by, "spearman_bands": {"rho": rho, "p": p}, "adjacent": adj,
            "adjacent_distinguishable_z196": n_sig, "adjacent_pairs": len(order) - 1, "inversions": inversions,
            "bet_level_spearman": spearman(allb.pred, allb.clv)[0]}, bets


def calib(B):
    """OLS of realized CLV on predicted CLV at bet level: intercept, slope, residual SD, R2."""
    x, y = B.pred.values, B.clv.values
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    se = np.sqrt(np.diag(r @ r / (len(y) - 2) * np.linalg.inv(X.T @ X)))
    return {"a": float(b[0]), "b": float(b[1]), "b_se": float(se[1]), "resid_sd": float(r.std(ddof=2)),
            "r2": float(1 - r.var() / y.var()), "n": int(len(y))}


def interval_bets(D, lo, hi):
    return first_with(D, (D.pred >= lo) & (D.pred < hi))


def interval_stats(D, grid, n_min=10):
    """n (first-appearance bets) and mean pred of every interval [b_j, b_i) between boundaries b_0=+inf > grid > -inf."""
    b = [np.inf] + list(grid) + [-np.inf]
    st = {}
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            B = interval_bets(D, b[j], b[i])
            if len(B) >= n_min:
                st[(i, j)] = (len(B), float(B.pred.mean()))
    return b, st


def dp_partition(b, st, cal, scale=1.0, kmax=12):
    """Resolution limit. For each k, the partition of the predicted-CLV axis into k bands (cuts from the grid) that
    MAXIMIZES the smallest adjacent expected z = (E_up - E_down) / sqrt(SE_up^2 + SE_down^2), with E = a + b x mean
    pred (calibration `cal`) and SE = resid_sd / sqrt(n x scale) (scale = horizon seasons / 3 sample seasons).
    Returns {k: (best min-z, bands)}; the number of distinguishable bands = largest k with min-z >= 1.96."""
    a, bb, s = cal["a"], cal["b"], cal["resid_sd"]
    m = len(b) - 1
    E = {k: a + bb * v[1] for k, v in st.items()}
    SE = {k: s / math.sqrt(v[0] * scale) for k, v in st.items()}
    f = {(0, j): (np.inf, None) for (i, j) in st if i == 0}
    out = {}
    for k in range(1, kmax + 1):
        if k > 1:
            g = {}
            for (i, j) in st:
                if i == 0:
                    continue
                best = (-np.inf, None)
                for (h, i2), (v, _) in f.items():
                    if i2 != i:
                        continue
                    z = (E[(h, i)] - E[(i, j)]) / math.sqrt(SE[(h, i)] ** 2 + SE[(i, j)] ** 2)
                    c = min(v, z)
                    if c > best[0]:
                        best = (c, (h, i))
                if best[1] is not None:
                    g[(i, j)] = best
            back[k] = g
            f = g
        else:
            back = {1: f}
        ends = [(v, key) for key, (v, _) in f.items() if key[1] == m]
        if not ends:
            continue
        v, key = max(ends)
        bands, kk = [], k
        while key is not None:
            n, mp = st[key]
            bands.append({"lo": None if not np.isfinite(b[key[1]]) else r4(b[key[1]]),
                          "hi": None if not np.isfinite(b[key[0]]) else r4(b[key[0]]), "n": round(n * scale, 1),
                          "mean_pred": r4(mp), "exp_clv": r4(E[key]), "se": r4(SE[key])})
            key = back[kk][key][1] if kk > 1 else None
            kk -= 1
        out[k] = {"min_adjacent_z": r4(v) if np.isfinite(v) else None, "bands": bands[::-1]}
    kbest = max([k for k, v in out.items() if k == 1 or (v["min_adjacent_z"] or -1) >= Z], default=1)
    return {"n_distinguishable_bands": kbest, "by_k": out}


def min_gap_table(cal):
    """Predicted-CLV gap two bands need for their realized means to differ at z=1.96: 1.96*sqrt(2)*sd/(b*sqrt(n))."""
    return {str(n): r4(Z * math.sqrt(2) * cal["resid_sd"] / (max(cal["b"], 1e-6) * math.sqrt(n)))
            for n in (25, 50, 100, 150, 250, 400)}


def ml_tables():
    import grade_v2 as GV
    fz = json.loads((OUT / "grade_v2_frozen.json").read_text())
    chosen = fz["model_choice"]["chosen"]
    Xd = GV.table(DEV)
    Cd = GV.candidates(Xd)
    Fd = GV.featurize(Cd)
    y, w = Cd.clv.values, Cd.w.values
    p = np.full(len(Cd), np.nan)
    for s in DEV:
        te = (Cd.season == s).values
        p[te] = GV.make(chosen).fit(Fd[~te], y[~te], w[~te]).predict(Fd[te])
    Cd["pred"] = p
    # 2023-25 was already evaluated once by grade_v2.py with this frozen model; re-reading it here is DESCRIPTIVE.
    os.environ["GRADEV2_HOLDOUT"] = "I_HAVE_FROZEN"
    Xh = GV.table(HOLD)
    Ch = GV.candidates(Xh)
    Ch["pred"] = GV.load_frozen_model(fz).predict(GV.featurize(Ch))
    return Cd, Ch


def sizing(Dd, Dh, cal):
    """Stake schemes on first-appearance bets. Expected profit = sum(units x CLV); risk = SD of sum(units x pnl).
    edge_cal = a + b x pred (dev calibration). Kelly fraction for an edge e at decimal d: e / (d - 1)."""
    out = {}
    for lab, D, ns in (("dev_oof", Dd, len(DEV)), ("holdout_descriptive", Dh, len(HOLD))):
        res = {}
        defs = {
            "flat_Aplus(>=2.5%)": (0.025, lambda B: np.ones(len(B))),
            "flat_A_and_up(>=1.5%)": (0.015, lambda B: np.ones(len(B))),
            "prop_cal_B_and_up": (0.005, lambda B: np.clip((cal["a"] + cal["b"] * B.pred.values) / 0.02, 0, 2)),
            "kelly_cal_Aplus(1u=1%bankroll,quarter)": (0.025, lambda B: np.clip(
                25 * np.clip(cal["a"] + cal["b"] * B.pred.values, 0, None) / (B.dec.values - 1), 0, 3)),
            "kelly_cal_A_and_up(quarter)": (0.015, lambda B: np.clip(
                25 * np.clip(cal["a"] + cal["b"] * B.pred.values, 0, None) / (B.dec.values - 1), 0, 3)),
        }
        for k, (cut, uf) in defs.items():
            B = first_with(D, D.pred >= cut)
            u = uf(B)
            ep = float((u * B.clv.values).sum())
            pnl = float((u * B.pnl.values).sum())
            risk = float(np.sqrt(((u * (B.pnl.values - B.pnl.values.mean())) ** 2).sum()))
            rng = np.random.default_rng(1)
            bs = []
            for _ in range(1000):
                i = rng.integers(0, len(B), len(B))
                bs.append((u[i] * B.clv.values[i]).sum() / u[i].sum())
            res[k] = {"bets": int(len(B)), "units": r4(u.sum()), "unit_weighted_clv": r4(ep / u.sum()),
                      "unit_weighted_clv_se": r4(np.std(bs, ddof=1)), "exp_profit_units_per_season": r4(ep / ns),
                      "realized_pnl_units_per_season": r4(pnl / ns), "pnl_sd_per_season": r4(risk / math.sqrt(ns)),
                      "exp_profit_over_risk": r4(ep / risk) if risk else None, "median_dec": r4(np.median(B.dec))}
        out[lab] = res
    return out


def spread_v1():
    import grade_v2_spread as GS
    os.environ["GRADE2SP_HOLDOUT"] = "I_HAVE_FROZEN"   # already evaluated once in grade_v2_spread.md; descriptive
    out = {}
    tabs = {}
    for lab, seas in (("dev", DEV), ("holdout", HOLD)):
        X = GS.table(seas)
        X = X[X.clv.notna() & X.price.between(-145, 125)].copy()
        X = X.sort_values("v1_edge", ascending=False).drop_duplicates(["game_id", "requested_ts"])
        X["score"] = GS.v1_score(X)
        X["v1"] = GS.v1_letter(X.score)
        X["pred"] = X.score.astype(float)
        tabs[lab] = X
    order6 = ["A+", "A", "B+", "B", "C+", "C"]
    for lab, X in tabs.items():
        ns = 3
        res = {}
        # (a) the 6 production letters
        bets = {g: first_with(X, X.v1 == g) for g in order6}
        by = {g: basic(bets[g]) for g in order6}
        adj = {f"{a} vs {b}": diff_boot(bets[a], bets[b]) for a, b in zip(order6[:-1], order6[1:])}
        res["letters6"] = {"by_band": by, "spearman": spearman(range(6, 0, -1), [by[g].get("clv") for g in order6])[0],
                           "adjacent": adj,
                           "adjacent_distinguishable_z196": sum(1 for v in adj.values() if (v.get("z") or 0) >= Z)}
        # (b) finest possible: every integer score (C split into -1 / -2 / -3)
        scores = sorted(X.score.unique(), reverse=True)
        sb = {int(s): first_with(X, X.score == s) for s in scores}
        sb = {k: v for k, v in sb.items() if len(v) >= 10}
        res["scores"] = {str(s): basic(b) for s, b in sb.items()}
        res["scores_spearman"] = spearman(list(sb), [basic(b).get("clv") for b in sb.values()])[0]
        # (c) coarse alternatives
        alt = {"A+ | A..B+ | B..C": {"top": ["A+"], "mid": ["A", "B+"], "low": ["B", "C+", "C"]},
               "A+ | rest": {"top": ["A+"], "rest": ["A", "B+", "B", "C+", "C"]}}
        res["coarse"] = {}
        for k, grp in alt.items():
            bb = {n: first_with(X, X.v1.isin(gs)) for n, gs in grp.items()}
            names = list(grp)
            res["coarse"][k] = {"by_band": {n: basic(b) for n, b in bb.items()},
                                "adjacent": {f"{a} vs {b}": diff_boot(bb[a], bb[b]) for a, b in zip(names[:-1], names[1:])}}
        # (d) calibration of CLV on the integer score (bet level, union of per-letter first appearances)
        U = pd.concat(bets.values())
        c = calib(U)
        res["calibration_clv_on_score"] = {k: r4(v) if isinstance(v, float) else v for k, v in c.items()}
        bsv, stv = interval_stats(X, [3.5, 2.5, 1.5, 0.5, -0.5, -1.5], n_min=10)
        res["resolution_limit"] = {h: dp_partition(bsv, stv, c, sc, kmax=7)
                                   for h, sc in (("1_season", 1 / 3), ("3_seasons", 1.0), ("10_seasons", 10 / 3))}
        for d in list(res["letters6"]["by_band"].values()) + list(res["scores"].values()):
            d["mean_score"] = d.pop("pred_clv", None)
        res["score_points_needed_per_band_gap"] = {str(n): r4(Z * math.sqrt(2) * c["resid_sd"] / max(c["b"], 1e-6)
                                                               / math.sqrt(n)) for n in (50, 150, 400)}
        out[lab] = res
    return out


def run_gran():
    Cd, Ch = ml_tables()
    Dd, Dh = best_per_snapshot(Cd, Cd.pred.values), best_per_snapshot(Ch, Ch.pred.values)
    qs = np.quantile(Dd.pred, [8 / 9, 7 / 9, 6 / 9, 5 / 9, 4 / 9, 3 / 9, 2 / 9, 1 / 9])
    EQ9 = {g: float(c) for g, c in zip(NINE[:-1], qs)}
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
           "note": "Bands chosen on 2020-22 out-of-fold predictions only. 2023-25 was ALREADY used for the 4-band "
                   "grade v2 test (grade_v2.md); everything on 2023-25 here is descriptive, not a new test.",
           "ml": {"bands_nested": ML_NESTED, "bands_equal9_dev_quantiles": {k: r4(v) for k, v in EQ9.items()},
                  "dev_snapshot_best_pred_quantiles": {str(q): r4(np.quantile(Dd.pred, q))
                                                       for q in (0.1, 0.5, 0.9, 0.95, 0.98, 0.99)}}}
    for lab, D, ns in (("dev_oof", Dd, len(DEV)), ("holdout_descriptive", Dh, len(HOLD))):
        r = {}
        r["four"], b4 = eval_bands(D, ML_FOUR, FOUR, ns)
        r["nested9"], b9 = eval_bands(D, ML_NESTED, NINE, ns)
        r["equal9"], _ = eval_bands(D, EQ9, NINE, ns)
        r["calibration_bets_nested9"] = calib(pd.concat(b9.values()))
        r["calibration_bets_four"] = calib(pd.concat(b4.values()))
        # offer-level out-of-sample R2 of the prediction (game-weighted) and calibration slope (as grade_v2.py)
        C = Cd if lab == "dev_oof" else Ch
        W = C.w.values
        y, p = C.clv.values, C.pred.values
        ym = np.average(y, weights=W)
        r["offer_level"] = {"r2_oos": r4(1 - np.average((y - p) ** 2, weights=W) / np.average((y - ym) ** 2, weights=W)),
                            "calibration_slope": r4(np.sum(W * (p - np.average(p, weights=W)) * (y - ym)) /
                                                    np.sum(W * (p - np.average(p, weights=W)) ** 2)),
                            "spearman": spearman(p, y)[0]}
        res["ml"][lab] = r
    cal_dev = res["ml"]["dev_oof"]["calibration_bets_nested9"]
    cal_h = res["ml"]["holdout_descriptive"]["calibration_bets_nested9"]
    res["ml"]["min_pred_gap_for_distinguishable_bands(dev calibration)"] = min_gap_table(cal_dev)
    res["ml"]["min_pred_gap_for_distinguishable_bands(holdout calibration)"] = min_gap_table(cal_h)
    grid = np.round(np.arange(0.05, -0.0601, -0.0025), 4)
    bd, std_ = interval_stats(Dd, grid)
    bh, sth = interval_stats(Dh, grid)
    rl = {}
    for horizon, scale in (("1_season", 1 / 3), ("3_seasons", 1.0), ("10_seasons", 10 / 3)):
        rl[horizon] = {"dev_preds_dev_calibration": dp_partition(bd, std_, cal_dev, scale),
                       "holdout_preds_holdout_calibration": dp_partition(bh, sth, cal_h, scale)}
        print(horizon, {k: v["n_distinguishable_bands"] for k, v in rl[horizon].items()}, flush=True)
    res["ml"]["resolution_limit"] = rl
    res["ml"]["sizing"] = sizing(Dd, Dh, cal_dev)
    res["spread_v1"] = spread_v1()
    store("granularity", res)
    print(json.dumps(res, indent=1, default=_jd)[:60000])


# ================================================================================================ Q2 totals grade
KEYT = (37.0, 41.0, 44.0, 47.0, 51.0)


class VDist:
    """Vectorized view of the PRODUCTION totals distribution (src/nflpred/totals.py TotalDist.load()): identical
    cdf / mean_at arrays and the same row / push conventions as TotalDist.probs."""

    def __init__(self):
        from nflpred import totals as T
        self.T = T
        self.d = T.TotalDist.load()
        self.grid = T.MU_GRID
        self.cdf = self.d.cdf
        self.mean_at = self.d.mean_at

    def rows(self, mu):
        loc = np.interp(np.nan_to_num(np.asarray(mu, float), nan=45.0), self.mean_at, self.grid)
        return np.clip(np.round((loc - self.grid[0]) / 0.05), 0, len(self.grid) - 1).astype(int)

    def probs(self, mu, pt, side):
        mu, pt = np.asarray(mu, float), np.asarray(pt, float)
        r = self.rows(mu)
        fl = np.floor(pt).astype(int)
        integer = pt == np.floor(pt)
        p_over = 1 - self.cdf[r, fl]
        p_under = np.where(integer, self.cdf[r, np.ceil(pt).astype(int) - 1], self.cdf[r, fl])
        push = np.clip(1 - p_over - p_under, 0, None)
        win = np.where(np.asarray(side) == "over", p_over, p_under)
        bad = ~np.isfinite(mu) | ~np.isfinite(pt)
        return np.where(bad, np.nan, win), np.where(bad, np.nan, push)

    def ev(self, mu, pt, dec, side):
        w, pu = self.probs(mu, pt, side)
        return w * np.asarray(dec, float) + pu - 1

    def implied_mu(self, pt, nv_over):
        """E[T] whose P(over)/(P(over)+P(under)) at `pt` equals nv_over (vectorized TotalDist.implied_mu)."""
        pt, nv = np.asarray(pt, float), np.asarray(nv_over, float)
        out = np.full(len(pt), np.nan)
        for v in np.unique(pt[np.isfinite(pt)]):
            m = pt == v
            fl = int(math.floor(v))
            po = 1 - self.cdf[:, fl]
            pu = self.cdf[:, int(math.ceil(v)) - 1] if v == int(v) else self.cdf[:, fl]
            q = np.maximum.accumulate(po / np.maximum(po + pu, 1e-12))
            out[m] = np.interp(nv[m], q, self.mean_at)
        return out


def dec_of(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def imp_of(a):
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def stadium_type(g):
    """Fixed per stadium (known before the season): dome if the stadium's games are 'dome'; retractable if any game
    there is 'open'/'closed'. Never the game-day roof status (a closed roof can reveal bad weather)."""
    st = g.groupby("stadium_id").roof.agg(lambda r: set(r.dropna()))
    dome = {s for s, v in st.items() if "dome" in v}
    retr = {s for s, v in st.items() if v & {"open", "closed"}}
    return g.stadium_id.isin(dome).astype(float), g.stadium_id.isin(retr).astype(float)


def totals_table(seasons) -> pd.DataFrame:
    seasons = tuple(seasons)
    if set(seasons) & set(HOLD) and os.environ.get("GRADETOT_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("totals holdout is locked")
    tag = CACHE / f"tot_{min(seasons)}_{max(seasons)}.parquet"
    if tag.exists():
        return pd.read_parquet(tag)
    import replay_early_lines as R
    import totals as TR                    # scripts/research/totals.py (loaders + research filters)
    from nflpred import odds as O
    V = VDist()
    g = TR.load_games()
    g["dome"], g["retractable"] = stadium_type(g)
    gg = g[g.season.isin(seasons)]
    o = TR.load_totals(seasons)
    o = R.match_games(o, gg).merge(gg[["game_id", "kick"]], on="game_id")
    o = o[o.requested_ts < o.kick].copy()
    o["hours_before"] = (o.kick - o.requested_ts).dt.total_seconds() / 3600
    o["nv_over"] = imp_of(o.tot_over_price) / (imp_of(o.tot_over_price) + imp_of(o.tot_under_price))
    o["mu_b"] = V.implied_mu(o.tot_point.values, o.nv_over.values)
    k2 = ["game_id", "requested_ts"]
    S = o.groupby(k2).agg(mu_cons=("mu_b", "median"), pt_cons=("tot_point", "median"), disp=("mu_b", "std"),
                          n_books=("book", "nunique"), hours_before=("hours_before", "first")).reset_index()
    sh = o[o.book.isin(SHARP)].groupby(k2).agg(mu_sharp=("mu_b", "median"), pt_sharp=("tot_point", "median")) \
        .reset_index()
    S = S.merge(sh, on=k2, how="left").sort_values(k2)
    last = S.groupby("game_id").tail(1)[["game_id", "requested_ts", "mu_cons", "mu_sharp", "pt_cons", "hours_before"]]
    last = last.rename(columns={"requested_ts": "close_ts", "mu_cons": "mu_close_all", "mu_sharp": "mu_close_sharp",
                                "pt_cons": "pt_close", "hours_before": "close_hours"})
    last["mu_close"] = last.mu_close_sharp.fillna(last.mu_close_all)
    S = S.merge(last, on="game_id")
    S = S[S.close_hours <= 3]                                   # genuine pre-kickoff close only
    w7 = S[S.hours_before <= 7 * 24].groupby("game_id").agg(mu_first=("mu_cons", "first"), pt_first=("pt_cons", "first"))
    S = S.merge(w7.reset_index(), on="game_id", how="left")
    allowed = O.load_allowed_books() or set(BOOKS)
    B = o[o.book.isin(allowed)][k2 + ["book", "tot_point", "tot_over_price", "tot_under_price"]]
    ov = B.tot_over_price.values
    B = B.assign(overround=imp_of(B.tot_over_price) + imp_of(B.tot_under_price))
    parts = []
    for side in ("over", "under"):
        x = B.copy()
        x["side"] = side
        x["price"] = x[f"tot_{side}_price"]
        parts.append(x.drop(columns=["tot_over_price", "tot_under_price"]).rename(columns={"tot_point": "point"}))
    del ov
    X = pd.concat(parts, ignore_index=True).merge(S, on=k2)
    gc = ["game_id", "season", "week", "game_type", "home_team", "away_team", "total", "dome", "retractable"]
    X = X.merge(gg[gc], on="game_id")
    X = X[X.total.notna()]
    X = X[(X.hours_before >= 10 / 60) & (X.hours_before <= 7 * 24) & (X.requested_ts < X.close_ts)].copy()
    sd = X.side.values
    sg = np.where(sd == "over", 1.0, -1.0)
    X["dec"] = dec_of(X.price.values)
    pt, dc = X.point.values, X.dec.values
    X["ev_sharp_raw"] = V.ev(X.mu_sharp.values, pt, dc, sd)
    X["ev_cons_raw"] = V.ev(X.mu_cons.values, pt, dc, sd)
    X["clv"] = V.ev(X.mu_close.values, pt, dc, sd)
    X["clv_all"] = V.ev(X.mu_close_all.values, pt, dc, sd)
    X["clv_pts"] = sg * (X.pt_close.values - pt)                  # + = our number better than the closing median
    adj = np.where(sd == "over", X.total - pt, pt - X.total)
    X["pnl"] = np.where(adj > 0, X.dec - 1, np.where(adj < 0, -1.0, 0.0))
    X["pt_adv_cons"] = sg * (X.pt_cons.values - pt)               # + = we get a better number than the median
    X["pt_adv_sharp"] = sg * (X.pt_sharp.fillna(X.pt_cons).values - pt)
    X["move_mu"] = sg * (X.mu_cons - X.mu_first)
    X["move_pts"] = sg * (X.pt_cons - X.pt_first)
    best = X.groupby(k2 + ["side"]).ev_cons_raw.transform("max")
    X["best_gap"] = best - X.ev_cons_raw
    CACHE.mkdir(parents=True, exist_ok=True)
    X.to_parquet(tag)
    return X


def key_feats(point, side, cons):
    point, cons = np.asarray(point, float), np.asarray(cons, float)
    over = np.asarray(side) == "over"
    kr = np.zeros(len(point))
    onk = np.zeros(len(point))
    cross = np.zeros(len(point))
    adv = np.where(over, cons - point, point - cons)              # + = our number better
    lo, hi = np.minimum(point, cons), np.maximum(point, cons)
    for k in KEYT:
        kr += np.where(over, (point == k - 0.5).astype(float) - (point == k + 0.5),
                       (point == k + 0.5).astype(float) - (point == k - 0.5))
        onk += (point == k).astype(float)
        cross += ((k >= lo) & (k <= hi) & (np.abs(point - cons) > 1e-9)).astype(float)
    return kr, onk, np.sign(adv) * cross


TFEATURES = {
    "ev_sharp": "EV of our point+price under the sharp fair expected total NOW (median over lowvig/betonlineag/"
                "circasports/bookmaker of each book's implied E[T]); fallback ev_cons; clipped",
    "ev_cons": "EV under the all-book median implied E[T] now; clipped",
    "sharp_missing": "1 if no sharp book quotes the total at the snapshot",
    "pt_adv_cons": "our number vs the all-book median point, + = better for our side (lower for over); clipped +-2.5",
    "pt_adv_sharp": "our number vs the sharp median point (fallback all-book), + = better; clipped +-2.5",
    "key_right": "+1 over at k-0.5 / under at k+0.5, -1 over at k+0.5 / under at k-0.5, k in 37,41,44,47,51",
    "on_key": "1 if our point is exactly 37/41/44/47/51 (push possible)",
    "key_cross": "signed count of key totals between the median point and ours (+ = ours better)",
    "p_imp": "1/decimal price of our side (juice)",
    "overround": "the offering book's two-way overround (clipped 1.0..1.12)",
    "log_hours": "log(1 + hours before kickoff)",
    "move_mu": "all-book fair E[T] now minus at first sight (<= 7 days), toward our side (+ = toward over for overs); "
               "clipped +-6",
    "move_pts": "all-book median point now minus at first sight, toward our side; clipped +-6",
    "disp": "std across books of implied E[T] at the snapshot (points); clipped 0..3",
    "best_gap": "best allowed-book EV-vs-consensus for this side at the snapshot minus this offer's (0 = best); "
                "clipped 0..0.1",
    "is_over": "1 for over",
    "tot_level": "all-book fair E[T] now (minus 44)",
    "dome": "1 if the stadium is a fixed dome",
    "retractable": "1 if the stadium has a retractable roof",
    "week": "NFL week",
    **{f"bk_{b}": f"1 if book == {b}" for b in BOOKS[1:]},
}
TFEATS = list(TFEATURES)
TMONO = {"ev_sharp": 1, "ev_cons": 1}
TSETS = {
    "core": ["ev_sharp", "ev_cons"],
    "market": ["ev_sharp", "ev_cons", "sharp_missing", "pt_adv_cons", "move_mu", "disp", "log_hours", "is_over"],
    "small": ["ev_sharp", "ev_cons", "sharp_missing", "pt_adv_cons", "pt_adv_sharp", "key_right", "on_key",
              "key_cross", "p_imp", "move_mu", "move_pts", "disp", "log_hours", "is_over", "tot_level", "best_gap"],
    "all": TFEATS,
}


def tfeaturize(X: pd.DataFrame) -> pd.DataFrame:
    c = lambda s: np.clip(s, *EV_CLIP)  # noqa: E731
    F = pd.DataFrame(index=X.index)
    F["ev_sharp"] = c(X.ev_sharp_raw.fillna(X.ev_cons_raw).fillna(0.0))
    F["ev_cons"] = c(X.ev_cons_raw.fillna(0.0))
    F["sharp_missing"] = X.mu_sharp.isna().astype(float)
    F["pt_adv_cons"] = np.clip(X.pt_adv_cons.fillna(0.0), -2.5, 2.5)
    F["pt_adv_sharp"] = np.clip(X.pt_adv_sharp.fillna(0.0), -2.5, 2.5)
    kr, onk, cr = key_feats(X.point.values, X.side.values, X.pt_cons.fillna(X.point).values)
    F["key_right"], F["on_key"], F["key_cross"] = kr, onk, cr
    F["p_imp"] = 1 / X.dec
    F["overround"] = X.overround.clip(1.0, 1.12)
    F["log_hours"] = np.log1p(X.hours_before.clip(lower=0))
    F["move_mu"] = np.clip(X.move_mu.fillna(0.0), -6, 6)
    F["move_pts"] = np.clip(X.move_pts.fillna(0.0), -6, 6)
    F["disp"] = X.disp.fillna(0.5).clip(0, 3)
    F["best_gap"] = X.best_gap.fillna(0.0).clip(0, 0.1)
    F["is_over"] = (X.side == "over").astype(float)
    F["tot_level"] = (X.mu_cons - 44.0).fillna(0.0)
    F["dome"] = X.dome.astype(float)
    F["retractable"] = X.retractable.astype(float)
    F["week"] = X.week.astype(float)
    for b in BOOKS[1:]:
        F[f"bk_{b}"] = (X.book == b).astype(float)
    return F[TFEATS].astype(float)


# Dev decision (2020-22, before freeze): the first dev run with "EV >= 0 vs sharp or consensus" kept only 830 offers on
# 345 of 838 games (a totals quote must beat the consensus by the full ~4.5% vig to qualify), so most games would have
# no graded offer and the GBMs could not split. A grade has to exist for every offer shown, so candidates = ALL
# allowed-book quotes passing the price filter (the moneyline filter only removed junk quotes; totals junk is already
# removed by load_totals).
TCAND_MIN_EV = None


def tcandidates(X):
    ok = X.clv.notna() & X.price.between(-250, 200)
    anypos = ((X.ev_sharp_raw >= TCAND_MIN_EV) | (X.ev_cons_raw >= TCAND_MIN_EV)) if TCAND_MIN_EV is not None else True
    C = X[ok & anypos].reset_index(drop=True)
    C["w"] = 1.0 / C.groupby("game_id").game_id.transform("size")
    return C


class Ridge:
    kind = "ridge"

    def __init__(self, alpha, cols):
        self.alpha, self.cols = alpha, list(cols)

    def fit(self, F, y, w):
        F = F[self.cols]
        self.mu = F.mean().values
        self.sd = F.std().replace(0, 1).values
        Zm = (F.values - self.mu) / self.sd
        W = w / w.sum()
        ym = float((W * y).sum())
        zm = (W[:, None] * Zm).sum(0)
        Zc, yc = Zm - zm, y - ym
        A = (Zc * W[:, None]).T @ Zc + self.alpha * np.eye(Zm.shape[1]) / len(y)
        self.coef = np.linalg.solve(A, (Zc * W[:, None]).T @ yc)
        self.b0 = ym - zm @ self.coef
        return self

    def predict(self, F):
        return self.b0 + ((F[self.cols].values - self.mu) / self.sd) @ self.coef

    def to_json(self):
        return {"kind": "ridge", "alpha": self.alpha, "intercept": float(self.b0), "feature_order": self.cols,
                "features": {f: {"mean": float(m), "sd": float(s), "coef_std": float(c), "coef_raw": float(c / s)}
                             for f, m, s, c in zip(self.cols, self.mu, self.sd, self.coef)},
                "formula": "pred_clv = intercept + sum_f coef_std[f] * (x_f - mean[f]) / sd[f]"}


class Base:
    kind = "base"
    cols = ["ev_sharp"]

    def fit(self, F, y, w):
        return self

    def predict(self, F):
        return F.ev_sharp.values


class GBM:
    kind = "lgbm"

    def __init__(self, cols, **kw):
        self.cols, self.kw = list(cols), kw

    def fit(self, F, y, w):
        import lightgbm as lgb
        p = dict(objective="regression", learning_rate=0.03, num_leaves=7, min_data_in_leaf=400, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                 monotone_constraints=[TMONO.get(f, 0) for f in self.cols], verbose=-1, seed=7, num_threads=2)
        p.update(self.kw)
        n = p.pop("n_trees", 200)
        self.m = lgb.train(p, lgb.Dataset(F[self.cols].values, y, weight=w, feature_name=self.cols), n)
        return self

    def predict(self, F):
        return self.m.predict(F[self.cols].values)


TIERS = {"base": 0, "ridge_core": 1, "ridge_market": 2, "ridge_small": 3, "ridge_all": 4, "lgbm_small": 5,
         "lgbm_all": 6}
TCHOICE = ("complexity tiers base (raw EV vs sharp fair total) < ridge core < ridge market < ridge small < ridge all < "
           "lgbm small < lgbm all; move to the best model of a higher tier only if its LOSO (2020/21/22) game-weighted "
           "MSE beats the current choice by > 0.5% relative (rule of grade_v2 / grade_v2_spread)")


def tier(name):
    return next(v for k, v in TIERS.items() if name.startswith(k))


def tgrid():
    g = [("base_ev_sharp", lambda: Base())]
    for fs in ("core", "market", "small", "all"):
        for a in (1.0, 300.0, 3000.0):
            g.append((f"ridge_{fs}_a{a:g}", lambda a=a, fs=fs: Ridge(a, TSETS[fs])))
    for fs in ("small", "all"):
        for nt, lv in ((100, 4), (200, 4)):
            g.append((f"lgbm_{fs}_{nt}x{lv}", lambda nt=nt, lv=lv, fs=fs: GBM(TSETS[fs], n_trees=nt, num_leaves=lv)))
    return g


def tmake(name):
    return dict(tgrid())[name]()


def wmse(y, p, w):
    return float(np.sum(w * (y - p) ** 2) / np.sum(w))


def tchoose(cv):
    cur = "base_ev_sharp"
    for t in sorted(set(TIERS.values()))[1:]:
        ks = [k for k in cv if tier(k) == t]
        if ks:
            b = min(ks, key=lambda k: cv[k]["wmse"])
            if cv[b]["wmse"] < cv[cur]["wmse"] * 0.995:
                cur = b
    return cur


def importance(m):
    if isinstance(m, Ridge):
        return {f: r4(c) for f, c in zip(m.cols, m.coef)}
    if isinstance(m, Base):
        return {"ev_sharp": 1.0}
    imp = m.m.feature_importance("gain")
    return {f: r4(g / imp.sum()) for f, g in zip(m.cols, imp)}


# Chosen on dev (2020-22) only, BEFORE freeze. The chosen LightGBM compresses the top (snapshot-best pred: 95th pct
# +0.3%, 98th +0.9%; nothing >= +2.5%), so the moneyline cuts do not transfer. Dev OOF scan (first bet per game with
# pred >= cut): 0.0% -> 73/season, CLV +2.3% +- 0.6; +0.5% -> 45/season, +1.7% +- 0.7; +1.0% -> 17/season, +2.4% +- 1.2
# (too few). Bands 1% apart (Q1: closer bands cannot be told apart at these volumes): dev first-appearance A+ 218 bets
# +2.3%, A 356 +0.5%, B 558 -1.3%, C 763 -2.3% (rho 1). A+ = predicted CLV >= 0 (a standard -110 quote at the
# consensus number is about -4.5%).
TOT_THRESHOLDS = {"A+": 0.0, "A": -0.01, "B": -0.02}
TOT_STRATEGIES = [
    {"id": "S1_Aplus_flat", "min_grade": "A+",
     "desc": "One bet per game at the first snapshot where the game's best totals offer is A+; 1 unit."},
    {"id": "S2_A_and_up_flat", "min_grade": "A",
     "desc": "One bet per game at the first snapshot where the best totals offer is A or A+; 1 unit."},
]


def tletter(pred, th):
    return np.where(pred >= th["A+"], "A+", np.where(pred >= th["A"], "A", np.where(pred >= th["B"], "B", "C")))


def tsumm(B, ns):
    s = basic(B)
    if not s["bets"]:
        return s
    s["per_season"] = round(s["bets"] / ns, 1)
    s["clv_all_close"] = r4(B.clv_all.mean())
    s["clv_pts"] = r4(B.clv_pts.mean())
    s["pct_over"] = r4((B.side == "over").mean())
    s["median_hours_before"] = r4(B.hours_before.median())
    s["by_season"] = {int(k): {"n": int(len(d)), "clv": r4(d.clv.mean()), "roi": r4(d.pnl.mean())}
                      for k, d in B.groupby("season")}
    return s


def tevaluate(C, pred, th, ns):
    D = best_per_snapshot(C, pred)
    D["grade"] = tletter(D.pred.values, th)
    bets = {g: first_with(D, D.grade == g) for g in FOUR}
    bg = {g: tsumm(bets[g], ns) for g in FOUR}
    out = {"games": int(C.game_id.nunique()), "candidate_rows": int(len(C)), "by_grade_first_appearance": bg}
    out["letters_spearman"] = spearman([4, 3, 2, 1], [bg[g].get("clv") if bg[g]["bets"] else np.nan for g in FOUR])[0]
    out["adjacent"] = {f"{a} vs {b}": diff_boot(bets[a], bets[b]) for a, b in zip(FOUR[:-1], FOUR[1:])}
    out["Aplus_minus_C"] = diff_boot(bets["A+"], bets["C"])
    D["dec_bin"] = pd.qcut(D.pred.rank(method="first"), 10, labels=False)
    dd = D.groupby("dec_bin").agg(pred=("pred", "mean"), clv=("clv", "mean"), n=("clv", "size")).reset_index()
    out["deciles"] = [{"pred": r4(r.pred), "clv": r4(r.clv), "n": int(r.n)} for r in dd.itertuples()]
    out["deciles_spearman"] = spearman(dd.pred, dd.clv)[0]
    W = C.w.values
    y = C.clv.values
    ym = np.average(y, weights=W)
    pc = pred - np.average(pred, weights=W)
    out["offer_spearman"] = spearman(pred, y)[0]
    out["calibration_slope"] = r4(np.sum(W * pc * (y - ym)) / np.sum(W * pc ** 2))
    out["r2_oos_offer"] = r4(1 - np.average((y - pred) ** 2, weights=W) / np.average((y - ym) ** 2, weights=W))
    out["strategies"] = {s["id"]: tsumm(first_with(D, D.pred >= th[s["min_grade"]]), ns) for s in TOT_STRATEGIES}
    allb = pd.concat(bets.values())
    out["calibration_bets"] = {k: r4(v) if isinstance(v, float) else v for k, v in calib(allb).items()}
    return out, D


def run_totals_dev():
    X = totals_table(DEV)
    C = tcandidates(X)
    F = tfeaturize(C)
    y, w = C.clv.values, C.w.values
    print("offers", len(X), "candidates", len(C), "games", C.game_id.nunique(), "mean clv offers", r4(X.clv.mean()),
          "cands", r4(np.average(y, weights=w)), flush=True)
    var0 = wmse(y, np.full(len(y), np.average(y, weights=w)), w)
    cv, oof = {}, {}
    for name, mk in tgrid():
        p = np.full(len(C), np.nan)
        for s in DEV:
            te = (C.season == s).values
            p[te] = mk().fit(F[~te], y[~te], w[~te]).predict(F[te])
        oof[name] = p
        cv[name] = {"wmse": wmse(y, p, w), "r2_vs_const": r4(1 - wmse(y, p, w) / var0)}
        print(name, cv[name], flush=True)
    best = tchoose(cv)
    print("chosen", best, flush=True)
    p = oof[best]
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV),
           "offers": int(len(X)), "candidate_rows": int(len(C)), "games": int(C.game_id.nunique()),
           "clv_all_offers_mean": r4(X.clv.mean()), "clv_candidates_mean": r4(np.average(y, weights=w)),
           "close_has_sharp": r4(X.drop_duplicates("game_id").mu_close_sharp.notna().mean()),
           "books": sorted(X.book.unique()), "cv": cv, "chosen_model": best, "choice_rule": TCHOICE,
           "oof_pred_quantiles": {str(q): r4(np.quantile(p, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99)}}
    D = best_per_snapshot(C, p)
    res["snapshot_best_pred_quantiles"] = {str(q): r4(np.quantile(D.pred, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.98)}
    scan = []
    for cut in (-0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04):
        s = tsumm(first_with(D, D.pred >= cut), len(DEV))
        scan.append({"cut": cut, **{k: s.get(k) for k in ("bets", "per_season", "pred_clv", "clv", "clv_se", "roi",
                                                          "roi_se", "beat_close", "clv_all_close")}})
        print(scan[-1], flush=True)
    res["oof_threshold_scan"] = scan
    sc0 = []
    D0 = best_per_snapshot(C, oof["base_ev_sharp"])
    for cut in (0.0, 0.01, 0.02, 0.03):
        s = tsumm(first_with(D0, D0.pred >= cut), len(DEV))
        sc0.append({"cut": cut, **{k: s.get(k) for k in ("bets", "clv", "clv_se", "roi", "roi_se")}})
    res["oof_threshold_scan_base"] = sc0
    m = tmake(best).fit(F, y, w)
    res["fit_all_dev_importance"] = importance(m)
    res["fit_all_dev_model"] = m.to_json() if isinstance(m, Ridge) else {"kind": m.kind}
    res["feature_spearman_dev"] = {f: spearman(F[f], y)[0] for f in TFEATS if F[f].std() > 0}
    if TOT_THRESHOLDS:
        res["oof_eval_chosen"], _ = tevaluate(C, p, TOT_THRESHOLDS, len(DEV))
        res["oof_eval_base"], _ = tevaluate(C, oof["base_ev_sharp"], TOT_THRESHOLDS, len(DEV))
        res["thresholds"] = TOT_THRESHOLDS
    store("totals_dev", res)
    print(json.dumps({k: v for k, v in res.items() if k != "cv"}, indent=1, default=_jd)[:25000])


def run_totals_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen totals grade is final)")
    if not TOT_THRESHOLDS:
        raise SystemExit("set TOT_THRESHOLDS from the dev band scan first")
    J = json.loads(JSON.read_text())
    best = J["totals_dev"]["chosen_model"]
    X = totals_table(DEV)
    C = tcandidates(X)
    F = tfeaturize(C)
    m = tmake(best).fit(F, C.clv.values, C.w.values)
    if isinstance(m, Ridge):
        model = m.to_json()
    elif isinstance(m, Base):
        model = {"kind": "base", "feature_order": ["ev_sharp"], "formula": "pred_clv = ev_sharp (clipped)"}
    else:
        model = {"kind": "lgbm", "model_string": m.m.model_to_string(), "feature_order": m.cols}
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"), "grading_version": 2, "market": "totals",
          "what": "TOTALS GRADE v2 = predicted honest CLV of an over/under offer: EV of our point+price under the "
                  "closing fair expected total (median over sharp books of each book's implied E[T]; fallback all "
                  "books) at the last pre-kickoff snapshot, key-number distribution totals_dist.json.",
          "distribution": "src/nflpred/totals.py TotalDist.load() (totals_dist.json); EV = P(win) x decimal + P(push) - 1; "
                          "implied E[T] of a book = TotalDist.implied_mu(point, no-vig P(over)) (research used a vectorized "
                          "grid-interpolated equivalent, VDist.implied_mu: max difference 0.03 points); no-vig P(over) = "
                          "implied(over) / (implied(over) + implied(under))",
          "production_notes": "features need, per game: the snapshot history since first sight within 7 days "
                              "(move_mu, move_pts), all books' totals quotes at the snapshot (consensus, sharp, disp) and "
                              "the allowed-book quotes (best_gap). Bands apply to the raw model output.",
          "fit_seasons": list(DEV), "holdout": list(HOLD), "model_choice": {"chosen": best, "rule": TCHOICE},
          "model": model, "feature_order": model["feature_order"],
          "feature_definitions": {f: TFEATURES[f] for f in model["feature_order"]}, "ev_clip": list(EV_CLIP),
          "fair_reference_definitions": {"sharp_books": sorted(SHARP),
                                         "consensus": "median over ALL books in the feed of each book's implied E[T]",
                                         "book_filter": "prices -250..+200 with |price| >= 100, overround 1.00..1.12, "
                                                        "point 25..80 and within 5 of the snapshot median point"},
          "candidate_filter": "every allowed-book (my_books.json) over/under quote at snapshots 10 min .. 7 days before "
                              "kick (research: strictly before the reference closing snapshot); price -250..+200",
          "thresholds_pred_clv": TOT_THRESHOLDS, "letters": "A+ >= A+ cut; A >= A cut; B >= B cut; else C",
          "one_bet_per_game": "first snapshot where the game's best predicted-CLV totals offer reaches the grade; that "
                              "offer (book, side, point, price) is the bet",
          "strategies": TOT_STRATEGIES,
          "pass_bar": {"monotone": "Spearman rho across the 4 letters (first-appearance CLV) >= 0.8 and A+ highest",
                       "top_grade": "A+ realized CLV > 0 with one-sided p < 0.05 (t on bet-level CLV)",
                       "adopt_if": "both hold"},
          "evaluate_once": True}
    FROZEN.write_text(json.dumps(fz, indent=1, default=_jd))
    print(json.dumps({k: v for k, v in fz.items() if k != "model"}, indent=1, default=_jd))
    print(json.dumps(model, default=_jd)[:3000])


def load_frozen_tmodel(fz):
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


def run_totals_holdout():
    if os.environ.get("GRADETOT_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout locked: set GRADETOT_HOLDOUT=I_HAVE_FROZEN after freezing")
    if not FROZEN.exists():
        raise SystemExit("freeze first")
    J = json.loads(JSON.read_text())
    if "totals_holdout" in J:
        raise SystemExit("totals holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    th = fz["thresholds_pred_clv"]
    m = load_frozen_tmodel(fz)
    X = totals_table(HOLD)
    C = tcandidates(X)
    pred = m.predict(tfeaturize(C))
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"],
           "seasons": list(HOLD), "offers": int(len(X)), "books": sorted(X.book.unique()),
           "close_has_sharp": r4(X.drop_duplicates("game_id").mu_close_sharp.notna().mean())}
    res["eval"], D = tevaluate(C, pred, th, len(HOLD))
    F = tfeaturize(C)
    res["eval_base_ev_sharp"], _ = tevaluate(C, F.ev_sharp.values, th, len(HOLD))
    ev = res["eval"]
    bg = ev["by_grade_first_appearance"]
    top = bg["A+"]
    res["pass"] = {"monotone": bool(ev["letters_spearman"] is not None and ev["letters_spearman"] >= 0.8 and
                                    top.get("clv", -9) == max(bg[g].get("clv", -9) for g in FOUR if bg[g]["bets"])),
                   "top_grade": bool(top.get("bets", 0) > 2 and top["clv"] > 0 and top["clv_p_one_sided"] < 0.05)}
    res["pass"]["adopt"] = res["pass"]["monotone"] and res["pass"]["top_grade"]
    store("totals_holdout", res)
    print(json.dumps(res, indent=1, default=_jd)[:20000])


def run_totals_posthoc():
    """NOT pre-registered (after the one-shot holdout; descriptive): what is A+? Blind early-week unders, A+ by side and
    hours before kick, and A+ vs the plain 'early under with EV >= 0 vs sharp' rule."""
    if os.environ.get("GRADETOT_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("set GRADETOT_HOLDOUT=I_HAVE_FROZEN")
    J = json.loads(JSON.read_text())
    if "totals_holdout" not in J:
        raise SystemExit("run the holdout first")
    fz = json.loads(FROZEN.read_text())
    m, th = load_frozen_tmodel(fz), fz["thresholds_pred_clv"]
    out = {"note": "post-hoc, descriptive; not part of the pre-registered test"}
    for lab, seas in (("dev_in_sample_fit", DEV), ("holdout", HOLD)):
        X = totals_table(seas)
        C = tcandidates(X)
        C["pred"] = m.predict(tfeaturize(C))
        D = best_per_snapshot(C, C.pred.values)
        A = first_with(D, D.pred >= th["A+"])
        r = {"Aplus_by_side": {k: tsumm(d, 3) for k, d in A.groupby("side")},
             "Aplus_by_hours": {k: tsumm(d, 3) for k, d in A.groupby(pd.cut(A.hours_before, [0, 24, 72, 120, 200]).astype(str))}}
        early = C[C.hours_before >= 96]
        U = early[early.side == "under"].sort_values("ev_cons_raw", ascending=False) \
            .drop_duplicates(["game_id", "requested_ts"]).pipe(lambda d: first_with(d, d.price.notna()))
        r["blind_early_under_best_price(>=96h, first snapshot)"] = tsumm(U, 3)
        V = early[(early.side == "under") & (early.ev_sharp_raw >= 0)].sort_values("ev_sharp_raw", ascending=False) \
            .drop_duplicates(["game_id", "requested_ts"]).pipe(lambda d: first_with(d, d.price.notna()))
        r["early_under_ev_sharp>=0"] = tsumm(V, 3)
        both = set(A.game_id) & set(V.game_id)
        r["Aplus_not_in_early_under_rule"] = tsumm(A[~A.game_id.isin(both)], 3)
        r["early_under_rule_not_Aplus"] = tsumm(V[~V.game_id.isin(both)], 3)
        out[lab] = r
    store("totals_posthoc", out)
    print(json.dumps(out, indent=1, default=_jd)[:12000])


# ================================================================================================ report
def _p(x, se=None, d=2):
    if x is None:
        return "n/a"
    return f"{100 * x:+.{d}f}%" + (f" ± {100 * se:.{d}f}" if se is not None else "")


def _brow(name, s):
    if not s or not s.get("bets"):
        return f"| {name} | 0 | | | | | |"
    return (f"| {name} | {s['bets']} | {_p(s.get('pred_clv'))} | {_p(s['clv'], s.get('clv_se'))} | "
            f"{s.get('clv_p_one_sided')} | {100 * s['beat_close']:.0f}% | {_p(s.get('roi'), s.get('roi_se'), 1)} |")


BHEAD = ("| band | bets | pred CLV | realized CLV ± SE | p (one-sided) | beat close | ROI ± SE |\n"
         "|---|---|---|---|---|---|---|")


def _bands(bands, pct=True):
    f = (lambda x: _p(x)) if pct else (lambda x: f"{x:+.1f}")
    return "; ".join(f"[{'-∞' if b['lo'] is None else f(b['lo'])}, {'∞' if b['hi'] is None else f(b['hi'])}) "
                     f"n={b['n']} exp {_p(b['exp_clv'], b['se'])}" for b in bands)


def _adj(a):
    return "; ".join(f"{k}: {_p(v.get('diff'))} (z {v.get('z')})" for k, v in a.items())


def run_report():
    J = json.loads(JSON.read_text())
    G = J["granularity"]
    M = G["ml"]
    L = ["# Grade granularity (how many letters?) and a totals grade v2", "",
         "Code `scripts/research/grade_granular.py`; numbers `grade_granular.json`; frozen totals grade "
         "`grade_totals_frozen.json`.", "",
         "## Q1. Finer letters for the moneyline grade v2?", "",
         "Bands chosen on 2020-22 out-of-fold (LOSO) predictions only. **2023-25 was already used once for the "
         "4-band grade v2 test, so every 2023-25 number below is descriptive, not a new test.** Bets: one per "
         "(game, band) at the first snapshot where the game's best predicted-CLV offer falls in that band "
         "(the grade_v2.py convention).", "",
         f"* NESTED 9 (refines the production 4: old A+ -> A+/A, A -> A-/B+, B -> B/B-, C -> C+/C/C-): "
         + ", ".join(f"{k} >= {_p(v)}" for k, v in M["bands_nested"].items()) + ", else C-.",
         f"* EQUAL 9 (equal-count dev quantiles of snapshot-best predictions): "
         + ", ".join(f"{k} >= {_p(v)}" for k, v in M["bands_equal9_dev_quantiles"].items()) + ", else C-.", ""]
    for lab in ("dev_oof", "holdout_descriptive"):
        R = M[lab]
        for sch in ("four", "nested9", "equal9"):
            E = R[sch]
            order = FOUR if sch == "four" else NINE
            L += [f"### {lab}: {sch}", "", BHEAD] + [_brow(g, E["by_band"][g]) for g in order]
            L += ["", f"Spearman across bands {E['spearman_bands']['rho']}; inversions {E['inversions']}; adjacent pairs "
                      f"distinguishable at z >= 1.96: **{E['adjacent_distinguishable_z196']} of {E['adjacent_pairs']}**; "
                      f"bet-level rank corr {E['bet_level_spearman']}.", "",
                  "Adjacent differences (game-cluster bootstrap): " + _adj(E["adjacent"]), ""]
        c9 = R["calibration_bets_nested9"]
        L += [f"Bet-level calibration (realized on predicted, nested-9 bets): slope {c9['b']:.2f} ± {c9['b_se']:.2f}, "
              f"residual SD {100 * c9['resid_sd']:.1f}%, R² {c9['r2']:.3f}; offer-level OOS R² "
              f"{R['offer_level']['r2_oos']}, slope {R['offer_level']['calibration_slope']}.", ""]
    L += ["### Resolution limit", "",
          "Predicted-CLV gap two bands need so that their realized means differ at z = 1.96, by bets per band "
          "(1.96·√2·residual SD / (slope·√n)): dev calibration " +
          ", ".join(f"n={k}: {_p(v)}" for k, v in M["min_pred_gap_for_distinguishable_bands(dev calibration)"].items()) +
          "; holdout calibration " +
          ", ".join(f"n={k}: {_p(v)}" for k, v in M["min_pred_gap_for_distinguishable_bands(holdout calibration)"].items())
          + ".", "",
          "Optimal partition (dynamic programming over cuts every 0.25%: for k bands, maximize the smallest adjacent "
          "expected z; expected CLV = bet-level calibration line, SE = residual SD / sqrt(n), n = first-appearance bets "
          "of the interval scaled to the horizon). Distinguishable bands = largest k with min z >= 1.96:", ""]
    for h, v in M["resolution_limit"].items():
        for k, R in v.items():
            nb = R["n_distinguishable_bands"]
            L.append(f"* {h}, {k}: **{nb} distinguishable bands** (best min adjacent z for k = 2..6: " + ", ".join(
                f"{kk}: {R['by_k'][str(kk) if str(kk) in R['by_k'] else kk]['min_adjacent_z']}" for kk in range(2, 7)
                if (str(kk) in R["by_k"] or kk in R["by_k"])) + "): " + _bands(R["by_k"][str(nb) if str(nb) in R["by_k"] else nb]["bands"]))
    L += ["", "### Sizing (first-appearance bets; expected profit = Σ units × CLV)", "",
          "| scheme | sample | bets | units | unit-weighted CLV ± SE | exp. profit u/season | realized PnL u/season | "
          "exp. profit / PnL SD |", "|---|---|---|---|---|---|---|---|"]
    for lab, S in M["sizing"].items():
        for k, s in S.items():
            L.append(f"| {k} | {lab} | {s['bets']} | {s['units']} | {_p(s['unit_weighted_clv'], s['unit_weighted_clv_se'])} | "
                     f"{s['exp_profit_units_per_season']} | {s['realized_pnl_units_per_season']} | {s['exp_profit_over_risk']} |")
    SV = G["spread_v1"]
    L += ["", "## Q1b. Spread grade v1 (already 6 letters A+..C)", ""]
    for lab in ("dev", "holdout"):
        R = SV[lab]
        L += [f"### {lab} ({'2020-22' if lab == 'dev' else '2023-25, descriptive'})", "", BHEAD]
        L += [_brow(g, {**R["letters6"]["by_band"][g], "pred_clv": None}) + f" score {R['letters6']['by_band'][g]['mean_score']:+.2f}"
              for g in ["A+", "A", "B+", "B", "C+", "C"]]
        c = R["calibration_clv_on_score"]
        L += ["", f"Spearman {R['letters6']['spearman']}; adjacent distinguishable at z >= 1.96: "
                  f"**{R['letters6']['adjacent_distinguishable_z196']} of 5**. " + _adj(R["letters6"]["adjacent"]), "",
              "Per integer score (finest possible; C split): " + "; ".join(
                  f"{s}: {v['bets']} bets {_p(v['clv'], v['clv_se'])}" for s, v in R["scores"].items()) +
              f" (Spearman {R['scores_spearman']}).", "",
              f"CLV on score: {_p(c['b'])} per point (± {_p(c['b_se'])}), residual SD {_p(c['resid_sd'])}, R² {c['r2']}; "
              f"score points needed between bands: " + ", ".join(f"n={k}: {v}" for k, v in
                                                                 R["score_points_needed_per_band_gap"].items()), "",
              "Resolution limit (DP over score cuts, linear calibration): " + "; ".join(
                  f"{h}: **{v['n_distinguishable_bands']}** bands ("
                  + _bands(v["by_k"][str(v["n_distinguishable_bands"]) if str(v["n_distinguishable_bands"]) in v["by_k"]
                                     else v["n_distinguishable_bands"]]["bands"], pct=False) + ")"
                  for h, v in R["resolution_limit"].items()), "",
              "Coarse alternatives: " + " | ".join(
                  f"{k}: " + ", ".join(f"{n} {b['bets']} {_p(b['clv'], b['clv_se'])}" for n, b in v["by_band"].items())
                  + " (" + _adj(v["adjacent"]) + ")" for k, v in R["coarse"].items()), ""]
    if "totals_dev" in J:
        T = J["totals_dev"]
        L += ["## Q2. Totals grade v2", "",
              f"Dev 2020-22: {T['offers']} allowed-book offers, {T['candidate_rows']} candidates, {T['games']} games; "
              f"mean CLV all offers {_p(T['clv_all_offers_mean'])}, candidates {_p(T['clv_candidates_mean'])}; close has "
              f"a sharp book {100 * T['close_has_sharp']:.0f}%. Books: {', '.join(T['books'])}.", "",
              "LOSO R² vs constant: " + ", ".join(f"{k} {v['r2_vs_const']}" for k, v in T["cv"].items()) +
              f". Chosen: **{T['chosen_model']}** ({T['choice_rule']}).", "",
              "Fit on 2020-22 (coef per 1 SD or gain share): " + ", ".join(
                  f"{k} {v}" for k, v in sorted(T["fit_all_dev_importance"].items(), key=lambda kv: -abs(kv[1] or 0))), "",
              "Univariate offer-level Spearman with CLV (dev): " + ", ".join(
                  f"{k} {v}" for k, v in sorted(T["feature_spearman_dev"].items(), key=lambda kv: -abs(kv[1] or 0))[:14]), "",
              "Dev band scan (OOF, first bet per game with pred >= cut): " + "; ".join(
                  f"{_p(s['cut'], d=1)}: {s['bets']} bets, CLV {_p(s['clv'], s['clv_se'])}" for s in T["oof_threshold_scan"]), ""]
        THEAD = ("| grade | bets (/season) | pred CLV | realized CLV ± SE | p | beat close | ROI ± SE | CLV vs all-book close | "
                 "points vs close |\n|---|---|---|---|---|---|---|---|---|")

        def trow(n, s):
            if not s.get("bets"):
                return f"| {n} | 0 | | | | | | | |"
            return (f"| {n} | {s['bets']} ({s['per_season']}) | {_p(s.get('pred_clv'))} | {_p(s['clv'], s['clv_se'])} | "
                    f"{s['clv_p_one_sided']} | {100 * s['beat_close']:.0f}% | {_p(s['roi'], s['roi_se'], 1)} | "
                    f"{_p(s['clv_all_close'])} | {s['clv_pts']:+.2f} |")
        for lab, E in (("Dev 2020-22 (OOF)", T.get("oof_eval_chosen")),
                       ("HOLDOUT 2023-25 (frozen, run once)", (J.get("totals_holdout") or {}).get("eval")),
                       ("Holdout baseline: same bands on raw EV vs sharp fair (no model)",
                        (J.get("totals_holdout") or {}).get("eval_base_ev_sharp"))):
            if not E:
                continue
            L += [f"### {lab}", "", THEAD] + [trow(g, E["by_grade_first_appearance"][g]) for g in FOUR]
            L += [trow(k, s) for k, s in E["strategies"].items()]
            L += ["", f"Spearman across letters {E['letters_spearman']}; deciles rho {E['deciles_spearman']}; offer-level "
                      f"rank corr {E['offer_spearman']}; calibration slope {E['calibration_slope']}; OOS R² "
                      f"{E['r2_oos_offer']}. A+ minus C {_p(E['Aplus_minus_C'].get('diff'), E['Aplus_minus_C'].get('se'))}. "
                      "Adjacent: " + _adj(E["adjacent"]), "",
                  "Deciles (pred -> realized): " + ", ".join(f"{_p(d['pred'])}->{_p(d['clv'])}" for d in E["deciles"]), "",
                  "By season (A+): " + ", ".join(f"{k}: {v['n']} bets {_p(v['clv'])}" for k, v in
                                                 E["by_grade_first_appearance"]["A+"].get("by_season", {}).items()), ""]
        H = J.get("totals_holdout")
        if H:
            L += [f"**Verdict (pre-registered bar): {'ADOPT' if H['pass']['adopt'] else 'DO NOT ADOPT'}** -- monotone "
                  f"[{H['pass']['monotone']}], A+ CLV > 0 at p < 0.05 [{H['pass']['top_grade']}].", ""]
    P = J.get("totals_posthoc")
    if P:
        def q(d):
            return f"{d['bets']} bets, CLV {_p(d.get('clv'), d.get('clv_se'))}, ROI {_p(d.get('roi'), d.get('roi_se'), 1)}" \
                if d.get("bets") else "0 bets"
        L += ["### Post-hoc (after the one-shot holdout; descriptive, not part of the test): what is totals A+?", ""]
        for lab in ("dev_in_sample_fit", "holdout"):
            R = P[lab]
            L += [f"* {lab}: A+ by side: " + "; ".join(f"{k} {q(v)}" for k, v in R["Aplus_by_side"].items()) +
                  ". By hours before kick: " + "; ".join(f"{k} {q(v)}" for k, v in R["Aplus_by_hours"].items()) +
                  f". Blind early-week (>= 96 h) under at the best allowed price: {q(R['blind_early_under_best_price(>=96h, first snapshot)'])}"
                  f". Early under with EV >= 0 vs the sharp fair: {q(R['early_under_ev_sharp>=0'])}; A+ bets outside that "
                  f"rule: {q(R['Aplus_not_in_early_under_rule'])}."]
        L.append("")
    L += ["## Reading", "", READING]
    MD.write_text("\n".join(L) + "\n")
    print(MD.read_text())


READING = """**Q1 -- finer letters do not help.**

* Moneyline, 9 letters chosen on dev: realized CLV is still roughly ordered (Spearman 0.98 dev, 0.97 on 2023-25), but
  only 1 of 8 adjacent pairs is statistically distinguishable in either period (C vs C-), and 2023-25 shows two
  inversions (B+ -0.3% below B +0.8%; C+ = C). The 4 production letters: 2 of 3 adjacent pairs distinguishable in both
  periods (A+ vs A and B vs C; A vs B never). Equal-count 9-iles: 3 of 8.
* Why: a bet's realized CLV has a residual SD of ~10% (longshot prices), while the bet-level calibration slope is
  ~0.9. Two bands need a predicted-CLV gap of 1.96*sqrt(2)*SD/(slope*sqrt(n)) = 2.6% at 150 bets per band, 1.6% at 400.
  The useful (positive) range of predicted CLV is only about 0..+3.5%, so it holds ONE clean step (A+ vs the rest)
  plus the C tail. The DP resolution limit (optimistic: assumes the dev calibration line holds) is 3 bands for one
  season of bets, 6 over three seasons -- but only 2 of those 6 lie above 0% predicted CLV; the rest split the
  negative (never-bet) region.
* Recommendation: keep the 4 moneyline bands (A+ >= 2.5%, A >= 1.5%, B >= 0.5%, C); do not add +/-. Read them as
  3 tiers: A+ = bet, A/B = no edge shown (2023-25: +0.1% / -0.1%), C = avoid. If anything, a 5th band could split C
  (C vs C- is the one robust extra split), which has no betting value.
* Sizing: flat 1 unit on A+ only. Among the schemes (flat A+, flat A-and-up, units proportional to calibrated
  predicted CLV from B up, quarter-Kelly on calibrated edge), flat A+ had the best expected-profit-to-PnL-risk ratio
  in dev (0.28) and on 2023-25 (0.16; quarter-Kelly 0.25 / 0.13, proportional 0.20 / 0.13). Kelly mostly shrinks
  longshot stakes, which costs as much CLV as it saves variance; proportional sizing adds ~0-CLV A/B bets. The
  grade cannot rank within A+ (A+ split at 3.0%: +2.8% vs +1.8%, z 0.7), so predicted-CLV-scaled stakes are not
  supported.
* Spread grade v1 (6 letters): ranked in both periods (rho 1.0) but adjacent letters are mostly indistinguishable:
  2 of 5 pairs in 2020-22, 1 of 5 in 2023-25 (A+ vs A, z 2.0). CLV per score point fell from +1.0% (dev) to +0.5%
  (2023-25). Resolution: 3 bands over 3 seasons in dev, 2 in 2023-25. Its letters are best read as A+ (+2.2..3.0%)
  vs the rest; a 3-tier display (A+ / A..B+ / B..C) is the most the data supports.

**Q2 -- totals grade v2: passes its pre-registered bar, and it is a Tuesday-under detector.**

* Chosen model (rule fixed before looking): LightGBM on the 'small' feature set (LOSO R2 0.088 vs 0.066 for raw EV
  vs the sharp fair total). Gain: EV vs sharp 53%, over/under 16%, hours before kick 13%, total level 7%; key
  numbers, juice, book and dispersion add ~nothing. Bands (dev): A+ >= 0.0% predicted, A >= -1%, B >= -2%, else C.
* 2023-25 (frozen, one run): A+ 243 bets (81/season) CLV +1.48% +- 0.66 (p 0.012), +0.7 / +1.7 / +1.9% by season,
  same vs the all-book close (+1.36%), +0.97 points vs the closing number; ROI +7.5% +- 6.1 (noise). A -1.3%, B -2.0%,
  C -2.6%; letters rho 1.0; A+ minus C +4.0% +- 0.7. Raw EV vs sharp with the same bands fails (A+ -0.15%), so the
  model's side/timing terms carry the edge. Dev was stronger (A+ +2.3%) and the holdout calibration slope is 0.81.
* What A+ is (post-hoc): 95% unders, median 139 h before kick (Tuesday 14:10 UTC). Totals drift down during the week
  (both periods), so an early under at or better than the sharp fair beats the close: early unders with EV >= 0 vs the
  sharp fair were +2.3% (dev, 118) and +2.6% +- 0.9 (2023-25, 122), and every one of them was A+; the other A+ bets were
  +0.2% on 2023-25. Blind early unders at the best price were -1.2%. The earlier totals study's soft-vs-sharp rule (both
  sides) failed on 2023-25 because overs do not share the drift.
* Recommendation: totals grade v2 may be shown as a LABEL (like moneyline grade v2): A+ only means something. Do
  not start a track from this result alone; if a totals track is wanted, pre-register the simple rule (Tuesday/early
  under at the best allowed price with EV >= 0 vs LowVig/BetOnline) and paper-trade it in 2026, judged on CLV.
* Caveats: 2023-25 totals were used once before (totals.md), and that report (Tuesday unders cheaper than overs;
  soft-vs-sharp failed) was known when this study was designed; the user-specified feature list included over/under
  and hours. The candidate set was widened to all quotes after the first dev run (dev-only decision, before the
  freeze). Production needs each game's totals snapshot history (first-seen fair total) and all books' quotes; the
  research implied-total inversion is a grid version of TotalDist.implied_mu (max 0.03 points apart).
"""

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "gran"
    {"gran": run_gran, "totals_dev": run_totals_dev, "totals_freeze": run_totals_freeze,
     "totals_holdout": run_totals_holdout, "totals_posthoc": run_totals_posthoc, "report": run_report}[stage]()
