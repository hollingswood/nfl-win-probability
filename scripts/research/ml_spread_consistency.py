"""Moneyline vs spread consistency, with a total-aware key-number margin model.

Stages (outputs: output/research/ml_spread_consistency.{json,md}, ml_spread_consistency_frozen.json):
  python scripts/research/ml_spread_consistency.py part1      # fit total-aware margin model (1999-2019, LOSO-selected),
                                                              # validate on 2020-2025 vs single-weight models
  python scripts/research/ml_spread_consistency.py p2dev      # ML vs spread-implied ML, 2020-2022 only
  python scripts/research/ml_spread_consistency.py p3dev      # teaser legs (2020-22) and alt-line buys (2023) re-priced
  python scripts/research/ml_spread_consistency.py freeze     # writes ..._frozen.json (refuses overwrite)
  MLSP_HOLDOUT=I_HAVE_FROZEN python scripts/research/ml_spread_consistency.py holdout   # ONCE: ML 2023-25,
                                                              # teasers 2023-25, alt buys 2024-25
  python scripts/research/ml_spread_consistency.py report     # writes the .md
Env MLSP_CACHE=<dir> caches intermediate tables.

Model: scripts/research/margin_by_total.py (Normal(mu, sigma) x w(|k|; total), kernel-raked weights).
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
import margin_by_total as MT  # noqa: E402
import teasers_v2 as T2  # noqa: E402
from nflpred import spread_bets as SB  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "ml_spread_consistency.json"
FROZEN = OUT / "ml_spread_consistency_frozen.json"
MD = OUT / "ml_spread_consistency.md"
CACHE = os.environ.get("MLSP_CACHE")
FIT = tuple(range(1999, 2020))
VAL = tuple(range(2020, 2026))
DEV = (2020, 2021, 2022)
HOLD = (2023, 2024, 2025)
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
TBINS = [0, 41, 44, 47, 50, 80]
TLABS = ["<=41", "41.5-44", "44.5-47", "47.5-50", ">50"]


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Interval):
        return str(o)
    return str(o)


def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=_jd))


def load_json():
    return json.loads(JSON.read_text())


def mse(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (r4(x.mean()) if len(x) else None), None
    return r4(x.mean()), r4(x.std(ddof=1) / math.sqrt(len(x)))


def bll(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ============================================================================================ PART 1
def games_all():
    g = T2.games(range(1999, 2026))
    g = g[g.spread_line.notna() & g.total_line.notna()].copy()
    g["ref"] = MT.ref_totals(g)
    ih = T2.imp(g.home_spread_odds.fillna(-110).values)
    ia = T2.imp(g.away_spread_odds.fillna(-110).values)
    g["q"] = ih / (ih + ia)
    return g


class _Wrap:
    """Adapter so teasers_v2.Dist / single-weight models share the MarginModel API (total ignored)."""

    def __init__(self, sigma, w):
        self.d = MT._Dist(sigma, w)

    def implied_mu(self, hp, q, total=None, ref_total=None, spread_abs=None):
        return self.d.implied_mu(np.atleast_1d(hp), np.atleast_1d(q))

    def pmf(self, mu, total=None, ref_total=None, spread_abs=None):
        P = MT.normal_rows(mu, self.d.sigma) * self.d.w[None, :]
        return P / P.sum(1, keepdims=True)

    def side_probs(self, mu, sh, pt, total=None, ref_total=None, spread_abs=None):
        mu, pt, sh = np.atleast_1d(mu), np.atleast_1d(pt), np.atleast_1d(sh).astype(bool)
        pwh = self.d.p_gt(mu, -pt)
        pwa = 1 - self.d.p_gt(mu, pt) - self.d.p_eq(mu, pt)
        return np.where(sh, pwh, pwa), self.d.p_eq(mu, np.where(sh, -pt, pt))

    def win_tie(self, mu, total=None, ref_total=None, spread_abs=None):
        mu = np.atleast_1d(mu)
        return self.d.p_gt(mu, np.zeros(len(mu))), self.d.p_eq(mu, np.zeros(len(mu)))


def single_models():
    key, _, c = T2.load_dist()
    old = SB.load_rules()
    w_old = np.array([old["_weights"].get(abs(int(k)), 1.0) for k in MT.KS])
    return {"old_spread_rules": _Wrap(old["margin"]["sigma"], w_old), "teasers_v2_single": _Wrap(c["sigma"], key.w)}


def get_model(name="total_aware"):
    J = load_json()
    if name in ("total_aware", "pooled_1999_2019"):
        return MT.MarginModel(J["part1"]["model" if name == "total_aware" else "pooled_model"])
    if name in ("total_aware_2012", "pooled_2012_2019"):
        return MT.MarginModel(J["part1b"]["model" if name == "total_aware_2012" else "pooled_model"])
    if name == "chosen":
        return get_model(J["part1b"]["chosen_for_betting"])
    return single_models()[name]


def run_part1(fit_seasons=FIT, key_name="part1", sigmas=(13.4, 13.8, 14.2, 14.6), modes=("abs", "rel"),
              spread_variants=True):
    g = games_all()
    key, _, _ = T2.load_dist()
    g["mu0"] = key.implied_mu(-g.spread_line.values, g.q.values)
    F = g[g.season.isin(fit_seasons)].reset_index(drop=True)
    s0_ = min(fit_seasons)
    nb = 3 if len(fit_seasons) > 12 else 1
    folds = [F.season.between(s0_ + nb * i, s0_ + nb * i + nb - 1).values for i in range(len(fit_seasons) // nb)]

    def cv(**kw):
        ll = 0.0
        for te in folds:
            tr = ~te
            M = MT.MarginModel(MT.fit(F.mu0[tr], F.m[tr], F.total_line[tr], ref_total=F.ref[tr],
                                      spread_abs=F.spread_line[tr].abs(), **kw))
            ll += MT.loglik(M, F.mu0[te], F.m[te], F.total_line[te], F.ref[te], F.spread_line[te].abs()).sum()
        return ll / len(F)

    grid = []
    for s0 in sigmas:
        for sl in (0.0, 0.1):
            grid.append({"sigma0": s0, "sigma_slope": sl, "h": None})
    for s0 in sigmas[1:3]:
        for mode in modes:
            for h in (2.5, 4.0, 6.0):
                for sm in (20.0, 80.0, 300.0):
                    grid.append({"sigma0": s0, "sigma_slope": 0.0, "h": h, "smooth": sm, "mode": mode})
    for hs in ((1.0, 2.0) if spread_variants else ()):
        grid.append({"sigma0": 14.2, "sigma_slope": 0.0, "h": None, "h_spread": hs, "smooth": 80.0})
        grid.append({"sigma0": 14.2, "sigma_slope": 0.0, "h": 4.0, "h_spread": hs, "smooth": 80.0, "mode": "abs"})
    for c in grid:
        c["loso_loglik"] = round(cv(**{k: v for k, v in c.items() if k != "loso_loglik"}), 5)
        print(c, flush=True)
    pooled = max([c for c in grid if c["h"] is None and c.get("h_spread") is None], key=lambda c: c["loso_loglik"])
    best = max(grid, key=lambda c: c["loso_loglik"])
    kw = lambda c: {k: v for k, v in c.items() if k != "loso_loglik"}  # noqa: E731
    # final fits on all of 1999-2019; mu refreshed through each model once (price-implied under that model)
    fits = {}
    for name, c in (("pooled_model", pooled), ("model", best)):
        p = MT.fit(F.mu0, F.m, F.total_line, ref_total=F.ref, spread_abs=F.spread_line.abs(), **kw(c))
        M = MT.MarginModel(p)
        mu1 = M.implied_mu(-F.spread_line.values, F.q.values, F.total_line.values, F.ref.values)
        p = MT.fit(mu1, F.m, F.total_line, ref_total=F.ref, spread_abs=F.spread_line.abs(), **kw(c))
        p["selected_by"] = f"leave-season-block-out log likelihood of exact margin, {fit_seasons[0]}-{fit_seasons[-1]}"
        fits[name] = p
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "fit_seasons": [fit_seasons[0], fit_seasons[-1]],
           "grid": grid, "chosen": best, "chosen_pooled": pooled, "model": fits["model"],
           "pooled_model": fits["pooled_model"]}
    store(key_name, res)
    if key_name == "part1":
        res.update(validate(g))
        store(key_name, res)
        print(json.dumps(res["validation"]["summary"], indent=1, default=_jd))


def run_part1b():
    """Same model class fit on 2012-2019 only (the era of teasers_v2's single-weight fit), then validation of all
    models split 2020-2022 (dev) / 2023-2025 / 2020-2025. The model used in parts 2-3 is chosen on 2020-2022 only."""
    run_part1(tuple(range(2012, 2020)), "part1b", sigmas=(13.0, 13.4, 13.8), modes=("abs",), spread_variants=False)
    g = games_all()
    J = load_json()
    res = J["part1b"]
    res["validation_by_period"] = {}
    for lab, seas in (("2020-2022", DEV), ("2023-2025", HOLD), ("2020-2025", VAL)):
        res["validation_by_period"][lab] = validate(g, seas, extra=True)["validation"]
    dv = res["validation_by_period"]["2020-2022"]["per_model"]
    res["chosen_for_betting"] = min(("total_aware", "total_aware_2012"), key=lambda k: dv[k]["exact_margin_logloss"])
    res["chosen_rule"] = "lower 2020-2022 exact-margin log loss of total_aware (1999-2019) vs total_aware_2012"
    store("part1b", res)
    for lab, v in res["validation_by_period"].items():
        print(lab, json.dumps(v["summary"], default=_jd))
    print("chosen:", res["chosen_for_betting"])


def validate(g, seasons=VAL, extra=False) -> dict:
    """Out-of-sample calibration: exact margin, exact 3/7/10/14, covers at +-0.5 around key numbers."""
    V = g[g.season.isin(list(seasons))].reset_index(drop=True)
    models = {"total_aware": get_model("total_aware"), "pooled_1999_2019": get_model("pooled_1999_2019"),
              **single_models()}
    if extra:
        models["total_aware_2012"] = get_model("total_aware_2012")
        models["pooled_2012_2019"] = get_model("pooled_2012_2019")
    fav_home = V.spread_line.values > 0           # nflverse spread_line > 0 = home favored
    fm = np.where(fav_home, V.m.values, -V.m.values)     # favorite's margin
    V["tb"] = pd.cut(V.total_line, TBINS, labels=TLABS)
    out = {"n_games": int(len(V)), "per_model": {}, "by_total": [], "weights": {}}
    LL = {}
    lines = (1.5, 2.5, 3.5, 6.5, 7.5, 9.5, 10.5, 13.5, 14.5)
    for name, M in models.items():
        mu = M.implied_mu(-V.spread_line.values, V.q.values, V.total_line.values, V.ref.values,
                          V.spread_line.abs().values)
        P = M.pmf(mu, V.total_line.values, V.ref.values, V.spread_line.abs().values)
        ll = -np.log(np.maximum(P[np.arange(len(V)), np.clip(V.m.values, -60, 60) + 60], 1e-12))
        LL[name] = {"exact": ll}
        # favorite-oriented pmf
        Pf = np.where(fav_home[:, None], P, P[:, ::-1])
        rec = {"exact_margin_logloss": r4(ll.mean())}
        for k in (3, 7, 10, 14):
            pk = Pf[:, 60 + k] + Pf[:, 60 - k]
            yk = (np.abs(V.m.values) == k).astype(float)
            LL[name][f"e{k}"] = bll(pk, yk)
            rec[f"exact{k}"] = {"pred": r4(pk.sum()), "act": int(yk.sum()), "logloss": r4(LL[name][f"e{k}"].mean())}
            V[f"p{k}_{name}"] = pk
        cov = []
        for L in lines:
            pc = Pf[:, KSI(L):].sum(1)                     # P(fav margin > L)
            yc = (fm > L).astype(float)
            cov.append(bll(pc, yc))
            rec[f"fav_gt_{L}"] = {"pred": r4(pc.mean()), "act": r4(yc.mean())}
        LL[name]["cover"] = np.mean(cov, axis=0)
        rec["cover_logloss_mean_9_lines"] = r4(LL[name]["cover"].mean())
        out["per_model"][name] = rec
    base = LL["total_aware"]
    summ = {}
    for name in models:
        if name == "total_aware":
            continue
        d = {}
        for k in ("exact", "e3", "e7", "cover"):
            diff = LL[name][k] - base[k]                  # + = total-aware better
            if name in ("teasers_v2_single", "old_spread_rules", "pooled_2012_2019") and "total_aware_2012" in LL:
                d2 = LL[name][k] - LL["total_aware_2012"][k]
                d.setdefault("ta2012_" + k, {"gain_per_game": r4(d2.mean()),
                                             "se": r4(d2.std(ddof=1) / math.sqrt(len(d2)))})
            d[k] = {"gain_per_game": r4(diff.mean()), "se": r4(diff.std(ddof=1) / math.sqrt(len(diff)))}
        summ[f"total_aware_vs_{name}"] = d
    out["summary"] = summ
    for (tb), e in V.groupby("tb", observed=True):
        row = {"total": str(tb), "n": int(len(e))}
        for k in (3, 7):
            row[f"act{k}"] = int((e.m.abs() == k).sum())
            for name in models:
                row[f"pred{k}_{name}"] = r4(e[f"p{k}_{name}"].sum())
        out["by_total"].append(row)
    # fit-period check (in-sample for total_aware) by total bin, 1999-2019
    Fg = g[g.season.isin(FIT)].reset_index(drop=True)
    Fg["tb"] = pd.cut(Fg.total_line, TBINS, labels=TLABS)
    fb = []
    for name in ("total_aware", "pooled_1999_2019"):
        M = models[name]
        mu = M.implied_mu(-Fg.spread_line.values, Fg.q.values, Fg.total_line.values, Fg.ref.values)
        P = M.pmf(mu, Fg.total_line.values, Fg.ref.values)
        Fg[f"p3_{name}"] = P[:, 57] + P[:, 63]
        Fg[f"p7_{name}"] = P[:, 53] + P[:, 67]
    for tb, e in Fg.groupby("tb", observed=True):
        fb.append({"total": str(tb), "n": int(len(e)), "act3": int((e.m.abs() == 3).sum()),
                   "pred3_total_aware": r4(e.p3_total_aware.sum()), "pred3_pooled": r4(e.p3_pooled_1999_2019.sum()),
                   "act7": int((e.m.abs() == 7).sum()), "pred7_total_aware": r4(e.p7_total_aware.sum()),
                   "pred7_pooled": r4(e.p7_pooled_1999_2019.sum())})
    out["fit_period_by_total"] = fb
    M = models["total_aware"]
    for t in (38.0, 41.0, 44.0, 47.0, 50.0, 53.0):
        w = M.weights_at(M.tcov(np.array([t]))[0])
        out["weights"][str(t)] = {str(k): r4(w[60 + k]) for k in (0, 1, 2, 3, 4, 6, 7, 8, 10, 14, 17, 21)}
    # P(exact 3) and P(exact 7) for a 3-pt and 7-pt favourite at -110 by total (illustration)
    ill = []
    for t in (38.0, 41.0, 44.0, 47.0, 50.0, 53.0):
        for sp in (3.0, 7.0):
            mu = M.implied_mu(np.array([-sp]), np.array([0.5]), np.array([t]))
            P = M.pmf(mu, np.array([t]))[0]
            pw, pt = M.win_tie(mu, np.array([t]))
            ill.append({"total": t, "fav_by": sp, "mu": r4(mu[0]), "p_fav_by_3": r4(P[63]), "p_fav_by_7": r4(P[67]),
                        "p_fav_win_no_tie": r4(pw[0] / (1 - pt[0]))})
    out["illustration"] = ill
    return {"validation": out}


def KSI(L):
    """index into KS of the smallest integer > L"""
    return int(math.floor(L)) + 1 + 60


# ============================================================================================ PART 2
def nv(h, a):
    ih, ia = T2.imp(h), T2.imp(a)
    return ih / (ih + ia), ih + ia - 1


def load_pinnacle(seasons, g):
    fs = [ROOT / "data" / "historical_odds" / "pinnacle" / f"nfl_odds_{s}.csv.gz" for s in seasons]
    ps = [pd.read_csv(f).assign(season=s) for f, s in zip(fs, seasons) if f.exists()]
    if not ps:
        return pd.DataFrame(columns=["game_id", "requested_ts", "p_pin"])
    p = pd.concat(ps, ignore_index=True)
    p["requested_ts"] = pd.to_datetime(p.requested_ts, utc=True)
    p["commence"] = pd.to_datetime(p.commence_time, utc=True)
    p = T2.match(p, g)
    p = p[p.ml_home.notna() & p.ml_away.notna()]
    p["p_pin"], _ = nv(p.ml_home.values, p.ml_away.values)
    return p.groupby(["game_id", "requested_ts"]).p_pin.median().reset_index()


def build_ml(seasons, M, single=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (rows, snaps). rows: one per (game, snapshot, book) with that book's ML no-vig prob and the win prob
    implied by that book's OWN spread + juice (total-aware model); snaps: per (game, snapshot) consensus/sharp/pinnacle
    references. Only snapshots <= 7 days and >= 10 min before kickoff."""
    tag = f"ml_{min(seasons)}_{max(seasons)}"
    if CACHE and (Path(CACHE) / f"{tag}_rows.parquet").exists():
        return (pd.read_parquet(Path(CACHE) / f"{tag}_rows.parquet"),
                pd.read_parquet(Path(CACHE) / f"{tag}_snaps.parquet"))
    g = T2.games(seasons)
    gg = games_all()
    g["ref"] = gg.set_index("game_id").ref.reindex(g.game_id).values
    o = T2.match(T2.load_odds(seasons), g)
    o = o.merge(g[["game_id", "kick", "week", "gameday", "m", "total_line", "spread_line", "ref"]], on="game_id")
    o["hours_before"] = (o.kick - o.requested_ts).dt.total_seconds() / 3600
    o = o[(o.hours_before >= 10 / 60) & (o.hours_before <= 7 * 24)]
    o = o.sort_values("last_update").drop_duplicates(["game_id", "requested_ts", "book"], keep="last")
    # spread validity (main line)
    sp_ok = (o.sp_home_point.notna() & o.sp_home_price.notna() & o.sp_away_price.notna() &
             ((o.sp_home_point + o.sp_away_point).abs() < 1e-9) & o.sp_home_price.between(-145, 125) &
             o.sp_away_price.between(-145, 125) & ((o.sp_home_point * 2) % 1 == 0))
    med = o[sp_ok].groupby(["game_id", "requested_ts"]).sp_home_point.transform("median")
    sp_ok &= (o.sp_home_point - med.reindex(o.index)).abs() <= 2.5
    q, ovr = nv(o.sp_home_price.fillna(-110).values, o.sp_away_price.fillna(-110).values)
    o["q_home"], o["sp_overround"] = q, ovr
    sp_ok &= o.sp_overround.between(-0.01, 0.12)
    o["sp_ok"] = sp_ok
    ml_ok = o.ml_home.notna() & o.ml_away.notna() & o.ml_home.abs().between(100, 2500) & o.ml_away.abs().between(100, 2500)
    p, ov = nv(o.ml_home.fillna(100).values, o.ml_away.fillna(100).values)
    o["p_ml"], o["ml_overround"] = np.where(ml_ok, p, np.nan), np.where(ml_ok, ov, np.nan)
    ml_ok &= o.ml_overround.between(-0.01, 0.15)
    o.loc[~ml_ok, "p_ml"] = np.nan
    o["ml_ok"] = ml_ok
    # total at the snapshot: consensus of the totals file at or before the snapshot; fallback nflverse close
    t = T2.match(T2.load_totals(seasons), g)
    t = t[t.tot_point.notna()].groupby(["game_id", "requested_ts"]).tot_point.median().rename("tot_snap").reset_index()
    o = pd.merge_asof(o.sort_values("requested_ts"), t.sort_values("requested_ts"), on="requested_ts", by="game_id",
                      direction="backward")
    o["tot_fallback"] = o.tot_snap.isna()
    o["tot"] = o.tot_snap.fillna(o.total_line)
    S = o[o.sp_ok]
    mu = M.implied_mu(S.sp_home_point.values, S.q_home.values, S.tot.values, S.ref.values)
    pw, pt = M.win_tie(mu, S.tot.values, S.ref.values)
    o.loc[S.index, "mu_sp"] = mu
    o.loc[S.index, "p_sp"] = pw / (1 - pt)
    o.loc[S.index, "p_tie"] = pt
    if single is not None:
        mu1 = single.implied_mu(S.sp_home_point.values, S.q_home.values)
        pw1, pt1 = single.win_tie(mu1)
        o.loc[S.index, "p_sp_single"] = pw1 / (1 - pt1)
    k2 = ["game_id", "requested_ts"]
    agg = dict(p_ml_cons=("p_ml", "median"), p_sp_cons=("p_sp", "median"), mu_cons=("mu_sp", "median"),
               pt_cons=("sp_home_point", "median"), n_books=("book", "nunique"), p_tie_cons=("p_tie", "median"))
    if single is not None:
        agg["p_sp1_cons"] = ("p_sp_single", "median")
    snaps = o.groupby(k2).agg(**agg)
    sh = o[o.book.isin(SHARP)].groupby(k2).agg(p_ml_sharp=("p_ml", "median"), p_sp_sharp=("p_sp", "median"),
                                               mu_sharp=("mu_sp", "median"))
    snaps = snaps.join(sh).reset_index()
    pin = load_pinnacle(seasons, g)
    snaps = snaps.merge(pin, on=k2, how="left")
    snaps = snaps.merge(o.drop_duplicates(k2)[k2 + ["season", "week", "gameday", "kick", "hours_before", "m", "tot",
                                                    "tot_fallback", "ref", "spread_line", "total_line"]], on=k2)
    # closing references (last snapshot of the game in our files): spread-implied (sharp->all) and ML (closing_fair)
    last = snaps.sort_values("requested_ts").groupby("game_id").tail(1)
    snaps = snaps.merge(last[["game_id", "p_sp_sharp", "p_sp_cons", "p_ml_sharp", "p_ml_cons"]].rename(columns={
        "p_sp_sharp": "pc_sp_sharp", "p_sp_cons": "pc_sp_cons", "p_ml_sharp": "pc_ml_sharp", "p_ml_cons": "pc_ml_cons"}),
        on="game_id")
    import edge_lab as EL
    cf = EL.closing_fair(list(seasons))[["game_id", "p_close_sharp", "p_close_all"]]
    snaps = snaps.merge(cf, on="game_id", how="left")
    snaps["p_close"] = snaps.p_close_sharp.fillna(snaps.p_close_all)
    snaps["pc_sp"] = snaps.pc_sp_sharp.fillna(snaps.pc_sp_cons)
    lp = snaps[snaps.p_pin.notna()].sort_values("requested_ts").groupby("game_id").tail(1)
    snaps = snaps.merge(lp[["game_id", "p_pin", "requested_ts"]].rename(
        columns={"p_pin": "pc_pin", "requested_ts": "pc_pin_ts"}), on="game_id", how="left")
    rows = o[["game_id", "requested_ts", "book", "ml_home", "ml_away", "p_ml", "ml_overround", "sp_home_point",
              "sp_home_price", "sp_away_price", "q_home", "sp_overround", "mu_sp", "p_sp", "p_tie", "sp_ok", "ml_ok"]
             + (["p_sp_single"] if single is not None else [])]
    if CACHE:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        rows.to_parquet(Path(CACHE) / f"{tag}_rows.parquet")
        snaps.to_parquet(Path(CACHE) / f"{tag}_snaps.parquet")
    return rows, snaps


def windows(S: pd.DataFrame) -> pd.DataFrame:
    """tue / fri / sun / kick tags (teasers_v2 definitions) on a per-(game, snapshot) frame."""
    X = S.copy()
    X["hours_before"] = (X.kick - X.requested_ts).dt.total_seconds() / 3600
    return T2.assign_windows(X)


def bets_table(rows, snaps, allowed) -> pd.DataFrame:
    """One row per (game, snapshot, allowed book, side) with the ML price and fair probs from several references."""
    B = rows[rows.book.isin(allowed) & rows.ml_ok].merge(snaps, on=["game_id", "requested_ts"])
    if "pc_pin" not in B:
        B["pc_pin"] = np.nan
    out = []
    for side in ("home", "away"):
        x = B.copy()
        h = side == "home"
        x["side"] = side
        x["ml"] = x[f"ml_{side}"]
        f = (lambda c: x[c]) if h else (lambda c: 1 - x[c])
        for c in ("p_ml", "p_sp", "p_ml_cons", "p_sp_cons", "p_ml_sharp", "p_sp_sharp", "p_pin", "p_close",
                  "pc_sp", "pc_ml_sharp", "pc_pin"):
            x[c + "_s"] = f(c)
        x["sp_pt_side"] = x.sp_home_point if h else -x.sp_home_point
        x["cons_pt_side"] = x.pt_cons if h else -x.pt_cons
        x["side_margin"] = x.m if h else -x.m
        out.append(x)
    X = pd.concat(out, ignore_index=True)
    X["dec"] = T2.dec(X.ml.values)
    X["pnl"] = np.where(X.side_margin > 0, X.dec - 1, np.where(X.side_margin < 0, -1.0, 0.0))
    X["won"] = np.where(X.side_margin == 0, np.nan, (X.side_margin > 0).astype(float))
    pt = X.p_tie_cons.fillna(0.003)
    # EVs (tie = refund). p_*_s are P(win | no tie).
    for ref in ("p_sp", "p_sp_cons", "p_sp_sharp", "p_ml_sharp", "p_pin"):
        X[f"ev_{ref}"] = (X[ref + "_s"] * X.dec - 1) * (1 - pt)
    X["p_ref_s"] = X[["p_sp_sharp_s", "p_ml_sharp_s"]].mean(axis=1)                 # blend: sharp spread + sharp ML
    X["p_ref_s"] = X.p_ref_s.fillna(X.p_sp_cons_s)
    X["ev_blend"] = (X.p_ref_s * X.dec - 1) * (1 - pt)
    X["clv"] = X.dec * X.p_close_s - 1
    X["clv_sp"] = X.dec * X.pc_sp_s - 1
    X["clv_pin"] = X.dec * X.pc_pin_s - 1          # 2024-25 only (Pinnacle file), last Pinnacle snapshot
    X["gap_self"] = X.p_sp_s - X.p_ml_s           # + = this book's own spread says our side is likelier than its ML
    return X


def seg_mask(X, seg):
    p = X.cons_pt_side
    if seg == "all":
        return np.ones(len(X), bool)
    if seg == "small_dog":           # +1 .. +3.5
        return (p >= 1) & (p <= 3.5)
    if seg == "small_fav":           # -1 .. -3
        return (p <= -1) & (p >= -3)
    if seg == "dog_3to3.5":
        return (p >= 2.5) & (p <= 3.5)
    if seg == "around7_dog":
        return (p >= 6) & (p <= 8)
    if seg == "around7_fav":
        return (p <= -6) & (p >= -8)
    if seg == "dog":
        return p > 0
    if seg == "fav":
        return p < 0
    if seg == "dog_4to6.5":
        return (p >= 4) & (p <= 6.5)
    if seg == "fav_3.5to6.5":
        return (p <= -3.5) & (p >= -6.5)
    raise ValueError(seg)


def tot_mask(X, tot):
    if tot is None:
        return np.ones(len(X), bool)
    lo, hi = tot
    return (X.tot >= lo) & (X.tot <= hi)


def pick(X, rule):
    """Apply a rule; one bet per game: the first window snapshot with a qualifying book, best EV there."""
    w = rule["window"]
    if w == "kick":
        Y = X[X.is_last]
    elif w == "any":
        Y = X[X.win.isin(["tue", "fri", "sun"]) | X.is_last]
    elif w == "early":                                    # first of tue / fri / sun that qualifies
        Y = X[X.win.isin(["tue", "fri", "sun"])]
    else:
        Y = X[X.win == w]
    Y = Y[seg_mask(Y, rule["seg"]) & tot_mask(Y, rule.get("tot"))]
    ev = Y[f"ev_{rule['ref']}"]
    Y = Y[ev >= rule["th"]]
    if rule.get("self_th") is not None:
        Y = Y[Y.gap_self * Y.dec >= rule["self_th"]]
    if rule.get("confirm") is not None:            # second reference must also be >= confirm
        Y = Y[Y[f"ev_{rule['confirm'][0]}"] >= rule["confirm"][1]]
    Y = Y[Y.ml.between(-400, 400)]
    if Y.empty:
        return Y
    Y = Y.sort_values(f"ev_{rule['ref']}", ascending=False).drop_duplicates(["game_id", "requested_ts"])
    return Y.sort_values("requested_ts").drop_duplicates("game_id")


def summ_bets(Y, nseasons=1):
    if Y is None or len(Y) == 0:
        return {"bets": 0}
    roi, roi_se = mse(Y.pnl)
    clv, clv_se = mse(Y.clv)
    clvs, clvs_se = mse(Y.clv_sp)
    w = Y.won.dropna()
    return {"bets": int(len(Y)), "per_season": round(len(Y) / nseasons, 1), "avg_ml": int(np.median(Y.ml)),
            "win_rate": r4(w.mean()), "pred_win_ref": r4(Y.p_ref_s.mean()), "pred_win_close": r4(Y.p_close_s.mean()),
            "roi": roi, "roi_se": roi_se, "clv": clv, "clv_se": clv_se, "clv_t": r4(clv / clv_se) if clv_se else None,
            "clv_spread_close": clvs, "clv_spread_close_se": clvs_se,
            "clv_pinnacle": mse(Y.clv_pin)[0] if "clv_pin" in Y else None,
            "clv_pinnacle_se": mse(Y.clv_pin)[1] if "clv_pin" in Y else None,
            "n_pin": int(Y.clv_pin.notna().sum()) if "clv_pin" in Y else 0,
            "books": Y.book.value_counts().to_dict()}


def p2_descriptive(X, snaps) -> dict:
    """Which probability is right? ML-implied vs spread-implied (total-aware / single) vs outcomes, by segment."""
    out = {}
    W = windows(snaps)
    res = []
    for wname in ("tue", "fri", "sun", "kick"):
        S = W[W.is_last] if wname == "kick" else W[W.win == wname]
        S = S[S.m != 0].dropna(subset=["p_ml_cons", "p_sp_cons"])
        y = (S.m > 0).astype(float).values
        rec = {"window": wname, "n": int(len(S))}
        for c in ("p_ml_cons", "p_sp_cons", "p_sp1_cons", "p_ml_sharp", "p_sp_sharp", "p_pin"):
            if c in S and S[c].notna().sum() > 50:
                e = S[S[c].notna()]
                rec[c] = r4(bll(e[c], (e.m > 0)).mean())
        d = bll(S.p_sp_cons, y) - bll(S.p_ml_cons, y)
        rec["sp_minus_ml_logloss"] = r4(d.mean())
        rec["sp_minus_ml_se"] = r4(d.std(ddof=1) / math.sqrt(len(d)))
        res.append(rec)
    out["accuracy_by_window"] = res
    # favourite-perspective gap: ML consensus vs spread-implied consensus, by number and total (kick window)
    S = W[W.is_last].dropna(subset=["p_ml_cons", "p_sp_cons"]).copy()
    fh = S.pt_cons < 0
    S["fav_pt"] = np.where(fh, S.pt_cons, -S.pt_cons)
    S["pml_f"] = np.where(fh, S.p_ml_cons, 1 - S.p_ml_cons)
    S["psp_f"] = np.where(fh, S.p_sp_cons, 1 - S.p_sp_cons)
    S["psp1_f"] = np.where(fh, S.p_sp1_cons, 1 - S.p_sp1_cons)
    S["fwin"] = np.where(S.m == 0, np.nan, ((np.where(fh, S.m, -S.m)) > 0).astype(float))
    S["nb"] = pd.cut(-S.fav_pt, [-0.1, 1.25, 2.25, 2.75, 3.25, 3.75, 6.25, 6.75, 7.25, 7.75, 9.75, 30],
                     labels=["0-1", "1.5-2", "2.5", "3", "3.5", "4-6", "6.5", "7", "7.5", "8-9.5", "10+"])
    S["tb"] = pd.cut(S.tot, TBINS, labels=TLABS)
    rows = []
    for (nb), e in S.groupby("nb", observed=True):
        rows.append({"fav_by": str(nb), "n": int(len(e)), "p_ml": r4(e.pml_f.mean()), "p_sp_total_aware": r4(e.psp_f.mean()),
                     "p_sp_single": r4(e.psp1_f.mean()), "actual": r4(e.fwin.mean()),
                     "se": r4(math.sqrt(e.pml_f.mean() * (1 - e.pml_f.mean()) / max(1, e.fwin.notna().sum())))})
    out["fav_by_number_kick"] = rows
    rows = []
    for (tb), e in S.groupby("tb", observed=True):
        rows.append({"total": str(tb), "n": int(len(e)), "gap_ml_minus_sp": r4((e.pml_f - e.psp_f).mean()),
                     "p_ml": r4(e.pml_f.mean()), "p_sp": r4(e.psp_f.mean()), "actual": r4(e.fwin.mean())})
    out["fav_by_total_kick"] = rows
    # same-book self consistency: mean (own spread-implied - own ML no-vig), favourite perspective, by book and number
    R = X[(X.side == "home")].copy()
    fh = R.cons_pt_side < 0
    R["gap_f"] = np.where(fh, R.p_sp - R.p_ml, R.p_ml - R.p_sp)
    R["nb"] = pd.cut(np.abs(R.cons_pt_side), [-0.1, 1.25, 2.25, 2.75, 3.25, 3.75, 6.25, 6.75, 7.25, 7.75, 9.75, 30],
                     labels=["0-1", "1.5-2", "2.5", "3", "3.5", "4-6", "6.5", "7", "7.5", "8-9.5", "10+"])
    R = R[(R.win != "") | R.is_last]
    out["self_gap_fav_by_book_number"] = (R.groupby(["book", "nb"], observed=True).gap_f.agg(["mean", "count"])
                                          .round(4).reset_index().to_dict("records"))
    return out


def p2_grid():
    rules = []
    for w in ("fri", "sun", "kick", "any"):
        for seg in ("all", "dog", "fav", "small_dog", "small_fav", "dog_3to3.5", "around7_dog", "around7_fav",
                    "dog_4to6.5", "fav_3.5to6.5"):
            for ref in ("p_sp_sharp", "p_sp_cons", "ev_self", "blend", "p_ml_sharp"):
                for th in (0.0, 0.01, 0.02, 0.03):
                    for tot in (None, (0, 43.5), (44, 80)):
                        r = {"window": w, "seg": seg, "ref": ref if ref != "ev_self" else "p_sp", "th": th, "tot": tot}
                        r["id"] = f"{w}|{seg}|{ref}|{th}|{tot}"
                        rules.append(r)
    return rules


def run_p2dev():
    M = get_model("chosen")
    single = get_model("teasers_v2_single")
    from nflpred import odds as O
    allowed = O.load_allowed_books()
    rows, snaps = build_ml(DEV, M, single)
    X = bets_table(rows, snaps, allowed)
    X = windows(X)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV),
           "rows": int(len(X)), "games": int(X.game_id.nunique()),
           "tot_fallback_share": r4(snaps.tot_fallback.mean())}
    res["descriptive"] = p2_descriptive(X, snaps)
    # EV distribution of allowed-book MLs vs each reference
    res["ev_dist"] = {ref: {"mean": r4(X[f"ev_{ref}"].mean()), "share_pos": r4((X[f"ev_{ref}"] > 0).mean()),
                            "share_ge2": r4((X[f"ev_{ref}"] >= 0.02).mean())}
                      for ref in ("p_sp", "p_sp_cons", "p_sp_sharp", "p_ml_sharp", "blend")}
    grid = []
    for r in p2_grid():
        Y = pick(X, r)
        s = summ_bets(Y, len(DEV))
        s.pop("books", None)
        grid.append({"id": r["id"], **s})
    res["grid"] = grid
    store("p2dev", res)
    G = pd.DataFrame(grid)
    G = G[G.bets >= 30].sort_values("clv_t", ascending=False)
    print(G.head(40).to_string())


# ============================================================================================ PART 3
FIXED = T2.FIXED
WONG = T2.WONG


def teaser_legs(seasons, M) -> pd.DataFrame:
    """teasers_v2-compatible leg table (fixed-price books only), probabilities from the total-aware model."""
    rows, snaps = build_ml(seasons, M, get_model("teasers_v2_single"))
    R = rows[rows.sp_ok & rows.book.isin(FIXED)].merge(snaps, on=["game_id", "requested_ts"])
    R["mu_sharp_f"] = R.mu_sharp.fillna(R.mu_cons)
    last = snaps.sort_values("requested_ts").groupby("game_id").tail(1)
    last = last.assign(mu_close_sharp=last.mu_sharp.fillna(last.mu_cons), mu_close_cons=last.mu_cons,
                       tot_close=last.tot)
    R = R.merge(last[["game_id", "mu_close_sharp", "mu_close_cons", "tot_close"]], on="game_id")
    single = get_model("teasers_v2_single")
    out = []
    for side in ("home", "away"):
        x = R.copy()
        h = side == "home"
        x["side"] = side
        x["point"] = x.sp_home_point if h else -x.sp_home_point
        x["price"] = x.sp_home_price if h else x.sp_away_price
        out.append(x)
    L = pd.concat(out, ignore_index=True)
    hs = (L.side == "home").values
    L["tpoint"] = L.point + T2.TEASE
    L["side_margin"] = np.where(hs, L.m, -L.m)
    L["y"] = np.sign(L.side_margin + L.tpoint).astype(int)
    for c, mu, tot in (("sharp", "mu_sharp_f", "tot"), ("cons", "mu_cons", "tot"),
                       ("close_sharp", "mu_close_sharp", "tot_close"), ("close_cons", "mu_close_cons", "tot_close")):
        pw, pp = M.side_probs(L[mu].values, hs, L.tpoint.values, L[tot].values, L.ref.values)
        L[f"pw_{c}"], L[f"pp_{c}"] = pw, pp
    # single-weight model, same legs: needs that model's own implied mu (recompute from consensus number via p_sp1?)
    # -> approximate with the total-aware mu (differences in implied mu are < 0.05 pts); leg prob from single dist.
    pw1, pp1 = single.side_probs(L.mu_sharp_f.values, hs, L.tpoint.values)
    L["pw_single"], L["pp_single"] = pw1, pp1
    L["tot_snap"] = L.tot
    L["wong_num"] = L.point.isin(WONG)
    L["typ"] = np.where(L.point.isin(WONG[:3]), "wong_dog", np.where(L.point.isin(WONG[3:]), "wong_fav",
                        np.where(L.point > 0, "dog", np.where(L.point < 0, "fav", "pk"))))
    L["hours_before"] = (L.kick - L.requested_ts).dt.total_seconds() / 3600
    L["alt_dec_key"] = np.nan
    L["alt_dec_norm"] = np.nan
    return T2.assign_windows(L.drop(columns=["win"], errors="ignore"))


def leg_cal(L, by, pcols=("pw_sharp", "pw_single")):
    X = L[L.is_last | L.win.isin(["fri"])].drop_duplicates(["game_id", "side", "point", "win", "is_last"])
    X = X[X.y != 0]
    rows = []
    for k, e in X.groupby(by, observed=True):
        rec = {"key": str(k), "n": int(len(e)), "actual": r4((e.y > 0).mean())}
        for c in pcols:
            pn = e[c] / np.maximum(1 - e[c.replace("pw_", "pp_")], 1e-9)
            rec[c] = r4(pn.mean())
            rec[c + "_ll"] = r4(bll(pn, e.y > 0).mean())
        rows.append(rec)
    return rows


def teaser_rules():
    rules = []
    for w in ("fri", "kick"):
        for legs in ("wong", "any"):
            for ev_min in (-1.0, 0.0, 0.02):
                for tot in ("all", "le44", "gt44"):
                    rules.append({"id": f"{w}|{legs}|ev{ev_min}|{tot}", "window": w, "fair": "sharp", "legs": legs,
                                  "ev_min": ev_min, "max_pairs": 3, "tot": tot})
    return rules


def teaser_eval(L, r, n):
    X = L
    if r["tot"] == "le44":
        X = L[L.tot <= 44]
    elif r["tot"] == "gt44":
        X = L[L.tot > 44]
    T = T2.select(X, {k: v for k, v in r.items() if k != "tot"}, "fixed")
    s = T2.summarize(T, n)
    s.pop("by_book", None)
    s.pop("leg_types", None)
    return s, T


def buy_ladder(seasons, M):
    """buy_points ladders re-priced with the total-aware model (fair = sharp/cons median of each book's
    total-aware price-implied mu at the matched main snapshot)."""
    import buy_points as BP
    L, _ = BP.ladders(seasons)
    _, snaps = build_ml(seasons, M, None)
    sn = snaps[["game_id", "requested_ts", "mu_sharp", "mu_cons", "tot", "ref"]].rename(
        columns={"requested_ts": "main_ts", "mu_sharp": "mu_sharp_ta", "mu_cons": "mu_cons_ta"})
    L = L.merge(sn, on=["game_id", "main_ts"], how="inner")
    L["mu_ta"] = L.mu_sharp_ta.fillna(L.mu_cons_ta)
    last = snaps.sort_values("requested_ts").groupby("game_id").tail(1)
    last = last.assign(mu_close_ta=last.mu_sharp.fillna(last.mu_cons), tot_close=last.tot)
    L = L.merge(last[["game_id", "mu_close_ta", "tot_close"]], on="game_id")
    hs = (L.side == "home").values
    pw, pp = M.side_probs(L.mu_ta.values, hs, L.point.values, L.tot.values, L.ref.values)
    L["pw_ta"], L["pp_ta"] = pw, pp
    L["ev_ta"] = pw * L.dec + pp - 1
    pw, pp = M.side_probs(L.mu_close_ta.values, hs, L.point.values, L.tot_close.values, L.ref.values)
    L["ev_close_ta"] = pw * L.dec + pp - 1
    return L


def buy_apply(L, r):
    X = L[(L.src == "alt") & L.price.between(-300, 300) & L.has_main]
    X = X[X.window == r["window"]]
    if r.get("books"):
        X = X[X.book.isin(r["books"])]
    if r.get("tot_max") is not None:
        X = X[X.tot <= r["tot_max"]]
    if r.get("tot_min") is not None:
        X = X[X.tot >= r["tot_min"]]
    if r.get("transitions"):
        ok = np.zeros(len(X), bool)
        for f, t in r["transitions"]:
            ok |= (((X.ref_point - f).abs() < 1e-9) & ((X.point - t).abs() < 1e-9)).values
        X = X[ok]
    if r.get("bought"):
        X = X[X.dist > 0]
    if r.get("cross_key"):
        X = X[X.cross_key]
    if r.get("min_ev") is not None:
        X = X[X.ev_ta >= r["min_ev"]]
    return X.sort_values("ev_ta").groupby("game_id").tail(1)


def buy_summ(X):
    if len(X) == 0:
        return {"bets": 0}
    roi, se = mse(X.profit)
    clv, cse = mse(X.ev_close_ta)
    ev, _ = mse(X.ev_ta)
    ev_old, _ = mse(X.ev_sharp)
    nop = X[X.y != 0]
    return {"bets": int(len(X)), "hit": r4((nop.y > 0).mean()) if len(nop) else None, "pushes": int((X.y == 0).sum()),
            "avg_price": r4(np.median(X.price)), "ev_total_aware": ev, "ev_single_model": ev_old, "roi": roi,
            "roi_se": se, "clv_ta": clv, "clv_ta_se": cse}


ONTO = [[2.5, 3.0], [-3.5, -3.0], [6.5, 7.0], [-7.5, -7.0]]
ONTO3 = [[2.5, 3.0], [-3.5, -3.0]]
OFF = [[3.0, 3.5], [-3.0, -2.5], [7.0, 7.5], [-7.0, -6.5]]


def buy_rules():
    rules = []
    for w in ("early", "close"):
        for name, tr in (("onto3or7", ONTO), ("onto3", ONTO3), ("off3or7", OFF), ("any", None)):
            for tot in ((None, None), (None, 41.5), (None, 43.5), (44.0, None)):
                for books in (None, ["betrivers"]):
                    for mev in (None, 0.0):
                        r = {"window": w, "transitions": tr, "tot_max": tot[1], "tot_min": tot[0], "books": books,
                             "min_ev": mev, "bought": tr is None}
                        r["id"] = f"{w}|{name}|tot{tot}|{books[0] if books else 'all'}|ev{mev}"
                        rules.append(r)
    return rules


def run_p3dev():
    M = get_model("chosen")
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds")}
    # ---- teasers 2020-2022
    L = teaser_legs(DEV, M)
    L["tb"] = pd.cut(L.tot, [0, 41, 44, 47, 80], labels=["<=41", "41.5-44", "44.5-47", ">47"])
    W = L[L.wong_num]
    res["teaser_leg_cal_wong_by_total_dev"] = leg_cal(W, "tb")
    res["teaser_leg_cal_wong_by_type_total_dev"] = leg_cal(W, ["typ", "tb"])
    g = games_all()
    C = closes_wong(g, M, range(2006, 2020))
    res["closes_wong_cal_by_total_2006_2019"] = C
    res["closes_wong_cal_by_total_2020_2022"] = closes_wong(g, M, DEV)
    tg = []
    for r in teaser_rules():
        s, _ = teaser_eval(L, r, len(DEV))
        tg.append({"id": r["id"], **s})
    res["teaser_grid"] = tg
    store("p3dev", res)
    print(pd.DataFrame(res["teaser_leg_cal_wong_by_total_dev"]).to_string())
    print(pd.DataFrame(C).to_string())
    print(pd.DataFrame(tg).to_string())
    # ---- alternate buys 2023
    B = buy_ladder((2023,), M)
    bg = []
    for r in buy_rules():
        s = buy_summ(buy_apply(B, r))
        if s["bets"] >= 10:
            bg.append({"id": r["id"], **s})
    res["buy_grid_2023"] = bg
    res["buy_calib_2023"] = buy_calib(B)
    store("p3dev", res)
    print(pd.DataFrame(bg).to_string())
    print(pd.DataFrame(res["buy_calib_2023"]).to_string())


def closes_wong(g, M, seasons):
    """Wong legs at nflverse closing numbers (+ closing juice): calibration by total, total-aware vs single."""
    single = get_model("teasers_v2_single")
    e = g[g.season.isin(list(seasons))].reset_index(drop=True)
    rows = []
    for side in ("home", "away"):
        hs = np.full(len(e), side == "home")
        pt = np.where(hs, -e.spread_line.values, e.spread_line.values)
        mu = M.implied_mu(-e.spread_line.values, e.q.values, e.total_line.values, e.ref.values)
        pw, pp = M.side_probs(mu, hs, pt + 6, e.total_line.values, e.ref.values)
        mu1 = single.implied_mu(-e.spread_line.values, e.q.values)
        pw1, pp1 = single.side_probs(mu1, hs, pt + 6)
        sm = np.where(hs, e.m.values, -e.m.values)
        rows.append(pd.DataFrame({"point": pt, "pw_sharp": pw, "pp_sharp": pp, "pw_single": pw1, "pp_single": pp1,
                                  "y": np.sign(sm + pt + 6), "tot": e.total_line.values, "season": e.season.values}))
    X = pd.concat(rows)
    X = X[X.point.isin(WONG) & (X.y != 0)]
    X["tb"] = pd.cut(X.tot, [0, 41, 44, 47, 80], labels=["<=41", "41.5-44", "44.5-47", ">47"])
    X["typ"] = np.where(X.point > 0, "dog", "fav")
    out = []
    for k, d in X.groupby(["typ", "tb"], observed=True):
        rec = {"key": str(k), "n": int(len(d)), "actual": r4((d.y > 0).mean())}
        for c in ("pw_sharp", "pw_single"):
            pn = d[c] / np.maximum(1 - d[c.replace("pw_", "pp_")], 1e-9)
            rec[c] = r4(pn.mean())
            rec[c + "_ll"] = r4(bll(pn, d.y > 0).mean())
        out.append(rec)
    return out


def buy_calib(B):
    """Bought / sold alt lines: predicted (total-aware vs single) vs actual win rate (no push), close window."""
    X = B[(B.window == "close") & (B.y != 0) & B.price.between(-300, 300)].copy()
    X["db"] = pd.cut(X.dist, [-4, -2.25, -0.75, -0.25, 0.25, 0.75, 2.25, 4],
                     labels=["sold 2.5-3.5", "sold 1-2", "sold 0.5", "main", "bought 0.5", "bought 1-2", "bought 2.5-3.5"])
    X["low"] = X.tot <= 43.5
    out = []
    for (db, low), d in X.groupby(["db", "low"], observed=True):
        pta = d.pw_ta / np.maximum(1 - d.pp_ta, 1e-9)
        ps = d.pw_sharp / np.maximum(1 - d.pp_sharp, 1e-9)
        out.append({"dist": str(db), "low_total": bool(low), "n": int(len(d)), "actual": r4((d.y > 0).mean()),
                    "pred_total_aware": r4(pta.mean()), "pred_single": r4(ps.mean())})
    return out


# ============================================================================================ freeze / holdout
# Chosen after reading the dev results only (p2dev: 2020-2022; p3dev: teasers 2020-2022, alt lines 2023).
P2_RULES = [
    {"id": "ML1_sun_dog_blend1", "window": "sun", "seg": "dog", "ref": "blend", "th": 0.01, "tot": None,
     "desc": "Sunday 14:10 UTC snapshot. Dog (consensus spread > 0) moneyline at an allowed book, price <= +400, with "
             "EV >= +1% vs fair = mean(sharp spread-implied P(win) [lowvig/betonlineag spread + juice through the "
             "total-aware margin model], sharp no-vig ML) (fallback: consensus spread-implied). Best-EV book, one bet "
             "per game.",
     "dev": {"bets": 68, "clv": 0.0236, "clv_se": 0.0068, "roi": 0.0012, "roi_se": 0.1812}},
    {"id": "ML2_early_dog_blend1_tot44", "window": "early", "seg": "dog", "ref": "blend", "th": 0.01, "tot": [44, 80],
     "desc": "Same bet at the FIRST of the Tue 14:10 / Fri 21:40 / Sun 14:10 UTC snapshots where it qualifies, only when "
             "the consensus total at the snapshot is >= 44 (dog ML value in higher-scoring games).",
     "dev": {"bets": 135, "clv": 0.0309, "clv_se": 0.0073, "roi": 0.0801, "roi_se": 0.1248}},
    {"id": "ML3_early_spread_gap2", "window": "early", "seg": "all", "ref": "p_sp_sharp", "th": 0.02,
     "confirm": ["p_ml_sharp", 0.0], "tot": None,
     "desc": "Pure ML-vs-spread inconsistency: first Tue/Fri/Sun snapshot where an allowed book's ML (either side) has "
             "EV >= +2% vs the SHARP SPREAD-implied P(win) (total-aware model) AND EV >= 0 vs the sharp no-vig ML.",
     "dev": {"bets": 102, "clv": 0.0243, "clv_se": 0.0077, "roi": -0.0149, "roi_se": 0.1394}},
]
P3_RULES = [
    {"id": "T1_fri_wong_le44", "kind": "teaser", "window": "fri", "fair": "sharp", "legs": "wong", "ev_min": -1.0,
     "max_pairs": 3, "tot": "le44",
     "desc": "Friday 21:40 UTC: Wong legs AT THE BOOK's number (+1.5..+2.5 / -7.5..-8.5) at DK/FD/MGM/CZR, consensus "
             "total <= 44; best-EV same-book pairs (total-aware leg probs) at the book's fixed 2-team price, no EV "
             "threshold, up to 3 disjoint pairs per week.",
     "dev": {"teasers": 33, "roi": 0.0471, "roi_se": 0.1589, "ev_close": -0.0275}},
    {"id": "T2_kick_wong_le44", "kind": "teaser", "window": "kick", "fair": "sharp", "legs": "wong", "ev_min": -1.0,
     "max_pairs": 3, "tot": "le44", "desc": "As T1 at each game's last snapshot (~75 min pre-kick).",
     "dev": {"teasers": 16, "roi": 0.3507, "roi_se": 0.2015, "ev_close": -0.0327}},
    {"id": "B1_close_buy_onto_3or7_tot_le43_5", "kind": "buy", "window": "close", "transitions": ONTO, "tot_max": 43.5,
     "desc": "~75-min close: buy onto 3 or 7 (+2.5->+3, -3.5->-3, +6.5->+7, -7.5->-7) at the same book's alt price, "
             "consensus total <= 43.5; best allowed book by total-aware EV; no EV filter; one per game.",
     "dev_2023": {"bets": 60, "roi": 0.0389, "roi_se": 0.1102, "ev_total_aware": -0.0418}},
]


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen rules are final)")
    J = load_json()
    for k in ("part1", "part1b", "p2dev", "p3dev"):
        if k not in J:
            raise SystemExit(f"run {k} first")
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
          "margin_model": f"{J['part1b']['chosen_for_betting']} (ml_spread_consistency.json part1/part1b), chosen on "
                          "2020-2022 exact-margin log loss",
          "moneyline": {"developed_on": list(DEV), "holdout": list(HOLD), "rules": P2_RULES,
                        "clv": "dec * closing sharp no-vig P(side) (edge_lab.closing_fair p_close_sharp, fallback "
                               "p_close_all) - 1; also reported: vs closing sharp spread-implied P, vs last Pinnacle ML",
                        "pass_if": "CLV - 2*SE > 0 AND ROI > 0 (edge); CLV t >= 2 alone -> paper-track"},
          "teasers_buys": {"teasers_developed_on": list(DEV), "teasers_holdout": list(HOLD),
                           "buys_developed_on": [2023], "buys_holdout": [2024, 2025], "rules": P3_RULES,
                           "pass_if": "ROI - 1.64*SE > 0 (model EV at close is negative for all three: expected fail)"},
          "evaluate_once": True}
    FROZEN.write_text(json.dumps(fz, indent=1, default=_jd))
    print(json.dumps(fz, indent=1, default=_jd))


def run_holdout():
    if os.environ.get("MLSP_HOLDOUT") != "I_HAVE_FROZEN":
        raise SystemExit("holdout is locked: set MLSP_HOLDOUT=I_HAVE_FROZEN after freezing")
    if not FROZEN.exists():
        raise SystemExit("freeze first")
    J = load_json()
    if "holdout" in J:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    from nflpred import odds as O
    M = get_model("chosen")
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"]}
    # ---- moneyline 2023-2025
    rows, snaps = build_ml(HOLD, M, get_model("teasers_v2_single"))
    X = windows(bets_table(rows, snaps, O.load_allowed_books()))
    res["ml"] = {}
    bets = {}
    for r in fz["moneyline"]["rules"]:
        r = {**r, "tot": tuple(r["tot"]) if r.get("tot") else None}
        Y = pick(X, r)
        res["ml"][r["id"]] = {**summ_bets(Y, len(HOLD)),
                              "by_season": {int(s): summ_bets(d, 1) for s, d in Y.groupby("season")}}
        bets[r["id"]] = Y[["game_id", "requested_ts", "win", "book", "side", "ml", "cons_pt_side", "tot", "ev_blend",
                           "ev_p_sp_sharp", "ev_p_ml_sharp", "clv", "clv_sp", "clv_pin", "pnl"]].to_dict("records")
    res["ml_bets"] = bets
    res["ml_descriptive"] = p2_descriptive(X, snaps)
    store("holdout", res)
    # ---- teasers 2023-2025, buys 2024-2025
    L = teaser_legs(HOLD, M)
    L["tb"] = pd.cut(L.tot, [0, 41, 44, 47, 80], labels=["<=41", "41.5-44", "44.5-47", ">47"])
    res["teaser_leg_cal_wong_by_total"] = leg_cal(L[L.wong_num], "tb")
    res["closes_wong_cal_by_total_2023_2025"] = closes_wong(games_all(), M, HOLD)
    B = buy_ladder((2024, 2025), M)
    res["p3"] = {}
    for r in fz["teasers_buys"]["rules"]:
        if r["kind"] == "teaser":
            s_, T = teaser_eval(L, r, len(HOLD))
            s_["by_season"] = {int(k): T2.summarize(d, 1).get("roi") for k, d in T.groupby("season")} if len(T) else {}
            res["p3"][r["id"]] = s_
        else:
            Xb = buy_apply(B, r)
            res["p3"][r["id"]] = {**buy_summ(Xb), "by_season": {int(k): buy_summ(d) for k, d in Xb.groupby("season")}}
    res["buy_calib_2024_2025"] = buy_calib(B)
    store("holdout", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("ml_bets",)}, indent=1, default=_jd)[:12000])


# ============================================================================================ report
def _tbl(rows, cols, heads=None):
    heads = heads or cols
    out = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else (f"{r[c]:.4g}" if isinstance(r.get(c), float) else str(r[c]))
                                     for c in cols) + " |")
    return "\n".join(out)


def _pct(x, se=None):
    if x is None:
        return "n/a"
    return f"{100 * x:+.1f}%" + (f" ± {100 * se:.1f}" if se is not None else "")


VERDICT = """## Verdict

1. **Total-aware key numbers: no out-of-sample gain.** Kernel-raked weights w(|margin|; total) do pick up the classic
   pattern in the fit years (1999-2019: w3 = 3.33 at total 38 vs 2.96 at 50; w7 rises slightly with the total), but
   most of it is era (low totals = 1999-2011). On 2020-2025 the total-aware model is no better than the same model
   without the total (exact-margin log loss {g_ex}, exact-3 {g3}, exact-7 {g7}, cover outcomes at ±0.5 around
   3/7/10/14 {gcov} nats/game; positive = total-aware better). Actual 2020-25 exact 3s went the *other* way: high totals
   (>47) had 97 vs ~75 predicted, low totals (<=41) 40 vs 42-47. Exact 7 is over-predicted by every model fit before
   2020 (131 actual vs 155-163; only the old 2012-14 weights, 135, are close) -- a post-2020 drift no total
   adjustment fixes. LOSO picked sigma constant in the total (sigma slope 0) and no spread dependence.
2. **Moneyline vs spread: markets are consistent to ~0.5 pp.** ML-implied and spread-implied (total-aware) win
   probabilities have equal log loss at every window (diff ≤ 0.0012 ± 0.0008); the ML market prices favourites
   0.2-0.9 pp richer than the spread at every total (no total-dependent mispricing); each allowed book's ML sits within
   ~±1 pp of its own spread (up to 2 pp on 10+ point favourites; vs a 4-5% ML hold), so same-book inconsistency is never bettable on its own. What is
   real: when the sharp (LowVig/BetOnline) spread implies a different P(win) than the sharp ML, the ML moves toward
   the spread by close (slope 0.2-0.6 dev, 0.1-0.25 holdout, Sun t≈3). Combined with soft-book ML prices, that
   produced two rules that held up on 2023-25 by CLV (see table); the total>=44 version failed.
3. **Teasers / buys re-priced: unchanged verdicts.** Total-aware leg probabilities differ from the single-weight
   model by 0.2-1.4 pp, and not systematically by total (the difference is mostly sigma), so no teaser or buy flips
   sign. Frozen T1 (Fri Wong legs, total <= 44):
   {t1}; T2 (pre-kick): {t2}; B1 (buy onto 3/7 at the close, total <= 43.5, real alt prices): {b1} -- model EV
   -3.9%. Historical 'low-total Wong dogs win more' (2006-2019 closes: 0.80 at totals <= 41 vs 0.70 at > 47) did not
   repeat in 2023-25 (0.71 vs 0.78) and the margin model does not reproduce it either way.
"""


def run_report():
    J = load_json()
    P1, P1b, H, FZ = J["part1"], J["part1b"], J["holdout"], json.loads(FROZEN.read_text())
    v = P1b["validation_by_period"]
    sm = v["2020-2025"]["summary"]["total_aware_vs_pooled_1999_2019"]
    sm2 = v["2020-2025"]["summary"]["total_aware_vs_pooled_2012_2019"]
    f = lambda d: f"{d['gain_per_game']:+.4f} ± {d['se']:.4f}"  # noqa: E731
    p3 = H["p3"]
    t1, t2, b1 = p3["T1_fri_wong_le44"], p3["T2_kick_wong_le44"], p3["B1_close_buy_onto_3or7_tot_le43_5"]
    L = ["# Moneyline vs spread consistency, with a total-aware key-number margin model", "",
         "Code: `scripts/research/ml_spread_consistency.py` (stages part1 -> part1b -> p2dev -> p3dev -> freeze -> holdout "
         "-> report), reusable model `scripts/research/margin_by_total.py`. Numbers: `ml_spread_consistency.json`; "
         "frozen rules: `ml_spread_consistency_frozen.json` (frozen " + FZ["frozen_at"] + ", before any 2023-25 "
         "moneyline / teaser or 2024-25 alt-line evaluation in this study). Caveat: the 2023-25 seasons were already "
         "used as holdout by earlier studies (teasers_v2, buy_points, edge_lab); the rules here were not tuned on them.",
         "",
         VERDICT.format(g_ex=f"{f(sm2['ta2012_exact'])} (2012-19 fit; 1999-2019 fit {f(sm['exact'])})",
                        g3=f(sm2["ta2012_e3"]), g7=f(sm2["ta2012_e7"]), gcov=f(sm2["ta2012_cover"]),
                        t1=f"{t1['teasers']} teasers, ROI {_pct(t1['roi'], t1['roi_se'])}, model EV at close {_pct(t1['ev_close'])}",
                        t2=f"{t2['teasers']} teasers, ROI {_pct(t2['roi'], t2['roi_se'])}",
                        b1=f"{b1['bets']} bets, ROI {_pct(b1['roi'], b1['roi_se'])}"),
         "## Moneyline holdout (2023-2025, evaluated once)", "",
         "| rule | dev 2020-22 bets / CLV / ROI | holdout bets | avg ML | win (close-implied) | ROI ± SE | CLV vs sharp ML close ± SE | CLV vs Pinnacle close (n) | CLV vs sharp spread close | per season CLV |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in FZ["moneyline"]["rules"]:
        h = H["ml"][r["id"]]
        d = r["dev"]
        ps = ", ".join(f"{k}: {_pct(x.get('clv'))} ({x['bets']})" for k, x in h["by_season"].items())
        L.append(f"| {r['id']} | {d['bets']} / {_pct(d['clv'], d['clv_se'])} / {_pct(d['roi'], d['roi_se'])} | {h['bets']} | "
                 f"{h['avg_ml']:+d} | {h['win_rate']} ({h['pred_win_close']}) | {_pct(h['roi'], h['roi_se'])} | "
                 f"{_pct(h['clv'], h['clv_se'])} | {_pct(h['clv_pinnacle'], h['clv_pinnacle_se'])} ({h['n_pin']}) | "
                 f"{_pct(h['clv_spread_close'], h['clv_spread_close_se'])} | {ps} |")
    L += ["", "Rules (exact):", ""] + [f"* **{r['id']}** -- {r['desc']}" for r in FZ["moneyline"]["rules"]]
    L += ["", "Pre-registered test: CLV - 2·SE > 0 AND ROI > 0 = edge; CLV t >= 2 = paper-track. ML1 and ML3 pass the "
          "letter (CLV t 2.7 and 2.4, ROI positive but ±13-15%), ML2 (total >= 44) fails (CLV -1.2%). Context: the "
          "best-priced allowed-book dog ML at the same Sunday snapshot, unselected, has CLV -2.5% (holdout), so the "
          "selection lifts CLV ~3.5 pts; ML1/ML3 overlap on 88 games. CLV fell from dev (+2.4%/+2.4%) to holdout "
          "(+1.0%/+1.6%), and ML1 is a close cousin of the live moneyline v2/v3 tracks (soft price vs sharp no-vig); "
          "its new ingredient is the sharp spread-implied P(win) in the fair price. Dog CLV relies on proportional "
          "de-vig of the closing sharp ML; the Pinnacle close (2024-25) agrees. The 'vs sharp spread close' column is "
          "inflated for dogs (the ML market prices favourites 0.5-0.9 pp richer than the spread in 2023-25), so it is "
          "shown for completeness, not as evidence.", "",
          "## 1. Margin model (fit 1999-2019 and 2012-2019; validated 2020-2025)", "",
          f"Chosen by leave-season-block-out log likelihood: 1999-2019 {P1['chosen']}; 2012-2019 {P1b['chosen']}. "
          f"Used for parts 2-3: **{P1b['chosen_for_betting']}** ({P1b['chosen_rule']}).", "",
          "Key-number weights of the 1999-2019 total-aware fit by total:", "",
          _tbl([{"total": t, **w} for t, w in v["2020-2025"]["weights"].items()],
               ["total", "0", "3", "6", "7", "10", "14"]), "",
          "Out-of-sample log loss by model (nats/game; lower is better):", ""]
    rows = []
    for per, vv in v.items():
        for mname, rec in vv["per_model"].items():
            rows.append({"period": per, "model": mname, "exact": rec["exact_margin_logloss"],
                         "exact3 pred/act": f"{rec['exact3']['pred']:.0f}/{rec['exact3']['act']}",
                         "exact7 pred/act": f"{rec['exact7']['pred']:.0f}/{rec['exact7']['act']}",
                         "e3 ll": rec["exact3"]["logloss"], "e7 ll": rec["exact7"]["logloss"],
                         "cover ll (9 lines)": rec["cover_logloss_mean_9_lines"]})
    L += [_tbl(rows, list(rows[0].keys())), "", "Exact 3 / 7 by closing total, 2020-2025 (actual vs predicted):", "",
          _tbl(v["2020-2025"]["by_total"], ["total", "n", "act3", "pred3_total_aware", "pred3_total_aware_2012",
                                            "pred3_teasers_v2_single", "act7", "pred7_total_aware",
                                            "pred7_total_aware_2012", "pred7_teasers_v2_single"]), "",
          "Same, fit period 1999-2019 (in-sample for the total-aware fit):", "",
          _tbl(v["2020-2025"]["fit_period_by_total"], list(v["2020-2025"]["fit_period_by_total"][0].keys())), "",
          "## 2. Moneyline vs spread, descriptive", "",
          "Log loss of the home-win outcome by reference and window (dev 2020-22 | holdout 2023-25):", "",
          _tbl(J["p2dev"]["descriptive"]["accuracy_by_window"], ["window", "n", "p_ml_cons", "p_sp_cons", "p_sp1_cons",
                                                                 "p_ml_sharp", "p_sp_sharp", "sp_minus_ml_logloss", "sp_minus_ml_se"]), "",
          _tbl(H["ml_descriptive"]["accuracy_by_window"], ["window", "n", "p_ml_cons", "p_sp_cons", "p_sp1_cons",
                                                           "p_ml_sharp", "p_sp_sharp", "p_pin", "sp_minus_ml_logloss", "sp_minus_ml_se"]), "",
          "Favourite P(win) at the last snapshot, ML (no-vig consensus) vs spread-implied, by number (dev | holdout):", "",
          _tbl(J["p2dev"]["descriptive"]["fav_by_number_kick"], ["fav_by", "n", "p_ml", "p_sp_total_aware", "p_sp_single", "actual", "se"]), "",
          _tbl(H["ml_descriptive"]["fav_by_number_kick"], ["fav_by", "n", "p_ml", "p_sp_total_aware", "p_sp_single", "actual", "se"]), "",
          "By total (dev | holdout): the ML-minus-spread gap does not depend on the total.", "",
          _tbl(J["p2dev"]["descriptive"]["fav_by_total_kick"], ["total", "n", "gap_ml_minus_sp", "p_ml", "p_sp", "actual"]), "",
          _tbl(H["ml_descriptive"]["fav_by_total_kick"], ["total", "n", "gap_ml_minus_sp", "p_ml", "p_sp", "actual"]), "",
          "## 3. Teasers and point buys re-priced", "",
          "Wong legs (Fri + last snapshot, book numbers) win rate vs total-aware and single-weight predictions, dev 2020-22 | holdout 2023-25:", "",
          _tbl(J["p3dev"]["teaser_leg_cal_wong_by_total_dev"], ["key", "n", "actual", "pw_sharp", "pw_single"]), "",
          _tbl(H["teaser_leg_cal_wong_by_total"], ["key", "n", "actual", "pw_sharp", "pw_single"]), "",
          "Wong legs at nflverse closes, 2006-2019 | 2023-2025:", "",
          _tbl(J["p3dev"]["closes_wong_cal_by_total_2006_2019"], ["key", "n", "actual", "pw_sharp", "pw_single"]), "",
          _tbl(H["closes_wong_cal_by_total_2023_2025"], ["key", "n", "actual", "pw_sharp", "pw_single"]), "",
          "Frozen teaser / buy rules, holdout:", ""]
    rows = []
    for r in FZ["teasers_buys"]["rules"]:
        h = p3[r["id"]]
        rows.append({"rule": r["id"], "n": h.get("teasers", h.get("bets")), "roi": _pct(h["roi"], h["roi_se"]),
                     "model EV": _pct(h.get("ev_close", h.get("ev_total_aware"))), "desc": r["desc"]})
    L += [_tbl(rows, ["rule", "n", "roi", "model EV", "desc"]), "",
          "Alt-line calibration, close, 2023 dev (low total = consensus <= 43.5): bought 1-2 pts in low totals won "
          "0.583 vs 0.558 predicted -- the hint behind B1 -- but in 2024-25 the same cell was 0.558 vs 0.557 and the "
          "excess moved to high totals (0.572 vs 0.557); the total-aware and single models predict the same (±0.1 pt).", ""]
    MD.write_text("\n".join(L) + "\n")
    print(MD.read_text()[:3000])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "part1"
    {"part1": run_part1, "part1b": run_part1b, "p2dev": run_p2dev, "p3dev": run_p3dev, "freeze": run_freeze, "holdout": run_holdout, "report": run_report}[stage]()
