"""NFL 6-point teaser research (Wong teasers), with a frozen-rule holdout.

Stages (run in order; each writes into output/research/):
  python scripts/research/teasers.py dev       # analysis on seasons <= 2022 only -> teasers.json["dev"]
  python scripts/research/teasers.py freeze    # writes teasers_frozen.json (refuses to overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/teasers.py holdout
                                               # 2023-2025, frozen rules evaluated ONCE -> teasers.json["holdout"]
  python scripts/research/teasers.py report    # renders teasers.md from teasers.json

Conventions
  * A "leg" is one team in one game, teased 6 points: covers iff side_margin + side_spread + 6 > 0,
    pushes iff == 0. side_spread < 0 = favorite. nflverse spread_line = expected HOME margin (+ = home fav).
  * Leg win rate = W / (W + L) (pushes excluded); push rate reported separately.
  * Teaser grading, two conventions (books differ; see md):
      tie_loses : a pushed leg makes the 2-team teaser lose.
      reduce    : a pushed leg is removed and the remaining leg is paid as a straight bet at -110;
                  both legs push -> stake refunded.
  * Teaser ROI (per 1 unit staked):
      implied  : from pooled leg outcome rates assuming independent legs (different games), CI by
                 bootstrap over (season, week) clusters.
      realized : legs paired inside each (season, week) in kickoff order (odd leg dropped), actual
                 grading, CI by bootstrap over (season, week) clusters.
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
OUT = ROOT / "output" / "research"
JSON = OUT / "teasers.json"
FROZEN = OUT / "teasers_frozen.json"
MD = OUT / "teasers.md"
DEV_LAST = 2022
HOLD = (2023, 2024, 2025)
WONG_DOG = (1.5, 2.0, 2.5)
WONG_FAV = (-7.5, -8.0, -8.5)
PRICES_2T = (-110, -120, -130, -140)
REDUCE_PRICE = -110
RNG_SEED = 20260930
NBOOT = 4000
PERIODS = [(1999, 2011), (2012, 2016), (2017, 2019), (2020, 2022), (2023, 2025)]


# ------------------------------------------------------------------------------------------- helpers
def dec(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def per_label(season: int) -> str:
    for a, b in PERIODS:
        if a <= season <= b:
            return f"{a}-{b}"
    return "other"


def total_bucket(t: float) -> str:
    if t < 41:
        return "a <41"
    if t < 44:
        return "b 41-43.5"
    if t < 47:
        return "c 44-46.5"
    if t < 49:
        return "d 47-48.5"
    return "e >=49"


def leg_type(spread: float) -> str:
    if spread in WONG_DOG:
        return "wong_dog"
    if spread in WONG_FAV:
        return "wong_fav"
    return "dog" if spread > 0 else ("fav" if spread < 0 else "pk")


# ------------------------------------------------------------------------------------------- legs
def closing_legs(seasons) -> pd.DataFrame:
    """Two rows per game (home, away) from nflverse closing lines."""
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[g.season.isin(list(seasons)) & g.home_score.notna() & g.spread_line.notna()].copy()
    g["kick"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"))
    rows = []
    for side, s in (("home", 1), ("away", -1)):
        x = g[["game_id", "season", "week", "game_type", "kick", "home_team", "away_team", "home_score",
               "away_score", "spread_line", "total_line", "roof", "div_game"]].copy()
        x["side"] = side
        x["team"] = np.where(s == 1, x.home_team, x.away_team)
        x["spread"] = -s * x.spread_line
        x["margin"] = s * (x.home_score - x.away_score)
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    return grade(L, "spread")


def grade(L: pd.DataFrame, col: str, tease: float = 6.0, prefix: str = "") -> pd.DataFrame:
    adj = L.margin + L[col] + tease
    L[prefix + "w"] = (adj > 0).astype(int)
    L[prefix + "p"] = (adj == 0).astype(int)
    L[prefix + "l"] = (adj < 0).astype(int)
    if not prefix:
        L["period"] = L.season.map(per_label)
        L["tb"] = L.total_line.map(total_bucket)
        L["typ"] = L[col].map(leg_type)
        L["wk"] = L.season.astype(str) + "-" + L.week.astype(str).str.zfill(2)
    return L


def rate_table(L: pd.DataFrame, by, prefix: str = "") -> list[dict]:
    w, p, l = prefix + "w", prefix + "p", prefix + "l"
    out = []
    for k, d in L.groupby(by, observed=True):
        W, P, Lo = int(d[w].sum()), int(d[p].sum()), int(d[l].sum())
        n = W + Lo
        r = W / n if n else float("nan")
        out.append({**dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,))),
                    "legs": W + P + Lo, "W": W, "P": P, "L": Lo, "win_rate": round(r, 4),
                    "se": round(math.sqrt(r * (1 - r) / n), 4) if n else None,
                    "push_rate": round(P / (W + P + Lo), 4)})
    return out


# ------------------------------------------------------------------------------------------- teaser economics
def breakevens() -> dict:
    be = {}
    for a in PRICES_2T:
        be[f"2team_{a}"] = round(math.sqrt(1 / dec(a)), 4)
    for a in (160, 180):
        be[f"3team6pt_+{a}"] = round((1 / dec(a)) ** (1 / 3), 4)
    for a in (-120, -110, 100):
        be[f"3team10pt_{a:+d}"] = round((1 / dec(a)) ** (1 / 3), 4)
    return be


def implied_roi(W, P, Lo, d, rule):
    n = W + P + Lo
    pw, pp = W / n, P / n
    if rule == "tie_loses":
        return pw * pw * d - 1
    red = dec(REDUCE_PRICE) - 1
    return pw * pw * (d - 1) + 2 * pw * pp * red - (1 - (pw + pp) ** 2)


def pair_legs(L: pd.DataFrame) -> pd.DataFrame:
    """Disjoint pairs within (season, week) in kickoff order; legs from the same game never paired."""
    rows = []
    for wk, d in L.sort_values(["kick", "game_id"]).groupby("wk"):
        d = d.drop_duplicates("game_id")  # never tease both sides of one game
        v = d[["w", "p", "l"]].values
        for i in range(0, len(v) - 1, 2):
            rows.append((wk, d.season.iloc[0], *v[i], *v[i + 1]))
    return pd.DataFrame(rows, columns=["wk", "season", "w1", "p1", "l1", "w2", "p2", "l2"])


def pair_pnl(P: pd.DataFrame, d: float, rule: str) -> np.ndarray:
    lose = (P.l1 == 1) | (P.l2 == 1)
    both_w = (P.w1 == 1) & (P.w2 == 1)
    if rule == "tie_loses":
        return np.where(both_w, d - 1, -1.0)
    red = dec(REDUCE_PRICE) - 1
    one_push = ((P.p1 + P.p2) == 1) & ~lose
    return np.where(lose, -1.0, np.where(both_w, d - 1, np.where(one_push, red, 0.0)))


def roi_block(L: pd.DataFrame, nboot: int = NBOOT) -> dict:
    """ROI at each price/rule, implied (leg-rate) and realized (paired), week-cluster bootstrap CIs."""
    rng = np.random.default_rng(RNG_SEED)
    wk = L.groupby("wk")[["w", "p", "l"]].sum()
    P = pair_legs(L)
    pwk = P.groupby("wk")
    wks = wk.index.values
    idx = rng.integers(0, len(wks), size=(nboot, len(wks)))
    Wb = wk.w.values[idx].sum(1)
    Pb = wk.p.values[idx].sum(1)
    Lb = wk.l.values[idx].sum(1)
    res = {"legs": int(len(L)), "seasons": int(L.season.nunique()),
           "legs_per_season": round(len(L) / L.season.nunique(), 1),
           "teasers": int(len(P)), "teasers_per_season": round(len(P) / L.season.nunique(), 1),
           "leg_win_rate": round(L.w.sum() / (L.w.sum() + L.l.sum()), 4),
           "leg_win_rate_se": round(math.sqrt((r := L.w.sum() / (L.w.sum() + L.l.sum())) * (1 - r) / (L.w.sum() + L.l.sum())), 4),
           "leg_push_rate": round(L.p.mean(), 4), "by_price": {}}
    # realized bootstrap over the same weeks (weeks without a pair contribute nothing)
    pw_ids = {k: i for i, k in enumerate(wks)}
    for a in PRICES_2T:
        d = dec(a)
        for rule in ("tie_loses", "reduce"):
            imp = implied_roi(L.w.sum(), L.p.sum(), L.l.sum(), d, rule)
            boot = np.array([implied_roi(Wb[i], Pb[i], Lb[i], d, rule) for i in range(nboot)])
            rec = {"implied_roi": round(imp, 4), "implied_ci90": [round(float(np.quantile(boot, .05)), 4),
                                                                 round(float(np.quantile(boot, .95)), 4)],
                   "implied_se": round(float(boot.std()), 4)}
            if len(P):
                pnl = pair_pnl(P, d, rule)
                s = pd.Series(pnl).groupby(P.wk.values).agg(["sum", "size"]).reindex(wks, fill_value=0)
                bs, bn = s["sum"].values[idx].sum(1), s["size"].values[idx].sum(1)
                bb = bs / np.maximum(bn, 1)
                rec.update({"realized_roi": round(float(pnl.mean()), 4),
                            "realized_ci90": [round(float(np.quantile(bb, .05)), 4), round(float(np.quantile(bb, .95)), 4)],
                            "realized_se": round(float(bb.std()), 4),
                            "realized_units": round(float(pnl.sum()), 2)})
            res["by_price"][f"{a}_{rule}"] = rec
    n = L.w.sum() + L.p.sum() + L.l.sum()
    pw = L.w.sum() / n
    res["three_team_6pt_implied_tie_loses"] = {f"+{a}": round(pw ** 3 * dec(a) - 1, 4) for a in (160, 180)}
    return res


def exec_legs(L: pd.DataFrame, S: pd.DataFrame, col: str = "best_late") -> pd.DataFrame:
    """Closing legs re-graded at an executable snapshot number (col) instead of the nflverse close."""
    M = L.drop(columns=["w", "p", "l"]).merge(S, on=["game_id", "side"], how="inner")
    M = M[M[col].notna()].copy()
    M["spread_exec"] = M[col]
    return grade(M, "spread_exec")


# ------------------------------------------------------------------------------------------- snapshots (2020+)
def snapshot_legs(holdout: bool) -> pd.DataFrame:
    """Per game side: best allowed-book spread at an early snapshot (Tue ~) and at the last snapshot."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import edge_lab as E
    t = E.load(holdout)
    t = t[t.point.notna() & t.sp_price.between(-150, 130)]
    t = t[t.hours_before >= 0.25]
    # early: first snapshot of the game week that is 72-150h before kickoff (Mon/Tue for a Sunday game)
    early = t[t.hours_before.between(72, 150)]
    first_ts = early.groupby("game_id").requested_ts.min().rename("ets")
    e = early.merge(first_ts, on="game_id")
    e = e[e.requested_ts == e.ets]
    last_ts = t.groupby("game_id").requested_ts.max().rename("lts")
    la = t.merge(last_ts, on="game_id")
    la = la[la.requested_ts == la.lts]
    def best(x, name):
        b = x.groupby(["game_id", "side"]).agg(**{f"best_{name}": ("point", "max"), f"cons_{name}": ("point", "median"),
                                                  f"hb_{name}": ("hours_before", "first")})
        return b
    B = best(e, "early").join(best(la, "late"), how="outer").reset_index()
    return B


def early_vs_close(C: pd.DataFrame, S: pd.DataFrame) -> dict:
    """C = closing legs for the same seasons; S = snapshot best numbers."""
    M = C.merge(S, on=["game_id", "side"], how="inner")
    out = {"games_matched": int(M.game_id.nunique())}
    for col in ("best_early", "cons_early", "best_late"):
        M = grade(M, col, prefix=col + "_")
    # move of the best early number relative to the close, for Wong-range legs
    def summarize(sel, lab, basis):
        d = M[sel]
        r = {"legs": int(len(d))}
        for col in ("spread", "best_early", "cons_early", "best_late"):
            pre = "" if col == "spread" else col + "_"
            W, Lo, P = d[pre + "w"].sum(), d[pre + "l"].sum(), d[pre + "p"].sum()
            r[f"rate_at_{col}"] = round(W / (W + Lo), 4) if W + Lo else None
            r[f"push_at_{col}"] = int(P)
        diff = d["best_early_w"] - d["best_early_l"] - (d["w"] - d["l"])  # per-leg (+1 win/-1 loss) difference
        r["mean_pts_better_early_vs_close"] = round(float((d.best_early - d.spread).mean()), 3)
        r["share_early_better"] = round(float((d.best_early > d.spread).mean()), 3)
        r["share_early_worse"] = round(float((d.best_early < d.spread).mean()), 3)
        r["legs_flipped_to_win_by_early"] = int(((d.best_early_w == 1) & (d.w == 0)).sum())
        r["legs_flipped_to_loss_by_early"] = int(((d.best_early_w == 0) & (d.w == 1)).sum())
        r["basis"] = basis
        out[lab] = r
    wong = lambda s: s.isin(WONG_DOG + WONG_FAV)
    summarize(wong(M.spread), "wong_at_close", "legs in Wong range at nflverse close")
    summarize(wong(M.best_early), "wong_at_best_early", "legs in Wong range at best allowed-book number, early (Mon/Tue) snapshot")
    summarize(wong(M.best_early) & ~wong(M.spread), "wong_early_only", "Wong at best early number but NOT at close")
    summarize(wong(M.spread) & ~wong(M.best_early), "wong_close_only", "Wong at close but NOT at best early number")
    summarize(wong(M.best_late), "wong_at_best_late", "legs in Wong range at best allowed-book number, last snapshot (~1h pre-kick)")
    # line value in teaser terms: dog legs that improve from +1.5 -> +2.5 etc. Just the distribution of moves
    d = M[wong(M.best_early)]
    out["move_dist_best_early_minus_close"] = {str(k): int(v) for k, v in (d.best_early - d.spread).round(1).value_counts().sort_index().items()}
    return out


# ------------------------------------------------------------------------------------------- rules
def apply_rule(L: pd.DataFrame, rule: dict, numcol: str = "spread") -> pd.DataFrame:
    x = L
    sp = x[numcol]
    m = sp.isin(rule["spreads"])
    if "total_max" in rule:
        m &= x.total_line <= rule["total_max"]
    if "game_types" in rule:
        m &= x.game_type.isin(rule["game_types"])
    return x[m]


def run_dev():
    seasons = range(1999, DEV_LAST + 1)
    L = closing_legs(seasons)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": [1999, DEV_LAST],
           "breakevens": breakevens()}
    teasable = L[L.spread.between(-10.5, 10.5)].copy()
    teasable["sb"] = teasable.spread
    res["by_spread_2012_2022"] = rate_table(teasable[teasable.season >= 2012], ["sb"])
    W = L[L.typ.isin(["wong_dog", "wong_fav"])]
    res["wong_by_period_type"] = rate_table(W, ["period", "typ"])
    res["wong_by_period"] = rate_table(W, ["period"])
    W12 = W[W.season >= 2012]
    res["wong_2012_2022_by_spread"] = rate_table(W12, ["spread"])
    res["wong_2012_2022_by_total"] = rate_table(W12, ["tb"])
    res["wong_2012_2022_by_total_type"] = rate_table(W12, ["tb", "typ"])
    res["wong_2012_2022_by_side_type"] = rate_table(W12, ["side", "typ"])
    res["wong_2012_2022_by_gametype"] = rate_table(W12, ["game_type"])
    res["wong_by_period_totalcut49"] = rate_table(W.assign(t49=np.where(W.total_line < 49, "<49", ">=49")), ["period", "t49"])
    res["wong_2012_2022_by_roof"] = rate_table(W12.assign(dome=W12.roof.isin(["dome", "closed"])), ["dome"])
    res["wong_2012_2022_by_div"] = rate_table(W12, ["div_game"])
    # ROI of simple rules on dev (2012-2022 and 2017-2022)
    rois = {}
    for lab, sub in (("wong_all", W), ("wong_total_lt49", W[W.total_line < 49])):
        for a, b in ((1999, 2011), (2012, 2022), (2017, 2022)):
            s = sub[sub.season.between(a, b)]
            rois[f"{lab}_{a}_{b}"] = roi_block(s)
    res["dev_rule_rois"] = rois
    # early vs close, 2020-2022
    S = snapshot_legs(False)
    res["early_vs_close_2020_2022"] = early_vs_close(L[L.season >= 2020], S)
    L20 = L[L.season >= 2020]
    ex = exec_legs(L20, S)
    rois["R1_close_2020_2022"] = roi_block(apply_rule(L20, {"spreads": list(WONG_DOG + WONG_FAV)}))
    rois["R2_close_2020_2022"] = roi_block(apply_rule(L20, {"spreads": list(WONG_DOG + WONG_FAV), "total_max": 48.5}))
    rois["R3_bestlate_2020_2022"] = roi_block(apply_rule(ex, {"spreads": list(WONG_DOG + WONG_FAV), "total_max": 48.5}, "spread_exec"))
    rois["wong_bestearly_2020_2022"] = roi_block(apply_rule(exec_legs(L20, S, "best_early"), {"spreads": list(WONG_DOG + WONG_FAV)}, "spread_exec"))
    store("dev", res)
    return res


# Candidate rules decided AFTER reading only the dev output (seasons <= 2022). See md for the reasoning.
CANDIDATES = [
    {"id": "R1_wong_classic",
     "desc": "Every leg whose nflverse closing spread is +1.5/+2/+2.5 (dog) or -7.5/-8/-8.5 (fav), all game types; "
             "pair legs within a week in kickoff order.",
     "spreads": list(WONG_DOG + WONG_FAV)},
    {"id": "R2_wong_total_le48_5",
     "desc": "R1 restricted to closing total_line <= 48.5 (the published 'total < 49' filter).",
     "spreads": list(WONG_DOG + WONG_FAV), "total_max": 48.5},
    {"id": "R3_wong_best_late_book_total_le48_5",
     "desc": "Executable version of R2: Wong range judged on the BEST allowed-book number at the last pre-kick snapshot "
             "(~1h before kickoff; my_books.json books), teased from that number; closing total <= 48.5. "
             "Evaluated only where snapshot data exists (2023-2025).",
     "spreads": list(WONG_DOG + WONG_FAV), "total_max": 48.5, "number": "best_late"},
]


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} already exists; frozen rules are never overwritten")
    dev = json.loads(JSON.read_text())["dev"]
    FROZEN.write_text(json.dumps({"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
                                  "developed_on": "seasons <= 2022 only (nflverse closes 1999-2022; snapshots 2020-2022)",
                                  "dev_generated": dev["generated"],
                                  "holdout": list(HOLD), "evaluate_once": True,
                                  "prices": list(PRICES_2T), "grading": ["tie_loses", "reduce"],
                                  "candidates": CANDIDATES}, indent=1))
    print("frozen", FROZEN)


def run_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout locked: freeze first and set EDGE_HOLDOUT")
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    if "holdout" in cur:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    L = closing_legs(HOLD)
    S = snapshot_legs(True)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"], "rules": {}}
    for r in fz["candidates"]:
        if r.get("number") == "best_late":
            sel = apply_rule(exec_legs(L, S), r, "spread_exec")
        else:
            sel = apply_rule(L, r)
        blk = roi_block(sel)
        blk["by_season"] = rate_table(sel, ["season"])
        blk["by_type"] = rate_table(sel, ["typ"])
        res["rules"][r["id"]] = blk
    # descriptive tables for the holdout period (reported after the freeze)
    W = L[L.typ.isin(["wong_dog", "wong_fav"])]
    res["wong_2023_2025_by_type"] = rate_table(W, ["typ"])
    res["wong_2023_2025_by_spread"] = rate_table(W, ["spread"])
    res["wong_2023_2025_by_total"] = rate_table(W, ["tb"])
    res["wong_2023_2025_by_side_type"] = rate_table(W, ["side", "typ"])
    res["wong_2023_2025_totalcut49"] = rate_table(W.assign(t49=np.where(W.total_line < 49, "<49", ">=49")), ["t49"])
    T = L[L.spread.between(-10.5, 10.5)].copy()
    res["by_spread_2023_2025"] = rate_table(T.assign(sb=T.spread), ["sb"])
    res["early_vs_close_2023_2025"] = early_vs_close(L, S)
    store("holdout", res)
    return res


def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float)[:20000])


# ------------------------------------------------------------------------------------------- report
def _md(rows, cols=None):
    if not rows:
        return "_none_\n"
    df = pd.DataFrame(rows)
    cols = cols or [c for c in df.columns if c not in ("se",)] + (["se"] if "se" in df.columns else [])
    df = df[cols]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df.itertuples(index=False):
        out += "| " + " | ".join(str(v) for v in r) + " |\n"
    return out


def _roi_rows(blocks: dict, prices=PRICES_2T, rule="tie_loses"):
    rows = []
    for name, b in blocks.items():
        for a in prices:
            r = b["by_price"][f"{a}_{rule}"]
            rows.append({"rule/sample": name, "legs": b["legs"], "leg_win": b["leg_win_rate"], "leg_se": b["leg_win_rate_se"],
                         "teasers/season": b["teasers_per_season"], "price": a,
                         "ROI implied [90% CI]": f"{r['implied_roi']:+.3f} [{r['implied_ci90'][0]:+.3f}, {r['implied_ci90'][1]:+.3f}]",
                         "ROI realized ± SE": f"{r.get('realized_roi', float('nan')):+.3f} ± {r.get('realized_se', float('nan')):.3f}",
                         "units": r.get("realized_units")})
    return rows


def run_report():
    J = json.loads(JSON.read_text())
    dev, hold = J["dev"], J.get("holdout")
    fz = json.loads(FROZEN.read_text())
    md = [NARRATIVE.strip() + "\n"]
    md.append("\n## 1. Break-even leg win rates\n")
    md.append(_md([{"bet": k, "break-even leg rate": v} for k, v in dev["breakevens"].items()]))
    md.append("\n## 2. Leg win rates (nflverse closing spreads, 6 points)\n")
    md.append("Win rate = W/(W+L); pushes shown separately. 2023-2025 rows were computed only after the freeze.\n")
    per = [r for r in dev["wong_by_period_type"]]
    if hold:
        per += [{"period": "2023-2025", **r} for r in hold["wong_2023_2025_by_type"]]
    md.append("\n### Wong legs by period and type\n" + _md(per, ["period", "typ", "legs", "W", "P", "L", "win_rate", "se"]))
    tc = dev["wong_by_period_totalcut49"] + ([{"period": "2023-2025", **r} for r in hold["wong_2023_2025_totalcut49"]] if hold else [])
    md.append("\n### Wong legs by period and closing total (< 49 vs >= 49)\n" + _md(tc, ["period", "t49", "legs", "W", "L", "win_rate", "se"]))
    md.append("\n### Wong legs by spread (2012-2022 dev)\n" + _md(dev["wong_2012_2022_by_spread"]))
    if hold:
        md.append("\n### Wong legs by spread (2023-2025)\n" + _md(hold["wong_2023_2025_by_spread"]))
    md.append("\n### Wong legs by total bucket x type (2012-2022 dev)\n" + _md(dev["wong_2012_2022_by_total_type"]))
    if hold:
        md.append("\n### Wong legs by total bucket (2023-2025)\n" + _md(hold["wong_2023_2025_by_total"]))
    md.append("\n### Wong legs by home/away x type (2012-2022 dev)\n" + _md(dev["wong_2012_2022_by_side_type"]))
    if hold:
        md.append("\n### Wong legs by home/away x type (2023-2025)\n" + _md(hold["wong_2023_2025_by_side_type"]))
    md.append("\n### Wong legs by game type / dome / division (2012-2022 dev)\n" + _md(dev["wong_2012_2022_by_gametype"])
              + "\n" + _md(dev["wong_2012_2022_by_roof"]) + "\n" + _md(dev["wong_2012_2022_by_div"]))
    md.append("\n### All teasable legs by closing spread, 2012-2022 (context: which numbers cross 3 and 7)\n" + _md(dev["by_spread_2012_2022"]))
    md.append("\n## 3. Early-week number vs close (2020-2022 dev; 2023-2025 after freeze)\n")
    md.append("best_early = best number among my_books.json books at the first snapshot 72-150h pre-kick (Mon/Tue for Sunday games); "
              "best_late = best allowed-book number at the last snapshot (~1h pre-kick); spread = nflverse close. "
              "rate_at_X = the SAME legs graded when teased from number X.\n")
    ev = []
    for lab, key in (("2020-2022", "early_vs_close_2020_2022"), ("2023-2025", "early_vs_close_2023_2025")):
        src = dev if key in dev else (hold or {})
        if key not in src:
            continue
        for sel, r in src[key].items():
            if isinstance(r, dict) and "legs" in r:
                ev.append({"seasons": lab, "selection": sel, "legs": r["legs"], "at_close": r["rate_at_spread"],
                           "at_best_early": r["rate_at_best_early"], "at_best_late": r["rate_at_best_late"],
                           "early_minus_close_pts": r["mean_pts_better_early_vs_close"],
                           "flip_to_win": r["legs_flipped_to_win_by_early"], "flip_to_loss": r["legs_flipped_to_loss_by_early"]})
    md.append(_md(ev))
    md.append("\n## 4. Development ROI (seasons <= 2022; in-sample for rule choice), tie_loses grading\n")
    md.append(_md(_roi_rows(dev["dev_rule_rois"], prices=(-120, -130, -140))))
    md.append("\n## 5. Frozen rules (" + fz["frozen_at"] + ", before any 2023-2025 evaluation)\n")
    for c in fz["candidates"]:
        md.append(f"- **{c['id']}**: {c['desc']}\n")
    if hold:
        md.append("\n## 6. Holdout 2023-2025 (evaluated once)\n")
        md.append("\n### tie_loses grading\n" + _md(_roi_rows(hold["rules"])))
        md.append("\n### reduce grading (pushed leg -> straight bet at -110)\n" + _md(_roi_rows(hold["rules"], rule="reduce")))
        for k, b in hold["rules"].items():
            md.append(f"\n**{k}** by season\n" + _md(b["by_season"]) + "\nby leg type\n" + _md(b["by_type"])
                      + f"\n3-team 6-pt implied ROI (tie_loses): {b['three_team_6pt_implied_tie_loses']}\n")
    md.append(BOOKS_NOTE)
    MD.write_text("".join(md))
    print("wrote", MD)


NARRATIVE = """
# NFL 6-point teasers (Wong teasers): does an edge survive today's prices?

Generated by `scripts/research/teasers.py` (stages dev -> freeze -> holdout -> report). Data: nflverse closing
spreads/totals 1999-2025 (all game types), and 2020-2025 multi-book snapshots (`edge_lab`, books in `my_books.json`).

## Verdict

**No bettable edge demonstrated at today's typical prices.** Wong legs (dog +1.5..+2.5, fav -7.5..-8.5 teased 6)
still win more often than a random teaser leg, but not by enough to clear -130/-140 pricing with any confidence:

* Frozen rules, 2023-2025 holdout (run once): leg win rates 0.744 (R1 all Wong), 0.752 (R2 total <= 48.5),
  0.765 (R3 best-book number ~1h pre-kick, total <= 48.5), each with SE ~0.028-0.029.
  Break-evens: 0.7385 (-120), 0.7518 (-130), 0.7638 (-140). So roughly: small positive at -120, break-even at -130,
  slightly negative at -140; every 90% CI spans roughly -15% to +20% ROI.
* Pooled 2012-2025 (dev + holdout), Wong legs with total < 49: 551-164 = 0.771 (SE 0.016). Implied 2-team ROI
  +8.9% at -120, +5.1% at -130, +1.8% at -140 -- and that pooled figure contains the development years used to pick the
  filter. Against -130 it is only ~1.2 SE above break-even; against -140, ~0.4 SE.
* Leg type split flipped: in 2023-2025 Wong dogs went 148-40 (0.787) but Wong favorites 41-25 (0.621, vs 0.76 in
  2012-2022). Not a frozen rule, so treat as a descriptive observation, not a new filter (it is ~2 SE and exactly
  the kind of split that regresses).
* The total < 49 filter held its direction out of sample (<49: 0.752 vs >=49: 0.688, n=32), but most Wong legs now have
  totals below 49 anyway, so it barely changes which bets you make.
* Early-week numbers did not help (section 3): legs in Wong range at the close won LESS often when graded at the best
  Mon/Tue number (dev 0.755 vs 0.776 at close; holdout 0.731 vs 0.744). Shopping for the best number ~1h before
  kickoff was neutral-to-slightly-positive (holdout R3 0.765 vs R2 0.752), within noise.
* 3-team 6-pt at +160 has a LOWER break-even (0.727) than 2-team at -120 (0.7385); if a book really offers +160 it is
  the least-bad teaser format on paper (holdout R2 implied +10.7%, but the ROI SE is ~0.13). At +140 (break-even 0.747)
  it is no better than 2-team -130.

Caveats: samples are small (~58-85 Wong legs/season, ~18-38 two-team teasers/season depending on rule and era); nflverse closing spreads are a
consensus number, not necessarily what your book posts (books shade +1.5/+2.5/-7.5 on teaser-heavy games); push grading
is nearly irrelevant because Wong legs are mostly half-point numbers (push rate < 1%), so tie_loses and reduce differ
only for +2/-8 legs. Totals used as filters are the nflverse CLOSING total (slight look-ahead for an early bet).

Grading assumptions: `tie_loses` (a pushed leg loses the 2-team teaser) and `reduce` (pushed leg dropped, remaining leg
paid as a straight bet at -110, both push = refund). Book-specific push rules were NOT verified.
ROI "implied" = p^2 x decimal - 1 from pooled leg outcomes (independent legs), 90% CI by bootstrap over (season, week);
"realized" = legs actually paired within each week in kickoff order (odd leg dropped), SE by the same week bootstrap.
"""

BOOKS_NOTE = """

## 7. Book pricing (practical)

One secondary source only -- bettingusa.com teaser guide (https://www.bettingusa.com/sports/teaser/, "last updated
June 5, 2026") lists 6-pt NFL teasers as: DraftKings 2-team -120 / 3-team +160; FanDuel -134 / +140; BetMGM -130 / +160;
Caesars -120 / +160; BetRivers uses dynamic teaser pricing (price depends on the legs). Not independently verified;
prices vary by state and change, and some books price teasers per-leg (worse on Wong-type legs).
ESPN Bet, Fanatics, Hard Rock: **unknown** (not found). Push rules per book: **unknown** (not verified) -- check house
rules; for Wong legs this matters only on +2 / -8.

Implication: at -120 (DK/Caesars per that source) Wong teasers have hovered around break-even-to-small-plus
historically; at -130/-134/-140 the measured edge is within noise of zero or negative. There is no evidence here that
justifies more than small recreational stakes.
"""


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": run_dev, "freeze": run_freeze, "holdout": run_holdout, "report": run_report}[stage]()
