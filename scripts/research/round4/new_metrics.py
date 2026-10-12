"""Round 4: metrics not tested before (pre-registered in output/research/round4/new_metrics_prereg.json, 2026-10-12).

A. QB EPA+CPOE composite: team CPOE (offense minus allowed) and the starting QB's CPOE, added to the production model
   and tested against the closing-line residual.
B. FTN charting quality (2022+): INT-worthy throws, catchable-ball rate, drop rate, QB-fault sacks (market test).
C. Bye overpricing (arXiv 2408.10867): fade teams off a bye against the closing spread.

Walk-forward: every rating uses only a team's (or QB's) strictly earlier games (shift(1) inside the decayed sums).
Run: PYTHONPATH=src python scripts/research/round4/new_metrics.py  -> output/research/round4/new_metrics.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]
from nflpred import data, features as F, model as Mo  # noqa: E402
import matchups_refs as MR  # noqa: E402

OUT = ROOT / "output" / "research" / "round4"
PRE = json.loads((OUT / "new_metrics_prereg.json").read_text())


def pval1(t):   # one-sided upper tail
    return 0.5 * math.erfc(t / math.sqrt(2))


def decayed_rate(rows: pd.DataFrame, key: str, num: str, den: str, hl: float, k: float) -> pd.Series:
    """Shrunk decayed mean of num/den over strictly earlier rows of `key` (rows sorted by date)."""
    Sn = MR._decayed_prior_sum(rows, key, num, hl)
    Sd = MR._decayed_prior_sum(rows, key, den, hl)
    return Sn / (Sd + k)


def load_pbp(seasons):
    cols = ["game_id", "season", "season_type", "week", "posteam", "defteam", "passer_player_id", "cpoe", "pass_attempt",
            "qb_dropback", "play_id", "sack", "interception"]
    fr = []
    for s in seasons:
        p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet", columns=cols)
        fr.append(p)
    p = pd.concat(fr, ignore_index=True)
    p["posteam"], p["defteam"] = F._norm_team(p["posteam"]), F._norm_team(p["defteam"])
    return p


def two_sided(g: pd.DataFrame, per: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """per: (game_id, team, cols...) offense-perspective sums. Returns long rows with o_ and d_ sums per team-game."""
    o = per.rename(columns={c: f"o_{c}" for c in cols})
    d = per.rename(columns={"team": "opp"}).rename(columns={c: f"d_{c}" for c in cols})
    long = pd.concat([g[["game_id", "gameday", "home_team", "away_team"]].rename(columns={"home_team": "team", "away_team": "opp"}),
                      g[["game_id", "gameday", "away_team", "home_team"]].rename(columns={"away_team": "team", "home_team": "opp"})])
    long = long.merge(o, on=["game_id", "team"], how="left").merge(d, on=["game_id", "opp"], how="left")
    return long.fillna({c: 0 for c in long.columns if c.startswith(("o_", "d_"))}).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)


def to_game(g: pd.DataFrame, long: pd.DataFrame, col: str, name: str) -> pd.Series:
    h = g[["game_id", "home_team"]].merge(long[["game_id", "team", col]], left_on=["game_id", "home_team"], right_on=["game_id", "team"], how="left")[col]
    a = g[["game_id", "away_team"]].merge(long[["game_id", "team", col]], left_on=["game_id", "away_team"], right_on=["game_id", "team"], how="left")[col]
    return pd.Series(h.values - a.values, index=g.index, name=name)


# ---------------------------------------------------------------- A. CPOE
def cpoe_features(g: pd.DataFrame) -> pd.DataFrame:
    p = load_pbp(range(2012, 2026))
    p = p[p.cpoe.notna() & (p.pass_attempt == 1)]
    team = p.groupby(["game_id", "posteam"]).agg(cp=("cpoe", "sum"), n=("cpoe", "size")).reset_index().rename(columns={"posteam": "team"})
    long = two_sided(g, team, ["cp", "n"])
    long["o_r"] = decayed_rate(long, "team", "o_cp", "o_n", 8, 150)
    long["d_r"] = decayed_rate(long, "team", "d_cp", "d_n", 8, 150)
    long["net"] = long.o_r - long.d_r
    out = pd.DataFrame({"game_id": g.game_id})
    out["m_cpoe"] = to_game(g, long, "net", "m_cpoe").values
    # starting QB's own CPOE (decayed over his earlier games, any team)
    qb = p.groupby(["game_id", "passer_player_id"]).agg(cp=("cpoe", "sum"), n=("cpoe", "size")).reset_index()
    qb = qb.merge(g[["game_id", "gameday"]], on="game_id")
    starters = pd.concat([g[["game_id", "gameday", "home_qb_id"]].rename(columns={"home_qb_id": "qb"}),
                          g[["game_id", "gameday", "away_qb_id"]].rename(columns={"away_qb_id": "qb"})]).dropna()
    rows = starters.merge(qb.rename(columns={"passer_player_id": "qb"}), on=["game_id", "gameday", "qb"], how="outer").fillna({"cp": 0, "n": 0})
    rows = rows.sort_values(["qb", "gameday", "game_id"]).reset_index(drop=True)
    rows["r"] = decayed_rate(rows, "qb", "cp", "n", 20, 200)
    r = rows.drop_duplicates(["game_id", "qb"])[["game_id", "qb", "r"]]
    h = g[["game_id", "home_qb_id"]].merge(r, left_on=["game_id", "home_qb_id"], right_on=["game_id", "qb"], how="left").r.fillna(0)
    a = g[["game_id", "away_qb_id"]].merge(r, left_on=["game_id", "away_qb_id"], right_on=["game_id", "qb"], how="left").r.fillna(0)
    out["qb_cpoe_diff"] = h.values - a.values
    return out


# ---------------------------------------------------------------- B. FTN charting
def ftn_features(g: pd.DataFrame) -> pd.DataFrame:
    fr = []
    for s in (2022, 2023, 2024, 2025):
        f = data._release("ftn_charting", "ftn_charting", [s])
        fr.append(f)
    f = pd.concat(fr, ignore_index=True).rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    p = load_pbp(range(2022, 2026))[["game_id", "play_id", "posteam", "qb_dropback", "pass_attempt", "sack"]]
    f["play_id"] = pd.to_numeric(f["play_id"], errors="coerce")
    x = p.merge(f, on=["game_id", "play_id"], how="inner")
    x = x[x.qb_dropback == 1]
    b = lambda c: x[c].fillna(False).astype(bool)
    aimed = (x.pass_attempt == 1) & ~b("is_throw_away")
    x["db"] = 1
    x["iw"] = b("is_interception_worthy").astype(int)
    x["aimed"] = aimed.astype(int)
    x["catch"] = (aimed & b("is_catchable_ball")).astype(int)
    x["drop"] = (aimed & b("is_catchable_ball") & b("is_drop")).astype(int)
    x["qbf"] = ((x.sack == 1) & b("is_qb_fault_sack")).astype(int)
    per = x.groupby(["game_id", "posteam"])[["db", "iw", "aimed", "catch", "drop", "qbf"]].sum().reset_index().rename(columns={"posteam": "team"})
    gg = g[g.season >= 2022].reset_index(drop=True)
    long = two_sided(gg, per, ["db", "iw", "aimed", "catch", "drop", "qbf"])
    spec = {"iw": ("iw", "db", 150, -1), "catch": ("catch", "aimed", 150, +1), "drop": ("drop", "catch", 100, -1), "qbfault": ("qbf", "db", 150, -1)}
    out = pd.DataFrame({"game_id": gg.game_id})
    for name, (num, den, k, good) in spec.items():
        o = decayed_rate(long, "team", f"o_{num}", f"o_{den}", 8, k)
        d = decayed_rate(long, "team", f"d_{num}", f"d_{den}", 8, k)
        lg_o = (long[f"o_{num}"].sum() / max(long[f"o_{den}"].sum(), 1))       # level only (cancels in the net)
        long[f"net_{name}"] = good * ((o - lg_o) - (d - lg_o))                    # offense good-oriented minus defense allowed
        out[f"m_{name}"] = to_game(gg, long, f"net_{name}", name).values
    # warm-up: ratings need a few weeks of 2022
    return out


# ---------------------------------------------------------------- C. bye
def bye_rule(g: pd.DataFrame) -> dict:
    x = g[(g.game_type == "REG") & g.home_score.notna() & g.spread_line.notna() & (g.week > 1)].copy()
    hb, ab = x.home_rest >= 13, x.away_rest >= 13
    x = x[hb ^ ab]                                  # exactly one team off a bye
    bye_home = (x.home_rest >= 13)
    margin = x.home_score - x.away_score
    d = margin - x.spread_line                      # home ATS margin vs close (+ = home covered)
    fade = np.where(bye_home, -d, d)               # + = the side we bet (opponent of the bye team) covered
    x = x.assign(fade=fade, bye_res=np.where(bye_home, d, -d))
    out = {}
    for lab, yrs in (("dev_2012_19", range(2012, 2020)), ("holdout_2020_25", range(2020, 2026))):
        y = x[x.season.isin(list(yrs))]
        w, l = int((y.fade > 0).sum()), int((y.fade < 0).sum())
        n = w + l
        cr = w / n if n else None
        z = (w - n * 0.5238) / math.sqrt(n * 0.5238 * 0.4762) if n else 0
        out[lab] = {"games": len(y), "fade_w": w, "fade_l": l, "cover_rate": round(cr, 4) if cr else None,
                    "p_vs_52_4": round(pval1(z), 4), "roi_at_110": round((w * 0.9091 - l) / n, 4) if n else None,
                    "bye_team_margin_vs_close_pts": round(float(y.bye_res.mean()), 2), "se": round(float(y.bye_res.std() / math.sqrt(len(y))), 2)}
    h = out["holdout_2020_25"]
    out["passed"] = bool(h["cover_rate"] and h["cover_rate"] > 0.5238 and h["p_vs_52_4"] < 0.05 and out["dev_2012_19"]["cover_rate"] > 0.5)
    return out


def market(d: pd.DataFrame, feats: list[str], dev, hold, signs: dict | None = None) -> dict:
    res, k = {}, len(feats)
    for c in feats:
        a = MR.ols_z(d[d.season.isin(list(dev))].res_m, d[d.season.isin(list(dev))][c])
        b = MR.ols_z(d[d.season.isin(list(hold))].res_m, d[d.season.isin(list(hold))][c])
        sign = (signs or {}).get(c) or (np.sign(a["b"]) if a.get("b") else 1)
        p = pval1(sign * b["t"]) if b.get("t") is not None else 1.0
        res[c] = {"dev": a, "holdout": b, "tested_sign": int(sign), "holdout_p_one_sided": round(p, 4), "pass": bool(p < 0.05 / k)}
    return res


def model_test(g: pd.DataFrame, extra: pd.DataFrame, feats: list[str]) -> dict:
    import experiment as X
    cache = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/advmetrics/prod_df.pkl")
    df = pd.read_pickle(cache) if cache.exists() else X.run("baseline")[1]
    df = df.drop(columns=[c for c in feats if c in df.columns]).merge(extra[["game_id"] + feats], on="game_id", how="left")
    for c in feats:
        df[c] = df[c].fillna(0.0)
    res = {}
    for name, add in (("baseline", []), ("cpoe_both", feats), ("m_cpoe_only", ["m_cpoe"]), ("qb_cpoe_only", ["qb_cpoe_diff"])):
        r = {}
        for lab, seasons in (("val_2015_19", X.VAL), ("hold_2020_25", X.HOLD)):
            ps, ys = [], []
            for s in seasons:
                tr = Mo.train_rows(df, before_season=s); te = df[(df.season == s) & df.home_win.notna()]
                ps.append(Mo.fit_predict(tr, te, F.FEATURES + add, "margin")); ys.append(te.home_win.values)
            r[lab] = round(Mo.score(np.concatenate(ys), np.concatenate(ps))["log_loss"], 5)
        res[name] = r
    b = res["baseline"]
    for k, v in res.items():
        v["d_val"], v["d_hold"] = round(v["val_2015_19"] - b["val_2015_19"], 5), round(v["hold_2020_25"] - b["hold_2020_25"], 5)
        v["adopt"] = bool(v["d_val"] <= -0.001 and v["d_hold"] <= -0.001)
    return res


def persistence(d: pd.DataFrame, cols: list[str]) -> dict:
    return {c: int(d[c].notna().sum()) for c in cols}


def main():
    g = MR.load_games()
    out = {"prereg": "output/research/round4/new_metrics_prereg.json"}
    A = cpoe_features(g)
    dA = g.merge(A, on="game_id")
    out["A_qb_composite"] = {"model": model_test(g, A, ["m_cpoe", "qb_cpoe_diff"]),
                             "market": market(dA, ["m_cpoe", "qb_cpoe_diff"], range(2012, 2020), range(2020, 2026))}
    print(json.dumps(out["A_qb_composite"], indent=1, default=str)[:3000])
    B = ftn_features(g)
    dB = g.merge(B, on="game_id")
    dB = dB[~((dB.season == 2022))]                                    # 2022 = warm-up
    out["B_ftn_quality"] = {"market": market(dB, ["m_iw", "m_catch", "m_drop", "m_qbfault"], [2023], [2024, 2025], {"m_drop": -1}),
                            "n_games": {s: int((dB.season == s).sum()) for s in (2023, 2024, 2025)}}
    print(json.dumps(out["B_ftn_quality"], indent=1, default=str)[:3000])
    out["C_bye_overpricing"] = bye_rule(g)
    print(json.dumps(out["C_bye_overpricing"], indent=1))
    (OUT / "new_metrics.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
