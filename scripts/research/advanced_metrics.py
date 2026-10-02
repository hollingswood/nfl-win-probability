"""ADVANCED TEAM METRICS (havoc, PROE/pace, early/late downs, red zone, drives, OL rushing, luck, 4th-down
aggressiveness): do they add to the production model, and does the market misprice them?

Stages (numbers go to output/research/advanced_metrics.json; the .md is written by hand from it):
  python scripts/research/advanced_metrics.py build       # per team-game stats -> walk-forward ratings -> game features
  python scripts/research/advanced_metrics.py stability   # does each pre-game rating predict the same stat in-game?
  python scripts/research/advanced_metrics.py model       # production margin model + metric groups (experiment.py protocol)
  python scripts/research/advanced_metrics.py dev         # market tests: 2012-19 close residuals, 2020-22 close + Tue CLV
  python scripts/research/advanced_metrics.py freeze      # writes advanced_metrics_frozen.json (never overwritten)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/advanced_metrics.py holdout   # 2023-25, ONCE

Leakage discipline (same as matchups_refs.py)
  * Ratings are decayed sums (half-life 8 games, as production) over a team's STRICTLY earlier games (shift(1)),
    expressed as the deviation from a league rate computed from the last 256 team-games played before the Tuesday
    of the game week, shrunk with a pseudo-count k. Everything is known at the Tuesday 14:10 UTC snapshot.
  * Models used inside a metric (late-down expected conversion by down x distance, expected 4th-down go rate) are
    fit only on seasons before the target season (2012 = warm-up: its own season; 2012 rows only feed ratings).
  * Season-to-date record metrics (wins vs Pythagorean, close-game record) use only earlier games of the season.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]

from nflpred import features as F  # noqa: E402
import matchups_refs as MR  # noqa: E402  (reused helpers: league reference, decayed sums, OLS, CLV tables)

OUT = ROOT / "output" / "research"
JSON = OUT / "advanced_metrics.json"
FROZEN = OUT / "advanced_metrics_frozen.json"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/advmetrics")
DEV1 = range(2012, 2020)           # closing-line residual development
DEV2 = (2020, 2021, 2022)          # second development period (close residual + early-line CLV)
HOLD = (2023, 2024, 2025)
HL = 8
SD_SEASONS = range(2013, 2020)     # fixed z scale
FTN_URL = MR.FTN_URL


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def save_json(key, obj):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d[key] = MR.jsonable(obj)
    JSON.write_text(json.dumps(d, indent=1))


# ================================================================== per team-game raw stats (offense perspective)
PBP_COLS = ["game_id", "play_id", "posteam", "defteam", "play_type", "pass", "rush", "down", "ydstogo", "yardline_100",
            "epa", "success", "yards_gained", "first_down", "touchdown", "td_team", "interception", "fumble",
            "fumble_forced", "fumble_lost", "fumble_out_of_bounds", "fumble_recovery_1_team", "fumbled_1_team",
            "tackled_for_loss", "sack", "pass_defense_1_player_id", "xpass", "pass_oe", "wp", "qtr",
            "half_seconds_remaining", "game_seconds_remaining", "score_differential", "no_huddle", "qb_scramble",
            "fixed_drive", "fixed_drive_result", "drive_inside20", "drive_first_downs", "qb_kneel", "qb_spike",
            "season_type", "penalty", "posteam_type"]


def _xconv_table(late: pd.DataFrame) -> pd.Series:
    t = late.assign(d=late.ydstogo.clip(1, 20)).groupby(["down", "d"]).conv.agg(["sum", "count"])
    return (t["sum"] + 1) / (t["count"] + 2)


def _go_model(fd: pd.DataFrame):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = fd[["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining", "wp"]].fillna(0.5).values
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=40).fit(X, fd.go.values)


def team_game_raw() -> pd.DataFrame:
    cache = SCR / "raw.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    SCR.mkdir(parents=True, exist_ok=True)
    seasons = {}
    for s in range(2012, 2026):
        p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet", columns=PBP_COLS)
        for c in ("posteam", "defteam", "td_team", "fumble_recovery_1_team", "fumbled_1_team"):
            p[c] = F._norm_team(p[c].astype(object))
        p["season"] = s
        seasons[s] = p
    # ---- walk-forward helper models (fit on earlier seasons; 2012 on itself)
    late_all, fourth_all = {}, {}
    for s, p in seasons.items():
        sc = p[p.play_type.isin(["pass", "run"]) & p.down.isin([3, 4]) & p.qb_kneel.ne(1) & p.qb_spike.ne(1)]
        late_all[s] = sc.assign(conv=((sc.first_down == 1) | ((sc.touchdown == 1) & (sc.td_team == sc.posteam))) * 1.0)[
            ["down", "ydstogo", "conv"]]
        f4 = p[p.down.eq(4) & p.play_type.isin(["pass", "run", "punt", "field_goal"]) & p.qb_kneel.ne(1)].copy()
        f4["go"] = f4.play_type.isin(["pass", "run"]).astype(int)
        fourth_all[s] = f4
    out = []
    for s, p in seasons.items():
        prior = [x for x in seasons if x < s] or [s]
        xt = _xconv_table(pd.concat([late_all[x] for x in prior]))
        gm = _go_model(pd.concat([fourth_all[x] for x in prior[-4:]]))
        sc = p[p.play_type.isin(["pass", "run"]) & p.qb_kneel.ne(1) & p.qb_spike.ne(1) & p.epa.notna()].copy()
        ps = sc["pass"].eq(1)
        td_off = (sc.touchdown == 1) & (sc.td_team == sc.posteam)
        early = sc.down.isin([1, 2])
        late = sc.down.isin([3, 4])
        conv = ((sc.first_down == 1) | td_off) * 1.0
        key = list(zip(sc.down.fillna(0).astype(int), sc.ydstogo.clip(1, 20).fillna(1).astype(int)))
        xconv = pd.Series([xt.get(k, np.nan) for k in key], index=sc.index).where(late)
        havoc = ((sc.tackled_for_loss == 1) | (sc.sack == 1) | (sc.fumble_forced == 1) | (sc.interception == 1) |
                 sc.pass_defense_1_player_id.notna())
        neutral = early & sc.wp.between(0.2, 0.8) & (sc.half_seconds_remaining > 120) & sc.qtr.le(4)
        xp = sc.xpass.notna() & sc.pass_oe.notna()
        dr = sc.play_type.eq("run") & sc.qb_scramble.ne(1)
        y = sc.yards_gained.fillna(0)
        aly = np.where(y < 0, 1.2 * y, np.minimum(y, 4) + 0.5 * np.clip(y - 4, 0, 6))
        pd_n = sc.pass_defense_1_player_id.notna() & sc.interception.ne(1)
        d = pd.DataFrame({
            "game_id": sc.game_id, "team": sc.posteam, "opp": sc.defteam,
            "plays": 1.0, "havoc_n": havoc * 1.0,
            "xp_n": xp * 1.0, "proe_s": (sc.pass_oe / 100).where(xp, 0).fillna(0),
            "nxp_n": (xp & neutral) * 1.0, "nproe_s": (sc.pass_oe / 100).where(xp & neutral, 0).fillna(0),
            "ed_n": early * 1.0, "ed_epa_s": sc.epa.where(early, 0), "ed_sr_s": sc.success.where(early, 0).fillna(0),
            "ld_n": late * 1.0, "ld_conv_s": conv.where(late, 0), "ld_cx_s": (conv - xconv).where(late, 0).fillna(0),
            "ld_epa_s": sc.epa.where(late, 0),
            "dr_n": dr * 1.0, "stuff_n": (dr & (y <= 0)) * 1.0, "aly_s": np.where(dr, aly, 0),
            "second_s": np.where(dr, np.clip(y - 5, 0, 5), 0), "open_s": np.where(dr, np.clip(y - 10, 0, None), 0),
            "int_n": (ps & sc.interception.eq(1)) * 1.0, "intpd_n": (ps & (sc.interception.eq(1) | pd_n)) * 1.0,
        })
        g1 = d.groupby(["game_id", "team", "opp"]).sum()
        # ---- neutral pace: seconds between consecutive snaps of the same drive (pass/run/no_play rows)
        q = p[p.play_type.isin(["pass", "run", "no_play"])].sort_values(["game_id", "play_id"])
        q = q.assign(prev_t=q.groupby(["game_id", "fixed_drive"]).game_seconds_remaining.shift(1))
        q["dt"] = q.prev_t - q.game_seconds_remaining
        okp = (q.play_type.isin(["pass", "run"]) & q.dt.between(1, 50) & q.qtr.le(3)
               & q.score_differential.abs().le(7))
        pace = q[okp].groupby(["game_id", "posteam"]).agg(pace_n=("dt", "size"), pace_s=("dt", "sum"),
                                                         nohud_n=("no_huddle", "sum")).rename_axis(["game_id", "team"])
        # ---- drives
        dv = p[p.posteam.notna() & p.fixed_drive.notna() & p.play_type.isin(["pass", "run", "punt", "field_goal", "no_play"])]
        dv = dv.sort_values(["game_id", "play_id"])
        dsc = dv[dv.play_type.isin(["pass", "run", "punt", "field_goal"])]
        agg = dsc.groupby(["game_id", "fixed_drive"]).agg(
            team=("posteam", "first"), opp=("defteam", "first"), result=("fixed_drive_result", "first"),
            in20=("drive_inside20", "max"), fd=("drive_first_downs", "first"), start=("yardline_100", "first"),
            last_yl=("yardline_100", "last"), last_y=("yards_gained", "last"), last_type=("play_type", "last"),
            kneel=("qb_kneel", "min"))
        agg = agg[~agg.result.isin(["End of half", "End of game"]) & agg.kneel.ne(1) & agg.start.notna()].reset_index()
        tdd = agg.result.eq("Touchdown")
        end = np.where(tdd, 0, np.where(agg.last_type.isin(["pass", "run"]) & ~agg.result.isin(["Turnover"]),
                                        agg.last_yl - agg.last_y.fillna(0), agg.last_yl)).clip(0, 100)
        drv = pd.DataFrame({"game_id": agg.game_id, "team": agg.team, "opp": agg.opp, "drives": 1.0,
                            "pts_s": np.where(tdd, 7.0, np.where(agg.result.eq("Field goal"), 3.0, 0.0)),
                            "rz_n": agg.in20.fillna(0).clip(0, 1) * 1.0, "rz_td_n": (agg.in20.eq(1) & tdd) * 1.0,
                            "series_n": agg.fd.fillna(0) + 1.0, "fd_s": agg.fd.fillna(0) * 1.0,
                            "avail_den": agg.start * 1.0, "avail_s": (agg.start - end).clip(lower=0),
                            "start_s": 100.0 - agg.start, "three_out_n": (agg.fd.fillna(0).eq(0) & agg.result.eq("Punt")) * 1.0})
        g2 = drv.groupby(["game_id", "team", "opp"]).sum()
        # ---- 4th-down decisions (go vs expected)
        f4 = fourth_all[s]
        f4 = f4[f4.posteam.notna()]
        X = f4[["ydstogo", "yardline_100", "score_differential", "game_seconds_remaining", "wp"]].fillna(0.5).values
        f4 = f4.assign(pgo=gm.predict_proba(X)[:, 1])
        g4 = f4.groupby(["game_id", "posteam"]).agg(fourth_n=("go", "size"), go_n=("go", "sum"), pgo_s=("pgo", "sum"))
        g4["goox_s"] = g4.go_n - g4.pgo_s
        g4 = g4.rename_axis(["game_id", "team"])[["fourth_n", "goox_s"]]
        # ---- fumbles (all fumbles in the game with a known recovering team)
        fu = p[p.fumble.eq(1) & p.fumble_out_of_bounds.ne(1) & p.fumble_recovery_1_team.notna() & p.fumbled_1_team.notna()]
        fu = fu.assign(r=(fu.fumble_recovery_1_team == fu.fumbled_1_team) * 1.0, one=1.0)
        own = fu.groupby(["game_id", "fumbled_1_team"]).agg(own_fum=("one", "sum"), own_rec=("r", "sum"))
        own = own.rename_axis(["game_id", "team"])
        g = g1.join(g2, how="outer").reset_index().set_index(["game_id", "team"])
        g = g.join(pace, how="left").join(g4, how="left").join(own, how="left").fillna(0).reset_index()
        g["season"] = s
        out.append(g)
    u = pd.concat(out, ignore_index=True)
    # FTN interception-worthy throws (2022+): descriptive only
    iw = []
    for s in range(2022, 2026):
        f = MR._fetch(FTN_URL.format(s), MR.SCR / f"ftn_{s}.parquet")
        f = f[["nflverse_game_id", "nflverse_play_id", "is_interception_worthy"]].rename(
            columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        p = seasons[s][["game_id", "play_id", "posteam", "interception", "pass", "play_type"]]
        p = p[p.play_type.eq("pass")].merge(f.drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"])
        p["iw"] = p.is_interception_worthy.astype(float)
        iw.append(p.groupby(["game_id", "posteam"]).agg(iw_n=("iw", "sum"), ftn_db=("iw", "size")).reset_index()
                  .rename(columns={"posteam": "team"}))
    u = u.merge(pd.concat(iw), on=["game_id", "team"], how="left").fillna({"iw_n": 0, "ftn_db": 0})
    u.to_parquet(cache)
    return u


# ================================================================== ratings
# name: (numerator, denominator, pseudo-count k in denominator units, family)
STATS = {
    "havoc": ("havoc_n", "plays", 200, "havoc"),
    "proe": ("proe_s", "xp_n", 150, "style"),
    "nproe": ("nproe_s", "nxp_n", 100, "style"),
    "npace": ("pace_s", "pace_n", 100, "style"),
    "nohud": ("nohud_n", "pace_n", 150, "style"),
    "ed_epa": ("ed_epa_s", "ed_n", 150, "downs"),
    "ed_sr": ("ed_sr_s", "ed_n", 150, "downs"),
    "ld_conv": ("ld_conv_s", "ld_n", 60, "downs"),
    "ld_cx": ("ld_cx_s", "ld_n", 60, "luck"),
    "ld_epa": ("ld_epa_s", "ld_n", 60, "downs"),
    "rz_td": ("rz_td_n", "rz_n", 15, "luck"),
    "ppd": ("pts_s", "drives", 30, "drives"),
    "dsr": ("fd_s", "series_n", 60, "drives"),
    "avail": ("avail_s", "avail_den", 1500, "drives"),
    "start_fp": ("start_s", "drives", 30, "drives"),
    "three_out": ("three_out_n", "drives", 30, "drives"),
    "stuff": ("stuff_n", "dr_n", 100, "ol"),
    "aly": ("aly_s", "dr_n", 100, "ol"),
    "second": ("second_s", "dr_n", 100, "ol"),
    "open": ("open_s", "dr_n", 100, "ol"),
    "fumrec": ("own_rec", "own_fum", 10, "luck"),
    "int_share": ("int_n", "intpd_n", 30, "luck"),
    "int_worthy": ("int_n", "iw_n", 15, "luck_ftn"),
    "go_oe": ("goox_s", "fourth_n", 20, "fourth"),
}
# sign so that + = good for the OFFENSE (net team rating = o - d). Style stats keep their natural sign.
GOOD = {"stuff": -1, "three_out": -1, "int_share": -1, "int_worthy": -1, "havoc": -1}


def unit_ratings(g: pd.DataFrame) -> pd.DataFrame:
    cache = SCR / "ratings.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    u = team_game_raw().merge(g[["game_id", "gameday", "cutoff"]], on="game_id")
    dfn = u.rename(columns={"team": "opp", "opp": "team"})
    res = {}
    for side, rows in (("o", u), ("d", dfn)):
        lg = rows.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
        tgt = lg[["game_id", "team"]].copy()
        for name, (num, den, k, _) in STATS.items():
            L = MR._league_ref(u, num, den, lg.cutoff)
            Sn = MR._decayed_prior_sum(lg, "team", num, HL)
            Sd = MR._decayed_prior_sum(lg, "team", den, HL)
            sgn = GOOD.get(name, 1)
            tgt[f"{side}_{name}"] = sgn * (Sn - np.nan_to_num(L) * Sd) / (Sd + k)
            tgt[f"{side}_{name}_n"] = Sd
        res[side] = tgt
    r = res["o"].merge(res["d"], on=["game_id", "team"])
    r.to_parquet(cache)
    return r


def game_level(g: pd.DataFrame) -> pd.DataFrame:
    """Record-based team metrics per (game_id, team), using only the team's earlier games."""
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        x = g[["game_id", "season", "gameday", f"{side}_team", f"{opp}_team", f"{side}_score", f"{opp}_score"]].copy()
        x.columns = ["game_id", "season", "gameday", "team", "opp", "pf", "pa"]
        rows.append(x)
    t = pd.concat(rows).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    done = t.pf.notna()
    t["g"] = done * 1.0
    t["pf"], t["pa"] = t.pf.fillna(0), t.pa.fillna(0)
    t["w"] = np.where(done, (t.pf > t.pa) + 0.5 * (t.pf == t.pa), 0.0)
    t["close"] = (done & ((t.pf - t.pa).abs() <= 8)) * 1.0
    t["close_sur"] = np.where(t.close == 1, t.w - 0.5, 0.0)
    t["pdiff"] = t.pf - t.pa
    out = t[["game_id", "team", "opp", "season"]].copy()
    E = 2.37
    for tag, grp in (("ew", ["team"]), ("std", ["team", "season"])):
        if tag == "ew":
            S = {c: MR._decayed_prior_sum(t, "team", c, HL) for c in ("g", "w", "pf", "pa", "close", "close_sur", "pdiff")}
        else:
            S = {c: t.groupby(grp)[c].transform(lambda v: v.shift(1).fillna(0).cumsum()) for c in
                 ("g", "w", "pf", "pa", "close", "close_sur", "pdiff")}
        pf, pa = S["pf"] + 1e-9, S["pa"] + 1e-9
        pyth = pf ** E / (pf ** E + pa ** E)
        out[f"g_pyth_{tag}"] = np.where(S["g"] > 0, (S["w"] - S["g"] * pyth) / (S["g"] + 4), 0.0)
        out[f"g_close_{tag}"] = S["close_sur"] / (S["g"] + 4)
        out[f"g_pd_{tag}"] = S["pdiff"] / (S["g"] + 3)
        out[f"g_winpct_{tag}"] = np.where(S["g"] > 0, (S["w"] - 0.5 * S["g"]) / (S["g"] + 2), 0.0)
    # schedule strength faced: mean pre-game point-diff rating of earlier opponents (decayed)
    opp_r = out[["game_id", "team", "g_pd_ew"]].rename(columns={"team": "opp", "g_pd_ew": "opp_pd"})
    t = t.merge(opp_r, on=["game_id", "opp"], how="left")
    t["opp_pd"] = np.where(done, t.opp_pd.fillna(0), 0.0)
    t = t.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    sos = MR._decayed_prior_sum(t, "team", "opp_pd", HL) / (MR._decayed_prior_sum(t, "team", "g", HL) + 2)
    out = out.merge(t[["game_id", "team"]].assign(g_sos=sos.values), on=["game_id", "team"])
    return out.drop(columns=["opp", "season"])


def build():
    g = MR.load_games()
    r = unit_ratings(g).merge(game_level(g), on=["game_id", "team"], how="outer")
    rcols = [c for c in r.columns if c not in ("game_id", "team") and not c.endswith("_n")]
    H = g[["game_id", "season", "home_team", "away_team"]].merge(
        r.rename(columns=lambda c: c if c == "game_id" else "h_" + c), left_on=["game_id", "home_team"], right_on=["game_id", "h_team"])
    X = H.merge(r.rename(columns=lambda c: c if c == "game_id" else "a_" + c), left_on=["game_id", "away_team"],
                right_on=["game_id", "a_team"])
    out = pd.DataFrame({"game_id": X.game_id})
    sdm = lambda v: float(v[X.season.isin(list(SD_SEASONS))].std()) or 1.0
    for n in STATS:
        hn = X[f"h_o_{n}"] - X[f"h_d_{n}"]
        an = X[f"a_o_{n}"] - X[f"a_d_{n}"]
        out[f"m_{n}"] = hn - an                                    # + favours home
        out[f"t_{n}"] = X[f"h_o_{n}"] + X[f"a_d_{n}"] + X[f"a_o_{n}"] + X[f"h_d_{n}"]   # + = more of it in this game
        if n in ("stuff", "three_out", "int_share", "int_worthy", "havoc"):
            out[f"t_{n}"] = -out[f"t_{n}"]                         # natural orientation (more havoc/stuffs) for totals
        out[f"o_{n}_h"], out[f"o_{n}_a"] = X[f"h_o_{n}"], X[f"a_o_{n}"]
        out[f"d_{n}_h"], out[f"d_{n}_a"] = X[f"h_d_{n}"], X[f"a_d_{n}"]
    for c in [c for c in rcols if c.startswith("g_")]:
        out[f"m_{c[2:]}"] = X[f"h_{c}"] - X[f"a_{c}"]
        out[f"t_{c[2:]}"] = X[f"h_{c}"] + X[f"a_{c}"]
    # derived "unsustainable" gaps: late-down conversion over expected beyond what early downs explain; red-zone TD
    # rate beyond overall drive quality; fumble-recovery composite (own recovery minus opponents' recovery vs us)
    z = lambda c: out[c] / sdm(out[c])
    for side in ("h", "a"):
        for k in ("o", "d"):
            out[f"{k}_ldgap_{side}"] = out[f"{k}_ld_cx_{side}"] / sdm(out[f"{k}_ld_cx_{side}"]) - out[f"{k}_ed_sr_{side}"] / sdm(out[f"{k}_ed_sr_{side}"])
            out[f"{k}_rzgap_{side}"] = out[f"{k}_rz_td_{side}"] / sdm(out[f"{k}_rz_td_{side}"]) - out[f"{k}_avail_{side}"] / sdm(out[f"{k}_avail_{side}"])
    for n in ("ldgap", "rzgap"):
        out[f"m_{n}"] = (out[f"o_{n}_h"] - out[f"d_{n}_h"]) - (out[f"o_{n}_a"] - out[f"d_{n}_a"])
        out[f"t_{n}"] = out[f"o_{n}_h"] + out[f"d_{n}_a"] + out[f"o_{n}_a"] + out[f"d_{n}_h"]
    # luck composite (pre-specified): equal-weight z of the five 'unsustainable' margin signals
    out["m_luck5"] = (z("m_ld_cx") + z("m_rz_td") + z("m_fumrec") + z("m_close_ew") + z("m_pyth_ew")) / math.sqrt(5)
    out["t_luck_off"] = (z("t_ld_cx") + z("t_rz_td")) / math.sqrt(2)
    d = g.merge(out, on="game_id", how="left")
    d.to_parquet(SCR / "games_feats.parquet")
    print(d.shape)
    return d


def load_all() -> pd.DataFrame:
    p = SCR / "games_feats.parquet"
    return pd.read_parquet(p) if p.exists() else build()


M_FEATS = ["havoc", "proe", "nproe", "npace", "nohud", "ed_epa", "ed_sr", "ld_conv", "ld_cx", "ld_epa", "rz_td", "ppd",
           "dsr", "avail", "start_fp", "three_out", "stuff", "aly", "second", "open", "fumrec", "int_share", "go_oe",
           "ldgap", "rzgap", "pyth_ew", "pyth_std", "close_ew", "close_std", "winpct_std", "sos", "luck5"]
T_FEATS = ["havoc", "proe", "nproe", "npace", "nohud", "ed_epa", "ed_sr", "ld_conv", "ld_cx", "ld_epa", "rz_td", "ppd",
           "dsr", "avail", "start_fp", "three_out", "stuff", "aly", "second", "open", "int_share", "go_oe", "ldgap",
           "rzgap", "luck_off"]
# pre-specified "luck / unsustainable" hypotheses: (feature, expected sign of the residual slope)
#   home luckier/overperforming -> market overrates home -> margin residual NEGATIVE; offensive luck in the total ->
#   total overpriced -> total residual NEGATIVE
LUCK = {"m_ld_cx": -1, "m_ldgap": -1, "m_rz_td": -1, "m_rzgap": -1, "m_fumrec": -1, "m_int_share": -1,
        "m_close_ew": -1, "m_close_std": -1, "m_pyth_ew": -1, "m_pyth_std": -1, "m_luck5": -1,
        "t_ld_cx": -1, "t_rz_td": -1, "t_ldgap": -1, "t_rzgap": -1, "t_luck_off": -1}


# ================================================================== stability: is the rating informative about the next game?
def stage_stability():
    g = MR.load_games()
    u = team_game_raw().merge(g[["game_id", "gameday", "cutoff", "season"]], on="game_id", suffixes=("", "_g"))
    r = unit_ratings(g)
    res = {}
    for name, (num, den, k, fam) in STATS.items():
        L = MR._league_ref(u, num, den, u.cutoff)
        rate = (u[num] - np.nan_to_num(L) * u[den]) / u[den].where(u[den] > 0)
        x = u[["game_id", "team", "season"]].assign(o_real=rate * GOOD.get(name, 1), w=u[den])
        x = x.merge(r[["game_id", "team", f"o_{name}"]], on=["game_id", "team"])
        x = x[x.season.between(2013, 2025) & x.o_real.notna()]
        if name == "int_worthy":
            x = x[x.season >= 2023]
        res[name] = {"family": fam, "corr_pre_rating_vs_game": float(np.corrcoef(x[f"o_{name}"], x.o_real)[0, 1]) if len(x) > 50 else None,
                     "n": len(x)}
    # record metrics: pre-game value vs the same game's realized analogue
    gl = game_level(g)
    t = pd.concat([g[["game_id", "season", "home_team", "home_score", "away_score"]].rename(columns={"home_team": "team"}).assign(sg=1),
                   g[["game_id", "season", "away_team", "home_score", "away_score"]].rename(columns={"away_team": "team"}).assign(sg=-1)])
    t["m"] = t.sg * (t.home_score - t.away_score)
    t = t[t.m.notna() & t.season.between(2013, 2025)].merge(gl, on=["game_id", "team"])
    t["close_sur"] = np.where(t.m.abs() <= 8, (t.m > 0) + 0.5 * (t.m == 0) - 0.5, np.nan)
    for c in ("g_close_ew", "g_close_std"):
        v = t.dropna(subset=["close_sur"])
        res[c] = {"corr_pre_vs_next_close_game_surplus": float(v[c].corr(v.close_sur)), "n": len(v)}
    for c in ("g_pyth_ew", "g_pyth_std", "g_pd_ew", "g_winpct_std", "g_sos"):
        res[c] = {"corr_pre_vs_next_margin": float(t[c].corr(t.m)), "n": len(t)}
    # partial effect of record 'luck' on the next margin, controlling for the point-differential rating
    for c in ("g_pyth_ew", "g_pyth_std", "g_close_ew", "g_close_std", "g_sos"):
        v = t[["m", "g_pd_ew", c]].dropna()
        Xm = np.c_[np.ones(len(v)), v.g_pd_ew / v.g_pd_ew.std(), v[c] / v[c].std()]
        b = np.linalg.lstsq(Xm, v.m.values, rcond=None)[0]
        e = v.m.values - Xm @ b
        XtXi = np.linalg.inv(Xm.T @ Xm)
        se = np.sqrt(np.diag(XtXi @ (Xm.T * e ** 2) @ Xm @ XtXi))
        res[c]["next_margin_pts_per_sd_given_pd"] = {"b": float(b[2]), "se": float(se[2])}
    save_json("stability", res)
    for k, v in res.items():
        print(k, v)


# ================================================================== (a) production model
GROUPS = {
    "havoc": ["m_havoc"],
    "proe_pace": ["m_proe", "m_nproe", "m_npace", "m_nohud"],
    "early_late_downs": ["m_ed_epa", "m_ed_sr", "m_ld_conv", "m_ld_cx"],
    "drives": ["m_ppd", "m_dsr", "m_avail", "m_start_fp", "m_three_out", "m_rz_td"],
    "ol_rushing": ["m_stuff", "m_aly", "m_second", "m_open"],
    "luck": ["m_fumrec", "m_int_share", "m_close_ew", "m_pyth_ew", "m_ldgap", "m_rzgap"],
    "luck_composite": ["m_luck5"],
    "record_std": ["m_pyth_std", "m_close_std", "m_winpct_std"],
    "sos": ["m_sos"],
    "fourth_down": ["m_go_oe"],
    # follow-ups to the two groups with the best validation score (labelled as such in the report)
    "havoc_split": ["m_havoc_off", "m_havoc_def"],
    "havoc+ol_rushing": ["m_havoc", "m_stuff", "m_aly", "m_second", "m_open"],
}


def stage_model():
    import experiment as X
    from nflpred import model as Mo
    cache = SCR / "prod_df.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        _, df = X.run("baseline")
        df.to_pickle(cache)
    d = load_all()
    d["m_havoc_off"] = d.o_havoc_h - d.o_havoc_a       # havoc allowed by the offense (good-oriented)
    d["m_havoc_def"] = -(d.d_havoc_h - d.d_havoc_a)    # havoc generated by the defense
    cols = sorted({c for v in GROUPS.values() for c in v})
    df = df.merge(d[["game_id"] + cols], on="game_id", how="left")
    for c in cols:
        df[c] = df[c].fillna(0.0)
    sets = {"baseline": [], **GROUPS, "all": sorted({c for k, v in GROUPS.items() if k != "havoc_split" for c in v})}
    res = {}
    for name, extra in sets.items():
        feats = F.FEATURES + extra
        r = {}
        for label, seasons in (("val_2015_19", X.VAL), ("hold_2020_25", X.HOLD)):
            ps, ys = [], []
            for s in seasons:
                tr = Mo.train_rows(df, before_season=s)
                te = df[(df.season == s) & df.home_win.notna()]
                ps.append(Mo.fit_predict(tr, te, feats, "margin"))
                ys.append(te.home_win.values)
            r[label] = round(Mo.score(np.concatenate(ys), np.concatenate(ps))["log_loss"], 5)
        res[name] = {"n_extra": len(extra), **r}
        print(name, res[name])
    b = res["baseline"]
    for k in res:
        res[k]["d_val"] = round(res[k]["val_2015_19"] - b["val_2015_19"], 5)
        res[k]["d_hold"] = round(res[k]["hold_2020_25"] - b["hold_2020_25"], 5)
        res[k]["adopt"] = bool(res[k]["d_val"] <= -0.001 and res[k]["d_hold"] <= -0.001)
    save_json("model", res)


# ================================================================== (b) market tests
def bh(pvals: dict, q=0.10) -> list:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    keep = 0
    for i, (_, p) in enumerate(items, 1):
        if p <= q * i / m:
            keep = i
    return [k for k, _ in items[:keep]]


def p2(t):
    return 2 * pval(abs(t)) if t is not None else None


def tue_moves(sp, d, tt):
    tue = sp[(sp.snapname == "Tue 14:10") & (sp.hours_before <= 7 * 24)]
    gm = tue.groupby("game_id").agg(m_tue=("m_sharp", "first"), m_cons=("m_cons", "first"),
                                    mu_close=("mu_close", "first")).reset_index()
    gm["move"] = gm.mu_close - gm.m_tue.fillna(gm.m_cons)
    gm = gm.merge(d, on="game_id")
    tg = tt[tt.snap == "tue"].groupby("game_id").agg(mu_tue=("mu_tue", "first"), mu_c=("mu_close", "first")).reset_index()
    tg["mv_tue"] = tg.mu_c - tg.mu_tue
    tg = tg.merge(d, on="game_id")
    return gm, tg


def stage_dev():
    d = load_all()
    d = d[d.total.notna() & d.spread_line.notna()]
    d1, d2 = d[d.season.isin(list(DEV1))], d[d.season.isin(list(DEV2))]
    sp, tt = MR.spread_table(False), MR.totals_table(False)
    gm, tg = tue_moves(sp, d, tt)
    coef = {}
    for pre, tgt, mv, feats, mvdf in (("m_", "res_m", "move", M_FEATS, gm), ("t_", "res_t", "mv_tue", T_FEATS, tg)):
        for n in feats:
            c = pre + n
            coef[c] = {"close_2012_19": MR.ols_z(d1[tgt], d1[c]), "close_2020_22": MR.ols_z(d2[tgt], d2[c]),
                       "tue_move_2020_22": MR.ols_z(mvdf[mv], mvdf[c])}
    out = {"n_tests_close_2012_19": len(coef), "feature_coefs": coef}
    pv = {c: p2(v["close_2012_19"]["t"]) for c, v in coef.items() if v["close_2012_19"]["t"] is not None}
    out["bonferroni_close_2012_19"] = [c for c, p in pv.items() if p < 0.05 / len(pv)]
    out["bh10_close_2012_19"] = bh(pv)
    out["n_abs_t_gt_1.96_close_2012_19"] = int(sum(p < 0.05 for p in pv.values()))
    pv2 = {c: p2(v["close_2020_22"]["t"]) for c, v in coef.items() if v["close_2020_22"]["t"] is not None}
    out["n_abs_t_gt_1.96_close_2020_22"] = int(sum(p < 0.05 for p in pv2.values()))
    pvm = {c: p2(v["tue_move_2020_22"]["t"]) for c, v in coef.items() if v["tue_move_2020_22"]["t"] is not None}
    out["bonferroni_tue_move_2020_22"] = [c for c, p in pvm.items() if p < 0.05 / len(pvm)]
    # pre-specified luck family: one-sided in the hypothesised direction, Bonferroni over the family
    fam = {}
    for c, sgn in LUCK.items():
        v = coef[c]
        r = {}
        for per in ("close_2012_19", "close_2020_22", "tue_move_2020_22"):
            t = v[per]["t"]
            r[per] = {"b": v[per]["b"], "se": v[per]["se"], "p_one_sided": pval(sgn * t) if t is not None else None}
        fam[c] = r
    out["luck_family"] = {"n": len(LUCK), "bar": 0.05 / len(LUCK), "tests": fam}
    # luck rules: fade the luckier side at the Tuesday line (spread CLV) / at the close (ATS)
    out["luck_rules_dev"] = luck_rule_grid(d, d1, d2, sp, tt)
    save_json("dev", out)
    rows = sorted(coef.items(), key=lambda kv: -abs(kv[1]["close_2012_19"]["t"] or 0))
    for c, v in rows[:20]:
        f = lambda x: "%5.2f±%4.2f" % (x["b"], x["se"]) if x["b"] is not None else "   na    "
        print("%-14s d1 %s  d2 %s  tue %s" % (c, f(v["close_2012_19"]), f(v["close_2020_22"]), f(v["tue_move_2020_22"])))
    print("bonf", out["bonferroni_close_2012_19"], "bh", out["bh10_close_2012_19"], "n|t|>1.96",
          out["n_abs_t_gt_1.96_close_2012_19"], "/", len(pv), " d2:", out["n_abs_t_gt_1.96_close_2020_22"],
          " tue-move bonf:", out["bonferroni_tue_move_2020_22"])
    for c, r in fam.items():
        print("%-14s " % c + "  ".join("%s p=%.3f b=%+.2f" % (k[:8], vv["p_one_sided"] or 1, vv["b"] or 0) for k, vv in r.items()))
    print(json.dumps(MR.jsonable(out["luck_rules_dev"]), indent=0)[:4000])


def closing_ats(games: pd.DataFrame, side: pd.Series, n_rules=1) -> dict:
    x = games.assign(side=side)
    x = x[x.side.isin(["home", "away"]) & x.result.notna() & x.spread_line.notna()]
    price = np.where(x.side == "home", x.home_spread_odds, x.away_spread_odds).astype(float)
    price = np.where(np.isfinite(price), price, -110)
    decm = np.where(price > 0, 1 + price / 100, 1 + 100 / -price)
    adj = np.where(x.side == "home", x.result - x.spread_line, x.spread_line - x.result)
    pnl = np.where(adj > 0, decm - 1, np.where(adj < 0, -1.0, 0.0))
    if len(pnl) < 3:
        return {"bets": int(len(pnl))}
    t = pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl)))
    dec_ = adj != 0
    return {"bets": int(len(pnl)), "cover_ex_push": float((adj[dec_] > 0).mean()), "roi": float(pnl.mean()),
            "roi_t": float(t), "p_one_sided": pval(t), "pass": bool(pnl.mean() > 0 and pval(t) < 0.05 / n_rules),
            "by_season": {int(s): float(v) for s, v in pd.Series(pnl, index=x.season.values).groupby(level=0).mean().items()}}


def zs(d_all, c):
    return float(d_all.loc[d_all.season.isin(list(SD_SEASONS)), c].std())


def spread_rule(sp, d_all, feat, thr, fade=True, snap="Tue 14:10"):
    x = sp[(sp.snapname == snap) & (sp.hours_before <= 7 * 24)].merge(d_all[["game_id", feat]], on="game_id")
    zz = x[feat] / zs(d_all, feat) * np.where(x.side == "home", 1, -1) * (-1 if fade else 1)
    return MR.one_per_game(x[zz >= thr].assign(clv_now=x.sp_ev_sharp))


def total_rule(tt, d_all, feat, thr, side="under", snap="tue"):
    x = tt[(tt.snap == snap) & (tt.side == side)].merge(d_all[["game_id", feat]], on="game_id")
    zz = x[feat] / zs(d_all, feat) * (1 if side == "under" else -1)
    return MR.one_per_game(x[zz >= thr].assign(clv_now=x.ev_ref_now))


def luck_rule_grid(d, d1, d2, sp, tt):
    res = {}
    keep = ("bets", "clv", "clv_t", "roi", "cover_ex_push", "roi_t")
    for feat in ("m_luck5", "m_pyth_std", "m_pyth_ew", "m_close_std", "m_ld_cx", "m_ldgap", "m_rz_td", "m_fumrec", "m_int_share"):
        for thr in (1.0, 1.5, 2.0):
            r = {}
            for nm, gg in (("ats_close_2012_19", d1), ("ats_close_2020_22", d2)):
                zz = gg[feat] / zs(d, feat)
                sel = gg[zz.abs() >= thr]
                side = pd.Series(np.where(sel[feat] > 0, "away", "home"), index=sel.index)   # fade the luckier side
                r[nm] = {k: v for k, v in closing_ats(sel, side).items() if k in keep}
            r["tue_clv_2020_22"] = {k: v for k, v in MR.summ(spread_rule(sp, d, feat, thr)).items() if k in keep}
            res[f"{feat}|z|>={thr}"] = r
    for feat in ("t_luck_off", "t_rz_td", "t_ld_cx", "t_npace", "t_nproe", "t_havoc"):
        for thr in (1.0, 1.5):
            r = {}
            for nm, gg in (("ou_close_2012_19", d1), ("ou_close_2020_22", d2)):
                zz = gg[feat] / zs(d, feat)
                for side, cond in (("under", zz >= thr), ("over", zz <= -thr)):
                    sel = gg[cond]
                    r[f"{nm}_{side}"] = {k: v for k, v in MR.closing_ou(sel, pd.Series(side, index=sel.index)).items() if k in
                                         ("bets", "roi", "roi_t", "win_rate_ex_push")}
            for side in ("under", "over"):
                r[f"tue_clv_2020_22_{side}"] = {k: v for k, v in MR.summ(total_rule(tt, d, feat, thr, side)).items() if k in keep}
            res[f"{feat}|z|>={thr}"] = r
    return res


# ================================================================== (c) frozen rules (chosen on dev; see stage_dev output)
FROZEN_RULES = {
    "R1_ats_close_fade_fumble_luck": {
        "kind": "ats_close_fade", "market": "spread", "feature": "m_fumrec", "threshold": 1.5,
        "snapshot": "closing line (nflverse spread_line) at its actual closing juice",
        "rule": "bet AGAINST the team with the better walk-forward fumble-recovery luck (own-fumble recovery rate "
                "minus opponents' recovery rate vs it, shrunk, home-minus-away) when |z| >= 1.5 (2013-19 SD scale)",
        "metric": "ATS ROI at closing juice; one-sided t on per-bet profit",
        "dev": "2012-19: 312 bets, ROI +2.8% (t 0.52); 2020-22: 141 bets, ROI +9.7% (t 1.21); pre-specified luck "
               "hypothesis; rating has ~zero persistence (pre-game vs game corr 0.02)"},
    "R2_total_close_slow_pace_under": {
        "kind": "ou_close_under", "market": "total", "feature": "t_npace", "threshold": 1.0,
        "snapshot": "closing line (nflverse total_line) at its actual closing juice",
        "rule": "bet UNDER when the game's combined neutral-situation seconds-per-snap rating (both offenses and the "
                "pace both defenses allow) is >= +1.0 SD (slow game)",
        "metric": "O/U ROI at closing juice; one-sided t on per-bet profit",
        "dev": "2012-19: 317 bets, ROI +10.1% (t 1.86); 2020-22: 94 bets, ROI +7.3% (t 0.74); Tue CLV 2020-22 -0.9% "
               "(vs -1.2% for any best-price under)"},
    "R3_spread_tue_back_early_down_epa": {
        "kind": "spread_tue_back", "market": "spread", "feature": "m_ed_epa", "threshold": 2.0,
        "snapshot": "Tue 14:10 UTC (<= 7 days before kickoff), best allowed-book price, one bet per game",
        "rule": "back the side with the better net early-down (1st/2nd down) EPA rating when |z| >= 2.0: the "
                "Tuesday line under-reacts to it (+0.37 pt/SD move to the close, t 6.3, 2020-22)",
        "metric": "price-based CLV vs edge_lab.closing_fair (mu_close_sharp, else all books)",
        "dev": "2020-22: 44 bets, CLV +0.3% (t 0.17); thresholds 1.0/1.5 were -2.5%/-1.8%"},
}


def rule_bets(name, d_all, sp=None, tt=None, games=None):
    r = FROZEN_RULES[name]
    if r["kind"] == "spread_tue_fade":
        return spread_rule(sp, d_all, r["feature"], r["threshold"], fade=True)
    if r["kind"] == "spread_tue_back":
        return spread_rule(sp, d_all, r["feature"], r["threshold"], fade=False)
    if r["kind"] == "total_tue":
        return total_rule(tt, d_all, r["feature"], r["threshold"], r["side"])
    if r["kind"] == "ats_close_fade":
        zz = games[r["feature"]] / zs(d_all, r["feature"])
        sel = games[zz.abs() >= r["threshold"]]
        return sel, pd.Series(np.where(sel[r["feature"]] > 0, "away", "home"), index=sel.index)
    if r["kind"] == "ou_close_under":
        sel = games[games[r["feature"]] / zs(d_all, r["feature"]) >= r["threshold"]]
        return sel, pd.Series("under", index=sel.index)
    raise KeyError(name)


def stage_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; frozen rules are never overwritten")
    if not FROZEN_RULES:
        raise SystemExit("no rules defined")
    import datetime as dt
    FROZEN.write_text(json.dumps({"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                  "n_rules": len(FROZEN_RULES), "holdout": list(HOLD),
                                  "pass_bar": f"primary metric > 0 with one-sided p < 0.05/{len(FROZEN_RULES)}",
                                  "rules": FROZEN_RULES}, indent=1))
    print("frozen", list(FROZEN_RULES))


def stage_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout is locked: run freeze first and set EDGE_HOLDOUT")
    if JSON.exists() and "holdout" in json.loads(JSON.read_text()):
        raise SystemExit("holdout already run once")
    fr = json.loads(FROZEN.read_text())
    n = fr["n_rules"]
    d = load_all()
    d = d[d.total.notna() & d.spread_line.notna()]
    ho = d[d.season.isin(HOLD)]
    sp, tt = MR.spread_table(True), MR.totals_table(True)
    res = {"rules": {}}
    for name, r in fr["rules"].items():
        if r["kind"] == "ats_close_fade":
            sel, side = rule_bets(name, d, games=ho)
            res["rules"][name] = closing_ats(sel, side, n)
        elif r["kind"] == "ou_close_under":
            sel, side = rule_bets(name, d, games=ho)
            res["rules"][name] = MR.closing_ou(sel, side, n)
            b = total_rule(tt, d, r["feature"], r["threshold"], "under")
            res["rules"][name]["secondary_tue_clv"] = MR.summ(b, n)
        else:
            res["rules"][name] = MR.summ(rule_bets(name, d, sp=sp, tt=tt), n)
            if r["kind"].startswith("spread"):
                zz = ho[r["feature"]] / zs(d, r["feature"])
                sel = ho[zz.abs() >= r["threshold"]]
                sgn = -1 if "fade" in r["kind"] else 1
                side = pd.Series(np.where(sgn * sel[r["feature"]] > 0, "home", "away"), index=sel.index)
                res["rules"][name]["secondary_ats_close"] = closing_ats(sel, side, n)
    # descriptive (no verdicts): every coefficient on the holdout seasons
    gm, tg = tue_moves(sp, d, tt)
    coefs = {}
    for pre, tgt, mv, feats, mvdf in (("m_", "res_m", "move", M_FEATS, gm), ("t_", "res_t", "mv_tue", T_FEATS, tg)):
        for nn in feats:
            c = pre + nn
            coefs[c] = {"close": MR.ols_z(ho[tgt], ho[c]), "tue_move": MR.ols_z(mvdf[mv], mvdf[c])}
    coefs["m_int_worthy_ftn"] = {"close": MR.ols_z(ho.res_m, ho.m_int_worthy), "tue_move": MR.ols_z(gm.move, gm.m_int_worthy)}
    res["descriptive"] = {"coefs": coefs,
                          "n_abs_t_gt_1.96_close": int(sum(abs(v["close"]["t"] or 0) > 1.96 for v in coefs.values())),
                          "n": len(coefs),
                          "luck_family_one_sided_p": {c: pval(s * (coefs[c]["close"]["t"] or 0)) for c, s in LUCK.items()}}
    save_json("holdout", res)
    print(json.dumps(MR.jsonable(res["rules"]), indent=1))
    print("n|t|>1.96", res["descriptive"]["n_abs_t_gt_1.96_close"], "/", len(coefs))
    for c, p in res["descriptive"]["luck_family_one_sided_p"].items():
        print(c, round(p, 3), coefs[c]["close"])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "stability": stage_stability, "model": stage_model, "dev": stage_dev, "freeze": stage_freeze,
     "holdout": stage_holdout}[stage]()
