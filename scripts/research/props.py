"""NFL PLAYER-PROP PROJECTIONS: walk-forward means + predictive distributions, validated without market lines.

Stages (each caches into the scratch dir; the final one writes output/research/props.json):
  python scripts/research/props.py build      # player-game + team-game tables from pbp, prior-only features
  python scripts/research/props.py fit        # walk-forward fits for every season 2014..2025 (train = seasons < S)
  python scripts/research/props.py evaluate   # dev (2019-2022) and holdout (2023-2025) tables -> props.json
  output/research/props.md is written by hand from props.json.

Markets (outcomes are official-style box-score stats derived from nflverse play-by-play, REG+POST 2012-2025):
  pass_yds, pass_att, pass_cmp (starting QB), rush_yds (RB), rec_yds, receptions (WR/TE/RB).
  pbp-derived stats are checked against nflverse official weekly stats (2020-2024) in `build`
  (official weekly files for 2019/2025 are not published, so pbp is the only consistent source).

Settlement-matched universe: a player-game is graded when the player actually took >=1 offensive snap
(prop rules: action if the player participates). Eligibility for "a book would post this prop" uses only
prior data: QB = listed starting QB (games.parquet *_qb_id, known pre-kickoff); RB rush = last-8 played-games
mean rush yds >= 25; WR/TE/RB receiving = last-8 mean rec yds >= 20. All need >= 3 prior played games.

Decomposition (all inputs from games with an earlier date, except the pre-game lines and the active list):
  team opportunity  E[team pass att], E[team targets], E[team carries]  = ridge on implied team total,
                    expected margin, game total, team/opp decayed volume + pass-rate-over-expected, wind/temp/roof.
  player share      decayed share of team targets/carries/attempts in games he played (halflife 6 games),
                    renormalised by the decayed shares of the teammates who are ACTIVE this game ("late" model:
                    assumes the bet is placed after inactives, ~T-90 min) or not listed Out/Doubtful on the
                    final injury report ("early" model).
  efficiency        decayed yards/target, yards/carry, yards/att, catch rate, comp% shrunk to position means
                    (pseudo-counts K_*), times opponent multipliers (decayed, shrunk, position-specific for targets).
  structural mean   opportunity x share x efficiency x opponent.
  GBM mean          LightGBM (small, fixed params) on the structural pieces + raw features; walk-forward by season.
  Distribution      conditional empirical: for a new mean mu, take the 400 out-of-sample (mu_j, y_j) pairs from the
                    previous 4 seasons nearest in log(mu); samples = y_j * mu / mu_j (yards) or y_j (counts).
                    The SAME recipe is applied to the baselines, so CRPS differences isolate mean quality.
Baselines: season-to-date average (previous-season average in week 1), last-4 played-games average.

Weather caveat: games.parquet wind/temp are RECORDED conditions (near-perfect forecast); missing outdoor wind
(esp. 2022) is filled with the outdoor median plus a missing flag. Lines (spread/total) are nflverse CLOSING
lines - for an early-week bet the opener would be used instead (small difference).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "research"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/props")
SEASONS = list(range(2012, 2026))
DEV, HOLD = (2019, 2020, 2021, 2022), (2023, 2024, 2025)
HL_P, HL_T = 6.0, 8.0  # decay halflives (games) for player and team series
K = dict(ypt=60, ypc=100, ypa=200, catch=60, comp=200, opp=250)
MARKETS = {
    # market: (stat column, position filter, kind)
    "pass_yds": ("pass_yds", "QB", "yards"),
    "pass_att": ("pass_att", "QB", "count"),
    "pass_cmp": ("cmp", "QB", "count"),
    "rush_yds": ("rush_yds", "RB", "yards"),
    "rec_yds": ("rec_yds", "REC", "yards"),
    "receptions": ("rec", "REC", "count"),
}


# ----------------------------------------------------------------------------------------------- build
def _decayed_prior(df: pd.DataFrame, key: str, cols: list[str], hl: float, prefix: str) -> pd.DataFrame:
    """Prior-only exponentially decayed SUMS of cols within key (rows must be time-sorted). Row i gets
    sum_{j<i} d^(i-1-j) x_j, plus the decayed game count. Never includes the current row."""
    d = 0.5 ** (1.0 / hl)
    X = df[cols].to_numpy(float)
    out = np.zeros((len(df), len(cols) + 1))
    for _, idx in df.groupby(key, sort=False).indices.items():
        s = np.zeros(len(cols) + 1)
        for i in idx:  # idx is in row order
            out[i] = s
            s = s * d
            s[:-1] += np.nan_to_num(X[i])
            s[-1] += 1.0
    res = pd.DataFrame(out, columns=[f"{prefix}{c}" for c in cols] + [f"{prefix}n"], index=df.index)
    return res


def load_pbp() -> pd.DataFrame:
    cols = ["game_id", "season", "season_type", "week", "game_date", "posteam", "defteam", "play_type",
            "pass_attempt", "complete_pass", "sack", "rush_attempt", "two_point_attempt", "passing_yards",
            "receiving_yards", "rushing_yards", "passer_player_id", "receiver_player_id", "rusher_player_id",
            "air_yards", "pass_oe", "qb_kneel", "qb_spike"]
    return pd.concat([pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=cols) for s in SEASONS], ignore_index=True)


def player_game_stats(p: pd.DataFrame) -> pd.DataFrame:
    p = p[p.play_type.isin(["pass", "run", "qb_kneel", "qb_spike"]) & (p.two_point_attempt.fillna(0) != 1)]
    att = (p.pass_attempt == 1) & (p.sack.fillna(0) == 0)
    pa = p[att & p.passer_player_id.notna()].assign(
        pass_att=1, cmp=lambda x: x.complete_pass.fillna(0), pass_yds=lambda x: x.passing_yards.fillna(0))
    pa = pa.groupby(["game_id", "posteam", "passer_player_id"])[["pass_att", "cmp", "pass_yds"]].sum()
    pa.index.names = ["game_id", "team", "player_id"]
    sk = p[(p.sack.fillna(0) == 1) & p.passer_player_id.notna()].groupby(
        ["game_id", "posteam", "passer_player_id"]).size().rename("sacks")
    sk.index.names = ["game_id", "team", "player_id"]
    ru = p[(p.rush_attempt == 1) & p.rusher_player_id.notna()].assign(
        carries=1, rush_yds=lambda x: x.rushing_yards.fillna(0))
    ru = ru.groupby(["game_id", "posteam", "rusher_player_id"])[["carries", "rush_yds"]].sum()
    ru.index.names = ["game_id", "team", "player_id"]
    re = p[att & p.receiver_player_id.notna()].assign(
        targets=1, rec=lambda x: x.complete_pass.fillna(0), rec_yds=lambda x: x.receiving_yards.fillna(0),
        tgt_air=lambda x: x.air_yards.fillna(0))
    re = re.groupby(["game_id", "posteam", "receiver_player_id"])[["targets", "rec", "rec_yds", "tgt_air"]].sum()
    re.index.names = ["game_id", "team", "player_id"]
    out = pd.concat([pa, sk, ru, re], axis=1).fillna(0).reset_index()
    return out


def team_game_stats(p: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    q = p[p.play_type.isin(["pass", "run"]) & (p.two_point_attempt.fillna(0) != 1) & p.posteam.notna()]
    t = q.groupby(["game_id", "posteam"]).agg(
        defteam=("defteam", "first"), plays=("play_type", "size"), proe=("pass_oe", "mean"),
        dropbacks=("pass_attempt", "sum")).reset_index().rename(columns={"posteam": "team"})
    box = pg.groupby(["game_id", "team"])[["pass_att", "cmp", "pass_yds", "carries", "rush_yds", "targets",
                                           "rec", "rec_yds"]].sum().reset_index()
    return t.merge(box, on=["game_id", "team"], how="left")


def official_check(pg: pd.DataFrame) -> dict:
    res = {}
    for s in range(2020, 2025):
        f = SCR / f"spw_{s}.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f)
        o = o.rename(columns={"attempts": "o_att", "completions": "o_cmp", "passing_yards": "o_py",
                              "rushing_yards": "o_ry", "carries": "o_car", "receptions": "o_rec",
                              "receiving_yards": "o_recy", "targets": "o_tgt"})
        m = pg[pg.season == s].merge(o, on=["player_id", "season", "week"], how="inner")
        r = {}
        for a, b in [("pass_att", "o_att"), ("cmp", "o_cmp"), ("pass_yds", "o_py"), ("carries", "o_car"),
                     ("rush_yds", "o_ry"), ("rec", "o_rec"), ("rec_yds", "o_recy"), ("targets", "o_tgt")]:
            mm = m[(m[a] > 0) | (m[b] > 0)]
            r[a] = {"rows": int(len(mm)), "exact": round(float((mm[a] == mm[b]).mean()), 4),
                    "mae": round(float((mm[a] - mm[b]).abs().mean()), 3)}
        res[s] = r
    return res


def build():
    g = pd.read_parquet(RAW / "games.parquet")
    g = g[g.season.isin(SEASONS) & g.result.notna()].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    p = load_pbp()
    pg = player_game_stats(p)
    tg = team_game_stats(p, pg)

    # --- who played: snap counts (pfr id -> gsis) union pbp involvement
    pl = pd.read_parquet(RAW / "players.parquet")
    pfr2g = pl.dropna(subset=["pfr_id", "gsis_id"]).drop_duplicates("pfr_id").set_index("pfr_id").gsis_id
    sn = pd.concat([pd.read_parquet(RAW / f"snap_counts_{s}.parquet") for s in SEASONS], ignore_index=True)
    sn["player_id"] = sn.pfr_player_id.map(pfr2g)
    sn = sn.dropna(subset=["player_id"])
    sn = sn[sn.offense_snaps > 0][["game_id", "team", "player_id", "offense_snaps", "offense_pct", "position"]]
    sn = sn.rename(columns={"position": "snap_pos"}).drop_duplicates(["game_id", "player_id"])
    team_fix = {"LA": "LA", "LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}
    sn["team"] = sn.team.replace(team_fix)
    tg["team"] = tg.team.replace(team_fix)
    pg["team"] = pg.team.replace(team_fix)
    tg["defteam"] = tg.defteam.replace(team_fix)
    for c in ["home_team", "away_team"]:
        g[c] = g[c].replace(team_fix)
    a = pg.merge(sn, on=["game_id", "team", "player_id"], how="outer")
    a[["pass_att", "cmp", "pass_yds", "sacks", "carries", "rush_yds", "targets", "rec", "rec_yds", "tgt_air"]] = a[
        ["pass_att", "cmp", "pass_yds", "sacks", "carries", "rush_yds", "targets", "rec", "rec_yds",
         "tgt_air"]].fillna(0)
    a["in_snaps"] = a.offense_snaps.notna()
    pos = pl.drop_duplicates("gsis_id").set_index("gsis_id").position
    a["position"] = a.player_id.map(pos).fillna(a.snap_pos)
    a.loc[a.position.isin(["FB", "HB"]), "position"] = "RB"
    a = a[a.position.isin(["QB", "RB", "WR", "TE"])].copy()
    meta = g[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "spread_line",
              "total_line", "roof", "wind", "temp", "home_qb_id", "away_qb_id"]]
    a = a.merge(meta, on="game_id", how="inner")
    tg = tg.merge(meta, on="game_id", how="inner")
    a = a.sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)
    tg = tg.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)

    # --- game context (pre-game lines are legitimately known)
    for d in (a, tg):
        d["home"] = (d.team == d.home_team).astype(int)
        d["exp_margin"] = np.where(d.home == 1, d.spread_line, -d.spread_line)
        d["itt"] = d.total_line / 2 + d.exp_margin / 2
        d["dome"] = d.roof.isin(["dome", "closed"]).astype(int)
        d["wind_missing"] = ((d.dome == 0) & d.wind.isna()).astype(int)
        d["wind_f"] = np.where(d.dome == 1, 0.0, d.wind)
        d["temp_f"] = np.where(d.dome == 1, 70.0, d.temp)
        d["post"] = (d.game_type != "REG").astype(int)
    med_w = tg.loc[tg.dome == 0, "wind_f"].median()
    med_t = tg.loc[tg.dome == 0, "temp_f"].median()
    for d in (a, tg):
        d["wind_f"] = d.wind_f.fillna(med_w)
        d["temp_f"] = d.temp_f.fillna(med_t)
    a["opp"] = np.where(a.home == 1, a.away_team, a.home_team)
    tg["opp"] = np.where(tg.home == 1, tg.away_team, tg.home_team)
    a["is_starter_qb"] = ((a.player_id == np.where(a.home == 1, a.home_qb_id, a.away_qb_id))).astype(int)

    # --- team-level decayed priors (offense) and opponent (defense) priors
    tcols = ["plays", "pass_att", "targets", "carries", "pass_yds", "rush_yds", "cmp", "rec_yds", "dropbacks"]
    tg["proe_x_plays"] = tg.proe.fillna(0) * tg.plays
    tg = pd.concat([tg, _decayed_prior(tg, "team", tcols + ["proe_x_plays"], HL_T, "to_")], axis=1)
    tg = pd.concat([tg, _decayed_prior(tg, "opp", tcols, HL_T, "do_")], axis=1)  # what opp's defense allowed
    n = tg.to_n.replace(0, np.nan)
    for c in ["plays", "pass_att", "targets", "carries"]:
        tg[f"t_{c}_pg"] = tg[f"to_{c}"] / n
    tg["t_proe"] = tg.to_proe_x_plays / tg.to_plays.replace(0, np.nan)
    nd = tg.do_n.replace(0, np.nan)
    for c in ["plays", "pass_att", "carries"]:
        tg[f"o_{c}_pg"] = tg[f"do_{c}"] / nd
    # league means (prior seasons, roughly stable) used for shrinkage/multipliers
    lg_ypa = tg.pass_yds.sum() / tg.pass_att.sum()
    lg_ypc = tg.rush_yds.sum() / tg.carries.sum()
    lg_cmp = tg.cmp.sum() / tg.pass_att.sum()
    tg["opp_ypa_mult"] = ((tg.do_pass_yds + K["opp"] * lg_ypa) / (tg.do_pass_att + K["opp"])) / lg_ypa
    tg["opp_ypc_mult"] = ((tg.do_rush_yds + K["opp"] * lg_ypc) / (tg.do_carries + K["opp"])) / lg_ypc
    tg["opp_cmp_mult"] = ((tg.do_cmp + K["opp"] * lg_cmp) / (tg.do_pass_att + K["opp"])) / lg_cmp

    # position-specific receiving allowed (opp defense vs WR / TE / RB)
    a["pgrp"] = np.where(a.position.isin(["WR", "TE", "RB"]), a.position, "OTH")
    pr = a[a.pgrp != "OTH"].groupby(["game_id", "opp", "pgrp", "gameday"])[["targets", "rec_yds", "rec"]].sum()
    pr = pr.reset_index().sort_values(["gameday", "game_id"]).reset_index(drop=True)
    pr["key"] = pr.opp + "_" + pr.pgrp
    pr = pd.concat([pr, _decayed_prior(pr, "key", ["targets", "rec_yds", "rec"], HL_T, "dp_")], axis=1)
    lg = pr.groupby("pgrp")[["targets", "rec_yds", "rec"]].sum()
    pr["lg_ypt"] = pr.pgrp.map(lg.rec_yds / lg.targets)
    pr["lg_cr"] = pr.pgrp.map(lg.rec / lg.targets)
    pr["opp_ypt_mult"] = ((pr.dp_rec_yds + K["opp"] * pr.lg_ypt) / (pr.dp_targets + K["opp"])) / pr.lg_ypt
    pr["opp_cr_mult"] = ((pr.dp_rec + K["opp"] * pr.lg_cr) / (pr.dp_targets + K["opp"])) / pr.lg_cr
    a = a.merge(pr[["game_id", "opp", "pgrp", "opp_ypt_mult", "opp_cr_mult"]], on=["game_id", "opp", "pgrp"],
                how="left")
    a = a.merge(tg[["game_id", "team", "opp_ypa_mult", "opp_ypc_mult", "opp_cmp_mult"]], on=["game_id", "team"],
                how="left")

    # --- player-level priors over games he PLAYED (snaps>0 or any pbp involvement)
    a["played"] = a.in_snaps | (a[["pass_att", "carries", "targets", "sacks"]].sum(axis=1) > 0)
    a = a[a.played].copy()
    a = a.merge(tg[["game_id", "team", "pass_att", "targets", "carries"]].rename(
        columns={"pass_att": "tm_pass_att", "targets": "tm_targets", "carries": "tm_carries"}),
        on=["game_id", "team"], how="left")
    a["snap_pct"] = a.offense_pct.fillna(np.nan)
    a["snap_known"] = a.snap_pct.notna().astype(float)
    a["snap_pct0"] = a.snap_pct.fillna(0)
    a = a.sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)
    pcols = ["pass_att", "cmp", "pass_yds", "sacks", "carries", "rush_yds", "targets", "rec", "rec_yds", "tgt_air",
             "tm_pass_att", "tm_targets", "tm_carries", "snap_pct0", "snap_known"]
    a = pd.concat([a, _decayed_prior(a, "player_id", pcols, HL_P, "p_")], axis=1)
    a["n_prior"] = a.groupby("player_id").cumcount()
    a["days_since"] = a.groupby("player_id").gameday.diff().dt.days
    # shares (decayed, in games played), efficiency shrunk to position means
    a["tgt_share"] = a.p_targets / a.p_tm_targets.replace(0, np.nan)
    a["car_share"] = a.p_carries / a.p_tm_carries.replace(0, np.nan)
    a["att_share"] = a.p_pass_att / a.p_tm_pass_att.replace(0, np.nan)
    a["snap_ew"] = a.p_snap_pct0 / a.p_snap_known.replace(0, np.nan)
    a["adot"] = a.p_tgt_air / a.p_targets.replace(0, np.nan)
    posm = a.groupby("pgrp")[["rec_yds", "targets", "rec", "rush_yds", "carries"]].sum()
    a["pos_ypt"] = a.pgrp.map(posm.rec_yds / posm.targets)
    a["pos_cr"] = a.pgrp.map(posm.rec / posm.targets)
    a["pos_ypc"] = a.pgrp.map(posm.rush_yds / posm.carries)
    qbm = a[a.position == "QB"][["pass_yds", "pass_att", "cmp"]].sum()
    a["ypt_s"] = (a.p_rec_yds + K["ypt"] * a.pos_ypt) / (a.p_targets + K["ypt"])
    a["cr_s"] = (a.p_rec + K["catch"] * a.pos_cr) / (a.p_targets + K["catch"])
    a["ypc_s"] = (a.p_rush_yds + K["ypc"] * a.pos_ypc) / (a.p_carries + K["ypc"])
    a["ypa_s"] = (a.p_pass_yds + K["ypa"] * qbm.pass_yds / qbm.pass_att) / (a.p_pass_att + K["ypa"])
    a["comp_s"] = (a.p_cmp + K["comp"] * qbm.cmp / qbm.pass_att) / (a.p_pass_att + K["comp"])
    a["sack_rate"] = a.p_sacks / (a.p_pass_att + a.p_sacks).replace(0, np.nan)
    for c in ["tgt_share", "car_share", "att_share"]:
        a[c] = a[c].fillna(0)

    # --- baselines: season-to-date average (prev-season avg in week 1) and last-4 average (played games)
    for stat in ["pass_yds", "pass_att", "cmp", "rush_yds", "rec_yds", "rec"]:
        gp = a.groupby("player_id")[stat]
        a[f"l4_{stat}"] = gp.transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
        a[f"l8_{stat}"] = gp.transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
        gs = a.groupby(["player_id", "season"])[stat]
        std = gs.transform(lambda x: x.shift(1).expanding().mean())
        prev = a.groupby(["player_id", "season"])[stat].mean().rename("pv").reset_index()
        prev["season"] += 1
        a = a.merge(prev.rename(columns={"pv": f"pv_{stat}"}), on=["player_id", "season"], how="left")
        a[f"std_{stat}"] = std.values if len(std) == len(a) else std
        a[f"std_{stat}"] = a[f"std_{stat}"].fillna(a[f"pv_{stat}"]).fillna(a[f"l4_{stat}"])
        a = a.drop(columns=[f"pv_{stat}"])

    # --- teammate availability: renormalise shares by the shares of teammates who play (late) or who are not
    #     listed Out/Doubtful (early). Shares are those as of this game (prior-only).
    for sh in ["tgt_share", "car_share"]:
        a[f"act_sum_{sh}"] = a.groupby(["game_id", "team"])[sh].transform("sum")
    inj = pd.concat([pd.read_parquet(RAW / f"injuries_{s}.parquet") for s in SEASONS], ignore_index=True)
    inj["team"] = inj.team.replace(team_fix)
    out = inj[inj.report_status.isin(["Out", "Doubtful"])][["season", "week", "team", "gsis_id"]]
    out = out.drop_duplicates().rename(columns={"gsis_id": "player_id"})
    q = inj[inj.report_status == "Questionable"][["season", "week", "team", "gsis_id"]].drop_duplicates()
    # rotation entering each team-game = players who played in any of the team's previous 3 games
    tgames = a[["game_id", "team", "gameday", "season", "week"]].drop_duplicates().sort_values("gameday")
    tgames["tg_idx"] = tgames.groupby("team").cumcount()
    a = a.merge(tgames[["game_id", "team", "tg_idx"]], on=["game_id", "team"])
    last = a[["player_id", "team", "tg_idx", "tgt_share", "car_share", "gameday"]].copy()
    # latest share of each player as of each later team-game in the window (use shares they had after that game)
    # approximate with the share entering their most recent game + that game (decayed): recompute post-game share
    a["tgt_share_post"] = (a.p_targets * 0.5 ** (1 / HL_P) + a.targets) / (
        a.p_tm_targets * 0.5 ** (1 / HL_P) + a.tm_targets).replace(0, np.nan)
    a["car_share_post"] = (a.p_carries * 0.5 ** (1 / HL_P) + a.carries) / (
        a.p_tm_carries * 0.5 ** (1 / HL_P) + a.tm_carries).replace(0, np.nan)
    rows = []
    post = a[["player_id", "team", "tg_idx", "tgt_share_post", "car_share_post"]].fillna(0)
    for k in (1, 2, 3):
        r = post.copy()
        r["tg_idx"] = r.tg_idx + k
        r["lag"] = k
        rows.append(r)
    rot = pd.concat(rows).sort_values("lag").drop_duplicates(["player_id", "team", "tg_idx"])
    rot = rot.merge(tgames[["game_id", "team", "tg_idx", "season", "week"]], on=["team", "tg_idx"])
    rot = rot.merge(out.assign(listed_out=1), on=["season", "week", "team", "player_id"], how="left")
    rot["listed_out"] = rot.listed_out.fillna(0)
    early = rot[rot.listed_out == 0].groupby(["game_id", "team"])[["tgt_share_post", "car_share_post"]].sum()
    early.columns = ["early_sum_tgt", "early_sum_car"]
    vac_inj = rot[rot.listed_out == 1].groupby(["game_id", "team"])[["tgt_share_post", "car_share_post"]].sum()
    vac_inj.columns = ["vac_inj_tgt", "vac_inj_car"]
    # late vacated = rotation players who did NOT play this game
    played_set = a[["game_id", "player_id"]].assign(pl=1)
    rot = rot.merge(played_set, on=["game_id", "player_id"], how="left")
    vac_late = rot[rot.pl.isna()].groupby(["game_id", "team"])[["tgt_share_post", "car_share_post"]].sum()
    vac_late.columns = ["vac_late_tgt", "vac_late_car"]
    a = a.merge(early.reset_index(), on=["game_id", "team"], how="left")
    a = a.merge(vac_inj.reset_index(), on=["game_id", "team"], how="left")
    a = a.merge(vac_late.reset_index(), on=["game_id", "team"], how="left")
    a = a.merge(q.rename(columns={"gsis_id": "player_id"}).assign(questionable=1),
                on=["season", "week", "team", "player_id"], how="left")
    for c in ["early_sum_tgt", "early_sum_car", "vac_inj_tgt", "vac_inj_car", "vac_late_tgt", "vac_late_car",
              "questionable"]:
        a[c] = a[c].fillna(0)
    # early renormaliser: own share is in the rotation sum if he played recently; else add it
    a["early_sum_tgt"] = np.maximum(a.early_sum_tgt, a.tgt_share)
    a["early_sum_car"] = np.maximum(a.early_sum_car, a.car_share)

    tg.to_parquet(SCR / "team_games.parquet")
    a.to_parquet(SCR / "player_games.parquet")
    chk = official_check(a)
    json.dump(chk, open(SCR / "official_check.json", "w"), indent=1)
    print(a.shape, tg.shape)
    print(json.dumps(chk.get(2024, {}), indent=1))


# ------------------------------------------------------------------------------------------------- fit
TEAM_FEATS = ["itt", "exp_margin", "total_line", "t_plays_pg", "t_pass_att_pg", "t_targets_pg", "t_carries_pg",
              "t_proe", "o_plays_pg", "o_pass_att_pg", "o_carries_pg", "wind_f", "temp_f", "dome", "home",
              "wind_missing", "post"]


def fit_team(tg: pd.DataFrame, season: int) -> pd.DataFrame:
    """Ridge models for team pass attempts / targets / carries, trained on seasons < season."""
    from sklearn.linear_model import Ridge
    X = tg[TEAM_FEATS].copy()
    X["wind2"] = np.maximum(X.wind_f - 10, 0)
    X["margin_x_total"] = X.exp_margin * X.total_line / 45
    fill = X[tg.season < season].mean()
    X = X.fillna(fill)
    tr = (tg.season < season) & (tg.season >= 2013) & tg.to_n.gt(2)
    te = tg.season == season
    res = tg.loc[te, ["game_id", "team"]].copy()
    for y in ["pass_att", "targets", "carries"]:
        m = Ridge(alpha=1.0).fit(X[tr], tg.loc[tr, y])
        res[f"e_tm_{y}"] = m.predict(X[te])
    return res


PLAYER_FEATS = ["itt", "exp_margin", "total_line", "wind_f", "temp_f", "dome", "home", "post", "wind_missing",
                "tgt_share", "car_share", "att_share", "snap_ew", "adot", "ypt_s", "cr_s", "ypc_s", "ypa_s",
                "comp_s", "sack_rate", "opp_ypa_mult", "opp_ypc_mult", "opp_cmp_mult", "opp_ypt_mult",
                "opp_cr_mult", "n_prior", "days_since", "questionable", "e_tm_pass_att", "e_tm_targets",
                "e_tm_carries", "is_starter_qb"]


def structural(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Opportunity x share x efficiency x opponent. mode 'late' (actives known) or 'early' (injury report)."""
    s = pd.DataFrame(index=d.index)
    if mode == "late":
        rt = d.tgt_share / np.maximum(d.act_sum_tgt_share, 0.55)
        rc = d.car_share / np.maximum(d.act_sum_car_share, 0.55)
    else:
        rt = d.tgt_share / np.maximum(d.early_sum_tgt, 0.55)
        rc = d.car_share / np.maximum(d.early_sum_car, 0.55)
    # partial renormalisation (half-way): vacated volume is not fully redistributed to incumbents
    s["share_t"] = 0.5 * d.tgt_share + 0.5 * np.minimum(rt, 0.45)
    s["share_c"] = 0.5 * d.car_share + 0.5 * np.minimum(rc, 0.85)
    s["e_targets"] = d.e_tm_targets * s.share_t
    s["e_carries"] = d.e_tm_carries * s.share_c
    wind_pen = 1 - 0.006 * np.maximum(d.wind_f - 10, 0)
    s["e_att"] = d.e_tm_pass_att * np.clip(d.att_share.where(d.att_share > 0.5, 0.9), 0.5, 1.0)
    s["sm_pass_att"] = s.e_att
    s["sm_cmp"] = s.e_att * d.comp_s * d.opp_cmp_mult.fillna(1)
    s["sm_pass_yds"] = s.e_att * d.ypa_s * d.opp_ypa_mult.fillna(1) * wind_pen
    s["sm_rush_yds"] = s.e_carries * d.ypc_s * d.opp_ypc_mult.fillna(1)
    s["sm_rec_yds"] = s.e_targets * d.ypt_s * d.opp_ypt_mult.fillna(1) * wind_pen
    s["sm_rec"] = s.e_targets * d.cr_s * d.opp_cr_mult.fillna(1)
    return s


def universe(a: pd.DataFrame, market: str) -> pd.Series:
    stat, grp, _ = MARKETS[market]
    base = a.n_prior >= 3
    if grp == "QB":
        return base & (a.is_starter_qb == 1)
    if grp == "RB":
        return base & (a.position == "RB") & (a.l8_rush_yds >= 25)
    return base & a.position.isin(["WR", "TE", "RB"]) & (a.l8_rec_yds >= 20)


def train_pool(a: pd.DataFrame, market: str) -> pd.Series:
    """Broader training rows: same position group with some prior volume."""
    stat, grp, _ = MARKETS[market]
    base = a.n_prior >= 1
    if grp == "QB":
        return base & (a.is_starter_qb == 1)
    if grp == "RB":
        return base & (a.position == "RB") & (a.l8_rush_yds >= 10)
    return base & a.position.isin(["WR", "TE", "RB"]) & (a.l8_rec_yds >= 8)


def fit():
    import lightgbm as lgb
    a = pd.read_parquet(SCR / "player_games.parquet")
    tg = pd.read_parquet(SCR / "team_games.parquet")
    team_pred = pd.concat([fit_team(tg, s) for s in range(2014, 2026)])
    a = a.merge(team_pred, on=["game_id", "team"], how="left")
    a = a[a.season >= 2014].reset_index(drop=True)
    preds = []
    for mode in ["late", "early"]:
        st = structural(a, mode)
        d = pd.concat([a, st], axis=1)
        feats = PLAYER_FEATS + ["share_t", "share_c", "e_targets", "e_carries", "e_att"]
        feats += [f"vac_{'late' if mode == 'late' else 'inj'}_tgt", f"vac_{'late' if mode == 'late' else 'inj'}_car"]
        for market, (stat, grp, kind) in MARKETS.items():
            sm = f"sm_{stat}"
            pool = train_pool(d, market)
            uni = universe(d, market)
            fx = feats + [sm, f"l4_{stat}", f"std_{stat}"]
            for S in range(2015, 2026):
                tr = pool & (d.season < S)
                te = (d.season == S) & pool
                params = dict(objective="poisson" if kind == "count" else "regression", learning_rate=0.05, n_jobs=2,
                              n_estimators=250, num_leaves=15, min_child_samples=100, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
                m = lgb.LGBMRegressor(**params).fit(d.loc[tr, fx], d.loc[tr, stat].clip(lower=0 if kind == "count"
                                                                                         else None))
                if kind == "yards":
                    # yards can be negative; L2 objective is fine
                    pass
                r = d.loc[te, ["game_id", "player_id", "season", "week", "team", "position", stat, sm,
                               f"l4_{stat}", f"std_{stat}"]].copy()
                r["gbm"] = m.predict(d.loc[te, fx])
                r["in_universe"] = uni[te].values
                r = r.rename(columns={stat: "y", sm: "struct", f"l4_{stat}": "last4", f"std_{stat}": "std"})
                r["market"], r["mode"] = market, mode
                preds.append(r)
            print(mode, market, "done", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(SCR / "preds.parquet")
    print(P.groupby(["mode", "market"]).size())


# -------------------------------------------------------------------------------------------- evaluate
NN, LOOKBACK = 400, 4


def cond_samples(mu_tr: np.ndarray, y_tr: np.ndarray, mu: np.ndarray, kind: str) -> np.ndarray:
    """For each mu, NN nearest training pairs in log(mu) -> samples (n, NN)."""
    lm_tr = np.log(np.maximum(mu_tr, 0.5))
    o = np.argsort(lm_tr)
    lm_s, mu_s, y_s = lm_tr[o], mu_tr[o], y_tr[o]
    lm = np.log(np.maximum(mu, 0.5))
    pos = np.searchsorted(lm_s, lm)
    lo = np.clip(pos - NN // 2, 0, len(lm_s) - NN)
    idx = lo[:, None] + np.arange(NN)[None, :]
    if kind == "yards":
        return y_s[idx] * (np.maximum(mu, 0.5)[:, None] / np.maximum(mu_s[idx], 0.5))
    return y_s[idx].astype(float)


def crps_samples(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    S = np.sort(S, axis=1)
    m = S.shape[1]
    t1 = np.abs(S - y[:, None]).mean(axis=1)
    w = (2 * np.arange(1, m + 1) - m - 1)
    t2 = (S * w[None, :]).sum(axis=1) / (m * m)  # = 0.5 * E|X-X'|
    return t1 - t2


def evaluate():
    P = pd.read_parquet(SCR / "preds.parquet")
    chk = json.load(open(SCR / "official_check.json"))
    res = {"definitions": __doc__, "official_check": chk, "markets": {}}
    methods = ["std", "last4", "struct", "gbm"]
    for (mode, market), D in P.groupby(["mode", "market"]):
        kind = MARKETS[market][2]
        D = D.dropna(subset=["y"]).copy()
        for mth in methods:
            D[mth] = D[mth].fillna(D["std"]).fillna(D["last4"])
        D = D.dropna(subset=methods)
        for split, seasons in [("dev", DEV), ("holdout", HOLD)]:
            rows = {}
            cal_all = {}
            for mth in methods:
                maes, crpss, biases, n = [], [], [], 0
                pit, over_med, cal = [], [], []
                for S in seasons:
                    tr = D[(D.season >= S - LOOKBACK) & (D.season < S) & D.in_universe]
                    te = D[(D.season == S) & D.in_universe]
                    if len(te) == 0:
                        continue
                    smp = cond_samples(tr[mth].to_numpy(float), tr.y.to_numpy(float), te[mth].to_numpy(float),
                                       kind)
                    y = te.y.to_numpy(float)
                    med = np.median(smp, axis=1)
                    maes.append(np.abs(y - med))
                    crpss.append(crps_samples(smp, y))
                    biases.append(y - te[mth].to_numpy(float))
                    # randomised PIT for ties
                    lt = (smp < y[:, None]).mean(axis=1)
                    le = (smp <= y[:, None]).mean(axis=1)
                    rng = np.random.default_rng(S)
                    pit.append(lt + rng.random(len(y)) * (le - lt))
                    # P(over own median) vs realised (line = median rounded to x.5 so no pushes)
                    line = np.floor(med) + 0.5
                    over_med.append(np.c_[(smp > line[:, None]).mean(axis=1), (y > line)])
                    # calibration of P(over X) across a fixed grid of lines
                    grid = LINE_GRID[market]
                    for X in grid:
                        cal.append(np.c_[(smp > X).mean(axis=1), (y > X), np.full(len(y), X)])
                    n += len(y)
                if n == 0:
                    continue
                mae = float(np.mean(np.concatenate(maes)))
                crps = float(np.mean(np.concatenate(crpss)))
                pit = np.concatenate(pit)
                om = np.concatenate(over_med)
                cal = np.concatenate(cal)
                bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
                b = np.digitize(cal[:, 0], bins) - 1
                ctab = [{"bin": f"{bins[i]:.1f}-{min(bins[i+1],1):.1f}", "n": int((b == i).sum()),
                         "pred": round(float(cal[b == i, 0].mean()), 3), "obs": round(float(cal[b == i, 1].mean()), 3)}
                        for i in range(10) if (b == i).sum() > 0]
                ece = float(sum(abs(c["pred"] - c["obs"]) * c["n"] for c in ctab) / sum(c["n"] for c in ctab))
                rows[mth] = {"n": int(n), "mae_median": round(mae, 3), "crps": round(crps, 3),
                             "mean_bias_y_minus_mu": round(float(np.mean(np.concatenate(biases))), 3),
                             "pit_deciles": np.histogram(pit, bins=np.linspace(0, 1, 11))[0].tolist(),
                             "pit_max_dev_pct": round(float(np.max(np.abs(np.histogram(pit, bins=np.linspace(0, 1, 11))[0]
                                                                        / len(pit) - 0.1)) * 100), 2),
                             "over_own_median": {"pred": round(float(om[:, 0].mean()), 4),
                                                 "obs": round(float(om[:, 1].mean()), 4)},
                             "calib_ece": round(ece, 4), "calib_table": ctab}
            res["markets"].setdefault(market, {}).setdefault(mode, {})[split] = rows
            g = rows.get("gbm", {})
            print(f"{mode:5s} {market:10s} {split:7s} n={g.get('n')} " + " ".join(
                f"{m}:MAE={rows[m]['mae_median']:.2f}/CRPS={rows[m]['crps']:.2f}" for m in methods if m in rows),
                flush=True)
    # bootstrap-by-game difference GBM vs best baseline on holdout MAE (late mode)
    res["paired"] = paired_tests(P)
    res["breakeven_offsets"] = breakeven_offsets(P)
    res["credit_plan"] = credit_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "props.json", "w"), indent=1, default=float)


LINE_GRID = {
    "pass_yds": np.arange(179.5, 300, 20.0), "pass_att": np.arange(25.5, 42, 3.0),
    "pass_cmp": np.arange(16.5, 28, 2.0), "rush_yds": np.arange(29.5, 110, 10.0),
    "rec_yds": np.arange(19.5, 100, 10.0), "receptions": np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
}


def breakeven_offsets(P: pd.DataFrame) -> dict:
    """How far (in stat units) a line must sit from the model's median before the model's own distribution
    says the bet wins 52.38% (-110 break-even) / 55% / 57.5%. Holdout, late GBM. A model only profits if its
    median is right AND the book line is at least this far away - i.e. the market must be this wrong."""
    out = {}
    D = P[(P["mode"] == "late") & P.season.isin(HOLD)].dropna(subset=["y", "gbm"])
    for market, M in D.groupby("market"):
        kind = MARKETS[market][2]
        r = {}
        offs = []
        for S in HOLD:
            te = M[(M.season == S) & M.in_universe]
            tr = P[(P["mode"] == "late") & (P.market == market) & (P.season >= S - LOOKBACK) & (P.season < S)
                   & P.in_universe].dropna(subset=["y", "gbm"])
            smp = np.sort(cond_samples(tr.gbm.to_numpy(float), tr.y.to_numpy(float), te.gbm.to_numpy(float),
                                       kind), axis=1)
            offs.append(smp)
        smp = np.concatenate(offs)
        med = np.median(smp, axis=1)
        for target in (0.5238, 0.55, 0.575):
            # line L = quantile at (1-target) => P(over L) = target ; offset = median - L
            q = np.quantile(smp, 1 - target, axis=1)
            r[f"p{target}"] = {"median_offset_units": round(float(np.median(med - q)), 2),
                               "median_offset_pct_of_median": round(float(np.median((med - q) /
                                                                                     np.maximum(med, 1))) * 100, 1)}
        r["typical_median"] = round(float(np.median(med)), 1)
        out[market] = r
    return out


def credit_plan() -> dict:
    """The Odds API historical per-event cost: 10 x markets returned x regions; <=10 bookmakers = 1 region.
    Historical events list: 1 credit per call. Props history exists from 2023-05-03 (2023 season onward)."""
    ev = {"2023": 285, "2024": 285, "2025": 285}  # 272 REG + 13 POST each
    n = sum(ev.values())
    list_calls = 3 * 22 * 2  # ~one events-list call per week per snapshot type, ~22 weeks/season
    plans = {}
    for mk in (1, 2, 3, 4):
        for sn in (1, 2, 3):
            plans[f"{mk}mkt_x_{sn}snap"] = int(n * sn * mk * 10 + list_calls)
    return {"events_2023_2025": n, "cost_per_event_snapshot_market": 10, "events_list_calls_approx": list_calls,
            "totals_all_3_seasons": plans,
            "one_season_1mkt_1snap": 285 * 10,
            "plans_usd_month": {"20K": 30, "100K": 59, "5M": 119, "15M": 249}}


def paired_tests(P: pd.DataFrame) -> dict:
    """Holdout: absolute error of the mean-as-point-forecast, GBM vs baselines, game-clustered bootstrap."""
    out = {}
    D = P[(P["mode"] == "late") & P.in_universe & P.season.isin(HOLD)].dropna(subset=["y"]).copy()
    rng = np.random.default_rng(7)
    for market, M in D.groupby("market"):
        M = M.copy()
        for c in ["struct", "gbm", "last4"]:
            M[c] = M[c].fillna(M["std"])
        M = M.dropna(subset=["std", "last4", "gbm", "struct"])
        r = {}
        games = M.game_id.unique()
        gi = {g: i for i, g in enumerate(games)}
        gidx = M.game_id.map(gi).to_numpy()
        for base in ["std", "last4", "struct"]:
            dlt = (np.abs(M.y - M[base]) - np.abs(M.y - M.gbm)).to_numpy()  # >0 = GBM better
            per = np.bincount(gidx, weights=dlt, minlength=len(games))
            cnt = np.bincount(gidx, minlength=len(games))
            bs = []
            for _ in range(1000):
                s = rng.integers(0, len(games), len(games))
                bs.append(per[s].sum() / cnt[s].sum())
            r[f"gbm_vs_{base}_abs_err_gain"] = {"mean": round(float(dlt.mean()), 3),
                                                "ci95": [round(float(np.percentile(bs, 2.5)), 3),
                                                         round(float(np.percentile(bs, 97.5)), 3)]}
        r["corr_y_gbm"] = round(float(np.corrcoef(M.y, M.gbm)[0, 1]), 3)
        r["corr_y_std"] = round(float(np.corrcoef(M.y, M["std"])[0, 1]), 3)
        r["sd_y"] = round(float(M.y.std()), 2)
        r["rmse_gbm"] = round(float(np.sqrt(((M.y - M.gbm) ** 2).mean())), 2)
        r["r2_gbm"] = round(float(1 - ((M.y - M.gbm) ** 2).mean() / M.y.var()), 4)
        r["r2_std"] = round(float(1 - ((M.y - M["std"]) ** 2).mean() / M.y.var()), 4)
        out[market] = r
    return out


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    {"build": build, "fit": fit, "evaluate": evaluate}[sys.argv[1]]()
