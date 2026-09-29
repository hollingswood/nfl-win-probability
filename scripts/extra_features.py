"""Candidate features under test (not in production until they pass validation + holdout).

Every candidate uses only information available before kickoff:
  A. special teams:   rolling net EPA on kickoffs, punts, field goals, extra points
  B. injury clusters: starters (>=60% recent snap share) Out/Doubtful, by position group
  C. QB priors:       draft-capital prior for inexperienced QBs + rating fades with time off
  D. early season:    new head coach flag and early-week interactions
  E. referee:         referee's past home-margin tendency (shrunk), prior games only
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nflpred import features as F
from nflpred.injuries import player_snap_share, TEAM_MAP as INJ_TEAM_MAP

ST_TYPES = ("kickoff", "punt", "field_goal", "extra_point")


def _long(df):
    return pd.concat([
        df[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
        df[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "team"}),
    ])


def _to_diff(df, per_team: pd.DataFrame, col: str, name: str, flip=False):
    out = df.copy()
    for side in ("home", "away"):
        m = per_team[["game_id", "team", col]].rename(columns={"team": f"{side}_team", col: f"{side}_{name}"})
        out = out.merge(m, on=["game_id", f"{side}_team"], how="left")
    d = out[f"home_{name}"].fillna(0) - out[f"away_{name}"].fillna(0)
    out[f"{name}_diff"] = -d if flip else d
    return out


# ---------------------------------------------------------------- A. special teams
def special_teams(df, pbp):
    p = pbp[pbp["play_type"].isin(ST_TYPES) & pbp["epa"].notna()].copy()
    p["posteam"], p["defteam"] = F._norm_team(p["posteam"]), F._norm_team(p["defteam"])
    a = p.groupby(["game_id", "posteam"])["epa"].sum().rename("st_for").reset_index().rename(columns={"posteam": "team"})
    b = p.groupby(["game_id", "defteam"])["epa"].sum().rename("st_against").reset_index().rename(columns={"defteam": "team"})
    lg = _long(df).merge(a, on=["game_id", "team"], how="left").merge(b, on=["game_id", "team"], how="left")
    lg["st_net"] = lg["st_for"] - lg["st_against"]
    lg = lg.sort_values(["team", "gameday", "game_id"])
    lg["pre_st"] = lg.groupby("team")["st_net"].transform(
        lambda x: x.shift(1).ewm(halflife=F.TEAM_HALFLIFE, ignore_na=True).mean())
    return _to_diff(df, lg, "pre_st", "st")


# ---------------------------------------------------------------- B. injury clusters
GROUPS = {"ol": {"T", "G", "C", "OT", "OG", "OL", "LT", "RT", "LG", "RG"},
          "sec": {"CB", "S", "FS", "SS", "DB"},
          "skill": {"WR", "TE", "RB", "FB"},
          "front": {"DE", "DT", "NT", "LB", "OLB", "ILB", "MLB", "DL", "EDGE"}}


def injury_clusters(df, injuries, snaps, players, starter_share=0.6):
    sched = df[["game_id", "season", "week", "gameday", "home_team", "away_team"]]
    share = player_snap_share(snaps, sched).rename(columns={"pfr_player_id": "pfr_id"})
    pos = snaps.drop_duplicates("pfr_player_id", keep="last")[["pfr_player_id", "position"]].rename(
        columns={"pfr_player_id": "pfr_id", "position": "snap_pos"})
    inj = injuries.copy()
    inj["team"] = inj["team"].replace(INJ_TEAM_MAP)
    inj = inj[inj["report_status"].isin(["Out", "Doubtful"])].drop_duplicates(["season", "week", "team", "gsis_id"])
    inj = inj.merge(players[["gsis_id", "pfr_id"]].dropna(), on="gsis_id").merge(pos, on="pfr_id", how="left")
    long = pd.concat([sched[["game_id", "season", "week", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
                      sched[["game_id", "season", "week", "gameday", "away_team"]].rename(columns={"away_team": "team"})])
    inj = inj.merge(long, on=["season", "week", "team"])
    inj = pd.merge_asof(inj.sort_values("gameday"), share, on="gameday", by="pfr_id", allow_exact_matches=False)
    inj = inj[inj["share_ewm"] >= starter_share]
    pos_col = inj["snap_pos"].fillna(inj["position"])
    for gname, members in GROUPS.items():
        inj[gname] = pos_col.isin(members).astype(int)
    agg = inj.groupby(["game_id", "team"])[list(GROUPS)].sum().reset_index()
    out = df.copy()
    for g in GROUPS:
        out = _to_diff(out, agg.assign(**{g: agg[g]}), g, f"out_{g}", flip=True)  # positive = away missing more
    for side in ("home", "away"):
        out[f"{side}_ol_cluster"] = (out[f"{side}_out_ol"].fillna(0) >= 2).astype(int)
    out["ol_cluster_diff"] = out["away_ol_cluster"] - out["home_ol_cluster"]
    return out


# ---------------------------------------------------------------- C. QB priors
def draft_prior_table(drafts, pbp, sched, players, until_season=2014, first_n=150):
    """Dropback-weighted EPA over a QB's first `first_n` career dropbacks, by draft bucket.
    Only QBs whose careers began inside the data (rookie season 2012..until_season) and only
    seasons <= until_season, so the table never sees the validation or holdout years."""
    qbs = players[(players["position"] == "QB")]
    start = pd.to_numeric(qbs.get("rookie_season", qbs.get("entry_year")), errors="coerce")
    ok = set(qbs.loc[start.between(2012, until_season), "gsis_id"])
    p = pbp[(pbp["season"] <= until_season) & pbp["qb_dropback"].eq(1) & pbp["qb_epa"].notna()
            & pbp["id"].isin(ok)]
    p = p.merge(sched[["game_id", "gameday"]], on="game_id").sort_values("gameday")
    p = p[p.groupby("id").cumcount() < first_n]
    agg = p.groupby("id")["qb_epa"].agg(["sum", "size"]).reset_index().rename(columns={"id": "gsis_id"})
    agg = agg[agg["size"] >= 30].merge(drafts[["gsis_id", "round"]], on="gsis_id", how="left")
    agg["bucket"] = draft_bucket(agg["round"])
    t = agg.groupby("bucket")[["sum", "size"]].sum()
    return (t["sum"] / t["size"]).to_dict(), agg.groupby("bucket").size().to_dict()


def draft_bucket(rnd):
    r = pd.to_numeric(rnd, errors="coerce")
    return np.select([r == 1, r.between(2, 3), r >= 4], ["r1", "r2_3", "r4_7"], "udfa")


def qb_ratings_v2(pbp, sched, priors: dict, drafts, time_halflife_days=365, prior_n=None):
    prior_n = prior_n or F.QB_PRIOR_DROPBACKS
    bucket = dict(zip(drafts["gsis_id"], draft_bucket(drafts["round"])))
    p = pbp[pbp["qb_dropback"].eq(1) & pbp["qb_epa"].notna() & pbp["id"].notna()]
    qg = p.groupby(["id", "game_id"]).agg(epa=("qb_epa", "sum"), db=("qb_epa", "size")).reset_index()
    qg = qg.merge(sched[["game_id", "gameday"]], on="game_id").sort_values(["id", "gameday"])
    rows = []
    for qb, d in qg.groupby("id", sort=False):
        prior = priors.get(bucket.get(qb, "udfa"), F.QB_PRIOR)
        num = den = 0.0
        last = None
        for gd, e, n in zip(d["gameday"], d["epa"], d["db"]):
            if last is not None and time_halflife_days:
                fade = 0.5 ** ((gd - last).days / time_halflife_days)
                num, den = num * fade, den * fade
            num = num * F.QB_DECAY + e
            den = den * F.QB_DECAY + n
            last = gd
            rows.append((qb, gd, (num + prior * prior_n) / (den + prior_n), prior))
    return pd.DataFrame(rows, columns=["qb_id", "gameday", "qb_rating", "prior"]).sort_values("gameday"), bucket


def qb_priors(df, pbp, drafts, players=None, time_halflife_days=365, use_draft=True):
    sched = df[["game_id", "gameday"]]
    priors = draft_prior_table(drafts, pbp, df, players)[0] if use_draft else {}
    qbr, bucket = qb_ratings_v2(pbp, sched, priors, drafts, time_halflife_days)
    out = df.copy()
    for side in ("home", "away"):
        key = out[["gameday", f"{side}_qb_id"]].reset_index().rename(columns={f"{side}_qb_id": "qb_id"}).sort_values("gameday")
        key["qb_id"] = key["qb_id"].astype(qbr["qb_id"].dtype)
        m = pd.merge_asof(key, qbr[["qb_id", "gameday", "qb_rating"]], on="gameday", by="qb_id", allow_exact_matches=False)
        r = m.set_index("index")["qb_rating"].reindex(out.index)
        # also age a rating that exists but is stale (time since the QB's last game)
        default = out[f"{side}_qb_id"].map(lambda q: priors.get(bucket.get(q, "udfa"), F.QB_PRIOR))
        out[f"{side}_qb_rating"] = r.fillna(default)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_qb_change") or c == "qb_change_diff"])
    out = F.attach_qb_change(out)
    out["qb_diff"] = out["home_qb_rating"] - out["away_qb_rating"]
    out["fw_qb_diff"] = out["final_week"] * out["qb_diff"]
    return out, priors


# ---------------------------------------------------------------- D. early season
def early_season(df):
    out = df.copy().sort_values(["gameday", "game_id"])
    long = pd.concat([out[["season", "gameday", "home_team", "home_coach"]].set_axis(["season", "gameday", "team", "coach"], axis=1),
                      out[["season", "gameday", "away_team", "away_coach"]].set_axis(["season", "gameday", "team", "coach"], axis=1)])
    last = long.sort_values("gameday").groupby(["team", "season"])["coach"].last().reset_index()
    last["season"] += 1
    last = last.rename(columns={"coach": "prev_coach"})
    for side in ("home", "away"):
        m = out[["season", f"{side}_team", f"{side}_coach"]].merge(
            last.rename(columns={"team": f"{side}_team"}), on=["season", f"{side}_team"], how="left")
        out[f"{side}_new_coach"] = (m["prev_coach"].notna() & (m["prev_coach"] != m[f"{side}_coach"])).astype(int).values
    early = (out["week"] <= 4).astype(int)
    out["newcoach_diff"] = out["home_new_coach"] - out["away_new_coach"]
    out["early_elo"] = early * out["elo_diff"]
    out["early_qb"] = early * out["qb_diff"]
    out["early_pt"] = early * out["pt_diff_diff"]
    return out.sort_index()


# ---------------------------------------------------------------- E. referee
def referee(df, shrink=60):
    out = df.copy()
    d = out[["game_id", "gameday", "referee", "home_score", "away_score", "completed", "home_field"]].copy()
    d["margin"] = np.where(d["completed"] & (d["home_field"] == 1), d["home_score"] - d["away_score"], np.nan)
    d = d.sort_values(["gameday", "game_id"])
    league = d["margin"].expanding().mean().shift(1)  # league home margin before this game
    d["resid"] = d["margin"] - league
    # sum/count of residuals from this referee's strictly earlier dates
    g = d.groupby("referee")
    d["cum_sum"] = g["resid"].transform(lambda x: x.fillna(0).cumsum().shift(1))
    d["cum_n"] = g["resid"].transform(lambda x: x.notna().cumsum().shift(1))
    # same-day games by the same referee don't exist, so shift(1) = strictly earlier
    d["ref_home_edge"] = (d["cum_sum"].fillna(0) / (d["cum_n"].fillna(0) + shrink))
    out = out.merge(d[["game_id", "ref_home_edge"]], on="game_id", how="left")
    out["ref_home_edge"] = out["ref_home_edge"].fillna(0)
    return out


# ---------------------------------------------------------------- F. situational (schedule-derived)
AFC = {"BAL", "BUF", "CIN", "CLE", "DEN", "HOU", "IND", "JAX", "KC", "LV", "LAC", "MIA", "NE", "NYJ", "PIT", "TEN"}
GRASS = {"grass", "dessograss"}


def situational(df):
    out = df.copy()
    # conference game
    out["conf_game"] = (out["home_team"].isin(AFC) == out["away_team"].isin(AFC)).astype(int)
    # day of week (these change the size of home edge; learned as main effects)
    wd = pd.to_datetime(out["gameday"]).dt.dayofweek
    out["thu"], out["mon"] = (wd == 3).astype(int), (wd == 0).astype(int)
    # turf: does the visitor's home surface type differ from this venue's?
    s = out["surface"].fillna("").str.lower().str.strip()
    out["venue_grass"] = s.isin(GRASS).astype(int)
    home_surf = (out[out["location"] != "Neutral"].sort_values("gameday").groupby(["season", "home_team"])["venue_grass"]
                 .agg(lambda x: x.mode().iloc[0] if len(x) else np.nan).rename("team_grass").reset_index())
    m = out[["season", "away_team"]].merge(home_surf.rename(columns={"home_team": "away_team"}), on=["season", "away_team"], how="left")
    out["surface_mismatch"] = ((m["team_grass"].values != out["venue_grass"].values) & m["team_grass"].notna().values).astype(int)
    # per-team previous game (same season only)
    long = pd.concat([
        out[["game_id", "season", "gameday", "home_team", "home_score", "away_score", "overtime", "home_rest"]]
            .set_axis(["game_id", "season", "gameday", "team", "pf", "pa", "ot", "rest"], axis=1).assign(is_home=1),
        out[["game_id", "season", "gameday", "away_team", "away_score", "home_score", "overtime", "away_rest"]]
            .set_axis(["game_id", "season", "gameday", "team", "pf", "pa", "ot", "rest"], axis=1).assign(is_home=0),
    ]).sort_values(["team", "gameday"])
    g = long.groupby(["team", "season"])
    long["prev_home"] = g["is_home"].shift(1)
    long["prev_ot"] = g["ot"].shift(1).fillna(0)
    long["prev_margin"] = (g["pf"].shift(1) - g["pa"].shift(1))
    long["big_loss"] = (long["prev_margin"] <= -17).astype(int)
    long["big_win"] = (long["prev_margin"] >= 17).astype(int)
    long["off_bye"] = (long["rest"] >= 13).astype(int)
    for side in ("home", "away"):
        m = long[["game_id", "team", "prev_home", "prev_ot", "big_loss", "big_win", "off_bye"]].rename(
            columns={c: f"{side}_{c}" for c in ["prev_home", "prev_ot", "big_loss", "big_win", "off_bye"]} | {"team": f"{side}_team"})
        out = out.merge(m, on=["game_id", f"{side}_team"], how="left")
    out["home_stand"] = (out["home_prev_home"] == 1).astype(int)          # home team was also home last week
    out["road_trip"] = (out["away_prev_home"] == 0).astype(int)           # visitor was also on the road last week
    out["ot_diff"] = out["away_prev_ot"].fillna(0) - out["home_prev_ot"].fillna(0)   # + = visitor coming off OT
    out["bye_diff"] = out["home_off_bye"].fillna(0) - out["away_off_bye"].fillna(0)
    out["bounce_diff"] = out["home_big_loss"].fillna(0) - out["away_big_loss"].fillna(0)
    out["letdown_diff"] = out["home_big_win"].fillna(0) - out["away_big_win"].fillna(0)
    return out
