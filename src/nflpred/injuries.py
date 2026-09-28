"""Injury-report feature: how much of a team's usual playing time is missing.

For each game, sum the recent snap share of every non-QB player the team listed as Out or
Doubtful on that week's injury report (QBs are handled by the starting-QB features).
Snap share comes from the player's previous appearances only, so it's leak-free; injury
reports are published before kickoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.8}
SNAP_HALFLIFE = 4  # player appearances
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}


def player_snap_share(snaps: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Post-game EWMA of each player's snap share, keyed by date (for as-of joins)."""
    s = snaps.merge(sched[["game_id", "gameday"]], on="game_id", how="inner")
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0)
    s["unit"] = np.where(s["offense_pct"].fillna(0) >= s["defense_pct"].fillna(0), "off", "def")
    s = s.sort_values(["pfr_player_id", "gameday"])
    s["share_ewm"] = s.groupby("pfr_player_id")["share"].transform(
        lambda x: x.ewm(halflife=SNAP_HALFLIFE).mean())
    return s[["pfr_player_id", "gameday", "share_ewm", "unit"]].sort_values("gameday")


def team_injury_load(injuries: pd.DataFrame, snaps: pd.DataFrame, players: pd.DataFrame,
                     sched: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (game_id, team): inj_off, inj_def (sum of missing snap share),
    and inj_reported (1 if the team's injury report for that week exists)."""
    inj = injuries.copy()
    inj["team"] = inj["team"].replace(TEAM_MAP)
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")

    long = pd.concat([
        sched[["game_id", "season", "week", "gameday", "home_team"]].rename(columns={"home_team": "team"}),
        sched[["game_id", "season", "week", "gameday", "away_team"]].rename(columns={"away_team": "team"}),
    ])
    reported = inj[["season", "week", "team"]].drop_duplicates().assign(inj_reported=1)

    out = inj[inj["report_status"].isin(STATUS_WEIGHT) & (inj["position"] != "QB")].copy()
    out["w"] = out["report_status"].map(STATUS_WEIGHT)
    out = out.merge(players[["gsis_id", "pfr_id"]].dropna(), on="gsis_id", how="inner")
    out = out.merge(long, on=["season", "week", "team"], how="inner")

    share = player_snap_share(snaps, sched).rename(columns={"pfr_player_id": "pfr_id"})
    out = pd.merge_asof(out.sort_values("gameday"), share, on="gameday", by="pfr_id",
                        allow_exact_matches=False)  # strictly before this game
    out["lost"] = out["w"] * out["share_ewm"].fillna(0)
    agg = out.pivot_table(index=["game_id", "team"], columns="unit", values="lost",
                          aggfunc="sum", fill_value=0).reset_index()
    for u in ("off", "def"):
        if u not in agg:
            agg[u] = 0.0
    agg = agg.rename(columns={"off": "inj_off", "def": "inj_def"})[["game_id", "team", "inj_off", "inj_def"]]

    res = long.merge(agg, on=["game_id", "team"], how="left").merge(
        reported, on=["season", "week", "team"], how="left")
    res[["inj_off", "inj_def", "inj_reported"]] = res[["inj_off", "inj_def", "inj_reported"]].fillna(0)
    return res[["game_id", "team", "inj_off", "inj_def", "inj_reported"]]
