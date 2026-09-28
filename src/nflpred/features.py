"""Pre-game feature engineering.

Every feature for a game is computed ONLY from games that finished before that game's
kickoff date. `tests/test_leakage.py` enforces this by re-computing features with all
future results removed and asserting nothing changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_HALFLIFE = 8        # games; recent form weighting for team efficiency
QB_DECAY = 0.965         # per-game decay for QB rating (~20-game half-life)
QB_PRIOR = -0.10         # EPA/dropback prior (≈ replacement-level starter)
QB_PRIOR_DROPBACKS = 150  # strength of the prior, in dropbacks
ELO_K = 20.0
ELO_HFA = 48.0
ELO_MEAN = 1505.0
ELO_REVERT = 1 / 3
WP_FILTER: tuple[float, float] | None = None  # e.g. (0.05, 0.95): drop garbage-time plays
OPP_ADJUST = False
POWER_RATINGS = False    # weekly opponent-adjusted ratings (ratings.py): tested, no gain; off       # adjust each game's efficiency for the opponent's pre-game rating

# Relocated franchises: nflverse uses current abbreviations in pbp but schedules keep
# historical ones in some seasons; normalize so a team's history is continuous.
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

FEATURES = [
    "elo_diff", "qb_diff", "off_epa_diff", "def_epa_diff", "off_sr_diff", "def_sr_diff",
    "pass_epa_diff", "rush_epa_diff", "to_margin_diff", "pt_diff_diff", "rest_diff",
    "home_field", "div_game", "qb_change_diff", "inj_off_diff", "inj_def_diff",
    "fw_elo_diff", "fw_pt_diff_diff", "fw_qb_diff",
]

FEATURE_LABELS = {
    "elo_diff": "Team strength (Elo)",
    "qb_diff": "Starting QB edge",
    "off_epa_diff": "Offensive efficiency",
    "def_epa_diff": "Defensive efficiency",
    "off_sr_diff": "Offensive success rate",
    "def_sr_diff": "Defensive success rate",
    "pass_epa_diff": "Passing game",
    "rush_epa_diff": "Running game",
    "to_margin_diff": "Turnover margin",
    "pt_diff_diff": "Recent point differential",
    "rest_diff": "Rest advantage",
    "home_field": "Home field",
    "div_game": "Division game",
    "qb_change_diff": "Starter vs. team's usual QB",
    "inj_off_diff": "Offensive injuries",
    "inj_def_diff": "Defensive injuries",
    "fw_elo_diff": "Final week: strength",
    "fw_pt_diff_diff": "Final week: point diff",
    "fw_qb_diff": "Final week: QB",
}


def _norm_team(s: pd.Series) -> pd.Series:
    return s.replace(TEAM_MAP)


def prepare_schedule(games: pd.DataFrame, min_season: int = 2012) -> pd.DataFrame:
    g = games[(games["season"] >= min_season)].copy()
    for c in ("home_team", "away_team"):
        g[c] = _norm_team(g[c])
    g["gameday"] = pd.to_datetime(g["gameday"])
    g["completed"] = g["home_score"].notna() & g["away_score"].notna()
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- team efficiency
def team_game_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["pass"].eq(1) | pbp["rush"].eq(1)) & pbp["epa"].notna()
            & pbp["qb_kneel"].ne(1) & pbp["qb_spike"].ne(1)].copy()
    if WP_FILTER is not None and "wp" in p:
        lo, hi = WP_FILTER
        p = p[p["wp"].between(lo, hi) | p["wp"].isna()]
    p["posteam"] = _norm_team(p["posteam"])
    p["defteam"] = _norm_team(p["defteam"])
    p["fumbled_1_team"] = _norm_team(p["fumbled_1_team"])
    p["pass_epa"] = np.where(p["pass"].eq(1), p["epa"], np.nan)
    p["rush_epa"] = np.where(p["rush"].eq(1), p["epa"], np.nan)
    p["to"] = p["interception"].fillna(0) + (
        p["fumble_lost"].fillna(0) * (p["fumbled_1_team"] == p["posteam"]))
    off = p.groupby(["game_id", "posteam"]).agg(
        off_epa=("epa", "mean"), off_sr=("success", "mean"),
        pass_epa=("pass_epa", "mean"), rush_epa=("rush_epa", "mean"),
        to_committed=("to", "sum"), defteam=("defteam", "first"),
    ).reset_index().rename(columns={"posteam": "team"})
    opp = off[["game_id", "defteam", "off_epa", "off_sr", "to_committed"]].rename(
        columns={"defteam": "team", "off_epa": "def_epa", "off_sr": "def_sr",
                 "to_committed": "to_forced"})
    out = off.drop(columns="defteam").merge(opp, on=["game_id", "team"], how="left")
    out["to_margin"] = out["to_forced"] - out["to_committed"]
    return out


def team_long(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with points for/against."""
    h = sched[["game_id", "season", "gameday", "home_team", "away_team", "home_score", "away_score"]]
    home = h.rename(columns={"home_team": "team", "away_team": "opp",
                             "home_score": "pf", "away_score": "pa"})
    away = h.rename(columns={"away_team": "team", "home_team": "opp",
                             "away_score": "pf", "home_score": "pa"})
    return pd.concat([home, away], ignore_index=True)


def _pre_ewm(lg: pd.DataFrame, col: str) -> pd.Series:
    """EWMA over a team's strictly-prior games (shift(1) => the game itself is excluded)."""
    return lg.groupby("team")[col].transform(
        lambda x: x.shift(1).ewm(halflife=TEAM_HALFLIFE, ignore_na=True).mean())


def rolling_team_features(sched: pd.DataFrame, tgs: pd.DataFrame) -> pd.DataFrame:
    lg = team_long(sched).merge(tgs, on=["game_id", "team"], how="left")
    lg["pt_diff"] = lg["pf"] - lg["pa"]
    lg = lg.sort_values(["team", "gameday", "game_id"])
    stats = ["off_epa", "def_epa", "off_sr", "def_sr", "pass_epa", "rush_epa", "to_margin", "pt_diff"]
    for s in stats:
        lg[f"pre_{s}"] = _pre_ewm(lg, s)
    if OPP_ADJUST:
        # Credit offense for facing good defenses (and vice versa), using the opponent's
        # PRE-game rating only, then re-average. Still leak-free: game g's adjusted value
        # is only used for games after g.
        opp = lg[["game_id", "team"] + [f"pre_{s}" for s in ("off_epa", "def_epa", "off_sr", "def_sr")]]
        opp = opp.rename(columns={"team": "opp", **{c: "opp_" + c for c in opp.columns if c.startswith("pre_")}})
        lg = lg.merge(opp, on=["game_id", "opp"], how="left")
        for a, b in (("off_epa", "def_epa"), ("off_sr", "def_sr"), ("def_epa", "off_epa"), ("def_sr", "off_sr")):
            center = 0.44 if "sr" in b else 0.0  # fixed league-typical level (no peeking)
            lg[f"{a}_adj"] = lg[a] - (lg[f"opp_pre_{b}"].fillna(center) - center)
        lg = lg.sort_values(["team", "gameday", "game_id"])
        for a in ("off_epa", "def_epa", "off_sr", "def_sr"):
            lg[f"pre_{a}"] = _pre_ewm(lg, f"{a}_adj")
    return lg[["game_id", "team"] + [f"pre_{s}" for s in stats]]


# ---------------------------------------------------------------- QB
def qb_ratings(pbp: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Post-game shrunk EPA/dropback state for each QB after each game they played."""
    p = pbp[pbp["qb_dropback"].eq(1) & pbp["qb_epa"].notna() & pbp["id"].notna()]
    qg = p.groupby(["id", "game_id"]).agg(epa=("qb_epa", "sum"), db=("qb_epa", "size")).reset_index()
    qg = qg.merge(sched[["game_id", "gameday"]], on="game_id").sort_values(["id", "gameday"])
    rows = []
    for qb, d in qg.groupby("id", sort=False):
        num = den = 0.0
        for gd, e, n in zip(d["gameday"], d["epa"], d["db"]):
            num = num * QB_DECAY + e
            den = den * QB_DECAY + n
            rows.append((qb, gd, (num + QB_PRIOR * QB_PRIOR_DROPBACKS) / (den + QB_PRIOR_DROPBACKS)))
    return pd.DataFrame(rows, columns=["qb_id", "gameday", "qb_rating"]).sort_values("gameday")


def attach_qb(sched: pd.DataFrame, qbr: pd.DataFrame) -> pd.DataFrame:
    out = sched.copy()
    for side in ("home", "away"):
        # Fallback: if starter unknown, use the team's most recent starter.
        long = pd.concat([
            out[["gameday", "home_team", "home_qb_id"]].set_axis(["gameday", "team", "qb"], axis=1),
            out[["gameday", "away_team", "away_qb_id"]].set_axis(["gameday", "team", "qb"], axis=1),
        ]).sort_values("gameday")
        long["qb"] = long.groupby("team")["qb"].ffill()
        last = long.drop_duplicates(["gameday", "team"], keep="last")
        filled = out[["gameday", f"{side}_team"]].merge(
            last.rename(columns={"team": f"{side}_team"}), on=["gameday", f"{side}_team"], how="left")["qb"]
        out[f"{side}_qb_id"] = out[f"{side}_qb_id"].fillna(pd.Series(filled.values, index=out.index))

        key = out[["gameday", f"{side}_qb_id"]].reset_index().rename(columns={f"{side}_qb_id": "qb_id"})
        key = key.sort_values("gameday")
        m = pd.merge_asof(key, qbr, on="gameday", by="qb_id", allow_exact_matches=False)
        out[f"{side}_qb_rating"] = m.set_index("index")["qb_rating"].reindex(out.index).fillna(QB_PRIOR)
    return out


def attach_qb_change(df: pd.DataFrame) -> pd.DataFrame:
    """How today's starter compares with the QBs who produced the team's recent stats.

    Team efficiency features are built from recent games, so if the usual starter is out,
    they overstate (or understate) the team. qb_change = starter rating - EWMA of the team's
    previous starters' pre-game ratings (strictly prior games).
    """
    long = pd.concat([
        df[["game_id", "gameday", "home_team", "home_qb_rating", "completed"]].set_axis(
            ["game_id", "gameday", "team", "qbr", "completed"], axis=1),
        df[["game_id", "gameday", "away_team", "away_qb_rating", "completed"]].set_axis(
            ["game_id", "gameday", "team", "qbr", "completed"], axis=1),
    ]).sort_values(["team", "gameday", "game_id"])
    played = long["qbr"].where(long["completed"])
    long["team_qb_base"] = played.groupby(long["team"]).transform(
        lambda x: x.shift(1).ewm(halflife=TEAM_HALFLIFE, ignore_na=True).mean())
    long["qb_change"] = (long["qbr"] - long["team_qb_base"]).fillna(0.0)
    out = df.copy()
    for side in ("home", "away"):
        m = long[["game_id", "team", "qb_change"]].rename(
            columns={"team": f"{side}_team", "qb_change": f"{side}_qb_change"})
        out = out.merge(m, on=["game_id", f"{side}_team"], how="left")
    out["qb_change_diff"] = out["home_qb_change"] - out["away_qb_change"]
    return out


# ---------------------------------------------------------------- Elo
def elo_ratings(sched: pd.DataFrame) -> pd.DataFrame:
    elo: dict[str, float] = {}
    last_season: dict[str, int] = {}
    pre_h, pre_a = [], []
    for r in sched.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            if t not in elo:
                elo[t] = ELO_MEAN
            elif last_season.get(t) != r.season:
                elo[t] = ELO_MEAN * ELO_REVERT + elo[t] * (1 - ELO_REVERT)
            last_season[t] = r.season
        eh, ea = elo[r.home_team], elo[r.away_team]
        pre_h.append(eh)
        pre_a.append(ea)
        if not r.completed:
            continue
        hfa = 0.0 if r.location == "Neutral" else ELO_HFA
        exp_h = 1 / (1 + 10 ** (-(eh + hfa - ea) / 400))
        mov = r.home_score - r.away_score
        act = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
        wdiff = (eh + hfa - ea) if mov > 0 else (ea - eh - hfa)
        mult = np.log(abs(mov) + 1) * 2.2 / (wdiff * 0.001 + 2.2) if mov != 0 else 1.0
        delta = ELO_K * mult * (act - exp_h)
        elo[r.home_team] = eh + delta
        elo[r.away_team] = ea - delta
    out = sched[["game_id"]].copy()
    out["home_elo"] = pre_h
    out["away_elo"] = pre_a
    return out


# ---------------------------------------------------------------- assemble
def build_features(games: pd.DataFrame, pbp: pd.DataFrame, inj_data: tuple | None = None,
                   ngs_data: tuple | None = None) -> pd.DataFrame:
    """inj_data: optional (injuries, snap_counts, players); ngs_data: optional (passing, rushing, receiving)."""
    sched = prepare_schedule(games)
    tgs = team_game_stats(pbp)
    roll = rolling_team_features(sched, tgs)
    df = attach_qb(sched, qb_ratings(pbp, sched))
    df = attach_qb_change(df)
    df = df.merge(elo_ratings(sched), on="game_id")
    for side in ("home", "away"):
        r = roll.rename(columns={c: f"{side}_{c[4:]}" for c in roll.columns if c.startswith("pre_")})
        df = df.merge(r.rename(columns={"team": f"{side}_team"}), on=["game_id", f"{side}_team"], how="left")

    if inj_data is not None:
        from .injuries import team_injury_load
        il = team_injury_load(*inj_data, sched)
        for side in ("home", "away"):
            df = df.merge(il.rename(columns={"team": f"{side}_team", "inj_off": f"{side}_inj_off",
                                             "inj_def": f"{side}_inj_def", "inj_reported": f"{side}_inj_reported"}),
                          on=["game_id", f"{side}_team"], how="left")
    else:
        for side in ("home", "away"):
            df[[f"{side}_inj_off", f"{side}_inj_def", f"{side}_inj_reported"]] = 0.0
    # Positive = the AWAY team is missing more, i.e. favors home.
    df["inj_off_diff"] = df["away_inj_off"].fillna(0) - df["home_inj_off"].fillna(0)
    df["inj_def_diff"] = df["away_inj_def"].fillna(0) - df["home_inj_def"].fillna(0)

    if POWER_RATINGS:
        from .ratings import power_ratings
        df["margin"] = df["home_score"] - df["away_score"]
        net = roll_net = None
        tg = tgs.assign(net_epa=tgs["off_epa"] - tgs["def_epa"])[["game_id", "team", "net_epa"]]
        df = df.merge(tg.rename(columns={"team": "home_team", "net_epa": "home_net_epa_g"}), on=["game_id", "home_team"], how="left")
        df = df.merge(tg.rename(columns={"team": "away_team", "net_epa": "away_net_epa_g"}), on=["game_id", "away_team"], how="left")
        df["epa_margin"] = df["home_net_epa_g"] - df["away_net_epa_g"]
        for tgt in ("margin", "epa_margin"):
            df = df.merge(power_ratings(df, tgt), on="game_id", how="left")
        df["pr_pts_diff"] = (df["home_margin_rating"] - df["away_margin_rating"]).fillna(0)
        df["pr_epa_diff"] = (df["home_epa_margin_rating"] - df["away_epa_margin_rating"]).fillna(0)
    else:
        df["pr_pts_diff"] = 0.0
        df["pr_epa_diff"] = 0.0

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["qb_diff"] = df["home_qb_rating"] - df["away_qb_rating"]
    for s in ["off_epa", "off_sr", "pass_epa", "rush_epa", "to_margin", "pt_diff"]:
        df[f"{s}_diff"] = df[f"home_{s}"] - df[f"away_{s}"]
    # Defense: lower allowed is better, so flip sign so positive = home advantage.
    df["def_epa_diff"] = df["away_def_epa"] - df["home_def_epa"]
    df["def_sr_diff"] = df["away_def_sr"] - df["home_def_sr"]
    # Final regular-season week: teams with nothing to play for rest starters, so normal
    # strength edges matter less. The schedule is known in advance, so this is pre-game info.
    last_wk = df[df["game_type"] == "REG"].groupby("season")["week"].max()
    df["final_week"] = ((df["game_type"] == "REG") & (df["week"] == df["season"].map(last_wk))).astype(int)
    for f in ("elo_diff", "pt_diff_diff", "qb_diff"):
        df[f"fw_{f}"] = df["final_week"] * df[f].fillna(0)
    df["rest_diff"] = (df["home_rest"] - df["away_rest"]).clip(-7, 7)

    # Travel / time zones / body clock (schedule-derived, known in advance)
    from .travel import travel_features
    df = df.merge(travel_features(sched), on="game_id", how="left")

    # Weather: game-time temp/wind from the schedule (forecast for upcoming games, see weather.py).
    outdoor = df["roof"].isin(["outdoors", "open"])
    wind = df["wind"].where(outdoor, 0).fillna(0).clip(0, 30)
    temp = df["temp"].where(outdoor, 70).fillna(60)
    df["wind_pass"] = wind / 10 * df["pass_epa_diff"].fillna(0)   # wind shrinks passing edges
    warm = warm_weather_teams(sched)
    cold = (temp <= 35).astype(int)
    df["cold_edge"] = cold * (df["away_team"].map(warm).fillna(0) - df["home_team"].map(warm).fillna(0))

    if ngs_data is not None:
        from .ngs import ngs_features
        df = ngs_features(df, *ngs_data)
    else:
        df[["qb_cpoe_diff", "qb_ttt_diff", "ryoe_diff", "sep_diff"]] = 0.0
    df["home_field"] = (df["location"] != "Neutral").astype(int)
    df["div_game"] = df["div_game"].fillna(0).astype(int)

    df["home_win"] = np.where(df["completed"], (df["home_score"] > df["away_score"]).astype(float), np.nan)
    df.loc[df["completed"] & (df["home_score"] == df["away_score"]), "home_win"] = np.nan  # ties
    df["vegas_home_prob"] = vegas_prob(df["home_moneyline"], df["away_moneyline"])
    df[FEATURES] = df[FEATURES].fillna(0.0)
    return df


def warm_weather_teams(sched: pd.DataFrame) -> pd.Series:
    """1 for teams whose home stadium is a dome/closed roof or in a warm climate (lat < 34)."""
    from .travel import STADIUMS
    h = sched[sched["location"] != "Neutral"]
    last = h.sort_values("gameday").groupby("home_team").tail(8)
    roof = last.groupby("home_team")["roof"].agg(lambda r: r.value_counts().index[0])
    sid = last.groupby("home_team")["stadium_id"].agg(lambda r: r.value_counts().index[0])
    lat = sid.map(lambda x: STADIUMS.get(x, (40, 0, ""))[0])
    return (roof.isin(["dome", "closed"]) | (lat < 34)).astype(int)


def _implied(ml: pd.Series) -> pd.Series:
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def vegas_prob(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """No-vig home win probability from moneylines."""
    h, a = _implied(home_ml.astype(float)), _implied(away_ml.astype(float))
    return pd.Series(h / (h + a), index=home_ml.index)
