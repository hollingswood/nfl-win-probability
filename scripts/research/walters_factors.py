"""Out-of-sample test of Billy Walters' published, FIXED NFL game factors ("Gambler", values as of the end
of the 2022-23 season). Nothing here is fit to the data: the factor table is implemented as published
(interpretations documented in INTERPRETATIONS below), so 2023-2025 is a genuine out-of-sample test.
Thresholds (|H| >= 1 / 1.5 / 2 points) were specified before looking at results; every one is reported.

Each factor unit = 0.20 points added to that side's power rating.  H = net factor points, home minus visitor.

Tests
  1. distribution of H (2003-2025)
  2. does H predict (home margin - closing spread_line)?  slope by period + ATS at the close (-110)
  3. does H predict line movement (early-week consensus -> price-implied close), 2020-2025 snapshots;
     bet early at the best allowed-book price on the H side: price-based CLV + ROI
  4. does H (or components) improve our walk-forward model (mu_model) or the closing line on 2023-25?
  5. per-factor residual signal, 2003-22 vs 2023-25 (multiple-comparison caution)

    cd /home/claude/nfl && EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts \
        python scripts/research/walters_factors.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from nflpred import features as F                                   # noqa: E402
from nflpred.travel import STADIUMS, team_bases, travel_features   # noqa: E402

OUT = ROOT / "output" / "research"
SCRATCH = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/walters")
U = 0.20                       # points per factor unit
MILES_2000_KM = 2000 * 1.609344
NIGHT_HOUR = 19.0              # kickoff >= 7:00 pm ET counts as a night game (all prime-time slots are 8:15-8:30)

INTERPRETATIONS = [
    "Team codes normalised to franchises (OAK->LV, SD->LAC, STL->LA); each team's home stadium/base per season = "
    "its most-used non-neutral home venue (travel.team_bases); STADIUMS lat/lon/tz for distances and zones.",
    "Turf: team's home-turf type per season = majority of its non-neutral home games, grass/dessograss = grass, "
    "everything else artificial (blank surface -> nearest season of that team). Same type +1 visitor, opposite +1 home.",
    "Prime-time slots: Thursday/Sunday/Monday with kickoff >= 7pm ET (Thanksgiving afternoon games are not TNF). "
    "'Coming off MNF' requires the MNF game to be the team's previous game within 7 days (not across a bye).",
    "Overtime: previous game (same season) went to OT and was <= 10 days ago (a bye in between cancels it); "
    "'home' OT game = team was the non-neutral home team.",
    "3rd away game in four: the visitor's current game plus its previous 3 games this season contain >= 3 "
    "non-home games (away or neutral).",
    "Bye: regular-season game, week > 1, >= 13 days since the team's previous game this season. Quality = tercile of "
    "the team's pre-game Elo (src features.elo_ratings parameters, re-run to rank vs all teams' current Elo at that date): "
    "bottom third 'below-average' 4 (5 away), middle 'average' 5 (6), top 'great' 7 (8). Both teams can get it.",
    "Playoff bye: Divisional-round team whose previous game was a regular-season game -> +1 home (bye teams host). "
    "Regular bye units are not applied in the playoffs; the 2-week Super Bowl gap is not a bye.",
    "Super Bowl: previous season's SB winner gets +4 in its 1st game and +2 in games 2-4 of the next season; "
    "SB loser's opponent gets the same.",
    "Travel: visitor's base-to-venue great-circle distance >= 2000 miles -> +1 home. Short-trip visitor bonuses "
    "(non-neutral games only): TB/JAX/MIA +1, DAL/HOU +1, ATL/CAR +1, IND/CIN +1, CHI/GB +1, LA metro pair "
    "(LA/LAC) +2 and Bay Area pair (SF/OAK, through 2019) +2 as the 'SF/LA area' analogue, LV vs LA/LAC +1, any "
    "pair of PHI/NYG/NYJ/WAS/NE/BAL/BUF +1 except NYG/NYJ +2 and BAL/WAS +2. Pairs keyed on bases, so e.g. SD-LA "
    "or OAK-LA pre-2020 are not bonused.",
    "Zones from base tz: East = New_York/Detroit/Indianapolis/Toronto, Central = Chicago, Mountain = Denver/Phoenix "
    "(ARI as Mountain), Pacific = Los_Angeles (incl. LV).",
    "10 a.m. games: kickoff before 2pm ET at a venue in the Eastern/Central zone (incl. Mexico City) for a "
    "Pacific-base team -> +2 opponent, Mountain-base team -> +1 opponent.",
    "Night games (kickoff >= 7pm ET): penalties East 6, Central 3, Mountain 1, Pacific 0; net units = "
    "penalty(eastern team) - penalty(western team) to the more western team.",
    "Second consecutive game >= 2 time zones from home: visitor's |tz shift| >= 2 in this game AND in its previous "
    "game this season -> +2 home.",
    "Bounce back: previous game (same season, any rest) lost by >= 19 -> +2, >= 29 -> +4 (not cumulative).",
    "Warm-weather team = base latitude < 34.5 or Las Vegas (MIA TB JAX HOU NO ATL DAL ARI LA LAC SD LV); "
    "cold-climate dome team = home roof mostly dome/closed/retractable and not warm (DET, MIN except 2014-15, "
    "IND, STL). Only the VISITOR can be 'visiting'; cold OUTDOOR = roof outdoors/open with nflverse temp "
    "(recorded at game time, i.e. not strictly ex-ante). Warm: <=35F .25, <=30 .50, <=25 .75, <=20 1.00, <=15 1.25, "
    "<=10 1.75 to home. Dome: 20<t<=30 .25, 10<t<=20 .50, t<=10 .75 to home.",
    "Rain from play-by-play `weather` text (2012+ only; 2003-11 rain = 0): description before 'Temp:' mentions "
    "rain/showers/drizzle and not 'chance/possible/no rain/%' -> +0.25 visitor; heavy/hard/downpour/pouring -> +0.75. "
    "Outdoor/open roof only. Recorded conditions, not forecast.",
    "W values: main H treats them as W units (x 0.2 pt); H_wpts treats them as points (reported side by side).",
    "Skipped: 'Variable' items (matchups, snow, heavy wind), all E-factors (subjective) and QB-specific W-factors.",
    "Neutral-site games keep the schedule's designated home team for all factors.",
]

AFC = {"BAL", "BUF", "CIN", "CLE", "DEN", "HOU", "IND", "JAX", "KC", "LV", "LAC", "MIA", "NE", "NYJ", "PIT", "TEN"}
EAST_TZ = {"America/New_York", "America/Detroit", "America/Indiana/Indianapolis", "America/Toronto"}
ZONE = {**{z: "E" for z in EAST_TZ}, "America/Chicago": "C", "America/Mexico_City": "C",
        "America/Denver": "M", "America/Phoenix": "M", "America/Los_Angeles": "P"}
NIGHT_PEN = {"E": 6, "C": 3, "M": 1, "P": 0}
LA_BASES, BAY_BASES = {"LAX01", "LAX97", "LAX99"}, {"OAK00", "SFO00", "SFO01"}
EAST_GROUP = {"PHI", "NYG", "NYJ", "WAS", "NE", "BAL", "BUF"}
SHORT_PAIRS = {frozenset(p): v for p, v in [
    (("TB", "JAX"), 1), (("TB", "MIA"), 1), (("JAX", "MIA"), 1), (("DAL", "HOU"), 1), (("ATL", "CAR"), 1),
    (("IND", "CIN"), 1), (("CHI", "GB"), 1)]}

COMPONENTS = ["turf", "division", "conference", "home_tnf", "home_snf", "home_mnf", "home_off_mnf",
              "away_off_mnf", "three_away", "off_ot", "bye", "playoff_bye", "super_bowl", "travel_2000",
              "short_trip", "tz_10am", "night_tz", "two_tz", "bounce_back", "w_warm_cold", "w_dome_cold", "w_rain"]
W_COMPONENTS = ["w_warm_cold", "w_dome_cold", "w_rain"]


# ============================================================================ schedule + context
def load_schedule() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    s = F.prepare_schedule(g, min_season=1999)
    s = s[s.completed].reset_index(drop=True)
    hm = s.gametime.astype(str).str.split(":", expand=True)
    s["hour"] = pd.to_numeric(hm[0], errors="coerce").fillna(13) + pd.to_numeric(hm[1], errors="coerce").fillna(0) / 60
    s["overtime"] = s.overtime.fillna(0).astype(int)
    s["result"] = s.home_score - s.away_score
    return s


def elo_terciles(s: pd.DataFrame) -> pd.DataFrame:
    """Pre-game Elo (same update rule/params as features.elo_ratings) and each team's percentile rank vs all
    active teams' CURRENT ratings at that date (no future information)."""
    elo, last = {}, {}
    out = []

    def cur(t, season):
        e = elo[t]
        return e if last[t] == season else F.ELO_MEAN * F.ELO_REVERT + e * (1 - F.ELO_REVERT)

    for r in s.itertuples(index=False):
        for t in (r.home_team, r.away_team):
            if t not in elo:
                elo[t] = F.ELO_MEAN
            elif last.get(t) != r.season:
                elo[t] = F.ELO_MEAN * F.ELO_REVERT + elo[t] * (1 - F.ELO_REVERT)
            last[t] = r.season
        active = [cur(t, r.season) for t in elo if last[t] >= r.season - 1]
        eh, ea = elo[r.home_team], elo[r.away_team]
        rk = lambda e: float(np.mean([x < e for x in active]) + 0.5 * np.mean([x == e for x in active]))
        out.append((r.game_id, eh, ea, rk(eh), rk(ea)))
        hfa = 0.0 if r.location == "Neutral" else F.ELO_HFA
        exp_h = 1 / (1 + 10 ** (-(eh + hfa - ea) / 400))
        mov = r.home_score - r.away_score
        if F.PT_CAP is not None:
            mov = float(np.clip(mov, -F.PT_CAP, F.PT_CAP))
        act = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
        wdiff = (eh + hfa - ea) if mov > 0 else (ea - eh - hfa)
        mult = np.log(abs(mov) + 1) * 2.2 / (wdiff * 0.001 + 2.2) if mov != 0 else 1.0
        d = F.ELO_K * mult * (act - exp_h)
        elo[r.home_team], elo[r.away_team] = eh + d, ea - d
    return pd.DataFrame(out, columns=["game_id", "home_elo", "away_elo", "home_elo_pct", "away_elo_pct"])


def team_season_context(s: pd.DataFrame) -> pd.DataFrame:
    """(season, team) -> base stadium, base zone, home-turf type, dome flag, warm flag."""
    base = team_bases(s)
    h = s[s.location != "Neutral"].copy()
    h["turf"] = np.where(h.surface.astype(str).str.strip().isin(["grass", "dessograss"]), "grass",
                         np.where(h.surface.astype(str).str.strip() == "", None, "art"))
    turf = h.dropna(subset=["turf"]).groupby(["season", "home_team"]).turf.agg(lambda x: x.value_counts().index[0])
    dome = h.groupby(["season", "home_team"]).roof.agg(lambda x: float(x.isin(["dome", "closed", "open"]).mean() >= 0.5))
    ctx = base.set_index(["season", "team"])
    ctx["turf"] = turf.rename_axis(["season", "team"])
    ctx["dome"] = dome.rename_axis(["season", "team"])
    ctx = ctx.reset_index().sort_values(["team", "season"])
    ctx["turf"] = ctx.groupby("team").turf.transform(lambda x: x.ffill().bfill())
    ctx["lat"] = ctx.base_id.map(lambda b: STADIUMS.get(b, (40, 0, ""))[0])
    ctx["zone"] = ctx.base_id.map(lambda b: ZONE.get(STADIUMS.get(b, (0, 0, "America/New_York"))[2], "E"))
    ctx["warm"] = ((ctx.lat < 34.5) | (ctx.base_id == "VEG00")).astype(int)
    ctx["cold_dome"] = ((ctx.dome == 1) & (ctx.warm == 0)).astype(int)
    return ctx


def rain_levels() -> pd.Series:
    cache = SCRATCH / "rain.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("game_id").rain
    rows = []
    for yr in range(2012, 2026):
        p = ROOT / "data" / "raw" / f"pbp_{yr}.parquet"
        if not p.exists():
            continue
        w = pd.read_parquet(p, columns=["game_id", "weather"]).drop_duplicates("game_id")
        rows.append(w)
    w = pd.concat(rows)

    def lvl(txt):
        if not isinstance(txt, str):
            return 0.0
        d = txt.split("Temp:")[0].lower()
        if not re.search(r"rain|shower|drizzle", d):
            return 0.0
        if re.search(r"chance|possible|no rain|%|percent|likely|threat", d):
            return 0.0
        return 0.75 if re.search(r"heavy|hard|downpour|pouring|torrential", d) else 0.25

    w["rain"] = w.weather.map(lvl)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    w[["game_id", "rain"]].to_parquet(cache)
    return w.set_index("game_id").rain


def build_factors() -> pd.DataFrame:
    cache = SCRATCH / "factors.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    s = load_schedule()
    tr = travel_features(s)
    s = s.merge(tr, on="game_id", how="left").merge(elo_terciles(s), on="game_id", how="left")
    ctx = team_season_context(s)
    for side in ("home", "away"):
        c = ctx.rename(columns={"team": f"{side}_team"}).rename(
            columns={k: f"{side}_{k}" for k in ("base_id", "turf", "dome", "lat", "zone", "warm", "cold_dome")})
        s = s.merge(c[["season", f"{side}_team", f"{side}_base_id", f"{side}_turf", f"{side}_zone", f"{side}_warm",
                       f"{side}_cold_dome"]], on=["season", f"{side}_team"], how="left")
    s["venue_zone"] = s.stadium_id.map(lambda b: ZONE.get(STADIUMS.get(b, (0, 0, "Europe/London"))[2], "X"))
    s["rain"] = s.game_id.map(rain_levels()).fillna(0.0)
    s["outdoor"] = s.roof.isin(["outdoors", "open"])

    # ---- long (team-game) table for "coming off" factors
    keep = ["game_id", "season", "week", "game_type", "gameday", "weekday", "hour", "location", "overtime"]
    lh = s[keep + ["home_team", "result", "home_tz_shift"]].rename(
        columns={"home_team": "team", "home_tz_shift": "tz"}).assign(side="home")
    la = s[keep + ["away_team", "result", "away_tz_shift"]].rename(
        columns={"away_team": "team", "away_tz_shift": "tz"}).assign(side="away")
    la["result"] = -la.result
    L = pd.concat([lh, la]).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    L["is_home"] = (L.side == "home") & (L.location != "Neutral")
    L["away_flag"] = (~L.is_home).astype(int)
    L["mnf"] = (L.weekday == "Monday") & (L.hour >= NIGHT_HOUR)
    g = L.groupby(["team", "season"])
    for c in ("gameday", "mnf", "is_home", "overtime", "result", "tz", "game_type"):
        L[f"prev_{c}"] = g[c].shift(1)
    L["days_rest"] = (L.gameday - L.prev_gameday).dt.days
    L["game_no"] = g.cumcount() + 1
    L["nonhome4"] = g.away_flag.transform(lambda x: x.rolling(4, min_periods=1).sum())
    has_prev = L.prev_gameday.notna()
    L["off_mnf"] = has_prev & (L.prev_mnf == True) & (L.days_rest <= 7)          # noqa: E712
    L["off_mnf_home"] = L.off_mnf & (L.prev_is_home == True)                # noqa: E712
    L["off_ot"] = has_prev & (L.prev_overtime == 1) & (L.days_rest <= 10)
    L["off_ot_home"] = L.off_ot & (L.prev_is_home == True)                  # noqa: E712
    L["off_bye"] = has_prev & (L.days_rest >= 13) & (L.game_type == "REG") & (L.week > 1)
    L["playoff_bye"] = (L.game_type == "DIV") & (L.prev_game_type == "REG")
    L["bounce"] = np.where(L.prev_result <= -29, 4, np.where(L.prev_result <= -19, 2, 0))
    L["two_tz"] = has_prev & (L.tz.abs() >= 2) & (L.prev_tz.abs() >= 2)
    # Super Bowl carry-over
    sb = s[s.game_type == "SB"][["season", "home_team", "away_team", "result"]]
    sbw = {r.season + 1: (r.home_team if r.result > 0 else r.away_team) for r in sb.itertuples()}
    sbl = {r.season + 1: (r.away_team if r.result > 0 else r.home_team) for r in sb.itertuples()}
    L["sb_win"] = [(sbw.get(se) == t) for se, t in zip(L.season, L.team)]
    L["sb_lose"] = [(sbl.get(se) == t) for se, t in zip(L.season, L.team)]
    L["sb_units"] = np.where(L.game_no == 1, 4, np.where(L.game_no <= 4, 2, 0))
    lcols = ["off_mnf", "off_mnf_home", "off_ot", "off_ot_home", "off_bye", "playoff_bye", "bounce", "two_tz",
             "sb_win", "sb_lose", "sb_units", "nonhome4", "days_rest"]
    for side in ("home", "away"):
        m = L[L.side == side][["game_id"] + lcols].rename(columns={c: f"{side}_{c}" for c in lcols})
        s = s.merge(m, on="game_id", how="left")

    f = pd.DataFrame({"game_id": s.game_id})
    hz, az = s.home_zone, s.away_zone
    # turf / alignment
    f["turf"] = np.where(s.home_turf == s.away_turf, -1, 1)
    f["division"] = -s.div_game.fillna(0).astype(int)
    f["conference"] = (s.home_team.isin(AFC) != s.away_team.isin(AFC)).astype(int)
    # schedule
    f["home_tnf"] = 2 * ((s.weekday == "Thursday") & (s.hour >= NIGHT_HOUR))
    f["home_snf"] = 4 * ((s.weekday == "Sunday") & (s.hour >= NIGHT_HOUR))
    f["home_mnf"] = 2 * ((s.weekday == "Monday") & (s.hour >= NIGHT_HOUR))
    f["home_off_mnf"] = -4 * (s.home_off_mnf & ~s.home_off_mnf_home)
    f["away_off_mnf"] = np.where(s.away_off_mnf, np.where(s.away_off_mnf_home, 6, 8), 0)
    f["three_away"] = 2 * (s.away_nonhome4 >= 3)
    f["off_ot"] = (np.where(s.home_off_ot, -np.where(s.home_off_ot_home, 4, 2), 0)
                   + np.where(s.away_off_ot, np.where(s.away_off_ot_home, 4, 2), 0))

    def bye_units(pct, away):
        u = np.where(pct < 1 / 3, 4, np.where(pct > 2 / 3, 7, 5))
        return u + int(away)
    f["bye"] = (np.where(s.home_off_bye, bye_units(s.home_elo_pct, False), 0)
                - np.where(s.away_off_bye, bye_units(s.away_elo_pct, True), 0))
    f["playoff_bye"] = (s.home_playoff_bye & ~s.away_playoff_bye).astype(int)
    f["super_bowl"] = (np.where(s.home_sb_win, s.home_sb_units, 0) - np.where(s.away_sb_win, s.away_sb_units, 0)
                       - np.where(s.home_sb_lose, s.home_sb_units, 0) + np.where(s.away_sb_lose, s.away_sb_units, 0))
    # travel
    f["travel_2000"] = (s.away_km >= MILES_2000_KM).astype(int)

    def short(r):
        if r.location == "Neutral":
            return 0
        hb, ab = r.home_base_id, r.away_base_id
        if hb in LA_BASES and ab in LA_BASES:
            return 2
        if hb in BAY_BASES and ab in BAY_BASES:
            return 2
        if {hb, ab} & {"VEG00"} and ({hb, ab} & LA_BASES):
            return 1
        pair = frozenset((r.home_team, r.away_team))
        if pair <= EAST_GROUP:
            return 2 if pair in (frozenset(("NYG", "NYJ")), frozenset(("BAL", "WAS"))) else 1
        return SHORT_PAIRS.get(pair, 0)
    f["short_trip"] = -np.array([short(r) for r in s.itertuples(index=False)])
    # time zones
    early = (s.hour < 14) & s.venue_zone.isin(["E", "C"])
    pen10 = {"P": 2, "M": 1}
    f["tz_10am"] = (np.where(early, az.map(pen10).fillna(0), 0) - np.where(early, hz.map(pen10).fillna(0), 0))
    night = s.hour >= NIGHT_HOUR
    f["night_tz"] = np.where(night, az.map(NIGHT_PEN) - hz.map(NIGHT_PEN), 0)
    f["two_tz"] = 2 * s.away_two_tz.fillna(False).astype(bool)
    f["bounce_back"] = s.home_bounce.fillna(0) - s.away_bounce.fillna(0)
    # W factors (in W units)
    t = s.temp
    cold = s.outdoor & t.notna()
    warm_w = np.select([t <= 10, t <= 15, t <= 20, t <= 25, t <= 30, t <= 35], [1.75, 1.25, 1.0, 0.75, 0.5, 0.25], 0)
    dome_w = np.select([t <= 10, t <= 20, t <= 30], [0.75, 0.5, 0.25], 0)
    f["w_warm_cold"] = np.where(cold & (s.away_warm == 1), warm_w, 0.0)
    f["w_dome_cold"] = np.where(cold & (s.away_cold_dome == 1), dome_w, 0.0)
    f["w_rain"] = -np.where(s.outdoor, s.rain, 0.0)

    f = f.astype({c: float for c in COMPONENTS})
    s_units = f[[c for c in COMPONENTS if c not in W_COMPONENTS]].sum(axis=1)
    w_units = f[W_COMPONENTS].sum(axis=1)
    f["H"] = U * (s_units + w_units)
    f["H_wpts"] = U * s_units + w_units
    meta = s[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "result", "spread_line",
              "location", "weekday", "hour", "home_zone", "away_zone", "venue_zone"]]
    out = meta.merge(f, on="game_id")
    out.to_parquet(cache)
    return out


# ============================================================================ stats helpers
def ols(y, X):
    """OLS with intercept; returns (coef, HC1 se) excluding intercept."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    meat = (X * e[:, None] ** 2).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
    return b[1:], np.sqrt(np.diag(V))[1:]


def slope_row(d, ycol="resid", xcol="H"):
    d = d[d[ycol].notna() & d[xcol].notna()]
    if len(d) < 30 or d[xcol].std() == 0:
        return {"n": int(len(d))}
    b, se = ols(d[ycol], d[[xcol]])
    return {"n": int(len(d)), "slope": round(float(b[0]), 3), "se": round(float(se[0]), 3),
            "t": round(float(b[0] / se[0]), 2)}


def ats(d, thr, hcol="H"):
    b = d[d[hcol].abs() >= thr]
    adj = np.sign(b[hcol]) * b.resid
    n_all, pushes = len(b), int((adj == 0).sum())
    w, l = int((adj > 0).sum()), int((adj < 0).sum())
    if w + l == 0:
        return {"bets": n_all}
    p = w / (w + l)
    pnl = np.where(adj > 0, 100 / 110, np.where(adj < 0, -1.0, 0.0))
    return {"bets": n_all, "per_season": round(n_all / b.season.nunique(), 1), "w": w, "l": l, "push": pushes,
            "cover": round(p, 3), "cover_se": round(math.sqrt(p * (1 - p) / (w + l)), 3),
            "z_vs_524": round((p - 0.5238) / math.sqrt(0.5238 * 0.4762 / (w + l)), 2),
            "roi": round(float(pnl.mean()), 3), "roi_se": round(float(pnl.std(ddof=1) / math.sqrt(n_all)), 3)}


def mse_compare(y, base, alt):
    d1, d2 = (y - base) ** 2, (y - alt) ** 2
    diff = d2 - d1
    return {"n": int(len(y)), "mse_base": round(float(d1.mean()), 2), "mse_alt": round(float(d2.mean()), 2),
            "delta": round(float(diff.mean()), 3), "delta_se": round(float(diff.std(ddof=1) / math.sqrt(len(y))), 3)}


PERIODS = {"2003-2022 (Walters in-sample)": (2003, 2022), "2003-2012": (2003, 2012), "2013-2022": (2013, 2022),
           "2020-2022": (2020, 2022), "2023-2025 (out-of-sample)": (2023, 2025)}


def per(d, lo, hi):
    return d[d.season.between(lo, hi)]


# ============================================================================ tests
def test1(d):
    out = {}
    for lab, (lo, hi) in {"2003-2022": (2003, 2022), "2023-2025": (2023, 2025)}.items():
        x = per(d, lo, hi)
        r = {"games": int(len(x))}
        for h in ("H", "H_wpts"):
            v = x[h]
            r[h] = {"mean": round(float(v.mean()), 3), "sd": round(float(v.std()), 3),
                    "p05": round(float(v.quantile(.05)), 2), "p50": round(float(v.median()), 2),
                    "p95": round(float(v.quantile(.95)), 2), "min": round(float(v.min()), 2), "max": round(float(v.max()), 2),
                    "share_abs_ge_0.5": round(float((v.abs() >= .5).mean()), 3),
                    "share_abs_ge_1": round(float((v.abs() >= 1).mean()), 3),
                    "share_abs_ge_1.5": round(float((v.abs() >= 1.5).mean()), 3),
                    "share_abs_ge_2": round(float((v.abs() >= 2).mean()), 3)}
        r["active_share"] = {c: round(float((x[c] != 0).mean()), 3) for c in COMPONENTS}
        out[lab] = r
    return out


def test2(d, close):
    out = {"slope_resid_on_H": {}, "slope_resid_on_H_wpts": {}, "raw_margin_on_H": {}, "H_given_line": {},
           "ats": {}, "ats_wpts": {}, "price_close_2020_25": {}}
    for lab, (lo, hi) in PERIODS.items():
        x = per(d, lo, hi)
        out["slope_resid_on_H"][lab] = slope_row(x, "resid", "H")
        out["slope_resid_on_H_wpts"][lab] = slope_row(x, "resid", "H_wpts")
        out["raw_margin_on_H"][lab] = slope_row(x, "result", "H")
        out.setdefault("slope_resid_on_H_ex_night", {})[lab] = slope_row(x.assign(Hx=x.H - U * x.night_tz), "resid", "Hx")
        b, se = ols(x.result, x[["spread_line", "H"]])
        out["H_given_line"][lab] = {"b_line": round(float(b[0]), 3), "b_H": round(float(b[1]), 3),
                                    "se_H": round(float(se[1]), 3)}
    for lab, (lo, hi) in {"2003-2022": (2003, 2022), "2023-2025": (2023, 2025)}.items():
        x = per(d, lo, hi)
        out["ats"][lab] = {str(t): ats(x, t) for t in (0.5, 1.0, 1.5, 2.0)}
        out["ats_wpts"][lab] = {str(t): ats(x, t, "H_wpts") for t in (1.0, 1.5, 2.0)}
    c = d.merge(close[["game_id", "mu_close_all"]], on="game_id")
    c["resid_px"] = c.result - c.mu_close_all
    for lab, (lo, hi) in {"2020-2022": (2020, 2022), "2023-2025": (2023, 2025)}.items():
        out["price_close_2020_25"][lab] = slope_row(per(c, lo, hi), "resid_px", "H")
    return out


def test3(d):
    import edge_lab as E
    import line_move_model as LM
    out = {}
    for lab, hold, seasons in (("2020-2022", False, E.DEV), ("2023-2025", True, E.HOLD)):
        t = E.load(hold)
        s, off = LM.build(t, seasons)
        first = s.sort_values("requested_ts").groupby("game_id").head(1)
        first = first.merge(d[["game_id", "H", "H_wpts", "spread_line"]], on="game_id")
        first["move_px"] = first.mu_close_all - first.mu_px_all
        first["move_raw"] = first.spread_line - first.m_cons
        r = {"games": int(len(first)), "median_hours_before_first": round(float(first.hours_before.median()), 1),
             "move_px_on_H": slope_row(first, "move_px", "H"), "move_raw_on_H": slope_row(first, "move_raw", "H"),
             "move_px_on_H_wpts": slope_row(first, "move_px", "H_wpts")}
        # bets: first snapshot, best allowed-book price on the H side
        o = off[off.sp_ok].merge(first[["game_id", "requested_ts", "H"]], on=["game_id", "requested_ts"])
        o = o[o.sp_clv.notna()]
        best = o.sort_values("sp_ev_now", ascending=False).groupby(["game_id", "side"]).head(1)
        base = {"bets": int(len(best)), "clv": round(float(best.sp_clv.mean()), 4),
                "roi": round(float(best.sp_pnl.mean()), 4)}
        r["baseline_both_sides_best_price"] = base
        r["bet_H_side_early"] = {str(thr): side_bets(best, best.H, thr) for thr in (0.5, 1.0, 1.5, 2.0)}
        out[lab] = r
        EARLY[lab] = (first, best)
    return out


EARLY: dict = {}


def side_bets(best, signal, thr):
    """Bet the side `signal` (home-signed) favours where |signal| >= thr, at the best allowed-book price at the
    first snapshot. 'excess' = CLV of that side minus the mean CLV of both sides' best prices in the same game
    (removes the vig/shopping level, isolating direction); its t is the directional test."""
    sig = pd.Series(np.asarray(signal, float), index=best.index)
    pair = best.groupby("game_id").sp_clv.transform("mean")
    n_sides = best.groupby("game_id").sp_clv.transform("size")
    pick = (sig.abs() >= thr) & (np.where(best.side == "home", 1, -1) * np.sign(sig) > 0) & (n_sides == 2)
    b = best[pick]
    if len(b) < 3:
        return {"bets": int(len(b))}
    clv, pnl, exc = b.sp_clv.values, b.sp_pnl.values, (b.sp_clv - pair[pick]).values
    tt = lambda v: round(float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))), 2)
    return {"bets": int(len(b)), "clv": round(float(clv.mean()), 4), "clv_t": tt(clv),
            "excess_clv": round(float(exc.mean()), 4), "excess_t": tt(exc),
            "beat_close": round(float((clv > 0).mean()), 3), "roi": round(float(pnl.mean()), 4),
            "roi_se": round(float(pnl.std(ddof=1) / math.sqrt(len(pnl))), 4)}


def test6(d):
    """POST-HOC follow-up on night_tz, the strongest in-sample factor (chosen by its 2003-22 t, then checked
    2023-25). Robustness cuts are descriptive, not selected on."""
    x = d[d.night_tz != 0].copy()
    x["west_home"] = x.night_tz > 0
    x["units"] = x.night_tz.abs()
    x["toward"] = np.sign(x.night_tz) * x.resid
    x["west_team"] = np.where(x.west_home, x.home_team, x.away_team)

    def cell(v):
        v = np.asarray(v, float)
        if len(v) < 5:
            return {"n": int(len(v))}
        p = (v > 0).sum() / max((v != 0).sum(), 1)
        return {"n": int(len(v)), "resid_toward": round(float(v.mean()), 2),
                "se": round(float(v.std(ddof=1) / math.sqrt(len(v))), 2), "cover": round(float(p), 3)}
    out = {"by_period": {}, "cuts_2003_22": {}, "cuts_2023_25": {}, "by_season_2020_25": {}}
    for lab, (lo, hi) in PERIODS.items():
        out["by_period"][lab] = cell(per(x, lo, hi).toward)
    for key, (lo, hi) in (("cuts_2003_22", (2003, 2022)), ("cuts_2023_25", (2023, 2025))):
        y = per(x, lo, hi)
        out[key] = {"west_home": cell(y[y.west_home].toward), "west_away": cell(y[~y.west_home].toward),
                    "kick_ge_20:00": cell(y[y.hour >= 20].toward),
                    "units<=2": cell(y[y.units <= 2].toward), "units_3": cell(y[y.units == 3].toward),
                    "units>=5": cell(y[y.units >= 5].toward),
                    "excl_SEA": cell(y[(y.home_team != "SEA") & (y.away_team != "SEA")].toward),
                    "regular_season": cell(y[y.game_type == "REG"].toward)}
    for se in range(2020, 2026):
        out["by_season_2020_25"][str(se)] = cell(x[x.season == se].toward)
    # early-line view: does the market move toward the western team, and does betting it early earn CLV?
    out["early"] = {}
    for lab, (first, best) in EARLY.items():
        f = first.merge(d[["game_id", "night_tz"]], on="game_id")
        f["night_pts"] = f.night_tz * U
        ff = f[f.night_tz != 0]
        mv = (np.sign(ff.night_tz) * ff.move_px)
        b = best.merge(d[["game_id", "night_tz"]], on="game_id")
        out["early"][lab] = {"games": int(len(ff)),
                             "move_toward_west_pts": round(float(mv.mean()), 3),
                             "move_se": round(float(mv.std(ddof=1) / math.sqrt(len(mv))), 3),
                             "bet_west_early": side_bets(b, b.night_tz, 1)}
    return out


def test4(d, close):
    rp = pd.read_parquet(ROOT / "data" / "replay_predictions.parquet")[["game_id", "mu_model"]]
    x = d.merge(rp, on="game_id").merge(close[["game_id", "mu_close_all"]], on="game_id", how="left")
    x["resid_model"] = x.result - x.mu_model
    out = {"model_resid_on_H": {}, "model_resid_components_2023_25": {}}
    for lab, (lo, hi) in {"2020-2022": (2020, 2022), "2023-2025": (2023, 2025)}.items():
        out["model_resid_on_H"][lab] = slope_row(per(x, lo, hi), "resid_model", "H")
    tr, te = per(x, 2020, 2022), per(x, 2023, 2025)
    # fixed (Walters) weight: add H as published
    out["fixed_add_H_to_model_2023_25"] = mse_compare(te.result.values, te.mu_model.values, (te.mu_model + te.H).values)
    out["fixed_add_H_to_close_2023_25"] = mse_compare(te.result.values, te.spread_line.values, (te.spread_line + te.H).values)
    full = per(d, 2023, 2025)
    out["fixed_add_H_to_close_2023_25_all_games"] = mse_compare(full.result.values, full.spread_line.values,
                                                                 (full.spread_line + full.H).values)
    # fitted blends (fit 2020-22 for the model; 2003-22 for the closing line) -> evaluate 2023-25
    def fit_pred(train, test, cols):
        b, _ = ols(train.result, train[cols])
        X = np.column_stack([np.ones(len(train))] + [train[c] for c in cols])
        a = np.linalg.lstsq(X, train.result.values, rcond=None)[0]
        return a[0] + test[cols].values @ a[1:], a
    p0, a0 = fit_pred(tr, te, ["mu_model"])
    p1, a1 = fit_pred(tr, te, ["mu_model", "H"])
    out["fitted_model_plus_H_2023_25"] = {**mse_compare(te.result.values, p0, p1), "coef_H": round(float(a1[-1]), 3)}
    q0, _ = fit_pred(tr, te, ["mu_model", "spread_line"])
    q1, a2 = fit_pred(tr, te, ["mu_model", "spread_line", "H"])
    out["fitted_model_line_plus_H_2023_25"] = {**mse_compare(te.result.values, q0, q1), "coef_H": round(float(a2[-1]), 3)}
    trc = per(d, 2003, 2022)
    r0, _ = fit_pred(trc, full, ["spread_line"])
    r1, a3 = fit_pred(trc, full, ["spread_line", "H"])
    out["fitted_close_plus_H_2023_25"] = {**mse_compare(full.result.values, r0, r1), "coef_H": round(float(a3[-1]), 3)}
    comps = [c for c in COMPONENTS if (te[c] != 0).sum() >= 5]
    r0c, _ = fit_pred(trc, full, ["spread_line"])
    trc_ok = [c for c in comps if trc[c].std() > 0]
    r1c, a4 = fit_pred(trc, full, ["spread_line"] + [c for c in trc_ok])
    out["fitted_close_plus_all_components_2023_25"] = mse_compare(full.result.values, r0c, r1c)
    for c in comps:
        out["model_resid_components_2023_25"][c] = slope_row(te.assign(**{c: te[c] * U}), "resid_model", c)
    return out


def test5(d):
    out = {}
    for c in COMPONENTS:
        pts = U if c in W_COMPONENTS else U          # main interpretation: everything in 0.2-pt units
        row = {}
        for lab, (lo, hi) in {"2003-2022": (2003, 2022), "2023-2025": (2023, 2025)}.items():
            x = per(d, lo, hi)
            a = x[x[c] != 0]
            if len(a) < 5:
                row[lab] = {"n": int(len(a))}
                continue
            sgn = np.sign(a[c])
            v = sgn * a.resid
            row[lab] = {"n": int(len(a)), "walters_pts": round(float((a[c].abs() * pts).mean()), 2),
                        "resid_toward": round(float(v.mean()), 2), "se": round(float(v.std(ddof=1) / math.sqrt(len(a))), 2),
                        "t": round(float(v.mean() / (v.std(ddof=1) / math.sqrt(len(a)))), 2),
                        "cover": round(float((v > 0).sum() / max((v != 0).sum(), 1)), 3),
                        "raw_margin_toward": round(float((sgn * a.result).mean()), 2),
                        "line_toward": round(float((sgn * a.spread_line).mean()), 2)}
        out[c] = row
    return out


# ============================================================================ report
def fmt_slope(r):
    if "slope" not in r:
        return f"n={r.get('n')}"
    return f"{r['slope']:+.2f} ± {r['se']:.2f} (t {r['t']:+.1f}, n {r['n']})"


def report(res) -> str:
    L = ["# Walters 'Gambler' game factors: fixed-table out-of-sample test", "",
         "Published factor values (end of 2022-23) applied without fitting; 1 unit = 0.20 pt; "
         "H = home minus visitor factor points. Residual = home margin − closing `spread_line` (nflverse). "
         "2023-2025 is out-of-sample for the published table. All thresholds pre-specified; all reported.", "",
         "## Interpretations", ""] + [f"- {x}" for x in INTERPRETATIONS] + [""]
    t1 = res["test1"]
    L += ["## 1. Distribution of H", "", "| period | games | mean | sd | p05 | p95 | |H|≥0.5 | |H|≥1 | |H|≥1.5 | |H|≥2 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in t1.items():
        for h in ("H", "H_wpts"):
            q = r[h]
            L.append(f"| {lab} {h} | {r['games']} | {q['mean']:+.2f} | {q['sd']:.2f} | {q['p05']:+.2f} | {q['p95']:+.2f} | "
                     f"{q['share_abs_ge_0.5']:.1%} | {q['share_abs_ge_1']:.1%} | {q['share_abs_ge_1.5']:.1%} | {q['share_abs_ge_2']:.1%} |")
    L += ["", "Share of games where each component is non-zero (2003-22 / 2023-25): " + ", ".join(
        f"{c} {t1['2003-2022']['active_share'][c]:.0%}/{t1['2023-2025']['active_share'][c]:.0%}" for c in COMPONENTS), ""]
    t2 = res["test2"]
    L += ["## 2. Does H predict the result beyond the closing line?", "",
          "Slope of residual (margin − spread_line) on H: 1 = market ignores the factors, 0 = fully priced.", "",
          "| period | resid ~ H | resid ~ H_wpts | resid ~ H without night_tz | margin ~ H (no line) | b_H in margin ~ line + H |",
          "|---|---|---|---|---|---|"]
    for lab in PERIODS:
        g = t2["H_given_line"][lab]
        L.append(f"| {lab} | {fmt_slope(t2['slope_resid_on_H'][lab])} | {fmt_slope(t2['slope_resid_on_H_wpts'][lab])} | "
                 f"{fmt_slope(t2['slope_resid_on_H_ex_night'][lab])} | {fmt_slope(t2['raw_margin_on_H'][lab])} | {g['b_H']:+.2f} ± {g['se_H']:.2f} |")
    L += ["", "Against the price-implied close (`edge_lab.closing_fair` mu_close_all): " + "; ".join(
        f"{k}: {fmt_slope(v)}" for k, v in t2["price_close_2020_25"].items()), "",
          "ATS at the closing spread_line, betting the side H favours (−110; break-even 52.4%):", "",
          "| period | |H|≥ | bets | /season | W-L-P | cover ± SE | z vs 52.4% | ROI ± SE |", "|---|---|---|---|---|---|---|---|"]
    for key, nm in (("ats", "H"), ("ats_wpts", "H_wpts")):
        for lab, rows in t2[key].items():
            for thr, a in rows.items():
                if "cover" not in a:
                    continue
                L.append(f"| {lab} ({nm}) | {thr} | {a['bets']} | {a['per_season']} | {a['w']}-{a['l']}-{a['push']} | "
                         f"{a['cover']:.3f} ± {a['cover_se']:.3f} | {a['z_vs_524']:+.2f} | {a['roi']:+.3f} ± {a['roi_se']:.3f} |")
    t3 = res["test3"]
    L += ["", "## 3. Line movement (early-week consensus → price-implied close), 2020-2025", "",
          "First snapshot ≤ 9 days before kickoff (≥ 3 h). move_px = mu_close_all − price-implied consensus at that snapshot "
          "(home-signed); move_raw = spread_line − consensus point.", "",
          "| period | games | median h before | move_px ~ H | move_raw ~ H | move_px ~ H_wpts |", "|---|---|---|---|---|---|"]
    for lab, r in t3.items():
        L.append(f"| {lab} | {r['games']} | {r['median_hours_before_first']} | {fmt_slope(r['move_px_on_H'])} | "
                 f"{fmt_slope(r['move_raw_on_H'])} | {fmt_slope(r['move_px_on_H_wpts'])} |")
    L += ["", "Bet the H side at the first snapshot, best allowed-book price (price-based CLV vs honest close):", "",
          "Excess CLV = H-side CLV minus the mean CLV of both sides' best prices in the same game (strips the "
          "~3% vig/shopping level; its t is the directional test).", "",
          "| period | |H|≥ | bets | CLV (t) | excess CLV (t) | beat close | ROI ± SE |", "|---|---|---|---|---|---|---|"]
    for lab, r in t3.items():
        b = r["baseline_both_sides_best_price"]
        L.append(f"| {lab} | baseline: both sides every game | {b['bets']} | {b['clv']:+.4f} | 0 | | {b['roi']:+.4f} |")
        for thr, a in r["bet_H_side_early"].items():
            if "clv" in a:
                L.append(f"| {lab} | {thr} | {a['bets']} | {a['clv']:+.4f} ({a['clv_t']:+.1f}) | "
                         f"{a['excess_clv']:+.4f} ({a['excess_t']:+.1f}) | {a['beat_close']:.3f} | "
                         f"{a['roi']:+.4f} ± {a['roi_se']:.4f} |")
    t4 = res["test4"]
    L += ["", "## 4. Does H improve our model or the closing line (2023-25)?", "",
          f"- residual (margin − mu_model) on H: " + "; ".join(f"{k}: {fmt_slope(v)}" for k, v in t4["model_resid_on_H"].items())]
    for k in ("fixed_add_H_to_model_2023_25", "fixed_add_H_to_close_2023_25", "fixed_add_H_to_close_2023_25_all_games",
              "fitted_model_plus_H_2023_25", "fitted_model_line_plus_H_2023_25", "fitted_close_plus_H_2023_25",
              "fitted_close_plus_all_components_2023_25"):
        v = t4[k]
        extra = f", coef_H {v['coef_H']:+.2f}" if "coef_H" in v else ""
        L.append(f"- {k}: MSE {v['mse_base']} → {v['mse_alt']} (Δ {v['delta']:+.3f} ± {v['delta_se']:.3f}, n {v['n']}{extra})")
    L += ["", "Model residual on each component (points), 2023-25: " + "; ".join(
        f"{c} {fmt_slope(v)}" for c, v in t4["model_resid_components_2023_25"].items()), ""]
    t5 = res["test5"]
    L += ["## 5. Individual factors", "",
          "For games where the factor is active: Walters' average size (pts) vs the realised residual toward the favoured "
          "side (margin − spread_line), and the line's own lean toward that side. ~22 factors × 2 periods → "
          "treat |t| < 3 as noise (Bonferroni 5% ≈ |t| 3.0).", "",
          "| factor | 03-22 n | Walters pts | resid toward ± SE (t) | line toward | 23-25 n | resid toward ± SE (t) | line toward |",
          "|---|---|---|---|---|---|---|---|"]
    for c, r in t5.items():
        a, b = r.get("2003-2022", {}), r.get("2023-2025", {})
        fa = f"{a['resid_toward']:+.2f} ± {a['se']:.2f} ({a['t']:+.1f})" if "se" in a else "–"
        fb = f"{b['resid_toward']:+.2f} ± {b['se']:.2f} ({b['t']:+.1f})" if "se" in b else "–"
        L.append(f"| {c} | {a.get('n', 0)} | {a.get('walters_pts', b.get('walters_pts', '–'))} | {fa} | {a.get('line_toward', '–')} | "
                 f"{b.get('n', 0)} | {fb} | {b.get('line_toward', '–')} |")
    t6 = res["test6_night_tz_posthoc"]

    def c6(c):
        return f"{c['resid_toward']:+.2f} ± {c['se']:.2f} (n {c['n']}, cover {c['cover']:.3f})" if "se" in c else f"n {c['n']}"
    L += ["", "## 6. POST-HOC: night-game time-zone factor (strongest in-sample factor)", "",
          "Selected because it had the largest 2003-22 t; 2023-25 is the check. Cuts below are descriptive. "
          "'toward' = residual (margin − spread_line) in favour of the more western team.", "",
          "| period | resid toward west ± SE |", "|---|---|"]
    L += [f"| {k} | {c6(v)} |" for k, v in t6["by_period"].items()]
    L += ["", "| cut | 2003-22 | 2023-25 |", "|---|---|---|"]
    L += [f"| {k} | {c6(t6['cuts_2003_22'][k])} | {c6(t6['cuts_2023_25'][k])} |" for k in t6["cuts_2003_22"]]
    L += ["", "By season: " + "; ".join(f"{k}: {c6(v)}" for k, v in t6["by_season_2020_25"].items()), ""]
    for lab, e in t6["early"].items():
        b = e["bet_west_early"]
        bt = (f"CLV {b['clv']:+.4f} (t {b['clv_t']:+.1f}), excess {b['excess_clv']:+.4f} (t {b['excess_t']:+.1f}), "
              f"ROI {b['roi']:+.3f} ± {b['roi_se']:.3f}, n {b['bets']}") if "clv" in b else f"n {b['bets']}"
        L.append(f"- {lab}: early→close move toward west {e['move_toward_west_pts']:+.3f} ± {e['move_se']:.3f} pts "
                 f"(n {e['games']}); bet west early at best price: {bt}")
    L += ["", "## Verdict", "", res.get("verdict", ""), ""]
    return "\n".join(L)


def main():
    import edge_lab as E
    d = build_factors()
    d = d[d.season.between(2003, 2025) & d.spread_line.notna() & d.result.notna()].copy()
    d["resid"] = d.result - d.spread_line
    close = E.closing_fair()
    res = {"interpretations": INTERPRETATIONS, "test1": test1(d), "test2": test2(d, close)}
    res["test3"] = test3(d)
    res["test4"] = test4(d, close)
    res["test5"] = test5(d)
    res["test6_night_tz_posthoc"] = test6(d)
    vfile = SCRATCH / "verdict.md"
    res["verdict"] = vfile.read_text() if vfile.exists() else "(see final summary)"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "walters_factors.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "walters_factors.md").write_text(report(res))
    print(report(res))


if __name__ == "__main__":
    main()
