"""Travel distance, time-zone and body-clock features.

Each team's "home base" for a season is the stadium where it played most of its home games.
For every game we compute how far each team traveled, how many time zones it crossed, and
what time kickoff felt like on its body clock (schedule `gametime` is US Eastern).
All of this is known from the schedule before kickoff.
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# stadium_id -> (lat, lon, IANA time zone)
STADIUMS = {
    "ATL00": (33.758, -84.401, "America/New_York"), "ATL97": (33.755, -84.401, "America/New_York"),
    "BAL00": (39.278, -76.623, "America/New_York"), "BOS00": (42.091, -71.264, "America/New_York"),
    "BUF00": (42.774, -78.787, "America/New_York"), "BUF01": (43.641, -79.389, "America/Toronto"),
    "CAR00": (35.226, -80.853, "America/New_York"), "CHI98": (41.862, -87.617, "America/Chicago"),
    "CIN00": (39.095, -84.516, "America/New_York"), "CLE00": (41.506, -81.700, "America/New_York"),
    "DAL00": (32.748, -97.093, "America/Chicago"), "DEN00": (39.744, -105.020, "America/Denver"),
    "DET00": (42.340, -83.046, "America/Detroit"), "FRA00": (50.069, 8.645, "Europe/Berlin"),
    "GER00": (48.219, 11.625, "Europe/Berlin"), "GNB00": (44.501, -88.062, "America/Chicago"),
    "HOU00": (29.685, -95.411, "America/Chicago"), "IND00": (39.760, -86.164, "America/Indiana/Indianapolis"),
    "JAX00": (30.324, -81.637, "America/New_York"), "KAN00": (39.049, -94.484, "America/Chicago"),
    "LAX01": (33.953, -118.339, "America/Los_Angeles"), "LAX97": (33.864, -118.261, "America/Los_Angeles"),
    "LAX99": (34.014, -118.288, "America/Los_Angeles"), "LON00": (51.556, -0.280, "Europe/London"),
    "LON01": (51.456, -0.342, "Europe/London"), "LON02": (51.604, -0.066, "Europe/London"),
    "MAD01": (40.453, -3.688, "Europe/Madrid"), "MEL00": (-37.820, 144.983, "Australia/Melbourne"),
    "MEX00": (19.303, -99.150, "America/Mexico_City"), "MIA00": (25.958, -80.239, "America/New_York"),
    "MIN00": (44.974, -93.258, "America/Chicago"), "MIN01": (44.974, -93.258, "America/Chicago"),
    "MIN98": (44.977, -93.225, "America/Chicago"), "MUN01": (48.219, 11.625, "Europe/Berlin"),
    "NAS00": (36.166, -86.771, "America/Chicago"), "NOR00": (29.951, -90.081, "America/Chicago"),
    "NYC01": (40.814, -74.074, "America/New_York"), "OAK00": (37.752, -122.201, "America/Los_Angeles"),
    "PAR00": (48.924, 2.360, "Europe/Paris"), "PHI00": (39.901, -75.168, "America/New_York"),
    "PHO00": (33.528, -112.263, "America/Phoenix"), "PIT00": (40.447, -80.016, "America/New_York"),
    "RIO00": (-22.912, -43.230, "America/Sao_Paulo"), "SAO00": (-23.545, -46.474, "America/Sao_Paulo"),
    "SDG00": (32.783, -117.120, "America/Los_Angeles"), "SEA00": (47.595, -122.332, "America/Los_Angeles"),
    "SFO00": (37.714, -122.386, "America/Los_Angeles"), "SFO01": (37.403, -121.970, "America/Los_Angeles"),
    "STL00": (38.633, -90.189, "America/Chicago"), "TAM00": (27.976, -82.503, "America/New_York"),
    "VEG00": (36.091, -115.184, "America/Los_Angeles"), "WAS00": (38.908, -76.864, "America/New_York"),
}
ET = ZoneInfo("America/New_York")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _utc_offset_hours(tz: str, when: datetime) -> float:
    return ZoneInfo(tz).utcoffset(when.replace(tzinfo=None)).total_seconds() / 3600


def team_bases(sched: pd.DataFrame) -> pd.DataFrame:
    """(season, team) -> stadium_id of the team's usual home stadium that season."""
    h = sched[sched["location"] != "Neutral"]
    base = (h.groupby(["season", "home_team"])["stadium_id"]
            .agg(lambda s: s.value_counts().index[0]).reset_index()
            .rename(columns={"home_team": "team", "stadium_id": "base_id"}))
    return base


def travel_features(sched: pd.DataFrame) -> pd.DataFrame:
    base = team_bases(sched)
    out = sched[["game_id", "season", "gameday", "gametime", "stadium_id", "home_team", "away_team"]].copy()
    for side in ("home", "away"):
        out = out.merge(base.rename(columns={"team": f"{side}_team", "base_id": f"{side}_base"}),
                        on=["season", f"{side}_team"], how="left")
        # teams without a home game yet this season (rare): fall back to venue for home, prior season base
        out[f"{side}_base"] = out[f"{side}_base"].fillna(out["stadium_id"] if side == "home" else None)

    def row_feats(r):
        venue = STADIUMS.get(r.stadium_id)
        try:
            hh, mm = (int(x) for x in str(r.gametime).split(":"))
        except ValueError:
            hh, mm = 13, 0
        kick_et = datetime(r.gameday.year, r.gameday.month, r.gameday.day, hh, mm)
        et_off = _utc_offset_hours("America/New_York", kick_et)
        res = []
        for b in (r.home_base, r.away_base):
            base = STADIUMS.get(b)
            if venue is None or base is None:
                res += [0.0, 0.0, float(hh + mm / 60)]
                continue
            km = float(haversine_km(base[0], base[1], venue[0], venue[1]))
            tz_shift = _utc_offset_hours(venue[2], kick_et) - _utc_offset_hours(base[2], kick_et)
            body = hh + mm / 60 + (_utc_offset_hours(base[2], kick_et) - et_off)  # kickoff on body clock
            res += [km, tz_shift, body]
        return res

    vals = np.array([row_feats(r) for r in out.itertuples(index=False)])
    out[["home_km", "home_tz_shift", "home_body_hour", "away_km", "away_tz_shift", "away_body_hour"]] = vals
    # Positive values favor the home team.
    out["travel_diff"] = (out["away_km"] - out["home_km"]) / 1000.0
    out["tz_diff"] = out["away_tz_shift"].abs() - out["home_tz_shift"].abs()
    out["early_body_diff"] = (out["away_body_hour"] < 11).astype(int) - (out["home_body_hour"] < 11).astype(int)
    out["body_hour_diff"] = out["home_body_hour"] - out["away_body_hour"]
    return out[["game_id", "home_km", "away_km", "home_tz_shift", "away_tz_shift",
                "home_body_hour", "away_body_hour", "travel_diff", "tz_diff", "early_body_diff", "body_hour_diff"]]
