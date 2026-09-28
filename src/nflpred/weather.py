"""Kickoff weather forecast for upcoming outdoor games (Open-Meteo, free, no key).

Historical games use the temp/wind recorded in the nflverse schedule. For games that haven't
been played, this fetches the hourly forecast at the venue for the kickoff hour.
Forecasts are only available ~16 days out; beyond that, fields stay empty.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .travel import STADIUMS

URL = "https://api.open-meteo.com/v1/forecast"


def _kickoff_utc(gameday: pd.Timestamp, gametime: str) -> datetime:
    hh, mm = (int(x) for x in str(gametime or "13:00").split(":"))
    et = datetime(gameday.year, gameday.month, gameday.day, hh, mm, tzinfo=ZoneInfo("America/New_York"))
    return et.astimezone(ZoneInfo("UTC"))


def parse_forecast(payload: dict, kickoff_utc: datetime) -> dict:
    """Pick the forecast hour closest to kickoff. Units: °F, mph, % precip chance."""
    h = payload["hourly"]
    times = [datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("UTC")) for t in h["time"]]
    i = min(range(len(times)), key=lambda k: abs((times[k] - kickoff_utc).total_seconds()))
    return {"temp_f": h["temperature_2m"][i], "wind_mph": h["wind_speed_10m"][i],
            "gust_mph": h.get("wind_gusts_10m", [None] * len(times))[i],
            "precip_pct": h.get("precipitation_probability", [None] * len(times))[i]}


def fetch_forecast(stadium_id: str, gameday: pd.Timestamp, gametime: str, timeout: float = 15) -> dict | None:
    if stadium_id not in STADIUMS:
        return None
    lat, lon, _ = STADIUMS[stadium_id]
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "UTC", "forecast_days": 16,
        "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=timeout) as r:
        payload = json.load(r)
    return parse_forecast(payload, _kickoff_utc(gameday, gametime))


def forecasts_for(games: pd.DataFrame) -> dict[str, dict]:
    """game_id -> forecast for outdoor/open-roof games. Failures are skipped (logged)."""
    out = {}
    for g in games.itertuples():
        if g.roof not in ("outdoors", "open"):
            continue
        try:
            f = fetch_forecast(g.stadium_id, g.gameday, g.gametime)
            if f:
                out[g.game_id] = f
        except Exception as e:  # network blocked, API down: predictions still publish
            print(f"weather: {g.game_id}: {e}")
    return out
