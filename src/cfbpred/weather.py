"""College game weather (Open-Meteo, free, no key).

* Venues: CFBD /venues (one call, data/cfb/raw/venues.json.gz), falling back to the FBS team-stadium list.
* History (research): for each completed game since 2021 at an open-air venue, the GFS forecast for the game
  window as it stood 1 and 2 days before (Open-Meteo Previous Runs API: `*_previous_day1/2`) plus the latest run
  (close to what actually happened). Saved to data/cfb/weather/game_weather.json, resumable, rate-limited
  (Open-Meteo free tier: 600 calls/min, 5,000/hour, 10,000/day). Using the forecast that existed at bet time keeps the
  test honest; recorded weather would leak (the NFL wind study's mistake, see output/research/totals.md).
* Live: kickoff-window forecast for this week's games, shown on the college page and logged to
  history/cfb/weather_log.jsonl (first time each forecast was seen) for a later honest test.
Display/research only: nothing here places a bet.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import data as D

ROOT = Path(__file__).resolve().parents[2]
WDIR = ROOT / "data" / "cfb" / "weather"
HIST = ROOT / "history" / "cfb"
PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FCST_URL = "https://api.open-meteo.com/v1/forecast"
VARS = ["wind_speed_10m", "wind_gusts_10m", "temperature_2m", "precipitation"]
# 10 variables: Open-Meteo counts more than 10 as extra calls
HOURLY_PREV = ",".join([f"{v}{s}" for v in ("wind_speed_10m", "wind_gusts_10m") for s in ("", "_previous_day1", "_previous_day2")]
                       + [f"{v}{s}" for v in ("temperature_2m", "precipitation") for s in ("", "_previous_day2")])
WINDOW_H = 3   # kickoff hour + the next 3 hours (a college game runs ~3.5 h)


def venues() -> dict[int, dict]:
    """venue_id -> {lat, lon, dome, elevation_m, tz, name}."""
    out: dict[int, dict] = {}
    for y in range(2014, datetime.now(timezone.utc).year + 1):
        for t in D._load(f"teams_fbs_{y}.json.gz"):
            loc = t.get("location") or {}
            if loc.get("id") and loc.get("latitude") is not None:
                out[int(loc["id"])] = {"lat": float(loc["latitude"]), "lon": float(loc["longitude"]), "dome": bool(loc.get("dome")),
                                       "elevation_m": float(loc["elevation"]) if loc.get("elevation") else None,
                                       "tz": loc.get("timezone"), "name": loc.get("name")}
    for v in D._load("venues.json.gz"):
        lat = v.get("latitude", (v.get("location") or {}).get("x"))
        lon = v.get("longitude", (v.get("location") or {}).get("y"))
        if v.get("id") is None or lat is None or lon is None:
            continue
        out.setdefault(int(v["id"]), {"lat": float(lat), "lon": float(lon), "dome": bool(v.get("dome")),
                                      "elevation_m": float(v["elevation"]) if v.get("elevation") else None,
                                      "tz": v.get("timezone"), "name": v.get("name")})
    return out


def pull_venues(key: str) -> int:
    from .fetch import RAW, get
    data = get("venues", {}, key)
    if data:
        RAW.mkdir(parents=True, exist_ok=True)
        with gzip.open(RAW / "venues.json.gz", "wt") as f:
            json.dump(data, f)
    return len(data or [])


def _tbd(season: int) -> set[int]:
    return {g["id"] for st in ("regular", "postseason") for g in D._load(f"games_{season}_{st}.json.gz") if g.get("startTimeTBD")}


def window(payload: dict, kickoff: datetime, suffixes=("", "_previous_day1", "_previous_day2")) -> dict:
    """Mean wind/temp, max gust and summed precip over the kickoff window, per forecast vintage. mph, °F, inches."""
    h = payload.get("hourly") or {}
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in h.get("time", [])]
    k0 = kickoff.replace(minute=0, second=0, microsecond=0)
    idx = [i for i, t in enumerate(times) if k0 <= t <= k0 + timedelta(hours=WINDOW_H)]
    out = {}
    for suf, tag in zip(suffixes, ("d0", "d1", "d2")):
        def vals(v):
            xs = h.get(v + suf) or []
            return [xs[i] for i in idx if i < len(xs) and xs[i] is not None]
        w, g, t, p = vals("wind_speed_10m"), vals("wind_gusts_10m"), vals("temperature_2m"), vals("precipitation")
        out[tag] = {"wind_mph": round(sum(w) / len(w), 1) if w else None, "gust_mph": round(max(g), 1) if g else None,
                    "temp_f": round(sum(t) / len(t), 1) if t else None, "precip_in": round(sum(p), 2) if p else None}
    return out


def _get(url: str, params: dict, timeout: float = 20) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout) as r:
        return json.load(r)


def backfill(seasons: list[int], max_calls: int = 4000, sleep: float = 0.15, deadline: datetime | None = None,
             getter=_get, checkpoint=None) -> dict:
    """Resumable; saves every 200 calls. Returns counts."""
    WDIR.mkdir(parents=True, exist_ok=True)
    path = WDIR / "game_weather.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    V = venues()
    calls, skipped, failed = 0, {"no_venue": 0, "dome": 0, "tbd": 0}, 0
    for s in seasons:
        G = D.games([s])
        G = G[G.completed & ((G.home_div == "fbs") | (G.away_div == "fbs"))]
        tbd = _tbd(s)
        for r in G.itertuples():
            gid = str(int(r.game_id))
            if gid in store:
                continue
            v = V.get(int(r.venue_id)) if r.venue_id == r.venue_id and r.venue_id is not None else None
            if v is None:
                skipped["no_venue"] += 1; continue
            if v["dome"]:
                store[gid] = {"dome": True}; skipped["dome"] += 1; continue
            if int(r.game_id) in tbd:
                skipped["tbd"] += 1; continue
            if calls >= max_calls or (deadline and datetime.now(timezone.utc) >= deadline):
                path.write_text(json.dumps(store))
                return {"calls": calls, "saved": len(store), "skipped": skipped, "failed": failed, "done": False}
            ko = datetime.fromisoformat(str(r.start)).astimezone(timezone.utc)
            end = ko + timedelta(hours=WINDOW_H + 1)
            hourly = HOURLY_PREV
            try:
                pay = getter(PREV_URL, {"latitude": v["lat"], "longitude": v["lon"], "hourly": hourly, "models": "gfs_seamless",
                                        "start_date": ko.date().isoformat(), "end_date": end.date().isoformat(), "timezone": "UTC",
                                        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"})
                store[gid] = {"dome": False, "kickoff": ko.isoformat(timespec="minutes"), **window(pay, ko)}
            except urllib.error.HTTPError as e:
                failed += 1
                if e.code == 429:            # over the hourly/daily limit: stop and resume next run
                    path.write_text(json.dumps(store))
                    return {"calls": calls, "saved": len(store), "skipped": skipped, "failed": failed, "done": False, "rate_limited": True}
            except Exception:
                failed += 1
            calls += 1
            if calls % 200 == 0:
                path.write_text(json.dumps(store))
                print(f"  {calls} calls, {len(store)} games saved, {failed} failed", flush=True)
            if checkpoint and calls % 500 == 0:
                checkpoint(f"{len(store)} games, {failed} failed calls")
            time.sleep(sleep)
    path.write_text(json.dumps(store))
    return {"calls": calls, "saved": len(store), "skipped": skipped, "failed": failed, "done": True}


def forecast(games: list[dict], now: datetime, getter=_get) -> dict[int, dict]:
    """game_id -> kickoff-window forecast for this week's open-air games (16-day horizon). Logs first sightings."""
    V = venues()
    gv = {}
    for s in {datetime.fromisoformat(g["start"]).year for g in games}:
        G = D.games([s])
        gv.update({int(r.game_id): r.venue_id for r in G.itertuples()})
    tbd = set().union(*(_tbd(s) for s in {datetime.fromisoformat(g["start"]).year for g in games})) if games else set()
    out = {}
    for g in games:
        vid = gv.get(g["game_id"])
        v = V.get(int(vid)) if vid == vid and vid is not None else None
        ko = datetime.fromisoformat(g["start"]).astimezone(timezone.utc)
        if v is None or g.get("completed") or ko - now > timedelta(days=15) or ko < now:
            continue
        if v["dome"]:
            out[g["game_id"]] = {"dome": True}; continue
        if g["game_id"] in tbd:
            out[g["game_id"]] = {"tbd": True}; continue
        try:
            pay = getter(FCST_URL, {"latitude": v["lat"], "longitude": v["lon"], "hourly": ",".join(VARS), "forecast_days": 16,
                                    "timezone": "UTC", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"})
            w = window(pay, ko, suffixes=("",))["d0"]
            out[g["game_id"]] = {"dome": False, **w, "hours_before": round((ko - now).total_seconds() / 3600, 1)}
        except Exception as e:
            print("cfb weather:", g["game_id"], str(e)[:80])
    if out:
        HIST.mkdir(parents=True, exist_ok=True)
        with (HIST / "weather_log.jsonl").open("a") as f:
            for gid, w in out.items():
                if not w.get("dome") and not w.get("tbd"):
                    f.write(json.dumps({"seen_at": now.isoformat(timespec="minutes"), "game_id": gid, **w}) + "\n")
    return out


def probe(seasons: list[int], getter=_get) -> dict:
    """One call per season on an open-air game: fails fast if the API errors or returns no older forecasts."""
    V, out = venues(), {}
    for s in seasons:
        G = D.games([s])
        G = G[G.completed & G.venue_id.notna()]
        r = next((r for r in G.itertuples() if V.get(int(r.venue_id), {}).get("dome") is False), None)
        if r is None:
            continue
        v, ko = V[int(r.venue_id)], datetime.fromisoformat(str(r.start)).astimezone(timezone.utc)
        pay = getter(PREV_URL, {"latitude": v["lat"], "longitude": v["lon"], "hourly": HOURLY_PREV, "models": "gfs_seamless",
                                "start_date": ko.date().isoformat(), "end_date": (ko + timedelta(hours=WINDOW_H + 1)).date().isoformat(),
                                "timezone": "UTC", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch"})
        out[s] = window(pay, ko)
    if not any((w.get("d2") or {}).get("wind_mph") is not None for w in out.values()):
        raise SystemExit(f"Previous Runs API returned no 2-day-old forecasts: {out}")
    return out


def git_checkpoint(msg: str) -> None:
    """In GitHub Actions: commit + push the weather file so progress is visible and survives a killed job."""
    import subprocess
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    sh = lambda c: subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True)
    sh('git config user.name "nfl-bot" && git config user.email "nfl-bot@users.noreply.github.com"')
    sh("git add data/cfb")
    if sh(f'git commit -q -m "CFB weather backfill: {msg}"').returncode == 0:
        for _ in range(3):
            if sh("git pull -q --rebase && git push -q").returncode == 0:
                break
            time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025,2026")
    ap.add_argument("--max-calls", type=int, default=4000)
    ap.add_argument("--minutes", type=float, default=330, help="stop (and save) after this many minutes")
    ap.add_argument("--hourly-chunks", action="store_true", help="loop: max-calls per hour until done or out of time")
    a = ap.parse_args()
    key = os.environ.get("CFBD_API_KEY")
    if key and not (D.RAW / "venues.json.gz").exists():
        print("venues:", pull_venues(key))
    seasons = [int(x) for x in a.seasons.split(",")]
    t0 = time.time()
    print("probe:", json.dumps(probe(seasons)), f"{(time.time() - t0) / len(seasons):.1f} s/call", flush=True)
    deadline = datetime.now(timezone.utc) + timedelta(minutes=a.minutes)
    while True:
        t0 = datetime.now(timezone.utc)
        t1 = time.time()
        rep = backfill(seasons, a.max_calls, deadline=deadline, checkpoint=git_checkpoint)
        rep["seconds_per_call"] = round((time.time() - t1) / max(1, rep["calls"]), 2)
        git_checkpoint(f"{rep['saved']} games ({rep['seconds_per_call']} s/call)")
        print(json.dumps(rep), flush=True)
        if rep["done"] or not a.hourly_chunks or datetime.now(timezone.utc) + timedelta(minutes=62) >= deadline:
            break
        time.sleep(max(0.0, 3720 - (datetime.now(timezone.utc) - t0).total_seconds()))


if __name__ == "__main__":
    main()
