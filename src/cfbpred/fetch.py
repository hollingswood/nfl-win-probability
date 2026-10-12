"""Pull college football data from CollegeFootballData.com (CFBD) into data/cfb/raw/.

Needs env CFBD_API_KEY (free tier: 1,000 calls/month; a full 2014-2026 pull is ~160 calls).
Each response is saved as-is (gzipped JSON) so parsing can change without re-downloading:
  data/cfb/raw/<endpoint>_<year>[_<seasonType>].json.gz
Already-downloaded past seasons are skipped; the current season is always refreshed.

Endpoints: games, lines (opening + closing spread/total per provider), advanced game stats
(EPA/success rate etc.), team talent, returning production, recruiting rankings, SP+ ratings,
FBS team list.
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
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "cfb" / "raw"
API = "https://api.collegefootballdata.com"
CALLS = {"n": 0}

PER_TYPE = ["games", "lines", "stats/game/advanced"]          # regular + postseason
PER_YEAR = ["talent", "player/returning", "recruiting/teams", "ratings/sp", "teams/fbs", "player/portal", "ppa/players/season"]


def get(path: str, params: dict, key: str, tries: int = 3):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                CALLS["n"] += 1
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"CFBD rejected the key ({e.code}); check the CFBD_API_KEY secret")
            if e.code == 429 or e.code >= 500:
                time.sleep(5 * (i + 1))
                continue
            print(f"  {path} {params}: HTTP {e.code}")
            return None
        except Exception as e:
            print(f"  {path} {params}: {e}")
            time.sleep(3)
    return None


def _name(path: str, year: int, st: str | None) -> Path:
    base = path.replace("/", "_")
    return RAW / (f"{base}_{year}_{st}.json.gz" if st else f"{base}_{year}.json.gz")


def pull(years: list[int], key: str, refresh_current: bool = True) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    cur = date.today().year if date.today().month >= 7 else date.today().year - 1
    saved = []
    for y in years:
        jobs = [(p, st) for p in PER_TYPE for st in ("regular", "postseason")] + [(p, None) for p in PER_YEAR]
        for path, st in jobs:
            out = _name(path, y, st)
            if out.exists() and not (refresh_current and y >= cur and st):   # yearly tables don't change in-season
                continue
            params = {"year": y}
            if st:
                params["seasonType"] = st
            if path in ("stats/game/advanced", "ppa/players/season"):
                params["excludeGarbageTime"] = "true"
            data = get(path, params, key)
            if data is None:
                continue
            with gzip.open(out, "wt") as f:
                json.dump(data, f)
            saved.append(out.name)
            print(f"{out.name}: {len(data) if isinstance(data, list) else 'ok'} records")
    return {"saved": len(saved), "calls": CALLS["n"]}


def pull_passing(years: list[int], key: str) -> dict:
    """Per-game passing stats (who actually played QB) -> games_players_passing_<year>_<type>_<week>.json.gz.
    One call per week; finished weeks are not re-downloaded. Used to find starting-QB changes (news-timing study)."""
    RAW.mkdir(parents=True, exist_ok=True)
    saved = 0
    for y in years:
        for st, weeks in (("regular", range(1, 17)), ("postseason", range(1, 2))):
            for wk in weeks:
                out = RAW / f"games_players_passing_{y}_{st}_{wk}.json.gz"
                if out.exists():
                    continue
                data = get("games/players", {"year": y, "week": wk, "seasonType": st, "category": "passing"}, key)
                if not data:
                    continue
                with gzip.open(out, "wt") as f:
                    json.dump(data, f)
                saved += 1
    return {"passing_weeks_saved": saved, "calls": CALLS["n"]}


def pull_havoc(years: list[int], key: str) -> dict:
    """Per-game havoc (research, round 4): CFBD stats/game/havoc per season (one call each); if that endpoint is not
    available, per-week team box scores (games/teams: tackles for loss, sacks, interceptions, passes deflected)."""
    RAW.mkdir(parents=True, exist_ok=True)
    out = {"havoc_files": 0, "teambox_files": 0}
    for y in years:
        for st in ("regular", "postseason"):
            p = RAW / f"stats_game_havoc_{y}_{st}.json.gz"
            if p.exists():
                continue
            data = get("stats/game/havoc", {"year": y, "seasonType": st}, key)
            if data:
                with gzip.open(p, "wt") as f:
                    json.dump(data, f)
                out["havoc_files"] += 1
                continue
            for wk in (range(1, 17) if st == "regular" else range(1, 2)):
                q = RAW / f"games_teams_{y}_{st}_{wk}.json.gz"
                if q.exists():
                    continue
                d2 = get("games/teams", {"year": y, "week": wk, "seasonType": st}, key)
                if d2:
                    with gzip.open(q, "wt") as f:
                        json.dump(d2, f)
                    out["teambox_files"] += 1
    out["calls"] = CALLS["n"]
    return out


def pull_current(season: int, key: str, now=None, roster_max_age_days: int = 7) -> dict:
    """In-season extras for the news roster check (cfbpred.roster): the CFBD roster (weekly) and passing stats for
    every finished regular-season week not saved yet (one call each)."""
    import time as _t
    from datetime import datetime, timedelta, timezone
    now = now or datetime.now(timezone.utc)
    RAW.mkdir(parents=True, exist_ok=True)
    out, n0 = {"roster": False, "passing_weeks": []}, CALLS["n"]
    p = RAW / f"roster_{season}.json.gz"
    if not p.exists() or _t.time() - p.stat().st_mtime > roster_max_age_days * 86400:
        data = get("roster", {"year": season}, key)
        if data:
            with gzip.open(p, "wt") as f:
                json.dump(data, f)
            out["roster"] = len(data)
    gp = _name("games", season, "regular")
    if gp.exists():
        last: dict = {}
        for g in json.load(gzip.open(gp)):
            if g.get("week") and g.get("startDate"):
                t = datetime.fromisoformat(g["startDate"].replace("Z", "+00:00"))
                last[g["week"]] = max(last.get(g["week"], t), t)
        for wk in sorted(w for w, t in last.items() if t + timedelta(hours=8) < now):
            f = RAW / f"games_players_passing_{season}_regular_{wk}.json.gz"
            if f.exists():
                continue
            data = get("games/players", {"year": season, "week": wk, "seasonType": "regular", "category": "passing"}, key)
            if data:
                with gzip.open(f, "wt") as fh:
                    json.dump(data, fh)
                out["passing_weeks"].append(wk)
    out["calls"] = CALLS["n"] - n0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2014-2026")
    ap.add_argument("--passing", action="store_true", help="also pull per-game passing stats (QB starters)")
    ap.add_argument("--havoc", action="store_true", help="also pull per-game havoc (research)")
    a = ap.parse_args(argv)
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise SystemExit("CFBD_API_KEY not set")
    if "-" in a.years:
        lo, hi = (int(x) for x in a.years.split("-"))
        years = list(range(lo, hi + 1))
    else:
        years = [int(x) for x in a.years.split(",")]
    res = pull(years, key)
    if a.passing:
        res.update(pull_passing([y for y in years if y >= 2021], key))
    if a.havoc:
        res.update(pull_havoc(years, key))
    print(json.dumps(res))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### CFB data pull\n\n```\n{json.dumps(res, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
