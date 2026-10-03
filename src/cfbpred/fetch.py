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
PER_YEAR = ["talent", "player/returning", "recruiting/teams", "ratings/sp", "teams/fbs"]


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
            if out.exists() and not (refresh_current and y >= cur):
                continue
            params = {"year": y}
            if st:
                params["seasonType"] = st
            if path == "stats/game/advanced":
                params["excludeGarbageTime"] = "true"
            data = get(path, params, key)
            if data is None:
                continue
            with gzip.open(out, "wt") as f:
                json.dump(data, f)
            saved.append(out.name)
            print(f"{out.name}: {len(data) if isinstance(data, list) else 'ok'} records")
    return {"saved": len(saved), "calls": CALLS["n"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2014-2026")
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
    print(json.dumps(res))
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as f:
            f.write(f"### CFB data pull\n\n```\n{json.dumps(res, indent=2)}\n```\n")


if __name__ == "__main__":
    main()
