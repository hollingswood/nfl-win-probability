"""Round 6: does the tone of a team's headlines in the week before a game predict the result beyond the closing line?

Pre-registered in output/research/round6/news_headlines_prereg.json (committed before any headline was scored).
Stages:
  probe  (Actions) check that Google News RSS honours after:/before: date ranges
  fetch  (Actions) for every 2023-2025 game and both teams: Google News RSS search "<team name> NFL" between
         kickoff-7d and kickoff -> data/research/news_headlines/<season>.jsonl.gz (title, source, published)
  score  (local) VADER compound score of each title; per team-game: mean, share negative, count, using only items
         published >= 2 h before kickoff
  test   (local) dev 2023, holdout 2024-25, as pre-registered
No Claude credits are used.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "research" / "news_headlines"
UA = {"User-Agent": "Mozilla/5.0 (research; nfl-win-probability)"}
NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
         "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
         "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
         "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
         "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
         "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
         "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def search(q: str, after: str, before: str) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f'"{q}" after:{after} before:{before}', "hl": "en-US", "gl": "US", "ceid": "US:en"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.iter("item"):
        pub = it.findtext("pubDate") or ""
        try:
            p = parsedate_to_datetime(pub).isoformat()
        except Exception:
            p = None
        out.append({"title": (it.findtext("title") or "").strip(), "source": it.findtext("source"), "published": p})
    return out


def games(seasons):
    from nflpred import data as D
    import pandas as pd
    g = D.load_schedules()
    g = g[g.season.isin(seasons) & g.home_score.notna()].copy()
    day = pd.to_datetime(g.gameday).dt.strftime("%Y-%m-%d")
    g["kick"] = pd.to_datetime(day + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g


def probe():
    res = {}
    for q, a, b in (("Kansas City Chiefs", "2024-10-01", "2024-10-08"), ("Detroit Lions", "2023-12-10", "2023-12-17")):
        items = search(q, a, b)
        pubs = sorted(i["published"] for i in items if i["published"])
        res[q] = {"n": len(items), "first": pubs[:1], "last": pubs[-1:], "sample": [i["title"] for i in items[:3]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def fetch(seasons, pause: float = 1.2):
    g = games(seasons)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        path = OUT / f"{s}.jsonl.gz"
        done = set()
        if path.exists():
            with gzip.open(path, "rt") as f:
                done = {(r["game_id"], r["team"]) for r in map(json.loads, f)}
        with gzip.open(path, "at") as f:
            for r in g[g.season == s].itertuples():
                for team in (r.home_team, r.away_team):
                    if (r.game_id, team) in done or team not in NAMES:
                        continue
                    a = (r.kick - timedelta(days=7)).strftime("%Y-%m-%d")
                    b = (r.kick + timedelta(days=1)).strftime("%Y-%m-%d")
                    for attempt in range(4):
                        try:
                            items = search(NAMES[team], a, b)
                            break
                        except Exception as e:
                            print("retry", r.game_id, team, e)
                            time.sleep(10 * (attempt + 1))
                    else:
                        items = None
                    if items is not None:
                        f.write(json.dumps({"game_id": r.game_id, "team": team, "kick": r.kick.isoformat(), "items": items}) + "\n")
                        f.flush()
                    time.sleep(pause)
        print("season", s, "done")


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "probe":
        probe()
    elif stage == "fetch":
        fetch([int(x) for x in sys.argv[2].split(",")])
