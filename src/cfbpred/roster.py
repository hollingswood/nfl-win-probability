"""College roster check for AI-read news (team verification).

The AI news reader often files a player under his old school (transfers: e.g. Dylan Raiola Nebraska -> Oregon in 2026)
or under the other team named in a matchup story. Before an item reaches the page its player is looked up in this
season's data:
  1. passing stats (data/cfb/raw/games_players_passing_<season>_*.json.gz): the team he actually played for, latest week
  2. CFBD roster (data/cfb/raw/roster_<season>.json.gz, refreshed weekly)
  3. transfer portal (data/cfb/raw/player_portal_<season>.json.gz): destination school
verify() returns {"roster": match | corrected | ambiguous | not_found, "team_verified": school or None}. The AI's own
fields are never changed; the page files the item under team_verified and hides items whose team can't be verified.
"""
from __future__ import annotations

import glob
import gzip
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "cfb" / "raw"
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")


def norm(name: str | None) -> str:
    s = _SUFFIX.sub("", (name or "").lower().replace("’", "'"))
    return " ".join(re.sub(r"[^a-z ]", "", s.replace("-", " ")).split())


def _wk(f: str) -> tuple:
    parts = Path(f).name.replace(".json.gz", "").split("_")
    return (parts[-2] != "regular", int(parts[-1]))          # regular weeks in order, then postseason


@lru_cache(maxsize=4)
def index(season: int, raw: str = str(RAW)) -> dict:
    """normalized name -> {"played": team or None, "roster": set(teams), "portal": team or None}."""
    idx: dict = {}
    ent = lambda n: idx.setdefault(n, {"played": None, "roster": set(), "portal": None})
    for f in sorted(glob.glob(f"{raw}/games_players_passing_{season}_*.json.gz"), key=_wk):
        try:
            games = json.load(gzip.open(f))
        except Exception:
            continue
        for g in games or []:
            for t in g.get("teams", []):
                for c in t.get("categories", []):
                    for ty in c.get("types", []):
                        for a in ty.get("athletes", []):
                            if a.get("name") and t.get("team"):
                                ent(norm(a["name"]))["played"] = t["team"]     # later weeks overwrite earlier
    p = Path(raw) / f"roster_{season}.json.gz"
    if p.exists():
        for x in json.load(gzip.open(p)) or []:
            n = norm(f"{x.get('firstName', '')} {x.get('lastName', '')}")
            if n and x.get("team"):
                ent(n)["roster"].add(x["team"])
    p = Path(raw) / f"player_portal_{season}.json.gz"
    if p.exists():
        for x in json.load(gzip.open(p)) or []:
            n = norm(f"{x.get('firstName', '')} {x.get('lastName', '')}")
            if n and x.get("destination"):
                ent(n)["portal"] = x["destination"]
    return idx


def verify(player: str | None, team: str | None, season: int, idx: dict | None = None) -> dict:
    if not player:
        return {"roster": "no_player", "team_verified": team}
    e = (idx if idx is not None else index(season)).get(norm(player))
    if not e:
        return {"roster": "not_found", "team_verified": None}
    teams = ({e["played"]} if e["played"] else set()) | e["roster"] | ({e["portal"]} if e["portal"] else set())
    if e["played"]:                       # he has thrown a pass this season: that team is the truth
        best = e["played"]
    elif len(e["roster"]) == 1:
        best = next(iter(e["roster"]))
    elif not e["roster"] and e["portal"]:
        best = e["portal"]
    else:
        best = None
    if team in teams and (best is None or best == team or len(e["roster"]) > 1):
        return {"roster": "match", "team_verified": team}
    if best:
        return {"roster": "match" if best == team else "corrected", "team_verified": best}
    return {"roster": "ambiguous", "team_verified": None}
