"""Tidy college football tables from data/cfb/raw (see fetch.py)."""
from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "cfb" / "raw"


def _load(name: str) -> list:
    p = RAW / name
    if not p.exists():
        return []
    with gzip.open(p, "rt") as f:
        return json.load(f) or []


def games(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for st in ("regular", "postseason"):
            for g in _load(f"games_{y}_{st}.json.gz"):
                rows.append({"game_id": g["id"], "season": g["season"], "week": g["week"], "season_type": g["seasonType"],
                             "start": g.get("startDate"), "neutral": bool(g.get("neutralSite")),
                             "conf_game": bool(g.get("conferenceGame")), "completed": bool(g.get("completed")),
                             "home": g["homeTeam"], "away": g["awayTeam"], "home_conf": g.get("homeConference"),
                             "away_conf": g.get("awayConference"), "home_div": g.get("homeClassification"),
                             "away_div": g.get("awayClassification"), "home_pts": g.get("homePoints"),
                             "away_pts": g.get("awayPoints"), "home_elo": g.get("homePregameElo"),
                             "away_elo": g.get("awayPregameElo"), "venue_id": g.get("venueId")})
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
    df["margin"] = df["home_pts"] - df["away_pts"]
    df["total"] = df["home_pts"] + df["away_pts"]
    return df


def lines(years) -> pd.DataFrame:
    """One row per game: median across providers of closing spread/total (home line: + = home underdog),
    opening spread/total where any provider reports one, and per-provider counts."""
    rows = []
    for y in years:
        for st in ("regular", "postseason"):
            for g in _load(f"lines_{y}_{st}.json.gz"):
                L = g.get("lines") or []
                med = lambda k: statistics.median([x[k] for x in L if x.get(k) is not None]) if any(x.get(k) is not None for x in L) else None
                rows.append({"game_id": g["id"], "spread_close": med("spread"), "spread_open": med("spreadOpen"),
                             "total_close": med("overUnder"), "total_open": med("overUnderOpen"),
                             "n_providers": len(L), "providers": ",".join(sorted({x["provider"] for x in L})),
                             "home_ml": med("homeMoneyline"), "away_ml": med("awayMoneyline")})
    return pd.DataFrame(rows)


def adv_stats(years) -> pd.DataFrame:
    rows = []
    for y in years:
        for st in ("regular", "postseason"):
            for r in _load(f"stats_game_advanced_{y}_{st}.json.gz"):
                o, d = r.get("offense") or {}, r.get("defense") or {}
                rows.append({"game_id": r["gameId"], "season": r["season"], "team": r["team"], "opponent": r["opponent"],
                             "off_ppa": o.get("ppa"), "off_sr": o.get("successRate"), "off_expl": o.get("explosiveness"),
                             "off_plays": o.get("plays"), "off_pass_ppa": (o.get("passingPlays") or {}).get("ppa"),
                             "off_rush_ppa": (o.get("rushingPlays") or {}).get("ppa"),
                             "def_ppa": d.get("ppa"), "def_sr": d.get("successRate"), "def_plays": d.get("plays")})
    return pd.DataFrame(rows)


def yearly(name: str, years) -> pd.DataFrame:
    out = []
    for y in years:
        for r in _load(f"{name}_{y}.json.gz"):
            out.append({"year_req": y, **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}})
    return pd.DataFrame(out)
