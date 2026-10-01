"""Night-game body-clock paper track (night_west_rules.json).

In night games (kickoff 7pm ET or later) between teams based in different time zones, back the
more western team's spread just before kickoff. Research (output/research/walters_factors.md):
the western team beat the closing spread by ~2 points in 2003-22 and again in 2023-25. Graded on
results (binomial test vs break-even), since the line does not move toward this factor.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import bets as ML
from . import spread_bets as SB

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "night_west_rules.json"
ORDER = {"E": 0, "C": 1, "M": 2, "P": 3}


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def zone_of(team: str, r: dict) -> str | None:
    for z, teams in r["qualify"]["zones"].items():
        if team in teams:
            return z
    return None


def evaluate(game: dict, r: dict, now: datetime | None = None) -> dict | None:
    q = r["qualify"]
    reasons = game["night_west_check"] = []
    kick = datetime.fromisoformat(game["kickoff_utc"]) if game.get("kickoff_utc") else None
    if kick is None:
        reasons.append("no kickoff time")
        return None
    if kick.astimezone(ZoneInfo("America/New_York")).hour < q["night_kickoff_et_hour_at_least"]:
        return None  # not a night game: not this track's business
    zh, za = zone_of(game["home_team"], r), zone_of(game["away_team"], r)
    if zh is None or za is None or zh == za:
        return None
    side = "home" if ORDER[zh] > ORDER[za] else "away"
    lo = (game.get("context") or {}).get("live_odds") or {}
    books = lo.get("spreads_by_book") or []
    game["night_west"] = {"side": side, "team": game[f"{side}_team"],
                          "zones": f"{game['away_team']} {za} @ {game['home_team']} {zh}"}
    now = now or datetime.now(timezone.utc)
    hours = (kick - now).total_seconds() / 3600
    if hours > q["bet_within_hours_of_kickoff"] or hours <= 0:
        reasons.append(f"bets only in the last {q['bet_within_hours_of_kickoff']} h before kickoff")
    if not books or lo.get("consensus_home_margin") is None:
        reasons.append("no live spreads")
        return None
    sr = SB.load_rules()
    mu = lo["consensus_home_margin"]
    best = None
    for b in books:
        point, price = b[f"{side}_point"], b[f"{side}_price"]
        if not (q["min_american_odds"] <= price <= q["max_american_odds"]):
            continue
        ev = SB.side_ev(mu, point, price, side, sr)[0]
        if best is None or ev > best["ev"]:
            best = {"point": point, "price": price, "book": b["book"], "ev": round(ev, 4)}
    if best is None:
        reasons.append("no price in allowed range")
        return None
    game["night_west"].update(best)
    if reasons:
        return None
    return {"id": f"{game['game_id']}:night_west:{side}", "track": "night_west", "rules_version": r["version"],
            "placed_at": now.isoformat(timespec="minutes"), "game_id": game["game_id"], "season": game["season"],
            "week": game["week"], "gameday": game["gameday"], "side": side, "team": game[f"{side}_team"],
            "opponent": game["away_team" if side == "home" else "home_team"], "point": best["point"],
            "price": best["price"], "book": best["book"], "edge": best["ev"], "units": r["sizing"]["units"],
            "status": "open"}


def record(ledger: list[dict], r: dict) -> dict:
    v = r["validation"]
    g = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    w = sum(b["result"] == "win" for b in g)
    l = sum(b["result"] == "loss" for b in g)
    n = w + l
    staked = sum(b["units"] for b in g if b["result"] != "push")
    profit = sum(b["profit_units"] for b in g)
    be = sum(1 / ML.decimal(b["price"]) for b in g if b["result"] != "push") / n if n else 0.524
    z = (w - n * be) / math.sqrt(n * be * (1 - be)) if n else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2)) if n else 1.0
    clv = [b["clv"] for b in g if "clv" in b]
    roi = profit / staked if staked else 0.0
    checks = {"enough_bets": len(g) >= v["min_bets"], "cover_rate_significant": n > 0 and p < v["cover_rate_p_below"],
              "roi_positive": roi > v["realized_roi_above"]}
    return {"graded": len(g), "wins": w, "losses": l, "units_staked": round(staked, 2),
            "profit_units": round(profit, 2), "roi": round(roi, 4),
            "avg_clv": round(sum(clv) / len(clv), 4) if clv else 0.0, "clv_p_value": round(p, 4),
            "checks": {"enough_bets": checks["enough_bets"], "clv_positive_and_significant": checks["cover_rate_significant"],
                       "roi_positive": checks["roi_positive"]},
            "passed": all(checks.values()), "min_bets": v["min_bets"], "test": "cover-rate p-value shown as p"}


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None) -> dict:
    r = r or load_rules()
    sr = SB.load_rules()
    path = history_dir / "paper_bets_night_west.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    if any(b.get("status") == "open" for b in ledger) and len(games):
        closes = SB.closing_margins(history_dir, sr)
        ledger = [SB.grade(b, games, sr, closes) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    new = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r)
        if bet and g["game_id"] not in have:
            new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = record(ledger, r)
    for g in pred.get("upcoming", []):
        v = g.get("night_west")
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else "pending"
            v["reasons"] = [] if logged else list(g.get("night_west_check") or [])
    return {"track": "night_west", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": [], "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}
