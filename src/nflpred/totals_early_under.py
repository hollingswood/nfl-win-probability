"""Early-week under paper track (totals_early_under_rules.json): a new 2026 hypothesis from the totals grade study.

At the Tuesday 14:10 UTC (7:10am AZ) run, for games 96 h..7 days from kickoff: take every book's totals quote at
this snapshot (research filters, grade_totals.research_rows), the sharp fair expected total (median over
LowVig/BetOnline/Circa/Bookmaker of each book's implied E[T]), and bet the UNDER at the allowed book whose point +
price has the highest EV under that fair total, if EV >= 0. Flat 1 unit, one bet per game. CLV vs our own closing
totals snapshot (totals.closing_totals, as totals_wind). Market-only; the totals grade is recorded as a label.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML
from . import grade_totals as GT
from . import ml_v4
from . import totals as T

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "totals_early_under_rules.json"
WINDOW_TXT = "only bets at the Tue 7:10am AZ run, 96 h-7 days before kickoff"


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def in_window(now: datetime, r: dict) -> bool:
    return ml_v4.in_window(now, r)


def best_under(quotes, allowed: set | None, r: dict, dist) -> tuple[dict | None, dict | None]:
    """(refs, best allowed-book under vs the sharp fair total) from raw [key, title, point, over, under] quotes."""
    q = r["qualify"]
    rows = GT.research_rows(quotes)
    R = GT.refs(rows, dist)
    if R is None:
        return None, None
    sharp = [x["mu"] for x in rows if x["key"] in set(q["sharp_books"])]
    mu_sharp = float(pd.Series(sharp).median()) if sharp else None
    R = dict(R, mu_sharp=mu_sharp)
    if mu_sharp is None:
        return R, None
    best = None
    for x in rows:
        if allowed is not None and x["key"] not in allowed:
            continue
        if not (q["min_american_odds"] <= x["under"] <= q["max_american_odds"]):
            continue
        ev = float(dist.ev(mu_sharp, x["point"], x["under"], "under"))
        if best is None or ev > best["_ev"]:
            price = int(x["under"]) if float(x["under"]).is_integer() else x["under"]
            best = {"book": x["book"], "book_key": x["key"], "point": x["point"], "price": price, "_ev": ev}
    if best is not None:  # thresholds compare the exact EV; the stored value is rounded
        best["ev_sharp"] = round(best["_ev"], 4)
    return R, best


def evaluate(game: dict, r: dict, dist, now: datetime | None = None, allowed: set | None | bool = False) -> dict | None:
    from . import odds as O
    q = r["qualify"]
    now = now or datetime.now(timezone.utc)
    allowed = O.load_allowed_books() if allowed is False else allowed
    reasons = game["early_under_check"] = []
    game["early_under"] = None
    tot = ((game.get("context") or {}).get("live_odds") or {}).get("totals") or {}
    if not tot.get("all_quotes"):
        reasons.append("no live totals")
        return None
    R, best = best_under(tot["all_quotes"], allowed, r, dist)
    if R is None:
        reasons.append("no live totals")
        return None
    if best is None:
        reasons.append("no sharp-book total (or no allowed-book under)")
        return None
    try:
        hours = (datetime.fromisoformat(game["kickoff_utc"]) - now).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        hours = None
    ev_exact = best.pop("_ev")
    game["early_under"] = {**best, "sharp_total": round(R["mu_sharp"], 2), "consensus_total": round(R["mu_cons"], 2),
                           "hours_before": None if hours is None else round(hours, 1)}
    if ev_exact < q["min_ev_vs_sharp_fair_total"]:
        reasons.append(f"best under only {best['ev_sharp']:+.1%} vs the sharp fair total "
                       f"(need {q['min_ev_vs_sharp_fair_total']:+.0%})")
    if hours is None or hours < q["min_hours_before_kickoff"]:
        reasons.append(f"less than {q['min_hours_before_kickoff']} h before kickoff (rule: 96 h-7 days)")
    elif hours > q["max_hours_before_kickoff"] or not in_window(now, r):
        reasons.append(WINDOW_TXT)
    if reasons:
        return None
    return {"id": f"{game['game_id']}:early_under", "track": "totals_early_under", "rules_version": r["version"],
            "placed_at": now.isoformat(timespec="minutes"), "game_id": game["game_id"], "season": game["season"],
            "week": game["week"], "gameday": game["gameday"], "side": "under",
            "team": f"{game['away_team']}@{game['home_team']} UNDER", "opponent": "",
            "point": best["point"], "price": best["price"], "book": best["book"], "book_key": best["book_key"],
            "edge": best["ev_sharp"], "sharp_total": round(R["mu_sharp"], 2), "consensus_total": round(R["mu_cons"], 2),
            "hours_before": round(hours, 1), "units": r["sizing"]["units"], "status": "open",
            **GT.bet_fields(game, "under", best["book_key"], best["point"], best["price"])}  # label only


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None, now=None) -> dict:
    r = r or load_rules()
    dist = T.default_dist()
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    if len(games) and any(b.get("status") == "open" for b in ledger):
        closes = T.closing_totals(history_dir, dist)
        ledger = [T.grade(b, games, dist, closes) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    new = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r, dist, now)
        if bet and g["game_id"] not in have:
            new.append(bet)
            have.add(g["game_id"])
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    for g in pred.get("upcoming", []):
        v = g.get("early_under")
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else ("lean" if v["ev_sharp"] >= 0 else "pass")
            v["reasons"] = [] if logged else list(g.get("early_under_check") or [])
    return {"track": "totals_early_under", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": [], "by_totals_grade": GT.by_grade(ledger), "new": new,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}
