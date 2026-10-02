"""Moneyline v4 paper track: moneyline vs the sharp books' SPREAD (moneyline_v4_rules.json).

When an allowed book's moneyline is worth >= 2% more than the win probability implied by the sharp
books' spread + juice (through the total-aware margin model, margin_total.py) and is not negative-EV
vs the sharp moneyline, bet it, but only at the Tue/Sun 7:10am AZ and Friday report runs (the
pre-registered windows). Market-only: no model input. Same grading/validation as v1/v2.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML
from . import grading as G
from . import grade_v2 as GV2
from . import margin_total as MT

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "moneyline_v4_rules.json"
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_MODEL = None


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = MT.load()
    return _MODEL


def in_window(now: datetime, r: dict) -> bool:
    tol = r["qualify"]["window_tolerance_minutes"]
    for day, hm in r["qualify"]["windows_utc"]:
        h, m = (int(x) for x in hm.split(":"))
        t = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if now.weekday() == DAYS[day] and abs((now - t).total_seconds()) <= tol * 60:
            return True
    return False


def spread_implied_home(lo: dict, r: dict) -> float | None:
    rows = lo.get("sharp_spreads") or []
    tot = (lo.get("totals") or {}).get("median_point") or 44.0
    cons = lo.get("consensus_home_margin")
    ps = []
    for hp, hpr, apr in rows:
        if cons is not None and abs(-hp - cons) > r["qualify"]["max_spread_point_gap_vs_consensus"]:
            continue
        ih = -hpr / (-hpr + 100) if hpr < 0 else 100 / (hpr + 100)
        ia = -apr / (-apr + 100) if apr < 0 else 100 / (apr + 100)
        q = ih / (ih + ia)
        M = _model()
        mu = M.implied_mu(hp, q, tot)
        pw, pt = M.win_tie(mu, tot)
        ps.append(float(pw[0] / (1 - pt[0])))
    return round(statistics.median(ps), 4) if ps else None


def evaluate(game: dict, r: dict, now: datetime | None = None) -> dict | None:
    q = r["qualify"]
    now = now or datetime.now(timezone.utc)
    lo = (game.get("context") or {}).get("live_odds") or {}
    reasons = game["ml_v4_check"] = []
    p_sp = spread_implied_home(lo, r) if lo else None
    p_ml = lo.get("sharp_home_prob")
    if p_sp is None or p_ml is None or not lo.get("best_home_ml") or not lo.get("best_away_ml"):
        reasons.append("no sharp spread/moneyline")
        game["ml_v4"] = None
        return None
    best = None
    for side in ("home", "away"):
        bk = lo[f"best_{side}_ml"]
        d = ML.decimal(bk["price"])
        ps = p_sp if side == "home" else 1 - p_sp
        pm = p_ml if side == "home" else 1 - p_ml
        c = {"side": side, "team": game[f"{side}_team"], "price": bk["price"], "book": bk["book"],
             "p_spread": round(ps, 4), "p_sharp_ml": round(pm, 4), "p_needed": round(1 / d, 4),
             "ev_spread": round(ps * d - 1, 4), "ev_ml": round(pm * d - 1, 4)}
        if best is None or c["ev_spread"] > best["ev_spread"]:
            best = c
    game["ml_v4"] = best
    if best["ev_spread"] < q["min_ev_vs_sharp_spread_implied"]:
        reasons.append(f"moneyline only {best['ev_spread']:+.1%} vs sharp spread (need {q['min_ev_vs_sharp_spread_implied']:.0%})")
    if best["ev_ml"] < q["min_ev_vs_sharp_moneyline"]:
        reasons.append("negative vs sharp moneyline")
    if not (q["min_american_odds"] <= best["price"] <= q["max_american_odds"]):
        reasons.append("price outside allowed range")
    if not in_window(now, r):
        reasons.append("only bets at the Tue/Sun 7:10am and Fri 2:40pm AZ runs")
    if reasons:
        return None
    opp = game["away_team"] if best["side"] == "home" else game["home_team"]
    return {"id": f"{game['game_id']}:ml_v4:{best['side']}", "track": "moneyline_v4", "rules_version": r["version"],
            "placed_at": now.isoformat(timespec="minutes"), "game_id": game["game_id"], "season": game["season"],
            "week": game["week"], "gameday": game["gameday"], "side": best["side"], "team": best["team"],
            "opponent": opp, "price": best["price"], "book": best["book"], "p_spread": best["p_spread"],
            "p_sharp_ml": best["p_sharp_ml"], "edge": best["ev_spread"],
            "units": ML.kelly_units(best["p_spread"], best["price"], r["sizing"]), "status": "open",
            **GV2.bet_fields(game, best["side"], best["price"])}  # grade v2: label only, never qualifies


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None, now=None) -> dict:
    r = r or load_rules()
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    if len(games):
        ledger = [ML.grade(b, games) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    cands = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r, now)
        if bet and g["game_id"] not in have:
            cands.append((bet, g))
    cap, new = r["sizing"]["max_units_per_week"], []
    for bet, g in sorted(cands, key=lambda x: -x[0]["edge"]):
        used = sum(b["units"] for b in ledger + new
                   if b.get("status") != "void" and b["season"] == bet["season"] and b["week"] == bet["week"])
        if used + bet["units"] > cap + 1e-9:
            g["ml_v4_check"].append(f"weekly exposure cap ({cap:g} units) reached")
            continue
        new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    for g in pred.get("upcoming", []):
        v = g.get("ml_v4")
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else ("lean" if v["ev_spread"] > 0 else "pass")
            v["reasons"] = [] if logged else list(g.get("ml_v4_check") or [])
    return {"track": "moneyline_v4", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": G.by_grade(ledger), "by_grade_v2": GV2.by_grade(ledger), "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}
