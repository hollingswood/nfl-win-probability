"""Moneyline v2 paper track: soft book vs sharp book line shopping (pre-registered in
moneyline_v2_rules.json; candidate C4 of the 2026-09-30 edge search).

A side qualifies when the best price among my_books.json is worth >= 2% more than the sharp books'
no-vig fair probability (lowvig, BetOnline, Circa, Bookmaker) AND the model's own probability
says the price is not negative-EV, once the injury report is out and both QBs are confirmed.
Same locking, weekly cap, grading (CLV vs the closing no-vig line) and validation mechanics as
the moneyline v1 track in bets.py; separate ledger history/paper_bets_ml_v2.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML
from . import grading as G

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "moneyline_v2_rules.json"
V3_RULES_PATH = ROOT / "moneyline_v3_rules.json"


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def evaluate(game: dict, r: dict) -> dict | None:
    q = r["qualify"]
    lo = (game.get("context") or {}).get("live_odds") or {}
    reasons = game[r.get("game_key", "ml_v2") + "_check"] = []
    if not lo.get("best_home_ml") or not lo.get("best_away_ml"):
        reasons.append("no live odds")
        return None
    ref = r.get("reference_field", "sharp_home_prob")
    if lo.get(ref) is None:
        reasons.append("no sharp-book price")
        game[r.get("game_key", "ml_v2")] = None
        return None
    best = None
    for side, bk in (("home", lo["best_home_ml"]), ("away", lo["best_away_ml"])):
        p_sharp = lo[ref] if side == "home" else 1 - lo[ref]
        p_model = game["home_win_prob"] if side == "home" else 1 - game["home_win_prob"]
        p_cons = lo["consensus_home_prob"] if side == "home" else 1 - lo["consensus_home_prob"]
        d = ML.decimal(bk["price"])
        cand = {"side": side, "team": game[f"{side}_team"], "price": bk["price"], "book": bk["book"],
                "p_sharp": round(p_sharp, 4), "p_model": round(p_model, 4), "p_market": round(p_cons, 4),
                "p_needed": round(1 / d, 4), "ev_sharp": round(p_sharp * d - 1, 4),
                "ev_model": round(p_model * d - 1, 4), "gap": abs(1 / d - p_cons)}
        if best is None or cand["ev_sharp"] > best["ev_sharp"]:
            best = cand
    game[r.get("game_key", "ml_v2")] = best
    if best["ev_sharp"] < q["min_ev_vs_sharp"]:
        reasons.append(f"best price only {best['ev_sharp']:+.1%} vs sharp books (need {q['min_ev_vs_sharp']:.0%})")
    if best["ev_model"] < q["min_ev_vs_model"]:
        reasons.append("model disagrees")
    if best["gap"] > q["max_price_gap_vs_consensus"] or not (q["min_american_odds"] <= best["price"] <= q["max_american_odds"]):
        reasons.append("price looks like a data error")
    qbs = game.get("qb_status") or {}
    if q["require_both_starting_qbs_confirmed"] and any((qbs.get(s) or {}).get("play_prob", 1) < 1 for s in ("home", "away")):
        reasons.append("starting QB not confirmed")
    if q["require_injury_report_published"] and not game.get("injury_report"):
        reasons.append("injury report not out")
    if reasons:
        return None
    opp = game["away_team"] if best["side"] == "home" else game["home_team"]
    return {"id": f"{game['game_id']}:{r['track']}:{best['side']}", "track": r["track"], "rules_version": r["version"],
            "placed_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "game_id": game["game_id"], "season": game["season"], "week": game["week"], "gameday": game["gameday"],
            "side": best["side"], "team": best["team"], "opponent": opp, "price": best["price"], "book": best["book"],
            "p_sharp": best["p_sharp"], "p_model": best["p_model"], "p_market": best["p_market"],
            "edge": best["ev_sharp"], "ev_model": best["ev_model"],
            "units": ML.kelly_units(best["p_sharp"], best["price"], r["sizing"]), "status": "open"}


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None) -> dict:
    r = r or load_rules()
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / r.get("ledger", "paper_bets_ml_v2.json")
    ledger = json.loads(path.read_text()) if path.exists() else []
    ledger = [ML.grade(b, games) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    cands = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r)
        if bet and g["game_id"] not in have:
            cands.append((bet, g))
    cap, new = r["sizing"]["max_units_per_week"], []
    for bet, g in sorted(cands, key=lambda x: -x[0]["edge"]):
        used = sum(b["units"] for b in ledger + new
                   if b.get("status") != "void" and b["season"] == bet["season"] and b["week"] == bet["week"])
        if used + bet["units"] > cap + 1e-9:
            g[r.get("game_key", "ml_v2") + "_check"].append(f"weekly exposure cap ({cap:g} units) reached")
            continue
        new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    for g in pred.get("upcoming", []):
        v = g.get(r.get("game_key", "ml_v2"))
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else ("lean" if v["ev_sharp"] > 0 else "pass")
            v["reasons"] = [] if logged else list(g.get(r.get("game_key", "ml_v2") + "_check") or [])
    return {"track": r["track"], "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": G.by_grade(ledger), "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}


def process_v3(pred: dict, games: pd.DataFrame, history_dir: Path) -> dict:
    """Moneyline v3: identical to v2 except the fair price comes from Pinnacle + LowVig + BetOnline."""
    return process(pred, games, history_dir, load_rules(V3_RULES_PATH))
