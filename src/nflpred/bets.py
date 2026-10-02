"""Paper betting ("shadow mode") and the switch to live recommendations.

Every run:
  1. For each upcoming game with live multi-book odds, compute the model + market blended
     probability (weights frozen in betting_rules.json, fit on 2015-2019 only).
  2. A side qualifies if its expected value at the BEST available moneyline clears the threshold
     and the safety checks pass (QBs confirmed, injury report out, line not moving against us).
  3. A qualifying side is recorded ONCE in history/paper_bets.json at that run's best price.
     Later runs never change it, exactly like a bet you placed.
  4. Finished games are graded: result, profit in units, and closing line value (CLV).
  5. Validation status is recomputed. Until the pre-registered test in betting_rules.json passes,
     mode is "shadow": bets are shown as paper bets only. When it passes, mode becomes "live":
     the dashboard highlights qualifying bets and the workflow opens a GitHub issue as an alert.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import grading as G

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "betting_rules.json"


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def decimal(american: float) -> float:
    return 1 + american / 100 if american > 0 else 1 + 100 / -american


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def blend_prob(p_model_home: float, p_market_home: float, w: dict) -> float:
    z = w["model"] * _logit(p_model_home) + w["vegas"] * _logit(p_market_home) + w["intercept"]
    return 1 / (1 + math.exp(-z))


def kelly_units(p: float, american: float, sizing: dict) -> float:
    b = decimal(american) - 1
    f = (b * p - (1 - p)) / b  # full-Kelly fraction of bankroll
    units = sizing["kelly_fraction"] * f * 100 / sizing["unit_pct_of_bankroll"]
    return round(min(max(units, sizing["min_units"]), sizing["max_units"]), 2)


def first_seen_market(history_dir: Path) -> dict[str, float]:
    """game_id -> consensus home probability the first time we saw live odds for it."""
    first: dict[str, float] = {}
    for f in sorted(history_dir.glob("predictions_*.json")):
        try:
            ups = json.loads(f.read_text()).get("upcoming", [])
        except Exception:
            continue
        for g in ups:
            lo = (g.get("context") or {}).get("live_odds")
            if lo and g["game_id"] not in first:
                first[g["game_id"]] = lo["consensus_home_prob"]
    return first


def evaluate(game: dict, rules: dict, first_market: float | None) -> dict | None:
    """Return the qualifying paper bet for this game, or None. `reasons` explains skips."""
    q, lo = rules["qualify"], (game.get("context") or {}).get("live_odds")
    game["bet_check"] = reasons = []
    if not lo or not lo.get("best_home_ml") or not lo.get("best_away_ml"):
        reasons.append("no live odds")
        return None
    p_home = blend_prob(game["home_win_prob"], lo["consensus_home_prob"], rules["probability"])
    best = None
    for side, p, bk, mkt in (("home", p_home, lo["best_home_ml"], lo["consensus_home_prob"]),
                             ("away", 1 - p_home, lo["best_away_ml"], 1 - lo["consensus_home_prob"])):
        edge = p * decimal(bk["price"]) - 1
        if best is None or edge > best["edge"]:
            best = {"side": side, "p": p, "price": bk["price"], "book": bk["book"], "edge": edge, "market": mkt}
    first_side = None if first_market is None else (first_market if best["side"] == "home" else 1 - first_market)
    moved = None if first_side is None else 100 * (best["market"] - first_side)
    p_model_side = game["home_win_prob"] if best["side"] == "home" else 1 - game["home_win_prob"]
    qbc = game.get("qb_change") or {}
    qb_flag = any(abs(qbc.get(s) or 0) > G.QB_CHANGE_FLAG for s in ("home", "away"))
    gr = G.grade("moneyline", best["edge"], 100 * (p_model_side - best["market"]), moved, None, qb_flag)
    game["moneyline"] = {"side": best["side"], "team": game[f"{best['side']}_team"], "price": best["price"],
                         "book": best["book"], "p_ours": round(best["p"], 4), "p_market": round(best["market"], 4),
                         "p_needed": round(1 / decimal(best["price"]), 4), "edge": round(best["edge"], 4), **gr}
    if best["edge"] < q["min_edge_at_best_price"]:
        reasons.append(f"best edge {best['edge']:+.1%} below {q['min_edge_at_best_price']:.0%}")
    if not (q["min_american_odds"] <= best["price"] <= q["max_american_odds"]):
        reasons.append("price outside allowed range")
    qbs = game.get("qb_status") or {}
    if q["require_both_starting_qbs_confirmed"] and any(
            (qbs.get(s) or {}).get("play_prob", 1) < 1 for s in ("home", "away")):
        reasons.append("starting QB not confirmed")
    if q["require_injury_report_published"] and not game.get("injury_report"):
        reasons.append("injury report not out")
    if q["require_line_not_moved_away_since_first_seen"] and first_market is not None:
        first_side = first_market if best["side"] == "home" else 1 - first_market
        if best["market"] < first_side - 0.02:
            reasons.append("line moved against this side since first seen")
    if reasons:
        return None
    team = game["home_team"] if best["side"] == "home" else game["away_team"]
    opp = game["away_team"] if best["side"] == "home" else game["home_team"]
    return {
        "id": f"{game['game_id']}:{best['side']}", "rules_version": rules["version"],
        "placed_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "game_id": game["game_id"], "season": game["season"], "week": game["week"], "gameday": game["gameday"],
        "side": best["side"], "team": team, "opponent": opp, "price": best["price"], "book": best["book"],
        "p_model": round(game["home_win_prob"] if best["side"] == "home" else 1 - game["home_win_prob"], 4),
        "p_market": round(best["market"], 4), "p_blend": round(best["p"], 4), "edge": round(best["edge"], 4),
        "units": kelly_units(best["p"], best["price"], rules["sizing"]), "status": "open",
        "grade": gr["grade"], "grade_why": gr["why"], "grading_version": gr["grading_version"],
        **_grade_v2_fields(game, best["side"], best["price"]),  # moneyline grade v2: label only
    }


def _grade_v2_fields(game: dict, side: str, price) -> dict:
    try:
        from . import grade_v2
        return grade_v2.bet_fields(game, side, price)
    except Exception:
        return {"grade_v2": None, "predicted_clv": None}


def grade(bet: dict, games: pd.DataFrame) -> dict:
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    r = g.iloc[0]
    hs, as_ = r["home_score"], r["away_score"]
    won = (hs > as_) if bet["side"] == "home" else (as_ > hs)
    push = hs == as_
    dec = decimal(bet["price"])
    b = dict(bet)
    b["status"] = "graded"
    b["result"] = "push" if push else ("win" if won else "loss")
    b["profit_units"] = 0.0 if push else round(bet["units"] * (dec - 1) if won else -bet["units"], 3)
    close_home = r.get("vegas_home_prob")
    if close_home is not None and not pd.isna(close_home):
        close_side = close_home if bet["side"] == "home" else 1 - close_home
        b["closing_prob"] = round(float(close_side), 4)
        b["clv"] = round(dec * float(close_side) - 1, 4)  # EV of our price at the closing fair odds
    b["final"] = f"{int(as_)}-{int(hs)}"
    return b


def record(ledger: list[dict], rules: dict) -> dict:
    """Validation status against the pre-registered test."""
    v = rules["validation"]
    graded = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == rules["version"]]
    staked = sum(b["units"] for b in graded if b["result"] != "push")
    profit = sum(b["profit_units"] for b in graded)
    clvs = [b["clv"] for b in graded if "clv" in b]
    n = len(clvs)
    mean = sum(clvs) / n if n else 0.0
    sd = (sum((c - mean) ** 2 for c in clvs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    z = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else 0.0
    p_value = 0.5 * math.erfc(z / math.sqrt(2)) if n > 1 else 1.0  # one-sided: is mean CLV > 0?
    roi = profit / staked if staked else 0.0
    checks = {
        "enough_bets": len(graded) >= v["min_bets"],
        "clv_positive_and_significant": mean > 0 and p_value < v["avg_clv_positive_with_p_below"],
        "roi_positive": roi > v["realized_roi_above"],
    }
    wins = sum(b["result"] == "win" for b in graded)
    losses = sum(b["result"] == "loss" for b in graded)
    return {"graded": len(graded), "wins": wins, "losses": losses, "units_staked": round(staked, 2),
            "profit_units": round(profit, 2), "roi": round(roi, 4), "avg_clv": round(mean, 4),
            "clv_p_value": round(p_value, 4), "checks": checks, "passed": all(checks.values()),
            "min_bets": v["min_bets"]}


def process(pred: dict, games: pd.DataFrame, history_dir: Path, rules: dict | None = None) -> dict:
    """Run the whole cycle for one pipeline run. Mutates nothing but the ledger file."""
    rules = rules or load_rules()
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "paper_bets.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    ledger = [grade(b, games) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    first = first_seen_market(history_dir)
    cands = []
    for g in pred.get("upcoming", []):
        if g["game_id"] in have:
            continue
        bet = evaluate(g, rules, first.get(g["game_id"]))
        if bet:
            cands.append((bet, g))
    cap = rules["sizing"].get("max_units_per_week")
    new = []
    for bet, g in sorted(cands, key=lambda x: -x[0]["edge"]):  # biggest edges first
        used = sum(b["units"] for b in ledger + new
                   if b.get("status") != "void" and b["season"] == bet["season"] and b["week"] == bet["week"])
        if cap is not None and used + bet["units"] > cap + 1e-9:
            g["bet_check"].append(f"weekly exposure cap ({cap:g} units) reached")
            continue
        new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = record(ledger, rules)
    mode = "live" if rec["passed"] else "shadow"
    for g in pred.get("upcoming", []):  # final verdict for the card: bet / lean / pass
        v = g.get("moneyline")
        if v:
            logged = any(b["game_id"] == g["game_id"] and b.get("status") != "void" for b in ledger)
            v["verdict"] = "bet" if logged else ("lean" if v["edge"] > 0 else "pass")
            v["reasons"] = [] if logged else list(g.get("bet_check") or [])
    try:
        from . import grade_v2 as _gv2
        bg2 = _gv2.by_grade(ledger)
    except Exception:
        bg2 = []
    return {"mode": mode, "rules_version": rules["version"], "new": new, "by_grade": G.by_grade(ledger),
            "by_grade_v2": bg2,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:],
            "record": rec}
