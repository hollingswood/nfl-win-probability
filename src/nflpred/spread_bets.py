"""Spread paper-bet track (separate rules and record from the moneyline track).

Per game: expected home margin = blend of the model's margin and the consensus market margin
(weights frozen in spread_rules.json, fit on 2015-2019), then a key-number-aware distribution
(margins.py) gives P(cover), P(push), P(lose) for EVERY sportsbook's actual line and price.
The best (book, side) is the one with the highest expected value, so shopping for -3 instead of
-3.5 is a real, priced "bought" half-point. Buying extra points at a book is shown as an estimate
only (the free odds plan has no alternate-line prices) and is never paper-bet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import bets as ml
from . import margins as K

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "spread_rules.json"


def load_rules(path: Path = RULES_PATH) -> dict:
    r = json.loads(path.read_text())
    r["_weights"] = {int(k): v for k, v in r["key_numbers"]["weights"].items()}
    return r


def expected_margin(model_margin: float, market_margin: float, r: dict) -> float:
    m = r["margin"]
    return m["model"] * model_margin + m["market"] * market_margin + m["intercept"]


def side_ev(mu: float, point: float, price: int, side: str, r: dict) -> tuple[float, float, float]:
    """EV per unit, P(win), P(push) of taking `side` at `point` (that side's own spread) and `price`."""
    home_line = point if side == "home" else -point
    hc, pu, ac = (x[0] for x in K.cover_probs(np.array([mu]), r["margin"]["sigma"], np.array([home_line]), r["_weights"]))
    pw = hc if side == "home" else ac
    return float(pw * ml.decimal(price) + pu - 1), float(pw), float(pu)


def _shift_price(price: int, cents: int) -> int:
    """Make an American price `cents` worse (e.g. -110 -> -130, +105 -> -115)."""
    if price < 0:
        return price - cents
    q = price - cents
    return q if q >= 100 else -(200 - q)


def buy_point_options(mu, point, price, side, r):
    """Estimated EV of buying 0.5 and 1.0 points at typical costs (informational)."""
    c = r["buy_points_estimate"]
    out, cur_point, cur_price = [], point, price
    for _ in range(2):
        nxt = cur_point + 0.5
        crosses = {abs(cur_point), abs(nxt)}
        cost = c["onto_or_off_3"] if 3 in crosses else c["onto_or_off_7"] if 7 in crosses else c["default_cents"]
        cur_point, cur_price = nxt, _shift_price(cur_price, cost)
        ev, _, _ = side_ev(mu, cur_point, cur_price, side, r)
        out.append({"point": cur_point, "est_price": cur_price, "ev": round(ev, 4)})
    return out


def analyze(game: dict, r: dict) -> dict | None:
    """Spread view for one game (used for the card and for qualification)."""
    lo = (game.get("context") or {}).get("live_odds") or {}
    books = lo.get("spreads_by_book") or []
    if not books or lo.get("consensus_home_margin") is None or game.get("model_home_margin") is None:
        return None
    mu = expected_margin(game["model_home_margin"], lo["consensus_home_margin"], r)
    best = None
    for b in books:
        for side in ("home", "away"):
            point, price = b[f"{side}_point"], b[f"{side}_price"]
            ev, pw, pu = side_ev(mu, point, price, side, r)
            if best is None or ev > best["ev"]:
                best = {"side": side, "team": game[f"{side}_team"], "point": point, "price": price, "book": b["book"],
                        "ev": round(ev, 4), "p_cover": round(pw, 4), "p_push": round(pu, 4)}
    best["buy_options"] = buy_point_options(mu, best["point"], best["price"], best["side"], r)
    return {"expected_home_margin": round(mu, 2), "best": best, "books": len(books)}


def evaluate(game: dict, r: dict, first_margin: float | None) -> dict | None:
    a = analyze(game, r)
    game["spread"] = a
    reasons = game["spread_check"] = []
    if not a:
        reasons.append("no live spreads")
        return None
    q, b = r["qualify"], a["best"]
    if b["ev"] < q["min_edge_at_best_price"]:
        reasons.append(f"best edge {b['ev']:+.1%} below {q['min_edge_at_best_price']:.0%}")
    if not (q["min_american_odds"] <= b["price"] <= q["max_american_odds"]):
        reasons.append("price outside allowed range")
    qbs = game.get("qb_status") or {}
    if q["require_both_starting_qbs_confirmed"] and any((qbs.get(s) or {}).get("play_prob", 1) < 1 for s in ("home", "away")):
        reasons.append("starting QB not confirmed")
    if q["require_injury_report_published"] and not game.get("injury_report"):
        reasons.append("injury report not out")
    now_margin = game["context"]["live_odds"]["consensus_home_margin"]
    if first_margin is not None:
        moved = (now_margin - first_margin) if b["side"] == "away" else (first_margin - now_margin)
        if moved >= q["require_line_not_moved_away_since_first_seen_points"]:
            reasons.append("line moved against this side since first seen")
    if reasons:
        return None
    p_nopush = b["p_cover"] / max(1 - b["p_push"], 1e-9)
    return {"id": f"{game['game_id']}:spread:{b['side']}", "track": "spread", "rules_version": r["version"],
            "placed_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "game_id": game["game_id"], "season": game["season"], "week": game["week"], "gameday": game["gameday"],
            "side": b["side"], "team": b["team"], "opponent": game["away_team" if b["side"] == "home" else "home_team"],
            "point": b["point"], "price": b["price"], "book": b["book"], "edge": b["ev"],
            "p_cover": b["p_cover"], "p_push": b["p_push"], "expected_home_margin": a["expected_home_margin"],
            "units": ml.kelly_units(p_nopush, b["price"], r["sizing"]), "status": "open"}


def grade(bet: dict, games: pd.DataFrame, r: dict) -> dict:
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    x = g.iloc[0]
    margin = x["home_score"] - x["away_score"]
    adj = (margin + bet["point"]) if bet["side"] == "home" else (-margin + bet["point"])
    out = dict(bet, status="graded", final=f"{int(x['away_score'])}-{int(x['home_score'])}")
    out["result"] = "win" if adj > 0 else "push" if adj == 0 else "loss"
    out["profit_units"] = round(bet["units"] * (ml.decimal(bet["price"]) - 1), 3) if adj > 0 else (0.0 if adj == 0 else -bet["units"])
    close = x.get("spread_line")  # closing expected home margin (+ = home favored)
    if close is not None and not pd.isna(close):
        out["closing_home_margin"] = float(close)
        # CLV: our line and price valued with the closing market as the expected margin
        out["clv"] = round(side_ev(float(close), bet["point"], bet["price"], bet["side"], r)[0], 4)
        out["clv_points"] = round(bet["point"] - ((-float(close)) if bet["side"] == "home" else float(close)), 1)
    return out


def first_seen_margin(history_dir: Path) -> dict[str, float]:
    first = {}
    for f in sorted(history_dir.glob("predictions_*.json")):
        try:
            ups = json.loads(f.read_text()).get("upcoming", [])
        except Exception:
            continue
        for g in ups:
            lo = (g.get("context") or {}).get("live_odds") or {}
            if lo.get("consensus_home_margin") is not None and g["game_id"] not in first:
                first[g["game_id"]] = lo["consensus_home_margin"]
    return first


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None) -> dict:
    r = r or load_rules()
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / "paper_bets_spread.json"
    ledger = json.loads(path.read_text()) if path.exists() else []
    ledger = [grade(b, games, r) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    first = first_seen_margin(history_dir)
    cands = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r, first.get(g["game_id"]))  # always run: also fills the card's spread view
        if bet and g["game_id"] not in have:
            cands.append((bet, g))
    cap, new = r["sizing"]["max_units_per_week"], []
    for bet, g in sorted(cands, key=lambda x: -x[0]["edge"]):
        used = sum(b["units"] for b in ledger + new
                   if b.get("status") != "void" and b["season"] == bet["season"] and b["week"] == bet["week"])
        if used + bet["units"] > cap + 1e-9:
            g["spread_check"].append(f"weekly exposure cap ({cap:g} units) reached")
            continue
        new.append(bet)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ml.record(ledger, r)
    return {"track": "spread", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}
