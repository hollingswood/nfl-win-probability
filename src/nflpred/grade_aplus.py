"""A+ grade paper tracks: bet EVERY A+ offer of a grade, to test whether the grade keeps its edge on 2026 lines.

Until now the grades were labels only, and the "record by grade" covered just the bets some other track happened to
place. These three tracks (frozen 2026-10-02, before any results) bet each A+ offer once, flat 1 unit:

  * aplus_ml     (grade_aplus_ml_rules.json)     moneyline grade v2 (grade_v2.py, game['grade_v2'] = best side).
                 Research output/research/grade_v2.md, S1_Aplus_flat: 2023-25 holdout +2.34% ± 0.98 CLV, 163 bets.
                 Graded with bets.grade (CLV vs the nflverse closing no-vig probability).
  * aplus_spread (grade_aplus_spread_rules.json) original spread grade v1 (grading.py via spread_bets.evaluate).
                 Research output/research/grade_v2_spread.md ('v1 A+', scripts/research/grade_v2_spread.v1_eval):
                 2023-25 +2.2% ± 1.3 CLV, 49 bets (borderline, p 0.05). Candidate exactly as v1_eval: at ANY
                 snapshot (no injury-report / QB-confirmed eligibility), the allowed-book offer with the highest
                 blend EV among offers passing the research spread filter (sp_ok: both prices -145..+125, half
                 points, overround -1%..12%, point within 2.5 of the all-book median), graded with grading.grade.
                 Graded with spread_bets.grade + closing_margins (closing PRICES, our own last snapshot).
  * aplus_totals (grade_aplus_totals_rules.json) totals grade (grade_totals.py, game['totals_grade'] = best
                 offer). Research output/research/grade_granular.md Q2: 2023-25 +1.48% ± 0.66 CLV, 243 bets (~95%
                 early unders). Graded with totals.grade + closing_totals (side-aware).

Protocol (the research convention): one bet per game per market at the FIRST snapshot where the game's best offer
is A+, at that offer's book, number and price; later snapshots never change it. Snapshots = every full run AND every
hourly watch (the research used our historical run times, which are sparser: more looks can only find more A+
moments). Candidates only 10 min .. 7 days before kickoff (the research tables' window). The grades themselves are
unchanged and still never qualify, veto or size any other track's bet.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML

ROOT = Path(__file__).resolve().parents[2]
RULE_FILES = {"ml": "grade_aplus_ml_rules.json", "spread": "grade_aplus_spread_rules.json",
              "totals": "grade_aplus_totals_rules.json"}
RESULT_KEYS = {"ml": "aplus_ml_bets", "spread": "aplus_spread_bets", "totals": "aplus_totals_bets"}
VIEW_KEYS = {"ml": "aplus_ml", "spread": "aplus_spread", "totals": "aplus_totals"}
EXCLUDE = {"pinnacle", "kalshi", "prophetx", "polymarket", "novig", "betopenly"}  # not in the research feed


def load_rules(market: str, root: Path | None = None) -> dict:
    return json.loads(((root or ROOT) / RULE_FILES[market]).read_text())


def _hours(game: dict, now: datetime) -> float | None:
    try:
        return (datetime.fromisoformat(game["kickoff_utc"]) - now).total_seconds() / 3600
    except (KeyError, TypeError, ValueError):
        return None


def _window_reason(game: dict, now: datetime, r: dict) -> str | None:
    q = r["qualify"]
    h = _hours(game, now)
    if h is None:
        return "no kickoff time"
    if h < q["min_hours_before_kickoff"]:
        return "too close to kickoff"
    if h > q["max_hours_before_kickoff"]:
        return "more than 7 days before kickoff (research window)"
    return None


# ------------------------------------------------------------------------------------------------ offers
def ml_offer(game: dict) -> tuple[dict | None, list[str]]:
    """The game's graded moneyline offer (best predicted CLV side) and why it is not A+."""
    g = game.get("grade_v2")
    if not g:
        return None, ["not graded"]
    o = {"side": g["side"], "team": g["team"], "price": g["price"], "book": g.get("book"),
         "grade": g.get("grade"), "predicted_clv": g.get("predicted_clv")}
    return o, ([] if o["grade"] == "A+" else [f"grade {o['grade']} (needs A+)"])


def _imp(a: float) -> float:
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def research_spread_rows(lo: dict, r: dict) -> list[dict]:
    """Allowed-book spreads_by_book rows that pass the research sp_ok filter (ml_spread_consistency.build_ml):
    both prices within the bounds, home point = -away point, half points, point within 2.5 of the median valid
    point over ALL books at the snapshot (by_book, exchanges/Pinnacle left out as in the research feed; fallback the
    allowed rows), overround -1%..12%."""
    q = r["qualify"]
    lo_p, hi_p = q["min_american_odds"], q["max_american_odds"]

    def ok(hp, ap_, hpr, apr):
        try:
            hp, hpr, apr = float(hp), float(hpr), float(apr)
        except (TypeError, ValueError):
            return False
        if ap_ is not None and abs(hp + float(ap_)) > 1e-9:
            return False
        return lo_p <= hpr <= hi_p and lo_p <= apr <= hi_p and (hp * 2) % 1 == 0

    rows = [b for b in lo.get("spreads_by_book") or []
            if ok(b.get("home_point"), b.get("away_point"), b.get("home_price"), b.get("away_price"))]
    allp = [float(d["sp"][0]) for k, d in (lo.get("by_book") or {}).items()
            if k not in EXCLUDE and (d or {}).get("sp") and ok(d["sp"][0], None, d["sp"][1], d["sp"][2])]
    pts = allp or [float(b["home_point"]) for b in rows]
    if not pts:
        return []
    med = statistics.median(pts)
    return [b for b in rows if abs(float(b["home_point"]) - med) <= q["max_points_from_median"]
            and q["min_overround"] <= _imp(b["home_price"]) + _imp(b["away_price"]) - 1 <= q["max_overround"]]


def spread_offer(game: dict, r: dict, sr: dict, first_margin: float | None) -> tuple[dict | None, list[str]]:
    """Grade v1 of the snapshot's best blend-EV offer among research-filtered rows (as v1_eval), without touching
    the game's own spread view."""
    from . import spread_bets as SB
    lo = (game.get("context") or {}).get("live_odds") or {}
    if not lo.get("spreads_by_book"):
        return None, ["no live spreads"]
    rows = research_spread_rows(lo, r)
    if not rows:
        return None, ["no spread offer passes the research filters (prices -145..+125)"]
    g2 = dict(game, context=dict(game.get("context") or {}, live_odds=dict(lo, spreads_by_book=rows)))
    SB.evaluate(g2, sr, first_margin)  # fills g2['spread'] with the v1 grade of the best offer
    a = g2.get("spread")
    if not a:
        return None, ["no live spreads"]
    b = a["best"]
    o = {"side": b["side"], "team": b["team"], "point": b["point"], "price": b["price"], "book": b["book"],
         "ev": b["ev"], "p_cover": b["p_cover"], "p_push": b["p_push"], "grade": b.get("grade"),
         "score": b.get("score"), "why": b.get("why") or [], "grading_version": b.get("grading_version"),
         "expected_home_margin": a["expected_home_margin"]}
    return o, ([] if o["grade"] == "A+" else [f"grade {o['grade']} (needs A+)"])


def totals_offer(game: dict) -> tuple[dict | None, list[str]]:
    t = game.get("totals_grade")
    if not t:
        return None, list(game.get("totals_grade_check") or ["not graded"])
    o = {"side": t["side"], "point": t["point"], "price": t["price"], "book": t.get("book"),
         "book_key": t.get("book_key"), "grade": t.get("grade"), "predicted_clv": t.get("predicted_clv"),
         "consensus_total": t.get("consensus_total"), "sharp_total": t.get("sharp_total")}
    return o, ([] if o["grade"] == "A+" else [f"grade {o['grade']} (needs A+)"])


# ------------------------------------------------------------------------------------------------ bets
def _base(market: str, game: dict, r: dict, now: datetime, side: str) -> dict:
    return {"id": f"{game['game_id']}:aplus_{market}:{side}", "track": f"aplus_{market}",
            "rules_version": r["version"], "placed_at": now.isoformat(timespec="minutes"),
            "game_id": game["game_id"], "season": game["season"], "week": game["week"], "gameday": game["gameday"],
            "side": side, "units": r["sizing"]["units"], "status": "open",
            "hours_before": round(_hours(game, now), 1)}


def make_bet(market: str, game: dict, o: dict, r: dict, now: datetime) -> dict:
    b = _base(market, game, r, now, o["side"])
    if market == "ml":
        opp = game["away_team"] if o["side"] == "home" else game["home_team"]
        b.update(team=o["team"], opponent=opp, price=o["price"], book=o["book"], edge=o["predicted_clv"],
                 grade_v2="A+", predicted_clv=o["predicted_clv"], grading_version=2)
    elif market == "spread":
        opp = game["away_team"] if o["side"] == "home" else game["home_team"]
        b.update(team=o["team"], opponent=opp, point=o["point"], price=o["price"], book=o["book"], edge=o["ev"],
                 p_cover=o["p_cover"], p_push=o["p_push"], expected_home_margin=o["expected_home_margin"],
                 grade="A+", grade_why=o["why"], grade_score=o["score"], grading_version=o["grading_version"])
    else:
        b.update(team=f"{game['away_team']}@{game['home_team']} {o['side'].upper()}", opponent="",
                 point=o["point"], price=o["price"], book=o["book"], book_key=o["book_key"], edge=o["predicted_clv"],
                 totals_grade="A+", predicted_clv=o["predicted_clv"], consensus_total=o["consensus_total"],
                 sharp_total=o["sharp_total"], grading_version=2)
    return b


def _by_grade(market: str, ledger: list[dict]) -> dict:
    try:
        if market == "ml":
            from . import grade_v2
            return {"by_grade_v2": grade_v2.by_grade(ledger)}
        if market == "totals":
            from . import grade_totals
            return {"by_totals_grade": grade_totals.by_grade(ledger)}
        from . import grading
        return {"by_grade": grading.by_grade(ledger)}
    except Exception:
        return {}


def process(market: str, pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None,
            now: datetime | None = None) -> dict:
    """One run of an A+ track: grade open bets (when results are passed), lock new A+ offers, set the per-game view."""
    r = r or load_rules(market)
    now = now or datetime.now(timezone.utc)
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    if market == "spread":
        from . import spread_bets as SB
        sr = SB.load_rules()
        first = SB.first_seen_margin(history_dir)
    if len(games) and any(b.get("status") == "open" for b in ledger):
        if market == "ml":
            ledger = [ML.grade(b, games) if b.get("status") == "open" else b for b in ledger]
        elif market == "spread":
            closes = SB.closing_margins(history_dir, sr)
            ledger = [SB.grade(b, games, sr, closes) if b.get("status") == "open" else b for b in ledger]
        else:
            from . import totals as T
            dist = T.default_dist()
            closes = T.closing_totals(history_dir, dist)
            ledger = [T.grade(b, games, dist, closes) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    new, views = [], {}
    for g in pred.get("upcoming", []):
        try:
            if market == "ml":
                o, reasons = ml_offer(g)
            elif market == "spread":
                o, reasons = spread_offer(g, r, sr, first.get(g["game_id"]))
            else:
                o, reasons = totals_offer(g)
        except Exception as e:  # one bad game never blocks the track
            o, reasons = None, [f"error: {e}"]
        w = _window_reason(g, now, r)
        if w and o is not None:
            reasons = reasons + [w]
        views[g["game_id"]] = (o, reasons)
        if o is not None and not reasons and g["game_id"] not in have:
            new.append(make_bet(market, g, o, r, now))
            have.add(g["game_id"])
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    vk = VIEW_KEYS[market]
    for g in pred.get("upcoming", []):
        o, reasons = views.get(g["game_id"], (None, []))
        bet = next((b for b in ledger if b["game_id"] == g["game_id"] and b.get("status") != "void"), None)
        if bet is not None:  # locked: show the logged bet, not the current offer
            v = {k: bet.get(k) for k in ("side", "team", "point", "price", "book", "predicted_clv", "edge")}
            if market == "totals":
                v["team"] = bet["side"].capitalize()
            v.update(grade="A+", verdict="bet", reasons=[], placed_at=bet["placed_at"], ledger_team=bet["team"],
                     current_grade=(o or {}).get("grade"))
        elif o is not None:
            v = dict(o, verdict="pass", reasons=list(reasons))
            if market == "totals":
                v["team"] = o["side"].capitalize()
        else:
            v = None
        g[vk] = v
    return {"track": f"aplus_{market}", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec,
            **_by_grade(market, ledger)}


def process_ml(pred, games, history_dir, r=None, now=None) -> dict:
    return process("ml", pred, games, history_dir, r, now)


def process_spread(pred, games, history_dir, r=None, now=None) -> dict:
    return process("spread", pred, games, history_dir, r, now)


def process_totals(pred, games, history_dir, r=None, now=None) -> dict:
    return process("totals", pred, games, history_dir, r, now)


def alert_lines(result: dict) -> list[str]:
    """One line per NEW bet of an A+ track that has passed its validation (mode 'live')."""
    lines = []
    for market, key in RESULT_KEYS.items():
        res = result.get(key) or {}
        if res.get("mode") != "live":
            continue
        for x in res.get("new", []):
            if market == "ml":
                lines.append(f"- **{x['team']}** moneyline {x['price']:+d} at {x['book']} vs {x['opponent']} "
                             f"({x['gameday']}): moneyline grade A+ (pred CLV {x['predicted_clv']:+.1%}), stake {x['units']}u")
            elif market == "spread":
                lines.append(f"- **{x['team']} {x['point']:+g}** ({x['price']:+d}) at {x['book']} vs {x['opponent']} "
                             f"({x['gameday']}): spread grade A+, stake {x['units']}u")
            else:
                lines.append(f"- **{x['team']} {x['point']}** ({x['price']:+d}) at {x['book']} ({x['gameday']}): "
                             f"totals grade A+ (pred CLV {x['predicted_clv']:+.1%}), stake {x['units']}u")
    return lines
