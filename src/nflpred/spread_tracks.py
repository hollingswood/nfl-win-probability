"""Two spread paper tracks frozen from the 7-strategy screen (2026-10-03):

* preseason_prior (preseason_prior_rules.json): weeks 2-8, back the side preseason win totals favor when the
  market is 4+ points away from the preseason-implied margin; bet in the last 3 h before kickoff; judged on results.
* tuesday_move (tuesday_move_rules.json): at the Tuesday 14:10 UTC run, predict the Tuesday->close move from
  Tuesday-only information and bet the top decile in the direction of the move; judged on CLV in points.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import exchanges as X
from .odds import TEAM_ABBR

ROOT = Path(__file__).resolve().parents[2]
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
SKIP_BOOKS = {"pinnacle", "kalshi", "polymarket", "novig", "prophetx", "betopenly"}


def load_rules(name: str) -> dict:
    return json.loads((ROOT / f"{name}_rules.json").read_text())


def win_totals(season: int) -> dict:
    out = {}
    with open(ROOT / "data/research_futures/win_totals_2013_2026.csv") as f:
        for r in csv.DictReader(f):
            if int(r["season"]) == season:
                out[FIX.get(r["team"], r["team"])] = float(r["line_adj"])
    return out


def dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def best_side(lo: dict, side: str) -> dict | None:
    """Best (point, price) for a side among the allowed books (spreads_by_book is allowed-only)."""
    best = None
    for r in lo.get("spreads_by_book") or []:
        pt, pr = r.get(f"{side}_point"), r.get(f"{side}_price")
        if pt is None or pr is None:
            continue
        if best is None or (pt, dec(pr)) > (best["point"], dec(best["price"])):
            best = {"point": pt, "price": pr, "book": r.get("book")}
    return best


def kickoff(g: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(g["kickoff_utc"])
    except (KeyError, TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------- grading
def closing_point(bet: dict, files: list, cache: dict) -> float | None:
    """Consensus home point (sportsbooks, median) in our last odds snapshot before kickoff."""
    ko = datetime.fromisoformat(bet["kickoff_utc"])
    for f in reversed(files):
        if X._snap_time(f) >= ko:
            continue
        if f not in cache:
            cache[f] = X.load_events(f)
        for ev in cache[f]:
            if (TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])) != (bet["home_team"], bet["away_team"]):
                continue
            pts = []
            for bk in ev.get("bookmakers", []):
                if bk.get("key") in SKIP_BOOKS:
                    continue
                for m in bk.get("markets", []):
                    if m["key"] == "spreads":
                        for o in m["outcomes"]:
                            if o["name"] == ev["home_team"] and o.get("point") is not None:
                                pts.append(o["point"])
            if pts:
                return statistics.median(pts)
        # event absent from this snapshot: keep looking back
    return None


def grade(bet: dict, games: pd.DataFrame, files: list, cache: dict) -> dict:
    g = games[games["game_id"] == bet["game_id"]] if len(games) else games
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    r = g.iloc[0]
    margin = float(r["home_score"]) - float(r["away_score"])
    d = (margin + bet["point"]) if bet["side"] == "home" else (-margin + bet["point"])
    b = dict(bet, status="graded", final=f"{int(r['away_score'])}-{int(r['home_score'])}")
    b["result"] = "push" if d == 0 else ("win" if d > 0 else "loss")
    b["profit_units"] = 0.0 if d == 0 else round(bet["units"] * (dec(bet["price"]) - 1) if d > 0 else -bet["units"], 3)
    cp = closing_point(bet, files, cache)
    if cp is not None:
        close_side = cp if bet["side"] == "home" else -cp
        b["close_point"] = close_side
        b["clv_points"] = round(bet["point"] - close_side, 2)
    return b


def record(ledger: list[dict], r: dict, test: str) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    wins = sum(b["result"] == "win" for b in gr); losses = sum(b["result"] == "loss" for b in gr)
    staked = sum(b["units"] for b in gr if b["result"] != "push"); profit = sum(b["profit_units"] for b in gr)
    roi = profit / staked if staked else 0.0
    clv = [b["clv_points"] for b in gr if b.get("clv_points") is not None]
    mclv = sum(clv) / len(clv) if clv else 0.0
    sd = statistics.pstdev(clv) if len(clv) > 1 else 0.0
    if test == "clv":
        z = mclv / (sd / math.sqrt(len(clv))) if len(clv) > 1 and sd > 0 else 0.0
    else:
        n = wins + losses
        z = (wins - n * 0.5238) / math.sqrt(n * 0.5238 * 0.4762) if n else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2)) if (wins + losses) else 1.0
    checks = {"enough_bets": len(gr) >= r["validation"]["min_bets"], "significant": p < 0.05, "roi_ok": roi >= 0 if test == "clv" else roi > 0}
    return {"graded": len(gr), "wins": wins, "losses": losses, "units_staked": round(staked, 2), "profit_units": round(profit, 2),
            "roi": round(roi, 4), "avg_clv": round(mclv, 3), "clv_unit": "points", "p_value": round(p, 4),
            "test": "mean CLV points > 0" if test == "clv" else "cover rate > 52.38%",
            "checks": checks, "passed": all(checks.values()), "min_bets": r["validation"]["min_bets"],
            "clv_p_value": round(p, 4), "avg_clv_note": "points"}


def _run(name: str, pred: dict, games: pd.DataFrame, history_dir: Path, evaluate, test: str, now: datetime) -> dict:
    r = load_rules(name)
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    files = X.snapshot_files(history_dir); cache: dict = {}
    if len(games):
        ledger = [grade(b, games, files, cache) if b.get("status") == "open" else b for b in ledger]
    have = {b["game_id"] for b in ledger if b.get("status") != "void"}
    new = []
    for g in pred.get("upcoming", []):
        bet = evaluate(g, r, now)
        if bet and g["game_id"] not in have:
            new.append(bet); have.add(g["game_id"])
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = record(ledger, r, test)
    return {"track": name, "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"], "new": new,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}


def _bet(g: dict, r: dict, now: datetime, side: str, best: dict, extra: dict) -> dict:
    team = g[f"{side}_team"]
    return {"id": f"{g['game_id']}:{r['track']}:{side}", "track": r["track"], "rules_version": r["version"],
            "placed_at": now.isoformat(timespec="minutes"), "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "gameday": g["gameday"], "kickoff_utc": g["kickoff_utc"], "home_team": g["home_team"], "away_team": g["away_team"],
            "side": side, "team": team, "opponent": g["away_team" if side == "home" else "home_team"],
            "point": best["point"], "price": best["price"], "book": best["book"], "units": r["sizing"]["units"],
            "status": "open", **extra}


# ------------------------------------------------------------------------------------- preseason prior
def preseason_evaluate(g: dict, r: dict, now: datetime) -> dict | None:
    q = r["qualify"]
    reasons = g["preseason_prior_check"] = []
    if not (q["weeks"][0] <= int(g["week"]) <= q["weeks"][1]):
        reasons.append("only weeks 2-8"); return None
    lo = (g.get("context") or {}).get("live_odds") or {}
    wt = win_totals(int(g["season"]))
    h, a = wt.get(g["home_team"]), wt.get(g["away_team"])
    mkt = lo.get("consensus_home_margin")
    if h is None or a is None or mkt is None:
        reasons.append("no preseason win total or live spread"); return None
    pre = 2.48 * (0 if g.get("neutral_site") else 1) + 2.085 * (h - a)
    gap = pre - mkt
    side = "home" if gap > 0 else "away"
    g["preseason_prior"] = {"pre_home_margin": round(pre, 2), "market_home_margin": mkt, "gap": round(gap, 2), "side": side,
                            "team": g[f"{side}_team"]}
    if abs(gap) < q["min_gap_points"]:
        reasons.append(f"market within {abs(gap):.1f} pts of the preseason number (needs {q['min_gap_points']:g})"); return None
    ko = kickoff(g)
    if ko is None or ko - now > timedelta(hours=q["bet_within_hours_of_kickoff"]) or ko - now < timedelta(minutes=q["min_minutes_before_kickoff"]):
        reasons.append(f"bets only in the last {q['bet_within_hours_of_kickoff']} h before kickoff"); return None
    best = best_side(lo, side)
    if not best:
        reasons.append("no spread at your books"); return None
    qbc = (g.get("qb_change") or {}).get(side)
    return _bet(g, r, now, side, best, {"gap": round(gap, 2), "pre_home_margin": round(pre, 2), "market_home_margin": mkt,
                                         "edge": round(abs(gap) / 100, 4), "qb_changed_label": bool(qbc and abs(qbc) > 0.01)})


def process_preseason(pred: dict, games: pd.DataFrame, history_dir: Path, now: datetime | None = None) -> dict:
    return _run("preseason_prior", pred, games, history_dir, preseason_evaluate, "results", now or datetime.now(timezone.utc))


# ------------------------------------------------------------------------------------- tuesday move
def in_window(now: datetime, r: dict) -> bool:
    day, hm = r["qualify"]["window_utc"]
    h, m = (int(x) for x in hm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return now.weekday() == DAYS[day] and abs((now - t).total_seconds()) <= r["qualify"]["window_tolerance_minutes"] * 60


def _imp(a):
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def tuesday_features(g: dict, lo: dict) -> dict | None:
    bb = {k: v for k, v in (lo.get("by_book") or {}).items() if k not in SKIP_BOOKS}
    pts = [v["sp"][0] for v in bb.values() if v.get("sp")]
    if not pts or g.get("model_home_margin_no_inj") is None:
        return None
    pt = statistics.median(pts)
    mlh = [v["ml"][0] for v in bb.values() if v.get("ml")]; mla = [v["ml"][1] for v in bb.values() if v.get("ml")]
    ih, ia = (_imp(statistics.median(mlh)), _imp(statistics.median(mla))) if mlh and mla else (0.5, 0.5)
    pin = ((lo.get("by_book") or {}).get("pinnacle") or {}).get("sp")
    wt = win_totals(int(g["season"])); h, a = wt.get(g["home_team"]), wt.get(g["away_team"])
    pre = (2.48 + 2.085 * (h - a) + pt) if (int(g["week"]) <= 8 and h is not None and a is not None) else 0.0
    tot = ((lo.get("totals") or {}).get("consensus_total"))
    return {"pt": pt, "f_model": g["model_home_margin_no_inj"] + pt, "f_pre": pre, "f_pin": (pin[0] - pt) if pin else 0.0,
            "f_mlgap": ih / (ih + ia) - 0.5 + 0.03 * pt, "f_disp": statistics.stdev(pts) if len(pts) > 1 else 0.0,
            "f_abs": abs(pt), "f_tot": tot if tot is not None else 44.0}


def tuesday_evaluate(g: dict, r: dict, now: datetime) -> dict | None:
    q = r["qualify"]
    reasons = g["tuesday_move_check"] = []
    lo = (g.get("context") or {}).get("live_odds") or {}
    f = tuesday_features(g, lo)
    if f is None:
        reasons.append("no live spreads"); return None
    c = q["coef"]
    pred = c["const"] + sum(c[k] * f[k] for k in c if k != "const")
    side = "away" if pred > 0 else "home"
    g["tuesday_move"] = {"predicted_move_pts": round(pred, 2), "side": side, "team": g[f"{side}_team"], "tuesday_point": f["pt"]}
    if abs(pred) < q["min_abs_predicted_move_pts"]:
        reasons.append(f"predicted move {abs(pred):.2f} pts (needs {q['min_abs_predicted_move_pts']:g})")
    qc = g.get("qb_change") or {}; qs = g.get("qb_status") or {}
    if any(abs(qc.get(s) or 0) > 0.01 for s in ("home", "away")) or any((qs.get(s) or {}).get("play_prob", 1) < 1 for s in ("home", "away")):
        reasons.append("starting QB uncertain or changed")
    ko = kickoff(g)
    if ko is None or not (q["kickoff_hours"][0] <= (ko - now).total_seconds() / 3600 <= q["kickoff_hours"][1]):
        reasons.append("kickoff not 3.5-7 days away")
    if not in_window(now, r):
        reasons.append("only bets at the Tuesday 7:10am AZ run")
    if reasons:
        return None
    best = best_side(lo, side)
    if not best:
        reasons.append("no spread at your books"); return None
    return _bet(g, r, now, side, best, {"predicted_move_pts": round(pred, 2), "tuesday_point": f["pt"], "edge": round(abs(pred) / 100, 4)})


def process_tuesday(pred: dict, games: pd.DataFrame, history_dir: Path, now: datetime | None = None) -> dict:
    return _run("tuesday_move", pred, games, history_dir, tuesday_evaluate, "clv", now or datetime.now(timezone.utc))
