"""Tuesday star-unders props paper track (props_unders_rules.json).

At the Tuesday 14:10 UTC run (+-50 min) every upcoming game within 7 days is fetched once for rush yds, receptions and
receiving yds props (3 credits per game). Every player-market already quoted then gets one UNDER bet at the allowed book
with the best under (highest point, then best price) on the main line. Graded on results from nflverse weekly stats.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "props_unders_rules.json"
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def in_window(now: datetime, r: dict) -> bool:
    day, hm = r["qualify"]["window_utc"]
    h, m = (int(x) for x in hm.split(":"))
    if now.weekday() != DAYS[day]:
        return False
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return abs((now - t).total_seconds()) <= r["qualify"]["window_tolerance_minutes"] * 60


def parse(resp: dict, r: dict) -> list[dict]:
    """-> main-line quotes {market, book, title, player, point, over, under}."""
    lo, hi = r["qualify"]["main_line_price_range"]
    out = {}
    for bk in (resp or {}).get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") not in r["markets"]:
                continue
            for o in m.get("outcomes", []):
                name, who, pt, px = o.get("name"), o.get("description"), o.get("point"), o.get("price")
                if name not in ("Over", "Under") or not who or pt is None or px is None:
                    continue
                q = out.setdefault((m["key"], bk.get("key"), who, float(pt)), {"market": m["key"], "book": bk.get("key"),
                                   "title": bk.get("title", bk.get("key")), "player": who, "point": float(pt)})
                q["over" if name == "Over" else "under"] = px
    ok = [q for q in out.values() if q.get("over") is not None and q.get("under") is not None
          and lo <= q["over"] <= hi and lo <= q["under"] <= hi]
    main = {}  # one main line per book and player: the point priced closest to even
    for q in ok:
        k = (q["market"], q["book"], q["player"])
        bal = abs(1 / ML.decimal(q["over"]) - 1 / ML.decimal(q["under"]))
        if k not in main or bal < main[k][0]:
            main[k] = (bal, q)
    return [q for _, q in main.values()]


def picks(quotes: list[dict], allowed: set | None, r: dict) -> list[dict]:
    by = {}
    for q in quotes:
        by.setdefault((q["market"], q["player"]), []).append(q)
    out = []
    for (mk, player), qs in by.items():
        if len({q["book"] for q in qs}) < r["qualify"]["min_books_quoting"]:
            continue
        mine = [q for q in qs if allowed is None or q["book"] in allowed]
        if not mine:
            continue
        best = max(mine, key=lambda q: (q["point"], ML.decimal(q["under"])))
        out.append({"market": mk, "player": player, "point": best["point"], "price": best["under"], "book": best["title"],
                    "book_key": best["book"], "n_books": len({q["book"] for q in qs})})
    return out


def grade(bet: dict, games: pd.DataFrame, stats_for, now: datetime, r: dict) -> dict:
    from . import player_stats as PS
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    stats, snaps = stats_for(int(bet["season"]))
    status, val = PS.stat_value(bet["game_id"], bet["player"], stats, snaps, r["markets"][bet["market"]])
    if status == "open":
        if now.date() - pd.Timestamp(bet["gameday"]).date() > timedelta(days=10) and len(stats) and (stats["game_id"] == bet["game_id"]).any():
            return dict(bet, status="void", void_reason="no stat line and no snap counts 10 days after the game")
        return bet
    if status == "void":
        return dict(bet, status="void", void_reason=val)
    v = float(val)
    res = "win" if v < bet["point"] else "push" if v == bet["point"] else "loss"
    d = ML.decimal(bet["price"])
    return dict(bet, status="graded", result=res, final=f"{v:g}", stat=v,
                profit_units=round(bet["units"] * (d - 1), 3) if res == "win" else (0.0 if res == "push" else -bet["units"]))


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    pnl = [b["profit_units"] for b in gr if b["result"] != "push"]
    n = len(pnl)
    m = sum(pnl) / n if n else 0.0
    sd = (sum((x - m) ** 2 for x in pnl) / (n - 1)) ** 0.5 if n > 1 else 0.0
    p = 0.5 * math.erfc((m / (sd / math.sqrt(n))) / math.sqrt(2)) if n > 1 and sd > 0 else 1.0
    checks = {"enough_bets": len(gr) >= r["validation"]["min_bets"], "roi_positive_and_significant": m > 0 and p < r["validation"]["roi_positive_with_p_below"]}
    by = {}
    for b in gr:
        x = by.setdefault(b["market"], {"graded": 0, "wins": 0, "profit_units": 0.0})
        x["graded"] += 1; x["wins"] += b["result"] == "win"; x["profit_units"] = round(x["profit_units"] + b["profit_units"], 2)
    return {"graded": len(gr), "wins": sum(b["result"] == "win" for b in gr), "losses": sum(b["result"] == "loss" for b in gr),
            "profit_units": round(sum(b["profit_units"] for b in gr), 2), "roi": round(m, 4), "roi_p_value": round(p, 4),
            "avg_clv": 0.0, "clv_p_value": 1.0, "checks": {**checks, "roi_positive": m > 0, "clv_positive_and_significant": False},
            "passed": all(checks.values()), "min_bets": r["validation"]["min_bets"], "by_market": by, "test": "roi"}


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None, now: datetime | None = None,
            fetch=None, stats_for=None, allowed: set | None | bool = False) -> dict:
    from . import odds as O
    from .props_receptions import _default_stats_for
    r = r or load_rules()
    now = now or datetime.now(timezone.utc)
    allowed = O.load_allowed_books() if allowed is False else allowed
    path, spath = history_dir / r["ledger"], history_dir / r["state"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    state = json.loads(spath.read_text()) if spath.exists() else {}
    checked = state.setdefault("checked", {})
    if len(games) and any(b.get("status") == "open" for b in ledger):
        sf = stats_for or _default_stats_for()
        try:
            ledger = [grade(b, games, sf, now, r) if b.get("status") == "open" else b for b in ledger]
        except Exception as e:
            print("props unders: grading skipped:", e)
    have = {(b["game_id"], b["market"], b["player"]) for b in ledger}
    new, entries, calls = [], [], 0
    if fetch is not None and in_window(now, r):
        for g in pred.get("upcoming", []):
            eid = ((g.get("context") or {}).get("live_odds") or {}).get("event_id")
            try:
                kick = datetime.fromisoformat(g["kickoff_utc"])
            except (KeyError, TypeError, ValueError):
                continue
            if not eid or g["game_id"] in checked or not (now < kick <= now + timedelta(days=r["qualify"]["kickoff_within_days"])):
                continue
            try:
                resp = fetch(eid, markets=",".join(r["markets"]))
                calls += 1
            except Exception as e:
                print(f"props unders: {g['game_id']} fetch failed ({e})"); continue
            checked[g["game_id"]] = now.isoformat(timespec="minutes")
            entries.append({"event_id": eid, "game_id": g["game_id"], "kind": "tue_open", "fetched_at": now.isoformat(timespec="seconds"),
                            "commence_time": g.get("kickoff_utc"), "response": resp})
            from .player_stats import norm_name
            for c in picks(parse(resp, r), allowed, r):
                if (g["game_id"], c["market"], c["player"]) in have:
                    continue
                stat = {"rushing_yards": "rush yds", "receptions": "receptions", "receiving_yards": "rec yds"}[r["markets"][c["market"]]]
                new.append({"id": f"{g['game_id']}:under:{c['market']}:{norm_name(c['player']).replace(' ', '_')}", "track": "props_unders_tue",
                            "rules_version": r["version"], "placed_at": now.isoformat(timespec="minutes"), "game_id": g["game_id"],
                            "season": g["season"], "week": g["week"], "gameday": g["gameday"], "kickoff_utc": g.get("kickoff_utc"),
                            "event_id": eid, "market": c["market"], "player": c["player"], "side": "under", "point": c["point"],
                            "price": c["price"], "book": c["book"], "book_key": c["book_key"], "n_books": c["n_books"], "edge": None,
                            "team": f"{c['player']} Under {c['point']:g} {stat}", "opponent": f"{g['away_team']}@{g['home_team']}",
                            "stat_label": stat, "units": r["sizing"]["units"], "status": "open"})
    if entries:
        O.save_props_snapshot(history_dir, entries, now)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    spath.write_text(json.dumps(state, indent=1))
    return {"track": "props_unders_tue", "mode": "shadow", "rules_version": r["version"], "by_grade": [], "new": new,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-30:], "record": record(ledger, r), "api_calls": calls}
