"""Receptions line-shopping props paper track (props_receptions_rules.json; research rule rec_shop_early).

At each game's early snapshot (Friday 21:40 UTC for Sunday-UTC kickoffs, kickoff - 24 h otherwise, +-50 min; the
hourly watch runs at :23) we fetch that event's player_receptions odds once (~1 credit) and, for every allowed-book
over/under quote on the main line (both prices -200..+170), compare its price with the leave-one-out median no-vig
probability of >= 3 OTHER books quoting the same player and point. Best-EV side/book per player; bet it (1 unit) if
2% <= EV < 50%. For games with open bets the event is fetched again 10-90 min before kickoff (the close, for CLV).
Raw responses: history/props_<UTC>.json.gz. Results from nflverse weekly player stats (player_stats.py).
"""
from __future__ import annotations

import gzip
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "props_receptions_rules.json"
MARKET = "player_receptions"
CLOSE_MIN, CLOSE_MAX = 10, 90          # minutes before kickoff for the closing fetch
DAYS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def _imp(a: float) -> float:
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


# ------------------------------------------------------------------------------------------------ windows
def early_time(kick: datetime, r: dict) -> datetime:
    """Research 'early' snapshot: Friday 21:40 UTC before a kickoff on a Sunday (UTC), else kickoff - 24 h."""
    q = r["qualify"]
    if kick.weekday() == DAYS["Sun"]:
        day, hm = q["friday_window_utc"]
        h, m = (int(x) for x in hm.split(":"))
        back = (kick.weekday() - DAYS[day]) % 7
        return (kick - timedelta(days=back)).replace(hour=h, minute=m, second=0, microsecond=0)
    return kick - timedelta(hours=q["other_games_hours_before_kickoff"])


def in_early_window(now: datetime, kick: datetime, r: dict) -> bool:
    return abs((now - early_time(kick, r)).total_seconds()) <= r["qualify"]["window_tolerance_minutes"] * 60


def in_close_window(now: datetime, kick: datetime) -> bool:
    return CLOSE_MIN * 60 <= (kick - now).total_seconds() <= CLOSE_MAX * 60


# ------------------------------------------------------------------------------------------------ quotes
def parse_event(resp: dict, r: dict) -> list[dict]:
    """Odds API event response -> main-line O/U quotes {book, title, player, point, over, under, p_nv}."""
    lo, hi = r["qualify"]["main_line_price_range"]
    out = {}
    for bk in (resp or {}).get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != MARKET:
                continue
            for o in m.get("outcomes", []):
                name, who, pt, px = o.get("name"), o.get("description"), o.get("point"), o.get("price")
                if name not in ("Over", "Under") or not who or pt is None or px is None:
                    continue
                k = (bk.get("key"), who, float(pt))
                q = out.setdefault(k, {"book": bk.get("key"), "title": bk.get("title", bk.get("key")), "player": who,
                                       "point": float(pt)})
                q["over" if name == "Over" else "under"] = int(px) if float(px).is_integer() else float(px)
    quotes = []
    for q in out.values():
        if q.get("over") is None or q.get("under") is None:
            continue
        if not (lo <= q["over"] <= hi and lo <= q["under"] <= hi):
            continue  # milestone / alternate quotes (research main-line filter)
        io, iu = _imp(q["over"]), _imp(q["under"])
        q["p_nv"] = io / (io + iu)
        quotes.append(q)
    return quotes


def candidates(quotes: list[dict], allowed: set | None, r: dict) -> list[dict]:
    """Allowed-book quotes vs the leave-one-out median no-vig P(over) of >= 3 other books at the same point."""
    q = r["qualify"]
    by = {}
    for x in quotes:
        by.setdefault((x["player"], x["point"]), []).append(x)
    out = []
    for x in quotes:
        if allowed is not None and x["book"] not in allowed:
            continue
        others = [y["p_nv"] for y in by[(x["player"], x["point"])] if y is not x]
        if len(others) < q["min_other_books_same_point"]:
            continue
        p = statistics.median(others)
        for side, price, ps in (("over", x["over"], p), ("under", x["under"], 1 - p)):
            out.append({"player": x["player"], "point": x["point"], "side": side, "price": price, "book": x["title"],
                        "book_key": x["book"], "p_fair": round(ps, 4), "n_other": len(others),
                        "ev": ps * ML.decimal(price) - 1})   # unrounded: thresholds compare the exact EV
    return out


def select(cands: list[dict], r: dict) -> list[dict]:
    """Best-EV quote per player; kept only if min_ev <= EV < max_ev (research `pick`)."""
    q = r["qualify"]
    best = {}
    for c in sorted(cands, key=lambda c: -c["ev"]):
        best.setdefault(c["player"], c)
    return [c for c in best.values() if q["min_ev"] <= c["ev"] < q["max_ev"]]


def evaluate_event(game: dict, resp: dict, r: dict, now: datetime, allowed: set | None) -> list[dict]:
    quotes = parse_event(resp, r)
    picks = select(candidates(quotes, allowed, r), r)
    from .player_stats import norm_name
    bets = []
    for c in sorted(picks, key=lambda c: -c["ev"]):
        bets.append({"id": f"{game['game_id']}:rec:{norm_name(c['player']).replace(' ', '_')}",
                     "track": "props_receptions", "rules_version": r["version"],
                     "placed_at": now.isoformat(timespec="minutes"), "game_id": game["game_id"],
                     "season": game["season"], "week": game["week"], "gameday": game["gameday"],
                     "kickoff_utc": game.get("kickoff_utc"), "event_id": resp.get("id"), "market": MARKET,
                     "player": c["player"], "side": c["side"], "point": c["point"], "price": c["price"],
                     "book": c["book"], "book_key": c["book_key"], "p_fair": c["p_fair"], "n_other": c["n_other"],
                     "edge": round(c["ev"], 4), "team": f"{c['player']} {c['side'].title()} {c['point']:g}",
                     "opponent": f"{game['away_team']}@{game['home_team']}", "units": r["sizing"]["units"],
                     "status": "open"})
    return bets


# ------------------------------------------------------------------------------------------------ snapshots
def load_snapshots(history_dir: Path) -> list[dict]:
    out = []
    for f in sorted(history_dir.glob("props_*.json.gz")):
        try:
            out += json.loads(gzip.open(f, "rt").read())
        except Exception:
            continue
    return out


def closing_prob(bet: dict, snaps: list[dict], r: dict) -> tuple[float | None, str | None]:
    """Median no-vig P(our side) among books quoting the same point in our last props snapshot of the event
    taken before kickoff (<= 3 h)."""
    try:
        kick = datetime.fromisoformat(bet["kickoff_utc"])
    except (KeyError, TypeError, ValueError):
        return None, None
    best = None
    for s in snaps:
        if s.get("event_id") != bet.get("event_id") and s.get("game_id") != bet["game_id"]:
            continue
        try:
            ts = datetime.fromisoformat(s["fetched_at"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts < kick and (kick - ts).total_seconds() <= 3 * 3600 and (best is None or ts > best[0]):
            best = (ts, s)
    if best is None:
        return None, None
    from .player_stats import norm_name
    me = norm_name(bet["player"])
    ps = [q["p_nv"] for q in parse_event(best[1].get("response"), r)
          if q["point"] == bet["point"] and norm_name(q["player"]) == me]
    if not ps:
        return None, f"no same-point quote at the close ({best[0].isoformat(timespec='minutes')})"
    p = statistics.median(ps)
    return (p if bet["side"] == "over" else 1 - p), \
        f"same-point no-vig median of {len(ps)} books at {best[0].isoformat(timespec='minutes')}"


def grade(bet: dict, games: pd.DataFrame, snaps: list[dict], r: dict, stats_for, now: datetime) -> dict:
    """stats_for(season) -> (weekly stats, snap counts). Leaves the bet open until the result is known."""
    from . import player_stats as PS
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    stats, snapc = stats_for(int(bet["season"]))
    status, val = PS.receptions(bet["game_id"], bet["player"], stats, snapc)
    if status == "open":
        stale = now.date() - pd.Timestamp(bet["gameday"]).date() > timedelta(days=10)
        if not (stale and len(stats) and (stats["game_id"] == bet["game_id"]).any()):
            return dict(bet, grade_note=val)
        status, val = "void", "no stat line and no snap counts 10 days after the game"
    if status == "void":
        return dict(bet, status="void", void_reason=val)
    rec = int(val)
    res = "win" if (rec > bet["point"] if bet["side"] == "over" else rec < bet["point"]) else \
        "push" if rec == bet["point"] else "loss"
    out = dict(bet, status="graded", result=res, final=f"{rec} rec", receptions=rec)
    out.pop("grade_note", None)
    d = ML.decimal(bet["price"])
    out["profit_units"] = round(bet["units"] * (d - 1), 3) if res == "win" else (0.0 if res == "push" else -bet["units"])
    p, src = closing_prob(bet, snaps, r)
    if p is not None:
        out["closing_prob"] = round(p, 4)
        out["clv"] = round(p * d - 1, 4)
    out["clv_source"] = src or "no closing props snapshot"
    return out


def _default_stats_for():
    from . import player_stats as PS
    cache = {}

    def f(season):
        if season not in cache:
            cache[season] = (PS.load_week_stats(season), PS.load_snaps(season))
        return cache[season]
    return f


# ------------------------------------------------------------------------------------------------ process
def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None, now: datetime | None = None,
            fetch=None, stats_for=None, allowed: set | None | bool = False) -> dict:
    """One pipeline run. `fetch(event_id) -> response` (None = offline: no API calls)."""
    from . import odds as O
    r = r or load_rules()
    now = now or datetime.now(timezone.utc)
    allowed = O.load_allowed_books() if allowed is False else allowed
    path, spath = history_dir / r["ledger"], history_dir / r["state"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    state = json.loads(spath.read_text()) if spath.exists() else {}
    checked = state.setdefault("early_checked", {})
    # 1. grade finished games (full runs only: `games` has results)
    if len(games) and any(b.get("status") == "open" for b in ledger):
        snaps = load_snapshots(history_dir)
        sf = stats_for or _default_stats_for()
        try:
            ledger = [grade(b, games, snaps, r, sf, now) if b.get("status") == "open" else b for b in ledger]
        except Exception as e:  # stats download failed: try again next run
            print("props receptions: grading skipped:", e)
    have = {(b["game_id"], b.get("player")) for b in ledger if b.get("status") != "void"}
    open_games = {b["game_id"] for b in ledger if b.get("status") == "open"}
    entries, new, calls = [], [], 0
    for g in pred.get("upcoming", []):
        eid = ((g.get("context") or {}).get("live_odds") or {}).get("event_id")
        try:
            kick = datetime.fromisoformat(g["kickoff_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        view = {"early_check_at": early_time(kick, r).isoformat(timespec="minutes"),
                "checked_at": checked.get(g["game_id"]), "bets": 0}
        g["props_receptions"] = view
        if not eid or fetch is None or now >= kick:
            continue
        kind = "early" if g["game_id"] not in checked and in_early_window(now, kick, r) else \
            "close" if g["game_id"] in open_games and in_close_window(now, kick) else None
        if not kind:
            continue
        try:
            resp = fetch(eid)
            calls += 1
        except Exception as e:
            print(f"props receptions: {g['game_id']} fetch failed ({e})")
            continue
        entries.append({"event_id": eid, "game_id": g["game_id"], "kind": kind,
                        "fetched_at": now.isoformat(timespec="seconds"), "commence_time": g.get("kickoff_utc"),
                        "response": resp})
        if kind == "early":
            checked[g["game_id"]] = view["checked_at"] = now.isoformat(timespec="minutes")
            bets = [b for b in evaluate_event(g, resp, r, now, allowed) if (b["game_id"], b["player"]) not in have]
            view["quoted_players"] = len({q["player"] for q in parse_event(resp, r)})
            new += bets
    if entries:
        O.save_props_snapshot(history_dir, entries, now)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    spath.write_text(json.dumps(state, indent=1))
    for g in pred.get("upcoming", []):
        v = g.get("props_receptions")
        if v is not None:
            bs = [b for b in ledger if b["game_id"] == g["game_id"] and b.get("status") != "void"]
            v["bets"] = len(bs)
            v["picks"] = [f"{b['team']} {b['price']:+d} {b['book']}" for b in bs][:12]
    rec = ML.record(ledger, r)
    return {"track": "props_receptions", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "by_grade": [], "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec,
            "api_calls": calls}
