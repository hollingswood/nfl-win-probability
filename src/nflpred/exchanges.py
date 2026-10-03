"""Exchanges / prediction markets: fee-adjusted prices and the exchange_value paper track.

Venues come from exchange_value_rules.json: Kalshi, Robinhood (Kalshi's book + Robinhood's fees),
Polymarket US, ProphetX (bettable) and Novig, BetOpenly (reference only). Every quote is turned into
the all-in cost of a $1 contract after that venue's fees, then compared with Pinnacle's no-vig price
at the same point.

* attach(pred, ...)  -> g["exchanges"] on each upcoming game (dashboard: Kalshi/Robinhood prices etc.)
* process(pred, ...) -> paper track: bet when a bettable venue beats Pinnacle fair by >= 2% after fees.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import bets as ML
from .odds import TEAM_ABBR, load_allowed_books

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "exchange_value_rules.json"
MARKETS = ("h2h", "spreads", "totals")


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def implied(american: float) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def american(dec: float) -> int:
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def all_in_cost(p: float, venue: dict) -> float:
    """Cost of a contract paying $1, after fees (p = quoted price as a probability)."""
    f = venue["fee"]
    if f == "kalshi_taker":
        return p + 0.07 * p * (1 - p)
    if f == "robinhood":
        return p + 0.01 + min(0.01, 0.07 * p * (1 - p))
    if f == "polymarket_taker":
        return p + 0.05 * p * (1 - p)
    if f == "winnings_commission":
        return 1 / (1 + (1 / p - 1) * (1 - venue.get("commission", 0.0)))
    return p


# ------------------------------------------------------------------------------------- snapshots
def _snap_time(path: Path) -> datetime:
    return datetime.strptime(path.name.split("odds_")[1][:15], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)


def snapshot_files(history_dir: Path) -> list[Path]:
    return sorted((p for p in history_dir.glob("odds_20*.json*") if p.name[5:6].isdigit()), key=_snap_time)


def load_events(path: Path) -> list[dict]:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("events", [])


def _ts(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    except ValueError:
        return None


def _side(mk: str, o: dict, ev: dict) -> str | None:
    if mk == "totals":
        return {"Over": "over", "Under": "under"}.get(o["name"])
    return "home" if o["name"] == ev["home_team"] else "away" if o["name"] == ev["away_team"] else None


def _pinnacle_fair(ev: dict) -> dict:
    """{market: {(side, point): fair prob}} from Pinnacle, both sides required."""
    out = {}
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != "pinnacle":
            continue
        for m in bk.get("markets", []):
            os_ = m.get("outcomes", [])
            if m["key"] not in MARKETS or len(os_) != 2:
                continue
            ps = {(_side(m["key"], o, ev), o.get("point")): implied(o["price"]) for o in os_}
            if None in {k[0] for k in ps}:
                continue
            z = sum(ps.values())
            out[m["key"]] = {k: v / z for k, v in ps.items()}
            out[m["key"] + "_updated"] = m.get("last_update")
    return out


def quotes(ev: dict, r: dict, snap_at: datetime, allowed: set | None = None) -> dict:
    """Per market: fair map and every venue's fee-adjusted quote per side, plus the best allowed sportsbook."""
    fair = _pinnacle_fair(ev)
    books = {bk["key"]: {m["key"]: m for m in bk.get("markets", [])} for bk in ev.get("bookmakers", [])}
    max_age = timedelta(minutes=r["qualify"]["max_quote_age_minutes"])
    out = {}
    for mk in MARKETS:
        rows = []
        for vk, v in r["venues"].items():
            m = books.get(v["source"], {}).get(mk)
            if not m or len(m.get("outcomes", [])) != 2:
                continue
            upd = _ts(m.get("last_update"))
            fresh = upd is None or snap_at - upd <= max_age
            for o in m["outcomes"]:
                side = _side(mk, o, ev)
                if side is None:
                    continue
                p = implied(o["price"])
                cost = all_in_cost(p, v)
                f = (fair.get(mk) or {}).get((side, o.get("point")))
                rows.append({"venue": vk, "label": v["label"], "bettable": v["bettable"], "side": side,
                             "point": o.get("point"), "quoted_cents": round(100 * p, 1),
                             "all_in_cents": round(100 * cost, 2), "price": american(1 / cost),
                             "fair": round(f, 4) if f is not None else None,
                             "ev": round(f / cost - 1, 4) if f is not None else None, "fresh": fresh})
        best_book = {}
        for bk, mks in books.items():
            if allowed is not None and bk not in allowed:
                continue
            for o in (mks.get(mk) or {}).get("outcomes", []):
                k = (_side(mk, o, ev), o.get("point"))
                d = ML.decimal(o["price"])
                if k[0] and d > best_book.get(k, (0, ""))[0]:
                    best_book[k] = (d, bk)
        for row in rows:
            bb = best_book.get((row["side"], row["point"]))
            if bb:
                row["best_book"], row["best_book_price"] = bb[1], american(bb[0])
                if row["fair"] is not None:
                    row["best_book_ev"] = round(row["fair"] * bb[0] - 1, 4)
        out[mk] = rows
    p_upd = _ts(fair.get("h2h_updated"))
    out["pinnacle_fresh"] = p_upd is None or snap_at - p_upd <= max_age
    return out


def _game_key(ev: dict) -> tuple | None:
    h, a = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
    return (h, a) if h and a else None


def _match(game: dict, events: list[dict]) -> dict | None:
    for ev in events:
        if _game_key(ev) == (game["home_team"], game["away_team"]):
            ko = _ts(ev.get("commence_time"))
            if ko and abs((ko.date() - pd.Timestamp(game["gameday"]).date()).days) <= 1:
                return ev
    return None


def _display(q: dict) -> dict:
    """Compact card view: Kalshi + Robinhood moneylines and the best bettable venue per side/market."""
    view = {}
    for row in q.get("h2h", []):
        if row["venue"] in ("kalshi", "robinhood"):
            view.setdefault(row["venue"], {})[row["side"]] = {k: row[k] for k in ("quoted_cents", "all_in_cents", "price", "ev")}
    best = {}
    for mk in MARKETS:
        for row in q.get(mk, []):
            if not row["bettable"] or row["ev"] is None:
                continue
            k = f"{mk}:{row['side']}"
            if k not in best or row["ev"] > best[k]["ev"]:
                best[k] = {kk: row.get(kk) for kk in ("label", "side", "point", "price", "ev", "best_book", "best_book_price", "best_book_ev")}
    view["best"] = best
    return view


def attach(pred: dict, history_dir: Path, r: dict | None = None) -> None:
    """Attach fee-adjusted exchange prices from the newest snapshot to every upcoming game."""
    r = r or load_rules()
    files = snapshot_files(history_dir)
    if not files:
        return
    snap_at = _snap_time(files[-1])
    events = load_events(files[-1])
    allowed = load_allowed_books()
    for g in pred.get("upcoming", []):
        ev = _match(g, events)
        if ev:
            g["exchanges"] = {"checked_at": snap_at.isoformat(timespec="minutes"), **_display(quotes(ev, r, snap_at, allowed))}


# ------------------------------------------------------------------------------------- paper track
def _closing_fair(bet: dict, files: list[Path], cache: dict) -> float | None:
    """Pinnacle no-vig prob of the bet's side at the same point, last snapshot before kickoff."""
    ko = _ts(bet["kickoff_utc"])
    for f in reversed(files):
        t = _snap_time(f)
        if t >= ko:
            continue
        if f not in cache:
            cache[f] = {ev.get("id"): ev for ev in load_events(f)}
        ev = cache[f].get(bet["event_id"])
        if not ev:
            continue
        fair = _pinnacle_fair(ev).get(bet["market"])
        if fair is None:
            continue  # Pinnacle missing in this snapshot: look further back
        return fair.get((bet["side"], bet.get("point")))  # None if the point moved
    return None


def grade(bet: dict, games: pd.DataFrame, files: list[Path], cache: dict) -> dict:
    g = games[games["game_id"] == bet["game_id"]] if len(games) else games
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    row = g.iloc[0]
    hs, as_ = float(row["home_score"]), float(row["away_score"])
    pt = bet.get("point") or 0.0
    if bet["market"] == "h2h":
        diff = (hs - as_) if bet["side"] == "home" else (as_ - hs)
    elif bet["market"] == "spreads":
        diff = (hs + pt - as_) if bet["side"] == "home" else (as_ + pt - hs)
    else:
        diff = (hs + as_ - pt) if bet["side"] == "over" else (pt - hs - as_)
    dec = 1 / bet["all_in_cost"]
    b = dict(bet, status="graded", final=f"{int(as_)}-{int(hs)}")
    b["result"] = "push" if diff == 0 else ("win" if diff > 0 else "loss")
    b["profit_units"] = 0.0 if diff == 0 else round(bet["units"] * (dec - 1) if diff > 0 else -bet["units"], 3)
    cf = _closing_fair(bet, files, cache)
    if cf is not None:
        b["closing_prob"] = round(cf, 4)
        b["clv"] = round(dec * cf - 1, 4)
    return b


def evaluate(game: dict, ev: dict, r: dict, snap_at: datetime, now: datetime, have: set) -> list[dict]:
    q_ = r["qualify"]
    ko = _ts(ev.get("commence_time"))
    if not ko or ko - now > timedelta(days=q_["kickoff_within_days"]) or ko - now < timedelta(minutes=q_["min_minutes_before_kickoff"]):
        return []
    q = quotes(ev, r, snap_at, load_allowed_books())
    if not q.get("pinnacle_fresh"):
        return []
    lo, hi = q_["fair_prob_range"]
    out = []
    for mk in q_["markets"]:
        if f"{game['game_id']}:{mk}" in have:
            continue
        cands = [x for x in q.get(mk, []) if x["bettable"] and x["fresh"] and x["ev"] is not None
                 and q_["min_ev"] <= x["ev"] < q_["max_ev"] and lo <= x["fair"] <= hi]
        if not cands:
            continue
        x = max(cands, key=lambda c: c["ev"])
        if mk == "totals":
            label = f"{game['away_team']}@{game['home_team']} {x['side'].upper()} {x['point']}"
        else:
            team = game[f"{x['side']}_team"]
            label = team + ("" if mk == "h2h" else f" {x['point']:+g}")
        out.append({"id": f"{game['game_id']}:exch:{mk}:{x['side']}", "track": "exchange_value",
                    "rules_version": r["version"], "placed_at": now.isoformat(timespec="minutes"),
                    "game_id": game["game_id"], "season": game["season"], "week": game["week"],
                    "gameday": game["gameday"], "kickoff_utc": ko.isoformat(), "event_id": ev.get("id"),
                    "market": mk, "side": x["side"], "point": x["point"], "team": label,
                    "venue": x["venue"], "book": x["label"], "quoted_cents": x["quoted_cents"],
                    "all_in_cost": round(x["all_in_cents"] / 100, 4), "price": x["price"], "fair": x["fair"],
                    "edge": x["ev"], "best_book": x.get("best_book"), "best_book_price": x.get("best_book_price"),
                    "best_book_ev": x.get("best_book_ev"), "units": r["sizing"]["units"], "status": "open"})
    return out


def process(pred: dict, games: pd.DataFrame, history_dir: Path, r: dict | None = None,
            now: datetime | None = None) -> dict:
    r = r or load_rules()
    now = now or datetime.now(timezone.utc)
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    files = snapshot_files(history_dir)
    cache: dict = {}
    if len(games):
        ledger = [grade(b, games, files, cache) if b.get("status") == "open" else b for b in ledger]
    new = []
    if files:
        snap_at = _snap_time(files[-1])
        if now - snap_at <= timedelta(hours=2):  # only act on a fresh snapshot
            events = load_events(files[-1])
            have = {f"{b['game_id']}:{b['market']}" for b in ledger if b.get("status") != "void"}
            for g in pred.get("upcoming", []):
                ev = _match(g, events)
                if ev:
                    new += evaluate(g, ev, r, snap_at, now, have)
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = ML.record(ledger, r)
    return {"track": "exchange_value", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"],
            "new": new, "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec,
            "by_venue": _by_venue(ledger)}


def _by_venue(ledger: list[dict]) -> list[dict]:
    out = {}
    for b in ledger:
        if b.get("status") != "graded":
            continue
        v = out.setdefault(b["venue"], {"venue": b["venue"], "bets": 0, "profit_units": 0.0, "clv": []})
        v["bets"] += 1
        v["profit_units"] = round(v["profit_units"] + b["profit_units"], 3)
        if "clv" in b:
            v["clv"].append(b["clv"])
    return [{**v, "clv": round(sum(v["clv"]) / len(v["clv"]), 4) if v["clv"] else None} for v in out.values()]
