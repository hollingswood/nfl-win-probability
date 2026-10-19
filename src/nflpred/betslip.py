"""Links that open a paper bet at its book, ready to place: the phone alert's main button and a 'Bet at <book>' link next
to every open paper bet on both pages.

The Odds API returns sportsbook links when asked (includeLinks / includeSids, set in odds.fetch and
cfbpred.odds_live.fetch; same credit cost): an OUTCOME link adds that exact selection to the book's bet slip, so only
the stake is left to enter. Checked live 2026-10-19: DraftKings (?outcomes=), FanDuel (addToBetslip) and theScore Bet
(market_selection_id) all return one for every NFL moneyline, spread and total; Pinnacle returns none.

link_for(bet) -> {"url", "kind": "slip" | "game" | "book", "book", "note"} or None, from the newest saved odds snapshot:
  slip  the same side AT THE SAME NUMBER is on the slip (the price shown there is the live one: check it),
  game  the book's page for the game: the number moved, a prop, or no slip link for that outcome.
Never a slip at a different number: the alert would say one line and the slip another.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOK_KEYS = {"draftkings": "draftkings", "fanduel": "fanduel", "thescorebet": "espnbet", "espnbet": "espnbet",
             "thescore": "espnbet", "betmgm": "betmgm", "caesars": "williamhill_us", "williamhillus": "williamhill_us",
             "fanatics": "fanatics", "betrivers": "betrivers", "hardrockbet": "hardrockbet", "bet365": "bet365"}
SHORT = {"draftkings": "DK", "fanduel": "FanDuel", "espnbet": "theScore"}
DK_EVENT = "https://sportsbook.draftkings.com/event/{event}?outcomes={outcome}"


def _k(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def book_key(b: dict) -> str | None:
    return b.get("book_key") or BOOK_KEYS.get(_k(b.get("book")))


def latest_events(folder: Path) -> list[dict]:
    files = sorted(folder.glob("odds_*.json.gz"))
    for f in reversed(files[-3:]):            # newest snapshot that parses
        try:
            with gzip.open(f, "rt") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else data.get("events") or []
        except (OSError, ValueError):
            continue
    return []


def _market(b: dict) -> str:
    """Odds API market of a paper bet (same reading as placed.bet_key, which needs a game id)."""
    from .placed import SPREAD_TRACKS
    m, t, side = str(b.get("market") or "").lower(), str(b.get("track") or "").lower(), str(b.get("side") or "").lower()
    if side in ("over", "under") or m in ("total", "totals") or "total" in t:
        return "totals"
    if m in ("spread", "spreads") or any(k in t for k in SPREAD_TRACKS):
        return "spreads"
    return "h2h"


def _outcome_name(b: dict, ev: dict, mk: str) -> str | None:
    side = str(b.get("side") or "").lower()
    if mk == "totals":
        return side.capitalize() if side in ("over", "under") else None
    if side in ("home", "away"):
        return ev.get(f"{side}_team")
    t = _k(b.get("team"))
    for name in (ev.get("home_team"), ev.get("away_team")):
        if t and (_k(name) == t or _k(name).startswith(t)):
            return name
    return None


def _build_dk(bm: dict, oc: dict) -> str | None:
    """DraftKings slip from source ids when the feed gives ids but no link."""
    if bm.get("sid") and oc.get("sid"):
        return DK_EVENT.format(event=bm["sid"], outcome=oc["sid"])
    return None


def link_for(b: dict, events: list[dict], event_id: str | None = None) -> dict | None:
    bk = book_key(b)
    eid = event_id or b.get("event_id")
    if not bk or not eid:
        return None
    ev = next((e for e in events if e.get("id") == eid), None)
    if not ev:
        return None
    bm = next((x for x in ev.get("bookmakers", []) if x.get("key") == bk), None)
    name = (bm or {}).get("title") or b.get("book")
    out = {"book": name, "short": SHORT.get(bk, name)}
    if not bm:
        return None
    if b.get("player"):                 # props: the game's page (prop selections aren't in the main feed)
        return {**out, "url": bm["link"], "kind": "game", "note": None} if bm.get("link") else None
    mk = _market(b)
    m = next((x for x in bm.get("markets", []) if x.get("key") == mk), None)
    want = _outcome_name(b, ev, mk)
    pt = b.get("point")
    note = None
    if m and want:
        same = [o for o in m.get("outcomes", []) if o.get("name") == want]
        oc = next((o for o in same if mk == "h2h" or (pt is not None and o.get("point") == pt)), None)
        if oc:
            url = oc.get("link") or (bk == "draftkings" and _build_dk(bm, oc))
            if url:
                return {**out, "url": url, "kind": "slip", "note": None}
        elif same and mk != "h2h":
            note = f"{name} now has {same[0].get('point'):+g}" if mk == "spreads" and same[0].get("point") is not None \
                else f"{name} now has {same[0].get('point'):g}" if same[0].get("point") is not None else "the number moved"
    url = (m or {}).get("link") or bm.get("link")
    if url:
        return {**out, "url": url, "kind": "game", "note": note}
    return None


def event_id_of(b: dict, pred: dict | None) -> str | None:
    """Odds API event id: college bets carry it; NFL bets are matched through the game's live odds."""
    if b.get("event_id"):
        return b["event_id"]
    for g in (pred or {}).get("upcoming", []):
        if g.get("game_id") == b.get("game_id"):
            return ((g.get("context") or {}).get("live_odds") or {}).get("event_id")
    return None


def for_bet(b: dict, pred: dict | None = None, college: bool | None = None) -> dict | None:
    """Slip link for one bet, from the right sport's newest snapshot (fail-safe: None)."""
    try:
        college = college if college is not None else bool(b.get("event_id")) and not b.get("game_id")
        folder = ROOT / "history" / ("cfb" if college else "")
        return link_for(b, latest_events(folder), event_id_of(b, pred))
    except Exception as e:
        print("betslip: no link for", b.get("id"), e)
        return None


def label(s: dict) -> str:
    """Button text: 'Bet at DK' (slip ready), 'Open FanDuel' (game page: check the number)."""
    return f"Bet at {s['short']}" if s["kind"] == "slip" else f"Open {s['short']}"


def urls(blocks: dict, pred: dict | None, college: bool) -> dict:
    """paper bet id -> slip link, for every open/new bet in the given *_bets blocks (dashboard)."""
    events = latest_events(ROOT / "history" / ("cfb" if college else ""))
    out = {}
    for k, block in blocks.items():
        if not (k.endswith("_bets") and isinstance(block, dict)):
            continue
        for b in (block.get("open") or []) + (block.get("new") or []):
            if b.get("id") and b["id"] not in out and b.get("status", "open") == "open":
                try:
                    s = link_for(b, events, event_id_of(b, pred))
                except Exception:
                    s = None
                if s:
                    out[b["id"]] = s
    return out
