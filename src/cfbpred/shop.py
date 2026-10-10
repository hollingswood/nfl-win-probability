"""College shop-vs-sharp paper track (cfb_shop_rules.json), frozen from scripts/research/cfb/shop_screen.py.

Every NCAAF odds snapshot: price each spread, total and moneyline quote at the allowed books against Pinnacle's no-vig
line at the same moment (any number, key-number pricing in cfbpred.dist). Bet the best quote that clears the market's
EV bar; one bet per game per market, locked at the first qualifying snapshot. Stake shown as quarter Kelly
(% of bankroll, capped); graded flat 1 unit for the record.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import data as D
from . import dist as DI
from .pipeline import team_matcher
from .tracks import HIST, dec, load, snap_time, snapshots

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "cfb_shop_rules.json"


RULES_V3 = ROOT / "cfb_shop_v3_rules.json"
RULES_APLUS = ROOT / "cfb_aplus_rules.json"


def books_of(r: dict) -> list[str]:
    """Rules books: a list, or "my_books" = the accounts in my_books.json (allowed_books)."""
    if r.get("books") == "my_books":
        return list(json.loads((ROOT / "my_books.json").read_text()).get("allowed_books", []))
    return list(r["books"])


def load_rules(path: Path = RULES) -> dict:
    return json.loads(path.read_text())


def _markets(bk: dict) -> dict:
    return {m["key"]: {o["name"]: o for o in m["outcomes"]} for m in bk.get("markets", [])}


def sharp(ev: dict) -> dict:
    """Pinnacle -> mu_m (expected home margin), mu_t (expected total), q_ml (no-vig home win)."""
    out: dict = {}
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != "pinnacle":
            continue
        mk = _markets(bk)
        h, a = ev["home_team"], ev["away_team"]
        sp, tt, h2 = mk.get("spreads", {}), mk.get("totals", {}), mk.get("h2h", {})
        try:
            if h in sp and a in sp and sp[h].get("point") is not None:
                out["mu_m"] = DI.mu_from_spread(sp[h]["point"], sp[h]["price"], sp[a]["price"])
                out["pin_spread"] = sp[h]["point"]
            if "Over" in tt and "Under" in tt and tt["Over"].get("point") is not None:
                out["mu_t"] = DI.mu_from_total(tt["Over"]["point"], tt["Over"]["price"], tt["Under"]["price"])
                out["pin_total"] = tt["Over"]["point"]
            if h in h2 and a in h2:              # v2 (2026-10-04): Shin vig removal; v1 used multiplicative
                from nflpred.devig import shin
                out["q_ml"] = shin(h2[h]["price"], h2[a]["price"])
        except Exception:
            pass
    return out


def probs(market: str, side: str, point, S: dict):
    """(win, push, lose) under the sharp line, or None if that market has no sharp quote."""
    if market == "spread" and "mu_m" in S:
        home_pt = point if side == "home" else -point
        return DI.spread_probs(S["mu_m"], home_pt, side)
    if market == "total" and "mu_t" in S:
        return DI.total_probs(S["mu_t"], point, side)
    if market == "ml" and "q_ml" in S:
        q = S["q_ml"] if side == "home" else 1 - S["q_ml"]
        return q, 0.0, 1 - q
    return None


def _haircut(w: float, l: float, price: float, market: str, r: dict) -> tuple[float, float]:
    """v3 sizing: shrink the win probability so EV drops by the market's average EV-to-CLV gap (2021-25:
    totals 1.9%, spreads 1.3%, moneylines 0). v1/v2 rules have no haircut."""
    h = (r["sizing"].get("ev_haircut") or {}).get(market, 0.0)
    if not h:
        return w, l
    d = dec(price)
    return max(0.0, w - h / d), min(1.0, l + h / d)


def kelly_pct(w: float, l: float, american: float, r: dict) -> float:
    b = dec(american) - 1
    f = (w * b - l) / (b * (w + l)) if w + l > 0 else 0.0
    return round(max(0.0, min(r["sizing"]["max_pct_bankroll"], 100 * r["sizing"]["kelly_fraction"] * f)), 2)


GRADES = [(0.04, "A+"), (0.02, "A"), (0.0, "B")]   # EV vs the sharp line; 2021-25 CLV rose with each step
                                                  # (output/research/cfb/grade_buckets.json)


def letter(edge: float | None) -> str | None:
    if edge is None:
        return None
    return next((g for th, g in GRADES if edge >= th), "C")


def offers(ev: dict, books: list[str], S: dict | None = None) -> dict:
    """Best quote per market and side among `books`, priced vs the sharp line: {market: {side: offer}}."""
    S = S if S is not None else sharp(ev)
    out: dict = {}
    for bk in ev.get("bookmakers", []):
        if bk.get("key") not in books:
            continue
        mk = _markets(bk)
        q = []
        for side, name in (("home", ev["home_team"]), ("away", ev["away_team"])):
            o = mk.get("spreads", {}).get(name)
            if o and o.get("point") is not None:
                q.append(("spread", side, o["point"], o["price"]))
            o = mk.get("h2h", {}).get(name)
            if o:
                q.append(("ml", side, None, o["price"]))
        for side, name in (("over", "Over"), ("under", "Under")):
            o = mk.get("totals", {}).get(name)
            if o and o.get("point") is not None:
                q.append(("total", side, o["point"], o["price"]))
        for market, side, point, price in q:
            pr = probs(market, side, point, S) if S else None
            e = round(DI.ev(*pr, price), 4) if pr else None
            cur = out.setdefault(market, {}).get(side)
            better = cur is None or (e is not None and (cur["edge"] is None or e > cur["edge"])) or \
                (e is None and cur["edge"] is None and (market == "ml" and price > cur["price"] or
                 market == "spread" and (point, price) > (cur["point"], cur["price"]) or
                 market == "total" and ((point < cur["point"]) if side == "over" else (point > cur["point"]))))
            if better:
                out[market][side] = {"point": point, "price": price, "book": bk.get("title", bk["key"]), "book_key": bk["key"],
                                     "edge": e, "grade": letter(e), "p_win": round(pr[0], 4) if pr else None,
                                     "p_push": round(pr[1], 4) if pr else None}
    return out


def candidates(ev: dict, r: dict) -> list[dict]:
    S = sharp(ev)
    if not S:
        return []
    lo, hi = r["qualify"]["price_range"]
    best: dict = {}
    books = books_of(r)
    for bk in ev.get("bookmakers", []):
        if bk.get("key") not in books:
            continue
        mk = _markets(bk)
        quotes = []
        for side, name in (("home", ev["home_team"]), ("away", ev["away_team"])):
            o = mk.get("spreads", {}).get(name)
            if o and o.get("point") is not None:
                quotes.append(("spread", side, o["point"], o["price"]))
            o = mk.get("h2h", {}).get(name)
            if o:
                quotes.append(("ml", side, None, o["price"]))
        for side, name in (("over", "Over"), ("under", "Under")):
            o = mk.get("totals", {}).get(name)
            if o and o.get("point") is not None:
                quotes.append(("total", side, o["point"], o["price"]))
        for market, side, point, price in quotes:
            if not (lo <= price <= hi):
                continue
            pr = probs(market, side, point, S)
            if pr is None:
                continue
            w, p, l = pr
            e = DI.ev(w, p, l, price)
            bar = (r["qualify"].get("min_ev_side") or {}).get(side, r["qualify"]["min_ev"][market])
            if e >= bar and (market not in best or e > best[market]["edge"]):
                team = ev["home_team"] if side == "home" else ev["away_team"] if side == "away" else side.capitalize()
                best[market] = {"market": market, "side": side, "team": team, "point": point, "price": price,
                                "book": bk.get("title", bk["key"]), "book_key": bk["key"], "edge": round(e, 4),
                                "p_win": round(w, 4), "p_push": round(p, 4), "kelly_pct": kelly_pct(*_haircut(w, l, price, market, r), price, r),
                                "sharp": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in S.items()}}
    return list(best.values())


def closing_probs(bet: dict, files: list[Path], cache: dict):
    ko = datetime.fromisoformat(bet["kickoff_utc"])
    for f in reversed(files):
        if snap_time(f) >= ko:
            continue
        if f not in cache:
            cache[f] = {e["id"]: e for e in load(f)}
        ev = cache[f].get(bet["event_id"])
        if ev:
            pr = probs(bet["market"], bet["side"], bet.get("point"), sharp(ev))
            if pr is not None:
                return pr
    return None


def settle(b: dict, margin: float, total: float) -> float:
    """> 0 win, 0 push, < 0 loss."""
    m, s, pt = b["market"], b["side"], b.get("point")
    if m == "ml":
        return margin if s == "home" else -margin
    if m == "spread":
        return margin + pt if s == "home" else -margin + pt
    return total - pt if s == "over" else pt - total


def grade(ledger: list[dict], files: list[Path]) -> list[dict]:
    open_ = [b for b in ledger if b.get("status") == "open"]
    if not open_:
        return ledger
    g = D.games(sorted({b["season"] for b in open_}))
    g = g[g.completed & g.margin.notna()]
    match = team_matcher(sorted(set(g.home) | set(g.away)))
    idx = {(r.home, r.away): r for r in g.itertuples()}
    cache: dict = {}
    out = []
    for b in ledger:
        if b.get("status") != "open":
            out.append(b); continue
        gg = idx.get((match(b["home"]), match(b["away"])))
        if gg is None or abs((gg.start - datetime.fromisoformat(b["kickoff_utc"])).total_seconds()) > 36 * 3600:
            out.append(b); continue
        x = settle(b, float(gg.margin), float(gg.total))
        nb = dict(b, status="graded", final=f"{int(gg.away_pts)}-{int(gg.home_pts)}",
                  result="push" if x == 0 else ("win" if x > 0 else "loss"))
        nb["profit_units"] = 0.0 if x == 0 else round(b["units"] * (dec(b["price"]) - 1) if x > 0 else -b["units"], 3)
        pr = closing_probs(b, files, cache)
        if pr is not None:
            nb["clv"] = round(DI.ev(*pr, b["price"]), 4)
        out.append(nb)
    return out


def counted(b: dict, r: dict) -> bool:
    """Bets that count for the track: this version's, plus (when the rules say so) every earlier bet at your books."""
    if b.get("rules_version") == r["version"]:
        return True
    return bool(r.get("count_earlier_bets_at_my_books")) and b.get("book_key") in books_of(r)


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and counted(b, r)]
    staked = sum(b["units"] for b in gr if b["result"] != "push")
    profit = sum(b["profit_units"] for b in gr)
    clv = [b["clv"] for b in gr if "clv" in b]
    m = sum(clv) / len(clv) if clv else 0.0
    sd = (sum((c - m) ** 2 for c in clv) / (len(clv) - 1)) ** 0.5 if len(clv) > 1 else 0.0
    p = 0.5 * math.erfc((m / (sd / math.sqrt(len(clv)))) / math.sqrt(2)) if len(clv) > 1 and sd > 0 else 1.0
    roi = profit / staked if staked else 0.0
    checks = {"enough_bets": len(gr) >= r["validation"]["min_bets"],
              "clv_positive_and_significant": m > 0 and p < r["validation"]["avg_clv_positive_with_p_below"], "roi_positive": roi > 0}
    by_market = {}
    for mk in ("spread", "total", "ml"):
        x = [b for b in gr if b["market"] == mk]
        c = [b["clv"] for b in x if "clv" in b]
        by_market[mk] = {"graded": len(x), "profit_units": round(sum(b["profit_units"] for b in x), 2),
                         "avg_clv": round(sum(c) / len(c), 4) if c else None}
    return {"graded": len(gr), "wins": sum(b["result"] == "win" for b in gr), "losses": sum(b["result"] == "loss" for b in gr),
            "pushes": sum(b["result"] == "push" for b in gr), "units_staked": round(staked, 2), "profit_units": round(profit, 2),
            "roi": round(roi, 4), "avg_clv": round(m, 4), "clv_p_value": round(p, 4), "checks": checks,
            "passed": all(checks.values()), "min_bets": r["validation"]["min_bets"], "by_market": by_market}


def process(now: datetime | None = None, rules_path: Path = RULES) -> dict:
    r = load_rules(rules_path)
    now = now or datetime.now(timezone.utc)
    path = HIST / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    files = snapshots()
    ledger = grade(ledger, files)
    new = []
    if files and now - snap_time(files[-1]) <= timedelta(hours=2):
        have = {(b["event_id"], b["market"]) for b in ledger if counted(b, r)}
        season = now.year if now.month >= 7 else now.year - 1
        q = r["qualify"]
        try:
            mine = set(json.loads((ROOT / "my_books.json").read_text()).get("allowed_books", []))
        except Exception:
            mine = set()
        for ev in load(files[-1]):
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if not (timedelta(minutes=q["min_minutes_before_kickoff"]) <= ko - now <= timedelta(days=q["max_days_before_kickoff"])):
                continue
            for c in candidates(ev, r):
                if (ev["id"], c["market"]) in have:
                    continue
                have.add((ev["id"], c["market"]))
                vtag = f":v{r['version']}" if r.get("books") == "my_books" else ""   # keeps ids unique across versions
                new.append({"id": f"{r.get('id_prefix', 'cfbshop')}:{ev['id']}:{c['market']}:{c['side']}{vtag}", "track": r["track"], "rules_version": r["version"],
                            "placed_at": now.isoformat(timespec="minutes"), "event_id": ev["id"], "season": season,
                            "kickoff_utc": ko.isoformat(), "home": ev["home_team"], "away": ev["away_team"],
                            "gameday": ko.date().isoformat(), "units": r["sizing"]["flat_units_for_grading"], "status": "open",
                            "your_book": c["book_key"] in mine, **c})
    ledger += new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1))
    cur = [b for b in ledger if counted(b, r)]   # bets at books you don't have are kept in the file but never shown or counted
    return {"track": r["track"], "mode": "shadow", "rules_version": r["version"], "new": new,
            "books": books_of(r), "open": [b for b in cur if b.get("status") == "open"],
            "recent_graded": [b for b in cur if b.get("status") == "graded"][-40:], "record": record(ledger, r)}
