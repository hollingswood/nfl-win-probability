"""Futures line-shopping paper track (futures_shop_rules.json): best Super Bowl price at any book vs the consensus fair price
(median of each book's power-devigged probabilities). Runs on each new futures snapshot (history/futures/)."""
from __future__ import annotations

import gzip
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
RULES = ROOT / "futures_shop_rules.json"
AZ_BOOKS = {"draftkings", "fanduel", "espnbet", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"}


def implied(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def power_devig(probs: dict) -> dict:
    """Scale p_i^k so they sum to 1 (k > 1 shrinks longshots more than favorites)."""
    lo, hi = 1.0, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(p ** k for p in probs.values()) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return {t: p ** k for t, p in probs.items()}


def snapshot_view(snap: dict, market: str) -> dict:
    evs = (snap.get("markets") or {}).get(market) or []
    ev = evs[0] if isinstance(evs, list) and evs else None
    if not ev:
        return {}
    fair_by_book, best = {}, {}
    for bk in ev.get("bookmakers", []):
        m = next((m for m in bk.get("markets", []) if m["key"] == "outrights"), None)
        if not m:
            continue
        px = {o["name"]: o["price"] for o in m["outcomes"]}
        fair_by_book[bk["key"]] = power_devig({t: implied(p) for t, p in px.items()})
        for t, p in px.items():
            if bk["key"] not in AZ_BOOKS:       # fair price uses every book; bets only at Arizona books
                continue
            if t not in best or p > best[t]["price"]:
                best[t] = {"price": p, "book": bk.get("title", bk["key"])}
    teams = set().union(*[set(f) for f in fair_by_book.values()]) if fair_by_book else set()
    out = {}
    for t in teams:
        ps = [f[t] for f in fair_by_book.values() if t in f]
        if len(ps) < 2 or t not in best:
            continue
        fair = statistics.median(ps)
        out[t] = {"fair": round(fair, 4), "n_books": len(ps), **best[t], "ev": round(fair * dec(best[t]["price"]) - 1, 4)}
    return out


def process(now: datetime | None = None) -> dict:
    r = json.loads(RULES.read_text())
    now = now or datetime.now(timezone.utc)
    files = sorted((HIST / "futures").glob("futures_*.json.gz"))
    path = HIST / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    if not files:
        return {"track": "futures_shop", "new": [], "open": ledger, "board": {}}
    with gzip.open(files[-1], "rt") as f:
        snap = json.load(f)
    view = snapshot_view(snap, r["market"])
    q, new = r["qualify"], []
    for t, v in sorted(view.items(), key=lambda kv: -kv[1]["ev"]):
        recent = [b for b in ledger if b["team"] == t and now - datetime.fromisoformat(b["placed_at"]) < timedelta(days=q["one_bet_per_team_per_days"])]
        if v["n_books"] >= q["min_books"] and v["ev"] >= q["min_ev"] and v["fair"] >= q["min_fair_prob"] and not recent:
            new.append({"id": f"fut:{t}:{now:%Y%m%d}", "track": "futures_shop", "rules_version": r["version"], "market": "super_bowl_winner",
                        "team": t, "price": v["price"], "book": v["book"], "fair": v["fair"], "edge": v["ev"], "n_books": v["n_books"],
                        "units": r["sizing"]["units"], "placed_at": now.isoformat(timespec="minutes"), "snapshot": files[-1].name, "status": "open"})
    ledger += new
    path.write_text(json.dumps(ledger, indent=1))
    board = dict(sorted(view.items(), key=lambda kv: -kv[1]["ev"])[:10])
    return {"track": "futures_shop", "mode": "shadow", "rules_version": r["version"], "new": new,
            "open": [b for b in ledger if b["status"] == "open"], "board": board, "snapshot": files[-1].name,
            "record": {"graded": 0, "wins": 0, "losses": 0, "profit_units": 0.0, "roi": 0.0, "avg_clv": 0.0, "clv_p_value": 1.0,
                       "checks": {"enough_bets": False, "clv_positive_and_significant": False, "roi_positive": False}, "passed": False,
                       "min_bets": r["validation"]["min_bets"], "note": "graded after the Super Bowl"}}


if __name__ == "__main__":
    out = process()
    print(json.dumps({k: out[k] for k in ("new", "board")}, indent=1))
    pp = ROOT / "output" / "predictions.json"
    if pp.exists():
        P = json.loads(pp.read_text()); P["futures_shop_bets"] = out; pp.write_text(json.dumps(P, indent=1, default=str))
