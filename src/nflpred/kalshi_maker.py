"""Kalshi resting-order (maker) paper track (kalshi_maker_rules.json). Simulated: nothing is posted.

Each run: (1) settle last run's hypothetical orders from Kalshi's public trade history (fill only if a trade printed
strictly below our bid), (2) post new hypothetical YES bids at Pinnacle Shin fair minus a margin where that bid would be
the best bid, (3) grade filled orders after the game (result + CLV vs Pinnacle's Shin close).
Raw Kalshi responses from the first runs are saved to history/kalshi/ so the formats can be checked.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .devig import shin

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
RULES = ROOT / "kalshi_maker_rules.json"


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def _get(url: str, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "nfl-win-probability/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _cents(m: dict, key: str):
    """Kalshi reports prices in cents (yes_bid) or dollars (yes_bid_dollars) depending on API version."""
    if m.get(key) is not None:
        return float(m[key])
    d = m.get(key + "_dollars")
    return float(d) * 100 if d not in (None, "") else None


def _ts(x) -> datetime | None:
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None


def fetch_markets(series: str, api: str, getter=_get) -> list[dict]:
    out, cursor = [], None
    for _ in range(10):
        q = {"series_ticker": series, "status": "open", "limit": 200}
        if cursor:
            q["cursor"] = cursor
        d = getter(f"{api}/markets?{urllib.parse.urlencode(q)}")
        out += d.get("markets") or []
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def fetch_trades(ticker: str, since: datetime, until: datetime, api: str, getter=_get) -> list[dict]:
    q = {"ticker": ticker, "min_ts": int(since.timestamp()), "max_ts": int(until.timestamp()), "limit": 1000}
    return getter(f"{api}/markets/trades?{urllib.parse.urlencode(q)}").get("trades") or []


# ------------------------------------------------------------------ matching a Kalshi market to an odds event
def _tokens(s: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if t not in {"at", "vs", "the"}]


def _fits(label: str, team: str) -> bool:
    """Every label word is a prefix of a word in the team name ('Los Angeles C' fits 'Los Angeles Chargers')."""
    lt, tt = _tokens(label), _tokens(team)
    return bool(lt) and all(any(w.startswith(x) for w in tt) for x in lt)


def match_team(label: str, ev: dict) -> str | None:
    """'home'/'away' if the market's team label fits exactly one of the event's teams."""
    hits = [side for side in ("home", "away") if _fits(label, ev[f"{side}_team"])]
    return hits[0] if len(hits) == 1 else None


def pinnacle_fair(ev: dict) -> tuple[float, datetime | None] | None:
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != "pinnacle":
            continue
        for m in bk.get("markets", []):
            if m["key"] == "h2h":
                px = {o["name"]: o["price"] for o in m["outcomes"]}
                if ev["home_team"] in px and ev["away_team"] in px:
                    return shin(px[ev["home_team"]], px[ev["away_team"]]), _ts(m.get("last_update") or bk.get("last_update"))
    return None


def _latest_events(sport: str) -> tuple[datetime | None, list[dict], list[Path]]:
    import gzip
    d = HIST if sport == "nfl" else HIST / "cfb"
    files = sorted(p for p in d.glob("odds_20*.json*") if p.name[5:6].isdigit())
    if not files:
        return None, [], []
    f = files[-1]
    t = datetime.strptime(f.name[5:20], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)
    with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)) as fh:
        evs = json.load(fh)
    return t, evs, files


def match_market(m: dict, events: list[dict], partner: dict | None = None) -> tuple[dict, str] | None:
    """Kalshi game-winner market -> (odds event, side). Kalshi titles name one team ("Buffalo wins"), so the game's
    other market (same event_ticker) must fit the other team; kickoff within 36 h of the expected expiration minus 3.5 h."""
    label = m.get("yes_sub_title") or m.get("subtitle") or ""
    other_label = (partner or {}).get("yes_sub_title") or m.get("no_sub_title") or ""
    exp = _ts(m.get("expected_expiration_time") or m.get("close_time"))
    found = []
    for ev in events:
        ko = _ts(ev.get("commence_time"))
        if exp and ko and abs((exp - timedelta(hours=3.5) - ko).total_seconds()) > 36 * 3600:
            continue
        side = match_team(label, ev)
        if not side:
            continue
        other = "away" if side == "home" else "home"
        if other_label and not _fits(other_label, ev[f"{other}_team"]):
            continue
        found.append((ev, side))
    return found[0] if len(found) == 1 else None


# ------------------------------------------------------------------ settle / post / grade
def maker_fee(p: float) -> float:
    return 0.0175 * p * (1 - p)


def settle(order: dict, trades: list[dict]) -> dict:
    bid = order["bid_cents"]
    prices = [(_cents(t, "yes_price"), t) for t in trades]
    through = [p for p, _ in prices if p is not None and p < bid]
    touch = [p for p, _ in prices if p is not None and p <= bid]
    return dict(order, status="filled" if through else "expired", fill="through" if through else ("touch" if touch else None),
                n_trades=len(trades))


def closing_fair(bet: dict) -> float | None:
    import gzip
    d = HIST if bet["sport"] == "nfl" else HIST / "cfb"
    ko = _ts(bet["kickoff_utc"])
    for f in sorted((p for p in d.glob("odds_20*.json*") if p.name[5:6].isdigit()), reverse=True):
        t = datetime.strptime(f.name[5:20], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)
        if t >= ko:
            continue
        with (gzip.open(f, "rt") if f.suffix == ".gz" else open(f)) as fh:
            ev = next((e for e in json.load(fh) if e.get("id") == bet["event_id"]), None)
        if ev:
            pf = pinnacle_fair(ev)
            if pf:
                return pf[0] if bet["side"] == "home" else 1 - pf[0]
        if ko - t > timedelta(days=2):
            break
    return None


def result_of(bet: dict) -> bool | None:
    """True if the team we bought YES on won; None while unknown."""
    try:
        if bet["sport"] == "nfl":
            from .data import load_schedules
            from .odds import TEAM_ABBR
            s = load_schedules()
            h, a = TEAM_ABBR.get(bet["home"]), TEAM_ABBR.get(bet["away"])
            g = s[(s.home_team == h) & (s.away_team == a) & s.result.notna()]
            g = g[abs((g.gameday.astype("datetime64[ns]") - datetime.fromisoformat(bet["kickoff_utc"][:10])).dt.days) <= 2]
            if g.empty:
                return None
            m = float(g.iloc[0].result)
        else:
            from cfbpred import data as CD
            from cfbpred.pipeline import team_matcher
            G = CD.games([bet["season"]])
            G = G[G.completed & G.margin.notna()]
            match = team_matcher(sorted(set(G.home) | set(G.away)))
            g = G[(G.home == match(bet["home"])) & (G.away == match(bet["away"]))]
            if g.empty:
                return None
            m = float(g.iloc[0].margin)
        if m == 0:
            return None
        return (m > 0) == (bet["side"] == "home")
    except Exception:
        return None


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    pnl = [b["profit"] for b in gr]
    clv = [b["clv"] for b in gr if b.get("clv") is not None]
    m = sum(clv) / len(clv) if clv else 0.0
    sd = (sum((c - m) ** 2 for c in clv) / (len(clv) - 1)) ** 0.5 if len(clv) > 1 else 0.0
    p = 0.5 * math.erfc((m / (sd / math.sqrt(len(clv)))) / math.sqrt(2)) if len(clv) > 1 and sd > 0 else 1.0
    staked = sum(b["cost"] for b in gr)
    roi = sum(pnl) / staked if staked else 0.0
    checks = {"enough_bets": len(gr) >= r["validation"]["min_fills"], "clv_positive_and_significant": m > 0 and p < r["validation"]["avg_clv_positive_with_p_below"], "roi_positive": roi > 0}
    posted = [b for b in ledger if b.get("rules_version") == r["version"]]
    return {"profit_units": round(sum(pnl), 2), "clv_positive": m > 0,
            "posted": len(posted), "filled": sum(b.get("status") in ("filled", "graded") for b in posted),
            "touch_only": sum(b.get("fill") == "touch" for b in posted), "graded": len(gr),
            "wins": sum(b["result"] == "win" for b in gr), "losses": sum(b["result"] == "loss" for b in gr),
            "profit_per_contract": round(sum(pnl), 3), "roi": round(roi, 4), "avg_clv": round(m, 4), "clv_p_value": round(p, 4),
            "checks": checks, "passed": all(checks.values()), "min_bets": r["validation"]["min_fills"]}


def process(now: datetime | None = None, getter=_get) -> dict:
    r = load_rules()
    now = now or datetime.now(timezone.utc)
    api, q = r["api"], r["qualify"]
    HIST.mkdir(parents=True, exist_ok=True)
    lpath, spath = HIST / r["ledger"], HIST / r["state"]
    ledger = json.loads(lpath.read_text()) if lpath.exists() else []
    state = json.loads(spath.read_text()) if spath.exists() else {"active": [], "probe_saved": {}}
    rep = {"settled": 0, "filled": 0, "posted": 0, "errors": []}
    # 1) settle last run's orders
    for o in state.get("active", []):
        until = min(now, _ts(o["expires_at"]))
        try:
            tr = fetch_trades(o["ticker"], _ts(o["posted_at"]), until, api, getter)
            s = settle(o, tr)
            rep["settled"] += 1
            if s["status"] == "filled" or s.get("fill") == "touch":
                ledger.append(s); rep["filled"] += s["status"] == "filled"
        except Exception as e:
            rep["errors"].append(f"trades {o['ticker']}: {str(e)[:80]}")
        time.sleep(0.2)
    state["active"] = []
    # 2) grade filled orders
    for b in ledger:
        if b.get("status") != "filled" or _ts(b["kickoff_utc"]) > now - timedelta(hours=4):
            continue
        won = result_of(b)
        if won is None:
            continue
        p = b["bid_cents"] / 100
        b["fee"] = round(maker_fee(p), 4)
        b["cost"] = round(p + b["fee"], 4)
        b["result"] = "win" if won else "loss"
        b["profit"] = round((1 - p - b["fee"]) if won else (-p - b["fee"]), 4)
        cf = closing_fair(b)
        b["clv"] = round(cf - b["cost"], 4) if cf is not None else None
        b["status"] = "graded"
    # 3) post new hypothetical orders
    for sport, series_list in r["series"].items():
        snap_t, events, _ = _latest_events(sport)
        if not events or now - snap_t > timedelta(hours=q["max_pinnacle_age_hours"]):
            continue
        for series in series_list:
            try:
                mk = fetch_markets(series, api, getter)
            except Exception as e:
                rep["errors"].append(f"markets {series}: {str(e)[:80]}"); continue
            key = f"{sport}:{series}"
            if mk and not state.get("probe_saved", {}).get(key):
                (HIST / "kalshi").mkdir(exist_ok=True)
                (HIST / "kalshi" / f"probe_{series}.json").write_text(json.dumps(mk[:6], indent=1))
                state.setdefault("probe_saved", {})[key] = now.isoformat(timespec="minutes")
            rep[f"markets_{series}"] = len(mk)
            matched = 0
            by_event = {}
            for m in mk:
                by_event.setdefault(m.get("event_ticker"), []).append(m)
            for m in mk:
                partner = next((x for x in by_event.get(m.get("event_ticker"), []) if x is not m), None)
                hit = match_market(m, events, partner)
                if not hit:
                    continue
                matched += 1
                ev, side = hit
                ko = _ts(ev["commence_time"])
                if not (timedelta(minutes=q["min_minutes_before_kickoff"]) <= ko - now <= timedelta(days=q["max_days_before_kickoff"])):
                    continue
                pf = pinnacle_fair(ev)
                if not pf:
                    continue
                fair = pf[0] if side == "home" else 1 - pf[0]
                if not (q["min_fair"] <= fair <= q["max_fair"]):
                    continue
                bid = math.floor(100 * fair) - q["margin_cents"]
                yb, ya = _cents(m, "yes_bid"), _cents(m, "yes_ask")
                if yb is None or ya is None or not (bid > yb and bid < ya):
                    continue
                expires = min(now + timedelta(minutes=75), ko - timedelta(minutes=q["min_minutes_before_kickoff"]))
                state["active"].append({"id": f"kmk:{m['ticker']}:{now:%Y%m%dT%H%M}", "track": "kalshi_maker", "rules_version": r["version"],
                                        "sport": sport, "ticker": m["ticker"], "event_id": ev["id"], "home": ev["home_team"], "away": ev["away_team"],
                                        "side": side, "team": ev[f"{side}_team"], "kickoff_utc": ko.isoformat(), "season": ko.year if ko.month >= 3 else ko.year - 1,
                                        "fair": round(fair, 4), "bid_cents": bid, "yes_bid": yb, "yes_ask": ya,
                                        "posted_at": now.isoformat(timespec="minutes"), "expires_at": expires.isoformat(timespec="minutes"), "status": "active"})
                rep["posted"] += 1
            rep[f"matched_{series}"] = matched
    lpath.write_text(json.dumps(ledger, indent=1))
    spath.write_text(json.dumps(state, indent=1))
    def view(b):   # fields the dashboard's track tables expect
        c = b.get("cost") or b["bid_cents"] / 100
        am = round(-100 * c / (1 - c)) if c >= 0.5 else round(100 * (1 - c) / c)
        return {**b, "price": am, "book": f"Kalshi {b['bid_cents']}¢ bid", "week": b.get("kickoff_utc", "")[5:10], "profit_units": b.get("profit"),
                "final": b.get("result", ""), "gameday": b["kickoff_utc"][:10], "opponent": b["away"] if b["side"] == "home" else b["home"],
                "edge": round(b["fair"] - c, 4), "units": 1, "placed_at": b["posted_at"]}
    filled = [b for b in ledger if b.get("status") in ("filled", "graded")]
    return {"track": "kalshi_maker", "mode": "shadow", "rules_version": r["version"], "report": rep,
            "active": state["active"], "new": [], "open": [view(b) for b in filled if b["status"] == "filled"],
            "recent_graded": [view(b) for b in filled if b["status"] == "graded"][-25:], "record": record(ledger, r)}


if __name__ == "__main__":
    out = process()
    print(json.dumps({k: out[k] for k in ("report", "record")}, indent=1))
    pp = ROOT / "output" / "predictions.json"       # shown on the NFL page's paper-bets table
    if pp.exists():
        P = json.loads(pp.read_text())
        P["kalshi_maker_bets"] = out
        pp.write_text(json.dumps(P, indent=1, default=str))
