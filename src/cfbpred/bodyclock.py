"""College body-clock paper track (body_clock_rules.json): rule F2 from the factor screen, tracked at Tyler's request
even though it failed the screen. Home team vs a visitor from >= 2 time zones west, kickoff before 1 pm local."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import data as D
from . import shop as SH
from .pipeline import team_matcher
from .tracks import HIST, dec, load, snap_time, snapshots

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "body_clock_rules.json"
TZ_OFF = {"America/New_York": -5, "America/Detroit": -5, "America/Indiana/Indianapolis": -5, "America/Kentucky/Louisville": -5,
          "America/Chicago": -6, "America/Denver": -7, "America/Boise": -7, "America/Phoenix": -7, "America/Los_Angeles": -8,
          "Pacific/Honolulu": -10, "America/Indiana/Knox": -6, "America/Indiana/Tell_City": -6, "America/Menominee": -6,
          "America/North_Dakota/Center": -6, "America/Anchorage": -9}


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def team_tz(season: int) -> dict:
    out = {}
    for y in (season - 1, season):
        for t in D._load(f"teams_fbs_{y}.json.gz"):
            tz = TZ_OFF.get((t.get("location") or {}).get("timezone"))
            if tz is not None:
                out[t["school"]] = tz
    return out


def qualifies(game, tz: dict, r: dict) -> dict | None:
    """game: CFBD games row. Returns the reason dict when the F2 rule applies."""
    if game.neutral or game.home_div != "fbs" or game.away_div != "fbs":
        return None
    h, a = tz.get(game.home), tz.get(game.away)
    if h is None or a is None:
        return None
    local_hour = (game.start.hour + h) % 24
    if h - a >= r["qualify"]["min_tz_gap_hours"] and local_hour < r["qualify"]["max_local_kickoff_hour"]:
        return {"tz_gap_hours": h - a, "local_kickoff_hour": int(local_hour)}
    return None


def best_home_spread(ev: dict, books: list[str]) -> dict | None:
    best = None
    for bk in ev.get("bookmakers", []):
        if bk.get("key") not in books:
            continue
        for m in bk.get("markets", []):
            if m["key"] != "spreads":
                continue
            for o in m["outcomes"]:
                if o["name"] == ev["home_team"] and o.get("point") is not None:
                    if best is None or (o["point"], o["price"]) > (best["point"], best["price"]):
                        best = {"point": o["point"], "price": o["price"], "book": bk.get("title", bk["key"]), "book_key": bk["key"]}
    return best


def grade(ledger: list[dict], files: list[Path]) -> list[dict]:
    open_ = [b for b in ledger if b.get("status") == "open"]
    if not open_:
        return ledger
    g = D.games(sorted({b["season"] for b in open_}))
    g = g[g.completed & g.margin.notna()]
    idx = {int(r.game_id): r for r in g.itertuples()}
    cache: dict = {}
    out = []
    for b in ledger:
        gg = idx.get(b.get("game_id")) if b.get("status") == "open" else None
        if gg is None:
            out.append(b); continue
        x = float(gg.margin) + b["point"]
        nb = dict(b, status="graded", final=f"{int(gg.away_pts)}-{int(gg.home_pts)}", result="push" if x == 0 else ("win" if x > 0 else "loss"))
        nb["profit_units"] = 0.0 if x == 0 else round(b["units"] * (dec(b["price"]) - 1) if x > 0 else -b["units"], 3)
        pr = SH.closing_probs({**b, "market": "spread", "side": "home"}, files, cache)
        if pr is not None:
            nb["clv"] = round(SH.DI.ev(*pr, b["price"]), 4)
        out.append(nb)
    return out


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    w, l = sum(b["result"] == "win" for b in gr), sum(b["result"] == "loss" for b in gr)
    n = w + l
    z = (w - n * 0.5238) / math.sqrt(n * 0.5238 * 0.4762) if n else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2)) if n else 1.0
    clv = [b["clv"] for b in gr if "clv" in b]
    profit = sum(b["profit_units"] for b in gr)
    checks = {"enough_bets": n >= r["validation"]["min_bets"], "cover_rate_significant": p < r["validation"]["cover_rate_p_below"]}
    return {"graded": len(gr), "wins": w, "losses": l, "pushes": len(gr) - n, "cover_rate": round(w / n, 3) if n else None,
            "p_value": round(p, 4), "profit_units": round(profit, 2), "roi": round(profit / n, 4) if n else 0.0,
            "avg_clv": round(sum(clv) / len(clv), 4) if clv else None, "checks": checks, "passed": all(checks.values()),
            "min_bets": r["validation"]["min_bets"]}


def process(now: datetime | None = None) -> dict:
    r = load_rules()
    now = now or datetime.now(timezone.utc)
    path = HIST / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    files = snapshots()
    ledger = grade(ledger, files)
    new, upcoming = [], []
    season = now.year if now.month >= 7 else now.year - 1
    G = D.games([season])
    G = G[~G.completed]
    G = G[(G.start > now) & (G.start < now + timedelta(days=8))]
    tz = team_tz(season)
    qual = {int(x.game_id): (x, q) for x in G.itertuples() if (q := qualifies(x, tz, r))}
    for gid, (x, q) in qual.items():
        upcoming.append({"game_id": gid, "home": x.home, "away": x.away, "kickoff_utc": x.start.isoformat(), **q})
    if qual and files and now - snap_time(files[-1]) <= timedelta(hours=2):
        have = {b["game_id"] for b in ledger}
        lo, hi = r["qualify"]["bet_window_hours_before_kickoff"]
        match = team_matcher(sorted({x.home for x, _ in qual.values()} | {x.away for x, _ in qual.values()}))
        by_pair = {(x.home, x.away): (gid, x, q) for gid, (x, q) in qual.items()}
        for ev in load(files[-1]):
            hit = by_pair.get((match(ev["home_team"]), match(ev["away_team"])))
            if not hit or hit[0] in have:
                continue
            gid, x, q = hit
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            hrs = (ko - now).total_seconds() / 3600
            if not (lo <= hrs <= hi):
                continue
            best = best_home_spread(ev, r["books"])
            if not best or not (r["qualify"]["price_range"][0] <= best["price"] <= r["qualify"]["price_range"][1]):
                continue
            new.append({"id": f"cfbclock:{gid}", "track": "cfb_body_clock", "rules_version": r["version"], "game_id": gid,
                        "event_id": ev["id"], "placed_at": now.isoformat(timespec="minutes"), "season": season,
                        "kickoff_utc": ko.isoformat(), "home": ev["home_team"], "away": ev["away_team"], "team": ev["home_team"],
                        "market": "spread", "side": "home", "units": r["sizing"]["units"], "status": "open", **best, **q})
            have.add(gid)
    ledger += new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1))
    return {"track": "cfb_body_clock", "mode": "shadow", "rules_version": r["version"], "new": new, "qualifying_games": upcoming,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-25:], "record": record(ledger, r)}
