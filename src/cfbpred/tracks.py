"""College football moneyline paper track (cfb_ml_rules.json): rules P1 (soft books vs Pinnacle no-vig) and
P2 (moneyline vs Pinnacle spread), frozen from the price screen. Runs on every NCAAF odds snapshot."""
from __future__ import annotations

import gzip
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from . import data as D
from .pipeline import team_matcher

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
RULES = ROOT / "cfb_ml_rules.json"
ND = NormalDist()


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def imp(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def snapshots() -> list[Path]:
    return sorted((HIST / "cfb").glob("odds_*.json.gz"))


def snap_time(p: Path) -> datetime:
    return datetime.strptime(p.name[5:20], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)


def load(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return json.load(f)


def pinnacle(ev: dict) -> dict | None:
    for bk in ev.get("bookmakers", []):
        if bk.get("key") != "pinnacle":
            continue
        mk = {m["key"]: {o["name"]: o for o in m["outcomes"]} for m in bk.get("markets", [])}
        h2, sp = mk.get("h2h", {}), mk.get("spreads", {})
        out = {}
        if ev["home_team"] in h2 and ev["away_team"] in h2:
            ph, pa = imp(h2[ev["home_team"]]["price"]), imp(h2[ev["away_team"]]["price"])
            out["fair_h"] = ph / (ph + pa)
        if ev["home_team"] in sp and ev["away_team"] in sp and sp[ev["home_team"]].get("point") is not None:
            ch, ca = imp(sp[ev["home_team"]]["price"]), imp(sp[ev["away_team"]]["price"])
            out["pt"], out["cover_h"] = sp[ev["home_team"]]["point"], ch / (ch + ca)
        return out or None
    return None


def candidates(ev: dict, r: dict) -> list[dict]:
    pin = pinnacle(ev)
    if not pin or "fair_h" not in pin:
        return []
    sd = r["qualify"]["spread_sd"]
    p_sp = None
    if "pt" in pin:
        mu = -pin["pt"] + sd * ND.inv_cdf(min(max(pin["cover_h"], 1e-6), 1 - 1e-6))
        p_sp = ND.cdf(mu / sd)
    lo, hi = r["qualify"]["price_range"]
    out = []
    for side in ("home", "away"):
        name = ev[f"{side}_team"]
        best = None
        for bk in ev.get("bookmakers", []):
            if bk.get("key") not in r["books"]:
                continue
            for m in bk.get("markets", []):
                if m["key"] == "h2h":
                    for o in m["outcomes"]:
                        if o["name"] == name and (best is None or dec(o["price"]) > dec(best["price"])):
                            best = {"price": o["price"], "book": bk.get("title", bk["key"])}
        if not best or not (lo <= best["price"] <= hi):
            continue
        pf = pin["fair_h"] if side == "home" else 1 - pin["fair_h"]
        ev_f = pf * dec(best["price"]) - 1
        rules = []
        if ev_f >= r["qualify"]["min_ev"]:
            rules.append(("P1", ev_f))
        if p_sp is not None:
            ps = p_sp if side == "home" else 1 - p_sp
            ev_s = ps * dec(best["price"]) - 1
            if ev_s >= r["qualify"]["min_ev"] and ev_f >= 0:
                rules.append(("P2", ev_s))
        for rule, edge in rules:
            out.append({"rule": rule, "side": side, "team": name, "price": best["price"], "book": best["book"],
                        "edge": round(edge, 4), "p_fair": round(pf, 4)})
    return out


def closing_fair(bet: dict, files: list[Path], cache: dict) -> float | None:
    ko = datetime.fromisoformat(bet["kickoff_utc"])
    for f in reversed(files):
        if snap_time(f) >= ko:
            continue
        if f not in cache:
            cache[f] = {e["id"]: e for e in load(f)}
        ev = cache[f].get(bet["event_id"])
        if ev:
            pin = pinnacle(ev)
            if pin and "fair_h" in pin:
                return pin["fair_h"] if bet["side"] == "home" else 1 - pin["fair_h"]
    return None


def grade(ledger: list[dict], files: list[Path]) -> list[dict]:
    open_ = [b for b in ledger if b.get("status") == "open"]
    if not open_:
        return ledger
    seasons = sorted({b["season"] for b in open_})
    g = D.games(seasons)
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
        m = float(gg.margin)
        won = m > 0 if b["side"] == "home" else m < 0
        nb = dict(b, status="graded", final=f"{int(gg.away_pts)}-{int(gg.home_pts)}",
                  result="push" if m == 0 else ("win" if won else "loss"))
        nb["profit_units"] = 0.0 if m == 0 else round(b["units"] * (dec(b["price"]) - 1) if won else -b["units"], 3)
        cf = closing_fair(b, files, cache)
        if cf is not None:
            nb["closing_prob"] = round(cf, 4); nb["clv"] = round(dec(b["price"]) * cf - 1, 4)
        out.append(nb)
    return out


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    staked = sum(b["units"] for b in gr if b["result"] != "push"); profit = sum(b["profit_units"] for b in gr)
    clv = [b["clv"] for b in gr if "clv" in b]
    m = sum(clv) / len(clv) if clv else 0.0
    sd = (sum((c - m) ** 2 for c in clv) / (len(clv) - 1)) ** 0.5 if len(clv) > 1 else 0.0
    z = m / (sd / math.sqrt(len(clv))) if len(clv) > 1 and sd > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2)) if len(clv) > 1 else 1.0
    roi = profit / staked if staked else 0.0
    checks = {"enough_bets": len(gr) >= r["validation"]["min_bets"], "clv_positive_and_significant": m > 0 and p < 0.05, "roi_positive": roi > 0}
    by_rule = {}
    for rule in ("P1", "P2"):
        x = [b for b in gr if b["rule"] == rule]
        c = [b["clv"] for b in x if "clv" in b]
        by_rule[rule] = {"graded": len(x), "profit_units": round(sum(b["profit_units"] for b in x), 2),
                         "avg_clv": round(sum(c) / len(c), 4) if c else None}
    return {"graded": len(gr), "wins": sum(b["result"] == "win" for b in gr), "losses": sum(b["result"] == "loss" for b in gr),
            "units_staked": round(staked, 2), "profit_units": round(profit, 2), "roi": round(roi, 4), "avg_clv": round(m, 4),
            "clv_p_value": round(p, 4), "checks": checks, "passed": all(checks.values()), "min_bets": r["validation"]["min_bets"],
            "by_rule": by_rule}


def process(now: datetime | None = None) -> dict:
    r = load_rules()
    now = now or datetime.now(timezone.utc)
    path = HIST / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    files = snapshots()
    ledger = grade(ledger, files)
    new = []
    if files and now - snap_time(files[-1]) <= timedelta(hours=2):
        have = {(b["event_id"], b["rule"]) for b in ledger}
        season = now.year if now.month >= 7 else now.year - 1
        for ev in load(files[-1]):
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if ko - now < timedelta(minutes=r["qualify"]["min_minutes_before_kickoff"]):
                continue
            for c in candidates(ev, r):
                if (ev["id"], c["rule"]) in have:
                    continue
                have.add((ev["id"], c["rule"]))
                new.append({"id": f"cfb:{ev['id']}:{c['rule']}:{c['side']}", "track": "cfb_ml", "rules_version": r["version"],
                            "placed_at": now.isoformat(timespec="minutes"), "event_id": ev["id"], "season": season,
                            "kickoff_utc": ko.isoformat(), "home": ev["home_team"], "away": ev["away_team"],
                            "gameday": ko.date().isoformat(), "units": r["sizing"]["units"], "status": "open", **c})
    ledger += new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1))
    return {"track": "cfb_ml", "mode": "shadow", "rules_version": r["version"], "new": new,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-25:], "record": record(ledger, r)}
