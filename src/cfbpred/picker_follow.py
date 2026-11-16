"""Follow-a-picker paper track (picker_follow_rules.json, v1 frozen 2026-11-15): bet every counted pick of the listed
published pickers (v1: Brandon Marcello, college) at your books' best price in the first odds snapshot after we first
saw the pick. The picks scorer (nflpred.picks_score.run) writes the queue of counted picks; the college pipeline turns
them into bets, grades them and builds the page view."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "picker_follow_rules.json"
HIST = ROOT / "history"
MKT = {"spread": "spreads", "total": "totals", "moneyline": "h2h"}
SHORT = {"spread": "spread", "total": "total", "moneyline": "ml"}


def _t(s):
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def load_rules(path: Path = RULES) -> dict:
    return json.loads(path.read_text())


def _read(p: Path, default):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def pick_id(x: dict) -> str:
    return f"follow:{x['picker'].lower().replace(' ', '_')}:{x['game_id']}:{x['market']}:{str(x['side']).lower().replace(' ', '_')}"


def queue_picks(rows: list[dict], hist: Path = HIST, rules: dict | None = None) -> int:
    """Called by picks_score.run with its counted picks (graded + pending): keep the followed pickers' picks."""
    r = rules or load_rules()
    start = _t(r["start"])
    p = hist / r["queue"]
    q = _read(p, {})
    n0 = len(q)
    for x in rows:
        if x.get("picker") in r["pickers"] and x.get("sport") in r["sports"] and _t(x.get("seen_at")) and _t(x["seen_at"]) >= start:
            k = pick_id(x)
            if k not in q:
                q[k] = {kk: x.get(kk) for kk in ("picker", "sport", "game_id", "market", "side", "line", "seen_at", "kick",
                                                 "home", "away", "week", "outlet", "link", "reason")}
    if len(q) != n0 or not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(q, indent=1))
    return len(q) - n0


def _event(snap: list[dict], x: dict):
    from nflpred.picks_score import cfb_matcher
    m = cfb_matcher([x["home"], x["away"]])
    kick = _t(x["kick"])
    for ev in snap:
        if {m(ev["home_team"]), m(ev["away_team"])} == {x["home"], x["away"]} and \
                abs((_t(ev["commence_time"]) - kick).total_seconds()) < 36 * 3600:
            return ev
    return None


def best_offer(ev: dict, x: dict, books: list[str]) -> dict | None:
    """Best quote at `books` for the pick: spread/total best number then price; moneyline best price."""
    from nflpred.picks_score import cfb_matcher
    mk = MKT[x["market"]]
    if mk == "totals":
        name, side = str(x["side"]).capitalize(), str(x["side"]).lower()
    else:
        m = cfb_matcher([x["side"]])
        name = next((t for t in (ev["home_team"], ev["away_team"]) if m(t) == x["side"]), None)
        if name is None:
            return None
        side = "home" if name == ev["home_team"] else "away"
    best = None
    for bk in ev.get("bookmakers", []):
        if bk.get("key") not in books:
            continue
        for mm in bk.get("markets", []):
            if mm["key"] != mk:
                continue
            for o in mm["outcomes"]:
                if o["name"] != name:
                    continue
                pt = o.get("point")
                if mk != "h2h" and pt is None:
                    continue
                # better number: spread higher (+4 > +3, -3 > -4); over lower; under higher
                key = (o["price"],) if mk == "h2h" else ((-pt if side == "over" else pt), o["price"])
                if best is None or key > best["_key"]:
                    best = {"_key": key, "price": o["price"], "point": pt, "book": bk.get("title", bk["key"]),
                            "book_key": bk["key"], "side": side, "team": name if mk != "totals" else None}
    if best:
        best.pop("_key")
    return best


def make_bets(q: dict, ledger: list[dict], files: list[Path], r: dict, now: datetime) -> list[dict]:
    from cfbpred.shop import books_of
    from cfbpred.tracks import load, snap_time
    books = books_of(r)
    have = {b["id"] for b in ledger}
    new = []
    for k, x in q.items():
        if k in have:
            continue
        seen, kick = _t(x["seen_at"]), _t(x["kick"])
        last_ok = min(seen + timedelta(hours=12), kick - timedelta(minutes=10))
        cand = [f for f in files if seen <= snap_time(f) <= last_ok]
        if not cand:
            if now > last_ok:      # window passed without a snapshot: record the skip once
                ledger.append({"id": k, "track": r["track"], "rules_version": r["version"], "status": "skipped",
                               "reason": "no odds snapshot in the entry window", **_meta(x)})
            continue
        f = cand[0]
        ev = _event(load(f), x)
        off = best_offer(ev, x, books) if ev else None
        if not off:
            ledger.append({"id": k, "track": r["track"], "rules_version": r["version"], "status": "skipped",
                           "reason": "game not found in the odds feed" if not ev else "none of your books quoted it", **_meta(x)})
            continue
        b = {"id": k, "track": r["track"], "rules_version": r["version"], "status": "open", **_meta(x),
             "event_id": ev["id"], "home": ev["home_team"], "away": ev["away_team"], "kickoff_utc": ev["commence_time"].replace("Z", "+00:00"),
             "market": SHORT[x["market"]], "side": off["side"], "team": off["team"], "point": off["point"], "price": off["price"],
             "book": off["book"], "book_key": off["book_key"], "units": r["units"], "placed_at": snap_time(f).isoformat(timespec="minutes"),
             "rule": x["picker"], "season": kick.year if kick.month >= 7 else kick.year - 1}
        ledger.append(b)
        new.append(b)
    return new


def _meta(x: dict) -> dict:
    return {"picker": x["picker"], "game_id": x["game_id"], "pick_market": x["market"], "pick_side": x["side"],
            "pick_line": x.get("line"), "seen_at": x["seen_at"], "cfbd_home": x["home"], "cfbd_away": x["away"],
            "outlet": x.get("outlet"), "link": x.get("link"), "pick_reason": x.get("reason")}


def grade(ledger: list[dict], files: list[Path]) -> list[dict]:
    from cfbpred import data as D, shop as SH
    from cfbpred.tracks import dec
    op = [b for b in ledger if b.get("status") == "open"]
    if not op:
        return ledger
    g = D.games(sorted({b["season"] for b in op}))
    g = g[g.completed & g.margin.notna()]
    idx = {str(int(r.game_id)): r for r in g.itertuples()}
    cache: dict = {}
    for i, b in enumerate(ledger):
        gg = idx.get(str(b.get("game_id"))) if b.get("status") == "open" else None
        if gg is None:
            continue
        # CFBD orientation -> the odds feed's (neutral sites can swap home/away)
        flip = b["cfbd_home"] != b["home"] and _same(b["cfbd_home"], b["away"])
        hp, ap = (float(gg.away_pts), float(gg.home_pts)) if flip else (float(gg.home_pts), float(gg.away_pts))
        x = SH.settle(b, hp - ap, hp + ap)
        res = "push" if x == 0 else "win" if x > 0 else "loss"
        nb = dict(b, status="graded", result=res, final=f"{int(gg.away_pts)}-{int(gg.home_pts)}",
                  profit_units=0.0 if res == "push" else round(b["units"] * (dec(b["price"]) - 1) if res == "win" else -b["units"], 3))
        pr = SH.closing_probs(b, files, cache)
        if pr is not None:
            nb["clv"] = round(float(SH.DI.ev(*pr, b["price"])), 4)
        ledger[i] = nb
    return ledger


def _same(cfbd_name: str, feed_name: str) -> bool:
    from nflpred.picks_score import cfb_matcher
    return cfb_matcher([cfbd_name])(feed_name) == cfbd_name


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    w, l = sum(b["result"] == "win" for b in gr), sum(b["result"] == "loss" for b in gr)
    prof = [b["profit_units"] for b in gr]
    n = len(prof)
    mean = sum(prof) / n if n else 0.0
    sd = math.sqrt(sum((p - mean) ** 2 for p in prof) / (n - 1)) if n > 1 else 0.0
    p_roi = 0.5 * math.erfc((mean / (sd / math.sqrt(n))) / math.sqrt(2)) if n > 1 and sd > 0 else 1.0
    clv = [b["clv"] for b in gr if b.get("clv") is not None]
    cm = sum(clv) / len(clv) if clv else None
    csd = math.sqrt(sum((c - cm) ** 2 for c in clv) / (len(clv) - 1)) if len(clv) > 1 else 0.0
    p_clv = 0.5 * math.erfc((cm / (csd / math.sqrt(len(clv)))) / math.sqrt(2)) if len(clv) > 1 and csd > 0 else 1.0
    v = r["validation"]
    checks = {"enough_bets": bool(n >= v["min_bets"]), "roi_significant": bool(p_roi < v["roi_p_below"]),
              "clv_positive_and_significant": bool(cm is not None and cm > 0 and p_clv < 0.05)}
    return {"graded": n, "wins": w, "losses": l, "pushes": n - w - l, "profit_units": round(sum(prof), 2),
            "roi": round(float(mean), 4), "roi_p_value": round(float(p_roi), 4), "avg_clv": round(float(cm), 4) if cm is not None else None,
            "clv_p_value": round(float(p_clv), 4), "checks": checks, "passed": checks["enough_bets"] and checks["roi_significant"],
            "min_bets": v["min_bets"], "skipped": sum(b.get("status") == "skipped" for b in ledger)}


def process(now: datetime | None = None, hist: Path = HIST, files: list[Path] | None = None) -> dict:
    from cfbpred.tracks import snapshots
    r = load_rules()
    now = now or datetime.now(timezone.utc)
    path = hist / r["ledger"]
    ledger = _read(path, [])
    files = files if files is not None else snapshots()
    new = make_bets(_read(hist / r["queue"], {}), ledger, files, r, now)
    ledger = grade(ledger, files)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1))
    return {"track": r["track"], "mode": "shadow", "rules_version": r["version"], "pickers": r["pickers"], "new": new,
            "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-25:], "record": record(ledger, r)}
