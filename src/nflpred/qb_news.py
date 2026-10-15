"""QB-news speed paper track (qb_news_rules.json, pre-registered 2026-10-12).

When the AI news reader logs a strict starting-QB call (certainty >= 0.9, full name matched to the Sleeper roster,
accurate source, about the next game), bet the side the news favors at your books if the consensus spread has not
yet moved half a point that way since the news appeared. Paper only; judged on closing line value with prices.

Runs in the hourly watch and in the 15-minute QB-news check (qb_news_fast.yml). Fresh odds are fetched only when a
signal qualifies, and that fetch is saved as an ordinary odds snapshot.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import exchanges as X
from . import roster as RS
from .odds import TEAM_ABBR

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "qb_news_rules.json"
SKIP_BOOKS = {"pinnacle", "kalshi", "polymarket", "novig", "prophetx", "betopenly", "lowvig", "betonlineag"}


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def _t(s: str | None) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def recent_signals(history_dir: Path, now: datetime, hours: float) -> list[dict]:
    p = history_dir / "news_llm.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines()[-4000:]:
        try:
            s = json.loads(line)
        except Exception:
            continue
        t = _t(s.get("seen_at"))
        if t and now - timedelta(hours=hours) <= t <= now and s.get("is_starting_qb_news"):
            out.append(s)
    return out


def news_time(s: dict) -> datetime:
    """When the news appeared: the earlier of published and first seen."""
    ts = [t for t in (_t(s.get("published")), _t(s.get("seen_at"))) if t]
    return min(ts)


def same_player(a: str | None, b: str | None) -> bool:
    return bool(a and b) and RS.key(a) == RS.key(b)


def classify(s: dict, g: dict, r: dict) -> tuple[str | None, str]:
    """-> (bet side 'home'/'away' or None, reason)."""
    q = r["qualify"]
    team = s.get("team_verified") or s.get("team")
    if team not in (g["home_team"], g["away_team"]):
        return None, "not this game"
    side = "home" if team == g["home_team"] else "away"
    other = "away" if side == "home" else "home"
    v = s.get("verify") or {}
    if v.get("roster") != "match" or v.get("roster_source") != "sleeper" or len((s.get("player") or "").split()) < 2:
        return None, "player not matched by full name on the Sleeper roster"
    if (s.get("certainty") or 0) < q["min_certainty"]:
        return None, f"certainty {s.get('certainty')} < {q['min_certainty']}"
    if s.get("source_muted") or (s.get("source_weight") is not None and s["source_weight"] < q["min_source_weight"]):
        return None, "source accuracy below the bar"
    if s.get("game_week_relevant") is False:
        return None, "not about the next game"
    starter = g.get(f"{side}_qb")
    sig = s.get("signal")
    if same_player(s.get("player"), starter) and sig in q["starter_unavailable"]:
        return other, f"starter {s['player']} {sig}"
    if not same_player(s.get("player"), starter) and sig in q["backup_named_starter"] and (s.get("position") or "QB").upper() == "QB":
        return other, f"backup {s['player']} named starter over {starter}"
    if same_player(s.get("player"), starter) and sig in q["starter_cleared"]:
        st = ((g.get("qb_status") or {}).get(side) or {}).get("status")
        if (st or "").lower() in q["starter_cleared_needs_listed_status"]:
            return side, f"starter {s['player']} cleared ({sig}) after being listed {st}"
        return None, "starter cleared but was not listed on the injury report"
    return None, f"signal {sig} does not qualify"


def consensus_home_point(ev: dict) -> float | None:
    pts = []
    for bk in ev.get("bookmakers", []):
        if bk.get("key") in SKIP_BOOKS:
            continue
        for m in bk.get("markets", []):
            if m.get("key") == "spreads":
                for o in m.get("outcomes", []):
                    if o.get("name") == ev.get("home_team") and o.get("point") is not None:
                        pts.append(o["point"])
    return statistics.median(pts) if pts else None


def _ev_for(events: list[dict], g: dict) -> dict | None:
    for ev in events:
        if (TEAM_ABBR.get(ev.get("home_team")), TEAM_ABBR.get(ev.get("away_team"))) == (g["home_team"], g["away_team"]):
            return ev
    return None


def point_before(files: list[Path], cache: dict, g: dict, t: datetime) -> float | None:
    """Consensus home point in our last odds snapshot at or before t."""
    for f in reversed(files):
        if X._snap_time(f) > t:
            continue
        if f not in cache:
            cache[f] = X.load_events(f)
        ev = _ev_for(cache[f], g)
        if ev:
            return consensus_home_point(ev)
    return None


def best_spread(ev: dict, side: str, allowed: set) -> dict | None:
    name = ev["home_team"] if side == "home" else ev["away_team"]
    best = None
    for bk in ev.get("bookmakers", []):
        if bk.get("key") not in allowed:
            continue
        for m in bk.get("markets", []):
            if m.get("key") != "spreads":
                continue
            for o in m.get("outcomes", []):
                if o.get("name") == name and o.get("point") is not None:
                    d = 1 + o["price"] / 100 if o["price"] > 0 else 1 + 100 / abs(o["price"])
                    if best is None or (o["point"], d) > (best["point"], best["_d"]):
                        best = {"point": o["point"], "price": o["price"], "book": bk.get("title", bk["key"]), "_d": d}
    if best:
        best.pop("_d")
    return best


def prev_game_end(team: str, before: datetime, schedule: pd.DataFrame | None, hours: float) -> datetime | None:
    if schedule is None or schedule.empty:
        return None
    s = schedule[(schedule.home_team == team) | (schedule.away_team == team)]
    ko = pd.to_datetime(s.gameday.astype(str) + " " + s.gametime.fillna("13:00").astype(str)).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    ko = ko[ko < pd.Timestamp(before) - pd.Timedelta(hours=1)]
    return (ko.max() + pd.Timedelta(hours=hours)).to_pydatetime() if len(ko) else None


def record(ledger: list[dict], r: dict) -> dict:
    gr = [b for b in ledger if b.get("status") == "graded" and b.get("rules_version") == r["version"]]
    wins = sum(b["result"] == "win" for b in gr); losses = sum(b["result"] == "loss" for b in gr)
    staked = sum(b["units"] for b in gr if b["result"] != "push"); profit = sum(b["profit_units"] for b in gr)
    clv = [b["clv"] for b in gr if b.get("clv") is not None]
    m = sum(clv) / len(clv) if clv else 0.0
    sd = statistics.stdev(clv) if len(clv) > 1 else 0.0
    p = 0.5 * math.erfc((m / (sd / math.sqrt(len(clv)))) / math.sqrt(2)) if len(clv) > 1 and sd > 0 else 1.0
    roi = profit / staked if staked else 0.0
    v = r["validation"]
    checks = {"enough_bets": len(gr) >= v["min_bets"], "clv_positive_and_significant": m > 0 and p < v["avg_clv_positive_with_p_below"],
              "roi_positive": roi > v["realized_roi_above"]}
    return {"graded": len(gr), "wins": wins, "losses": losses, "pushes": len(gr) - wins - losses, "units_staked": round(staked, 2),
            "profit_units": round(profit, 2), "roi": round(roi, 4), "avg_clv": round(m, 4), "clv_p_value": round(p, 4),
            "checks": checks, "passed": all(checks.values()), "min_bets": v["min_bets"]}


def process(pred: dict, games: pd.DataFrame, history_dir: Path, now: datetime | None = None, fetch=None,
            schedule: pd.DataFrame | None = None, r: dict | None = None) -> dict:
    from . import spread_bets as SB
    r = r or load_rules()
    now = now or datetime.now(timezone.utc)
    q = r["qualify"]
    path = history_dir / r["ledger"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    if len(games):
        rs = SB.load_rules()
        closes = SB.closing_margins(history_dir, rs)
        ledger = [SB.grade(b, games, rs, closes) if b.get("status") == "open" else b for b in ledger]
    if schedule is None:
        try:
            from . import data
            schedule = data.load_schedules()
        except Exception:
            schedule = None
    have = {b["game_id"] for b in ledger}
    allowed = set(json.loads((ROOT / "my_books.json").read_text()).get("allowed_books", []))
    checks, cands = [], []
    for s in recent_signals(history_dir, now, q["signal_age_hours_max"]):
        for g in pred.get("upcoming", []):
            side, why = classify(s, g, r)
            if why == "not this game":
                continue
            ko = _t(g.get("kickoff_utc"))
            row = {"player": s.get("player"), "team": s.get("team_verified") or s.get("team"), "signal": s.get("signal"),
                   "game_id": g["game_id"], "reason": why, "seen_at": s.get("seen_at")}
            if side and (ko is None or not (q["kickoff_window_hours"][0] <= (ko - now).total_seconds() / 3600 <= q["kickoff_window_hours"][1])):
                side, row["reason"] = None, "kickoff not 30 min to 6 days away"
            pe = prev_game_end(row["team"], ko or now, schedule, q["published_after_previous_game_end_hours"]) if side else None
            if side and pe and news_time(s) < pe:
                side, row["reason"] = None, "published before the team's previous game ended"
            if side and g["game_id"] in have:
                side, row["reason"] = None, "already bet this game"
            checks.append(row)
            if side:
                cands.append((s, g, side, row))
    new = []
    if cands:
        events = None
        try:
            if fetch is not None:
                events = fetch()
                if events:     # saved like any other snapshot (also the 'before' point for later news)
                    with gzip.open(history_dir / f"odds_{now:%Y-%m-%dT%H%M}.json.gz", "wt") as f:
                        f.write(json.dumps(events))
        except Exception as e:
            print("qb_news: odds fetch failed:", e)
        files = X.snapshot_files(history_dir); cache: dict = {}
        if events is None and files:
            events = X.load_events(files[-1])
        for s, g, side, row in cands:
            if g["game_id"] in have:
                continue
            ev = _ev_for(events or [], g)
            if not ev:
                row["reason"] = "game not in the odds feed"; continue
            p_now, p0 = consensus_home_point(ev), point_before(files, cache, g, news_time(s))
            if p_now is None or p0 is None:
                row["reason"] = "no spread before or after the news"; continue
            moved = (p0 - p_now) if side == "home" else (p_now - p0)   # + = market moved toward the bet side
            row["moved_pts"] = round(moved, 2)
            if moved >= q["max_line_move_pts"]:
                row["reason"] = f"line already moved {moved:.1f} pts toward {g[f'{side}_team']}"; continue
            best = best_spread(ev, side, allowed)
            if not best:
                row["reason"] = "no spread at your books"; continue
            team = g[f"{side}_team"]
            bet = {"id": f"{g['game_id']}:qb_news:{side}", "track": "qb_news", "rules_version": r["version"],
                   "placed_at": now.isoformat(timespec="minutes"), "game_id": g["game_id"], "season": g.get("season"),
                   "week": g.get("week"), "gameday": g.get("gameday"), "kickoff_utc": g.get("kickoff_utc"),
                   "home_team": g["home_team"], "away_team": g["away_team"], "side": side, "team": team,
                   "opponent": g["away_team" if side == "home" else "home_team"], "units": r["sizing"]["units"], "status": "open",
                   **best, "market": "spread", "edge": None, "news": {k: s.get(k) for k in ("player", "signal", "certainty", "outlet", "source", "link", "title", "published", "seen_at")},
                   "why": row["reason"], "consensus_before": p0, "consensus_at_bet": p_now, "moved_pts": round(moved, 2)}
            new.append(bet); have.add(g["game_id"]); row["reason"] = "BET " + row["reason"]
    ledger += new
    path.write_text(json.dumps(ledger, indent=2))
    rec = record(ledger, r)
    return {"track": "qb_news", "mode": "live" if rec["passed"] else "shadow", "rules_version": r["version"], "new": new,
            "checks": checks[-40:], "open": [b for b in ledger if b.get("status") == "open"],
            "recent_graded": [b for b in ledger if b.get("status") == "graded"][-20:], "record": rec}


def main() -> int:
    """15-minute QB-news check: update predictions.json's qb_news_bets (no other tracks run)."""
    from . import odds as O
    from . import notify
    pp = ROOT / "output" / "predictions.json"
    pred = json.loads(pp.read_text())
    res = process(pred, pd.DataFrame(columns=["game_id", "completed"]), ROOT / "history", fetch=O.fetch)
    prev = pred.get("qb_news_bets") or {}
    res["recent_graded"] = prev.get("recent_graded", res["recent_graded"])
    pred["qb_news_bets"] = res
    try:
        from . import heads_up
        heads_up.annotate(pred, ROOT / "history")
    except Exception as e:
        print("heads-up failed:", e)
    pp.write_text(json.dumps(pred, indent=1, default=str))
    if res["new"]:
        notify.new_bets({"upcoming": pred.get("upcoming", []), "qb_news_bets": res}, set())
    print(json.dumps({"new": [b["id"] for b in res["new"]], "checks": res["checks"][-10:]}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
