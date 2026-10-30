"""Props + news paper track (props_news_rules.json v1, frozen 2026-10-30): when the AI news reader learns a regular
WR/TE/RB is out, bet the OVER on his next men up before the books adjust; graded on CLV and points moved by the close.

Runs with the QB-news check (every ~15 min and in the hourly watch): qb_news.main -> props_news.process.
Fetches props only when a trigger qualifies (and once near kickoff for games with open bets).
"""
from __future__ import annotations

import gzip
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "props_news_rules.json"
STAT_COL = {"player_receptions": "receptions", "player_reception_yds": "receiving_yards", "player_rush_yds": "rushing_yards"}
LABEL = {"player_receptions": "rec", "player_reception_yds": "rec yds", "player_rush_yds": "rush yds"}


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def _t(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _imp(a: float) -> float:
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def _dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def snap_shares(snaps: pd.DataFrame, team: str, before: datetime | None = None) -> dict:
    """player -> (position, average offense_pct over the team's last 3 games)."""
    s = snaps[snaps["team"] == team]
    if s.empty:
        return {}
    games = sorted(s["game_id"].unique(), key=lambda g: s[s.game_id == g]["week"].iloc[0])[-3:]
    last = s[s.game_id.isin(games)]
    out = {}
    for p, d in last.groupby("player"):
        out[p] = (str(d["position"].iloc[-1]), float(d["offense_pct"].sum()) / len(games))
    return out


def triggers(signals: list[dict], game: dict, shares_for, r: dict, now: datetime, done: set) -> list[dict]:
    """Qualifying 'regular is out' news for this game -> [{player, team, position, beneficiaries, seen_at}]."""
    from .player_stats import norm_name
    t, out = r["trigger"], []
    try:
        kick = datetime.fromisoformat(game["kickoff_utc"])
    except (KeyError, TypeError, ValueError):
        return []
    if (kick - now).total_seconds() < t["min_hours_before_kickoff"] * 3600:
        return []
    outs_by_team: dict = {}
    for s in signals:
        if s.get("signal") in t["signals"]:
            outs_by_team.setdefault(s.get("team_verified") or s.get("team"), set()).add(norm_name(s.get("player") or ""))
    for side in ("home", "away"):
        team, since = game[f"{side}_team"], _t(game.get(f"{side}_prev_end"))
        for s in signals:
            if (s.get("team_verified") or s.get("team")) != team or s.get("signal") not in t["signals"]:
                continue
            if float(s.get("certainty") or 0) < t["min_certainty"] or s.get("game_week_relevant") is False:
                continue
            if t["roster_match_required"] and (s.get("verify") or {}).get("roster") != "match":
                continue
            pub, seen = _t(s.get("published")) or _t(s.get("seen_at")), _t(s.get("seen_at"))
            if not seen or (now - seen).total_seconds() > t["first_seen_within_hours"] * 3600:
                continue
            if t["published_after_previous_game"] and since and pub and pub < since:
                continue
            key = f"{game['game_id']}:{norm_name(s['player'])}"
            if key in done:
                continue
            sh = shares_for(team)
            me = next(((pos, v) for p, (pos, v) in sh.items() if norm_name(p) == norm_name(s["player"])), None)
            if not me:
                continue
            pos = (s.get("position") or me[0] or "").upper()
            pos = "WR" if pos.startswith("WR") else "TE" if pos.startswith("TE") else "RB" if pos in ("RB", "HB", "FB") else pos
            if pos not in t["positions"] or me[1] < 0.5:
                continue
            ben = sorted(((p, v) for p, (ps, v) in sh.items() if ps == pos and 0.15 <= v <= 0.85
                          and norm_name(p) != norm_name(s["player"]) and norm_name(p) not in outs_by_team.get(team, set())),
                         key=lambda x: -x[1])[:2]
            out.append({"key": key, "player": s["player"], "team": team, "position": pos, "share": round(me[1], 3),
                        "beneficiaries": [{"player": p, "share": round(v, 3)} for p, v in ben],
                        "seen_at": s.get("seen_at"), "published": s.get("published"), "link": s.get("link")})
            done.add(key)
    return out


def quotes(resp: dict) -> list[dict]:
    """Main-line O/U quotes {book, title, market, player, point, over, under, p_nv} (both prices in -200..+170)."""
    out = {}
    for bk in (resp or {}).get("bookmakers", []):
        for m in bk.get("markets", []):
            for o in m.get("outcomes", []):
                name, who, pt, px = o.get("name"), o.get("description"), o.get("point"), o.get("price")
                if name not in ("Over", "Under") or not who or pt is None or px is None:
                    continue
                q = out.setdefault((bk["key"], m["key"], who, float(pt)), {"book": bk["key"], "title": bk.get("title", bk["key"]),
                                                                          "market": m["key"], "player": who, "point": float(pt)})
                q["over" if name == "Over" else "under"] = px
    res = []
    for q in out.values():
        if q.get("over") is None or q.get("under") is None or not (-200 <= q["over"] <= 170 and -200 <= q["under"] <= 170):
            continue
        io, iu = _imp(q["over"]), _imp(q["under"])
        q["p_nv"] = io / (io + iu)
        res.append(q)
    return res


def bets_for(trig: dict, game: dict, resp: dict, r: dict, allowed: set | None, now: datetime) -> list[dict]:
    from .player_stats import norm_name
    q = quotes(resp)
    out = []
    for b in trig["beneficiaries"]:
        for mk in r["markets"][trig["position"]]:
            mine = [x for x in q if x["market"] == mk and norm_name(x["player"]) == norm_name(b["player"])
                    and (allowed is None or x["book"] in allowed)]
            if not mine:
                continue
            best = min(mine, key=lambda x: (x["point"], -_dec(x["over"])))
            out.append({"id": f"{game['game_id']}:pnews:{mk}:{norm_name(b['player']).replace(' ', '_')}",
                        "track": "props_news", "rules_version": r["version"], "placed_at": now.isoformat(timespec="minutes"),
                        "game_id": game["game_id"], "season": game["season"], "week": game["week"], "gameday": game["gameday"],
                        "kickoff_utc": game["kickoff_utc"], "event_id": resp.get("id"), "market": mk, "player": b["player"],
                        "side": "over", "point": best["point"], "price": best["over"], "book": best["title"], "book_key": best["book"],
                        "units": 1.0, "status": "open", "edge": None, "stat_label": LABEL[mk],
                        "team": f"{b['player']} Over {best['point']:g} {LABEL[mk]}",
                        "opponent": f"{game['away_team']}@{game['home_team']}",
                        "why": f"{trig['player']} ({trig['team']} {trig['position']}, {trig['share']:.0%} of snaps) reported out; "
                               f"{b['player']} plays {b['share']:.0%} of snaps", "news_link": trig.get("link"),
                        "news_seen_at": trig.get("seen_at")})
    return out


def _close(bet: dict, snaps: list[dict]) -> tuple[float | None, float | None, str | None]:
    """(no-vig P(over) at the same point, median close point, source) from our last fetch <= 3 h before kickoff."""
    from .player_stats import norm_name
    kick = _t(bet["kickoff_utc"])
    best = None
    for s in snaps:
        ts = _t(s.get("fetched_at"))
        if s.get("event_id") == bet.get("event_id") and ts and kick and ts < kick and (kick - ts) <= timedelta(hours=3) \
                and (best is None or ts > best[0]):
            best = (ts, s)
    if not best:
        return None, None, None
    q = [x for x in quotes(best[1]["response"]) if x["market"] == bet["market"] and norm_name(x["player"]) == norm_name(bet["player"])]
    same = [x["p_nv"] for x in q if x["point"] == bet["point"]]
    pts = [x["point"] for x in q]
    return (statistics.median(same) if same else None, statistics.median(pts) if pts else None,
            f"our props fetch at {best[0].isoformat(timespec='minutes')}")


def grade(bet: dict, games: pd.DataFrame, snaps: list[dict], stats_for, now: datetime) -> dict:
    from . import player_stats as PS
    g = games[games["game_id"] == bet["game_id"]]
    if g.empty or not bool(g["completed"].iloc[0]):
        return bet
    stats, sn = stats_for(int(bet["season"]))
    status, val = PS.stat_value(bet["game_id"], bet["player"], stats, sn, STAT_COL[bet["market"]])
    if status == "open":
        return bet
    if status == "void":
        return dict(bet, status="void", void_reason=val)
    v = float(val)
    res = "win" if v > bet["point"] else "push" if v == bet["point"] else "loss"
    out = dict(bet, status="graded", result=res, final=f"{v:g} {bet['stat_label']}", stat=v)
    out["profit_units"] = round(_dec(bet["price"]) - 1, 3) if res == "win" else (0.0 if res == "push" else -1.0)
    p, pt, src = _close(bet, snaps)
    if p is not None:
        out["closing_prob"], out["clv"] = round(p, 4), round(p * _dec(bet["price"]) - 1, 4)
    if pt is not None:
        out["points_moved"] = round(pt - bet["point"], 2)
    out["clv_source"] = src or "no closing props fetch"
    return out


def _snaps(history_dir: Path) -> list[dict]:
    out = []
    for f in sorted(history_dir.glob("propsnews_*.json.gz")):
        try:
            out += json.loads(gzip.open(f, "rt").read())
        except Exception:
            continue
    return out


def recent_out_signals(history_dir: Path, now: datetime, hours: float) -> list[dict]:
    p = history_dir / "news_llm.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines()[-6000:]:
            try:
                s = json.loads(line)
            except ValueError:
                continue
            t = _t(s.get("seen_at"))
            if t and now - timedelta(hours=hours) <= t <= now:
                out.append(s)
    return out


def process(pred: dict, history_dir: Path, now: datetime | None = None, fetch=None, allowed: set | None | bool = False,
            signals: list[dict] | None = None, snaps_for=None, games: pd.DataFrame | None = None, stats_for=None) -> dict:
    """fetch(event_id, markets) -> Odds API event response (None = offline)."""
    from . import odds as O
    r = load_rules()
    now = now or datetime.now(timezone.utc)
    allowed = O.load_allowed_books() if allowed is False else allowed
    path, spath = history_dir / r["ledger"], history_dir / r["state"]
    ledger = json.loads(path.read_text()) if path.exists() else []
    state = json.loads(spath.read_text()) if spath.exists() else {"done": [], "triggers": []}
    done = set(state["done"])
    raw, new, calls = [], [], 0
    sig = signals if signals is not None else recent_out_signals(history_dir, now, r["trigger"]["first_seen_within_hours"] + 1)
    season = max((g.get("season") for g in pred.get("upcoming", [])), default=None)
    cache: dict = {}

    def shares_for(team):
        if "snaps" not in cache:
            try:
                cache["snaps"] = snaps_for() if snaps_for else __import__("nflpred.data", fromlist=["x"]).load_snaps([season])
            except Exception as e:
                print("props news: snap counts unavailable:", e)
                cache["snaps"] = pd.DataFrame(columns=["team", "game_id", "week", "player", "position", "offense_pct"])
        return snap_shares(cache["snaps"], team)

    open_games = {b["game_id"] for b in ledger if b.get("status") == "open"}
    for g in pred.get("upcoming", []):
        eid = ((g.get("context") or {}).get("live_odds") or {}).get("event_id")
        kick = _t(g.get("kickoff_utc"))
        if not eid or not kick or now >= kick:
            continue
        trig = triggers(sig, g, shares_for, r, now, done) if sig else []
        for tr in trig:
            state["triggers"].append({**{k: tr[k] for k in ("key", "player", "team", "position", "share", "seen_at")},
                                      "beneficiaries": tr["beneficiaries"], "at": now.isoformat(timespec="minutes")})
            mks = sorted({m for m in r["markets"][tr["position"]]})
            if fetch is None or not tr["beneficiaries"]:
                continue
            try:
                resp = fetch(eid, ",".join(mks)); calls += 1
            except Exception as e:
                print("props news: fetch failed", e); continue
            raw.append({"event_id": eid, "game_id": g["game_id"], "kind": "trigger", "fetched_at": now.isoformat(timespec="minutes"),
                        "response": resp})
            have = {b["id"] for b in ledger}
            for b in bets_for(tr, g, resp, r, allowed, now):
                if b["id"] not in have:
                    ledger.append(b); new.append(b); have.add(b["id"])
        # close fetch for games with open bets, 10-90 min before kickoff
        if fetch is not None and g["game_id"] in open_games and timedelta(minutes=10) <= kick - now <= timedelta(minutes=90) \
                and g["game_id"] not in state.setdefault("closed", []):
            mks = sorted({b["market"] for b in ledger if b["game_id"] == g["game_id"] and b.get("status") == "open"})
            try:
                resp = fetch(eid, ",".join(mks)); calls += 1
                raw.append({"event_id": eid, "game_id": g["game_id"], "kind": "close",
                            "fetched_at": now.isoformat(timespec="minutes"), "response": resp})
                state["closed"].append(g["game_id"])
            except Exception as e:
                print("props news: close fetch failed", e)
    if raw:
        with gzip.open(history_dir / f"propsnews_{now.strftime('%Y-%m-%dT%H%M')}.json.gz", "wt") as f:
            json.dump(raw, f)
    if games is not None and len(games) and any(b.get("status") == "open" for b in ledger):
        snaps = _snaps(history_dir)
        sf = stats_for
        if sf is None:
            from .props_receptions import _default_stats_for
            sf = _default_stats_for()
        try:
            ledger = [grade(b, games, snaps, sf, now) if b.get("status") == "open" else b for b in ledger]
        except Exception as e:
            print("props news: grading skipped:", e)
    state["done"] = sorted(done)[-5000:]
    state["triggers"] = state["triggers"][-500:]
    path.write_text(json.dumps(ledger, indent=1))
    spath.write_text(json.dumps(state, indent=1))
    gr = [b for b in ledger if b.get("status") == "graded"]
    cl = [b["clv"] for b in gr if isinstance(b.get("clv"), (int, float))]
    pm = [b["points_moved"] for b in gr if isinstance(b.get("points_moved"), (int, float))]
    return {"track": "props_news", "rules_version": r["version"], "mode": "paper", "new": new,
            "open": [b for b in ledger if b.get("status") == "open"], "recent_graded": gr[-30:], "calls": calls,
            "triggers_recent": state["triggers"][-10:],
            "record": {"n": len(gr), "w": sum(b["result"] == "win" for b in gr), "l": sum(b["result"] == "loss" for b in gr),
                       "units": round(sum(b.get("profit_units", 0) for b in gr), 2),
                       "avg_clv": round(sum(cl) / len(cl), 4) if cl else None,
                       "avg_points_moved": round(sum(pm) / len(pm), 3) if pm else None}}
