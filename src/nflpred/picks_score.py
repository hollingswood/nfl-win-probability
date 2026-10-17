"""Score logged published picks (history/picks_log.jsonl) and run the pre-registered hot-pickers test
(picks_hot_rules.json). Logging and scoring only; nothing bets.

Each pick seen before kickoff is matched to its game (NFL: nflverse schedule; college: CFBD) and graded at the CLOSING
line: spreads and totals 1 unit at -110 (pushes don't count), moneylines straight up and in units at the closing price.
'Line value' = points the market moved toward the pick between the picker's stated number and the close.
Writes history/picks_scores.json: per-picker records, the hot list for the coming week, and the hot test so far.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import picks_check as PC

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
RULES = ROOT / "picks_hot_rules.json"
NICK = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
        "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
        "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LV": "Raiders", "LAC": "Chargers", "LA": "Rams",
        "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
        "PHI": "Eagles", "PIT": "Steelers", "SF": "49ers", "SEA": "Seahawks", "TB": "Buccaneers", "TEN": "Titans",
        "WAS": "Commanders"}
WIN_110 = 100 / 110


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def nfl_abbr(name: str) -> str | None:
    n = (name or "").lower()
    hits = [a for a, nick in NICK.items() if nick.lower() in n]
    if len(hits) == 1:
        return hits[0]
    if (name or "").upper() in NICK:
        return name.upper()
    return None


def _dec(a) -> float | None:
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if math.isnan(a) or a == 0:
        return None
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def nfl_games() -> pd.DataFrame:
    from . import data
    try:
        g = data.load_schedules(refresh=True)      # nflverse schedule with this week's results
    except Exception:
        g = data.load_schedules()
    g = g[g.season >= 2026].copy()
    day = pd.to_datetime(g.gameday).dt.strftime("%Y-%m-%d")
    g["kick"] = pd.to_datetime(day + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g


def cfb_games() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cfbpred import data as D
    yrs = sorted({datetime.now(timezone.utc).year - 1, datetime.now(timezone.utc).year})
    g = D.games(yrs)
    L = D.lines(yrs, prefer=D.PREFERRED_PROVIDERS)
    g = g.merge(L, on="game_id", how="left")
    g["kick"] = pd.to_datetime(g.start, utc=True)
    return g


def drop_superseded(picks: list[dict], hist: Path) -> tuple[list[dict], int]:
    """An article's latest read lists every pick it still shows (picks_reads.json 'keys'); picks logged from that article
    earlier but missing from its latest read were misreads and are dropped (picks_check layer 4)."""
    try:
        reads = json.loads((hist / "picks_reads.json").read_text())
    except (OSError, ValueError):
        return picks, 0
    latest = {r["link"]: (r.get("last", ""), set(r["keys"])) for r in reads.values() if r.get("link") and "keys" in r}
    from .picks_log import pick_key
    keep, n = [], 0
    for p in picks:
        lk = latest.get(p.get("link"))
        if lk and p.get("seen_at", "") <= lk[0] and "|".join(map(str, pick_key(p))) not in lk[1]:
            n += 1; continue
        keep.append(p)
    return keep, n


def grade_pick(p: dict, nfl: pd.DataFrame, cfb: pd.DataFrame, cmatch) -> dict | None:
    """-> {game_id, week, kick, result: 'win'|'loss'|'push', units, line_value} or None (unmatched / not final / late)."""
    seen = datetime.fromisoformat(p["seen_at"])
    mk, side = p.get("market"), str(p.get("pick") or "")
    if p.get("sport") == "nfl":
        h, a = nfl_abbr(p.get("home_team")), nfl_abbr(p.get("away_team"))
        if not h or not a:
            return None
        g = nfl[(((nfl.home_team == h) & (nfl.away_team == a)) | ((nfl.home_team == a) & (nfl.away_team == h)))
                & (nfl.kick > pd.Timestamp(seen) - timedelta(days=1)) & (nfl.kick < pd.Timestamp(seen) + timedelta(days=10))]
        if g.empty:
            return None
        r = g.sort_values("kick").iloc[0]
        if r.kick <= pd.Timestamp(seen):
            return {"late": True}
        if pd.isna(r.home_score):
            return {"pending": True, "kick": r.kick.isoformat(), "away_actual": r.away_team, "home_actual": r.home_team}
        hs, as_, home, away = float(r.home_score), float(r.away_score), r.home_team, r.away_team
        home_margin_line = float(r.spread_line) if pd.notna(r.spread_line) else None     # + = home favored
        total = float(r.total_line) if pd.notna(r.total_line) else None
        ml = {home: _dec(r.home_moneyline), away: _dec(r.away_moneyline)}
        picked = nfl_abbr(side)
        gid, week, kick = r.game_id, int(r.week), r.kick
    else:
        h, a = cmatch(p.get("home_team") or ""), cmatch(p.get("away_team") or "")
        if not h or not a:
            return None
        g = cfb[(((cfb.home == h) & (cfb.away == a)) | ((cfb.home == a) & (cfb.away == h)))
                & (cfb.kick > pd.Timestamp(seen) - timedelta(days=1)) & (cfb.kick < pd.Timestamp(seen) + timedelta(days=10))]
        if g.empty:
            return None
        r = g.sort_values("kick").iloc[0]
        if r.kick <= pd.Timestamp(seen):
            return {"late": True}
        if not r.completed or pd.isna(r.home_pts):
            return {"pending": True, "kick": r.kick.isoformat(), "away_actual": r.away, "home_actual": r.home}
        hs, as_, home, away = float(r.home_pts), float(r.away_pts), r.home, r.away
        home_margin_line = -float(r.spread_close) if pd.notna(r.spread_close) else None
        total = float(r.total_close) if pd.notna(r.total_close) else None
        ml = {home: _dec(r.home_ml), away: _dec(r.away_ml)}
        picked = cmatch(side)
        gid, week, kick = str(r.game_id), int(r.week), r.kick
    out = {"game_id": str(gid), "week": week, "kick": kick.isoformat(), "away_actual": away, "home_actual": home}
    if mk == "total":
        if total is None or side.lower() not in ("over", "under"):
            return None
        why = PC.grade_check("total", p.get("line"), total)
        if why:
            return {"excluded": why}
        d = (hs + as_ - total) * (1 if side.lower() == "over" else -1)
        out["close_line"], out["final"] = total, f"{int(as_)}-{int(hs)}"
        out["result"] = "push" if d == 0 else "win" if d > 0 else "loss"
        out["units"] = 0.0 if d == 0 else (WIN_110 if d > 0 else -1.0)
        st = p.get("line")
        if isinstance(st, (int, float)) and abs(st - total) <= 7:
            out["line_value"] = round((total - st) * (1 if side.lower() == "over" else -1), 2)
        return out
    if picked not in (home, away):
        return None
    pm = (hs - as_) if picked == home else (as_ - hs)
    if mk == "spread":
        if home_margin_line is None:
            return None
        close_pick = -home_margin_line if picked == home else home_margin_line    # picked team's spread at the close
        why = PC.grade_check("spread", p.get("line"), close_pick)
        if why:
            return {"excluded": why}
        d = pm + close_pick
        out["close_line"], out["final"] = close_pick, f"{int(as_)}-{int(hs)}"
        out["result"] = "push" if d == 0 else "win" if d > 0 else "loss"
        out["units"] = 0.0 if d == 0 else (WIN_110 if d > 0 else -1.0)
        st = p.get("line")
        if isinstance(st, (int, float)) and abs(st - close_pick) <= 7:
            out["line_value"] = round(st - close_pick, 2)    # + = picker's number was better than the close
        return out
    if mk == "moneyline":
        out["final"] = f"{int(as_)}-{int(hs)}"
        d0 = ml.get(picked)
        out["close_line"] = (round((d0 - 1) * 100) if d0 >= 2 else round(-100 / (d0 - 1))) if d0 else None
        out["result"] = "win" if pm > 0 else "loss" if pm < 0 else "push"
        d = ml.get(picked)
        out["units"] = (round(d - 1, 3) if pm > 0 else -1.0 if pm < 0 else 0.0) if d else None
        return out
    return None


# Display/cleaning only (frozen hot-picker rule v1 counts "named analysts or accounts"): one name per person or
# outlet, and items that are not anyone's pick (market prices quoted in an article, fan-forum predictions) are skipped.
NAME_ALIASES = {"zach cohen": "Zachary Cohen", "sportsline": "SportsLine Projection Model"}
OUTLET_ALIASES = {"yahoo_nfl": "Yahoo Sports", "yahoo_cfb": "Yahoo Sports", "cbs_nfl": "CBS Sports", "cbs_cfb": "CBS Sports",
                  "CBS Sports (SportsLine)": "SportsLine", "vsin": "VSiN", "sdsouth": "Saturday Down South", "fox_nfl": "FOX Sports",
                  "pfn": "Pro Football Network"}
MARKET_OUTLETS = {"polymarket", "kalshi", "betmgm", "caesars sports", "draftkings", "fanduel", "fanatics", "espn bet", "thescore bet"}


def outlet_of(p: dict) -> str:
    o = p.get("outlet") or p.get("source") or ""
    if str(o).startswith("bsky:"):
        return "Bluesky"
    return OUTLET_ALIASES.get(o, o)


def not_a_pick(p: dict) -> str | None:
    """Why a logged item isn't anyone's pick (None = it is): odds/market prices quoted in an article, fan-forum posts."""
    words = f"{p.get('confidence_words') or ''}".lower()
    if not (p.get("analyst") or "").strip() and str(p.get("outlet") or "").lower() in MARKET_OUTLETS:
        return "market price, not a pick"
    if "per share" in words or "trading at $" in words:
        return "market price, not a pick"
    if " fans:" in f" {(p.get('title') or '').lower()}" or "fans divid" in (p.get("title") or "").lower():
        return "fan-forum prediction"
    return None


def picker_of(p: dict) -> str:
    a = (p.get("analyst") or "").strip()
    if p.get("source") == "bluesky":
        h = (p.get("outlet") or "bluesky").split(" (")[0]
        return "@" + h[5:].replace(".bsky.social", "") if h.startswith("bsky:") else h
    if a:
        return NAME_ALIASES.get(a.lower(), a)
    o = outlet_of(p)
    return o if "model" in o.lower() else f"{o} staff"


def is_named(picker: str) -> bool:
    return not picker.endswith(" staff")


def _p_cover(w: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(((w - 0.5) - mu) / sd / math.sqrt(2))


def _detail(p: dict) -> dict:
    return {"picker": picker_of(p), "sport": p.get("sport"), "seen_at": p.get("seen_at"), "away": p.get("away_team"),
            "home": p.get("home_team"), "market": p.get("market"), "pick": p.get("pick"), "line": p.get("line"),
            "link": p.get("link"), "quote": (p.get("confidence_words") or "")[:160]}


def table(graded: list[dict], logged: dict) -> list[dict]:
    """Per-picker records from graded picks, plus pickers whose logged picks are all still pending."""
    by: dict = {}
    for x in graded:
        b = by.setdefault(x["picker"], {"picker": x["picker"], "outlet": x["outlet"], "n": 0, "w": 0, "l": 0, "p": 0, "units": 0.0,
                                        "lv": [], "sports": set()})
        b["n"] += 1; b["sports"].add(x["sport"])
        b[{"win": "w", "loss": "l", "push": "p"}[x["result"]]] += 1
        b["units"] += x.get("units") or 0.0
        if x.get("line_value") is not None:
            b["lv"].append(x["line_value"])
    pickers = []
    for b in by.values():
        d = b["w"] + b["l"]
        pickers.append({"picker": b["picker"], "outlet": b["outlet"], "sports": sorted(b["sports"]), "graded": b["n"], "wins": b["w"],
                        "losses": b["l"], "pushes": b["p"], "win_pct": round(b["w"] / d, 3) if d else None, "units": round(b["units"], 2),
                        "avg_line_value": round(sum(b["lv"]) / len(b["lv"]), 2) if b["lv"] else None, "n_line_value": len(b["lv"])})
    have = {x["picker"] for x in pickers}
    for x in pickers:
        lg = logged.get(x["picker"], {})
        x["logged"], x["pending"] = lg.get("n", x["graded"]), max(0, lg.get("n", x["graded"]) - x["graded"] - lg.get("late", 0))
    for k, lg in logged.items():      # pickers with nothing graded yet (their games are still to be played)
        if isinstance(k, str) and k not in have:
            pickers.append({"picker": k, "outlet": lg["outlet"], "sports": sorted(lg["sports"]), "graded": 0, "wins": 0, "losses": 0,
                            "pushes": 0, "win_pct": None, "units": 0.0, "avg_line_value": None, "n_line_value": 0,
                            "logged": lg["n"], "pending": max(0, lg["n"] - lg["late"])})
    pickers.sort(key=lambda x: (-(x["graded"] > 0), -(x["units"]), -x["graded"], -x["logged"]))
    return pickers


def run(hist: Path = HIST, now: datetime | None = None, nfl=None, cfb=None) -> dict:
    now = now or datetime.now(timezone.utc)
    r = load_rules()
    path = hist / "picks_log.jsonl"
    picks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    picks, superseded = drop_superseded(picks, hist)          # picks_check layer 4: an article's latest read wins
    nfl = nfl if nfl is not None else nfl_games()
    try:
        cfb = cfb if cfb is not None else cfb_games()
    except Exception as e:
        print("picks: college games unavailable:", e)
        cfb = pd.DataFrame(columns=["home", "away", "kick", "completed", "home_pts"])
    from cfbpred.pipeline import team_matcher
    cmatch = team_matcher(sorted(set(cfb.home) | set(cfb.away))) if len(cfb) else (lambda s: None)
    graded, seen_keys, late = [], set(), 0
    logged: dict = {}                 # every distinct NFL / college pick per picker (graded, pending or late)
    pending: list = []                # recent picks whose games haven't been graded yet (shown when you open a picker)
    # last-name-only bylines ("Hummer") -> the full name the same outlet uses elsewhere ("Chris Hummer")
    full: dict = {}
    for p in picks:
        n = (p.get("analyst") or "").strip()
        if len(n.split()) >= 2:
            full.setdefault((p.get("source"), n.split()[-1].lower()), set()).add(n)
    for p in picks:
        n = (p.get("analyst") or "").strip()
        if n and len(n.split()) == 1 and len(full.get((p.get("source"), n.lower()), ())) == 1:
            p["analyst"] = next(iter(full[(p.get("source"), n.lower())]))
    skipped: dict = {"dropped when the article was re-read": superseded} if superseded else {}
    for p, why in zip(picks, PC.offline_reasons(picks)):     # picks_check: wrong side / both sides, also for old picks
        p["_check"] = why
    for p in picks:
        why = p.pop("_check", None) or not_a_pick(p)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        k = (picker_of(p), p.get("sport"), p.get("home_team"), p.get("away_team"), p.get("market"), str(p.get("pick")).lower())
        if k in seen_keys:            # the same pick repeated in another article / run
            continue
        seen_keys.add(k)
        if p.get("sport") in ("nfl", "cfb"):
            for key in (picker_of(p), (picker_of(p), p.get("sport"))):     # all sports, and per sport
                lg = logged.setdefault(key, {"n": 0, "outlet": outlet_of(p), "sports": set(), "late": 0})
                lg["n"] += 1; lg["sports"].add(p.get("sport"))
        try:
            g = grade_pick(p, nfl, cfb, cmatch)
        except Exception:
            g = None
        if g and g.get("excluded"):               # picks_check layer 3: the stated line can't be this game's
            skipped[g["excluded"]] = skipped.get(g["excluded"], 0) + 1
            continue
        if g and g.get("pending"):                # matched to a game not played yet
            pending.append(dict(_detail(p), kick=g["kick"], away=g["away_actual"], home=g["home_actual"]))
            continue
        if not g:                                  # no matching game (not counted, not shown as pending)
            continue
        if g.get("late"):
            late += 1
            for key in (picker_of(p), (picker_of(p), p.get("sport"))):
                if key in logged:
                    logged[key]["late"] += 1
            continue
        graded.append({**g, "picker": picker_of(p), "sport": p.get("sport"), "market": p.get("market"), "pick": p.get("pick"),
                       "line": p.get("line"), "seen_at": p["seen_at"], "outlet": outlet_of(p),
                       "away": g.get("away_actual") or p.get("away_team"), "home": g.get("home_actual") or p.get("home_team"),
                       "link": p.get("link")})
    pickers = table(graded, {k: v for k, v in logged.items() if isinstance(k, str)})
    # hot test: per sport and week, hot = >= 60% with >= 8 graded over the 4 previous weeks; follow next week's picks
    test = {"followed": 0, "w": 0, "l": 0, "units": 0.0, "others_w": 0, "others_l": 0}
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        for wk in sorted({x["week"] for x in rows}):
            prior = [x for x in rows if wk - 4 <= x["week"] < wk]
            rec: dict = {}
            for x in prior:
                t = rec.setdefault(x["picker"], [0, 0]); t[0] += x["result"] == "win"; t[1] += x["result"] == "loss"
            hot = {k for k, (w, l) in rec.items() if is_named(k) and w + l >= 8 and w / (w + l) >= 0.60}
            for x in rows:
                if x["week"] != wk or x["result"] == "push":
                    continue
                if x["picker"] in hot:
                    test["followed"] += 1; test["w"] += x["result"] == "win"; test["l"] += x["result"] == "loss"; test["units"] += x.get("units") or 0
                else:
                    test["others_w"] += x["result"] == "win"; test["others_l"] += x["result"] == "loss"
    n = test["w"] + test["l"]
    test["cover_rate"] = round(test["w"] / n, 4) if n else None
    no = test["others_w"] + test["others_l"]
    test["others_rate"] = round(test["others_w"] / no, 4) if no else None
    test["p_value"] = round(_p_cover(test["w"], n, r["validation"]["cover_rate_vs_close_above"]), 4) if n else None
    v = r["validation"]
    test["passed"] = bool(n >= v["min_followed_picks"] and test["cover_rate"] > v["cover_rate_vs_close_above"]
                          and test["p_value"] < v["one_sided_p_below"] and (test["others_rate"] is None or test["cover_rate"] > test["others_rate"]))
    test["units"] = round(test["units"], 2)
    # hot list for the coming week (each sport's next week)
    hot_now = []
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        if not rows:
            continue
        nxt = max(x["week"] for x in rows) + 1
        rec = {}
        for x in rows:
            if nxt - 4 <= x["week"] < nxt:
                t = rec.setdefault(x["picker"], [0, 0]); t[0] += x["result"] == "win"; t[1] += x["result"] == "loss"
        for k, (w, l) in rec.items():
            if is_named(k) and w + l >= 8 and w / (w + l) >= 0.60:
                hot_now.append({"picker": k, "sport": sport, "week": nxt, "record": f"{w}-{l}"})
    by_sport = {}
    for sport in ("nfl", "cfb"):   # each page shows its own league
        gs = [x for x in graded if x["sport"] == sport]
        lg = {k[0]: v for k, v in logged.items() if isinstance(k, tuple) and k[1] == sport}
        tb = table(gs, lg)
        shown = {x["picker"] for x in tb[:250]}
        detail: dict = {}
        norm = (lambda t: nfl_abbr(t) or str(t)) if sport == "nfl" else (lambda t: cmatch(t or "") or str(t))
        gseen = set()
        for x in sorted(gs, key=lambda x: x["kick"], reverse=True):
            gk = (x["picker"], x["game_id"], x["market"], norm(x["pick"]))
            if gk in gseen:      # same pick logged twice with home/away swapped
                continue
            gseen.add(gk)
            if x["picker"] in shown:
                d = detail.setdefault(x["picker"], {"pending": [], "graded": []})
                if len(d["graded"]) < 40:
                    d["graded"].append({k: x.get(k) for k in ("kick", "week", "away", "home", "market", "pick", "line", "close_line",
                                                              "result", "units", "line_value", "final", "link")})
        done = {(x["picker"], x["market"], norm(x["pick"])) for x in gs}
        seen_p = set()
        for x in sorted((y for y in pending if y["sport"] == sport), key=lambda y: y["seen_at"], reverse=True):
            key = (x["picker"], x["market"], norm(x["pick"]), norm(x["home"]))
            if key[:3] in done or key in seen_p:     # already graded under another spelling, or a repeat
                continue
            seen_p.add(key)
            if x["picker"] in shown:
                d = detail.setdefault(x["picker"], {"pending": [], "graded": []})
                if len(d["pending"]) < 30:
                    d["pending"].append(x)
        by_sport[sport] = {"pickers": tb[:250], "picks_by_picker": detail, "n_pickers": len(tb), "picks_graded": len(gs),
                           "picks_logged": sum(v["n"] for v in lg.values()), "hot_now": [h for h in hot_now if h["sport"] == sport],
                           "overall": {"w": sum(x["result"] == "win" for x in gs), "l": sum(x["result"] == "loss" for x in gs),
                                       "units": round(sum(x.get("units") or 0 for x in gs), 2),
                                       "avg_line_value": round(sum(x["line_value"] for x in gs if x.get("line_value") is not None) /
                                                               max(1, sum(x.get("line_value") is not None for x in gs)), 2)}}
    out = {"generated_at": now.isoformat(timespec="minutes"), "rules_version": r["version"], "picks_logged": len(picks), "not_picks_skipped": skipped, "by_sport": by_sport,
           "picks_graded": len(graded), "late_picks_dropped": late, "pickers": pickers[:250], "n_pickers": len(pickers), "hot_now": hot_now, "hot_test": test,
           "overall": {"w": sum(x["result"] == "win" for x in graded), "l": sum(x["result"] == "loss" for x in graded),
                       "units": round(sum(x.get("units") or 0 for x in graded), 2),
                       "avg_line_value": round(sum(x["line_value"] for x in graded if x.get("line_value") is not None) /
                                               max(1, sum(x.get("line_value") is not None for x in graded)), 2)}}
    (hist / "picks_scores.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    res = run()
    print(json.dumps({k: res[k] for k in ("picks_logged", "picks_graded", "late_picks_dropped", "overall", "hot_test")}, indent=1))
    for p in res["pickers"][:15]:
        print(p)
