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
        if pd.isna(r.home_score) or r.kick <= pd.Timestamp(seen):
            return {"late": True} if r.kick <= pd.Timestamp(seen) else None
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
            return None
        hs, as_, home, away = float(r.home_pts), float(r.away_pts), r.home, r.away
        home_margin_line = -float(r.spread_close) if pd.notna(r.spread_close) else None
        total = float(r.total_close) if pd.notna(r.total_close) else None
        ml = {home: _dec(r.home_ml), away: _dec(r.away_ml)}
        picked = cmatch(side)
        gid, week, kick = str(r.game_id), int(r.week), r.kick
    out = {"game_id": str(gid), "week": week, "kick": kick.isoformat()}
    if mk == "total":
        if total is None or side.lower() not in ("over", "under"):
            return None
        d = (hs + as_ - total) * (1 if side.lower() == "over" else -1)
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
        d = pm + close_pick
        out["result"] = "push" if d == 0 else "win" if d > 0 else "loss"
        out["units"] = 0.0 if d == 0 else (WIN_110 if d > 0 else -1.0)
        st = p.get("line")
        if isinstance(st, (int, float)) and abs(st - close_pick) <= 7:
            out["line_value"] = round(st - close_pick, 2)    # + = picker's number was better than the close
        return out
    if mk == "moneyline":
        out["result"] = "win" if pm > 0 else "loss" if pm < 0 else "push"
        d = ml.get(picked)
        out["units"] = (round(d - 1, 3) if pm > 0 else -1.0 if pm < 0 else 0.0) if d else None
        return out
    return None


def picker_of(p: dict) -> str:
    a = (p.get("analyst") or "").strip()
    if p.get("source") == "bluesky":
        return (p.get("outlet") or "bluesky").split(" (")[0]
    return a or f"{p.get('outlet') or p.get('source')} (unnamed)"


def _p_cover(w: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(((w - 0.5) - mu) / sd / math.sqrt(2))


def run(hist: Path = HIST, now: datetime | None = None, nfl=None, cfb=None) -> dict:
    now = now or datetime.now(timezone.utc)
    r = load_rules()
    path = hist / "picks_log.jsonl"
    picks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
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
    for p in picks:
        k = (picker_of(p), p.get("sport"), p.get("home_team"), p.get("away_team"), p.get("market"), str(p.get("pick")).lower())
        if k in seen_keys:            # the same pick repeated in another article / run
            continue
        seen_keys.add(k)
        if p.get("sport") in ("nfl", "cfb"):
            lg = logged.setdefault(picker_of(p), {"n": 0, "outlet": p.get("outlet") or p.get("source"), "sports": set(), "late": 0})
            lg["n"] += 1; lg["sports"].add(p.get("sport"))
        try:
            g = grade_pick(p, nfl, cfb, cmatch)
        except Exception:
            g = None
        if not g:
            continue
        if g.get("late"):
            late += 1
            if picker_of(p) in logged:
                logged[picker_of(p)]["late"] += 1
            continue
        graded.append({**g, "picker": picker_of(p), "sport": p.get("sport"), "market": p.get("market"), "pick": p.get("pick"),
                       "line": p.get("line"), "seen_at": p["seen_at"], "outlet": p.get("outlet") or p.get("source")})
    # per picker
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
        if k not in have:
            pickers.append({"picker": k, "outlet": lg["outlet"], "sports": sorted(lg["sports"]), "graded": 0, "wins": 0, "losses": 0,
                            "pushes": 0, "win_pct": None, "units": 0.0, "avg_line_value": None, "n_line_value": 0,
                            "logged": lg["n"], "pending": max(0, lg["n"] - lg["late"])})
    pickers.sort(key=lambda x: (-(x["graded"] > 0), -(x["units"]), -x["graded"], -x["logged"]))
    # hot test: per sport and week, hot = >= 60% with >= 8 graded over the 4 previous weeks; follow next week's picks
    H = r["hot"]
    test = {"followed": 0, "w": 0, "l": 0, "units": 0.0, "others_w": 0, "others_l": 0}
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        for wk in sorted({x["week"] for x in rows}):
            prior = [x for x in rows if wk - 4 <= x["week"] < wk]
            rec: dict = {}
            for x in prior:
                t = rec.setdefault(x["picker"], [0, 0]); t[0] += x["result"] == "win"; t[1] += x["result"] == "loss"
            hot = {k for k, (w, l) in rec.items() if w + l >= 8 and w / (w + l) >= 0.60}
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
            if w + l >= 8 and w / (w + l) >= 0.60:
                hot_now.append({"picker": k, "sport": sport, "week": nxt, "record": f"{w}-{l}"})
    out = {"generated_at": now.isoformat(timespec="minutes"), "rules_version": r["version"], "picks_logged": len(picks),
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
