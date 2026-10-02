"""Outcome audit of the AI news reader (history/news_llm.jsonl) and its pre-registered promotion rule.

Run on full runs (pipeline update). For every AI signal whose game has finished, check it against ground truth:

  * participation — nflverse snap counts (any offense/defense/special-teams snap = played); the weekly
    stats_player file (player_stats.load_week_stats) is the fallback while snap counts are not yet published.
    'out' / 'injured_reserve' / 'doubtful' / 'suspended' are correct if the player did NOT play;
    'will_start' / 'expected_to_play' / 'returning' are correct if he played.
  * starting-QB signals (is_starting_qb_news, QB) — compared to the actual starter (schedule home_qb_id /
    away_qb_id): positive signals are correct if he started, 'out'/'injured_reserve'/'doubtful'/'suspended'/
    'benched' if he did not.
  * official final injury report (nflverse injuries) — agreement recorded alongside (not the precision basis).

Lead time: hours between first seen and (a) the official final report time proxy (16:00 ET two days before
the game, one day for Thursday games: Friday for Sunday games) and (b) for QB signals, the consensus spread move
(median over books, from history/odds_*.json[.gz]) in the 6 h after first seen vs the 6 h before.

Units: one audited unit per (game, player, signal), timed at the first time we saw it (the feeds repeat
stories); a unit is in the promotion rule's population if any of its copies met the rule's conditions.
Game assignment uses `team_verified` (roster.py); signals whose team is unclear are not audited.

Output: history/news_audit.json. The promotion rule lives in news_rules.json (frozen; bump the version to
change it). `promotion_check` reports eligibility only: nothing here feeds the model.
"""
from __future__ import annotations

import gzip
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .player_stats import match as name_match, norm_name

ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = ROOT / "news_rules.json"
ET = ZoneInfo("America/New_York")
NEG = {"out", "injured_reserve", "doubtful", "suspended"}   # expect: did not play
POS = {"will_start", "expected_to_play", "returning"}       # expect: played
QB_NEG = NEG | {"benched"}                                  # expect: did not start
MAX_DAYS_AHEAD = 10                                         # signal -> the team's next game within 10 days (Thu -> next Sun)
MOVE_HOURS = 6


def _ts(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def load_signals(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            s = json.loads(line)
        except ValueError:
            continue
        if isinstance(s, dict) and _ts(s.get("seen_at")):
            out.append(s)
    return out


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def signal_team(s: dict) -> str | None:
    """Team used for game assignment: team_verified once checked (None = team unclear), else the AI's team."""
    return s.get("team_verified") if "verify" in s else s.get("team")


def is_qb_signal(s: dict) -> bool:
    pos = (s.get("position") or "").upper()
    return bool(s.get("is_starting_qb_news")) and pos in ("QB", "")


def kickoff(g) -> datetime:
    gd = pd.Timestamp(g["gameday"])
    hh, mm = (int(x) for x in str(g.get("gametime") or "13:00").split(":"))
    return datetime(gd.year, gd.month, gd.day, hh, mm, tzinfo=ET).astimezone(timezone.utc)


def report_time(g) -> datetime:
    """Proxy for the official final injury report: 16:00 ET two days before the game (Friday for Sunday games,
    Saturday for Monday), one day before for Thursday games."""
    gd = pd.Timestamp(g["gameday"]).date()
    back = 1 if gd.weekday() == 3 else 2
    d = gd - timedelta(days=back)
    return datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET).astimezone(timezone.utc)


def assign_game(s: dict, games: pd.DataFrame) -> dict | None:
    team, seen = signal_team(s), _ts(s["seen_at"])
    if not team:
        return None
    g = games[(games["home_team"] == team) | (games["away_team"] == team)]
    best = None
    for _, r in g.iterrows():
        k = kickoff(r)
        if seen < k <= seen + timedelta(days=MAX_DAYS_AHEAD) and (best is None or k < best[0]):
            best = (k, r)
    return None if best is None else best[1].to_dict()


# ---------------------------------------------------------------------------------------------- ground truth
def played(s: dict, game: dict, team: str, snaps: pd.DataFrame, stats: pd.DataFrame,
           gsis_to_pfr: dict) -> bool | None:
    """True/False from snap counts (stats_player as fallback); None while neither is published for the game."""
    gid = game["game_id"]
    sn = snaps[snaps["game_id"] == gid] if len(snaps) else snaps
    if len(sn):
        sn = sn[sn["team"] == team] if "team" in sn else sn
        gsis = (s.get("verify") or {}).get("gsis_id")
        pfr = gsis_to_pfr.get(gsis) if gsis else None
        row = sn[sn["pfr_player_id"] == pfr] if pfr and "pfr_player_id" in sn else sn.iloc[0:0]
        if row.empty:
            hit = name_match(s.get("player") or "", {i: {norm_name(p)} for i, p in zip(sn.index, sn["player"])})
            row = sn.loc[[hit]] if hit is not None else row
        if row.empty:
            return False  # snap counts list everyone who took a snap
        tot = sum(float(row.iloc[0].get(c) or 0) for c in ("offense_snaps", "defense_snaps", "st_snaps"))
        return tot > 0
    st = stats[stats["game_id"] == gid] if len(stats) and "game_id" in stats else stats.iloc[0:0]
    if len(st):
        gsis = (s.get("verify") or {}).get("gsis_id")
        if gsis and "player_id" in st and (st["player_id"] == gsis).any():
            return True
        cands = {i: {norm_name(x) for x in (r.get("player_display_name"), r.get("player_name")) if isinstance(x, str)}
                 for i, r in st.iterrows()}
        if name_match(s.get("player") or "", cands) is not None:
            return True
    return None  # a missing stat line is not proof he sat


def started(s: dict, game: dict, side: str, id_to_name: dict) -> bool | None:
    qid = game.get(f"{side}_qb_id")
    if qid is None or (isinstance(qid, float) and pd.isna(qid)):
        return None
    gsis = (s.get("verify") or {}).get("gsis_id")
    if gsis:
        return gsis == qid
    nm = game.get(f"{side}_qb_name") or id_to_name.get(qid) or ""
    return name_match(s.get("player") or "", {0: {norm_name(nm), norm_name(id_to_name.get(qid, ""))} - {""}}) == 0


def official_status(s: dict, game: dict, team: str, injuries: pd.DataFrame) -> tuple[bool, str | None]:
    """(team had a final report that week, player's report_status or None)."""
    if injuries is None or not len(injuries):
        return False, None
    w = injuries[(injuries["season"] == game["season"]) & (injuries["week"] == game["week"])
                 & (injuries["team"] == team)]
    if not w["report_status"].notna().any():
        return False, None
    gsis = (s.get("verify") or {}).get("gsis_id")
    row = w[w["gsis_id"] == gsis] if gsis else w.iloc[0:0]
    if row.empty:
        hit = name_match(s.get("player") or "", {i: {norm_name(n)} for i, n in zip(w.index, w["full_name"].fillna(""))})
        row = w.loc[[hit]] if hit is not None else row
    st = row["report_status"].dropna()
    return True, (str(st.iloc[-1]) if len(st) else None)


def report_agrees(signal: str, status: str | None) -> bool | None:
    if signal in ("out", "suspended"):
        return status == "Out"
    if signal == "doubtful":
        return status in ("Doubtful", "Out")
    if signal in POS:
        return status not in ("Doubtful", "Out")
    return None  # injured_reserve: IR players are not on the weekly report


# ---------------------------------------------------------------------------------------------- odds
def spread_series(history_dir: Path) -> dict[tuple[str, str, str], list[tuple[datetime, float]]]:
    """(home, away, commence date) -> [(snapshot time, consensus home margin)] from saved odds snapshots."""
    from . import odds as O
    out: dict = {}
    files = list(history_dir.glob("odds_*.json")) + list(history_dir.glob("odds_*.json.gz"))
    for f in files:
        try:
            ts = datetime.strptime(f.name.split(".")[0][5:], "%Y-%m-%dT%H%M").replace(tzinfo=timezone.utc)
            events = json.loads(gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text())
        except Exception:
            continue
        for ev in events or []:
            home, away = O.TEAM_ABBR.get(ev.get("home_team")), O.TEAM_ABBR.get(ev.get("away_team"))
            if not home or not away:
                continue
            pts = []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") != "spreads":
                        continue
                    for o in mk.get("outcomes", []):
                        if o.get("name") == ev["home_team"] and o.get("point") is not None:
                            pts.append(-float(o["point"]))
            if pts:
                out.setdefault((home, away, str(ev.get("commence_time", ""))[:10]), []).append(
                    (ts, statistics.median(pts)))
    for v in out.values():
        v.sort()
    return out


def _at(series, t: datetime, tol_h: float = 3) -> float | None:
    """Value of the last snapshot at or before t (no older than tol_h hours)."""
    best = None
    for ts, v in series:
        if ts <= t:
            best = (ts, v)
        else:
            break
    return best[1] if best and (t - best[0]) <= timedelta(hours=tol_h) else None


def line_move(series, seen: datetime, home: bool, now: datetime) -> dict | None:
    """Team-perspective consensus margin move in the MOVE_HOURS after vs before first seen."""
    if not series or seen + timedelta(hours=MOVE_HOURS) > now:
        return None
    v0, va, vb = (_at(series, seen), _at(series, seen + timedelta(hours=MOVE_HOURS)),
                  _at(series, seen - timedelta(hours=MOVE_HOURS)))
    if v0 is None or va is None:
        return None
    sgn = 1 if home else -1
    return {"after": round(sgn * (va - v0), 2), "before": None if vb is None else round(sgn * (v0 - vb), 2)}


def _series_for(spreads: dict, game: dict):
    k = kickoff(game)
    for d in (k.date(), k.date() - timedelta(days=1)):
        v = spreads.get((game["home_team"], game["away_team"], d.isoformat()))
        if v:
            return v
    return None


# ---------------------------------------------------------------------------------------------- audit
def _pct(c, n):
    return round(100 * c / n, 1) if n else None


def audit(signals: list[dict], games: pd.DataFrame, snaps: pd.DataFrame, stats: pd.DataFrame,
          injuries: pd.DataFrame, players: pd.DataFrame | None = None, spreads: dict | None = None,
          now: datetime | None = None, rules: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rules = rules if rules is not None else load_rules()
    pop = rules.get("population", {})
    players = players if players is not None else pd.DataFrame(columns=["gsis_id", "pfr_id", "display_name"])
    gsis_to_pfr = dict(zip(players["gsis_id"], players["pfr_id"])) if "pfr_id" in players else {}
    id_to_name = dict(zip(players["gsis_id"], players["display_name"])) if "display_name" in players else {}
    spreads = spreads or {}
    games = games[games["gameday"].notna()].copy()
    roster_counts: dict = {}
    units: dict = {}
    skipped = {"team_unclear": 0, "no_game": 0, "not_game_week": 0}
    for s in sorted(signals, key=lambda x: _ts(x["seen_at"])):
        st = (s.get("verify") or {}).get("roster", "unchecked")
        roster_counts[st] = roster_counts.get(st, 0) + 1
        if not s.get("game_week_relevant"):
            skipped["not_game_week"] += 1
            continue
        team = signal_team(s)
        if not team:
            skipped["team_unclear"] += 1
            continue
        g = assign_game(s, games)
        if g is None:
            skipped["no_game"] += 1
            continue
        who = (s.get("verify") or {}).get("gsis_id") or norm_name(s.get("player") or "")
        k = (g["game_id"], who, s.get("signal"))
        if k not in units:
            units[k] = (s, g, team, [])
        units[k][3].append(s)

    def eligible(x: dict) -> bool:  # news_rules.json population (would be allowed to act as an override)
        return bool(is_qb_signal(x) and x.get("game_week_relevant")
                    and float(x.get("certainty") or 0) >= pop.get("min_certainty", 0.8)
                    and (x.get("verify") or {}).get("roster") == pop.get("roster", "match"))
    rows = []
    for s, g, team, group in units.values():
        sig, qb = s.get("signal"), is_qb_signal(s)
        side = "home" if g["home_team"] == team else "away"
        seen = _ts(s["seen_at"])
        row = {"seen_at": s["seen_at"], "player": s.get("player"), "team": team, "signal": sig, "qb": qb,
               "game_id": g["game_id"], "roster": (s.get("verify") or {}).get("roster", "unchecked"),
               "certainty": s.get("certainty"),
               "lead_vs_report_h": round((report_time(g) - seen).total_seconds() / 3600, 1)}
        if qb:
            row["line_move"] = line_move(_series_for(spreads, g), seen, side == "home", now)
        finished = pd.notna(g.get("home_score")) and kickoff(g) < now
        gradable = (sig in POS or sig in QB_NEG) if qb else (sig in POS or sig in NEG)
        if not gradable:
            row["verdict"] = "ungradable"
        elif not finished:
            row["verdict"] = "pending"
        else:
            if qb:
                truth = started(s, g, side, id_to_name)
                row["truth"] = None if truth is None else ("started" if truth else "did not start")
                ok = None if truth is None else (truth == (sig in POS))
            else:
                truth = played(s, g, team, snaps, stats, gsis_to_pfr)
                row["truth"] = None if truth is None else ("played" if truth else "did not play")
                ok = None if truth is None else (truth == (sig in POS))
            row["verdict"] = "pending" if ok is None else ("correct" if ok else "wrong")
            has_rep, status = official_status(s, g, team, injuries)
            if has_rep:
                row["official_status"] = status
                row["report_agrees"] = report_agrees(sig, status)
        row["in_rule_population"] = any(eligible(x) for x in group)  # first seen may be the mis-tagged copy
        rows.append(row)
    graded = [r for r in rows if r["verdict"] in ("correct", "wrong")]
    by_sig: dict = {}
    for r in graded:
        b = by_sig.setdefault(r["signal"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += r["verdict"] == "correct"
    for b in by_sig.values():
        b["pct"] = _pct(b["correct"], b["n"])
    qbg = [r for r in graded if r["qb"]]
    popg = [r for r in qbg if r["in_rule_population"]]
    leads = [r["lead_vs_report_h"] for r in rows if r["verdict"] != "ungradable"]
    moves = [r["line_move"] for r in rows if r.get("line_move")]
    befores = [m["before"] for m in moves if m["before"] is not None]
    neg_moves = [r["line_move"]["after"] for r in rows if r.get("line_move") and r["signal"] in QB_NEG]
    agree = [r["report_agrees"] for r in graded if r.get("report_agrees") is not None]
    res = {
        "generated_at": now.isoformat(timespec="minutes"),
        "signals": len(signals), "units": len(rows), "skipped": skipped, "roster_check": roster_counts,
        "graded": len(graded), "pending": sum(r["verdict"] == "pending" for r in rows),
        "overall": {"n": len(graded), "correct": sum(r["verdict"] == "correct" for r in graded),
                    "pct": _pct(sum(r["verdict"] == "correct" for r in graded), len(graded))},
        "by_signal": dict(sorted(by_sig.items())),
        "qb": {"n": len(qbg), "correct": sum(r["verdict"] == "correct" for r in qbg),
               "pct": _pct(sum(r["verdict"] == "correct" for r in qbg), len(qbg)),
               "pending": sum(r["qb"] and r["verdict"] == "pending" for r in rows)},
        "rule_population": {"n": len(popg), "correct": sum(r["verdict"] == "correct" for r in popg),
                            "pct": _pct(sum(r["verdict"] == "correct" for r in popg), len(popg)),
                            "pending": sum(r["in_rule_population"] and r["verdict"] == "pending" for r in rows)},
        "official_report_agreement": {"n": len(agree), "agree": sum(agree), "pct": _pct(sum(agree), len(agree))},
        "lead_vs_report_hours": {"n": len(leads), "median": round(statistics.median(leads), 1) if leads else None,
                                 "share_before_report": _pct(sum(x > 0 for x in leads), len(leads))},
        "qb_line_move_6h": {"n": len(moves),
                            "avg_abs_after": round(statistics.mean(abs(m["after"]) for m in moves), 2) if moves else None,
                            "avg_abs_before": round(statistics.mean(abs(x) for x in befores), 2) if befores else None,
                            "avg_after_negative_signals": round(statistics.mean(neg_moves), 2) if neg_moves else None},
        "rows": sorted([r for r in rows if r["verdict"] != "ungradable"], key=lambda r: r["seen_at"], reverse=True),
    }
    res["promotion"] = promotion_check(res, rules)
    return res


def promotion_check(audit_result: dict, rules: dict | None = None) -> dict:
    """Pre-registered rule (news_rules.json): AI QB signals may feed QB availability only once enough audited
    starting-QB signals of the eligible kind (certainty, roster match, game-week relevant) were right.
    Returns {"eligible", "reasons", "n", "correct", "precision", "rules_version"}; never applies anything."""
    rules = rules if rules is not None else load_rules()
    p = rules["promotion"]
    pop = audit_result.get("rule_population") or {}
    n, c = int(pop.get("n") or 0), int(pop.get("correct") or 0)
    prec = c / n if n else None
    reasons = []
    if n < p["min_audited_qb_signals"]:
        reasons.append(f"{n} of {p['min_audited_qb_signals']} audited starting-QB signals so far")
    if prec is None or prec < p["min_precision"]:
        reasons.append(f"precision {'n/a' if prec is None else f'{prec:.0%}'} (needs >= {p['min_precision']:.0%})")
    eligible = not reasons
    if eligible:
        reasons.append(f"{c}/{n} correct ({prec:.0%}) >= {p['min_precision']:.0%} on >= {p['min_audited_qb_signals']}")
    return {"eligible": eligible, "reasons": reasons, "n": n, "correct": c,
            "precision": None if prec is None else round(prec, 4), "rules_version": rules.get("version"),
            "min_n": p["min_audited_qb_signals"], "min_precision": p["min_precision"]}


def run(history_dir: Path, season: int, now: datetime | None = None) -> dict:
    """Load the data (cached nflverse files; the gate already refreshed them) and write history/news_audit.json."""
    from . import data, player_stats as PS
    signals = load_signals(history_dir / "news_llm.jsonl")
    games = data.load_schedules(refresh=False)
    games = games[games["season"] == season]
    try:
        snaps = data.load_snaps([season])
    except Exception:
        snaps = pd.DataFrame(columns=["game_id", "team", "player", "pfr_player_id"])
    stats = PS.load_week_stats(season, refresh=False)
    try:
        injuries = data.load_injuries([season])
    except Exception:
        injuries = pd.DataFrame(columns=["season", "week", "team", "gsis_id", "full_name", "report_status"])
    try:
        players = data.load_players()
    except Exception:
        players = None
    res = audit(signals, games, snaps, stats, injuries, players, spread_series(history_dir), now)
    (history_dir / "news_audit.json").write_text(json.dumps(res, indent=1, default=str))
    return res
