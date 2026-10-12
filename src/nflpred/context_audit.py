"""Audit of AI-read team context news (context_rules.json v1, pre-registered 2026-10-12). Logging only.

For each context item with a direction (helps / hurts that team) about the team's next game: did the consensus
spread move toward the implied side after we saw it (line move, points), and did that side cover the closing spread?
Per category, against the pre-registered candidate bar. Output: history/context_audit.json.
"""
from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import exchanges as X
from .odds import TEAM_ABBR
from .qb_news import consensus_home_point, _t

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "context_rules.json"


def _kicks(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule.copy()
    s["kick"] = pd.to_datetime(s.gameday.astype(str) + " " + s.gametime.fillna("13:00").astype(str)).dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC")
    return s


def _point_at(files, cache, home, away, t, before_kick=None):
    """Consensus home point in the last snapshot at or before t."""
    for f in reversed(files):
        ts = X._snap_time(f)
        if ts > t or (before_kick is not None and ts >= before_kick):
            continue
        if f not in cache:
            cache[f] = X.load_events(f)
        for ev in cache[f]:
            if (TEAM_ABBR.get(ev.get("home_team")), TEAM_ABBR.get(ev.get("away_team"))) == (home, away):
                p = consensus_home_point(ev)
                if p is not None:
                    return p
    return None


def _p_binom(w: int, n: int, p0: float) -> float:
    if not n:
        return 1.0
    z = (w - n * p0) / math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(z / math.sqrt(2))


def run(history_dir: Path = ROOT / "history", schedule: pd.DataFrame | None = None, now: datetime | None = None) -> dict:
    r = json.loads(RULES.read_text())
    now = now or datetime.now(timezone.utc)
    if schedule is None:
        from . import data
        schedule = data.load_schedules()
    S = _kicks(schedule[schedule.season >= 2026])
    p = history_dir / "news_context.jsonl"
    items = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    files = X.snapshot_files(history_dir); cache: dict = {}
    seen, rows = set(), []
    it = r["items"]
    for c in sorted(items, key=lambda x: x.get("seen_at", "")):
        if c.get("direction") not in it["directions"] or (c.get("certainty") or 0) < it["min_certainty"] or c.get("game_week_relevant") is False:
            continue
        t = min(x for x in (_t(c.get("published")), _t(c.get("seen_at"))) if x)
        team = c.get("team")
        g = S[((S.home_team == team) | (S.away_team == team)) & (S.kick > pd.Timestamp(t))].sort_values("kick")
        if g.empty:
            continue
        g = g.iloc[0]
        key = (team, c.get("category"), g.game_id)
        if key in seen:
            continue
        seen.add(key)
        side = "home" if g.home_team == team else "away"
        bet_side = side if c["direction"] == "helps" else ("away" if side == "home" else "home")
        row = {"game_id": g.game_id, "team": team, "category": c.get("category"), "direction": c["direction"],
               "seen": t.isoformat(timespec="minutes"), "summary": c.get("summary"), "implied_side": bet_side}
        kick = g.kick.to_pydatetime()
        p0 = _point_at(files, cache, g.home_team, g.away_team, t)
        if kick < now:
            pc = _point_at(files, cache, g.home_team, g.away_team, kick, before_kick=kick)
            if p0 is not None and pc is not None:
                row["line_move_toward"] = round((p0 - pc) if bet_side == "home" else (pc - p0), 2)
            if pc is not None and pd.notna(g.home_score):
                margin = float(g.home_score) - float(g.away_score)
                d = (margin + pc) if bet_side == "home" else (-margin - pc)
                row["result_vs_close"] = "push" if d == 0 else ("win" if d > 0 else "loss")
        rows.append(row)
    cats = {}
    bar = r["candidate_bar"]
    for cat in sorted({x["category"] for x in rows}):
        x = [y for y in rows if y["category"] == cat]
        mv = [y["line_move_toward"] for y in x if y.get("line_move_toward") is not None]
        w = sum(y.get("result_vs_close") == "win" for y in x); l = sum(y.get("result_vs_close") == "loss" for y in x)
        m = statistics.mean(mv) if mv else None
        sd = statistics.stdev(mv) if len(mv) > 1 else 0
        mp = 0.5 * math.erfc((m / (sd / math.sqrt(len(mv)))) / math.sqrt(2)) if mv and len(mv) > 1 and sd > 0 else 1.0
        cr = w / (w + l) if w + l else None
        cp = _p_binom(w, w + l, 0.5)
        n = w + l
        cand = cat != "other" and n >= bar["min_items"] and (
            (cr is not None and cr > bar["cover_rate_vs_close_above"] and cp < bar["one_sided_p_below"]) or
            (m is not None and m >= bar["or_avg_line_move_toward_pts_at_least"] and mp < bar["move_p_below"]))
        cats[cat] = {"items": len(x), "graded": n, "cover_rate_vs_close": round(cr, 3) if cr is not None else None, "cover_p": round(cp, 4),
                     "with_line_move": len(mv), "avg_line_move_toward": round(m, 2) if m is not None else None, "move_p": round(mp, 4),
                     "candidate": bool(cand)}
    out = {"generated_at": now.isoformat(timespec="minutes"), "rules_version": r["version"], "items": len(rows),
           "categories": cats, "recent": rows[-30:]}
    (history_dir / "context_audit.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    res = run()
    print(json.dumps({k: res[k] for k in ("items", "categories")}, indent=1))
