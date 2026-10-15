"""Per-game warnings for the dashboard and phone alerts (display only: never qualifies, vetoes or sizes a paper bet).

* qb_out  - a strict AI call that a game's projected starting QB is out (certainty >= 0.9, full name on the team's
            roster, published after the team's previous game). The model only learns this from the official injury
            report / overrides.json, so its number, and grades that lean on it, can be stale.
* split   - the US books' consensus spread and the sharp books' (Pinnacle/LowVig) spread disagree by 1.5+ points.
            Usually one side hasn't caught up with news yet; prices "beating the sharp line" can then be an illusion
            (2026-10-15: Jalen Hurts ruled out, US books and exchanges moved CAR -1.5 -> -3.5 within minutes while
            Pinnacle still showed CAR -1).
Writes game['heads_up'] = [ {kind, text, ...} ].
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_SIGNALS = {"out", "doubtful", "injured_reserve", "benched", "suspended"}
SPLIT_PTS = 1.5


def _t(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _signals(history_dir: Path, now: datetime, days: int = 8) -> list[dict]:
    p = history_dir / "news_llm.jsonl"
    if not p.exists():
        return []
    cut, out = now - timedelta(days=days), []
    for line in p.read_text().splitlines():
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = _t(s.get("published")) or _t(s.get("seen_at"))
        if t and cut <= t <= now + timedelta(hours=1):
            out.append(s)
    return out


def _same(a: str | None, b: str | None) -> bool:
    from .qb_news import same_player
    return bool(a and b and same_player(a, b))


def qb_out(g: dict, signals: list[dict]) -> list[dict]:
    out = []
    for side in ("home", "away"):
        team, qb, since = g.get(f"{side}_team"), g.get(f"{side}_qb"), _t(g.get(f"{side}_prev_end"))
        best = None
        for s in signals:
            if (s.get("team_verified") or s.get("team")) != team or not s.get("is_starting_qb_news"):
                continue
            if (s.get("verify") or {}).get("roster") != "match" or float(s.get("certainty") or 0) < 0.9:
                continue
            if s.get("game_week_relevant") is False or not _same(s.get("player"), qb):
                continue
            t = _t(s.get("published")) or _t(s.get("seen_at"))
            if since and t < since:
                continue
            if best is None or t > best[0]:
                best = (t, s)
        if best and best[1].get("signal") in OUT_SIGNALS:   # the latest strict call about the starter says he's out
            s = best[1]
            out.append({"kind": "qb_out", "team": team, "player": s.get("player"), "signal": s.get("signal"),
                        "source": s.get("outlet") or {"pft": "PFT", "yahoo": "Yahoo", "cbs": "CBS", "espn": "ESPN"}.get(s.get("source"), "Google News" if str(s.get("source", "")).startswith("google") else s.get("source")), "link": s.get("link"), "seen_at": s.get("seen_at"),
                        "text": f"{s.get('player')} ({team} QB) reported {str(s.get('signal')).replace('_', ' ')}: the model and "
                                f"grades may not reflect it until the injury report / depth chart catches up"})
    return out


def split(g: dict, pts: float = SPLIT_PTS) -> list[dict]:
    lo = (g.get("context") or {}).get("live_odds") or {}
    ss, cm = lo.get("sharp_spreads") or [], lo.get("consensus_home_margin")
    if not ss or cm is None:
        return []
    sm = -statistics.median(x[0] for x in ss)
    if abs(cm - sm) < pts:
        return []
    fmt = lambda m: (f"{g['home_team']} −{abs(m):g}" if m > 0 else f"{g['away_team']} −{abs(m):g}" if m < 0 else "pick'em")
    return [{"kind": "split", "consensus_margin": cm, "sharp_margin": round(sm, 2),
             "text": f"Books disagree: US books {fmt(cm)}, sharp books {fmt(sm)}. One side hasn't caught up (often news); "
                     f"prices that beat the sharp line here may be an illusion"}]


def annotate(pred: dict, history_dir: Path = ROOT / "history", now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    sig = _signals(history_dir, now)
    for g in pred.get("upcoming", []):
        try:
            g["heads_up"] = qb_out(g, sig) + split(g)
        except Exception as e:  # display only
            print("heads_up failed for", g.get("game_id"), e)
            g["heads_up"] = []
    return pred
