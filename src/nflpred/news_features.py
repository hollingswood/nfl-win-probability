"""Per-game news features from the AI news reader (news_llm.py), using only items published before a cutoff.

Inputs (history/): news_outlook.jsonl (C: per-team sentiment / point impact / novelty, prompt v3+),
news_context.jsonl (B: team context with helps/hurts), news_llm.jsonl (A: player availability).
For a team and a cutoff time, items published in the 7 days before the cutoff (and after the team's previous game,
when known) count once each (same team + link):

  impact_new   sum of impact_pts x certainty for items marked "new"   (what the market may not have priced yet)
  impact_all   sum of impact_pts x certainty for every item
  sentiment    certainty-weighted mean sentiment (-2..2)
  ctx_net      context items: helps - hurts (certainty-weighted)
  avail_out    starters reported out/doubtful/IR in that window (count, latest status per player)

`game_features(game, cutoff)` gives home, away and home-minus-away values. Display, the blind candidates
(blind_candidates.py) and the news tests use these; no paper track bets on them.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
OUT_SIGNALS = {"out", "doubtful", "injured_reserve", "suspended", "released_or_traded"}
_CACHE: dict = {}


def _t(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _load(name: str, hist: Path) -> list[dict]:
    p = hist / name
    key = (str(p), p.stat().st_mtime if p.exists() else 0)
    if key not in _CACHE:
        rows = []
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                r["_t"] = _t(r.get("published")) or _t(r.get("seen_at"))
                r["_seen"] = _t(r.get("seen_at")) or r["_t"]
                rows.append(r)
        _CACHE.clear()
        _CACHE[key] = rows
    return _CACHE[key]


def team_features(team: str, cutoff: datetime, since: datetime | None = None, hist: Path = HIST,
                  days: int = 7) -> dict:
    lo = max(cutoff - timedelta(days=days), since) if since else cutoff - timedelta(days=days)
    # an item counts only if we had SEEN it by the cutoff (no hindsight) and it was published in the window
    ok = lambda r: r.get("team") == team and r["_t"] and r["_seen"] and lo <= r["_t"] <= cutoff and r["_seen"] <= cutoff \
        and r.get("game_week_relevant", True) is not False
    out = {"impact_new": 0.0, "impact_all": 0.0, "sentiment": 0.0, "ctx_net": 0.0, "avail_out": 0, "n_items": 0}
    seen, ws, sw = set(), 0.0, 0.0
    for r in _load("news_outlook.jsonl", hist):
        if not ok(r) or (r.get("link"), r.get("team")) in seen:
            continue
        seen.add((r.get("link"), r.get("team")))
        c = float(r.get("certainty") or 0.5)
        imp = float(r.get("impact_pts") or 0) * c
        out["impact_all"] += imp
        if r.get("novelty") == "new":
            out["impact_new"] += imp
        sw += float(r.get("sentiment") or 0) * c
        ws += c
        out["n_items"] += 1
    out["sentiment"] = round(sw / ws, 3) if ws else 0.0
    cseen = set()
    for r in _load("news_context.jsonl", hist):
        if not ok(r) or (r.get("link"), r.get("category")) in cseen:
            continue
        cseen.add((r.get("link"), r.get("category")))
        c = float(r.get("certainty") or 0.5)
        out["ctx_net"] += c if r.get("direction") == "helps" else -c if r.get("direction") == "hurts" else 0.0
    status: dict = {}
    for r in _load("news_llm.jsonl", hist):
        if ok(r) and r.get("player"):
            k = r["player"].lower()
            if k not in status or r["_t"] >= status[k][0]:
                status[k] = (r["_t"], r.get("signal"))
    out["avail_out"] = sum(1 for _, sig in status.values() if sig in OUT_SIGNALS)
    out["impact_new"], out["impact_all"], out["ctx_net"] = (round(out[k], 3) for k in ("impact_new", "impact_all", "ctx_net"))
    return out


def game_features(home: str, away: str, cutoff: datetime, home_since=None, away_since=None, hist: Path = HIST) -> dict:
    h = team_features(home, cutoff, _t(home_since), hist)
    a = team_features(away, cutoff, _t(away_since), hist)
    f = {f"home_{k}": v for k, v in h.items()}
    f.update({f"away_{k}": v for k, v in a.items()})
    for k in h:
        f[f"diff_{k}"] = round(h[k] - a[k], 3)
    return f
