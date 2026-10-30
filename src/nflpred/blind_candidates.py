"""Live hooks for registered blind candidates (blind.py): name -> fn(predictions) -> [{game_id, kickoff, home_win_prob}].
Only registered names may appear here (test_blind enforces it); each logs before kickoff on every full run.

News candidates (output/research/round6/news_models_prereg.json, registered 2026-10-30): the production margin
plus a fixed multiple of the AI news reader's per-game news scores (news_features.py)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def _phi_inv(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def _sigma(margin: float, p: float) -> float:
    z = _phi_inv(min(max(p, 1e-4), 1 - 1e-4))
    s = margin / z if abs(z) > 0.05 and abs(margin) > 0.5 else 13.5
    return s if 5 < s < 25 else 13.5


_RUN: dict = {}


def _games(pred: dict, now: datetime | None = None):
    """Features for this run's upcoming games, computed and logged once per run (all candidates share them)."""
    if _RUN.get("id") == id(pred):
        return _RUN["rows"]
    from . import news_features as NF
    from .pipeline import _kickoff
    now = now or datetime.now(timezone.utc)
    rows, log = [], []
    for g in pred.get("upcoming", []):
        if g.get("model_home_margin") is None or g.get("home_win_prob") is None:
            continue
        ko = _kickoff(g)
        f = NF.game_features(g["home_team"], g["away_team"], now, g.get("home_prev_end"), g.get("away_prev_end"))
        rows.append((g, ko, f))
        log.append({"logged_at": now.isoformat(timespec="minutes"), "game_id": g["game_id"], "kickoff": ko, **f})
    p = ROOT / "history" / "news_game_features.jsonl"
    with p.open("a") as fh:
        for r in log:
            fh.write(json.dumps(r) + "\n")
    _RUN.update(id=id(pred), rows=rows)
    return rows


def _shift(pred: dict, fn) -> list[dict]:
    out = []
    for g, ko, f in _games(pred):
        m, p = float(g["model_home_margin"]), float(g["home_win_prob"])
        out.append({"game_id": g["game_id"], "kickoff": ko, "home_win_prob": round(_phi((m + fn(f)) / _sigma(m, p)), 4)})
    return out


LIVE = {
    "news_new_v1": lambda pred: _shift(pred, lambda f: 1.0 * f["diff_impact_new"]),
    "news_all_half_v1": lambda pred: _shift(pred, lambda f: 0.5 * f["diff_impact_all"]),
    "news_sent_v1": lambda pred: _shift(pred, lambda f: 1.0 * f["diff_sentiment"] + 0.5 * f["diff_ctx_net"]),
}
