"""Starting-QB availability for upcoming games.

Historical games use the QB who actually started. For games not yet played, the listed starter
may be hurt. If he is Questionable/Doubtful, the prediction becomes a mix:

    P(win) = P(plays) * P(win | he starts) + (1 - P(plays)) * P(win | replacement starts)

P(plays) comes from how often starting QBs with the same report status and final practice
participation actually started, 2012-2025 (see README). Overrides in `overrides.json` let you
apply late news the injury report hasn't caught up with.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# (report_status, final practice) -> historical probability the QB started
PLAY_PROB = {
    ("Questionable", "Full"): 0.86, ("Questionable", "Limited"): 0.53, ("Questionable", "Did Not"): 0.42,
    ("Questionable", None): 0.60, ("Doubtful", None): 0.05, ("Out", None): 0.0,
}
REPLACEMENT_RATING = -0.10  # EPA/dropback for an emergency starter (inexperienced starters averaged -0.06)
ROOT = Path(__file__).resolve().parents[2]


def play_prob(status, practice) -> float:
    if status not in ("Questionable", "Doubtful", "Out"):
        return 1.0
    return PLAY_PROB.get((status, practice), PLAY_PROB.get((status, None), 1.0))


def load_overrides(path: Path = ROOT / "overrides.json") -> dict:
    """{"<game_id>": {"home_qb_play_prob": 0.4, "note": "..."}}"""
    return json.loads(path.read_text()) if path.exists() else {}


def availability(games: pd.DataFrame, overrides: dict | None = None) -> pd.DataFrame:
    ov = overrides or {}
    out = pd.DataFrame(index=games.index)
    for side in ("home", "away"):
        pp = [play_prob(s, p) for s, p in zip(games.get(f"{side}_qb_status", [None] * len(games)),
                                              games.get(f"{side}_qb_practice", [None] * len(games)))]
        out[f"{side}_qb_play_prob"] = pp
        for i, gid in zip(games.index, games["game_id"]):
            v = ov.get(gid, {}).get(f"{side}_qb_play_prob")
            if v is not None:
                out.at[i, f"{side}_qb_play_prob"] = float(v)
    return out


def with_replacement(games: pd.DataFrame, side: str) -> pd.DataFrame:
    """Copy of games with `side`'s starter swapped for a replacement-level QB."""
    g = games.copy()
    delta = REPLACEMENT_RATING - g[f"{side}_qb_rating"]
    sign = 1 if side == "home" else -1
    g["qb_diff"] = g["qb_diff"] + sign * delta
    g["qb_change_diff"] = g["qb_change_diff"] + sign * delta
    g["fw_qb_diff"] = g["final_week"] * g["qb_diff"]
    return g


def blended_prob(model, games: pd.DataFrame, avail: pd.DataFrame, predict) -> np.ndarray:
    """Mix over the four starter scenarios (each side plays / sits)."""
    ph, pa = avail["home_qb_play_prob"].values, avail["away_qb_play_prob"].values
    both = predict(model, games)
    h_out = predict(model, with_replacement(games, "home"))
    a_out = predict(model, with_replacement(games, "away"))
    both_out = predict(model, with_replacement(with_replacement(games, "home"), "away"))
    return ph * pa * both + (1 - ph) * pa * h_out + ph * (1 - pa) * a_out + (1 - ph) * (1 - pa) * both_out
