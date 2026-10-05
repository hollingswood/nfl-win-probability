"""Weekly player stats (nflverse) and prop-name matching, for grading player-prop paper bets.

Stats: nflverse-data release `stats_player` (stats_player_week_<season>.parquet; the older `player_stats`
release is tried as a fallback), cached in data/raw. Snap counts (data.load_snaps) decide whether a player who
has no stat line played (0 receptions) or did not take an offensive snap (void, as in the research settlement).
Name matching is a minimal copy of scripts/research/props_backtest.py (norm_name / first_ok / fuzzy among the
game's players); production code never imports from scripts/research.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from . import data

SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
ALIAS = {"hollywood brown": "marquise brown", "amon ra stbrown": "amon ra st brown"}
SOURCES = (("stats_player", "stats_player_week"), ("player_stats", "player_stats"))


def norm_name(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " ").replace(".", "").replace("'", ""))
    s = " ".join(t for t in s.split() if t not in SUFFIX)
    return ALIAS.get(s, s)


def first_ok(a: str, b: str) -> bool:
    """First names must be close (Eli/Elijah, Drew/Andrew); rejects Deonte vs Damien Harris."""
    fa, fb = (a.split() or [""])[0], (b.split() or [""])[0]
    return fa in fb or fb in fa or SequenceMatcher(None, fa, fb).ratio() >= 0.6


def match(name: str, cands: dict) -> object | None:
    """cands: id -> set of normalized names (players of ONE game). Exact, then fuzzy (>= 0.85 with a 0.05 margin
    and close first names), then first initial + last name. None if ambiguous / not found."""
    n = norm_name(name)
    hit = [i for i, ns in cands.items() if n in ns]
    if len(hit) == 1:
        return hit[0]
    sc = sorted(((max((SequenceMatcher(None, n, x).ratio() for x in ns if first_ok(n, x)), default=0), i)
                 for i, ns in cands.items()), key=lambda t: -t[0])
    if sc and sc[0][0] >= 0.85 and (len(sc) == 1 or sc[1][0] < sc[0][0] - 0.05):
        return sc[0][1]
    toks = n.split()
    if toks and len(toks[0]) == 1:
        lm = [i for i, ns in cands.items() if any(x.split()[-1:] == toks[-1:] and x[:1] == n[:1] for x in ns)]
        if len(lm) == 1:
            return lm[0]
    return None


def load_week_stats(season: int, refresh: bool = True, raw_dir=data.RAW_DIR) -> pd.DataFrame:
    """Weekly player stats for a season (empty frame if no release file is available yet)."""
    for tag, stem in SOURCES:
        try:
            path = data._download(f"{data.BASE}/{tag}/{stem}_{season}.parquet", raw_dir / f"{stem}_{season}.parquet",
                                  refresh)
            d = pd.read_parquet(path)
            if "receptions" in d and "game_id" in d:
                return d
        except Exception:
            continue
    return pd.DataFrame(columns=["game_id", "player_id", "player_display_name", "player_name", "receptions"])


def load_snaps(season: int) -> pd.DataFrame:
    try:
        return data.load_snaps([season])
    except Exception:
        return pd.DataFrame(columns=["game_id", "player", "offense_snaps"])


def receptions(game_id: str, player: str, stats: pd.DataFrame, snaps: pd.DataFrame) -> tuple[str, object]:
    """('graded', receptions) | ('void', reason) | ('open', reason) for a player-prop bet on this game."""
    return stat_value(game_id, player, stats, snaps, "receptions")


def stat_value(game_id: str, player: str, stats: pd.DataFrame, snaps: pd.DataFrame, col: str) -> tuple[str, object]:
    """Same as receptions() for any weekly stat column (rushing_yards, receiving_yards, ...): a player with no stat
    line who took an offensive snap counts as 0; no offensive snap = void."""
    st = stats[stats["game_id"] == game_id] if len(stats) else stats
    if len(st):
        cands = {}
        for i, r in st.iterrows():
            ns = {norm_name(r.get("player_display_name") or "")}
            if isinstance(r.get("player_name"), str):
                ns.add(norm_name(r["player_name"]))
            cands[i] = {x for x in ns if x}
        hit = match(player, cands)
        if hit is not None:
            v = st.loc[hit, col] if col in st.columns else None
            return "graded", float(0 if v is None or pd.isna(v) else v)
    sn = snaps[snaps["game_id"] == game_id] if len(snaps) else snaps
    if not len(sn):
        return "open", "stats/snap counts not published yet" if not len(st) else "no stat line; snap counts not published yet"
    cands = {i: {norm_name(r)} for i, r in zip(sn.index, sn["player"])}
    hit = match(player, cands)
    if hit is not None and float(sn.loc[hit, "offense_snaps"] or 0) >= 1:
        return "graded", 0  # played on offense without a catch (no stat line)
    if not len(st):
        return "open", "stats not published yet"
    return "void", "no offensive snap (did not play)"
