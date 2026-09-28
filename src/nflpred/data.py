"""Download and load nflverse data (schedules + play-by-play).

Data is pulled from nflverse-data GitHub releases and cached as parquet in data/raw.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

PBP_COLS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type", "pass", "rush",
    "qb_kneel", "qb_spike", "epa", "success", "interception", "fumble_lost",
    "fumbled_1_team", "qb_dropback", "qb_epa", "id", "cpoe", "wp",
]


def _download(url: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def load_schedules(refresh: bool = False, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = _download(f"{BASE}/schedules/games.parquet", raw_dir / "games.parquet", refresh)
    g = pd.read_parquet(path)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def load_pbp(seasons, refresh_current: int | None = None, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load play-by-play for the given seasons. `refresh_current` forces a re-download of that season."""
    frames = []
    for s in seasons:
        path = _download(
            f"{BASE}/pbp/play_by_play_{s}.parquet",
            raw_dir / f"pbp_{s}.parquet",
            refresh=(s == refresh_current),
        )
        frames.append(pd.read_parquet(path, columns=PBP_COLS))
    return pd.concat(frames, ignore_index=True)


def _release(tag: str, name: str, seasons=None, refresh_current=None, raw_dir: Path = RAW_DIR):
    if seasons is None:
        return pd.read_parquet(_download(f"{BASE}/{tag}/{name}.parquet", raw_dir / f"{name}.parquet",
                                         refresh_current is not None))
    frames = []
    for s in seasons:
        try:
            path = _download(f"{BASE}/{tag}/{name}_{s}.parquet", raw_dir / f"{name}_{s}.parquet",
                             refresh=(s == refresh_current))
            frames.append(pd.read_parquet(path))
        except Exception:  # e.g. current season's file not published yet
            continue
    return pd.concat(frames, ignore_index=True)


def load_injuries(seasons, refresh_current=None):
    return _release("injuries", "injuries", seasons, refresh_current)


def load_snaps(seasons, refresh_current=None):
    return _release("snap_counts", "snap_counts", seasons, refresh_current)


def load_players(refresh=False):
    return _release("players", "players", None, 0 if refresh else None)
