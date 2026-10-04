"""What is being first on college QB news worth? (descriptive study, not a bet rule; 2021-2026)

Starter = the team's passer with the most attempts (CFBD per-game passing stats). A QB change = a different starter
than the team's previous game this season, where the previous starter had >= 10 attempts. The line move is Pinnacle's
spread-implied expected margin from the first snapshot 96+ h before kickoff (Sun/Mon) to its last pre-kickoff quote,
signed so that + = the market moved AGAINST the team that changed QB. Compared with all other games (|move| baseline).
"""
import glob
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS  # noqa: E402
import opener_screen as OS  # noqa: E402

RAW = PS.ROOT / "data" / "cfb" / "raw"


def starters() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RAW / "games_players_passing_*.json.gz")):
        season = int(re.findall(r"passing_(\d{4})_", f)[0])
        for g in json.load(gzip.open(f)):
            for t in g.get("teams", []):
                att = {}
                for c in t.get("categories", []):
                    for ty in c.get("types", []):
                        if ty["name"] == "C/ATT":
                            for a in ty["athletes"]:
                                try:
                                    att[(a["id"], a["name"])] = int(str(a["stat"]).split("/")[1])
                                except Exception:
                                    pass
                if att:
                    (pid, name), n = max(att.items(), key=lambda kv: kv[1])
                    rows.append({"game_id": int(g["id"]), "season": season, "team": t["team"], "qb_id": pid, "qb": name, "att": n})
    return pd.DataFrame(rows)


def main():
    S = starters()
    G = pd.concat([PS.D.games([s]) for s in range(2021, 2027)])[["game_id", "start", "home", "away"]]
    S = S.merge(G, on="game_id").sort_values("start")
    S["prev_qb"] = S.groupby(["season", "team"]).qb_id.shift(1)
    S["prev_att"] = S.groupby(["season", "team"]).att.shift(1)
    S["change"] = S.prev_qb.notna() & (S.qb_id != S.prev_qb) & (S.prev_att >= 10)
    ch = S[S.change].copy()
    ch["is_home"] = ch.team == ch.home
    O = PS.load("cfb")
    E = OS.event_map(O)
    pin = pd.read_parquet(PS.OUT / "pin_mus.parquet").dropna(subset=["mu_m"])
    pin = pin.join(E[["game_id"]], on="event_id", how="inner")
    pin["h"] = (pin.ko - pin.t).dt.total_seconds() / 3600
    early = pin[pin.h >= 96].sort_values("t").groupby("game_id").mu_m.first()
    late = pin[pin.h > 0].sort_values("t").groupby("game_id").mu_m.last()
    mv = (late - early).dropna().rename("home_move")          # + = moved toward home
    base = mv.abs()
    ch = ch.join(mv, on="game_id", how="inner")
    ch["move_against"] = np.where(ch.is_home, -ch.home_move, ch.home_move)
    games_changed = set(ch.game_id)
    other = base[~base.index.isin(games_changed)]
    out = {"qb_change_games": int(len(ch)),
           "mean_move_against_changed_team_pts": round(float(ch.move_against.mean()), 2),
           "median_move_against": round(float(ch.move_against.median()), 2),
           "share_moved_against_1pt_plus": round(float((ch.move_against >= 1).mean()), 3),
           "share_moved_against_3pt_plus": round(float((ch.move_against >= 3).mean()), 3),
           "mean_abs_move_changed": round(float(ch.move_against.abs().mean()), 2),
           "mean_abs_move_other_games": round(float(other.mean()), 2), "n_other_games": int(len(other)),
           "by_season": {int(s): {"n": int(len(g)), "mean_against": round(float(g.move_against.mean()), 2)} for s, g in ch.groupby("season")}}
    (PS.OUT / "qb_timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
