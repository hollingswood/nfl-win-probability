"""Tuesday-information model margins for 2020-2025 (fixes look-ahead in S3):
injury-report features zeroed (the final report is Friday) and a flag for games where either team's
starting QB differs from its previous game's starter (on Tuesday the starter would not be known for sure).
-> data/screen_tuesday.parquet"""
import datetime as dt, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from nflpred import model as M
from nflpred.pipeline import build

def main():
    df = build(refresh=False, today=dt.date(2026, 9, 1), live_news=False)
    long = pd.concat([df[["season", "gameday", "home_team", "home_qb_id"]].set_axis(["season", "gameday", "team", "qb"], axis=1),
                      df[["season", "gameday", "away_team", "away_qb_id"]].set_axis(["season", "gameday", "team", "qb"], axis=1)])
    long = long.sort_values("gameday"); long["prev_qb"] = long.groupby("team").qb.shift(1)
    prev = long.set_index(["gameday", "team"]).prev_qb
    parts = []
    for s in range(2020, 2026):
        te = df[(df.season == s) & df.home_win.notna()].copy()
        mdl = M.fit(df, before_season=s)
        te["mu_full"] = mdl.predict_margin(te)
        t2 = te.copy(); t2["inj_off_diff"] = 0.0; t2["inj_def_diff"] = 0.0
        te["mu_tue"] = mdl.predict_margin(t2)
        te["qb_same"] = [(prev.get((g, h)) == hq) and (prev.get((g, a)) == aq)
                         for g, h, a, hq, aq in zip(te.gameday, te.home_team, te.away_team, te.home_qb_id, te.away_qb_id)]
        parts.append(te[["game_id", "season", "week", "mu_full", "mu_tue", "qb_same"]])
    out = pd.concat(parts); out.to_parquet(ROOT / "data/screen_tuesday.parquet")
    print(out.shape, out.qb_same.mean(), (out.mu_full - out.mu_tue).abs().mean())

if __name__ == "__main__":
    main()
