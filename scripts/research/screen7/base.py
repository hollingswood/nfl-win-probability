"""Walk-forward model predictions 2014-2025 for the 7-strategy screen (each season predicted by a model
trained only on earlier seasons). -> data/screen_base.parquet"""
import datetime as dt
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from nflpred import model as M
from nflpred.pipeline import build

OUT = ROOT / "data" / "screen_base.parquet"

def main():
    df = build(refresh=False, today=dt.date(2026, 9, 1), live_news=False)
    parts = []
    for s in range(2014, 2026):
        te = df[(df.season == s) & df.home_win.notna()].copy()
        mdl = M.fit(df, before_season=s)
        te["mu_model"] = mdl.predict_margin(te); te["p_model"] = M.predict(mdl, te)
        parts.append(te)
    out = pd.concat(parts).reset_index(drop=True)
    keep = [c for c in out.columns if out[c].dtype != object or c in ("game_id", "gameday", "home_team", "away_team",
            "game_type", "weekday", "gametime", "home_qb_id", "away_qb_id", "roof", "home_coach", "away_coach")]
    out[keep].to_parquet(OUT)
    print(out.shape, out.season.min(), out.season.max())

if __name__ == "__main__":
    main()
