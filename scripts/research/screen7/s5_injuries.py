"""S5 (injury pricing) — pre-declared 2026-10-03 before results.
Non-QB injury load per team-game: sum over players listed Out/Doubtful on the FINAL report of
(position weight x the player's average snap share over his team's previous 4 games; previous season if none).
Weights: RB .5, WR .8, TE .5, OL .6, DL .6, LB .5, DB .6 (QBs excluded: QB news is priced separately).
diff = load_home - load_away; "big" = |diff| >= 90th percentile over all games.
 (a) outcome test 2013-2025: bet AGAINST the more-injured team at the closing number/price (nflverse) in big games.
 (b) timing test 2022-2025 (hourly odds): consensus home spread 1 h before the final report, 2 h after, and at the close;
     does the line keep moving against the injured team after the report (underreaction -> CLV for a Friday bet)?
Pass: one-sided p < 0.007 on ROI (a) or on mean post-report move in our favour (b), positive in most seasons."""
import glob, json, math, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
W = {"RB": .5, "FB": .3, "WR": .8, "TE": .5, "T": .6, "G": .6, "C": .6, "OL": .6, "OT": .6, "OG": .6,
     "DE": .6, "DT": .6, "NT": .6, "DL": .6, "LB": .5, "ILB": .5, "OLB": .5, "MLB": .5,
     "CB": .6, "S": .6, "FS": .6, "SS": .6, "DB": .6}
FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower().replace(" jr", "").replace(" sr", "").replace(" iii", "").replace(" ii", ""))

def dec(a):
    a = pd.to_numeric(a, errors="coerce").fillna(-110).to_numpy(float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))

def loads():
    rows = []
    for s in range(2013, 2026):
        sn = pd.read_parquet(ROOT / f"data/raw/snap_counts_{s}.parquet")
        sn = sn[sn.game_type == "REG"] if "game_type" in sn else sn
        sn["pct"] = sn[["offense_pct", "defense_pct"]].max(axis=1)
        sn["key"] = sn.player.map(norm); sn["team"] = sn.team.replace(FIX)
        prev = pd.read_parquet(ROOT / f"data/raw/snap_counts_{s-1}.parquet") if (ROOT / f"data/raw/snap_counts_{s-1}.parquet").exists() else None
        prev_avg = {}
        if prev is not None:
            prev["pct"] = prev[["offense_pct", "defense_pct"]].max(axis=1); prev["key"] = prev.player.map(norm)
            prev_avg = prev.groupby("key").pct.mean().to_dict()
        inj = pd.read_parquet(ROOT / f"data/raw/injuries_{s}.parquet")
        inj = inj[(inj.game_type == "REG") & inj.report_status.isin(["Out", "Doubtful"]) & (inj.position != "QB")].copy()
        inj["team"] = inj.team.replace(FIX); inj["key"] = inj.full_name.map(norm)
        raw = pd.read_parquet(ROOT / f"data/raw/injuries_{s}.parquet").assign(team=lambda d: d.team.replace(FIX))
        rep = raw.groupby(["team", "week"]).date_modified.max() if "date_modified" in raw else pd.Series(dtype=object)
        for (team, wk), d in inj.groupby(["team", "week"]):
            hist = sn[(sn.team == team) & (sn.week < wk) & (sn.week >= wk - 4)]
            avg = hist.groupby("key").pct.mean().to_dict()
            load = 0.0
            for r in d.itertuples():
                p = avg.get(r.key, prev_avg.get(r.key, 0.0))
                load += W.get(r.position, 0.4) * float(p)
            rows.append({"season": s, "week": wk, "team": team, "load": load, "report_ts": rep.get((team, wk))})
    return pd.DataFrame(rows)

def main():
    L = loads()
    G = pd.read_parquet(ROOT / "data/raw/games.parquet")
    G = G[(G.game_type == "REG") & G.spread_line.notna() & G.home_score.notna() & G.season.between(2013, 2025)].copy()
    for c in ("home_team", "away_team"): G[c] = G[c].replace(FIX)
    G = G.merge(L.rename(columns={"team": "home_team", "load": "lh", "report_ts": "rh"}), on=["season", "week", "home_team"], how="left")
    G = G.merge(L.rename(columns={"team": "away_team", "load": "la", "report_ts": "ra"}), on=["season", "week", "away_team"], how="left")
    G[["lh", "la"]] = G[["lh", "la"]].fillna(0.0)
    G["diff"] = G.lh - G.la
    thr = float(G["diff"].abs().quantile(0.9))
    G["res"] = (G.home_score - G.away_score) - G.spread_line
    B = G[G["diff"].abs() >= thr].copy()
    s = -np.sign(B["diff"])            # bet against the more-injured side (+1 = home)
    r = s * B.res; pr = np.where(s > 0, dec(B.home_spread_odds), dec(B.away_spread_odds))
    pnl = np.where(r > 0, pr - 1, np.where(r < 0, -1, 0)); m, se = pnl.mean(), pnl.std() / math.sqrt(len(pnl))
    per = pd.Series(pnl, index=B.season).groupby(level=0).mean()
    out = {"threshold_load_diff": round(thr, 3), "a_outcome": {"bets": int(len(B)), "cover": round(float((r > 0).sum() / max(1, (r != 0).sum())), 3),
           "roi": round(float(m), 4), "se": round(float(se), 4), "p": round(0.5 * math.erfc((m / se) / math.sqrt(2)), 4),
           "seasons_positive": f"{int((per > 0).sum())}/{len(per)}", "by_season": {int(k): round(float(v), 3) for k, v in per.items()}}}
    # also regression across all games: does the injury diff predict result - close?
    b = np.polyfit(G["diff"], G.res, 1)[0]; se_b = float(np.std(G.res - b * G["diff"]) / (np.std(G["diff"]) * math.sqrt(len(G))))
    out["a_regression_pts_per_unit_load"] = {"beta": round(float(b), 3), "se": round(se_b, 3), "note": "negative = injured home side does worse than the close implies"}
    # (b) timing with hourly odds 2022-2025
    fs = sorted(glob.glob(str(ROOT / "data/historical_odds/dense/nfl_odds_*.csv.gz")))
    O = pd.concat([pd.read_csv(f, usecols=["requested_ts", "commence_time", "home", "away", "book", "sp_home_point"]) for f in fs], ignore_index=True)
    O["t"] = pd.to_datetime(O.requested_ts, utc=True); O["ko"] = pd.to_datetime(O.commence_time, utc=True)
    O = O[(O.t < O.ko) & O.sp_home_point.notna()]
    O["home"] = O.home.replace(FIX); O["away"] = O.away.replace(FIX)
    C = O.groupby(["home", "away", "ko", "t"]).sp_home_point.median().reset_index()     # consensus home point per hour
    C["kd"] = C.ko.dt.tz_convert("America/New_York").dt.date
    Bt = B[B.season >= 2022].copy(); Bt["rep"] = pd.to_datetime(np.where(np.sign(Bt["diff"]) > 0, Bt.rh, Bt.ra), utc=True)
    Bt["gd"] = pd.to_datetime(Bt.gameday).dt.date
    mv = []
    for g in Bt.itertuples():
        c = C[(C.home == g.home_team) & (C.away == g.away_team) & (C.kd == g.gd)]
        if c.empty or pd.isna(g.rep):
            continue
        c = c.sort_values("t")
        pre = c[c.t <= g.rep - pd.Timedelta(hours=1)]; post = c[c.t >= g.rep + pd.Timedelta(hours=2)]
        if pre.empty or post.empty:
            continue
        side = -np.sign(g.diff)   # our bet side (+1 home)
        p0, p1, pc = pre.sp_home_point.iloc[-1], post.sp_home_point.iloc[0], c.sp_home_point.iloc[-1]
        # home point more negative = home more favored; move "our way" for a home bet = point falling
        mv.append({"season": g.season, "pre_to_post": float(-side * (p1 - p0)), "post_to_close": float(-side * (pc - p1)),
                   "pre_to_close": float(-side * (pc - p0))})
    M = pd.DataFrame(mv)
    if len(M):
        x = M.post_to_close
        out["b_timing_2022_25"] = {"games": int(len(M)), "mean_move_pre_to_post_pts": round(float(M.pre_to_post.mean()), 3),
                                   "mean_move_post_to_close_pts (our CLV if bet 2h after report)": round(float(x.mean()), 3),
                                   "se": round(float(x.std() / math.sqrt(len(x))), 3), "share_moved_our_way": round(float((x > 0).mean()), 3),
                                   "share_moved_against": round(float((x < 0).mean()), 3)}
    (ROOT / "output/research/screen7/s5_injuries.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
