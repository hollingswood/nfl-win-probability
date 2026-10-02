"""PLAYER PROPS vs THE MARKET at three snapshots (Tue open / Fri early / close): accuracy, line moves, model-vs-market,
cross-book shopping, systematic biases, anytime-TD pricing. Dev 2023, freeze <=4 rules, 2024-25 run ONCE.

Stages (cache in the scratch dir; holdout refuses to run without output/research/props_full_frozen.json):
  PYTHONPATH=src:scripts python scripts/research/props_full.py build    # props.py build (player/team-game tables)
  PYTHONPATH=src:scripts python scripts/research/props_full.py fit      # projections per snapshot mode + TD model
  PYTHONPATH=src:scripts python scripts/research/props_full.py prep     # load/match props, consensus per snapshot
  PYTHONPATH=src:scripts python scripts/research/props_full.py dev      # 2023 analyses + rule grids (scratch json)
  PYTHONPATH=src:scripts python scripts/research/props_full.py holdout  # frozen rules on 2024-25 + all descriptives
  output/research/props_full.md is written by hand from output/research/props_full.json.

Data: data/historical_odds/props/*.csv.gz (Odds API, regions=us). Snapshots (requested_ts):
  open  = Tuesday 14:10 UTC of game week (props are posted for only a minority of games by then: 14%/29%/65% of
          events in 2023/24/25 for the 5-market file), early = Fri 21:40 UTC (Sunday games; kickoff-24h otherwise),
  close = kickoff-75 min.
Markets: pass_yds, pass_att(empts), rush_yds, receptions, rec_yds (O/U) and anytime TD (only the 'Yes' price is
  in the feed; it is stored in the under_price column; there is no 'No' side, so no per-book de-vig).
O/U main-line filter (props_backtest): both prices within [-200, +170] (drops Kambi milestone/alt quotes).

Projections (scripts/research/props.py pipeline, refit here per information set; train = seasons < S):
  open  = no injury report: teammate shares renormalised over the whole recent rotation (players listed Out later
          are still counted), no questionable flag; game lines = the Tuesday snapshot (team totals + spread).
  early = props.py "early" (final Out/Doubtful report) with Friday-snapshot game lines.
  late  = props.py "late" (actual actives; valid at close, inactives come out at T-90) with close-snapshot lines.
  Training rows use nflverse closing lines (2014-22 have no snapshot lines); prediction rows for 2023-25 use the
  snapshot lines. Weather is still RECORDED weather (optimistic, as in props.py).
  Distribution = props.py conditional-empirical (400 nearest out-of-sample (mu,y) pairs, previous 4 seasons).
TD model: LightGBM binary on prior-only red-zone (inside-10/20) carry & target shares, decayed TD rate, overall
  shares, implied team total (snapshot), position; same three information sets.

Conventions (as props_backtest): per-book no-vig P(over) multiplicative; per-quote shift c = L - Q_S(1-p) moves the
model distribution S to price that quote; consensus line L* = modal point, p* = median no-vig at L*.
Market-implied (price-adjusted) median = model median + median_b c_b.
CLV (O/U): P_close(side at bet point) x decimal - 1; P_close = median no-vig of close books quoting the same point
  ("direct") else the close consensus probability moved along the late-model distribution shifted to the close,
  damped by k=0.55 ("anchored", k from props_backtest 2023).
CLV (TD Yes): p_fair_close x decimal - 1 where p_fair_close = calibration map (fit on 2023 close quotes vs outcomes)
  applied to the median close implied probability across books (all us books). Vig-free by construction of the map.
Settlement: player must take >=1 offensive snap, else void (no CLV, no PnL). Outcomes = pbp official-style stats.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import props as PR  # noqa: E402
import props_backtest as PBT  # noqa: E402

RAW = ROOT / "data" / "raw"
PROPS = ROOT / "data" / "historical_odds" / "props"
ODDS = ROOT / "data" / "historical_odds"
OUT = ROOT / "output" / "research"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/propsfull")
PROJ = SCR / "proj"
PR.SCR = PROJ  # props.py build/fit caches go to our scratch
SEASONS = (2023, 2024, 2025)
DEVS, HOLDS = (2023,), (2024, 2025)
ALLOWED = set(json.load(open(ROOT / "my_books.json"))["allowed_books"])
SNAPS = ("open", "early", "close")
MODE_OF = {"open": "open", "early": "early", "close": "late"}
MKT = {"player_pass_yds": "pass_yds", "player_pass_attempts": "pass_att", "player_rush_yds": "rush_yds",
       "player_receptions": "receptions", "player_reception_yds": "rec_yds", "player_anytime_td": "td"}
OU = ["pass_yds", "pass_att", "rush_yds", "receptions", "rec_yds"]
STAT = {"pass_yds": "pass_yds", "pass_att": "pass_att", "rush_yds": "rush_yds", "receptions": "rec",
        "rec_yds": "rec_yds", "td": "td"}
K_OFF = PBT.K_OFF
TEAM_FIX = {"LA": "LA", "LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX"}


def dec(am):
    return PBT.dec(am)


def snap_of(rt: pd.Series, h: pd.Series) -> np.ndarray:
    return np.where(h < 3, "close", np.where((rt.dt.dayofweek == 1) & (rt.dt.hour == 14) & (h > 30), "open", "early"))


# =============================================================================================== FIT
def _team_abbr_map() -> dict:
    """Full team name -> nflverse abbreviation, learned from the game-odds feed (home/away + h2h names absent), so
    build from a fixed list."""
    m = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
         "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
         "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
         "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
         "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
         "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
         "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
         "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
         "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
         "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
    return m


def snapshot_requests() -> pd.DataFrame:
    """(event_id, requested_ts, snap) of the props snapshots (any market file)."""
    fr = []
    for s in SEASONS:
        for f in [f"player_pass_yds+player_pass_attempts+player_rush_yds+player_receptions+player_anytime_td_{s}",
                  f"player_reception_yds_{s}"]:
            fr.append(pd.read_csv(PROPS / f"{f}.csv.gz", usecols=["requested_ts", "event_id", "commence_time"]
                                  ).drop_duplicates(["event_id", "requested_ts"]))
    r = pd.concat(fr).drop_duplicates(["event_id", "requested_ts"])
    rt, ct = pd.to_datetime(r.requested_ts, utc=True), pd.to_datetime(r.commence_time, utc=True)
    r["snap"] = snap_of(rt, (ct - rt).dt.total_seconds() / 3600)
    return r[["event_id", "requested_ts", "snap"]]


def game_lines_by_snap() -> pd.DataFrame:
    """(game_id, snap) -> itt_home, itt_away, total, margin_home, from the game-odds requests made at the SAME
    requested_ts as the props snapshot (team totals: derivatives file; spread: nfl_odds)."""
    nm = _team_abbr_map()
    req = snapshot_requests()
    out = []
    for s in SEASONS:
        t = pd.read_csv(ODDS / "derivatives" / f"spreads_h1+totals_h1+team_totals_{s}.csv.gz")
        t = t[(t.market == "team_totals")].dropna(subset=["point"])
        o = pd.read_csv(ODDS / f"nfl_odds_{s}.csv.gz", usecols=["requested_ts", "event_id", "commence_time", "home",
                                                                 "away", "sp_home_point"])
        t = t.merge(req, on=["event_id", "requested_ts"])
        o = o.merge(req, on=["event_id", "requested_ts"])
        t["abbr"] = t.player.map(nm)
        t = t.dropna(subset=["abbr"])
        tt = t.groupby(["event_id", "snap", "home", "away", "abbr"]).point.median().reset_index()
        hh = tt[tt.abbr == tt.home].rename(columns={"point": "itt_home"})[["event_id", "snap", "itt_home"]]
        aa = tt[tt.abbr == tt.away].rename(columns={"point": "itt_away"})[["event_id", "snap", "itt_away"]]
        sp = o.groupby(["event_id", "snap"]).sp_home_point.median().rename("sp_home").reset_index()
        ev = pd.concat([t[["event_id", "home", "away", "commence_time"]],
                        o[["event_id", "home", "away", "commence_time"]]]).drop_duplicates("event_id")
        ev["ct"] = pd.to_datetime(ev.commence_time, utc=True)
        L = hh.merge(aa, on=["event_id", "snap"], how="outer").merge(sp, on=["event_id", "snap"], how="outer")
        L = L.merge(ev, on="event_id", how="left")
        L["season"] = s
        out.append(L)
    L = pd.concat(out, ignore_index=True)
    gm = PBT.map_games(L[["event_id", "home", "away", "ct"]].drop_duplicates("event_id"))
    L = L.merge(gm[["event_id", "game_id"]], on="event_id", how="left").dropna(subset=["game_id"])
    L["total"] = L.itt_home + L.itt_away
    L["margin_home"] = np.where(L.sp_home.notna(), -L.sp_home, L.itt_home - L.itt_away)
    return L[["game_id", "snap", "total", "margin_home", "itt_home", "itt_away", "sp_home"]]


def apply_lines(df: pd.DataFrame, GL: pd.DataFrame, snap: str) -> pd.DataFrame:
    """Replace total_line / spread_line (and derived itt, exp_margin) by snapshot lines where available."""
    df = df.copy()
    g = GL[GL.snap == snap].drop_duplicates("game_id").set_index("game_id")
    tot = df.game_id.map(g.total)
    mh = df.game_id.map(g.margin_home)
    ok = tot.notna() & mh.notna()
    df.loc[ok, "total_line"] = tot[ok]
    df.loc[ok, "spread_line"] = mh[ok]
    df["exp_margin"] = np.where(df.home == 1, df.spread_line, -df.spread_line)
    df["itt"] = df.total_line / 2 + df.exp_margin / 2
    df["snap_lines"] = ok.astype(int)
    return df


def fit_team_lines(tg_tr: pd.DataFrame, tg_te: pd.DataFrame) -> pd.DataFrame:
    from sklearn.linear_model import Ridge

    def X_(t):
        X = t[PR.TEAM_FEATS].copy()
        X["wind2"] = np.maximum(X.wind_f - 10, 0)
        X["margin_x_total"] = X.exp_margin * X.total_line / 45
        return X
    Xtr = X_(tg_tr)
    fill = Xtr.mean()
    Xtr = Xtr.fillna(fill)
    Xte = X_(tg_te).fillna(fill)
    res = tg_te[["game_id", "team"]].copy()
    for y in ["pass_att", "targets", "carries"]:
        m = Ridge(alpha=1.0).fit(Xtr, tg_tr[y])
        res[f"e_tm_{y}"] = m.predict(Xte)
    return res


def structural_open(d: pd.DataFrame) -> pd.DataFrame:
    """props.structural 'early' but renormalising over the whole recent rotation (no injury report yet)."""
    e = d.copy()
    e["early_sum_tgt"] = np.maximum(d.early_sum_tgt + d.vac_inj_tgt, d.tgt_share)
    e["early_sum_car"] = np.maximum(d.early_sum_car + d.vac_inj_car, d.car_share)
    return PR.structural(e, "early")


def td_features(a: pd.DataFrame) -> pd.DataFrame:
    """Per player-game red-zone usage and TDs from pbp; prior-only decayed sums (props._decayed_prior)."""
    cols = ["game_id", "posteam", "play_type", "pass_attempt", "sack", "rush_attempt", "two_point_attempt",
            "receiver_player_id", "rusher_player_id", "yardline_100", "touchdown", "td_player_id"]
    p = pd.concat([pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=cols) for s in PR.SEASONS], ignore_index=True)
    p["posteam"] = p.posteam.replace(TEAM_FIX)
    q = p[p.play_type.isin(["pass", "run"]) & (p.two_point_attempt.fillna(0) != 1)]
    att = (q.pass_attempt == 1) & (q.sack.fillna(0) == 0)
    rz10, rz20 = q.yardline_100 <= 10, q.yardline_100 <= 20
    parts = []
    for nm, mask, idc in [("rz10_car", (q.rush_attempt == 1) & rz10, "rusher_player_id"),
                          ("rz20_car", (q.rush_attempt == 1) & rz20, "rusher_player_id"),
                          ("rz10_tgt", att & rz10, "receiver_player_id"),
                          ("rz20_tgt", att & rz20, "receiver_player_id")]:
        x = q[mask & q[idc].notna()].groupby(["game_id", "posteam", idc]).size().rename(nm)
        x.index.names = ["game_id", "team", "player_id"]
        parts.append(x)
    td = p[(p.touchdown == 1) & p.td_player_id.notna()].groupby(["game_id", "td_player_id"]).size().rename("tds")
    td.index.names = ["game_id", "player_id"]
    pp = pd.concat(parts, axis=1).fillna(0).reset_index()
    tm = pp.groupby(["game_id", "team"])[["rz10_car", "rz20_car", "rz10_tgt", "rz20_tgt"]].sum()
    tm.columns = ["tm_" + c for c in tm.columns]
    out = a[["game_id", "team", "player_id", "gameday"]].merge(pp, on=["game_id", "team", "player_id"], how="left")
    out = out.merge(tm.reset_index(), on=["game_id", "team"], how="left")
    out = out.merge(td.reset_index(), on=["game_id", "player_id"], how="left")
    c = ["rz10_car", "rz20_car", "rz10_tgt", "rz20_tgt", "tm_rz10_car", "tm_rz20_car", "tm_rz10_tgt", "tm_rz20_tgt",
         "tds"]
    out[c] = out[c].fillna(0)
    out.index = a.index
    dp = PR._decayed_prior(out, "player_id", c, 8.0, "q_")
    f = pd.DataFrame(index=a.index)
    for k in ["rz10_car", "rz20_car", "rz10_tgt", "rz20_tgt"]:
        f[f"{k}_sh"] = (dp[f"q_{k}"] / dp[f"q_tm_{k}"].replace(0, np.nan)).fillna(0)
        f[f"{k}_pg"] = (dp[f"q_{k}"] / dp.q_n.replace(0, np.nan)).fillna(0)
    f["td_pg"] = (dp.q_tds / dp.q_n.replace(0, np.nan)).fillna(0)
    f["td_any"] = (out.tds > 0).astype(int).to_numpy()
    return f


def fit():
    import lightgbm as lgb
    a = pd.read_parquet(PROJ / "player_games.parquet")
    tg = pd.read_parquet(PROJ / "team_games.parquet")
    GL = game_lines_by_snap()
    GL.to_parquet(SCR / "game_lines.parquet")
    a = a.sort_values(["gameday", "game_id", "team", "player_id"]).reset_index(drop=True)
    tdf = td_features(a)
    a = pd.concat([a, tdf], axis=1)
    # closing-line team predictions for every season (training rows + pre-2023 test rows)
    team_close = pd.concat([PR.fit_team(tg, s) for s in range(2014, 2026)])
    preds = []
    for mode in ["open", "early", "late"]:
        snap = {"open": "open", "early": "early", "late": "close"}[mode]
        vac = {"open": None, "early": "inj", "late": "late"}[mode]
        base = a.merge(team_close, on=["game_id", "team"], how="left")
        base = base[base.season >= 2014].reset_index(drop=True)
        feats = PR.PLAYER_FEATS + ["share_t", "share_c", "e_targets", "e_carries", "e_att"]
        if vac:
            feats += [f"vac_{vac}_tgt", f"vac_{vac}_car"]
        else:
            feats = [f for f in feats if f != "questionable"]
        tdx = ["rz10_car_sh", "rz20_car_sh", "rz10_tgt_sh", "rz20_tgt_sh", "rz10_car_pg", "rz20_car_pg",
               "rz10_tgt_pg", "rz20_tgt_pg", "td_pg", "is_rb", "is_wr", "is_te", "is_qb"]
        for S in range(2019, 2026):
            tr_rows = base[base.season < S]
            te_rows = base[base.season == S]
            if S in SEASONS:  # snapshot game lines at prediction time
                te_rows = apply_lines(te_rows, GL, snap)
                tgte = apply_lines(tg[tg.season == S], GL, snap)
                tp = fit_team_lines(tg[(tg.season < S) & (tg.season >= 2013) & tg.to_n.gt(2)], tgte)
                te_rows = te_rows.drop(columns=["e_tm_pass_att", "e_tm_targets", "e_tm_carries"]).merge(
                    tp, on=["game_id", "team"], how="left")
            D = pd.concat([tr_rows, te_rows], ignore_index=True)
            st = structural_open(D) if mode == "open" else PR.structural(D, mode)
            D = pd.concat([D, st], axis=1)
            if mode == "open":
                D["questionable"] = 0
            for c, pos in [("is_rb", "RB"), ("is_wr", "WR"), ("is_te", "TE"), ("is_qb", "QB")]:
                D[c] = (D.position == pos).astype(int)
            tr, te = D.season < S, D.season == S
            for market, (stat, grp, kind) in PR.MARKETS.items():
                if market == "pass_cmp":
                    continue
                pool, uni = PR.train_pool(D, market), PR.universe(D, market)
                fx = feats + [f"sm_{stat}", f"l4_{stat}", f"std_{stat}"]
                params = dict(objective="poisson" if kind == "count" else "regression", learning_rate=0.05,
                              n_jobs=2, n_estimators=250, num_leaves=15, min_child_samples=100, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=0)
                m = lgb.LGBMRegressor(**params).fit(D.loc[tr & pool, fx], D.loc[tr & pool, stat].clip(
                    lower=0 if kind == "count" else None))
                r = D.loc[te & pool, ["game_id", "player_id", "season", "week", "team", "position", stat]].copy()
                r["gbm"] = m.predict(D.loc[te & pool, fx])
                r["in_universe"] = uni[te & pool].values
                r = r.rename(columns={stat: "y"})
                r["market"], r["mode"] = market, mode
                preds.append(r)
            # anytime TD: all skill players with >=1 prior game
            poolt = D.position.isin(["QB", "RB", "WR", "TE"]) & (D.n_prior >= 1)
            fx = feats + tdx + ["sm_rec_yds", "sm_rush_yds", "l4_rec_yds", "l4_rush_yds"]
            m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, n_jobs=2, n_estimators=300, num_leaves=15,
                                   min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   reg_lambda=5.0, verbose=-1, random_state=0).fit(D.loc[tr & poolt, fx],
                                                                                   D.loc[tr & poolt, "td_any"])
            r = D.loc[te & poolt, ["game_id", "player_id", "season", "week", "team", "position", "td_any",
                                   "itt"]].copy()
            r["gbm"] = m.predict_proba(D.loc[te & poolt, fx])[:, 1]
            r["in_universe"] = True
            r = r.rename(columns={"td_any": "y"})
            r["market"], r["mode"] = "td", mode
            preds.append(r)
            print(mode, S, "done", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_parquet(SCR / "preds.parquet")
    print(P.groupby(["mode", "market"]).size())
    # TD model calibration / skill 2019-25 (out of sample by construction)
    T = P[P.market == "td"]
    for (mode, s), x in T.groupby(["mode", "season"]):
        p = np.clip(x.gbm, 1e-4, 1 - 1e-4)
        ll = -(x.y * np.log(p) + (1 - x.y) * np.log(1 - p)).mean()
        b = x.y.mean()
        ll0 = -(b * np.log(b) + (1 - b) * np.log(1 - b))
        print(f"TD {mode} {s} n={len(x)} rate={b:.3f} pred={p.mean():.3f} ll={ll:.4f} base={ll0:.4f}")


# ======================================================================================== DISTRIBUTIONS
def samples(mode: str, market: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Sorted conditional-empirical samples for 2023-25 rows of (mode, market); rows keyed game|player."""
    f = SCR / f"smp_{mode}_{market}.npy"
    g = SCR / f"smp_{mode}_{market}.parquet"
    if f.exists():
        return pd.read_parquet(g), np.load(f)
    P = pd.read_parquet(SCR / "preds.parquet")
    pm = {"receptions": "receptions", "rec_yds": "rec_yds", "pass_yds": "pass_yds", "pass_att": "pass_att",
          "rush_yds": "rush_yds"}[market]
    P = P[(P.market == pm) & (P["mode"] == mode)]
    kind = PR.MARKETS[pm][2]
    rows, smps = [], []
    for S in SEASONS:
        tr = P[(P.season >= S - PR.LOOKBACK) & (P.season < S) & P.in_universe].dropna(subset=["y", "gbm"])
        te = P[P.season == S].dropna(subset=["gbm"])
        smp = np.sort(PR.cond_samples(tr.gbm.to_numpy(float), tr.y.to_numpy(float), te.gbm.to_numpy(float), kind),
                      axis=1)
        rows.append(te[["game_id", "player_id", "gbm", "in_universe"]])
        smps.append(smp)
    R = pd.concat(rows, ignore_index=True)
    R["key"] = R.game_id + "|" + R.player_id
    S_ = np.concatenate(smps).astype(np.float32)
    keep = ~R.key.duplicated().to_numpy()
    R, S_ = R[keep].reset_index(drop=True), S_[keep]
    R.to_parquet(g)
    np.save(f, S_)
    return R, S_


# ================================================================================================ PREP
def load_quotes() -> pd.DataFrame:
    fr = []
    for s in SEASONS:
        for f in [f"player_pass_yds+player_pass_attempts+player_rush_yds+player_receptions+player_anytime_td_{s}",
                  f"player_reception_yds_{s}"]:
            x = pd.read_csv(PROPS / f"{f}.csv.gz")
            x["season"] = s
            fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d["market"] = d.market.map(MKT)
    d["ct"] = pd.to_datetime(d.commence_time, utc=True)
    d["rt"] = pd.to_datetime(d.requested_ts, utc=True)
    d["hours_before"] = (d.ct - d.rt).dt.total_seconds() / 3600
    d["snap"] = snap_of(d.rt, d.hours_before)
    td = d.market == "td"
    # anytime TD: only the Yes price exists (in under_price)
    d.loc[td, "yes_price"] = d.loc[td, "under_price"]
    d.loc[td, ["over_price", "under_price"]] = np.nan
    ou = d[~td].dropna(subset=["over_price", "under_price", "point"]).copy()
    main = ou.over_price.between(-200, 170) & ou.under_price.between(-200, 170)
    print("O/U quotes dropped as non-main (milestone/alt):", int((~main).sum()), "of", len(ou))
    ou = ou[main]
    ou["dec_o"], ou["dec_u"] = dec(ou.over_price), dec(ou.under_price)
    io, iu = 1 / ou.dec_o, 1 / ou.dec_u
    ou["p_nv"] = io / (io + iu)
    ou["hold"] = io + iu - 1
    t = d[td].dropna(subset=["yes_price"]).copy()
    t = t[t.yes_price.abs() >= 100]
    t["dec_y"] = dec(t.yes_price)
    t["p_imp"] = 1 / t.dec_y
    d = pd.concat([ou, t], ignore_index=True)
    # keep one quote per (event, snap, book, market, player, point): the latest requested within the snapshot
    d = d.sort_values("rt").drop_duplicates(["event_id", "snap", "book", "market", "player", "point"], keep="last")
    return d


def prep():
    SCR.mkdir(parents=True, exist_ok=True)
    d = load_quotes()
    gm = PBT.map_games(d)
    d = d.merge(gm, on="event_id", how="left")
    a = pd.read_parquet(PROJ / "player_games.parquet",
                        columns=["game_id", "team", "player_id", "season", "position", "pass_yds", "pass_att",
                                 "rush_yds", "rec", "rec_yds", "home", "spread_line", "total_line"])
    a = a[a.season.isin(SEASONS)]
    P = pd.read_parquet(SCR / "preds.parquet", columns=["game_id", "player_id", "market", "mode", "y"])
    tdy = P[(P.market == "td") & (P["mode"] == "late")].drop_duplicates(["game_id", "player_id"])
    a = a.merge(tdy[["game_id", "player_id", "y"]].rename(columns={"y": "td"}), on=["game_id", "player_id"],
                how="left")
    mp = PBT.match_players(d[d.game_id.notna()], a)
    d = d.merge(mp, on=["game_id", "player"], how="left")
    d["status"] = d.status.fillna("no_game")
    d = d.merge(a.drop(columns=["season"]).rename(columns={"home": "is_home"}), on=["game_id", "player_id"],
                how="left")
    d["y"] = np.nan
    for m, c in STAT.items():
        k = d.market == m
        d.loc[k, "y"] = d.loc[k, c]
    d["key"] = d.game_id.astype(str) + "|" + d.player_id.astype(str)
    d = d.drop(columns=["ct", "rt", "pass_yds", "pass_att", "rush_yds", "rec", "rec_yds", "td", "commence_time",
                        "requested_ts", "snapshot_ts"])
    d.to_parquet(SCR / "quotes.parquet")
    u = d.drop_duplicates(["event_id", "player", "market"])
    print(u.groupby(["season", "market", "status"]).size().unstack(fill_value=0))
    print(u[u.status == "played"].how.value_counts())


# ===================================================================================== ANALYSIS TABLE
def ou_table(market: str):
    """Played & modelled O/U quotes for one market with per-quote shift c (vs the snapshot's model mode), plus
    the model samples per mode."""
    d = pd.read_parquet(SCR / "quotes.parquet")
    d = d[(d.market == market) & (d.status == "played")].copy()
    if market == "rush_yds":
        d = d[d.position == "RB"]  # model universe: RB rushing (QB rush-yds props are not modelled)
    Smod = {}
    for mode in ["open", "early", "late"]:
        R, S_ = samples(mode, market)
        Smod[mode] = (pd.Series(np.arange(len(R)), index=R.key), S_, R)
    for mode in ["open", "early", "late"]:
        d[f"si_{mode}"] = d.key.map(Smod[mode][0])
    d = d[d[["si_open", "si_early", "si_late"]].notna().all(axis=1)].copy()
    for mode in ["open", "early", "late"]:
        d[f"si_{mode}"] = d[f"si_{mode}"].astype(int)
    d["c"] = np.nan
    for sn in SNAPS:
        m = (d.snap == sn).to_numpy()
        mode = MODE_OF[sn]
        S_ = Smod[mode][1][d.loc[m, f"si_{mode}"].to_numpy()]
        d.loc[m, "c"] = PBT.shift_for(S_, d.loc[m, "point"].to_numpy(float), d.loc[m, "p_nv"].to_numpy(float))
    return d, Smod


def consensus(q: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (k, sn), G in q.groupby(["key", "snap"], sort=False):
        vc = G.point.value_counts()
        top = vc[vc == vc.max()].index.to_numpy()
        L = top[np.argmin(np.abs(top - G.point.median()))]
        out.append((k, sn, L, G.loc[G.point == L, "p_nv"].median(), G.c.median(), len(G), G.hold.median()))
    return pd.DataFrame(out, columns=["key", "snap", "L", "p", "c", "nbooks", "hold"])


def pg_table(d: pd.DataFrame, Smod: dict) -> pd.DataFrame:
    """One row per player-game: consensus at each snapshot + model medians/probabilities."""
    C = consensus(d)
    W = C.pivot(index="key", columns="snap")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    base = d.drop_duplicates("key").set_index("key")[["game_id", "player_id", "season", "week", "team", "position",
                                                      "is_home", "spread_line", "y", "si_open", "si_early", "si_late"]]
    T = base.join(W).reset_index()
    for mode in ["open", "early", "late"]:
        S_ = Smod[mode][1][T[f"si_{mode}"].to_numpy()]
        T[f"med_{mode}"] = np.median(S_, axis=1)
        T[f"mu_{mode}"] = Smod[mode][2].gbm.to_numpy()[T[f"si_{mode}"].to_numpy()]
    for sn in SNAPS:
        if f"L_{sn}" not in T:
            T[f"L_{sn}"] = np.nan
            T[f"c_{sn}"] = np.nan
            T[f"p_{sn}"] = np.nan
        T[f"mkt_med_{sn}"] = T[f"med_{MODE_OF[sn]}"] + T[f"c_{sn}"]
        ok = T[f"L_{sn}"].notna().to_numpy()
        mode = MODE_OF[sn]
        S_ = Smod[mode][1][T.loc[ok, f"si_{mode}"].to_numpy()]
        T.loc[ok, f"pm_{sn}"] = (S_ > T.loc[ok, f"L_{sn}"].to_numpy()[:, None]).mean(axis=1)
    return T


def clustered(x: np.ndarray, g: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    x, g = x[m], np.asarray(g)[m]
    if len(x) == 0:
        return np.nan, np.nan, 0
    u, gi = np.unique(g, return_inverse=True)
    s = np.bincount(gi, weights=x - x.mean())
    return float(x.mean()), float(np.sqrt((s ** 2).sum()) / len(x)), int(len(x))


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# ------------------------------------------------------------------------------------ A1 accuracy
def accuracy(T: pd.DataFrame, seasons) -> dict:
    D = T[T.season.isin(seasons)]
    res = {}
    allc = D[[f"L_{s}" for s in SNAPS]].notna().all(axis=1)
    for lab, M in [("all_three", D[allc]), ("early_close", D[D.L_early.notna() & D.L_close.notna()])]:
        r = {"n": int(len(M))}
        if len(M) < 20:
            res[lab] = r
            continue
        g = M.game_id.to_numpy()
        for sn in SNAPS:
            if M[f"L_{sn}"].isna().any():
                continue
            e = (M.y - M[f"L_{sn}"]).abs()
            r[f"line_mae_{sn}"] = r4(e.mean())
            r[f"mktmed_mae_{sn}"] = r4((M.y - M[f"mkt_med_{sn}"]).abs().mean())
            r[f"model_mae_{sn}"] = r4((M.y - M[f"med_{MODE_OF[sn]}"]).abs().mean())
            nz = M.y != M[f"L_{sn}"]
            r[f"over_rate_{sn}"] = r4((M.y > M[f"L_{sn}"])[nz].mean())
            r[f"mean_pnv_over_{sn}"] = r4(M[f"p_{sn}"].mean())
        if lab == "all_three":
            dd = (M.y - M.L_open).abs() - (M.y - M.L_close).abs()
            m_, se, _ = clustered(dd.to_numpy(), g)
            r["open_minus_close_mae"] = [r4(m_), r4(se)]
            dd = (M.y - M.L_early).abs() - (M.y - M.L_close).abs()
            m_, se, _ = clustered(dd.to_numpy(), g)
            r["early_minus_close_mae"] = [r4(m_), r4(se)]
            dd = (M.y - M.L_open).abs() - (M.y - M.med_open).abs()
            m_, se, _ = clustered(dd.to_numpy(), g)
            r["open_line_minus_open_model_mae"] = [r4(m_), r4(se)]
        res[lab] = r
    return res


# ------------------------------------------------------------------------------------ A2 line moves
def moves(T: pd.DataFrame, fit_s, test_s) -> dict:
    """Open->close move of the price-adjusted market median; is it predictable from open-time and Friday info?"""
    D = T[T.mkt_med_open.notna() & T.mkt_med_close.notna() & T.mkt_med_early.notna()].copy()
    D["mv"] = D.mkt_med_close - D.mkt_med_open
    D["mv_oe"] = D.mkt_med_early - D.mkt_med_open
    D["mv_ec"] = D.mkt_med_close - D.mkt_med_early
    D["x_gap_open"] = D.med_open - D.mkt_med_open           # our open projection vs open line (known Tue)
    D["x_fri_info"] = D.med_early - D.med_open              # Friday info: injury report + game-line moves
    D["x_late_info"] = D.med_late - D.med_early             # actives + close game lines
    D["res_open"] = D.y - D.mkt_med_open
    out = {"n": {str(s): int((D.season == s).sum()) for s in SEASONS}}
    for s in SEASONS:
        x = D[D.season == s]
        if len(x) < 30:
            continue
        out[str(s)] = {"mean_abs_move_open_close": r4(x.mv.abs().mean()), "share_line_moved":
                       r4((x.L_open != x.L_close).mean()), "mean_abs_open_early": r4(x.mv_oe.abs().mean()),
                       "mean_abs_early_close": r4(x.mv_ec.abs().mean()),
                       "corr_move_with_open_residual": r4(np.corrcoef(x.mv, x.res_open)[0, 1]),
                       "corr_move_gap_open": r4(np.corrcoef(x.mv, x.x_gap_open)[0, 1]),
                       "corr_move_fri_info": r4(np.corrcoef(x.mv, x.x_fri_info)[0, 1])}
    from sklearn.linear_model import LinearRegression
    tr, te = D[D.season.isin(fit_s)], D[D.season.isin(test_s)]
    reg = {}
    for lab, xs in [("gap_open", ["x_gap_open"]), ("gap_open+fri", ["x_gap_open", "x_fri_info"]),
                    ("gap_open+fri+late", ["x_gap_open", "x_fri_info", "x_late_info"])]:
        if len(tr) < 30:
            break
        m = LinearRegression().fit(tr[xs], tr.mv)
        r = {"coef": [r4(c) for c in m.coef_], "r2_fit": r4(m.score(tr[xs], tr.mv)), "n_fit": int(len(tr))}
        if len(te) > 30:
            p = m.predict(te[xs])
            r["r2_test"] = r4(1 - ((te.mv - p) ** 2).sum() / ((te.mv - te.mv.mean()) ** 2).sum())
            r["dir_acc_test_when_|pred|>sd/4"] = r4(
                (np.sign(p) == np.sign(te.mv))[(np.abs(p) > te.mv.std() / 4) & (te.mv != 0)].mean())
            r["n_test"] = int(len(te))
        reg[lab] = r
    out["regression"] = reg
    return out


# --------------------------------------------------------------------------- O/U close valuation helper
def close_value(E: pd.DataFrame, d: pd.DataFrame, T: pd.DataFrame, Smod: dict, k: float = K_OFF) -> pd.DataFrame:
    """Adds p_close_over (direct at the same point, else anchored on the late-model distribution)."""
    E = E.merge(T[["key", "L_close", "p_close", "c_close"]], on="key", how="left")
    C = d[d.snap == "close"].groupby(["key", "point"]).p_nv.median().rename("p_close_direct").reset_index()
    E = E.merge(C, on=["key", "point"], how="left")
    Sl = Smod["late"][1][E.si_late.to_numpy()]
    cc = E.c_close.to_numpy(float)
    L = E.point.to_numpy(float)
    ok = ~np.isnan(cc)
    pL, pLs = np.full(len(E), np.nan), np.full(len(E), np.nan)
    pL[ok] = ((Sl[ok] + cc[ok, None]) > L[ok, None]).mean(axis=1)
    pLs[ok] = ((Sl[ok] + cc[ok, None]) > E.L_close.to_numpy(float)[ok, None]).mean(axis=1)
    anc = np.clip(E.p_close.to_numpy(float) + k * (pL - pLs), 0.005, 0.995)
    E["p_close_over"] = E.p_close_direct.fillna(pd.Series(anc, index=E.index))
    E["clv_src"] = np.where(E.p_close_direct.notna(), "direct", np.where(ok, "anchored", "none"))
    return E


def sides(E: pd.DataFrame, p_over_col: str, extra: list[str]) -> pd.DataFrame:
    cols = ["key", "game_id", "season", "week", "book", "point", "snap", "y", "clv_src", "p_close_over", "p_nv",
            "position", "team"] + extra
    o = E[cols + ["dec_o", p_over_col]].rename(columns={"dec_o": "dec", p_over_col: "p"})
    o["side"] = "over"
    u = E[cols + ["dec_u", p_over_col]].rename(columns={"dec_u": "dec", p_over_col: "p"})
    u["p"] = 1 - u.p
    u["side"] = "under"
    u["p_close_over"] = 1 - u.p_close_over
    u["p_nv"] = 1 - u.p_nv
    b = pd.concat([o, u], ignore_index=True).rename(columns={"p_close_over": "p_close"})
    b["ev"] = b.p * b.dec - 1
    return b


def grade(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    b["clv"] = b.p_close * b.dec - 1
    win = np.where(b.side == "yes", b.y > 0, np.where(b.side == "over", b.y > b.point, b.y < b.point))
    push = (b.y == b.point) & (b.side != "yes")
    b["pnl"] = np.where(push, 0.0, np.where(win, b.dec - 1, -1.0))
    b["win"] = np.where(push, np.nan, win.astype(float))
    return b


def summarize(b: pd.DataFrame, k_tests: int = 1) -> dict:
    from scipy.stats import norm
    if len(b) == 0:
        return {"n": 0}
    g = b.game_id.to_numpy()
    cm, cse, cn = clustered(b.clv.to_numpy(), g)
    rm, rse, _ = clustered(b.pnl.to_numpy(), g)
    weeks = b.groupby(["season", "week"]).ngroups
    t = cm / cse if cse and cse > 0 else np.nan
    return {"n": int(len(b)), "n_clv": cn, "clv": r4(cm), "clv_se": r4(cse), "clv_t": r4(t),
            "clv_p": r4(1 - norm.cdf(t)) if not np.isnan(t) else None,
            "roi": r4(rm), "roi_se": r4(rse), "hit": r4(np.nanmean(b.win)), "bets_per_week": round(len(b) / max(weeks, 1), 2),
            "weeks": int(weeks), "mean_ev": r4(b.ev.mean()), "mean_dec": r4(b.dec.mean()),
            "under_share": r4((b.side == "under").mean()) if "side" in b else None,
            "direct_share": r4((b.clv_src == "direct").mean()) if "clv_src" in b else None}


# ------------------------------------------------------------------------------------ A3 model at open
def blend_fit(T: pd.DataFrame, Smod: dict, sn: str, seasons) -> dict:
    """Log-likelihood of over/under at the consensus line for blend w (S + (1-w)c): grid on given seasons."""
    D = T[T.season.isin(seasons) & T[f"L_{sn}"].notna() & T[f"c_{sn}"].notna()]
    D = D[D.y != D[f"L_{sn}"]]
    mode = MODE_OF[sn]
    S_ = Smod[mode][1][D[f"si_{mode}"].to_numpy()]
    L = D[f"L_{sn}"].to_numpy(float)
    c = D[f"c_{sn}"].to_numpy(float)
    o = (D.y > L).to_numpy()
    res = {"n": int(len(D))}
    for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        p = np.clip(((S_ + ((1 - w) * c)[:, None]) > L[:, None]).mean(axis=1), 0.01, 0.99)
        res[str(w)] = r4(np.mean(np.where(o, np.log(p), np.log(1 - p))))
    mkt = np.clip(D[f"p_{sn}"].to_numpy(float), 0.01, 0.99)
    res["market_nv"] = r4(np.mean(np.where(o, np.log(mkt), np.log(1 - mkt))))
    return res


def model_bets(d, T, Smod, sn: str, w: float) -> pd.DataFrame:
    """All allowed-book quotes at snapshot sn with blend-w probabilities -> both sides."""
    E = d[(d.snap == sn) & d.book.isin(ALLOWED)].copy()
    E = E.merge(T[["key", f"c_{sn}"]], on="key", how="left")
    mode = MODE_OF[sn]
    S_ = Smod[mode][1][E[f"si_{mode}"].to_numpy()]
    X = S_ + ((1 - w) * E[f"c_{sn}"].to_numpy(float))[:, None]
    L = E.point.to_numpy(float)
    E["p_over_m"] = (X > L[:, None]).mean(axis=1)
    pp = (np.abs(X - L[:, None]) < 0.5).mean(axis=1) * (L % 1 == 0)
    E["p_over_m"] = E.p_over_m + 0.5 * pp  # crude push split (integer lines are rare)
    E = close_value(E, d, T, Smod)
    return sides(E, "p_over_m", ["spread_line", "is_home"])


def shop_bets(d, T, Smod, sn: str) -> pd.DataFrame:
    """Allowed-book quote vs leave-one-out median no-vig of >=3 OTHER books (any us book) at the same point."""
    Q = d[d.snap == sn]
    grp = Q.groupby(["key", "point"])
    allp = grp.p_nv.apply(list)
    E = Q[Q.book.isin(ALLOWED)].copy()
    lst = E.set_index(["key", "point"]).index.map(allp)
    loo, nb = [], []
    for ps, own in zip(lst, E.p_nv.to_numpy()):
        ps = list(ps)
        ps.remove(own)
        nb.append(len(ps))
        loo.append(np.median(ps) if len(ps) >= 3 else np.nan)
    E["p_loo"], E["n_other"] = loo, nb
    E = E[E.p_loo.notna()]
    E = close_value(E, d, T, Smod)
    return sides(E, "p_loo", ["n_other", "spread_line", "is_home"])


def pick(b: pd.DataFrame, thr: float, side="both", ev_max=0.5, one_per="key") -> pd.DataFrame:
    if side != "both":
        b = b[b.side == side]
    b = b.sort_values("ev", ascending=False).drop_duplicates(one_per)
    return grade(b[(b.ev >= thr) & (b.ev < ev_max)])


# ------------------------------------------------------------------------------------------ TD tables
def td_table():
    d = pd.read_parquet(SCR / "quotes.parquet")
    d = d[(d.market == "td") & (d.status == "played")].copy()
    d = d[d.y.notna()]  # TD outcome/model exist for players with >=1 prior offensive game (rookie debuts dropped)
    P = pd.read_parquet(SCR / "preds.parquet")
    P = P[P.market == "td"]
    P["key"] = P.game_id + "|" + P.player_id
    for mode in ["open", "early", "late"]:
        s = P[P["mode"] == mode].drop_duplicates("key").set_index("key")
        d[f"pm_{mode}"] = d.key.map(s.gbm)
        d[f"itt_{mode}"] = d.key.map(s.itt)
    d["pm"] = np.select([d.snap == "open", d.snap == "early"], [d.pm_open, d.pm_early], d.pm_late)
    C = d.groupby(["key", "snap"]).p_imp.median().rename("p_med").reset_index()
    N = d.groupby(["key", "snap"]).size().rename("nb").reset_index()
    W = C.merge(N, on=["key", "snap"]).pivot(index="key", columns="snap")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    d = d.merge(W.reset_index(), on="key", how="left")
    return d


def td_calib_fit(d: pd.DataFrame, seasons, col="p_med_close") -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    x = d[d.season.isin(seasons)].drop_duplicates("key").dropna(subset=[col])
    m = LogisticRegression(C=1e6).fit(PBT.logit(x[col].to_numpy())[:, None], x.y.astype(int))
    return float(m.intercept_[0]), float(m.coef_[0][0])


def td_fair(p, ab):
    return 1 / (1 + np.exp(-(ab[0] + ab[1] * PBT.logit(np.asarray(p, float)))))


def td_bets(d: pd.DataFrame, sn: str, ab_close: tuple, ab_snap: tuple, w: float, how: str) -> pd.DataFrame:
    """Yes bets at snapshot sn, allowed books. how='model': p = blend of calibrated market (at sn) and model in
    logit space with weight w on the model; how='shop': p = calibrated LOO median implied prob of >=3 other books."""
    E = d[(d.snap == sn) & d.book.isin(ALLOWED)].copy()
    if how == "shop":
        allp = d[d.snap == sn].groupby("key").p_imp.apply(list)
        loo = []
        for k, own in zip(E.key, E.p_imp):
            ps = list(allp[k])
            ps.remove(own)
            loo.append(np.median(ps) if len(ps) >= 3 else np.nan)
        E["p"] = td_fair(np.array(loo), ab_snap)
    else:
        pm = E[f"p_med_{sn}"].to_numpy(float)
        E["p"] = 1 / (1 + np.exp(-((1 - w) * PBT.logit(td_fair(pm, ab_snap)) + w * PBT.logit(E.pm.to_numpy(float)))))
    E = E[E.p.notna() & E.pm.notna()]
    E["dec"] = E.dec_y
    E["ev"] = E.p * E.dec - 1
    E["p_close"] = td_fair(E.p_med_close.to_numpy(float), ab_close)
    E["clv"] = E.p_close * E.dec - 1
    E["side"] = "yes"
    E["clv_src"] = np.where(E.p_med_close.notna(), "close_consensus", "none")
    E["pnl"] = np.where(E.y > 0, E.dec - 1, -1.0)
    E["win"] = (E.y > 0).astype(float)
    return E


def td_analysis(d: pd.DataFrame, seasons, ab: dict) -> dict:
    from sklearn.metrics import roc_auc_score
    res = {}
    D = d[d.season.isin(seasons)].drop_duplicates("key")
    for sn in SNAPS:
        x = D.dropna(subset=[f"p_med_{sn}", "pm_" + MODE_OF[sn]])
        if len(x) < 50:
            continue
        pm_ = x["pm_" + MODE_OF[sn]].to_numpy(float)
        pk = td_fair(x[f"p_med_{sn}"], ab[sn])
        yy = x.y.to_numpy(int)

        def ll(p):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
        res[sn] = {"n": int(len(x)), "td_rate": r4(yy.mean()), "mkt_implied_mean": r4(x[f"p_med_{sn}"].mean()),
                   "mkt_fair_mean": r4(pk.mean()), "model_mean": r4(pm_.mean()),
                   "auc_market": r4(roc_auc_score(yy, x[f"p_med_{sn}"])), "auc_model": r4(roc_auc_score(yy, pm_)),
                   "ll_market_calib": r4(ll(pk)), "ll_model": r4(ll(pm_)),
                   "ll_blend_half": r4(ll(1 / (1 + np.exp(-(0.5 * PBT.logit(pk) + 0.5 * PBT.logit(pm_))))))}
    # price buckets (close): favourite-longshot pattern
    x = D.dropna(subset=["p_med_close"]).copy()
    x["bk"] = pd.cut(x.p_med_close, [0, .1, .2, .3, .4, .5, .7, 1])
    res["close_buckets"] = {str(k): {"n": int(len(g)), "implied": r4(g.p_med_close.mean()), "rate": r4(g.y.mean())}
                            for k, g in x.groupby("bk", observed=True)}
    res["by_position_close"] = {k: {"n": int(len(g)), "implied": r4(g.p_med_close.mean()), "rate": r4(g.y.mean()),
                                    "model": r4(g.pm_late.mean())}
                                for k, g in x.groupby("position")}
    return res


def td_info_test(d: pd.DataFrame, sn: str, fit_s, test_s, ab) -> dict:
    from sklearn.linear_model import LogisticRegression
    mode = MODE_OF[sn]
    D = d.drop_duplicates("key").dropna(subset=[f"p_med_{sn}", f"pm_{mode}"]).copy()
    D["xm"], D["xo"] = PBT.logit(D[f"p_med_{sn}"].to_numpy(float)), PBT.logit(D[f"pm_{mode}"].to_numpy(float))
    tr, te = D[D.season.isin(fit_s)], D[D.season.isin(test_s)]
    if len(tr) < 100:
        return {"n_fit": int(len(tr))}
    m1 = LogisticRegression(C=1e6).fit(tr[["xm"]], tr.y.astype(int))
    m2 = LogisticRegression(C=1e6).fit(tr[["xm", "xo"]], tr.y.astype(int))
    r = {"n_fit": int(len(tr)), "coef_mkt_plus_model": [r4(m2.intercept_[0])] + [r4(c) for c in m2.coef_[0]]}
    if len(te) > 100:
        o = te.y.to_numpy(int)

        def ll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -(o * np.log(p) + (1 - o) * np.log(1 - p))
        g = ll(m1.predict_proba(te[["xm"]])[:, 1]) - ll(m2.predict_proba(te[["xm", "xo"]])[:, 1])
        mm, se, n = clustered(g, te.game_id.to_numpy())
        r.update({"n_test": n, "ll_gain_model": [r4(mm), r4(se)]})
    return r


# ------------------------------------------------------------------------------------- A5 biases
def biases(T: pd.DataFrame, d: pd.DataFrame, seasons, market: str) -> dict:
    res = {}
    D = T[T.season.isin(seasons)]
    for sn in SNAPS:
        x = D[D[f"L_{sn}"].notna() & (D[f"p_{sn}"] - 0.5).abs().lt(0.04) & (D.y != D[f"L_{sn}"])]
        if len(x) < 30:
            continue
        o = (x.y > x[f"L_{sn}"]).astype(float).to_numpy()
        m, se, n = clustered(o, x.game_id.to_numpy())
        # game script: team expected margin (nflverse closing spread, sign = team perspective)
        em = np.where(x.is_home == 1, x.spread_line, -x.spread_line)
        fav = {}
        for lab, mask in [("big_dog(<=-6)", em <= -6), ("dog(-6,-1]", (em > -6) & (em <= -1)),
                          ("pickish(-1,1)", (em > -1) & (em < 1)), ("fav[1,6)", (em >= 1) & (em < 6)),
                          ("big_fav(>=6)", em >= 6)]:
            if mask.sum() >= 20:
                mm, ss, nn = clustered(o[mask], x.game_id.to_numpy()[mask])
                fav[lab] = [nn, r4(mm), r4(ss)]
        res[sn] = {"n_50pct_lines": n, "over_rate": r4(m), "se": r4(se), "by_team_spread": fav}
    return res


def always_side(d: pd.DataFrame, T: pd.DataFrame, Smod: dict, sn: str, seasons, side: str) -> dict:
    E = d[(d.snap == sn) & d.book.isin(ALLOWED) & d.season.isin(seasons)].copy()
    E = close_value(E, d, T, Smod)
    E["p_mkt"] = E.p_nv
    b = sides(E, "p_mkt", ["spread_line", "is_home"])
    b = b[b.side == side]
    # best number for the side: over -> lowest point then best price; under -> highest point then best price
    asc = side == "over"
    b = b.sort_values(["key", "point", "dec"], ascending=[True, asc, False]).drop_duplicates("key")
    return summarize(grade(b))


# =============================================================================================== DEV
W_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
T_GRID = (0.0, 0.02, 0.04, 0.06, 0.08, 0.12)


def run_market(market: str, seasons, want_grid: bool) -> tuple[dict, dict]:
    d, Smod = ou_table(market)
    T = pg_table(d, Smod)
    d = d.merge(T[["key"]], on="key")
    res = {"accuracy": accuracy(T, seasons), "biases": biases(T, d, seasons, market),
           "blend_ll": {sn: blend_fit(T, Smod, sn, seasons) for sn in ["open", "early"]},
           "coverage": {sn: int(T[T.season.isin(seasons)][f"L_{sn}"].notna().sum()) for sn in SNAPS}}
    res["always_under"] = {sn: always_side(d, T, Smod, sn, seasons, "under") for sn in ["open", "early"]}
    res["always_over"] = {sn: always_side(d, T, Smod, sn, seasons, "over") for sn in ["open", "early"]}
    grids = {}
    if want_grid:
        for sn in ["open", "early"]:
            for w in W_GRID:
                b = model_bets(d[d.season.isin(seasons)], T, Smod, sn, w)
                for side in ["both", "under", "over"]:
                    for t in T_GRID:
                        s = summarize(pick(b, t, side))
                        grids[f"model_{sn}_w{w}_t{t}_{side}"] = s
            b = shop_bets(d[d.season.isin(seasons)], T, Smod, sn)
            for side in ["both", "under", "over"]:
                for t in (0.0, 0.01, 0.02, 0.03, 0.05):
                    grids[f"shop_{sn}_t{t}_{side}"] = summarize(pick(b, t, side))
    return res, grids, (d, T, Smod)


def dev():
    out = {"markets": {}, "grids": {}, "moves": {}}
    for market in OU:
        res, grids, (d, T, Smod) = run_market(market, DEVS, True)
        res["moves_2023_insample"] = moves(T, DEVS, ())
        out["markets"][market] = res
        out["grids"][market] = grids
        print("=" * 30, market, json.dumps(res, default=float)[:3000], flush=True)
        good = {k: v for k, v in grids.items() if v.get("n", 0) >= 30 and (v.get("clv") or -1) > -0.005}
        for k, v in sorted(good.items(), key=lambda kv: -(kv[1]["clv"] or -9))[:25]:
            print(k, {kk: v[kk] for kk in ["n", "clv", "clv_t", "roi", "roi_se", "hit", "bets_per_week",
                                            "under_share", "direct_share"]})
    # anytime TD
    td = td_table()
    ab = {sn: td_calib_fit(td, DEVS, f"p_med_{sn}") for sn in SNAPS}
    out["td"] = {"calib_2023": ab, "analysis": td_analysis(td, DEVS, ab)}
    g = {}
    for sn in ["open", "early"]:
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            E = td_bets(td[td.season.isin(DEVS)], sn, ab["close"], ab[sn], w, "model")
            for t in (0.0, 0.03, 0.05, 0.1, 0.2):
                g[f"td_model_{sn}_w{w}_t{t}"] = summarize(pick(E, t, "both", ev_max=2.0))
        E = td_bets(td[td.season.isin(DEVS)], sn, ab["close"], ab[sn], 0, "shop")
        for t in (0.0, 0.03, 0.05, 0.1, 0.2):
            g[f"td_shop_{sn}_t{t}"] = summarize(pick(E, t, "both", ev_max=2.0))
    out["grids"]["td"] = g
    print("=" * 30, "TD", json.dumps(out["td"], default=float))
    for k, v in g.items():
        if v.get("n", 0) >= 20:
            print(k, {kk: v[kk] for kk in ["n", "clv", "clv_t", "roi", "roi_se", "hit", "bets_per_week", "mean_ev"]})
    json.dump(out, open(SCR / "dev.json", "w"), indent=1, default=float)


# =========================================================================================== HOLDOUT
def rule_bets(r: dict, cache: dict) -> pd.DataFrame:
    m, sn = r["market"], r["snap"]
    if m == "td":
        td, ab = cache["td"]
        E = td_bets(td, sn, ab["close"], ab[sn], r.get("w", 0.0), r["kind"])
        return pick(E, r["thr"], "both", ev_max=r.get("ev_max", 2.0))
    out = []
    for mm in (m if isinstance(m, list) else [m]):
        d, T, Smod = cache[mm]
        b = model_bets(d, T, Smod, sn, r["w"]) if r["kind"] == "model" else shop_bets(d, T, Smod, sn)
        out.append(pick(b, r["thr"], r.get("side", "both"), ev_max=r.get("ev_max", 0.5)).assign(market=mm))
    return pd.concat(out, ignore_index=True)


def holdout():
    fz = json.load(open(OUT / "props_full_frozen.json"))
    k = len(fz["rules"])
    res = {"definitions": __doc__, "frozen": fz, "markets": {}, "rules": {}}
    cache = {}
    q = pd.read_parquet(SCR / "quotes.parquet")
    u = q.drop_duplicates(["event_id", "player", "market"])
    res["match"] = {f"{s}_{m}": g.status.value_counts().to_dict() for (s, m), g in u.groupby(["season", "market"])}
    res["events_by_snap"] = {f"{s}_{m}": g.groupby("snap").event_id.nunique().to_dict()
                             for (s, m), g in q.groupby(["season", "market"])}
    res["quotes_by_snap_book"] = {f"{s}_{sn}": g.book.value_counts().to_dict()
                                  for (s, sn), g in q[q.market != "td"].groupby(["season", "snap"])}
    for market in OU:
        mres = {}
        for lab, ss in [("2023", DEVS), ("2024_25", HOLDS)] + [(str(s), (s,)) for s in HOLDS]:
            r_, _, cache[market] = run_market(market, ss, False)
            mres[lab] = r_
        mres["moves"] = moves(cache[market][1], DEVS, HOLDS)
        mres["moves_fit2024_test2025"] = moves(cache[market][1], (2024,), (2025,))["regression"]
        res["markets"][market] = mres
        print(market, "done", flush=True)
    td = td_table()
    ab = {sn: td_calib_fit(td, DEVS, f"p_med_{sn}") for sn in SNAPS}
    cache["td"] = (td, ab)
    res["td"] = {"calib_2023": ab, "2023": td_analysis(td, DEVS, ab), "2024_25": td_analysis(td, HOLDS, ab),
                 "info_test": {sn: td_info_test(td, sn, DEVS, HOLDS, ab) for sn in SNAPS}}
    from scipy.stats import norm  # noqa: F401
    for r in fz["rules"]:
        b = rule_bets(r, cache)
        rr = {}
        for lab, ss in [("dev_2023", DEVS), ("holdout_2024_25", HOLDS), ("2024", (2024,)), ("2025", (2025,))]:
            rr[lab] = summarize(b[b.season.isin(ss)])
        h = rr["holdout_2024_25"]
        rr["pass"] = bool(h.get("n", 0) > 0 and h["clv"] is not None and h["clv"] > 0 and h["clv_p"] is not None
                          and h["clv_p"] < 0.05 / k and (rr["2024"].get("clv") or -1) > 0
                          and (rr["2025"].get("clv") or -1) > 0)
        res["rules"][r["name"]] = rr
        b.to_csv(SCR / f"holdout_bets_{r['name']}.csv", index=False)
        print(r["name"], json.dumps(rr, default=float), flush=True)
    json.dump(res, open(OUT / "props_full.json", "w"), indent=1, default=float)


def build():
    PROJ.mkdir(parents=True, exist_ok=True)
    PR.build()


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    {"build": build, "fit": fit, "prep": prep, "dev": dev, "holdout": holdout}[sys.argv[1]]()
