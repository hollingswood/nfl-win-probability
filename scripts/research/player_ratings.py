"""Player-level value model (RAPM on EPA/play) -> injury-aware team strength features.

Research script (not a model input). Question: can individual player ratings price an absence (and its
replacement) better than the production injury feature (Out/Doubtful x prior snap share), and does that
show up (a) in walk-forward log loss and (b) in early-week -> close line moves / CLV on injury games?

Method
  1. Plays 2016-2025: nflverse pbp (pass/rush, EPA) joined to nflverse `pbp_participation` (the 22 gsis ids
     on the field). Participation is only published from 2016, so features are 0 before 2016.
  2. RAPM: weighted ridge  EPA_play = sum(offense player effects) + sum(defense player effects)
     + home + on-field rookie counts.  Refit before EVERY (season, week) using only plays from games
     with gameday < first gameday of that week (3-season window, exponential time decay). Penalty and
     decay tuned on out-of-sample next-4-weeks play EPA in 2017-2019 only.
  3. Roles: each player's EWMA on-field share over his own prior appearances (half-life 4 appearances).
  4. Team-game features (non-QB; QBs are already handled by the production qb_* features):
       miss_*   = sum over Out (1.0) / Doubtful (0.8) players of  w * role * (rating - top-backup rating)
       lineup_* = expected available unit strength today (role-weighted ratings of likely-available
                  players, capped at 10 offense / 11 defense slots, gaps filled at replacement level)
       delta_*  = lineup today - EWMA of the lineups actually fielded in the team's last 8 games
                  (what the team EPA features already "saw"), all valued with today's ratings.
     Units: points per game (EPA/play x 62 plays). Diffs are signed so positive favors home.
  5. Evaluate: walk-forward ridge exactly as scripts/experiment.py (val 2015-19, holdout 2020-25);
     line move first-snapshot -> price-implied close and ATS residual vs close (2020-25 snapshots);
     price-based CLV of a rule frozen on 2020-22 and run once on 2023-25; leakage masking test.

Run:  cd /home/claude/nfl && EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts \
          python scripts/research/player_ratings.py [stage]
      stages: prep | tune | ratings | features | model | market | leak | report | all (default)
Cache dir: $PR_CACHE (participation downloads, plays, ratings). Default: system temp dir / nfl_player_ratings.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "research"
CACHE = Path(os.environ.get("PR_CACHE", Path(tempfile.gettempdir()) / "nfl_player_ratings"))
CACHE.mkdir(parents=True, exist_ok=True)
PART_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{}.parquet"
SEASONS = range(2016, 2026)
TM = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
STATUS_W = {"Out": 1.0, "Doubtful": 0.8}
PLAYS_PER_GAME = 62.0       # EPA/play -> points per game
EPA_CLIP = 4.0
WINDOW_DAYS = 3 * 365
ROLE_HALFLIFE = 4           # appearances (same as production injuries.py)
FIELDED_LAGS, FIELDED_HL = 8, 8.0
# tuned on 2017-19 (see `tune`); defaults overwritten by CACHE/tuned.json when present
PARAMS = {"lam_off": 3000.0, "lam_def": 3000.0, "hl_days": 365.0}

# ============================================================================ data
def _participation(season: int) -> pd.DataFrame:
    p = CACHE / f"pbp_participation_{season}.parquet"
    if not p.exists():
        urllib.request.urlretrieve(PART_URL.format(season), p)
    q = pd.read_parquet(p, columns=["nflverse_game_id", "play_id", "offense_players", "defense_players"])
    q["play_id"] = q["play_id"].astype(float)
    return q.rename(columns={"nflverse_game_id": "game_id"})


def schedule() -> pd.DataFrame:
    g = pd.read_parquet(RAW / "games.parquet")
    g["gameday"] = pd.to_datetime(g["gameday"])
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TM)
    return g


def prep():
    """plays.parquet (one row per scrimmage play), onfield.parquet (row, pid, side)."""
    if (CACHE / "plays.parquet").exists():
        return
    g = schedule()[["game_id", "gameday"]]
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam", "home_team", "pass", "rush",
            "qb_kneel", "qb_spike", "epa"]
    plays, on = [], []
    for s in SEASONS:
        p = pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=cols)
        p = p[(p["pass"].eq(1) | p["rush"].eq(1)) & p["epa"].notna() & p["qb_kneel"].ne(1) & p["qb_spike"].ne(1)]
        p = p.merge(_participation(s), on=["game_id", "play_id"], how="inner")
        p = p[p.offense_players.fillna("").str.len().gt(0) & p.defense_players.fillna("").str.len().gt(0)]
        plays.append(p)
        print(s, len(p), flush=True)
    p = pd.concat(plays, ignore_index=True).merge(g, on="game_id", how="inner")
    for c in ("posteam", "defteam", "home_team"):
        p[c] = p[c].replace(TM)
    p["home"] = (p.posteam == p.home_team).astype(np.int8)
    p["row"] = np.arange(len(p), dtype=np.int32)
    for side, col in (("off", "offense_players"), ("def", "defense_players")):
        x = p[["row", col]].assign(pid=p[col].str.split(";")).explode("pid")[["row", "pid"]]
        x = x[x.pid.notna() & x.pid.ne("")]
        x["side"] = side
        on.append(x)
    on = pd.concat(on, ignore_index=True)
    # drop plays with implausible counts
    n = on.groupby(["row", "side"]).size().unstack(fill_value=0)
    ok = n.index[(n["off"].between(10, 11)) & (n["def"].between(10, 11))]
    p = p[p.row.isin(ok)]
    on = on[on.row.isin(ok)]
    p[["row", "game_id", "play_id", "season", "week", "gameday", "posteam", "defteam", "home", "epa"]].to_parquet(
        CACHE / "plays.parquet")
    on.to_parquet(CACHE / "onfield.parquet")


def load_prep():
    p = pd.read_parquet(CACHE / "plays.parquet")
    on = pd.read_parquet(CACHE / "onfield.parquet")
    players = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "position_group", "rookie_season",
                                                                "draft_year", "display_name"])
    return p, on, players


# ============================================================================ RAPM design
class Design:
    """Global sparse design over all plays: [off players | def players | home, off_rookies, def_rookies]."""
    N_COV = 3

    def __init__(self, plays, on, players):
        self.plays = plays.sort_values("row").reset_index(drop=True)
        self.plays["r"] = np.arange(len(self.plays))
        rmap = pd.Series(self.plays.r.values, index=self.plays.row.values)
        ids = np.sort(on.pid.unique())
        self.ids = ids
        self.P = len(ids)
        pidx = pd.Series(np.arange(self.P), index=ids)
        r = rmap.reindex(on.row.values).values
        c = pidx.reindex(on.pid.values).values + np.where(on.side.values == "def", self.P, 0)
        # rookie counts
        rs = players.set_index("gsis_id")["rookie_season"].combine_first(players.set_index("gsis_id")["draft_year"])
        rs = rs[~rs.index.duplicated()]
        season = self.plays.season.values[r]
        rook = (rs.reindex(on.pid.values).values == season).astype(float)
        nr = len(self.plays)
        roff = np.bincount(r[on.side.values == "off"], weights=rook[on.side.values == "off"], minlength=nr)
        rdef = np.bincount(r[on.side.values == "def"], weights=rook[on.side.values == "def"], minlength=nr)
        cov = np.column_stack([self.plays.home.values.astype(float), roff, rdef])
        Xp = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(nr, 2 * self.P))
        self.X = sp.hstack([Xp, sp.csr_matrix(cov)], format="csr")
        self.y = self.plays.epa.clip(-EPA_CLIP, EPA_CLIP).values
        self.day = self.plays.gameday.values.astype("datetime64[D]").astype(np.int64)

    def fit(self, date, lam_off, lam_def, hl_days, x0=None, team_mode=False):
        d = np.datetime64(pd.Timestamp(date), "D").astype(np.int64)
        m = (self.day < d) & (self.day >= d - WINDOW_DAYS)
        if m.sum() < 1000:
            return None
        rows = np.flatnonzero(m)
        w = 0.5 ** ((d - self.day[rows]) / hl_days)
        X = self.X[rows]
        y = self.y[rows]
        ybar = np.average(y, weights=w)
        sw = np.sqrt(w)
        # column scaling => per-block penalties with a single lsqr damp
        scale = np.concatenate([np.ones(self.P), np.full(self.P, math.sqrt(lam_off / lam_def)),
                                np.full(self.N_COV, 30.0)])
        Xs = sp.diags(sw) @ X @ sp.diags(scale)
        z0 = None if x0 is None else x0 / scale
        sol = lsqr(Xs, sw * (y - ybar), damp=math.sqrt(lam_off), atol=1e-7, btol=1e-7, iter_lim=400, x0=z0)[0]
        beta = sol * scale
        # weighted play counts per player in window (for replacement-level calc)
        nplay = np.asarray((sp.diags(w) @ X[:, :2 * self.P]).sum(axis=0)).ravel()
        return {"beta": beta, "ybar": ybar, "nplay": nplay}

    def predict_rows(self, rows, fit):
        return self.X[rows] @ fit["beta"] + fit["ybar"]


def team_baseline_r2(dz: Design, date, horizon=28, lam=300.0, hl=365.0):
    """Comparator: same ridge with team (posteam/defteam) indicators instead of players."""
    d = np.datetime64(pd.Timestamp(date), "D").astype(np.int64)
    pl = dz.plays
    teams = np.sort(pd.unique(np.concatenate([pl.posteam.values, pl.defteam.values])))
    ti = pd.Series(np.arange(len(teams)), index=teams)
    n = len(pl)
    X = sp.csr_matrix((np.ones(2 * n), (np.r_[np.arange(n), np.arange(n)],
                                         np.r_[ti[pl.posteam].values, len(teams) + ti[pl.defteam].values])),
                      shape=(n, 2 * len(teams)))
    X = sp.hstack([X, sp.csr_matrix(pl.home.values.astype(float)[:, None])], format="csr")
    m = (dz.day < d) & (dz.day >= d - WINDOW_DAYS)
    rows = np.flatnonzero(m)
    w = 0.5 ** ((d - dz.day[rows]) / hl)
    ybar = np.average(dz.y[rows], weights=w)
    sw = np.sqrt(w)
    beta = lsqr(sp.diags(sw) @ X[rows], sw * (dz.y[rows] - ybar), damp=math.sqrt(lam), atol=1e-7, btol=1e-7)[0]
    te = np.flatnonzero((dz.day >= d) & (dz.day < d + horizon))
    return X[te] @ beta + ybar, te


# ============================================================================ tuning (2017-2019 only)
def week_starts(g=None) -> pd.DataFrame:
    g = schedule() if g is None else g
    w = g[g.season.isin(SEASONS)].groupby(["season", "week"]).gameday.min().reset_index()
    return w.rename(columns={"gameday": "start"})


def tune():
    plays, on, players = load_prep()
    dz = Design(plays, on, players)
    ws = week_starts()
    pts = ws[ws.season.isin([2017, 2018, 2019]) & ws.week.isin([3, 7, 11, 15])].start.tolist()
    grid = [(l, l, h) for l in (300.0, 1000.0, 3000.0, 8000.0, 20000.0) for h in (365.0, 730.0)]
    grid += [(1000.0, 3000.0, 365.0), (3000.0, 1000.0, 365.0), (1000.0, 8000.0, 365.0),
             (1000.0, 20000.0, 365.0), (2000.0, 8000.0, 365.0), (500.0, 8000.0, 365.0), (2000.0, 20000.0, 365.0)]
    res = []
    # baselines: mean-only and team-ridge
    sse0 = n = 0.0
    sse_t = {lt: 0.0 for lt in (30.0, 100.0, 300.0, 1000.0)}
    for d in pts:
        ybar = dz.fit(d, 1e9, 1e9, 365.0)["ybar"]
        for lt in sse_t:
            pt, te = team_baseline_r2(dz, d, lam=lt)
            sse_t[lt] += ((dz.y[te] - pt) ** 2).sum()
        sse0 += ((dz.y[te] - ybar) ** 2).sum()
        n += len(te)
    for lo, ld, h in grid:
        sse = 0.0
        for d in pts:
            f = dz.fit(d, lo, ld, h)
            dd = np.datetime64(pd.Timestamp(d), "D").astype(np.int64)
            te = np.flatnonzero((dz.day >= dd) & (dz.day < dd + 28))
            sse += ((dz.y[te] - dz.predict_rows(te, f)) ** 2).sum()
        r = {"lam_off": lo, "lam_def": ld, "hl_days": h, "oos_r2_pct": round(100 * (1 - sse / sse0), 3),
             "mse": sse / n}
        print(r, flush=True)
        res.append(r)
    best = min(res, key=lambda r: r["mse"])
    out = {"grid": res, "best": best, "team_ridge_oos_r2_pct": {str(k): round(100 * (1 - v / sse0), 3) for k, v in sse_t.items()},
           "n_test_plays": int(n), "fit_dates": [str(pd.Timestamp(d).date()) for d in pts]}
    json.dump(out, open(CACHE / "tuned.json", "w"), indent=1)
    return out


def params():
    p = CACHE / "tuned.json"
    if p.exists():
        b = json.load(open(p))["best"]
        return {k: b[k] for k in ("lam_off", "lam_def", "hl_days")}
    return PARAMS


# ============================================================================ rolling ratings
def rolling_ratings(dz: Design, weeks: pd.DataFrame, prm: dict) -> pd.DataFrame:
    """One rating set per (season, week), fit on plays strictly before that week's first gameday.
    Returns long frame: season, week, pid, side, rating (EPA/play; defense sign = EPA allowed), nplay."""
    out, x0 = [], None
    t0 = time.time()
    for i, r in enumerate(weeks.itertuples()):
        # NB: no warm start. scipy's lsqr damps the CORRECTION to x0, so x0 would silently change the
        # ridge prior to "previous week's ratings" (caught by the leakage/reproducibility test).
        f = dz.fit(r.start, prm["lam_off"], prm["lam_def"], prm["hl_days"])
        if f is None:
            continue
        b, n = f["beta"][:2 * dz.P], f["nplay"]
        keep = np.flatnonzero(n > 0)
        side = np.where(keep < dz.P, "off", "def")
        pid = dz.ids[keep % dz.P]
        df = pd.DataFrame({"season": r.season, "week": r.week, "pid": pid, "side": side,
                           "rating": b[keep].astype(np.float32), "nplay": n[keep].astype(np.float32)})
        out.append(df)
        if i % 20 == 0:
            print(f"  ratings {r.season}-{r.week:02d} {time.time() - t0:.0f}s", flush=True)
    return pd.concat(out, ignore_index=True)


# ============================================================================ roles & appearances
def appearances(plays, on) -> pd.DataFrame:
    """(game_id, team, pid, side, share, gameday): on-field share of the team's scrimmage plays."""
    x = on.merge(plays[["row", "game_id", "gameday", "posteam", "defteam"]], on="row")
    x["team"] = np.where(x.side == "off", x.posteam, x.defteam)
    a = x.groupby(["game_id", "gameday", "team", "side", "pid"]).size().rename("n").reset_index()
    tot = plays.groupby(["game_id", "posteam"]).size().rename("tot_off")
    tod = plays.groupby(["game_id", "defteam"]).size().rename("tot_def")
    a = a.merge(tot, left_on=["game_id", "team"], right_index=True, how="left") \
         .merge(tod, left_on=["game_id", "team"], right_index=True, how="left")
    a["share"] = np.where(a.side == "off", a.n / a.tot_off, a.n / a.tot_def)
    return a[["game_id", "gameday", "team", "side", "pid", "share"]]


def player_roles(app: pd.DataFrame) -> pd.DataFrame:
    """Post-game EWMA role per (pid, side) over his own appearances; as-of joined strictly before a date."""
    a = app.sort_values(["pid", "side", "gameday"]).copy()
    a["role"] = a.groupby(["pid", "side"])["share"].transform(lambda s: s.ewm(halflife=ROLE_HALFLIFE).mean())
    return a[["pid", "side", "gameday", "role"]]


# ============================================================================ team-game features
def _long_games(g):
    s = g[g.season.isin(SEASONS)]
    h = s[["game_id", "season", "week", "gameday", "home_team"]].rename(columns={"home_team": "team"})
    a = s[["game_id", "season", "week", "gameday", "away_team"]].rename(columns={"away_team": "team"})
    lg = pd.concat([h.assign(is_home=1), a.assign(is_home=0)], ignore_index=True)
    lg = lg.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    lg["k"] = lg.groupby("team").cumcount()
    return lg


def team_features(g, plays, on, players, ratings, injuries, target_ids=None) -> pd.DataFrame:
    """Per (game_id, team): miss_off/def, lineup_off/def, delta_off/def (points/game, + = good for team)."""
    lg = _long_games(g)
    app = appearances(plays, on).merge(lg[["game_id", "team", "k"]], on=["game_id", "team"])
    roles = player_roles(app)
    pos = players.drop_duplicates("gsis_id").set_index("gsis_id")["position_group"]
    tgt = lg if target_ids is None else lg[lg.game_id.isin(target_ids)]
    tgt = tgt.copy()

    # --- ratings as-of target week (value: offense = rating, defense = -rating), replacement level per unit
    rt = ratings.copy()
    rt["val"] = np.where(rt.side == "off", rt.rating, -rt.rating)
    repl = rt[rt.nplay < 150].groupby(["season", "week", "side"]).val.mean().rename("repl").reset_index()

    # --- injury report for the target week (final status; pre-game info)
    inj = injuries.copy()
    inj["team"] = inj["team"].replace(TM)
    inj = inj.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    inj = inj[["season", "week", "team", "gsis_id", "report_status"]].rename(columns={"gsis_id": "pid"})

    # --- candidate players for each target team-game: appeared for team in its last 8 games
    cand = []
    for lag in range(1, 9):
        c = tgt[["game_id", "team", "k", "season", "week", "gameday"]].assign(kk=lambda x: x.k - lag)
        c = c.merge(app[["team", "k", "pid", "side", "share"]].rename(columns={"k": "kk"}), on=["team", "kk"])
        c["lag"] = lag
        cand.append(c)
    cand = pd.concat(cand, ignore_index=True)
    fielded = cand.copy()  # used for delta (lineups actually fielded)
    cand = cand.sort_values("lag").drop_duplicates(["game_id", "team", "pid", "side"])  # most recent lag
    cand = cand.merge(inj, on=["season", "week", "team", "pid"], how="left")
    # role as of target date (strictly earlier appearances)
    cand = pd.merge_asof(cand.sort_values("gameday"), roles.sort_values("gameday"), on="gameday",
                         by=["pid", "side"], allow_exact_matches=False)
    cand = cand.merge(rt[["season", "week", "pid", "side", "val"]], on=["season", "week", "pid", "side"], how="left")
    cand = cand.merge(repl, on=["season", "week", "side"], how="left")
    cand["val"] = cand["val"].fillna(cand["repl"])
    cand["pos"] = cand.pid.map(pos).fillna("UNK")
    cand = cand[cand.pos != "QB"]
    cand["w_out"] = cand.report_status.map(STATUS_W).fillna(0.0)
    # availability: played in one of the last 2 games, or listed on this week's report (not Out/Doubtful)
    listed_ok = cand.report_status.notna() | False
    cand["avail"] = ((cand.lag <= 2) | (cand.report_status.isin(["Questionable"]) & listed_ok)) * (1 - cand.w_out)

    # injured players not seen in the last 8 games are irrelevant here (their absence is already in team stats)
    out_rows = cand[cand.w_out > 0].copy()
    # top backup: highest-role available same-pos teammate below the injured player's role
    bk = cand[(cand.avail > 0)][["game_id", "team", "side", "pos", "pid", "role", "val"]]
    m = out_rows.merge(bk, on=["game_id", "team", "side", "pos"], suffixes=("", "_b"))
    m = m[(m.pid_b != m.pid) & (m.role_b < m.role)]
    m = m.sort_values("role_b", ascending=False).drop_duplicates(["game_id", "team", "side", "pid"])
    out_rows = out_rows.merge(m[["game_id", "team", "side", "pid", "val_b"]], on=["game_id", "team", "side", "pid"],
                              how="left")
    out_rows["val_b"] = out_rows["val_b"].fillna(out_rows["repl"])
    out_rows["miss"] = out_rows.w_out * out_rows.role.fillna(0) * (out_rows.val - out_rows.val_b)
    out_rows["miss_raw"] = out_rows.w_out * out_rows.role.fillna(0) * out_rows.val
    out_rows["miss_repl"] = out_rows.w_out * out_rows.role.fillna(0) * (out_rows.val - out_rows.repl)
    out_rows["miss_role"] = out_rows.w_out * out_rows.role.fillna(0)
    out_rows["is_starter"] = (out_rows.role.fillna(0) >= 0.5).astype(int)
    # fresh absences only: played in the team's last game, Out/Doubtful now (team EPA stats haven't seen it yet)
    fresh = (out_rows.lag == 1).astype(float)
    out_rows["miss_new"] = fresh * out_rows.miss_repl
    out_rows["miss_newrole"] = fresh * out_rows.miss_role

    # lineup today: role-weighted value of available players, capped to unit slots, gap at replacement
    slots = {"off": 10.0, "def": 11.0}
    av = cand[cand.avail > 0].copy()
    av["rw"] = av.avail * av.role.fillna(0)
    lu = av.groupby(["game_id", "team", "side"]).apply(
        lambda d: pd.Series({"rsum": d.rw.sum(), "vsum": (d.rw * d.val).sum(), "repl": d.repl.iloc[0]}),
        include_groups=False).reset_index()
    cap = lu.side.map(slots)
    sc = np.minimum(1.0, cap / lu.rsum.clip(lower=1e-9))
    lu["lineup"] = lu.vsum * sc + (cap - lu.rsum * sc) * lu.repl.fillna(0)

    # fielded lineups over last 8 games (actual shares), valued with TODAY's ratings
    fd = fielded.merge(rt[["season", "week", "pid", "side", "val"]], on=["season", "week", "pid", "side"], how="left")
    fd = fd.merge(repl, on=["season", "week", "side"], how="left")
    fd["val"] = fd.val.fillna(fd.repl)
    fd["pos"] = fd.pid.map(pos).fillna("UNK")
    fd = fd[fd.pos != "QB"]
    fg = fd.assign(v=fd.share * fd.val).groupby(["game_id", "team", "side", "lag"]).v.sum().reset_index()
    fg["wt"] = 0.5 ** ((fg.lag - 1) / FIELDED_HL)
    fe = fg.groupby(["game_id", "team", "side"]).apply(lambda d: np.average(d.v, weights=d.wt),
                                                       include_groups=False).rename("fielded").reset_index()

    res = tgt[["game_id", "team", "season", "week", "gameday", "is_home"]].copy()
    for side in ("off", "def"):
        o = out_rows[out_rows.side == side].groupby(["game_id", "team"]).agg(
            **{f"miss_{side}": ("miss", "sum"), f"missraw_{side}": ("miss_raw", "sum"),
               f"missrepl_{side}": ("miss_repl", "sum"),
               f"missnew_{side}": ("miss_new", "sum"), f"missnewrole_{side}": ("miss_newrole", "sum"),
               f"missrole_{side}": ("miss_role", "sum"), f"nstart_out_{side}": ("is_starter", "sum")})
        res = res.merge(o, on=["game_id", "team"], how="left")
        l = lu[lu.side == side][["game_id", "team", "lineup"]].rename(columns={"lineup": f"lineup_{side}"})
        f = fe[fe.side == side][["game_id", "team", "fielded"]].rename(columns={"fielded": f"fielded_{side}"})
        res = res.merge(l, on=["game_id", "team"], how="left").merge(f, on=["game_id", "team"], how="left")
        res[f"delta_{side}"] = res[f"lineup_{side}"] - res[f"fielded_{side}"]
    num = [c for c in res.columns if c.split("_")[0] in ("miss", "missraw", "missrepl", "missnew", "lineup", "fielded", "delta")]
    res[num] = res[num] * PLAYS_PER_GAME  # -> points per game
    fill = [c for c in res.columns if c.split("_")[0] in ("miss", "missraw", "missrepl", "missnew", "missnewrole", "missrole", "nstart", "delta")]
    res[fill] = res[fill].fillna(0.0)
    return res


def game_features(tf: pd.DataFrame) -> pd.DataFrame:
    """Home-minus-away, signed so positive favors home."""
    h = tf[tf.is_home == 1].drop(columns=["is_home", "team"]).set_index("game_id")
    a = tf[tf.is_home == 0].drop(columns=["is_home", "team", "season", "week", "gameday"]).set_index("game_id")
    d = pd.DataFrame(index=h.index)
    for u in ("off", "def"):
        d[f"pr_miss_{u}_diff"] = a[f"miss_{u}"] - h[f"miss_{u}"]
        d[f"pr_missraw_{u}_diff"] = a[f"missraw_{u}"] - h[f"missraw_{u}"]
        d[f"pr_missrepl_{u}_diff"] = a[f"missrepl_{u}"] - h[f"missrepl_{u}"]
        d[f"pr_missnew_{u}_diff"] = a[f"missnew_{u}"] - h[f"missnew_{u}"]
        d[f"pr_missnewrole_{u}_diff"] = a[f"missnewrole_{u}"] - h[f"missnewrole_{u}"]
        d[f"pr_lineup_{u}_diff"] = h[f"lineup_{u}"] - a[f"lineup_{u}"]
        d[f"pr_delta_{u}_diff"] = h[f"delta_{u}"] - a[f"delta_{u}"]
        d[f"home_miss_{u}"], d[f"away_miss_{u}"] = h[f"miss_{u}"], a[f"miss_{u}"]
        d[f"home_nstart_out_{u}"], d[f"away_nstart_out_{u}"] = h[f"nstart_out_{u}"], a[f"nstart_out_{u}"]
    d["pr_miss_diff"] = d.pr_miss_off_diff + d.pr_miss_def_diff
    d["pr_missrepl_diff"] = d.pr_missrepl_off_diff + d.pr_missrepl_def_diff
    d["pr_missnew_diff"] = d.pr_missnew_off_diff + d.pr_missnew_def_diff
    d["pr_delta_diff"] = d.pr_delta_off_diff + d.pr_delta_def_diff
    d["pr_lineup_diff"] = d.pr_lineup_off_diff + d.pr_lineup_def_diff
    return d.reset_index()


def load_injuries():
    return pd.concat([pd.read_parquet(RAW / f"injuries_{s}.parquet") for s in SEASONS], ignore_index=True)


def build_all():
    p_r, p_f = CACHE / "ratings.parquet", CACHE / "game_feats.parquet"
    plays, on, players = load_prep()
    g = schedule()
    if not p_r.exists():
        dz = Design(plays, on, players)
        rolling_ratings(dz, week_starts(g), params()).to_parquet(p_r)
    if not p_f.exists():
        tf = team_features(g, plays, on, players, pd.read_parquet(p_r), load_injuries())
        tf.to_parquet(CACHE / "team_feats.parquet")
        game_features(tf).to_parquet(p_f)
    return pd.read_parquet(p_f)


# ============================================================================ (a) walk-forward model
FEATSETS = {
    "prod": [],
    "+miss": ["pr_miss_off_diff", "pr_miss_def_diff"],
    "+miss(net)": ["pr_miss_diff"],
    "+missraw": ["pr_missraw_off_diff", "pr_missraw_def_diff"],
    "+missrepl": ["pr_missrepl_off_diff", "pr_missrepl_def_diff"],
    "+missrepl(net)": ["pr_missrepl_diff"],
    "+missnew": ["pr_missnew_off_diff", "pr_missnew_def_diff"],
    "+missnew(net)": ["pr_missnew_diff"],
    "+missnewrole (fresh snap-share only)": ["pr_missnewrole_off_diff", "pr_missnewrole_def_diff"],
    "+delta": ["pr_delta_off_diff", "pr_delta_def_diff"],
    "+delta(net)": ["pr_delta_diff"],
    "+lineup": ["pr_lineup_off_diff", "pr_lineup_def_diff"],
    "+miss+delta": ["pr_miss_off_diff", "pr_miss_def_diff", "pr_delta_off_diff", "pr_delta_def_diff"],
    "swap inj->miss": ["pr_miss_off_diff", "pr_miss_def_diff", "-inj_off_diff", "-inj_def_diff"],
}


def model_eval(gf: pd.DataFrame) -> dict:
    import experiment as E
    from nflpred import model as M
    base, df = E.run("prod")
    df = df.merge(gf, on="game_id", how="left")
    cols = sorted({c for v in FEATSETS.values() for c in v if not c.startswith("-")})
    df[cols] = df[cols].fillna(0.0)
    res = {"harness_prod": base}

    def wf(feats, seasons, min_train=None):
        ps, ys = [], []
        for s in seasons:
            tr = M.train_rows(df, before_season=s)
            if min_train:
                tr = tr[tr.season >= min_train]
            te = df[(df.season == s) & df.home_win.notna()]
            ps.append(M.fit_predict(tr, te, feats)); ys.append(te.home_win.values)
        y, p = np.concatenate(ys), np.concatenate(ps)
        return M.score(y, p), p, y

    import experiment as E2  # noqa
    from nflpred.features import FEATURES
    rows = []
    for name, extra in FEATSETS.items():
        feats = [f for f in FEATURES if "-" + f not in extra] + [c for c in extra if not c.startswith("-")]
        r = {"set": name}
        for lab, seas in (("val", E.VAL), ("val1719", range(2017, 2020)), ("hold", E.HOLD)):
            sc, _, _ = wf(feats, seas)
            r[f"{lab}_ll"] = round(sc["log_loss"], 5)
            r[f"{lab}_brier"] = round(sc["brier"], 5)
        # sensitivity: train only on seasons with participation data
        sc, _, _ = wf(feats, E.HOLD, min_train=2016)
        r["hold_ll_train2016+"] = round(sc["log_loss"], 5)
        rows.append(r)
        print(r, flush=True)
    res["table"] = rows
    # paired bootstrap of holdout delta for the val-selected set (by game, 2020-25)
    best = min((r for r in rows if r["set"] != "prod"), key=lambda r: r["val_ll"])
    res["val_selected"] = best["set"]
    fb = [f for f in FEATURES if "-" + f not in FEATSETS[best["set"]]] + \
         [c for c in FEATSETS[best["set"]] if not c.startswith("-")]
    _, p0, y = wf(FEATURES, E.HOLD)
    _, p1, _ = wf(fb, E.HOLD)
    l0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    l1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    d = l1 - l0
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res["holdout_delta_selected"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                      float(np.percentile(bs, 97.5))]}
    # coefficients (points per unit) of the selected set fit on 2013-2025
    from sklearn.linear_model import Ridge
    tr = M.train_rows(df)
    X = tr[fb]; yv = (tr.home_score - tr.away_score).values
    rg = Ridge(alpha=M.RIDGE_ALPHA).fit((X - X.mean()) / X.std(), yv)
    res["coef_pts_per_unit_full"] = {f: float(c / s) for f, c, s in zip(fb, rg.coef_, X.std()) if f.startswith("pr_")
                                     or f.startswith("inj_")}
    df[["game_id", "season"] + cols].to_parquet(CACHE / "model_df_feats.parquet")
    return res


CLV_SIGNALS = ["prod_inj_net", "pr_miss_diff", "pr_missnew_diff"]


# ============================================================================ (b) market: line moves, ATS vs close, CLV
def _ols(y, X, names):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(x, float) for x in X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    # HC1 robust SE
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    return {nm: {"b": round(float(bb), 4), "se": round(float(s), 4), "t": round(float(bb / s), 2)}
            for nm, bb, s in zip(["const"] + names, b, se)} | {"n": int(n), "r2": round(1 - e.var() / y.var(), 4)}


def market_eval(gf: pd.DataFrame) -> dict:
    import edge_lab as EL
    from nflpred import spread_bets as SB
    import experiment as E  # production inj features for comparison
    _, df = E.run("prod")
    df = df[["game_id", "inj_off_diff", "inj_def_diff", "home_score", "away_score", "season"]].merge(gf, on="game_id")
    cf = EL.closing_fair()
    out = {}
    rules = SB.load_rules()
    tables = {"dev": EL.load(False)}
    if os.environ.get("EDGE_HOLDOUT") == "I_HAVE_FROZEN_CANDIDATES":
        tables["holdout"] = EL.load(True)
    frozen = None
    for lab, t in tables.items():
        gm = t.drop_duplicates("game_id")[["game_id", "m_first", "home_score", "away_score"]]
        x = gm.merge(cf[["game_id", "mu_close_all", "mu_close_sharp"]], on="game_id") \
              .merge(df.drop(columns=["home_score", "away_score"]), on="game_id")
        x = x[x.m_first.notna() & x.mu_close_all.notna()]
        x["move"] = x.mu_close_all - x.m_first
        x["ats_resid"] = (x.home_score - x.away_score) - x.mu_close_all
        x["inj_game"] = ((x.home_nstart_out_off + x.home_nstart_out_def + x.away_nstart_out_off
                          + x.away_nstart_out_def) > 0)
        r = {"n_games": len(x), "n_injury_games": int(x.inj_game.sum())}
        for sub, xs in (("all", x), ("injury_games", x[x.inj_game])):
            r[sub] = {
                "move~prod_inj": _ols(xs.move.values, [xs.inj_off_diff, xs.inj_def_diff], ["inj_off", "inj_def"]),
                "move~miss": _ols(xs.move.values, [xs.pr_miss_off_diff, xs.pr_miss_def_diff], ["miss_off", "miss_def"]),
                "move~prod_inj+miss": _ols(xs.move.values, [xs.inj_off_diff, xs.inj_def_diff, xs.pr_miss_diff],
                                           ["inj_off", "inj_def", "miss_net"]),
                "move~missnew": _ols(xs.move.values, [xs.pr_missnew_off_diff, xs.pr_missnew_def_diff],
                                     ["missnew_off", "missnew_def"]),
                "move~missnewrole": _ols(xs.move.values, [xs.pr_missnewrole_off_diff, xs.pr_missnewrole_def_diff],
                                         ["newrole_off", "newrole_def"]),
                "move~missnewrole+missnew": _ols(xs.move.values, [xs.pr_missnewrole_off_diff + xs.pr_missnewrole_def_diff,
                                                                  xs.pr_missnew_diff], ["newrole_net", "missnew_net"]),
                "ats_resid~missnew+newrole": _ols(xs.ats_resid.values, [xs.pr_missnew_diff,
                                                  xs.pr_missnewrole_off_diff + xs.pr_missnewrole_def_diff],
                                                  ["missnew_net", "newrole_net"]),
                "move~delta": _ols(xs.move.values, [xs.pr_delta_off_diff, xs.pr_delta_def_diff], ["delta_off", "delta_def"]),
                "ats_resid~miss+delta": _ols(xs.ats_resid.values, [xs.pr_miss_diff, xs.pr_delta_diff],
                                             ["miss_net", "delta_net"]),
                "ats_resid~prod_inj": _ols(xs.ats_resid.values, [xs.inj_off_diff, xs.inj_def_diff], ["inj_off", "inj_def"]),
            }
        # ---- CLV rules (frozen on dev): bet AGAINST the more-injured team when |signal| >= dev 90th pct
        x["prod_inj_net"] = x.inj_off_diff + x.inj_def_diff
        if lab == "dev":
            frozen = {"signals": {k: float(x[k].abs().quantile(0.9)) for k in CLV_SIGNALS},
                      "rule": "|signal| >= dev 90th percentile -> take the side the signal favors (spread)",
                      "timings": {"first": "first snapshot <=9 days out (uses the FINAL report: hindsight)",
                                  "eligible": "first snapshot after the final injury report (honest timing)"}}
        clv = {}
        for sig, thr in frozen["signals"].items():
            bets = x[x[sig].abs() >= thr][["game_id", sig, "mu_close_all"]].rename(columns={sig: "sig"})
            for timing in ("first", "eligible"):
                tt = t[t.game_id.isin(bets.game_id) & t.point.notna() & t.sp_price.notna() & (t.hours_before <= 216)]
                if timing == "eligible":
                    tt = tt[tt.eligible]
                snap = tt.groupby("game_id").requested_ts.transform("min")
                tt = tt[tt.requested_ts == snap].merge(bets, on="game_id")
                tt = tt[(tt.side == "home") == (tt.sig > 0)]
                tt["clv"] = [EL.spread_clv_price(mu, pt, int(pr), sd) for mu, pt, pr, sd in
                             zip(tt.mu_close_all, tt.point, tt.sp_price, tt.side)]
                best = tt.sort_values("clv", ascending=False).drop_duplicates("game_id")
                c = best.clv.dropna().values
                mv = (np.sign(best.sig) * (best.mu_close_all - np.where(best.side == "home", -best.point, best.point)))
                clv[f"{sig}|{timing}"] = {
                    "bets": len(c), "mean_clv": round(float(c.mean()), 4) if len(c) else None,
                    "t": round(float(c.mean() / (c.std(ddof=1) / math.sqrt(len(c)))), 2) if len(c) > 2 else None,
                    "beat_close": round(float((c > 0).mean()), 3) if len(c) else None,
                    "mean_pts_moved_our_way": round(float(mv.mean()), 2) if len(c) else None,
                    "median_hours_before": round(float(best.hours_before.median()), 1) if len(c) else None}
        # baseline: average best-book CLV of betting either side of every game at the same snapshot
        for timing in ("first", "eligible"):
            tt = t[t.game_id.isin(x.game_id) & t.point.notna() & t.sp_price.notna() & (t.hours_before <= 216)]
            if timing == "eligible":
                tt = tt[tt.eligible]
            tt = tt[tt.requested_ts == tt.groupby("game_id").requested_ts.transform("min")]
            tt = tt.merge(x[["game_id", "mu_close_all"]], on="game_id")
            tt["clv"] = [EL.spread_clv_price(mu, pt, int(pr), sd) for mu, pt, pr, sd in
                         zip(tt.mu_close_all, tt.point, tt.sp_price, tt.side)]
            b = tt.sort_values("clv", ascending=False).drop_duplicates(["game_id", "side"])
            clv[f"baseline_any_side|{timing}"] = {"bets": len(b), "mean_clv": round(float(b.clv.mean()), 4)}
        r["clv_rule"] = clv
        out[lab] = r
    out["frozen_rule"] = frozen
    return out


# ============================================================================ (c) leakage masking test
def leak_test(cutoffs=("2021-11-14", "2024-10-06")) -> dict:
    """Erase every play from `cutoff` onward (and later weeks' injury reports), refit the ratings for the
    cutoff week, recompute features for games ON the cutoff date and compare with the full run."""
    plays, on, players = load_prep()
    g = schedule()
    full_r = pd.read_parquet(CACHE / "ratings.parquet")
    inj = load_injuries()
    prm = params()
    full = pd.read_parquet(CACHE / "team_feats.parquet")
    res = {}
    for cut in cutoffs:
        c = pd.Timestamp(cut)
        ids = g.loc[g.gameday == c, "game_id"].tolist()
        wk = g.loc[g.gameday == c, ["season", "week"]].drop_duplicates()
        p2 = plays[plays.gameday < c]
        on2 = on[on.row.isin(p2.row)]
        g2 = g.copy()
        fut = g2.gameday >= c
        g2.loc[fut, ["home_score", "away_score", "result"]] = np.nan
        later = set(map(tuple, g2.loc[g2.gameday > c, ["season", "week"]].drop_duplicates().values)) - \
            set(map(tuple, wk.values))
        inj2 = inj[~inj[["season", "week"]].apply(tuple, axis=1).isin(later)]
        dz2 = Design(p2, on2, players)
        ws = week_starts(g2).merge(wk, on=["season", "week"])
        r2 = rolling_ratings(dz2, ws, prm)
        # ratings for the week must match the full run exactly (up to solver tolerance)
        a = full_r.merge(wk, on=["season", "week"]).set_index(["pid", "side"]).rating
        b = r2.set_index(["pid", "side"]).rating
        rat_diff = float((a - b.reindex(a.index)).abs().max())
        # features for the cutoff games: other weeks' ratings may be needed only for lags (not used) -> pass both
        tf2 = team_features(g2, p2, on2, players, pd.concat([full_r[full_r.set_index(["season", "week"]).index
                                                                     .isin(list(map(tuple, wk.values))) == False],
                                                             r2]), inj2, target_ids=ids)
        cols = ["miss_off", "miss_def", "lineup_off", "lineup_def", "delta_off", "delta_def"]
        f1 = full[full.game_id.isin(ids)].set_index(["game_id", "team"])[cols].sort_index()
        f2 = tf2.set_index(["game_id", "team"])[cols].sort_index()
        md = float((f1 - f2.reindex(f1.index)).abs().max().max())
        res[cut] = {"games": len(ids), "max_abs_rating_diff": rat_diff, "max_abs_feature_diff_pts": md,
                    "pass": bool(md < 1e-3)}
        print(cut, res[cut], flush=True)
    # negative control: a rating fit that is allowed to see the cutoff day's plays must differ
    c = pd.Timestamp(cutoffs[0])
    dz = Design(plays, on, players)
    f_ok = dz.fit(c, prm["lam_off"], prm["lam_def"], prm["hl_days"])
    f_leak = dz.fit(c + pd.Timedelta(days=1), prm["lam_off"], prm["lam_def"], prm["hl_days"])
    res["negative_control_max_rating_change_if_cutoff_day_included"] = float(np.abs(f_ok["beta"] - f_leak["beta"]).max())
    return res


# ============================================================================ report helpers
def top_players(n=12):
    r = pd.read_parquet(CACHE / "ratings.parquet")
    players = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "display_name", "position_group"])
    last = r[(r.season == 2025) & (r.week == r[r.season == 2025].week.max())]
    last = last.merge(players.drop_duplicates("gsis_id"), left_on="pid", right_on="gsis_id")
    last["pts_per_game"] = np.where(last.side == "off", last.rating, -last.rating) * PLAYS_PER_GAME
    out = {}
    for side in ("off", "def"):
        d = last[(last.side == side) & (last.nplay > 400)].sort_values("pts_per_game", ascending=False)
        out[side] = d.head(n)[["display_name", "position_group", "pts_per_game", "nplay"]].round(2).to_dict("records")
    sd = last[last.nplay > 400].groupby(["side", "position_group"]).pts_per_game.agg(["std", "mean", "size"])
    out["pts_by_pos"] = {f"{a}_{b}": {k: round(float(v), 3) for k, v in r.items()} for (a, b), r in sd.iterrows()}
    return out


def main(stage="all"):
    res_path = CACHE / "results.json"
    res = json.load(open(res_path)) if res_path.exists() else {}
    if stage in ("prep", "all"):
        prep()
    if stage in ("tune", "all") and not (CACHE / "tuned.json").exists():
        tune()
    res["tuning"] = json.load(open(CACHE / "tuned.json")) if (CACHE / "tuned.json").exists() else None
    if stage in ("ratings", "features", "all"):
        build_all()
    if stage in ("model", "all"):
        res["model"] = model_eval(build_all())
    if stage in ("market", "all"):
        res["market"] = market_eval(build_all())
    if stage in ("leak", "all"):
        res["leakage"] = leak_test()
    if stage in ("report", "all"):
        res["top_players_2025_end"] = top_players()
        gf = build_all()
        res["feature_summary"] = {c: {"mean": round(float(gf[c].mean()), 3), "sd": round(float(gf[c].std()), 3)}
                                  for c in gf.columns if c.startswith("pr_")}
    json.dump(res, open(res_path, "w"), indent=1, default=str)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "player_ratings.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
