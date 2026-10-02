"""Positional MATCHUPS (unit vs unit) and REFEREE CREWS on totals: is there information the market misses?

Stages (each writes into output/research/matchups_refs.json; the .md is written by hand from it):
  python scripts/research/matchups_refs.py build      # unit ratings + matchup + referee features -> scratch cache
  python scripts/research/matchups_refs.py dev        # 2013-2019 closing-line residuals, 2020-2022 early-line CLV
  python scripts/research/matchups_refs.py model      # walk-forward log loss of the production margin model + matchups
  python scripts/research/matchups_refs.py freeze     # writes matchups_refs_frozen.json from FROZEN_RULES (no overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/matchups_refs.py holdout   # 2023-2025, ONCE

Leakage discipline
  * Unit ratings: decayed sums over a team's strictly earlier games (shift(1)), relative to a league rate computed
    from team-games played before the Tuesday of the target game's week (so everything is known at the Tuesday
    snapshot). Shrinkage = pseudo-count k (in plays) toward the league rate (rating 0).
  * Referee ratings: decayed sums over the referee's strictly earlier games, league reference likewise.
  * Referee ASSIGNMENTS are assumed public on Thursday of game week (crew assignments typically leak/publish
    mid-week via Football Zebras / NFL). Referee rules are therefore only evaluated at the Friday 21:40 UTC
    snapshot, and Thursday games (whose Friday snapshot does not exist) are excluded.
  * Data: pbp (2012+), nflverse pbp_participation (was_pressure, pass rushers; usable 2016+),
    FTN charting (blitzers, play action; 2022+). Pre-2016 pressure / pre-2022 blitz ratings are 0 (= league).

Targets
  (1) margin residual = home margin - nflverse closing spread_line   (2) total residual = total - total_line
  (3) early-week CLV at allowed-book prices vs the price-based closing fair line (edge_lab.closing_fair for spreads;
      totals research TotalDist + closing sharp fair total for totals).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]

from nflpred import features as F  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "matchups_refs.json"
FROZEN = OUT / "matchups_refs_frozen.json"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/matchups")
DEV_CL = range(2013, 2020)          # closing-line residual development
DEV_EL = (2020, 2021, 2022)         # early-line CLV development
HOLD = (2023, 2024, 2025)
HL = 8                              # team half-life in games (as production)
REF_HL = 48                         # referee half-life in games (~3 seasons)
LG_WINDOW = 256                     # league reference: last 256 team-games before the week's Tuesday
FTN_URL = "https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{}.parquet"
PART_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{}.parquet"


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2))


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    return x


def save_json(key, obj):
    d = json.loads(JSON.read_text()) if JSON.exists() else {}
    d[key] = jsonable(obj)
    JSON.write_text(json.dumps(d, indent=1))


def _fetch(url, path):
    if not path.exists():
        import urllib.request
        SCR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)


# ================================================================== schedule
def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[g.season.between(2012, 2025)].copy()
    for c in ("home_team", "away_team"):
        g[c] = F._norm_team(g[c])
    g["gameday"] = pd.to_datetime(g.gameday)
    back = (g.gameday.dt.weekday - 1) % 7                     # days since the Tuesday on/before
    g["cutoff"] = g.gameday - pd.to_timedelta(np.where(back == 0, 7, back), unit="D")  # strictly-before Tuesday
    g["res_m"] = (g.home_score - g.away_score) - g.spread_line
    g["res_t"] = (g.home_score + g.away_score) - g.total_line
    return g.sort_values(["gameday", "game_id"]).reset_index(drop=True)


# ================================================================== per team-game unit stats (offense perspective)
STATS = {   # name: (numerator, denominator, pseudo-count k in denominator units, +1 = higher is good for OFFENSE)
    "sack": ("sack_n", "db", 150, -1),
    "hit": ("hit_n", "db", 150, -1),
    "press": ("press_n", "press_den", 150, -1),
    "pass_epa": ("pass_epa_s", "db", 150, +1),
    "clean_epa": ("clean_epa_s", "clean_n", 120, +1),
    "pressed_epa": ("pressed_epa_s", "press_n", 60, +1),
    "rush_epa": ("rush_epa_s", "rushes", 100, +1),
    "rush_sr": ("rush_sr_s", "rushes", 100, +1),
    "in_epa": ("in_epa_s", "in_n", 80, +1),
    "out_epa": ("out_epa_s", "out_n", 80, +1),
    "out_share": ("out_n", "rushes", 100, +1),
    "deep_val": ("deep_epa_s", "db", 200, +1),
    "deep_rate": ("deep_n", "att_ay", 150, +1),
    "expl": ("expl_n", "plays", 200, +1),
    "yac_epa": ("yac_epa_s", "comp", 80, +1),
    "nohud": ("nohud_n", "plays", 200, +1),
    "blitz_rate": ("blitz_n", "ftn_db", 150, -1),
    "blitz_epa": ("blitz_epa_s", "blitz_n", 60, +1),
    "noblitz_epa": ("noblitz_epa_s", "noblitz_n", 100, +1),
}


def team_game_units() -> pd.DataFrame:
    cache = SCR / "units.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["game_id", "play_id", "posteam", "defteam", "pass", "rush", "sack", "qb_hit", "epa", "success",
            "air_yards", "yards_gained", "run_gap", "run_location", "no_huddle", "complete_pass", "pass_attempt",
            "yac_epa", "qb_kneel", "qb_spike"]
    out = []
    for s in range(2012, 2026):
        p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet", columns=cols)
        p = p[(p["pass"].eq(1) | p["rush"].eq(1)) & p.epa.notna() & p.qb_kneel.ne(1) & p.qb_spike.ne(1)].copy()
        p["posteam"], p["defteam"] = F._norm_team(p.posteam), F._norm_team(p.defteam)
        p["was_pressure"], p["n_blitzers"] = np.nan, np.nan
        if s >= 2016:
            q = _fetch(PART_URL.format(s), SCR / f"part_{s}.parquet")
            q = q[["nflverse_game_id", "play_id", "was_pressure"]].dropna(subset=["play_id"])
            q = q.assign(play_id=q.play_id.astype(int), wp_=q.was_pressure.map({True: 1.0, False: 0.0, "True": 1.0,
                                                                                 "False": 0.0, 1: 1.0, 0: 0.0}))
            q = q.rename(columns={"nflverse_game_id": "game_id"}).drop(columns="was_pressure")
            p = p.drop(columns="was_pressure").merge(q.drop_duplicates(["game_id", "play_id"]),
                                                      on=["game_id", "play_id"], how="left") \
                .rename(columns={"wp_": "was_pressure"})
        if s >= 2022:
            f = _fetch(FTN_URL.format(s), SCR / f"ftn_{s}.parquet")
            f = f[["nflverse_game_id", "nflverse_play_id", "n_blitzers"]].rename(
                columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id", "n_blitzers": "nb"})
            p = p.drop(columns="n_blitzers").merge(f.drop_duplicates(["game_id", "play_id"]),
                                                    on=["game_id", "play_id"], how="left").rename(columns={"nb": "n_blitzers"})
        ps, ru = p["pass"].eq(1), p["rush"].eq(1)
        pr = ps & p.was_pressure.notna()
        inside = ru & (p.run_gap.eq("guard") | p.run_location.eq("middle"))
        outside = ru & p.run_gap.isin(["tackle", "end"])
        att_ay = ps & p.pass_attempt.eq(1) & p.air_yards.notna() & p.sack.ne(1)
        deep = att_ay & (p.air_yards >= 20)
        fb = ps & p.n_blitzers.notna()
        bl = fb & (p.n_blitzers > 0)
        e = p.epa
        d = pd.DataFrame({
            "game_id": p.game_id, "team": p.posteam, "opp": p.defteam,
            "plays": 1.0, "db": ps * 1.0, "sack_n": (ps & p.sack.eq(1)) * 1.0,
            "hit_n": (ps & (p.sack.eq(1) | p.qb_hit.eq(1))) * 1.0,
            "press_den": pr * 1.0, "press_n": (pr & p.was_pressure.eq(1)) * 1.0,
            "pass_epa_s": e.where(ps, 0), "clean_epa_s": e.where(pr & p.was_pressure.eq(0), 0),
            "clean_n": (pr & p.was_pressure.eq(0)) * 1.0, "pressed_epa_s": e.where(pr & p.was_pressure.eq(1), 0),
            "rushes": ru * 1.0, "rush_epa_s": e.where(ru, 0), "rush_sr_s": p.success.where(ru, 0).fillna(0),
            "in_n": inside * 1.0, "in_epa_s": e.where(inside, 0), "out_n": outside * 1.0, "out_epa_s": e.where(outside, 0),
            "att_ay": att_ay * 1.0, "deep_n": deep * 1.0, "deep_epa_s": e.where(deep, 0),
            "expl_n": ((ps & (p.yards_gained >= 20)) | (ru & (p.yards_gained >= 10))) * 1.0,
            "comp": (ps & p.complete_pass.eq(1)) * 1.0, "yac_epa_s": p.yac_epa.where(ps & p.complete_pass.eq(1), 0).fillna(0),
            "nohud_n": p.no_huddle.eq(1) * 1.0,
            "ftn_db": fb * 1.0, "blitz_n": bl * 1.0, "blitz_epa_s": e.where(bl, 0),
            "noblitz_n": (fb & ~bl) * 1.0, "noblitz_epa_s": e.where(fb & ~bl, 0),
        })
        out.append(d.groupby(["game_id", "team", "opp"]).sum().reset_index())
    u = pd.concat(out, ignore_index=True)
    SCR.mkdir(parents=True, exist_ok=True)
    u.to_parquet(cache)
    return u


def _league_ref(rows: pd.DataFrame, num: str, den: str, cutoff: pd.Series, window=LG_WINDOW) -> np.ndarray:
    """League rate over the last `window` team-games with gameday < cutoff (all rows have gameday)."""
    r = rows.sort_values("gameday")
    cn = np.concatenate([[0], np.cumsum(r[num].values)])
    cd = np.concatenate([[0], np.cumsum(r[den].values)])
    i = np.searchsorted(r.gameday.values, cutoff.values, side="left")
    lo = np.maximum(0, i - window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cd[i] - cd[lo] > 0, (cn[i] - cn[lo]) / (cd[i] - cd[lo]), np.nan)


def _decayed_prior_sum(lg: pd.DataFrame, key: str, col: str, hl: float) -> pd.Series:
    return lg.groupby(key)[col].transform(lambda x: x.shift(1).fillna(0).ewm(halflife=hl, adjust=True).mean()
                                          * _wsum(len(x), hl))


def _wsum(n, hl):
    """Sum of EWM weights so that ewm(adjust=True).mean() * wsum = decayed SUM (weights 1, a, a^2, ...)."""
    a = 0.5 ** (1 / hl)
    i = np.arange(n)
    return (1 - a ** (i + 1)) / (1 - a)


def unit_ratings(g: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, team): offense ratings o_<stat> and defense ratings d_<stat>, as shrunk deviations from the
    league rate in OFFENSE-good orientation (d_x > 0 = this defense lets offenses do better on x)."""
    u = team_game_units()
    gd = g[["game_id", "gameday", "cutoff", "season"]]
    u = u.merge(gd, on="game_id")
    num_cols = [c for c in u.columns if c not in ("game_id", "team", "opp", "gameday", "cutoff", "season")]
    # defense rows: what the opponent's offense did against this team
    dfn = u.rename(columns={"team": "opp", "opp": "team"})
    res = {}
    for side, rows in (("o", u), ("d", dfn)):
        lg = rows.sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
        target = lg[["game_id", "team", "cutoff"]].copy()
        for name, (num, den, k, sgn) in STATS.items():
            L = _league_ref(u, num, den, lg.cutoff)                      # league rate (same for o and d)
            Sn = _decayed_prior_sum(lg, "team", num, HL)
            Sd = _decayed_prior_sum(lg, "team", den, HL)
            target[f"{side}_{name}"] = sgn * (Sn - np.nan_to_num(L) * Sd) / (Sd + k)
            target[f"{side}_{name}_n"] = Sd
        res[side] = target.drop(columns="cutoff")
    return res["o"].merge(res["d"], on=["game_id", "team"])


# ================================================================== matchup features
def matchup_features(g: pd.DataFrame) -> pd.DataFrame:
    cache = SCR / "matchups.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = unit_ratings(g)
    base = g[["game_id", "season", "home_team", "away_team"]]
    H = base.merge(r.rename(columns=lambda c: c if c == "game_id" else "h_" + c), left_on=["game_id", "home_team"],
                   right_on=["game_id", "h_team"])
    X = H.merge(r.rename(columns=lambda c: c if c == "game_id" else "a_" + c), left_on=["game_id", "away_team"],
                right_on=["game_id", "a_team"])
    # z-scale each rating by its std on 2013-2019 rows (a fixed scale; no effect on t-stats)
    sd = {}
    for n in STATS:
        for s in ("o", "d"):
            v = pd.concat([X.loc[X.season.between(2013, 2019), f"h_{s}_{n}"], X.loc[X.season.between(2013, 2019), f"a_{s}_{n}"]])
            if n.startswith(("blitz", "noblitz")):
                v = pd.concat([X.loc[X.season == 2023, f"h_{s}_{n}"], X.loc[X.season == 2023, f"a_{s}_{n}"]])
            if n in ("press", "clean_epa", "pressed_epa"):
                v = pd.concat([X.loc[X.season.between(2017, 2019), f"h_{s}_{n}"], X.loc[X.season.between(2017, 2019), f"a_{s}_{n}"]])
            sd[(s, n)] = float(v.std()) or 1.0
    z = lambda side, s, n: X[f"{side}_{s}_{n}"] / sd[(s, n)]
    relu = lambda v: np.clip(v, 0, None)
    out = pd.DataFrame({"game_id": X.game_id})

    def side_feats(off, de):     # off = 'h' or 'a' offense vs the other team's defense; all offense-good oriented
        f = {}
        for n in ("sack", "hit", "press", "rush_epa", "rush_sr", "deep_val", "expl", "pass_epa", "yac_epa", "blitz_rate"):
            o, d = z(off, "o", n), z(de, "d", n)
            f[f"{n}_add"] = o + d
            f[f"{n}_int"] = o * d
            f[f"{n}_mm"] = -relu(-o) * relu(-d) + relu(o) * relu(d)       # compounding mismatch, either direction
        # pass-rush "mismatch": elite rush (d strongly negative) vs bad protection (o strongly negative)
        for n in ("sack", "hit", "press"):
            f[f"{n}_bad"] = -relu(-z(off, "o", n)) * relu(-z(de, "d", n))
        # run-style: offense's outside-run share x defense's (outside - inside) EPA allowed
        f["gap_style"] = (z(off, "o", "out_share")) * (z(de, "d", "out_epa") - z(de, "d", "in_epa"))
        # pressure sensitivity: defense's pressure rate x offense's collapse under pressure (clean - pressed EPA);
        # z(d press) > 0 = defense generates LESS pressure, so the product is offense-good oriented
        f["press_sens"] = z(de, "d", "press") * (z(off, "o", "clean_epa") - z(off, "o", "pressed_epa"))
        # blitz: defense blitz rate x offense's EPA vs blitz relative to vs non-blitz (FTN 2022+)
        f["blitz_sens"] = -z(de, "d", "blitz_rate") * (z(off, "o", "blitz_epa") - z(off, "o", "noblitz_epa"))
        # deep shot offense vs defense that allows deep shots
        f["deep_style"] = z(off, "o", "deep_rate") * z(de, "d", "deep_val")
        return f

    fh, fa = side_feats("h", "a"), side_feats("a", "h")
    for k in fh:
        out[f"m_{k}"] = (fh[k] - fa[k]).values       # margin: positive favours home
        out[f"t_{k}"] = (fh[k] + fa[k]).values       # total: positive = more offense
    out["t_nohud"] = (z("h", "o", "nohud") + z("a", "o", "nohud")).values
    out = out.merge(g[["game_id"]], on="game_id")
    out.to_parquet(cache)
    return out


# ================================================================== referee ratings
def ref_game_stats(g: pd.DataFrame) -> pd.DataFrame:
    cache = SCR / "ref_games.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for s in range(2012, 2026):
        p = pd.read_parquet(ROOT / "data" / "raw" / f"pbp_{s}.parquet",
                            columns=["game_id", "penalty", "penalty_type", "penalty_yards", "penalty_team", "home_team"])
        p = p[p.penalty.eq(1)]
        p["home_pen"] = (F._norm_team(p.penalty_team) == F._norm_team(p.home_team)) * 1.0
        p["dpi_hold"] = p.penalty_type.isin(["Defensive Pass Interference", "Defensive Holding",
                                             "Illegal Contact"]) * 1.0
        p["ohold"] = p.penalty_type.eq("Offensive Holding") * 1.0
        rows.append(p.groupby("game_id").agg(pens=("penalty", "sum"), pen_yds=("penalty_yards", "sum"),
                                             dpi=("dpi_hold", "sum"), ohold=("ohold", "sum"),
                                             home_pens=("home_pen", "sum")).reset_index())
    r = pd.concat(rows)
    r.to_parquet(cache)
    return r


REF_STATS = {"pens": 16, "pen_yds": 16, "dpi": 16, "ohold": 16, "res_t": 80, "home_pen_share": 30}


def ref_features(g: pd.DataFrame) -> pd.DataFrame:
    r = g[["game_id", "gameday", "cutoff", "season", "referee", "res_t", "total_line"]].merge(ref_game_stats(g), on="game_id", how="left")
    r["home_pen_share"] = r.home_pens - r.pens / 2
    r["referee"] = r.referee.fillna("UNK").str.strip()
    r["one"] = 1.0
    done = r.res_t.notna()
    for c in REF_STATS:
        r[c] = r[c].where(done)
    r = r.sort_values(["referee", "gameday", "game_id"]).reset_index(drop=True)
    out = r[["game_id", "referee"]].copy()
    lgrows = r[done].assign(team="x")
    for c, k in REF_STATS.items():
        rr = r.assign(v=r[c].fillna(0), w=r[c].notna() * 1.0)
        L = _league_ref(lgrows.assign(v=lgrows[c], w=1.0), "v", "w", r.cutoff, window=128)
        dev = (rr.v - np.nan_to_num(L) * rr.w)
        rr["dev"] = dev
        Sn = _decayed_prior_sum(rr, "referee", "dev", REF_HL)
        Sd = _decayed_prior_sum(rr, "referee", "w", REF_HL)
        out[f"ref_{c}"] = np.where(r.referee == "UNK", 0.0, Sn / (Sd + k))
    out["ref_n"] = np.where(r.referee == "UNK", 0.0, _decayed_prior_sum(r.assign(one=done * 1.0), "referee", "one", REF_HL))
    return out


def build():
    g = load_games()
    m = matchup_features(g)
    rf = ref_features(g)
    d = g.merge(m, on="game_id", how="left").merge(rf, on=["game_id", "referee"], how="left")
    d.to_parquet(SCR / "games_feats.parquet")
    return d


def load_all() -> pd.DataFrame:
    p = SCR / "games_feats.parquet"
    return pd.read_parquet(p) if p.exists() else build()


# ================================================================== analysis helpers
def ols_z(y: pd.Series, x: pd.Series) -> dict:
    """y ~ a + b * z(x), heteroskedasticity-robust (HC0) SE. b = effect of a 1-SD feature change."""
    v = y.notna() & x.notna()
    y, x = y[v].values.astype(float), x[v].values.astype(float)
    if len(y) < 30 or x.std() == 0:
        return {"b": None, "se": None, "t": None, "n": int(len(y))}
    x = (x - x.mean()) / x.std()
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi
    se = float(np.sqrt(V[1, 1]))
    return {"b": float(b[1]), "se": se, "t": float(b[1] / se), "n": int(len(y))}


def feat_cols(d, pre):
    return [c for c in d.columns if c.startswith(pre) and not c.endswith("_n")]


def usable(d, c, seasons):
    """Pressure features need participation (2016+, used from 2017 so ratings have history); blitz needs FTN
    (2022+, used from 2023)."""
    x = d[d.season.isin(list(seasons))]
    if "press" in c or "sens" in c and "blitz" not in c:
        x = x[x.season >= 2017]
    if "blitz" in c:
        x = x[x.season >= 2023]
    return x


SD_SEASONS = range(2013, 2020)
INDEX_FEATS = ["m_pass_epa_add", "m_expl_add", "m_press_add", "m_rush_sr_add"]   # frozen composite (see dev)


def matchup_index(df: pd.DataFrame, d_all: pd.DataFrame) -> pd.Series:
    """Equal-weight unit-strength index (home-positive), each feature scaled by its 2013-2019 SD; /sqrt(k)."""
    sd = {f: d_all.loc[d_all.season.isin(list(SD_SEASONS)), f].std() for f in INDEX_FEATS}
    return sum(df[f] / sd[f] for f in INDEX_FEATS) / math.sqrt(len(INDEX_FEATS))


def ref_z(df: pd.DataFrame, d_all: pd.DataFrame, c: str) -> pd.Series:
    sd = d_all.loc[d_all.season.isin(list(SD_SEASONS)), c].std()
    return df[c] / sd


# ------------------------------------------------------------------ early-line tables
def spread_table(holdout: bool) -> pd.DataFrame:
    """Every (allowed book, snapshot, side) spread bet 2020-22 (or 2023-25), with price-based CLV vs closing fair."""
    import edge_lab as E
    from nflpred import spread_bets as SB, margins as K
    t = E.load(holdout)
    cf = E.closing_fair()
    t = t.merge(cf[["game_id", "mu_close_sharp", "mu_close_all"]], on="game_id")
    t["mu_close"] = t.mu_close_sharp.fillna(t.mu_close_all)
    t = t[t.point.notna() & t.sp_price.notna() & t.sp_price.between(-200, 200) & t.mu_close.notna()].copy()
    r = SB.load_rules()
    sgn = np.where(t.side == "home", 1, -1)
    hc, pu, _ = K.cover_probs(sgn * t.mu_close.values, r["margin"]["sigma"], t.point.values, r["_weights"])
    t["clv"] = hc * t.sp_dec + pu - 1
    t["pnl"] = t.sp_pnl
    t["snapname"] = t.requested_ts.dt.day_name().str[:3] + " " + t.requested_ts.dt.strftime("%H:%M")
    return t


def totals_table(holdout: bool) -> pd.DataFrame:
    """Every (allowed book, snapshot, side) totals bet, price-based CLV vs the closing sharp fair total
    (scripts/research/totals.py TotalDist and build; cached in scratch)."""
    seasons = HOLD if holdout else DEV_EL
    p = SCR / f"totals_{'holdout' if holdout else 'dev'}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    import totals as TR
    g = TR.load_games()
    t, _, _ = TR.build(seasons, TR.TotalDist(g), g)
    t = t.rename(columns={"tot_clv": "clv", "tot_pnl": "pnl"})
    t.to_parquet(p)
    return t


def one_per_game(b: pd.DataFrame, score="clv_now") -> pd.DataFrame:
    """Live rule: first qualifying snapshot, best price there (highest EV under the snapshot's sharp fair line)."""
    if b.empty:
        return b
    return b.sort_values(["game_id", "requested_ts", score], ascending=[True, True, False]).groupby("game_id").head(1)


def summ(b: pd.DataFrame, n_rules: int = 1) -> dict:
    if b is None or len(b) < 3:
        return {"bets": 0 if b is None else len(b)}
    c = b.clv.values
    t = c.mean() / (c.std(ddof=1) / math.sqrt(len(c)))
    return {"bets": len(b), "clv": float(c.mean()), "clv_se": float(c.std(ddof=1) / math.sqrt(len(c))), "clv_t": float(t),
            "p_one_sided": pval(t), "pass": bool(c.mean() > 0 and pval(t) < 0.05 / n_rules),
            "roi": float(b.pnl.mean()), "roi_se": float(b.pnl.std(ddof=1) / math.sqrt(len(b))),
            "beat_close": float((c > 0).mean()),
            "by_season": {int(s): {"n": len(x), "clv": float(x.clv.mean()), "roi": float(x.pnl.mean())}
                          for s, x in b.groupby("season")}}


def closing_ou(games: pd.DataFrame, side: pd.Series, n_rules: int = 1) -> dict:
    """O/U (or ATS) graded at the closing nflverse line and its actual closing juice."""
    x = games.assign(side=side)
    x = x[x.side.isin(["over", "under"]) & x.total.notna() & x.total_line.notna()]
    price = np.where(x.side == "over", x.over_odds, x.under_odds).astype(float)
    price = np.where(np.isfinite(price), price, -110)
    decm = np.where(price > 0, 1 + price / 100, 1 + 100 / -price)
    adj = np.where(x.side == "over", x.total - x.total_line, x.total_line - x.total)
    pnl = np.where(adj > 0, decm - 1, np.where(adj < 0, -1.0, 0.0))
    if len(pnl) < 3:
        return {"bets": len(pnl)}
    t = pnl.mean() / (pnl.std(ddof=1) / math.sqrt(len(pnl)))
    dec_ = adj != 0
    return {"bets": int(len(pnl)), "win_rate_ex_push": float((adj[dec_] > 0).mean()),
            "breakeven": float(np.mean(1 / decm[dec_])), "roi": float(pnl.mean()), "roi_t": float(t),
            "p_one_sided": pval(t), "pass": bool(pnl.mean() > 0 and pval(t) < 0.05 / n_rules)}


# ================================================================== frozen rules (chosen on dev; see stage_dev)
FROZEN_RULES = {
    "R1_spread_tue_unit_index": {
        "market": "spread", "snapshot": "Tue 14:10 UTC (<=7 days before kickoff)",
        "rule": "bet the side favoured by the equal-weight unit-strength index (pass EPA, explosive-play, pressure, "
                "rush success: offense vs opposing defense, additive) when |index| >= 2.5; best allowed-book price "
                "(highest EV vs the snapshot's sharp fair line); one bet per game",
        "index_feats": INDEX_FEATS, "threshold": 2.5, "metric": "price-based CLV vs edge_lab.closing_fair"},
    "R2_total_close_low_penalty_crew_under": {
        "market": "total", "snapshot": "closing line (nflverse total_line, actual closing juice); referee known",
        "rule": "bet UNDER when the referee's shrunk penalties-per-game rating (z, 2013-19 scale) <= -1.0",
        "threshold": -1.0, "metric": "O/U ROI at actual closing juice vs break-even (one-sided t on per-bet profit)",
        "secondary": "same games at the Friday 21:40 snapshot (non-Thursday games): price-based CLV, informational"},
    "R3_total_close_sack_interaction_under": {
        "market": "total", "snapshot": "closing line (nflverse total_line, actual closing juice)",
        "rule": "bet UNDER at the close when t_sack_int (sum over both offense-vs-defense pairings of z(off sack-"
                "avoidance) x z(def sack-allowance)) z-score >= 1.0 (2013-19 scale)", "threshold": 1.0,
        "metric": "O/U ROI at actual closing juice vs break-even (one-sided t on per-bet profit)"},
}


def rule_bets(name, d_all, sp=None, tt=None, games=None):
    if name == "R1_spread_tue_unit_index":
        x = sp[(sp.snapname == "Tue 14:10") & (sp.hours_before <= 7 * 24)].merge(d_all[["game_id"] + INDEX_FEATS], on="game_id")
        x["s"] = matchup_index(x, d_all) * np.where(x.side == "home", 1, -1)
        return one_per_game(x[x.s >= 2.5].assign(clv_now=x.sp_ev_sharp))
    if name == "R2_total_close_low_penalty_crew_under":
        return games[ref_z(games, d_all, "ref_pens") <= -1.0]
    if name == "R2_fri_secondary":
        x = tt[(tt.snap == "fri") & (tt.weekday != "Thursday") & (tt.side == "under")].merge(
            d_all[["game_id", "ref_pens"]], on="game_id")
        x["z"] = ref_z(x, d_all, "ref_pens")
        return one_per_game(x[x.z <= -1.0].assign(clv_now=x.ev_ref_now))
    if name == "R3_total_close_sack_interaction_under":
        sd = d_all.loc[d_all.season.isin(list(SD_SEASONS)), "t_sack_int"].std()
        return games[(games.t_sack_int / sd) >= 1.0]
    raise KeyError(name)


# ================================================================== stages
def stage_dev():
    d = load_all()
    d = d[d.total.notna()]
    cl, el = d[d.season.isin(list(DEV_CL))], d[d.season.isin(list(DEV_EL))]
    out = {}
    # ---- 1. residual vs closing line, per feature (1-SD effect in points)
    coef = {}
    for pre, tgt in (("m_", "res_m"), ("t_", "res_t")):
        for c in feat_cols(d, pre):
            coef[c] = {"close_2013_19": ols_z(usable(d, c, DEV_CL)[tgt], usable(d, c, DEV_CL)[c]),
                       "close_2020_22": ols_z(usable(d, c, DEV_EL)[tgt], usable(d, c, DEV_EL)[c])}
    # ---- 2. early-week moves (Tuesday sharp fair -> closing fair), 2020-22
    sp = spread_table(False)
    tue = sp[(sp.snapname == "Tue 14:10") & (sp.hours_before <= 7 * 24)]
    gm = tue.groupby("game_id").agg(m_tue=("m_sharp", "first"), m_cons=("m_cons", "first"), mu_close=("mu_close", "first"),
                                    mu_model=("mu_model", "first")).reset_index()
    gm["move"] = gm.mu_close - gm.m_tue.fillna(gm.m_cons)
    gm = gm.merge(d, on="game_id")
    tt = totals_table(False)
    tg = tt.groupby("game_id").agg(mu_tue=("mu_tue", "first"), mu_fri=("mu_fri", "first"), mu_close=("mu_close", "first"),
                                   weekday=("weekday", "first")).reset_index().merge(d, on="game_id", suffixes=("", "_g"))
    tg["mv_tue"], tg["mv_fri"] = tg.mu_close - tg.mu_tue, tg.mu_close - tg.mu_fri
    for c in feat_cols(d, "m_"):
        coef[c]["tue_move_2020_22"] = ols_z(usable(gm, c, DEV_EL).move, usable(gm, c, DEV_EL)[c])
    for c in feat_cols(d, "t_"):
        coef[c]["tue_move_2020_22"] = ols_z(usable(tg, c, DEV_EL).mv_tue, usable(tg, c, DEV_EL)[c])
    out["feature_coefs"] = coef
    out["n_features_tested"] = len(coef)
    out["move_sd"] = {"spread_tue": float(gm.move.std()), "total_tue": float(tg.mv_tue.std()), "total_fri": float(tg.mv_fri.std())}
    # unit index: move regression with/without control for model disagreement
    gm["idx"] = matchup_index(gm, d)
    gm["dis"] = gm.mu_model - gm.m_tue.fillna(gm.m_cons)
    out["index_move"] = {"alone": ols_z(gm.move, gm.idx), "res_close_2013_19": ols_z(cl.res_m, matchup_index(cl, d)),
                         "res_close_2020_22": ols_z(el.res_m, matchup_index(el, d))}
    v = gm[["move", "idx", "dis"]].dropna()
    X = np.c_[np.ones(len(v)), (v.idx - v.idx.mean()) / v.idx.std(), (v.dis - v.dis.mean()) / v.dis.std()]
    b = np.linalg.lstsq(X, v.move.values, rcond=None)[0]
    e = v.move.values - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXi @ (X.T * e ** 2) @ X @ XtXi))
    out["index_move"]["with_model_disagreement"] = {"b_idx": b[1], "se_idx": se[1], "b_dis": b[2], "se_dis": se[2], "n": len(v)}
    # index CLV by snapshot and threshold (spread)
    x = sp.merge(d[["game_id"] + INDEX_FEATS], on="game_id")
    x = x[x.hours_before <= 8 * 24]
    x["s"] = matchup_index(x, d) * np.where(x.side == "home", 1, -1)
    snaps = {}
    # Monday 14:10 / Tuesday 00:00 snapshots are excluded: Monday-night teams' features would include that game
    for sn in ("Tue 14:10", "Wed 14:10", "Thu 14:10", "Fri 21:40", "Sat 14:10"):
        y = x[x.snapname == sn]
        base = y.sort_values(["game_id", "side", "sp_ev_sharp"], ascending=[True, True, False]).groupby(["game_id", "side"]).head(1)
        snaps[sn] = {"base_best_price_clv_both_sides": float(base.clv.mean())}
        for thr in (1.0, 1.5, 2.0, 2.5):
            snaps[sn][f"idx>={thr}"] = {k: v for k, v in summ(one_per_game(y[y.s >= thr].assign(clv_now=y.sp_ev_sharp))).items()
                                        if k in ("bets", "clv", "clv_t", "roi")}
    out["index_clv_by_snapshot"] = snaps
    # ---- 3. referee
    rg = d.merge(ref_game_stats(d), on="game_id")
    rr = {}
    for c in ("pens", "pen_yds", "dpi", "ohold", "home_pen_share"):
        y = rg[c] if c != "home_pen_share" else rg.home_pens - rg.pens / 2
        rr[c] = {"stability_corr_rating_vs_game_2013_22": float(rg.loc[rg.season.between(2013, 2022), f"ref_{c}"].corr(
            y[rg.season.between(2013, 2022)]))}
    rr["contemporaneous"] = {"corr_game_pens_vs_total_resid": float(rg.pens.corr(rg.res_t)),
                             "corr_game_dpi_vs_total_resid": float(rg.dpi.corr(rg.res_t))}
    for c in ("ref_pens", "ref_pen_yds", "ref_dpi", "ref_ohold", "ref_res_t", "ref_n"):
        rr[c + "_vs_total"] = {"close_2013_19": ols_z(cl.res_t, cl[c]), "close_2020_22": ols_z(el.res_t, el[c]),
                               "fri_move_2020_22": ols_z(tg[tg.weekday != "Thursday"].mv_fri, tg[tg.weekday != "Thursday"][c])}
        rr[c + "_vs_margin_close_2013_19"] = ols_z(cl.res_m, cl[c])
    # referee O/U at closing juice and Friday CLV, by threshold (dev)
    for c in ("ref_pens", "ref_dpi", "ref_res_t"):
        z = ref_z(cl, d, c)
        for thr in (1.0, 1.5):
            rr[f"{c}_close_ou_2013_19_z>={thr}_over"] = closing_ou(cl[z >= thr], pd.Series("over", index=cl[z >= thr].index))
            rr[f"{c}_close_ou_2013_19_z<=-{thr}_under"] = closing_ou(cl[z <= -thr], pd.Series("under", index=cl[z <= -thr].index))
        y = tt[(tt.snap == "fri") & (tt.weekday != "Thursday")].merge(d[["game_id", c]], on="game_id")
        y["z"] = ref_z(y, d, c)
        for thr in (1.0, 1.5):
            for side, cond in (("over", y.z >= thr), ("under", y.z <= -thr)):
                rr[f"{c}_fri_clv_2020_22_{side}_|z|>={thr}"] = {k: v for k, v in summ(one_per_game(
                    y[cond & (y.side == side)].assign(clv_now=y.ev_ref_now))).items() if k in ("bets", "clv", "clv_t", "roi")}
    out["referee"] = rr
    # ---- 4. frozen-rule candidates on dev
    U = lambda b: pd.Series("under", index=b.index)
    out["rules_dev"] = {
        "R1_tue_2020_22": summ(rule_bets("R1_spread_tue_unit_index", d, sp=sp)),
        "R2_close_ou_2013_19": closing_ou(b := rule_bets("R2_total_close_low_penalty_crew_under", d, games=cl), U(b)),
        "R2_close_ou_2020_22": closing_ou(b := rule_bets("R2_total_close_low_penalty_crew_under", d, games=el), U(b)),
        "R2_fri_clv_2020_22": summ(rule_bets("R2_fri_secondary", d, tt=tt)),
        "R3_close_ou_2013_19": closing_ou(b := rule_bets("R3_total_close_sack_interaction_under", d, games=cl), U(b)),
        "R3_close_ou_2020_22": closing_ou(b := rule_bets("R3_total_close_sack_interaction_under", d, games=el), U(b)),
    }
    save_json("dev", out)
    top = sorted(((c, v["close_2013_19"]["t"] or 0, (v["close_2020_22"]["t"] or 0), (v["tue_move_2020_22"]["t"] or 0))
                  for c, v in coef.items()), key=lambda r: -abs(r[1]))[:12]
    for r in top:
        print("%-22s close13-19 t=%5.2f  close20-22 t=%5.2f  tue-move t=%5.2f" % r)
    print(json.dumps(jsonable(out["index_move"]), indent=1))
    print(json.dumps(jsonable(out["rules_dev"]), indent=1)[:3000])


def stage_model():
    """Walk-forward log loss of the production margin model + matchup features (scripts/experiment.py protocol)."""
    import experiment as X
    from nflpred import model as Mo
    _, df = X.run("baseline")
    d = load_all()
    mcols = feat_cols(d, "m_")
    df = df.merge(d[["game_id"] + mcols], on="game_id", how="left")
    for c in mcols:
        df[c] = df[c].fillna(0.0)
    adds = [c for c in mcols if c.endswith("_add") and "blitz" not in c]
    inter = [c for c in mcols if (c.endswith(("_int", "_mm", "_bad")) or c in ("m_gap_style", "m_press_sens", "m_deep_style"))
             and "blitz" not in c]
    sets = {"baseline": [], "index4_add": INDEX_FEATS, "all_additive": adds, "all_interactions": inter,
            "mismatch_core": ["m_press_bad", "m_sack_bad", "m_gap_style", "m_deep_style", "m_press_sens", "m_rush_sr_mm"],
            "additive+interactions": adds + inter}
    res = {}
    for name, extra in sets.items():
        feats = F.FEATURES + extra
        r = {}
        for label, seasons in (("val_2015_19", X.VAL), ("hold_2020_25", X.HOLD)):
            ps, ys = [], []
            for s in seasons:
                tr = Mo.train_rows(df, before_season=s)
                te = df[(df.season == s) & df.home_win.notna()]
                ps.append(Mo.fit_predict(tr, te, feats, "margin"))
                ys.append(te.home_win.values)
            r[label] = round(Mo.score(np.concatenate(ys), np.concatenate(ps))["log_loss"], 4)
        res[name] = {"n_extra": len(extra), **r}
        print(name, res[name])
    best = min((k for k in res if k != "baseline"), key=lambda k: res[k]["val_2015_19"])
    res["selected_on_validation"] = best
    res["selected_delta_val"] = res[best]["val_2015_19"] - res["baseline"]["val_2015_19"]
    res["selected_delta_hold"] = res[best]["hold_2020_25"] - res["baseline"]["hold_2020_25"]
    res["adopt"] = bool(res["selected_delta_val"] < 0)   # must beat the baseline on validation to be considered
    save_json("model", res)


def stage_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; frozen rules are never overwritten")
    import datetime as dt
    FROZEN.write_text(json.dumps({"frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                  "n_rules": len(FROZEN_RULES), "pass_bar": f"holdout CLV > 0 (R3: ROI > 0) with "
                                  f"one-sided p < 0.05/{len(FROZEN_RULES)}", "holdout": list(HOLD),
                                  "rules": FROZEN_RULES}, indent=1))
    print("frozen", list(FROZEN_RULES))


def stage_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES" or not FROZEN.exists():
        raise SystemExit("holdout is locked: run freeze first and set EDGE_HOLDOUT")
    if JSON.exists() and "holdout" in json.loads(JSON.read_text()):
        raise SystemExit("holdout already run once")
    fr = json.loads(FROZEN.read_text())
    n = fr["n_rules"]
    d = load_all()
    d = d[d.total.notna()]
    ho = d[d.season.isin(HOLD)]
    sp, tt = spread_table(True), totals_table(True)
    U = lambda b: pd.Series("under", index=b.index)
    res = {"rules": {
        "R1_spread_tue_unit_index": summ(rule_bets("R1_spread_tue_unit_index", d, sp=sp), n),
        "R2_total_close_low_penalty_crew_under": closing_ou(b := rule_bets("R2_total_close_low_penalty_crew_under", d, games=ho), U(b), n),
        "R3_total_close_sack_interaction_under": closing_ou(b := rule_bets("R3_total_close_sack_interaction_under", d, games=ho), U(b), n)},
        "secondary": {"R2_fri_clv": summ(rule_bets("R2_fri_secondary", d, tt=tt), n)}}
    # descriptive (labelled, no verdicts): the same coefficients on the holdout seasons
    tue = sp[(sp.snapname == "Tue 14:10") & (sp.hours_before <= 7 * 24)]
    gm = tue.groupby("game_id").agg(m_tue=("m_sharp", "first"), m_cons=("m_cons", "first"),
                                    mu_close=("mu_close", "first")).reset_index().merge(d, on="game_id")
    gm["move"] = gm.mu_close - gm.m_tue.fillna(gm.m_cons)
    res["descriptive"] = {"index_tue_move": ols_z(gm.move, matchup_index(gm, d)),
                          "index_res_close": ols_z(ho.res_m, matchup_index(ho, d)),
                          "t_sack_int_res_close": ols_z(ho.res_t, ho.t_sack_int),
                          "ref_pens_res_close": ols_z(ho.res_t, ho.ref_pens),
                          "ref_res_t_res_close": ols_z(ho.res_t, ho.ref_res_t),
                          "ref_pens_stability": float(ho.merge(ref_game_stats(ho), on="game_id")[["ref_pens", "pens"]].corr().iloc[0, 1])}
    coefs = {}
    for pre, tgt in (("m_", "res_m"), ("t_", "res_t")):
        for c in feat_cols(d, pre):
            coefs[c] = ols_z(ho[tgt], ho[c])
    res["descriptive"]["feature_coefs_res_close"] = coefs
    ts = [v["t"] for v in coefs.values() if v["t"] is not None]
    res["descriptive"]["n_abs_t_gt_1.96"] = int(sum(abs(t) > 1.96 for t in ts))
    res["descriptive"]["n_features"] = len(ts)
    save_json("holdout", res)
    print(json.dumps(jsonable(res["rules"]), indent=1))
    print(json.dumps(jsonable({k: v for k, v in res["descriptive"].items() if k != "feature_coefs_res_close"}), indent=1))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": lambda: print(build().shape), "dev": stage_dev, "model": stage_model, "freeze": stage_freeze,
     "holdout": stage_holdout}[stage]()
