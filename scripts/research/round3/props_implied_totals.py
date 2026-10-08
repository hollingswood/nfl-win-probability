"""PROPS-IMPLIED TEAM SCORING vs TEAM-TOTAL / GAME-TOTAL MARKETS (cross-market consistency), NFL 2023-25.

Question: do player-prop markets imply team scoring that disagrees with the team-total / game-total markets, and
does betting the derivative toward the props beat the close?  Pre-declared 2026-10-07, BEFORE any outcome or CLV
of this design was looked at.  Stages:
    PYTHONPATH=src python scripts/research/round3/props_implied_totals.py calibrate   # 2023 only (may be re-run)
    PYTHONPATH=src python scripts/research/round3/props_implied_totals.py holdout     # 2024-25, ONCE (guarded)
Outputs: output/research/round3/props_implied_totals_calib.json (2023 constants + 2023 in-sample descriptives),
         output/research/round3/props_implied_totals.json (holdout), .md written from the json.

DATA
  Props: data/historical_odds/props (Odds API, regions=us), as loaded/matched by scripts/research/props_full.py
    `prep` (cached quotes.parquet: event->game_id via props_backtest.map_games, player->gsis id via
    props_backtest.match_players).  Snapshots: early = Fri 21:40 UTC (Sunday games; kickoff-24h otherwise),
    close = kickoff-75 min (open = Tue 14:10 is too sparse in 2023 and is not used).
    Anytime TD: only the Yes price exists (props_full puts it in p_imp = 1/decimal).
  Team totals: data/historical_odds/derivatives via scripts/research/derivatives.py build() (main-line filters,
    implied MEAN of every two-way quote under the derivatives.py team-points key-number pmf fit on 2012-22,
    snapshot consensus `dcons` = median implied mean over ALL books; close consensus `close_cons`).
  Game totals: data/historical_odds/totals + dense (tot_* columns); fg_fair filters of derivatives.py; per-quote
    implied expected total under nflpred.totals.TotalDist (totals_dist.json); consensus = median over all books.
  Outcomes: games.parquet scores; pbp TDs (td_player_id; team offensive TDs = pass/rush TD by posteam) and FGs.

TEAM ASSIGNMENT (no look-ahead): a prop row's team is the player's team in this game when he played; for players
  who end up inactive (props_full status did_not_play/unmatched) it is the team of his most recent EARLIER played
  game in the same season (pd.merge_asof by name), kept only if it is one of the two teams in the game.  Every
  quoted player counts, whether or not he later plays (the live system cannot know).

PROPS-IMPLIED TEAM POINTS (per team-game and snapshot sn in {early, close})
  TD signal (primary):
    per quote: lam_raw = -ln(1 - p_imp); fair lam = k_b * lam_raw**gamma, books b in CORE_TD (the six books that
      quote anytime TD in all three seasons).  k_b (per book) and gamma (global) are fit on 2023 by maximum
      likelihood of 1{player scored >= 1 TD} = 1 - exp(-lam) over played players' early+close quotes (void rule).
    player lam = median fair lam over core books (needs >= 2 books).
    lam9 = sum of the 9 largest player lams on the team (team needs >= 9 such players: books list 11-14 players
      per team and the depth of the list varies by season, so the tail is cut to keep the level comparable).
    E[off TDs] = r_sn * lam9, r_sn = sum(realised offensive TDs) / sum(lam9) on 2023 (coverage factor).
    PPTD = sum(points - 3*FGmade) / sum(all TDs) on 2023 (PAT/2pt/defensive-TD/safety value per TD).
    FG points = f0_sn + f1_sn * TT_sn (OLS of 3*FGmade on the snapshot team total, 2023).
    P_TD = PPTD * r_sn * lam9 + f0_sn + f1_sn * TT_sn + c0_sn, with c0_sn set so the 2023 mean of P_TD - TT_sn is
      0 (the level is anchored to the market, so the signal is purely cross-sectional; c0 is reported).
  Yards signal (secondary):
    per player/market: consensus line L* (modal point over all books, ties -> closest to the median), p* = median
      no-vig P(over) at L*; price-adjusted median m = L* + sigma_m * Phi^-1(p*), sigma = 60 pass yds, 20 rush yds.
    Y = max QB pass-yds m on the team + sum of the 2 largest rush-yds m on the team (needs both).
    P_YD = a_sn + b_sn * Y, OLS of the snapshot team total TT_sn on Y over 2023 team-games.
  Market team points TT_sn = derivatives.py team-total consensus implied mean (dcons / close_cons); fallback (no
    team-total quote) = derivatives.py full-game-implied team mean (fg_mean from game total and spread).
  gap_sn = P_sn - TT_sn (points; + = props imply more scoring than the team-total market).

RULES (4, all at the EARLY snapshot, flat 1u, thresholds fixed here, nothing tuned on outcomes)
  P1_TD / P1_YD (team total): gap_early >= +1.5 -> team-total OVER; <= -1.5 -> UNDER.  One bet per team-game.
  P2_TD / P2_YD (game total): gapG = gap_home + gap_away (both teams need the signal); >= +2.5 -> game OVER,
    <= -2.5 -> UNDER.  One bet per game.
  Books: Arizona-legal list AZ = draftkings, fanduel, espnbet, betmgm, williamhill_us, betrivers, fanatics,
    hardrockbet, ballybet (only those present in the feed can be bet).  Among the AZ quotes of the side at the
    betting snapshot take the max EV under the early all-book consensus (key-number pmf at dcons / TotalDist at
    the consensus expected total) = best number/price trade-off.  Team totals: the derivatives early snapshot
    (same requested time as the props snapshot).  Game totals: the totals/dense snapshot nearest to the props
    early request within [-10, +70] min (never a price older than 10 min before the signal).
  Informational variants (not tested): my_books.json books only (draftkings, fanduel, espnbet); threshold 0
    (bet the sign of every gap) as a dose-response / vig baseline.
CLV (the pass test): P_close(win) * decimal + P_close(push) - 1.  P_close(over) = median two-way no-vig P(over)
  among CLOSE quotes of ALL books at the SAME number ("direct"; push share from the pmf at the close consensus);
  if no close book quotes that number: pmf at the close consensus (team: close_cons; game: latest totals snapshot
  0-3 h before kickoff, median implied mean).  Results graded on final team / game points (push refunds; OT in).
STATISTICS: mean CLV and ROI with game-clustered SEs; one-sided p = 1 - Phi(mean/SE).
SPLIT: 2023 = calibration + in-sample descriptives only (all constants above fit there).  2024-25 = holdout, run
  once with the 2023 constants frozen in props_implied_totals_calib.json.
PASS (per rule, Bonferroni k = 4): holdout n >= 100 bets, mean CLV > 0 with one-sided p < 0.05/4 = 0.0125, and
  mean CLV > 0 in 2024 AND in 2025 separately.  ROI is reported, not tested (too noisy at these n).
LEAD-LAG (pre-declared secondary, reported for 2023 and 2024-25): per team-game, regress
  dTT = TT_close - TT_early on gap_early (beta_mkt: market moves toward props) and dP = P_close - P_early on
  -gap_early (beta_props: props move toward the market); same at game level with the game-total consensus and
  gapG.  "Market follows props" = holdout beta_mkt > 0 (one-sided p < 0.05) AND beta_mkt - beta_props > 0
  (game-cluster bootstrap, 2000 draws, one-sided p < 0.05).  Caveat declared up front: measurement noise in either
  side's early value reverts and inflates its own beta; that is why the two directions are compared.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research")]
import derivatives as D  # noqa: E402
from nflpred.totals import TotalDist  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad")
PF_QUOTES = SCRATCH / "propsfull" / "quotes.parquet"   # props_full.py prep output
SCR = SCRATCH / "round3"
OUT = ROOT / "output" / "research" / "round3"
CALIB = OUT / "props_implied_totals_calib.json"
RESULT = OUT / "props_implied_totals.json"
RAW = ROOT / "data" / "raw"
ODDS = ROOT / "data" / "historical_odds"
DEV, HOLD = (2023,), (2024, 2025)
SEASONS = DEV + HOLD
SNAPS = ("early", "close")
CORE_TD = ["betmgm", "betrivers", "bovada", "draftkings", "fanduel", "williamhill_us"]
AZ = {"draftkings", "fanduel", "espnbet", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet",
      "ballybet"}
MYB = set(json.load(open(ROOT / "my_books.json"))["allowed_books"])
TOPN = 9
SIG_YDS = {"pass_yds": 60.0, "rush_yds": 20.0}
THR_TEAM, THR_GAME = 1.5, 2.5
RULES = ["P1_TD", "P1_YD", "P2_TD", "P2_YD"]
K_TESTS = len(RULES)
TEAM_FIX = {"LAR": "LA", "STL": "LA", "SD": "LAC", "OAK": "LV", "JAC": "JAX", "WSH": "WAS"}


def phi_inv(p):
    from scipy.stats import norm
    return norm.ppf(np.clip(np.asarray(p, float), 0.02, 0.98))


def pval(t):
    return 0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else float("nan")


def clustered(x, g):
    x = np.asarray(x, float)
    m = ~np.isnan(x)
    x, g = x[m], np.asarray(g)[m]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, len(x)
    _, gi = np.unique(g, return_inverse=True)
    s = np.bincount(gi, weights=x - x.mean())
    return float(x.mean()), float(np.sqrt((s ** 2).sum()) / len(x)), len(x)


def ols_cl(x, y, g):
    """y = a + b x; returns b, clustered SE of b."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], np.asarray(g)[ok]
    X = np.column_stack([np.ones(len(x)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    _, gi = np.unique(g, return_inverse=True)
    U = np.zeros((gi.max() + 1, 2))
    np.add.at(U, gi, X * e[:, None])
    V = XtX_inv @ (U.T @ U) @ XtX_inv
    return float(b[1]), float(np.sqrt(V[1, 1])), float(b[0]), len(x)


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 5)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ============================================================================================ loading
def games() -> pd.DataFrame:
    g = pd.read_parquet(RAW / "games.parquet")
    g = g[g.season.isin(SEASONS) & g.home_score.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_FIX)
    g["gameday"] = pd.to_datetime(g.gameday)
    return g[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team", "home_score",
              "away_score"]]


def pbp_team_player() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(team-game: off_tds, all_tds, fg_made), (player-game: tds)."""
    cols = ["game_id", "posteam", "td_team", "touchdown", "pass_touchdown", "rush_touchdown", "field_goal_result",
            "td_player_id"]
    p = pd.concat([pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=cols) for s in SEASONS], ignore_index=True)
    for c in ("posteam", "td_team"):
        p[c] = p[c].replace(TEAM_FIX)
    td = p[p.touchdown == 1]
    off = td[(td.td_team == td.posteam) & ((td.pass_touchdown == 1) | (td.rush_touchdown == 1))]
    a = off.groupby(["game_id", "td_team"]).size().rename("off_tds")
    b = td.groupby(["game_id", "td_team"]).size().rename("all_tds")
    f = p[p.field_goal_result == "made"].groupby(["game_id", "posteam"]).size().rename("fg_made")
    a.index.names = b.index.names = f.index.names = ["game_id", "team"]
    T = pd.concat([a, b, f], axis=1).fillna(0).reset_index()
    pl = td[td.td_player_id.notna()].groupby(["game_id", "td_player_id"]).size().rename("tds").reset_index() \
        .rename(columns={"td_player_id": "player_id"})
    return T, pl


def team_games(g: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    h = g.assign(team=g.home_team, opp=g.away_team, side="home", pts=g.home_score)
    a = g.assign(team=g.away_team, opp=g.home_team, side="away", pts=g.away_score)
    tg = pd.concat([h, a], ignore_index=True)[["game_id", "season", "week", "gameday", "team", "opp", "side", "pts"]]
    tg = tg.merge(T, on=["game_id", "team"], how="left")
    tg[["off_tds", "all_tds", "fg_made"]] = tg[["off_tds", "all_tds", "fg_made"]].fillna(0)
    return tg


def load_props(g: pd.DataFrame) -> pd.DataFrame:
    if not PF_QUOTES.exists():
        raise SystemExit(f"missing {PF_QUOTES}: run scripts/research/props_full.py build, fit, prep first")
    q = pd.read_parquet(PF_QUOTES, columns=["event_id", "home", "away", "book", "market", "player", "point", "p_nv",
                                             "p_imp", "season", "snap", "game_id", "player_id", "status", "team",
                                             "position"])
    q = q[q.snap.isin(SNAPS) & q.market.isin(["td", "pass_yds", "rush_yds"]) & q.game_id.notna()].copy()
    q = q.merge(g[["game_id", "gameday", "home_team", "away_team"]], on="game_id", how="inner")
    # team for players who did not play: most recent earlier played game (same season, same name string)
    played = q[q.team.notna()][["season", "player", "gameday", "team", "position"]] \
        .drop_duplicates(["season", "player", "gameday"]).sort_values("gameday")
    miss = q[q.team.isna()].drop(columns=["team", "position"]).reset_index()
    miss = miss.sort_values("gameday")
    m = pd.merge_asof(miss, played, on="gameday", by=["season", "player"], allow_exact_matches=False)
    m = m.set_index("index")
    ok = (m.team == m.home_team) | (m.team == m.away_team)
    q.loc[m.index[ok], "team"] = m.team[ok]
    q.loc[m.index[ok], "position"] = m.position[ok]
    q["team_src"] = np.where(q.status == "played", "played", np.where(q.team.notna(), "prior_game", "none"))
    return q


# ============================================================================================ TD signal
def td_quotes(q: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    t = q[(q.market == "td") & q.book.isin(CORE_TD) & q.p_imp.between(0.005, 0.95)].copy()
    t = t.drop_duplicates(["game_id", "snap", "book", "player"], keep="last")
    t["lam_raw"] = -np.log(1 - t.p_imp)
    t = t.merge(pl, on=["game_id", "player_id"], how="left")
    t["tds"] = np.where(t.status == "played", t.tds.fillna(0), np.nan)
    return t


def fit_td_map(t: pd.DataFrame) -> dict:
    from scipy.optimize import minimize
    x = t[t.season.isin(DEV) & (t.status == "played") & t.tds.notna()]
    bi = x.book.map({b: i for i, b in enumerate(CORE_TD)}).to_numpy()
    lr = x.lam_raw.to_numpy()
    y = (x.tds > 0).to_numpy(float)

    def nll(th):
        k = np.exp(th[:len(CORE_TD)])[bi]
        lam = k * lr ** np.exp(th[-1])
        p = np.clip(1 - np.exp(-lam), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    r = minimize(nll, np.zeros(len(CORE_TD) + 1), method="L-BFGS-B")
    k = dict(zip(CORE_TD, np.exp(r.x[:len(CORE_TD)])))
    gam = float(np.exp(r.x[-1]))
    lam = np.array([k[b] for b in x.book]) * lr ** gam
    return {"k_book": k, "gamma": gam, "n_quotes": int(len(x)), "converged": bool(r.success),
            "mean_p_raw": float(x.p_imp.mean()), "mean_p_fair": float((1 - np.exp(-lam)).mean()),
            "td_rate": float(y.mean())}


def team_lam(t: pd.DataFrame, cal: dict) -> pd.DataFrame:
    t = t[t.team.notna()].copy()
    t["lam"] = t.book.map(cal["k_book"]) * t.lam_raw ** cal["gamma"]
    p = t.groupby(["game_id", "snap", "team", "player"]).agg(lam=("lam", "median"), nb=("book", "nunique"))
    p = p[p.nb >= 2].reset_index().sort_values("lam", ascending=False)
    top = p.groupby(["game_id", "snap", "team"]).head(TOPN)
    out = top.groupby(["game_id", "snap", "team"]).agg(lam9=("lam", "sum"), n9=("lam", "size")).reset_index()
    n = p.groupby(["game_id", "snap", "team"]).size().rename("n_players").reset_index()
    out = out.merge(n, on=["game_id", "snap", "team"])
    out.loc[out.n9 < TOPN, "lam9"] = np.nan
    return out


# ============================================================================================ yards signal
def team_yards(q: pd.DataFrame) -> pd.DataFrame:
    o = q[q.market.isin(["pass_yds", "rush_yds"]) & q.team.notna() & q.p_nv.notna() & q.point.notna()].copy()
    rows = []
    for (gid, sn, mk, tm, pl_), G in o.groupby(["game_id", "snap", "market", "team", "player"], sort=False):
        vc = G.point.value_counts()
        top = vc[vc == vc.max()].index.to_numpy()
        L = top[np.argmin(np.abs(top - G.point.median()))]
        p = G.loc[G.point == L, "p_nv"].median()
        rows.append((gid, sn, mk, tm, pl_, L, p))
    c = pd.DataFrame(rows, columns=["game_id", "snap", "market", "team", "player", "L", "p"])
    c["m"] = c.L + c.market.map(SIG_YDS) * phi_inv(c.p)
    pas = c[c.market == "pass_yds"].groupby(["game_id", "snap", "team"]).m.max().rename("pass_m")
    ru = c[c.market == "rush_yds"].sort_values("m", ascending=False).groupby(["game_id", "snap", "team"]).head(2)
    rus = ru.groupby(["game_id", "snap", "team"]).m.agg(["sum", "size"])
    rus = rus[rus["size"] >= 2]["sum"].rename("rush_m")
    Y = pd.concat([pas, rus], axis=1).dropna()
    Y["Y"] = Y.pass_m + Y.rush_m
    return Y.reset_index()


# ============================================================================================ team totals
def team_totals(fair) -> tuple[pd.DataFrame, pd.DataFrame]:
    q = pd.concat([D.build((2023,), fair), D.build((2024, 2025), fair)], ignore_index=True)
    q = q[q.mk.str.startswith("team_totals") & q.game_id.notna()].copy()
    q["side"] = q.mk.str.split(":").str[1]
    e = q[q.snap == "early"]
    agg = e.groupby(["game_id", "side"]).agg(TT_early=("dcons", "median"), fgm_early=("fg_mean", "median"),
                                             scale=("scale", "first"), X=("X", "first"),
                                             tt_rt=("requested_ts", "max"), n_tt_books=("book", "nunique"))
    c = q[q.snap == "close"].groupby(["game_id", "side"]).agg(TT_close=("imean", "median"),
                                                               fgm_close=("fg_mean", "median"))
    T = agg.join(c, how="outer").reset_index()
    T["TT_early"] = T.TT_early.fillna(T.fgm_early)
    T["TT_close"] = T.TT_close.fillna(T.fgm_close)
    return T, q


# ============================================================================================ game totals
def game_total_quotes(g: pd.DataFrame) -> pd.DataFrame:
    p = SCR / "game_totals.parquet"
    if p.exists():
        return pd.read_parquet(p)
    td = TotalDist.load()
    fr = []
    for s in SEASONS:
        fs = [ODDS / "totals" / f"nfl_odds_{s}.csv.gz"] + sorted((ODDS / "dense").glob(f"nfl_odds_{s}_*.csv.gz"))
        for f in fs:
            o = D._read(f, s)
            if "tot_point" not in o:
                continue
            fr.append(o[["requested_ts", "event_id", "commence", "home", "away", "season", "book", "tot_point",
                         "tot_over_price", "tot_under_price"]])
    t = pd.concat(fr, ignore_index=True).dropna(subset=["tot_point", "tot_over_price", "tot_under_price"])
    t = t.drop_duplicates(["event_id", "requested_ts", "book"])
    ok = (t.tot_over_price.between(-250, 200) & t.tot_under_price.between(-250, 200)
          & (t.tot_over_price.abs() >= 100) & (t.tot_under_price.abs() >= 100) & t.tot_point.between(25, 80))
    t = t[ok].copy()
    ov = D.imp(t.tot_over_price) + D.imp(t.tot_under_price)
    t = t[(ov > 1.0) & (ov < 1.12)].copy()
    t["nv"] = D.imp(t.tot_over_price) / (D.imp(t.tot_over_price) + D.imp(t.tot_under_price))
    med = t.groupby(["event_id", "requested_ts"]).tot_point.transform("median")
    t = t[(t.tot_point - med).abs() <= 5].copy()
    t["hb"] = (t.commence - t.requested_ts).dt.total_seconds() / 3600
    t = t[t.hb > 0]
    gg = D.load_games()
    t = D._match(t, gg[gg.season.isin(SEASONS)])
    # implied expected total per quote (cache by (point, nv rounded))
    key = list(zip(t.tot_point, t.nv.round(4)))
    cache = {}
    vals = []
    for k in key:
        if k not in cache:
            cache[k] = td.implied_mu(k[0], k[1])
        vals.append(cache[k])
    t["mu_q"] = vals
    t.to_parquet(p)
    return t


def game_total_snaps(t: pd.DataFrame, props_rt: pd.DataFrame) -> pd.DataFrame:
    """Per game: early betting snapshot (nearest to the props early request within [-10,+70] min) and close."""
    snaps = t.groupby(["game_id", "requested_ts"]).agg(mu=("mu_q", "median"), hb=("hb", "first"),
                                                         nb=("book", "nunique")).reset_index()
    m = snaps.merge(props_rt, on="game_id")
    m["d"] = (m.requested_ts - m.props_rt).dt.total_seconds() / 60
    m = m[(m.d >= -10) & (m.d <= 70)]
    m["ad"] = m.d.abs()
    e = m.sort_values("ad").drop_duplicates("game_id")[["game_id", "requested_ts", "mu", "d"]] \
        .rename(columns={"requested_ts": "gt_rt", "mu": "GT_early", "d": "gt_lag_min"})
    c = snaps[(snaps.hb > 0) & (snaps.hb <= 3)].sort_values("hb").drop_duplicates("game_id")
    c = c[["game_id", "requested_ts", "mu"]].rename(columns={"requested_ts": "gt_close_rt", "mu": "GT_close"})
    return e.merge(c, on="game_id", how="outer")


def props_requests(q: pd.DataFrame) -> pd.DataFrame:
    """Props EARLY request time per game, from the raw props files (requested_ts is not in the cache)."""
    fr = []
    for s in SEASONS:
        for f in (f"player_pass_yds+player_pass_attempts+player_rush_yds+player_receptions+player_anytime_td_{s}",
                  f"player_reception_yds_{s}"):
            fr.append(pd.read_csv(ODDS / "props" / f"{f}.csv.gz", usecols=["requested_ts", "event_id",
                                                                           "commence_time"]).drop_duplicates())
    r = pd.concat(fr).drop_duplicates(["event_id", "requested_ts"])
    rt, ct = pd.to_datetime(r.requested_ts, utc=True), pd.to_datetime(r.commence_time, utc=True)
    h = (ct - rt).dt.total_seconds() / 3600
    r = r[(h >= 3) & ~((rt.dt.dayofweek == 1) & (rt.dt.hour == 14) & (h > 30))]   # props_full 'early'
    r = r.assign(props_rt=pd.to_datetime(r.requested_ts, utc=True))
    ev = q.drop_duplicates("event_id")[["event_id", "game_id"]]
    return r.merge(ev, on="event_id").groupby("game_id").props_rt.max().reset_index()


# ============================================================================================ signal table
def build_table(fair, cal_td: dict | None = None):
    g = games()
    T, pl = pbp_team_player()
    tg = team_games(g, T)
    q = load_props(g)
    tdq = td_quotes(q, pl)
    if cal_td is None:
        cal_td = fit_td_map(tdq)
    lam = team_lam(tdq, cal_td)
    Y = team_yards(q)
    TT, ttq = team_totals(fair)
    gm = g.set_index("game_id")
    TT["team"] = np.where(TT.side == "home", TT.game_id.map(gm.home_team), TT.game_id.map(gm.away_team))
    tg = tg.merge(TT, on=["game_id", "team", "side"], how="left")
    for sn in SNAPS:
        L = lam[lam.snap == sn].drop(columns="snap").rename(columns={"lam9": f"lam9_{sn}", "n9": f"n9_{sn}",
                                                                      "n_players": f"np_{sn}"})
        tg = tg.merge(L, on=["game_id", "team"], how="left")
        y = Y[Y.snap == sn][["game_id", "team", "Y"]].rename(columns={"Y": f"Y_{sn}"})
        tg = tg.merge(y, on=["game_id", "team"], how="left")
    diag = {"team_src_share": q.drop_duplicates(["game_id", "snap", "market", "player"])
            .groupby(["season", "snap"]).team_src.value_counts(normalize=True).round(4).to_dict()}
    return tg, ttq, q, cal_td, diag, g


def calibrate_constants(tg: pd.DataFrame, cal_td: dict) -> dict:
    d = tg[tg.season.isin(DEV)]
    cal = {"td_map": cal_td, "TOPN": TOPN}
    cal["PPTD"] = float((d.pts - 3 * d.fg_made).sum() / d.all_tds.sum())
    for sn in SNAPS:
        TTc = f"TT_{sn}"
        x = d[d[TTc].notna()]
        b, _, a, _ = ols_cl(x[TTc], 3 * x.fg_made, x.game_id)
        z = x[x[f"lam9_{sn}"].notna()]
        r = float(z.off_tds.sum() / z[f"lam9_{sn}"].sum())
        raw = cal["PPTD"] * r * z[f"lam9_{sn}"] + a + b * z[TTc]
        c0 = float((z[TTc] - raw).mean())
        zy = x[x[f"Y_{sn}"].notna()]
        by, _, ay, _ = ols_cl(zy[f"Y_{sn}"], zy[TTc], zy.game_id)
        cal[sn] = {"fg_f0": a, "fg_f1": b, "r": r, "c0": c0, "n_td_teams": int(len(z)), "yd_a": ay, "yd_b": by,
                   "n_yd_teams": int(len(zy)),
                   "realised_off_tds_mean": float(z.off_tds.mean()), "lam9_mean": float(z[f"lam9_{sn}"].mean())}
    return cal


def apply_signals(tg: pd.DataFrame, cal: dict) -> pd.DataFrame:
    tg = tg.copy()
    for sn in SNAPS:
        c = cal[sn]
        TT = tg[f"TT_{sn}"]
        tg[f"P_TD_{sn}"] = cal["PPTD"] * c["r"] * tg[f"lam9_{sn}"] + c["fg_f0"] + c["fg_f1"] * TT + c["c0"]
        tg[f"P_YD_{sn}"] = c["yd_a"] + c["yd_b"] * tg[f"Y_{sn}"]
        for s in ("TD", "YD"):
            tg[f"gap_{s}_{sn}"] = tg[f"P_{s}_{sn}"] - TT
    return tg


# ============================================================================================ bets
def team_bets(tg: pd.DataFrame, ttq: pd.DataFrame, fair, sig: str, books: set, thr: float = THR_TEAM) -> pd.DataFrame:
    gap = tg[f"gap_{sig}_early"]
    sel = tg[gap.abs() >= thr].copy()
    sel["bet_side"] = np.where(sel[f"gap_{sig}_early"] > 0, "over", "under")
    e = ttq[(ttq.snap == "early") & ttq.book.isin(books)].copy()
    e = e.drop(columns=["season"]).merge(sel[["game_id", "side", "bet_side", f"gap_{sig}_early", "TT_early",
                                              "TT_close", "season", "week"]], on=["game_id", "side"])
    if e.empty:
        return e
    mk = "team_totals:home"  # same team-points pmf for home/away
    gt, eq, lt = D.probs(fair, mk, e.TT_early.values, e.x.values, np.full(len(e), 44.0))
    e["dec"] = np.where(e.bet_side == "over", D.dec(e.over_price), D.dec(e.under_price))
    e["ev_early"] = np.where(e.bet_side == "over", gt, lt) * e.dec + eq - 1
    e = e.sort_values("ev_early", ascending=False).drop_duplicates(["game_id", "side"])
    # close valuation
    c = ttq[ttq.snap == "close"]
    direct = c.groupby(["game_id", "side", "x"]).q_over.median().rename("q_close").reset_index()
    e = e.merge(direct, on=["game_id", "side", "x"], how="left")
    gt, eq, lt = D.probs(fair, mk, e.TT_close.values, e.x.values, np.full(len(e), 44.0))
    p_over_dir = e.q_close * (1 - eq)
    p_under_dir = (1 - e.q_close) * (1 - eq)
    pw = np.where(e.bet_side == "over", np.where(e.q_close.notna(), p_over_dir, gt),
                  np.where(e.q_close.notna(), p_under_dir, lt))
    e["clv"] = pw * e.dec + eq - 1
    e["clv_src"] = np.where(e.q_close.notna(), "direct", np.where(np.isfinite(e.TT_close), "pmf", "none"))
    e.loc[e.clv_src == "none", "clv"] = np.nan
    win = np.where(e.bet_side == "over", e.X > e.x, e.X < e.x)
    push = e.X == e.x
    e["pnl"] = np.where(push, 0.0, np.where(win, e.dec - 1, -1.0))
    e["win"] = np.where(push, np.nan, win.astype(float))
    e["gap"] = e[f"gap_{sig}_early"]
    return e[["game_id", "season", "week", "side", "book", "x", "bet_side", "dec", "ev_early", "gap", "TT_early",
              "TT_close", "X", "clv", "clv_src", "pnl", "win"]]


def game_bets(tg: pd.DataFrame, gtq: pd.DataFrame, gsn: pd.DataFrame, g: pd.DataFrame, sig: str,
              books: set, thr: float = THR_GAME) -> pd.DataFrame:
    td = TotalDist.load()
    w = tg.pivot_table(index="game_id", columns="side", values=f"gap_{sig}_early")
    w = w.dropna(subset=["home", "away"]) if {"home", "away"} <= set(w.columns) else w.iloc[:0]
    w["gapG"] = w.home + w.away
    sel = w[w.gapG.abs() >= thr].reset_index()[["game_id", "gapG"]]
    sel["bet_side"] = np.where(sel.gapG > 0, "over", "under")
    sel = sel.merge(gsn, on="game_id", how="inner").dropna(subset=["gt_rt"])
    e = gtq[gtq.book.isin(books)].merge(sel, left_on=["game_id", "requested_ts"], right_on=["game_id", "gt_rt"])
    if e.empty:
        return e
    e["dec"] = np.where(e.bet_side == "over", D.dec(e.tot_over_price), D.dec(e.tot_under_price))
    e["ev_early"] = [td.probs(mu, pt, sd)[0] * dc + td.probs(mu, pt, sd)[1] - 1
                     for mu, pt, sd, dc in zip(e.GT_early, e.tot_point, e.bet_side, e.dec)]
    e = e.sort_values("ev_early", ascending=False).drop_duplicates("game_id")
    c = gtq.merge(gsn[["game_id", "gt_close_rt"]], left_on=["game_id", "requested_ts"],
                  right_on=["game_id", "gt_close_rt"])
    direct = c.groupby(["game_id", "tot_point"]).nv.median().rename("q_close").reset_index()
    e = e.merge(direct, on=["game_id", "tot_point"], how="left")
    clv, src = [], []
    for r in e.itertuples():
        if not np.isfinite(r.GT_close):
            clv.append(np.nan)
            src.append("none")
            continue
        pw, pu = td.probs(r.GT_close, r.tot_point, r.bet_side)
        if np.isfinite(r.q_close):
            q = r.q_close if r.bet_side == "over" else 1 - r.q_close
            pw = q * (1 - pu)
            src.append("direct")
        else:
            src.append("pmf")
        clv.append(pw * r.dec + pu - 1)
    e["clv"], e["clv_src"] = clv, src
    sc = g.set_index("game_id")
    e["total_pts"] = e.game_id.map(sc.home_score + sc.away_score)
    e["season"] = e.game_id.map(sc.season)
    e["week"] = e.game_id.map(sc.week)
    win = np.where(e.bet_side == "over", e.total_pts > e.tot_point, e.total_pts < e.tot_point)
    push = e.total_pts == e.tot_point
    e["pnl"] = np.where(push, 0.0, np.where(win, e.dec - 1, -1.0))
    e["win"] = np.where(push, np.nan, win.astype(float))
    e["gap"] = e.gapG
    return e[["game_id", "season", "week", "book", "tot_point", "bet_side", "dec", "ev_early", "gap", "GT_early",
              "GT_close", "total_pts", "clv", "clv_src", "pnl", "win", "gt_lag_min"]]


def summarize(b: pd.DataFrame) -> dict:
    if b is None or len(b) == 0:
        return {"n": 0}
    g = b.game_id.to_numpy()
    cm, cse, cn = clustered(b.clv, g)
    rm, rse, _ = clustered(b.pnl, g)
    t = cm / cse if cse and cse > 0 else np.nan
    return {"n": int(len(b)), "n_clv": cn, "clv": cm, "clv_se": cse, "clv_t": t, "clv_p": pval(t),
            "roi": rm, "roi_se": rse, "hit": float(np.nanmean(b.win)),
            "over_share": float((b.bet_side == "over").mean()), "mean_abs_gap": float(b.gap.abs().mean()),
            "direct_share": float((b.clv_src == "direct").mean()),
            "bets_per_week": round(len(b) / max(b.groupby(["season", "week"]).ngroups, 1), 2),
            "books": b.book.value_counts().to_dict()}


def by_season(b: pd.DataFrame, seasons) -> dict:
    out = {str(s): summarize(b[b.season == s]) for s in seasons}
    if len(seasons) > 1:
        out["pooled"] = summarize(b[b.season.isin(seasons)])
    return out


# ============================================================================================ lead-lag
def lead_lag(tg: pd.DataFrame, gsn: pd.DataFrame, seasons, B=2000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    d = tg[tg.season.isin(seasons)]
    res = {}

    def one(x, ym, yp, gid):
        ok = np.isfinite(x) & np.isfinite(ym) & np.isfinite(yp)
        x, ym, yp, gid = x[ok], ym[ok], yp[ok], gid[ok]
        if len(x) < 30:
            return {"n": int(len(x))}
        bm, sm, _, n = ols_cl(x, ym, gid)
        bp, sp, _, _ = ols_cl(-x, yp, gid)
        u, gi = np.unique(gid, return_inverse=True)
        idx = [np.flatnonzero(gi == k) for k in range(len(u))]
        diffs = []
        for _ in range(B):
            s = np.concatenate([idx[k] for k in rng.integers(0, len(u), len(u))])
            X = np.column_stack([np.ones(len(s)), x[s]])
            b1 = np.linalg.lstsq(X, ym[s], rcond=None)[0][1]
            X2 = np.column_stack([np.ones(len(s)), -x[s]])
            b2 = np.linalg.lstsq(X2, yp[s], rcond=None)[0][1]
            diffs.append(b1 - b2)
        diffs = np.asarray(diffs)
        return {"n": n, "beta_mkt": bm, "beta_mkt_se": sm, "beta_mkt_p": pval(bm / sm),
                "beta_props": bp, "beta_props_se": sp, "beta_props_p": pval(bp / sp),
                "diff": bm - bp, "diff_boot_se": float(diffs.std()), "diff_boot_p_le0": float((diffs <= 0).mean()),
                "sd_gap": float(x.std()), "sd_dmkt": float(ym.std()), "sd_dprops": float(yp.std()),
                "mean_dmkt_when_gap_ge_thr": float(ym[x >= THR_TEAM].mean()) if (x >= THR_TEAM).any() else None,
                "mean_dmkt_when_gap_le_-thr": float(ym[x <= -THR_TEAM].mean()) if (x <= -THR_TEAM).any() else None}
    for sig in ("TD", "YD"):
        x = d[f"gap_{sig}_early"].to_numpy(float)
        ym = (d.TT_close - d.TT_early).to_numpy(float)
        yp = (d[f"P_{sig}_close"] - d[f"P_{sig}_early"]).to_numpy(float)
        res[f"team_{sig}"] = one(x, ym, yp, d.game_id.to_numpy())
        # game level: game-total consensus move vs summed gaps
        w = d.pivot_table(index="game_id", columns="side",
                          values=[f"gap_{sig}_early", f"P_{sig}_early", f"P_{sig}_close"])
        try:
            G = pd.DataFrame({"gapG": w[(f"gap_{sig}_early", "home")] + w[(f"gap_{sig}_early", "away")],
                              "dP": (w[(f"P_{sig}_close", "home")] + w[(f"P_{sig}_close", "away")])
                              - (w[(f"P_{sig}_early", "home")] + w[(f"P_{sig}_early", "away")])})
        except KeyError:
            res[f"game_{sig}"] = {"n": 0}
            continue
        G = G.join(gsn.set_index("game_id")[["GT_early", "GT_close"]], how="inner")
        res[f"game_{sig}"] = one(G.gapG.to_numpy(float), (G.GT_close - G.GT_early).to_numpy(float),
                                 G.dP.to_numpy(float), G.index.to_numpy())
    return res


def descriptives(tg: pd.DataFrame, seasons) -> dict:
    d = tg[tg.season.isin(seasons)]
    out = {}
    for sig in ("TD", "YD"):
        for sn in SNAPS:
            z = d[[f"gap_{sig}_{sn}", f"TT_{sn}", f"P_{sig}_{sn}", "pts", "game_id", "season"]].dropna()
            if len(z) < 30:
                continue
            res_b, res_se, _, _ = ols_cl(z[f"gap_{sig}_{sn}"], z.pts - z[f"TT_{sn}"], z.game_id)
            out[f"{sig}_{sn}"] = {
                "n": int(len(z)), "mean_gap": float(z[f"gap_{sig}_{sn}"].mean()),
                "sd_gap": float(z[f"gap_{sig}_{sn}"].std()),
                "mean_gap_by_season": z.groupby("season")[f"gap_{sig}_{sn}"].mean().round(3).to_dict(),
                "share_abs_gap_ge_1.5": float((z[f"gap_{sig}_{sn}"].abs() >= THR_TEAM).mean()),
                "corr_P_TT": float(np.corrcoef(z[f"P_{sig}_{sn}"], z[f"TT_{sn}"])[0, 1]),
                "corr_gap_TT": float(np.corrcoef(z[f"gap_{sig}_{sn}"], z[f"TT_{sn}"])[0, 1]),
                "mae_TT_vs_pts": float((z.pts - z[f"TT_{sn}"]).abs().mean()),
                "mae_P_vs_pts": float((z.pts - z[f"P_{sig}_{sn}"]).abs().mean()),
                "slope_pts_resid_on_gap": res_b, "slope_se": res_se}
    return out


def coverage(tg: pd.DataFrame, gsn: pd.DataFrame, seasons) -> dict:
    d = tg[tg.season.isin(seasons)]
    out = {}
    for s, z in d.groupby("season"):
        out[str(s)] = {"team_games": int(len(z)), "TT_early": int(z.TT_early.notna().sum()),
                       "TT_close": int(z.TT_close.notna().sum()),
                       "TD_early": int(z.gap_TD_early.notna().sum()), "TD_close": int(z.gap_TD_close.notna().sum()),
                       "YD_early": int(z.gap_YD_early.notna().sum()), "YD_close": int(z.gap_YD_close.notna().sum()),
                       "mean_players_listed_early": float(z.np_early.mean()),
                       "games_with_gt_early_snapshot": int(gsn[gsn.game_id.isin(z.game_id)].GT_early.notna().sum()),
                       "games_with_gt_close": int(gsn[gsn.game_id.isin(z.game_id)].GT_close.notna().sum())}
    return out


# ============================================================================================ stages
def run(stage: str):
    SCR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fair = D.load_fair()
    if stage == "holdout":
        if RESULT.exists():
            raise SystemExit(f"{RESULT} exists: the holdout has already been run once")
        cal = json.loads(CALIB.read_text())["calibration"]
        tg, ttq, q, _, diag, g = build_table(fair, cal["td_map"])
    else:
        tg, ttq, q, cal_td, diag, g = build_table(fair)
        cal = calibrate_constants(tg, cal_td)
    tg = apply_signals(tg, cal)
    props_rt = props_requests(q)
    gtq = game_total_quotes(g)
    gsn = game_total_snaps(gtq, props_rt)
    seasons = DEV if stage == "calibrate" else HOLD
    res = {"stage": stage, "definitions": __doc__, "coverage": coverage(tg, gsn, SEASONS if stage == "holdout"
                                                                         else DEV),
           "team_src_share": {str(k): v for k, v in diag["team_src_share"].items()},
           "descriptives": descriptives(tg, seasons), "lead_lag": lead_lag(tg, gsn, seasons), "rules": {},
           "rules_my_books_info": {}}
    if stage == "holdout":
        res["descriptives_2023_with_frozen_constants"] = descriptives(tg, DEV)
    for r in RULES:
        kind, sig = r.split("_")
        for lab, books in (("rules", AZ), ("rules_my_books_info", MYB)):
            b = team_bets(tg, ttq, fair, sig, books) if kind == "P1" else game_bets(tg, gtq, gsn, g, sig, books)
            b = b[b.season.isin(seasons)] if len(b) else b
            rr = by_season(b, seasons)
            if lab == "rules" and stage == "holdout":
                h = rr["pooled"]
                rr["pass"] = bool(h.get("n", 0) >= 100 and h["clv"] > 0 and h["clv_p"] < 0.05 / K_TESTS
                                  and (rr["2024"].get("clv") or -1) > 0 and (rr["2025"].get("clv") or -1) > 0)
            res[lab][r] = rr
            if lab == "rules" and len(b):
                b.to_csv(SCR / f"bets_{stage}_{r}.csv", index=False)
        # informational dose-response (NOT tested): sign of the gap only (threshold 0), AZ books
        b = team_bets(tg, ttq, fair, sig, AZ, 0.0) if kind == "P1" else game_bets(tg, gtq, gsn, g, sig, AZ, 0.0)
        b = b[b.season.isin(seasons)] if len(b) else b
        res.setdefault("info_threshold0", {})[r] = by_season(b, seasons)
    tg.to_parquet(SCR / f"teamgames_{stage}.parquet")
    if stage == "calibrate":
        out = {"calibration": cal, **res}
        CALIB.write_text(json.dumps(js(out), indent=1))
    else:
        res["calibration_used"] = cal
        RESULT.write_text(json.dumps(js(res), indent=1))
    show = {k: v for k, v in res.items() if k not in ("definitions",)}
    print(json.dumps(js(show), indent=1)[:20000])
    if stage == "calibrate":
        print(json.dumps(js(cal), indent=1))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "calibrate")
