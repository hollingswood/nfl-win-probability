"""Market-implied power ratings: what moves them, and where the market over/under-reacts (research only).

Idea: pros don't out-predict the market, they find where it OVER- or UNDER-reacts. Each week we filter team
ratings + HFA out of that week's lines (Kalman random walk, lines = near-noiseless observations of
r_home - r_away + HFA). For every game we then have
    prior  = the line implied by ratings BEFORE either team's last game result (one-step-ahead prediction)
    adj    = close - prior              (how much the market moved the matchup since last week)
    ats    = margin - close             (what the market still got wrong)
    true   = margin - prior = adj + ats (what the last week's information was actually worth)
Regressing adj / ats / true on decomposed last-game information (home minus away) gives, per component,
the market's weight, the error in that weight, and the "correct" weight; the three coefficients add up.

Components of a team's LAST game (team perspective): result surprise (margin - closing line), efficiency
EPA surprise (pass/rush EPA net, turnover plays excluded, residualised on the line), turnovers (INT net,
fumble-recovery luck, fumbles-forced net), non-offensive TDs (return/defensive/ST), garbage-time points,
close-game W/L, overtime, primetime (national TV) and public-team interactions. QB starter changes are
"injuries known": games where either team's starter changed are excluded from the core sample.

Discipline
  * closing-based tests developed on 2003-2019 (EPA components need pbp: 2012-2019), 2020-22 reported as a
    second dev block; early-line (look-ahead / re-open / close) tests developed on 2020-22 with price-based
    CLV (edge_lab.closing_fair mu_close_all) at the best allowed-book price (my_books.json).
  * <= 3 rules frozen into output/research/market_ratings_frozen.json (`freeze`, never overwritten), then
    2023-2025 run ONCE (`holdout`, needs EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES).

    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/market_ratings.py dev
    PYTHONPATH=src:scripts python scripts/research/market_ratings.py freeze
    EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/market_ratings.py holdout
    PYTHONPATH=src:scripts python scripts/research/market_ratings.py report
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import edge_lab as E  # noqa: E402
from nflpred import margins as K, odds as O, spread_bets as SB  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "market_ratings.json"
MD = OUT / "market_ratings.md"
FROZEN = OUT / "market_ratings_frozen.json"
SCRATCH = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/mktrat")
HIST = ROOT / "data" / "historical_odds"
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
PUBLIC = {"DAL", "KC", "GB", "PIT", "NE", "SF", "PHI"}
DEV_C, DEV_C2, DEV_E, HOLD = tuple(range(2003, 2020)), (2020, 2021, 2022), (2020, 2021, 2022), (2023, 2024, 2025)
SPR = SB.load_rules()
W, SIG = SPR["_weights"], SPR["margin"]["sigma"]
BREAKEVEN = 0.5238


# ============================================================================ games
def load_games() -> pd.DataFrame:
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[g.home_score.notna() & g.spread_line.notna()].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    g["gameday"] = pd.to_datetime(g.gameday)
    g["gametime"] = g.gametime.fillna("13:00")
    g["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()], utc=True)
    g["margin"] = g.home_score - g.away_score
    g["neutral"] = (g.location == "Neutral").astype(float)
    g["wk"] = np.where(g.game_type == "REG", g.week, g.week)  # playoffs continue the week count
    hh = g.gametime.fillna("13:00").str.slice(0, 2).astype(int)
    g["prime"] = ((hh >= 19) | g.weekday.isin(["Thursday", "Monday"])) & (g.game_type == "REG")
    return g.sort_values(["kick", "game_id"]).reset_index(drop=True)


# ============================================================================ Kalman market ratings
def kalman(g: pd.DataFrame, s_obs=1.5, q_w=0.25, rho=0.75, q_off=4.0, line_col="spread_line"):
    """Forward filter over (season, week). Returns per-game prior mean/var of the line (before that week's
    lines are absorbed) and the per-(season, week, team) posterior rating (after the week's lines)."""
    teams = sorted(set(g.home_team) | set(g.away_team))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    x = np.zeros(n + 1)
    x[n] = 2.5
    P = np.eye(n + 1) * 36.0
    P[n, n] = 1.0
    prior_mu = np.full(len(g), np.nan)
    prior_var = np.full(len(g), np.nan)
    post = []
    last_season = None
    for (s, w), d in g.groupby(["season", "wk"], sort=True):
        if last_season is not None and s != last_season:
            m = x[:n].mean()
            x[:n] = m + rho * (x[:n] - m)
            P[:n, :n] = rho ** 2 * P[:n, :n] + q_off * np.eye(n)
            P[n, n] += 0.25
        elif last_season is not None:
            P[:n, :n] += q_w * np.eye(n)
            P[n, n] += 0.002
        last_season = s
        hs = np.array([ix[t] for t in d.home_team])
        as_ = np.array([ix[t] for t in d.away_team])
        nh = 1.0 - d.neutral.values
        H = np.zeros((len(d), n + 1))
        H[np.arange(len(d)), hs] = 1.0
        H[np.arange(len(d)), as_] = -1.0
        H[:, n] = nh
        prior_mu[d.index.values] = H @ x
        prior_var[d.index.values] = np.einsum("ij,jk,ik->i", H, P, H)
        y = d[line_col].values
        Rm = np.eye(len(d)) * s_obs ** 2
        Sm = H @ P @ H.T + Rm
        Kg = P @ H.T @ np.linalg.inv(Sm)
        x = x + Kg @ (y - H @ x)
        P = (np.eye(n + 1) - Kg @ H) @ P
        P = 0.5 * (P + P.T)
        c = x[:n].mean()
        for t in set(d.home_team) | set(d.away_team):
            post.append((s, w, t, x[ix[t]] - c, x[n]))
    rat = pd.DataFrame(post, columns=["season", "wk", "team", "rating", "hfa"])
    return prior_mu, prior_var, rat


def tune_kalman(g: pd.DataFrame) -> dict:
    """Grid on one-step-ahead LINE prediction MSE (not outcomes), dev seasons only."""
    dev = g.season.isin(DEV_C).values
    best = None
    rows = []
    for s_obs in (0.75, 1.0, 1.5):
        for q_w in (0.25, 0.5, 0.75, 1.0, 1.5):
            for rho, q_off in ((0.45, 4.0), (0.6, 2.0), (0.6, 4.0), (0.75, 4.0), (0.75, 9.0)):
                pm, _, _ = kalman(g, s_obs, q_w, rho, q_off)
                e = (g.spread_line.values - pm)
                wk1 = dev & (g.week.values == 1)
                rest = dev & (g.week.values > 1) & (g.game_type.values == "REG")
                mse_rest, mse_w1 = float(np.mean(e[rest] ** 2)), float(np.mean(e[wk1] ** 2))
                rows.append({"s_obs": s_obs, "q_w": q_w, "rho": rho, "q_off": q_off,
                             "mse_wk2plus": round(mse_rest, 3), "mse_wk1": round(mse_w1, 3)})
    df = pd.DataFrame(rows)
    b = df.sort_values("mse_wk2plus").iloc[0]
    # season transition picked on week-1 MSE given the in-season pick
    sub = df[(df.s_obs == b.s_obs) & (df.q_w == b.q_w)].sort_values("mse_wk1").iloc[0]
    return {"s_obs": float(b.s_obs), "q_w": float(b.q_w), "rho": float(sub.rho), "q_off": float(sub.q_off),
            "grid": rows}


# ============================================================================ per team-game components (pbp)
def pbp_components(seasons) -> pd.DataFrame:
    cols = ["game_id", "play_id", "posteam", "defteam", "pass", "rush", "qb_kneel", "qb_spike", "epa", "interception",
            "fumble", "fumble_lost", "fumbled_1_team", "touchdown", "td_team", "special_teams_play", "qtr",
            "home_wp", "total_home_score", "total_away_score", "home_team", "away_team"]
    out = []
    for s in seasons:
        p = ROOT / "data" / "raw" / f"pbp_{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=cols)
        for c in ("posteam", "defteam", "fumbled_1_team", "td_team", "home_team", "away_team"):
            d[c] = d[c].replace(TEAM_MAP)
        d = d.sort_values(["game_id", "play_id"])
        sc = d[(d["pass"].eq(1) | d["rush"].eq(1)) & d.epa.notna() & d.qb_kneel.ne(1) & d.qb_spike.ne(1)].copy()
        sc["int"] = sc.interception.fillna(0)
        sc["fum_own"] = (sc.fumble.fillna(0) == 1) & (sc.fumbled_1_team == sc.posteam)
        sc["fum_lost_own"] = sc.fum_own & (sc.fumble_lost.fillna(0) == 1)
        sc["to_play"] = (sc["int"] == 1) | sc.fum_lost_own
        sc["eff_epa"] = np.where(sc.to_play, np.nan, sc.epa)
        off = sc.groupby(["game_id", "posteam"]).agg(
            eff=("eff_epa", "sum"), eff_n=("eff_epa", "count"), epa_all=("epa", "sum"), plays=("epa", "size"),
            ints=("int", "sum"), fum=("fum_own", "sum"), fl=("fum_lost_own", "sum")).reset_index()
        off = off.rename(columns={"posteam": "team"})
        # points scored per play (score columns are cumulative after the play)
        d["dh"] = d.groupby("game_id").total_home_score.diff().fillna(d.total_home_score).clip(lower=0)
        d["da"] = d.groupby("game_id").total_away_score.diff().fillna(d.total_away_score).clip(lower=0)
        gar = d[(d.qtr >= 4) & ((d.home_wp < 0.1) | (d.home_wp > 0.9))]
        gsum = gar.groupby("game_id").agg(gh=("dh", "sum"), ga=("da", "sum"), home=("home_team", "first"))
        nt = d[(d.touchdown == 1) & ((d.special_teams_play == 1) | (d.td_team != d.posteam)) & d.td_team.notna()]
        ntd = nt.groupby(["game_id", "td_team"]).size().rename("ntd").reset_index().rename(columns={"td_team": "team"})
        gm = d.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first")).reset_index()
        long = pd.concat([gm.rename(columns={"home": "team", "away": "opp"}).assign(is_home=1),
                          gm.rename(columns={"away": "team", "home": "opp"}).assign(is_home=0)], ignore_index=True)
        long = long.merge(off, on=["game_id", "team"], how="left")
        o2 = off.rename(columns={"team": "opp", **{c: "o_" + c for c in off.columns if c not in ("game_id", "team")}})
        long = long.merge(o2, on=["game_id", "opp"], how="left")
        long = long.merge(ntd, on=["game_id", "team"], how="left").merge(
            ntd.rename(columns={"team": "opp", "ntd": "o_ntd"}), on=["game_id", "opp"], how="left")
        long = long.merge(gsum[["gh", "ga"]].reset_index(), on="game_id", how="left")
        long[["ntd", "o_ntd", "gh", "ga"]] = long[["ntd", "o_ntd", "gh", "ga"]].fillna(0)
        r = pd.DataFrame({"game_id": long.game_id, "team": long.team})
        r["eff_net"] = long.eff - long.o_eff
        r["epa_net"] = long.epa_all - long.o_epa_all
        r["epa_pp_net"] = long.epa_all / long.plays - long.o_epa_all / long.o_plays
        r["int_net"] = long.o_ints - long.ints
        r["fum_luck"] = (long.o_fl - 0.5 * long.o_fum) - (long.fl - 0.5 * long.fum)
        r["fum_cnt_net"] = 0.5 * (long.o_fum - long.fum)
        r["to_net"] = r.int_net + (long.o_fl - long.fl)
        r["ntd_net"] = long.ntd - long.o_ntd
        r["garb_net"] = np.where(long.is_home == 1, long.gh - long.ga, long.ga - long.gh)
        out.append(r)
    return pd.concat(out, ignore_index=True)


# ============================================================================ panel
def team_games(g: pd.DataFrame, rat: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with team-perspective line/result, ordered within season."""
    base = ["game_id", "season", "week", "wk", "game_type", "kick", "prime", "spread_line", "margin", "overtime",
            "home_spread_odds", "away_spread_odds"]
    h = g[base + ["home_team", "away_team", "home_qb_id"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_qb_id": "qb"}).assign(is_home=1)
    a = g[base + ["away_team", "home_team", "away_qb_id"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_qb_id": "qb"}).assign(is_home=0)
    a["spread_line"] = -a.spread_line
    a["margin"] = -a.margin
    t = pd.concat([h, a], ignore_index=True).sort_values(["team", "kick"])
    t["ats"] = t.margin - t.spread_line
    t = t.merge(rat[["season", "wk", "team", "rating"]], on=["season", "wk", "team"], how="left")
    t["n_in_season"] = t.groupby(["team", "season"]).cumcount()
    return t.sort_values(["team", "kick"]).reset_index(drop=True)


COMP = ["ats", "eff_s", "int_net", "fum_luck", "fum_cnt_net", "ntd_net", "garb_net", "close_wl", "ot",
        "ats_prime", "ats_pub", "ats_big"]


def build_panel(params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache = SCRATCH / "panel.parquet"
    tcache = SCRATCH / "teamgames.parquet"
    pcache = SCRATCH / "kalman.json"
    if cache.exists() and tcache.exists() and pcache.exists():
        return pd.read_parquet(cache), pd.read_parquet(tcache), json.loads(pcache.read_text())
    g = load_games()
    if params is None:
        params = tune_kalman(g)
    pm, pv, rat = kalman(g, params["s_obs"], params["q_w"], params["rho"], params["q_off"])
    g["prior"] = pm
    g["prior_sd"] = np.sqrt(pv)
    t = team_games(g, rat)
    comp = pbp_components(range(2012, 2027))
    t = t.merge(comp, on=["game_id", "team"], how="left")
    # efficiency surprise: net non-turnover EPA residualised on the team's closing line (fit on 2012-19 only)
    m = t.season.between(2012, 2019) & t.eff_net.notna() & (t.game_type == "REG")
    b, a = np.polyfit(t.loc[m, "spread_line"], t.loc[m, "eff_net"], 1)
    params["eff_fit"] = {"slope": float(b), "intercept": float(a)}
    t["eff_s"] = t.eff_net - (a + b * t.spread_line)
    t["close_wl"] = np.where(t.margin.abs() <= 7, np.sign(t.margin), 0.0)
    t["ot"] = t.overtime.fillna(0) * np.sign(t.margin)
    t["ats_prime"] = t.ats * t.prime.astype(float)
    t["ats_pub"] = t.ats * t.team.isin(PUBLIC).astype(float)
    t["ats_big"] = np.sign(t.ats) * np.clip(t.ats.abs() - 14, 0, None)      # excess beyond a 14-pt surprise
    # previous game of the same team in the same season
    grp = t.groupby(["team", "season"])
    for c in COMP + ["qb", "rating", "kick", "spread_line", "margin", "eff_net", "epa_net", "to_net", "prime"]:
        t[f"p_{c}"] = grp[c].shift(1)
    for k in (2, 3):
        t[f"p{k}_rating"] = grp.rating.shift(k)
    t["p_ats2"] = grp.ats.shift(2)
    t["qb_chg"] = (t.qb != t.p_qb) & t.p_qb.notna()
    # season-to-date (strictly prior) means for slow-moving tests
    for c in ("eff_net", "ats", "margin"):
        t[f"std_{c}"] = grp[c].transform(lambda x: x.shift(1).expanding().mean())
    # standings to date (REG only) for motivation tests
    reg = t.game_type == "REG"
    t["win"] = np.where(t.margin > 0, 1.0, np.where(t.margin == 0, 0.5, 0.0))
    t.loc[reg, "w_to_date"] = t[reg].groupby(["team", "season"]).win.transform(lambda x: x.shift(1).cumsum().fillna(0))
    t.loc[reg, "gp"] = t[reg].groupby(["team", "season"]).cumcount()
    n_reg = g[g.game_type == "REG"].groupby("season").size() * 2 / g[g.game_type == "REG"].groupby("season").apply(
        lambda d: len(set(d.home_team) | set(d.away_team)))
    t["n_reg"] = t.season.map(n_reg.round())
    t["dead"] = reg & (t.w_to_date + (t.n_reg - t.gp) < t.n_reg / 2)
    # last season's final market rating and point differential per game (season-start tests)
    last = t[t.game_type == "REG"].groupby(["team", "season"]).agg(
        end_rating=("rating", "last"), pd_pg=("margin", "mean"), ats_pg=("ats", "mean")).reset_index()
    last["season"] += 1
    t = t.merge(last.rename(columns={"end_rating": "ly_rating", "pd_pg": "ly_pd", "ats_pg": "ly_ats"}),
                on=["team", "season"], how="left")
    # game-level panel (home row joined with away row)
    keep = ["game_id", "team"] + [c for c in t.columns if c.startswith(("p_", "p2_", "p3_", "std_", "ly_"))] + \
        ["qb_chg", "rating", "dead", "n_in_season", "w_to_date", "gp", "eff_net"]
    hh = t[t.is_home == 1][keep].rename(columns={c: "h_" + c for c in keep if c != "game_id"})
    aa = t[t.is_home == 0][keep].rename(columns={c: "a_" + c for c in keep if c != "game_id"})
    panel = g.merge(hh, on="game_id").merge(aa, on="game_id")
    panel["adj"] = panel.spread_line - panel.prior
    panel["ats_res"] = panel.margin - panel.spread_line
    panel["true"] = panel.margin - panel.prior
    for c in COMP:
        panel[f"d_{c}"] = panel[f"h_p_{c}"] - panel[f"a_p_{c}"]
    panel.to_parquet(cache)
    t.to_parquet(tcache)
    pcache.write_text(json.dumps(params))
    return panel, t, params


# ============================================================================ stats helpers
def ols(df: pd.DataFrame, y: str, xs: list[str], cluster: str | None = "cl") -> dict:
    """OLS with intercept and cluster-robust (by season-week) SEs."""
    d = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in xs])
    Y = d[y].values.astype(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    e = Y - X @ beta
    if cluster and cluster in d:
        meat = np.zeros((X.shape[1], X.shape[1]))
        for _, idx in d.groupby(cluster).indices.items():
            s = X[idx].T @ e[idx]
            meat += np.outer(s, s)
        G = d[cluster].nunique()
        k = X.shape[1]
        meat *= G / (G - 1) * (len(d) - 1) / (len(d) - k)
    else:
        meat = (X * e[:, None] ** 2).T @ X * len(d) / (len(d) - X.shape[1])
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - e.var() / Y.var() if Y.var() > 0 else float("nan")
    names = ["const"] + xs
    return {"n": int(len(d)), "r2": round(float(r2), 4),
            "coef": {k: [round(float(b), 4), round(float(s), 4)] for k, b, s in zip(names, beta, se)}}


def triple(df, xs, label=""):
    """Same regressors on adj / ats / true (coefficients add: true = adj + ats)."""
    return {"adj": ols(df, "adj", xs), "ats": ols(df, "ats_res", xs), "true": ols(df, "true", xs)}


def ats_record(b: pd.DataFrame) -> dict:
    """b: rows with side ('home'/'away'), margin, spread_line, home/away_spread_odds. Closing line, actual juice."""
    if len(b) == 0:
        return {"bets": 0}
    home = (b.side == "home").values
    res = np.where(home, b.margin - b.spread_line, -(b.margin - b.spread_line))
    price = np.where(home, b.home_spread_odds, b.away_spread_odds)
    price = np.where(np.isnan(price.astype(float)), -110.0, price.astype(float))
    dec = E._dec(price)
    pnl = np.where(res > 0, dec - 1, np.where(res < 0, -1.0, 0.0))
    dec_n = res != 0
    wins = int((res > 0).sum())
    n = int(dec_n.sum())
    p_one = 0.5 * math.erfc(((wins / max(n, 1)) - BREAKEVEN) / math.sqrt(BREAKEVEN * (1 - BREAKEVEN) / max(n, 1)) /
                            math.sqrt(2)) if n else 1.0
    return {"bets": int(len(b)), "per_season": round(len(b) / b.season.nunique(), 1), "wins": wins,
            "losses": int((res < 0).sum()), "pushes": int((res == 0).sum()),
            "win_rate": round(wins / n, 4) if n else None, "p_vs_52.4": round(p_one, 4),
            "roi": round(float(pnl.mean()), 4), "roi_se": round(float(pnl.std(ddof=1) / math.sqrt(len(pnl))), 4)}


# ============================================================================ early / look-ahead lines (2020+)
def _read_odds(kind: str, seasons) -> pd.DataFrame:
    fr = []
    for s in seasons:
        p = (HIST / kind if kind else HIST) / f"nfl_odds_{s}.csv.gz"
        if p.exists():
            fr.append(pd.read_csv(p).assign(season=s, src=kind or "main"))
    o = pd.concat(fr, ignore_index=True)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    return o


def _mu_px(d: pd.DataFrame) -> float:
    v = K.market_mu(zip(d.sp_home_point, d.sp_home_price, d.sp_away_point, d.sp_away_price), SIG, W)
    return np.nan if v is None else v


def _ev_spread(mu_home, side, point, dec):
    mu_home = np.asarray(mu_home, float)
    point = np.asarray(point, float)
    dec = np.asarray(dec, float)
    ok = ~np.isnan(mu_home) & ~np.isnan(point) & ~np.isnan(dec)
    home_line = np.where(side == "home", point, -point)
    hc, pu, ac = K.cover_probs(np.nan_to_num(mu_home), SIG, np.nan_to_num(home_line), W)
    pw = np.where(side == "home", hc, ac)
    return np.where(ok, pw * dec + pu - 1, np.nan)


def early_table(panel: pd.DataFrame, seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per game: look-ahead line (last snapshot before EITHER team's previous game kicked off), early line
    (first snapshot after BOTH previous games ended, kick+4h), price-implied close. Plus the allowed-book
    offers at the early snapshot (one row per book/side) for betting."""
    tag = "_".join(str(s) for s in seasons)
    c1, c2 = SCRATCH / f"early_{tag}.parquet", SCRATCH / f"offers_{tag}.parquet"
    if c1.exists() and c2.exists():
        return pd.read_parquet(c1), pd.read_parquet(c2)
    g = panel[panel.season.isin(seasons) & (panel.game_type == "REG")].copy()
    o = pd.concat([_read_odds("", seasons), _read_odds("openers", seasons)], ignore_index=True)
    ev = o.drop_duplicates("event_id")[["event_id", "home", "away", "commence", "season"]]
    m = ev.merge(g[["game_id", "home_team", "away_team", "kick", "season"]],
                 left_on=["home", "away", "season"], right_on=["home_team", "away_team", "season"])
    m = m[(m.commence - m.kick).abs().dt.total_seconds() < 48 * 3600]
    o = o.merge(m[["event_id", "game_id"]], on="event_id")
    o = o.merge(g[["game_id", "kick", "h_p_kick", "a_p_kick"]], on="game_id")
    o = o[o.requested_ts < o.kick - pd.Timedelta(minutes=30)]
    o = o[o.sp_home_point.notna()]
    cf = E.closing_fair()[["game_id", "mu_close_all", "mu_close_sharp", "p_close_all"]]
    allowed = O.load_allowed_books()
    rows, offers = [], []
    for gid, d in o.groupby("game_id"):
        r0 = d.iloc[0]
        pk = [x for x in (r0.h_p_kick, r0.a_p_kick) if pd.notna(x)]
        if not pk:
            continue
        la_cut, early_cut = min(pk), max(pk) + pd.Timedelta(hours=4)
        rec = {"game_id": gid}
        ts = np.sort(d.requested_ts.unique())
        la = [x for x in ts if x < la_cut]
        er = [x for x in ts if x >= early_cut]
        for lab, cand in (("la", la[-1:] if la else []), ("early", er[:1])):
            if not cand:
                continue
            s = d[d.requested_ts == cand[0]]
            # combine the two files at the same timestamp (same snapshot) -> dedupe books
            s = s.drop_duplicates("book")
            rec[f"{lab}_ts"] = cand[0]
            rec[f"{lab}_m"] = float((-s.sp_home_point).median())
            rec[f"{lab}_mu"] = _mu_px(s)
            rec[f"{lab}_nb"] = int(s.book.nunique())
            if lab == "early":
                a = s[s.book.isin(allowed)]
                for side in ("home", "away"):
                    x = a[["book"]].copy()
                    x["game_id"] = gid
                    x["side"] = side
                    x["point"] = a[f"sp_{side}_point"].values
                    x["price"] = a[f"sp_{side}_price"].values
                    offers.append(x)
        rows.append(rec)
    et = pd.DataFrame(rows).merge(cf, on="game_id", how="left")
    of = pd.concat(offers, ignore_index=True) if offers else pd.DataFrame()
    et.to_parquet(c1)
    of.to_parquet(c2)
    return et, of


# ============================================================================ analyses
SPEC_M0 = ["d_ats"]
SPEC_M1 = ["d_ats", "d_close_wl", "d_ot", "d_ats_prime", "d_ats_pub", "d_ats_big"]
SPEC_M2 = ["d_ats", "d_eff_s", "d_int_net", "d_fum_luck", "d_fum_cnt_net", "d_ntd_net", "d_garb_net"]
SPEC_M3 = ["d_eff_s", "d_int_net", "d_fum_luck", "d_fum_cnt_net", "d_ntd_net", "d_garb_net"]
ADJ_FIT = None  # set from 2003-19 in prep()


def prep(panel: pd.DataFrame) -> pd.DataFrame:
    """Game-level REG panel (both teams have a previous game this season) + derived multi-week columns."""
    global ADJ_FIT
    p = panel[(panel.game_type == "REG") & panel.h_p_ats.notna() & panel.a_p_ats.notna()].copy()
    p["cl"] = p.season * 100 + p.week
    p["qb_any"] = p.h_qb_chg | p.a_qb_chg
    for s in "ha":
        p[f"{s}_d1"] = p[f"{s}_rating"] - p[f"{s}_p_rating"]
        p[f"{s}_d2"] = p[f"{s}_p_rating"] - p[f"{s}_p2_rating"]
        p[f"{s}_d3"] = p[f"{s}_p2_rating"] - p[f"{s}_p3_rating"]
        up = (p[f"{s}_d1"] > 0.3) & (p[f"{s}_d2"] > 0.3) & (p[f"{s}_d3"] > 0.3)
        dn = (p[f"{s}_d1"] < -0.3) & (p[f"{s}_d2"] < -0.3) & (p[f"{s}_d3"] < -0.3)
        p[f"{s}_mom"] = np.where(up, 1.0, np.where(dn, -1.0, 0.0))
        p[f"{s}_chg3"] = p[f"{s}_rating"] - p[f"{s}_p3_rating"]
        p[f"{s}_ext"] = np.sign(p[f"{s}_rating"]) * np.clip(p[f"{s}_rating"].abs() - 6, 0, None)
        p[f"{s}_off"] = p[f"{s}_rating"] - p[f"{s}_ly_rating"]
        p[f"{s}_lyluck"] = p[f"{s}_ly_pd"] - p[f"{s}_ly_rating"]
    for c in ("mom", "chg3", "ext", "rating", "std_eff_net", "std_ats", "dead", "off", "lyluck"):
        p[f"d_{c}"] = p[f"h_{c}"].astype(float) - p[f"a_{c}"].astype(float)
    p["d_p_ats2"] = p.h_p_ats2 - p.a_p_ats2
    dev = p[p.season.isin(DEV_C) & ~p.qb_any]
    ADJ_FIT = [float(x) for x in np.polyfit(dev.d_ats, dev.adj, 1)]
    p["adj_fit"] = np.polyval(ADJ_FIT, p.d_ats)
    p["adj_news"] = p.adj - p.adj_fit
    return p


def _brief(r: dict) -> dict:
    return {"n": r["n"], "r2": r["r2"], **{k: v for k, v in r["coef"].items() if k != "const"}}


def test1_closing(p: pd.DataFrame, seasons) -> dict:
    d = p[p.season.isin(seasons) & ~p.qb_any]
    out = {"M0_result_surprise": {k: _brief(v) for k, v in triple(d, SPEC_M0).items()},
           "M1_context": {k: _brief(v) for k, v in triple(d, SPEC_M1).items()}}
    e = d[d.h_p_eff_s.notna() & d.a_p_eff_s.notna()]
    if len(e) > 100:
        out["M2_components_plus_result"] = {k: _brief(v) for k, v in triple(e, SPEC_M2).items()}
        out["M3_components_only"] = {k: _brief(v) for k, v in triple(e, SPEC_M3).items()}
    out["two_games_back"] = {k: _brief(v) for k, v in triple(d.dropna(subset=["d_p_ats2"]), ["d_ats", "d_p_ats2"]).items()}
    return out


def test2_multiweek(p: pd.DataFrame, seasons) -> dict:
    d = p[p.season.isin(seasons)]
    w4 = d[d.week >= 4]
    out = {
        "momentum_3_straight": _brief(ols(w4, "ats_res", ["d_mom"])),
        "momentum_share_of_teams": round(float((w4.h_mom != 0).mean()), 3),
        "rating_change_3g": _brief(ols(w4, "ats_res", ["d_chg3"])),
        "extreme_rating_beyond_6": _brief(ols(d, "ats_res", ["d_ext"])),
        "rating_level": _brief(ols(d, "ats_res", ["d_rating"])),
        "season_to_date_ats_ctrl_rating": _brief(ols(w4, "ats_res", ["d_std_ats", "d_rating"])),
        "late_dead_wk12plus": _brief(ols(d[d.week >= 12], "ats_res", ["d_dead"])),
        "late_dead_games": int((d[d.week >= 12].d_dead != 0).sum()),
    }
    e = w4.dropna(subset=["d_std_eff_net"])
    if len(e) > 100:
        out["efficiency_vs_market_rating"] = _brief(ols(e, "ats_res", ["d_std_eff_net", "d_rating"]))
    out["line_adjustment_overreaction"] = {
        "ats_on_adj": _brief(ols(d, "ats_res", ["adj"])),
        "ats_on_adj_split": _brief(ols(d, "ats_res", ["adj_fit", "adj_news"])),
        "ats_on_adj_noQBchange": _brief(ols(d[~d.qb_any], "ats_res", ["adj"]))}
    return out


def season_start(panel: pd.DataFrame, seasons) -> dict:
    d = panel[(panel.game_type == "REG") & panel.season.isin(seasons) & (panel.week <= 4)].copy()
    d["cl"] = d.season * 100 + d.week
    for s in "ha":
        d[f"{s}_off"] = d[f"{s}_rating"] - d[f"{s}_ly_rating"]
        d[f"{s}_lyluck"] = d[f"{s}_ly_pd"] - d[f"{s}_ly_rating"]
    for c in ("off", "lyluck", "ly_ats", "rating", "ly_pd"):
        d[f"d_{c}"] = d[f"h_{c}"] - d[f"a_{c}"]
    w1 = d[d.week == 1]
    return {"wk1_4_ats_on_offseason_rating_move": _brief(ols(d, "ats_res", ["d_off"])),
            "wk1_4_ats_on_last_year_pd_beyond_rating": _brief(ols(d, "ats_res", ["d_lyluck"])),
            "wk1_4_ats_on_last_year_ats": _brief(ols(d, "ats_res", ["d_ly_ats"])),
            "wk1_4_ats_on_ly_pd_ctrl_rating": _brief(ols(d, "ats_res", ["d_ly_pd", "d_rating"])),
            "wk1_line_vs_prior_on_ly_pd": _brief(ols(w1, "adj", ["d_ly_pd"]))}


def early_frame(p: pd.DataFrame, seasons) -> tuple[pd.DataFrame, pd.DataFrame]:
    et, of = early_table(p, seasons)
    d = p[p.season.isin(seasons)].merge(et, on="game_id")
    ok = d.la_ts.notna() & ((d.early_ts - d.la_ts) < pd.Timedelta(days=4))
    d.loc[~ok, ["la_mu", "la_m"]] = np.nan
    d["reopen"] = d.early_mu - d.la_mu
    d["drift"] = d.mu_close_all - d.early_mu
    d["atsc"] = d.margin - d.mu_close_all
    d["tru_la"] = d.margin - d.la_mu
    d["adj_e"] = d.early_mu - d.prior
    d["la_adj"] = d.la_mu - d.prior
    return d, of


def test3_early(d: pd.DataFrame) -> dict:
    c = d[~d.qb_any]
    out = {"games": int(len(d)), "games_with_lookahead": int(d.la_mu.notna().sum()),
           "early_hours_before_kick_median": round(float(((d.kick - d.early_ts).dt.total_seconds() / 3600).median()), 1),
           "abs_reopen_mean": round(float(d.reopen.abs().mean()), 3),
           "abs_drift_mean": round(float(d.drift.abs().mean()), 3),
           "abs_adj_e_mean": round(float(d.adj_e.abs().mean()), 3)}
    for nm, xs in (("M0", SPEC_M0), ("M2", ["d_ats", "d_ats_prime", "d_eff_s", "d_int_net", "d_fum_luck", "d_ntd_net", "d_garb_net"])):
        out[nm] = {y: _brief(ols(c, y, xs)) for y in ("reopen", "drift", "atsc", "tru_la", "adj_e")}
    out["ats_close_on_adj_split"] = _brief(ols(d, "atsc", ["adj_e", "drift"]))
    out["drift_on_lookahead_vs_rating"] = _brief(ols(d, "drift", ["la_adj"]))
    out["drift_on_lookahead_by_season"] = {int(s): _brief(ols(x, "drift", ["la_adj"])) for s, x in d.groupby("season")
                                           if x.la_adj.notna().sum() > 20}
    out["close_minus_lookahead_on_la_adj"] = _brief(ols(d.assign(y=d.mu_close_all - d.la_mu), "y", ["la_adj"]))
    return out


# ============================================================================ rules
def rule_prime(d: pd.DataFrame, thr: float = 12.0) -> pd.DataFrame:
    """R1: back the team whose LAST game was primetime and beat the closing spread big (fade the reverse).
    S = home prime-ATS-surprise - away prime-ATS-surprise; bet sign(S) when |S| >= thr. No QB change."""
    x = d[~d.qb_any]
    s = x.d_ats_prime
    b = x[s.abs() >= thr].copy()
    b["side"] = np.where(s[s.abs() >= thr] > 0, "home", "away")
    return b


def rule_bounce(d: pd.DataFrame, thr: float = 21.0) -> pd.DataFrame:
    """R2: back a team that failed to cover its last game by >= thr points (opponent did not); no QB change."""
    x = d[~d.qb_any]
    s = pd.Series(np.where(x.h_p_ats <= -thr, 1, 0) - np.where(x.a_p_ats <= -thr, 1, 0), index=x.index)
    b = x[s != 0].copy()
    b["side"] = np.where(s[s != 0] > 0, "home", "away")
    return b


def offers_frame(d: pd.DataFrame, of: pd.DataFrame) -> pd.DataFrame:
    o = of.merge(d[["game_id", "season", "early_mu", "early_m", "la_adj", "mu_close_all", "margin", "qb_any",
                    "d_ats_prime", "h_p_ats", "a_p_ats"]], on="game_id")
    o = o[o.point.notna() & o.price.between(-200, 200)].copy()
    cons_side = np.where(o.side == "home", -o.early_m, o.early_m)
    o = o[(o.point - cons_side).abs() <= 2.5]
    o["dec"] = E._dec(o.price.values)
    o["clv"] = _ev_spread(o.mu_close_all.values, o.side.values, o.point.values, o.dec.values)
    o["ev_now"] = _ev_spread(o.early_mu.values, o.side.values, o.point.values, o.dec.values)
    res = np.where(o.side == "home", o.margin + o.point, -o.margin + o.point)
    o["pnl"] = np.where(res > 0, o.dec - 1, np.where(res < 0, -1.0, 0.0))
    o["res"] = res
    return o


def rule_la_revert(o: pd.DataFrame, coef: dict, thr: float = 0.01) -> pd.DataFrame:
    """R3 (early line, CLV): the look-ahead line's deviation from the rating-implied line partly reverts by
    the close. Predicted close = early consensus mu + a + b * la_adj; take the best allowed-book offer at the
    early snapshot if its EV at the predicted close >= thr."""
    x = o[o.la_adj.notna()].copy()
    x["pred"] = x.early_mu + coef["const"] + coef["la_adj"] * x.la_adj
    x["ev"] = _ev_spread(x.pred.values, x.side.values, x.point.values, x.dec.values)
    x = x[x.ev >= thr]
    return x.sort_values("ev", ascending=False).groupby("game_id").head(1)


def early_version(o: pd.DataFrame, closing_bets: pd.DataFrame) -> pd.DataFrame:
    """Same side as a closing-line rule, taken at the early snapshot at the best allowed-book offer."""
    k = closing_bets[["game_id", "side"]]
    x = o.merge(k, on=["game_id", "side"])
    return x.sort_values("ev_now", ascending=False).groupby("game_id").head(1)


def clv_stats(b: pd.DataFrame) -> dict:
    if len(b) == 0:
        return {"bets": 0}
    v = b.clv.dropna().values
    z = v.mean() / (v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 2 and v.std() > 0 else 0.0
    dec = b.res != 0
    return {"bets": int(len(b)), "per_season": round(len(b) / b.season.nunique(), 1),
            "clv": round(float(v.mean()), 4), "clv_t": round(float(z), 2),
            "clv_p": round(0.5 * math.erfc(z / math.sqrt(2)), 4), "beat_close": round(float((v > 0).mean()), 3),
            "cover": round(float((b.res > 0).sum() / max(dec.sum(), 1)), 4),
            "roi": round(float(b.pnl.mean()), 4), "roi_se": round(float(b.pnl.std(ddof=1) / math.sqrt(len(b))), 4)}


def rule_grid_closing(p: pd.DataFrame, seasons) -> dict:
    """All closing-line rule variants looked at in development (reported, not just the frozen ones)."""
    d = p[p.season.isin(seasons)]
    out = {}
    for thr in (8, 12, 16):
        out[f"prime_thr{thr}"] = ats_record(rule_prime(d, thr))
    for thr in (14, 21):
        out[f"bounce_thr{thr}"] = ats_record(rule_bounce(d, thr))
    for thr in (2, 4, 6):
        x = d[d.adj.abs() >= thr].copy()
        x["side"] = np.where(x.adj < 0, "home", "away")
        out[f"fade_adj_thr{thr}"] = ats_record(x)
    x = d[(d.week >= 12) & (d.d_dead != 0)].copy()
    x["side"] = np.where(x.d_dead < 0, "home", "away")
    out["fade_dead_wk12"] = ats_record(x)
    for thr in (2, 3):
        x = d[d.d_int_net.abs() >= thr].dropna(subset=["d_int_net"]).copy()
        x["side"] = np.where(x.d_int_net > 0, "home", "away")
        out[f"int_net_thr{thr}"] = ats_record(x)
    return out


# ============================================================================ main
def _loso_la(d: pd.DataFrame, o: pd.DataFrame, thr: float) -> dict:
    parts = []
    for s in sorted(d.season.unique()):
        r = ols(d[d.season != s], "drift", ["la_adj"])["coef"]
        parts.append(rule_la_revert(o[o.season == s], {"const": r["const"][0], "la_adj": r["la_adj"][0]}, thr))
    return clv_stats(pd.concat(parts))


def dev() -> dict:
    panel, t, params = build_panel()
    p = prep(panel)
    res = {"kalman": {k: v for k, v in params.items() if k != "grid"}, "kalman_grid_top": sorted(
        params["grid"], key=lambda r: r["mse_wk2plus"])[:5], "adj_fit_2003_19": ADJ_FIT}
    res["sample"] = {"dev_2003_19_games": int(p.season.isin(DEV_C).sum()),
                     "dev_2003_19_noQBchange": int((p.season.isin(DEV_C) & ~p.qb_any).sum()),
                     "dev_2020_22_games": int(p.season.isin(DEV_C2).sum())}
    for lab, ss in (("dev_2003_2019", DEV_C), ("dev_2012_2019", tuple(range(2012, 2020))), ("dev_2020_2022", DEV_C2)):
        res[f"test1_{lab}"] = test1_closing(p, ss)
        res[f"test2_{lab}"] = test2_multiweek(p, ss)
        res[f"season_start_{lab}"] = season_start(panel, ss)
        res[f"rules_closing_{lab}"] = rule_grid_closing(p, ss)
    d, of = early_frame(p, DEV_E)
    res["test3_dev_2020_2022"] = test3_early(d)
    o = offers_frame(d, of)
    res["early_rules_dev_2020_2022"] = {
        "blind_offers_clv": round(float(o.clv.mean()), 4),
        "la_revert_loso_thr0": _loso_la(d, o, 0.0), "la_revert_loso_thr0.01": _loso_la(d, o, 0.01),
        "la_revert_loso_thr0.02": _loso_la(d, o, 0.02),
        "prime12_at_early_best_price": clv_stats(early_version(o, rule_prime(d, 12))),
        "bounce21_at_early_best_price": clv_stats(early_version(o, rule_bounce(d, 21)))}
    out = json.loads(JSON.read_text()) if JSON.exists() else {}
    out["dev"] = res
    JSON.write_text(json.dumps(out, indent=1, default=str))
    return res


def freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; never overwrite a freeze")
    panel, _, _ = build_panel()
    p = prep(panel)
    d, _ = early_frame(p, DEV_E)
    r = ols(d, "drift", ["la_adj"])["coef"]
    spec = {
        "frozen_on": "2026-10-01", "holdout": list(HOLD), "alpha_each": round(0.05 / 3, 4),
        "disclosure": ("During development a per-season table of the coefficient of next-game ATS residual on the "
                       "raw line adjustment (close - rating-implied prior) was printed for ALL seasons incl. "
                       "2023-25 (+0.33, +0.47, +0.12 vs dev about -0.1/-0.26). The 'fade the line adjustment' rule "
                       "is therefore NOT frozen and its 2023-25 numbers are contaminated/descriptive only. No other "
                       "holdout quantity was looked at before this freeze."),
        "rules": {
            "R1_prime_underreaction": {
                "market": "spread, closing line (nflverse spread_line), actual closing juice",
                "rule": "REG game, both teams' starting QB unchanged from their previous game. For each team "
                        "s_T = (margin - closing spread) of its previous game this season if that game was "
                        "primetime (Thu, Mon, or Sun kick >= 19:00 ET) else 0. S = s_home - s_away. "
                        "Bet home if S >= 12, away if S <= -12.",
                "thr": 12.0,
                "pass": "cover rate > 52.38% with one-sided binomial p < 0.0167 vs 52.38%"},
            "R2_bounce_after_ats_blowout": {
                "market": "spread, closing line, actual closing juice",
                "rule": "REG game, no QB change either team. Back the team whose previous game (this season) "
                        "missed the closing spread by >= 21 points, unless both did.",
                "thr": 21.0,
                "pass": "cover rate > 52.38% with one-sided binomial p < 0.0167"},
            "R3_lookahead_reversion_early_clv": {
                "market": "spread at the early snapshot (first snapshot after both teams' previous games ended "
                          "+4h), best allowed-book offer (my_books.json), point within 2.5 of consensus, "
                          "price in [-200, 200]",
                "rule": "la_adj = look-ahead consensus price-implied mu (last snapshot before either team's "
                        "previous kickoff, within 4 days of the early snapshot) - Kalman rating-implied prior. "
                        "pred_close = early_mu + const + b*la_adj. Bet the allowed offer with max EV at pred_close "
                        "if EV >= 0.01 (one per game).",
                "coef": {"const": r["const"][0], "la_adj": r["la_adj"][0]},
                "coef_se": {"const": r["const"][1], "la_adj": r["la_adj"][1]},
                "thr": 0.01,
                "pass": "mean price-based CLV (vs closing_fair mu_close_all) > 0 with one-sided p < 0.0167"},
        }}
    FROZEN.write_text(json.dumps(spec, indent=1))
    print(json.dumps(spec, indent=1))


def holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("holdout is locked: set EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES after freezing")
    spec = json.loads(FROZEN.read_text())
    out = json.loads(JSON.read_text())
    if "holdout" in out:
        raise SystemExit("holdout already run once; see market_ratings.json")
    panel, _, _ = build_panel()
    p = prep(panel)
    R = spec["rules"]
    h = p[p.season.isin(HOLD)]
    d, of = early_frame(p, HOLD)
    o = offers_frame(d, of)
    res = {"rules": {}}
    for rid, b in (("R1_prime_underreaction", rule_prime(h, R["R1_prime_underreaction"]["thr"])),
                   ("R2_bounce_after_ats_blowout", rule_bounce(h, R["R2_bounce_after_ats_blowout"]["thr"]))):
        s = ats_record(b)
        s["pass"] = bool(s.get("win_rate") and s["win_rate"] > BREAKEVEN and s["p_vs_52.4"] < spec["alpha_each"])
        s["by_season"] = {int(y): ats_record(g) for y, g in b.groupby("season")}
        s["same_side_at_early_best_price"] = clv_stats(early_version(o, b[b.season.isin(HOLD)]))
        res["rules"][rid] = s
    b = rule_la_revert(o, R["R3_lookahead_reversion_early_clv"]["coef"], R["R3_lookahead_reversion_early_clv"]["thr"])
    s = clv_stats(b)
    s["pass"] = bool(s.get("bets", 0) > 2 and s["clv"] > 0 and s["clv_p"] < spec["alpha_each"])
    s["by_season"] = {int(y): clv_stats(g) for y, g in b.groupby("season")}
    res["rules"]["R3_lookahead_reversion_early_clv"] = s
    # descriptive (same regressions as dev, run once)
    res["test1_holdout"] = test1_closing(p, HOLD)
    res["test2_holdout"] = test2_multiweek(p, HOLD)
    res["season_start_holdout"] = season_start(panel, HOLD)
    res["test3_holdout"] = test3_early(d)
    res["rules_closing_grid_holdout_descriptive"] = rule_grid_closing(p, HOLD)
    out["holdout"] = res
    out["frozen"] = spec
    JSON.write_text(json.dumps(out, indent=1, default=str))
    return res


def _c(x, k):
    v = x.get(k)
    return "n/a" if v is None else f"{v[0]:+.3f} ± {v[1]:.3f}"


def _rec(r):
    if not r or not r.get("bets"):
        return "0 bets"
    return f"{r['wins']}-{r['losses']}-{r['pushes']} ({100 * r['win_rate']:.1f}%), ROI {100 * r['roi']:+.1f}% ± {100 * r['roi_se']:.1f}"


def _clv(r):
    if not r or not r.get("bets"):
        return "0 bets"
    return (f"{r['bets']} bets, CLV {100 * r['clv']:+.2f}% (t {r['clv_t']:+.2f}, p {r['clv_p']:.3f}), "
            f"cover {100 * r['cover']:.1f}%, ROI {100 * r['roi']:+.1f}% ± {100 * r['roi_se']:.1f}")


def write_report():
    R = json.loads(JSON.read_text())
    d, h, F = R["dev"], R.get("holdout", {}), R.get("frozen", json.loads(FROZEN.read_text()))
    k = d["kalman"]
    L = ["# Market-implied power ratings: what moves them, where the market over/under-reacts", "",
         "Script: `scripts/research/market_ratings.py` (dev / freeze / holdout / report). Raw numbers: "
         "`output/research/market_ratings.json`. Frozen rules: `output/research/market_ratings_frozen.json`.", ""]
    L += ["## Verdict", "",
          "- **The market's week-to-week updating is close to efficient.** Lines move about 0.05-0.06 points per "
          "point of last-game result surprise (margin minus closing spread). The weight that would have been "
          "correct is 0.03-0.05. The next-game ATS coefficient is -0.007 to -0.020 in every block, and never "
          "significant.",
          "- **The classic hypothesis is not supported.** The market does not underweight efficiency. In the "
          "components-only regression, it slightly OVER-weights one-game EPA surprise. It already discounts return/defensive "
          "TDs and garbage-time points, and if anything it discounts garbage time too much. Turnover luck "
          "over-reaction is too small to bet.",
          "- **All 3 frozen rules failed the 2023-25 holdout** (Bonferroni α = 0.0167 each). R1 (primetime "
          "under-reaction) and R2 (bounce after an ATS blowout) were positive in every block but not significant. "
          "R3 (look-ahead reversion, early-line CLV) was flat.",
          "- **One structural fact replicated out of sample.** The part of the look-ahead line that deviates from "
          "the rating-implied line partly reverts by the close (slope about -0.10 to -0.13, t about 4 in both dev "
          "and holdout). It is too small to beat the price after vig.", ""]
    L += ["## Method", "",
          f"- **Ratings.** A Kalman random-walk filter on 32 team ratings plus HFA, observing each week's closing "
          f"lines (`spread_line`, 1999-2026). Tuned on one-step-ahead LINE prediction MSE, 2003-19 only: "
          f"s_obs {k['s_obs']}, q_week {k['q_w']}, season carry-over rho {k['rho']}, q_off {k['q_off']}.",
          "- **Prior.** For every game, `prior` is the line implied by ratings before either team's last result. "
          "From it: `adj = close - prior`, `ats = margin - close` and `true = margin - prior`, with "
          "**true = adj + ats**. Regressing all three on the same last-game components (home minus away) gives "
          "the market's weight, its error and the correct weight. Coefficients add exactly.",
          "- **Components of each team's previous game.** All are team perspective. In the JSON, the M2/M3 "
          "regressions under `test1_dev_2003_2019` use 2012-19 only, because pbp starts in 2012:",
          "  - ATS surprise.",
          "  - Efficiency-EPA surprise: net pass/rush EPA, turnover plays removed, residualised on the line (2012+ pbp).",
          "  - INT net and fumble-recovery luck (lost minus 0.5 × fumbles).",
          "  - Non-offensive TDs net and garbage-time points net (Q4, home WP < 0.1 or > 0.9).",
          "  - Close-game W/L, OT, ATS × primetime, ATS × public team, and the excess ATS beyond 14.",
          "- **Sample.** Core sample excludes games where either starting QB changed from the team's last game, "
          "which is the 'injuries known' control. SEs are clustered by season-week.",
          "- **Early lines (2020-25).** The look-ahead line is the last snapshot before either team's previous "
          "kickoff, from the main and openers files, within 4 days of the re-open. The early (re-open) line is "
          "the first snapshot after both previous games ended (+4h), median 158 h before kickoff. Both use the "
          "price-implied consensus mu. The close is `closing_fair` mu_close_all. CLV is valued at the best "
          "allowed-book offer.", ""]

    def tri(tag, blk, spec):
        x = blk.get(spec)
        if not x:
            return []
        rows = []
        for var in [c for c in x["adj"] if c not in ("n", "r2")]:
            rows.append(f"| {tag} | {var} | {_c(x['adj'], var)} | {_c(x['ats'], var)} | {_c(x['true'], var)} |")
        return rows
    L += ["## Test 1: what moves market ratings, and is the move the right size?", "",
          "Coefficients are points of line per unit of (home minus away) last-game component. "
          "'adj' is the market's move, 'ats' is the leftover error (negative means over-reaction) and 'true' is "
          "the correct weight. Format is coefficient ± clustered SE.", "",
          "| block (n) | component | adj (market) | ats (error) | true |", "|---|---|---|---|---|"]
    blocks = (("dev 2003-19", d["test1_dev_2003_2019"]), ("dev 2020-22", d["test1_dev_2020_2022"]),
              ("HOLDOUT 2023-25", h.get("test1_holdout", {})))
    for tag, b in blocks:
        if b:
            L += tri(f"{tag} ({b['M0_result_surprise']['adj']['n']})", b, "M0_result_surprise")
    for tag, b in blocks:
        if b:
            L += [r for r in tri(tag, b, "M1_context") if "d_ats_prime" in r or "d_ats_big" in r or "close_wl" in r]
    L += ["", "**Components only** (2012-19 dev uses pbp; same spec in 2020-22 and in the holdout):", "",
          "| block | component | adj (market) | ats (error) | true |", "|---|---|---|---|---|"]
    for tag, b in (("dev 2012-19", d["test1_dev_2012_2019"]), ("dev 2020-22", d["test1_dev_2020_2022"]),
                   ("HOLDOUT", h.get("test1_holdout", {}))):
        if b:
            L += tri(tag, b, "M3_components_only")
    L += ["", "**How to read it:**",
          "- **Efficiency.** Market weight on one-game efficiency surprise is 0.06 / 0.04 / 0.04 against a correct "
          "weight of 0.04 / 0.03 / 0.00. That is a mild over-reaction to single-game EPA, the opposite of the "
          "classic claim.",
          "- **Interceptions.** INT net is moved 0.19 per INT in every era, while 0.29-0.40 would have been "
          "correct. That is a consistent under-reaction, but no block is significant.",
          "- **Garbage time.** The market discounts garbage-time points (it reacts about -0.02 at the re-open), "
          "yet garbage-time points carry positive signal: ATS +0.03 / +0.06 / +0.08. It is a lead, not significant.",
          "- **Non-offensive TDs.** These are already removed from the move: adj is -0.19 per return TD when "
          "holding the score fixed.", ""]
    L += ["## Test 2: multi-week dynamics (ATS residual vs close)", "", "| test | dev 2003-19 | dev 2020-22 | holdout |",
          "|---|---|---|---|"]
    t2 = (d["test2_dev_2003_2019"], d["test2_dev_2020_2022"], h.get("test2_holdout", {}))
    for lab, key, var in (("3 straight rating rises (±1)", "momentum_3_straight", "d_mom"),
                          ("rating change over 3 games", "rating_change_3g", "d_chg3"),
                          ("extreme rating (beyond ±6)", "extreme_rating_beyond_6", "d_ext"),
                          ("rating level", "rating_level", "d_rating"),
                          ("season-to-date ATS (ctrl rating)", "season_to_date_ats_ctrl_rating", "d_std_ats"),
                          ("season-to-date efficiency (ctrl rating)", "efficiency_vs_market_rating", "d_std_eff_net"),
                          ("eliminated-proxy team, wk 12+", "late_dead_wk12plus", "d_dead")):
        L.append(f"| {lab} | " + " | ".join(_c(x.get(key, {}), var) if x.get(key) else "n/a" for x in t2) + " |")
    L.append("| line adjustment (close - prior) | " + " | ".join(
        _c(x["line_adjustment_overreaction"]["ats_on_adj"], "adj") if x else "n/a" for x in t2) + " (holdout CONTAMINATED, see disclosure) |")
    ss = (d["season_start_dev_2003_2019"], d["season_start_dev_2020_2022"], h.get("season_start_holdout", {}))
    L.append("| wk 1-4: offseason rating move | " + " | ".join(
        _c(x["wk1_4_ats_on_offseason_rating_move"], "d_off") if x else "n/a" for x in ss) + " |")
    L.append("| wk 1-4: last-yr point diff beyond rating | " + " | ".join(
        _c(x["wk1_4_ats_on_last_year_pd_beyond_rating"], "d_lyluck") if x else "n/a" for x in ss) + " |")
    L += ["", "**Findings:**",
          "- Momentum, regression of extreme ratings and season-start priors show nothing stable. No preseason "
          "win totals are in the repo, so offseason rating moves and last season's point differential stand in "
          "for them.",
          "- The eliminated-team proxy (cannot finish at least .500) is negative in every block, but its "
          "cover-rate rule loses in dev. Means and cover rates disagree.",
          "- Fading the week's line adjustment looked good in dev (-0.10, -0.26) and reversed in 2023-25 "
          "(+0.30). It was not frozen; see the disclosure.", ""]
    t3d, t3h = d["test3_dev_2020_2022"], h.get("test3_holdout", {})
    L += ["## Test 3: look-ahead → re-open → close (2020-25)", "",
          "| quantity | dev 2020-22 | holdout 2023-25 |", "|---|---|---|",
          f"| games / with look-ahead line | {t3d['games']} / {t3d['games_with_lookahead']} | {t3h.get('games')} / {t3h.get('games_with_lookahead')} |",
          f"| mean abs re-open (early - look-ahead), pts | {t3d['abs_reopen_mean']} | {t3h.get('abs_reopen_mean')} |",
          f"| mean abs drift (close - early), pts | {t3d['abs_drift_mean']} | {t3h.get('abs_drift_mean')} |"]
    for y, lab in (("reopen", "re-open per pt of result surprise"), ("drift", "drift per pt of result surprise"),
                   ("tru_la", "correct weight (margin - look-ahead)"), ("atsc", "ATS vs price close")):
        L.append(f"| {lab} | {_c(t3d['M0'][y], 'd_ats')} (R² {t3d['M0'][y]['r2']}) | "
                 f"{_c(t3h['M0'][y], 'd_ats') if t3h else 'n/a'} |")
    L.append(f"| drift on look-ahead deviation from rating line | {_c(t3d['drift_on_lookahead_vs_rating'], 'la_adj')} | "
             f"{_c(t3h['drift_on_lookahead_vs_rating'], 'la_adj') if t3h else 'n/a'} |")
    if t3h:
        L.append(f"| drift on efficiency surprise (M2) | {_c(t3d['M2']['drift'], 'd_eff_s')} | {_c(t3h['M2']['drift'], 'd_eff_s')} |")
        L.append(f"| re-open on garbage-time pts (M2) | {_c(t3d['M2']['reopen'], 'd_garb_net')} | {_c(t3h['M2']['reopen'], 'd_garb_net')} |")
    L += ["", "**Findings:**",
          "- The adjustment to a result happens almost entirely at the re-open: about 0.049 per point, with R² "
          "about 0.4 from the result alone. It is the right size: the correct weight is 0.040-0.045. Little "
          "result-driven drift remains afterwards.",
          "- What does drift is the look-ahead line's own deviation from the rating-implied line, which partly "
          "reverts. This replicated in the holdout (all three seasons negative).",
          "- **New in the holdout only.** After the re-open the line keeps moving toward efficiency (EPA) "
          "surprise: drift +0.019 per EPA point, t about 3.7, against about 0 in dev. It was not "
          "pre-registered, so it is a lead for paper tracking only.", ""]
    L += ["## Frozen rules and the one-time holdout", "",
          f"Frozen before the holdout ({F['frozen_on']}). α = {F['alpha_each']} each.", "",
          "| rule | dev 2003-19 | dev 2020-22 | HOLDOUT 2023-25 | pass |", "|---|---|---|---|---|"]
    rc = (d["rules_closing_dev_2003_2019"], d["rules_closing_dev_2020_2022"])
    hr = h.get("rules", {})
    if hr:
        L.append(f"| R1 primetime under-reaction (S ≥ 12), closing ATS | {_rec(rc[0]['prime_thr12'])} | {_rec(rc[1]['prime_thr12'])} | "
                 f"{_rec(hr['R1_prime_underreaction'])}, p {hr['R1_prime_underreaction']['p_vs_52.4']} | {hr['R1_prime_underreaction']['pass']} |")
        L.append(f"| R2 bounce after ATS miss ≥ 21, closing ATS | {_rec(rc[0]['bounce_thr21'])} | {_rec(rc[1]['bounce_thr21'])} | "
                 f"{_rec(hr['R2_bounce_after_ats_blowout'])}, p {hr['R2_bounce_after_ats_blowout']['p_vs_52.4']} | {hr['R2_bounce_after_ats_blowout']['pass']} |")
        L.append(f"| R3 look-ahead reversion, early best price, CLV | n/a | LOSO: {_clv(d['early_rules_dev_2020_2022']['la_revert_loso_thr0.01'])} | "
                 f"{_clv(hr['R3_lookahead_reversion_early_clv'])} | {hr['R3_lookahead_reversion_early_clv']['pass']} |")
        L += ["", "**Same side at the early line** (best allowed price):",
              f"- R1 holdout: {_clv(hr['R1_prime_underreaction']['same_side_at_early_best_price'])}.",
              f"- R2 holdout: {_clv(hr['R2_bounce_after_ats_blowout']['same_side_at_early_best_price'])}.",
              "- Both signals pay at the close, not by moving the line. Betting them early costs about 2-3% CLV "
              "against a -4.5% blind baseline.", "",
              "**Pooled across dev and holdout** (selection was made on dev, so this is optimistic):",
              "- R1: 331-273, 54.8%, z = 1.19 against 52.4%.",
              "- R2: 269-209, 56.3%, z = 1.71.", ""]
    L += ["**Disclosure.** " + F["disclosure"], "",
          "## Paper-track recommendation",
          "",
          "Nothing qualifies as an edge, and nothing should get real money. R2, and less so R1, can go on a "
          "zero-stake paper track at the close with actual juice, exactly as frozen. Each produces 20-30 bets a "
          "season. That is monitoring, not a test that will resolve soon. Separating a true 55% from 52.4% at "
          "2 SE needs about 1,400 bets, which is decades at this volume.",
          "",
          "- **R1.** In a regular-season game where neither team's starting QB changed from its previous game: "
          "let s = (previous-game margin minus its closing spread) if that game was primetime (Thu, Mon, or Sun "
          "at 7pm ET or later), else 0. Bet home ATS if s_home - s_away ≥ 12. Bet away if it is ≤ -12.",
          "- **R2.** Same QB filter. Back the team that missed its previous closing spread by 21 or more, unless "
          "both teams did.",
          "",
          "Post-hoc leads to re-test on 2026+ only:",
          "- Garbage-time points are over-discounted.",
          "- In the 2023+ market, drift continues toward efficiency surprise.",
          "- INT margin is under-weighted.", ""]
    MD.write_text("\n".join(L))
    print(MD.read_text())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "dev":
        r = dev()
        print(json.dumps(r, indent=1, default=str)[:20000])
    elif cmd == "freeze":
        freeze()
    elif cmd == "holdout":
        print(json.dumps(holdout(), indent=1, default=str)[:20000])
    elif cmd == "report":
        write_report()
    else:
        raise SystemExit(f"unknown command {cmd}")
