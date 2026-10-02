"""Season Monte Carlo simulator + historical futures test (win totals) + QB-injury futures analysis.

    cd /home/claude/nfl && PYTHONPATH=src:scripts python scripts/research/season_sim.py [--fetch]

Outputs: output/research/season_sim.md / season_sim.json / season_sim_2026.json,
         data/research_futures/win_totals_2013_2026.csv (written by --fetch, or if missing)

SIMULATOR (simulate()):
  * From any date D in a season, the remaining REG games are simulated N times; completed games keep
    their real result. Each remaining game: home margin = mu + (d_home(h) - d_away(h)) + eps,
    eps ~ N(0, sig_eps), d_t(h) = team rating error that is a random walk in weeks-ahead h
    (Var = tau_a^2 + tau_b^2 * h). tau_a, tau_b, sig_eps are estimated on 2015-2019 from the
    covariance of same-team residuals (see fit_uncertainty()). Ties are not simulated (continuous margin).
  * mu, two versions:
      market: latest closing spread where the line is already posted (the upcoming week only);
              otherwise ratings + HFA fitted (weighted ridge, recency half-life) to all spreads posted
              before D plus the upcoming week's lines, shrunk toward 0 by k(h) (fit on 2015-19).
      model : production MarginModel trained walk-forward (seasons < S). Each team's pre-game
              team-level inputs (Elo, QB rating, EPA/SR EWMAs, ...) are taken from its next game on/after D,
              then synthetic feature rows are built for every remaining game (schedule context = home field,
              division, rest, final week; injuries/OT set to 0). Team part shrunk by k(h) fit on 2015-19.
  * Standings: division winner, playoff seeds (7 per conference from 2020, 6 before), #1 seed.
    Tiebreakers (approximation): win pct, then head-to-head win pct among the tied teams (only if every
    tied pair played each other), then division record (only when all tied teams share a division), then
    conference record, then coin flip. Checked on completed 2015-2025 seasons: reproduces the real playoff
    fields except true coin-flip-level cases decided by common games / strength of victory. Wild cards:
    NFL step 1 is applied (only the best remaining team of each division is eligible at each pick).
    Not modelled: common games, strength of victory/schedule, points tiebreakers, restart of a
    multi-team tiebreak after one team is eliminated.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nflpred import model as M
from nflpred.features import FEATURES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "research"
FUT = ROOT / "data" / "research_futures"
WT_CSV = FUT / "win_totals_2013_2026.csv"
SCRATCH = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/seasonsim")
DEV, HOLD = list(range(2015, 2020)), list(range(2020, 2026))
N_SIMS = 10000
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}

DIVS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LV", "LAC"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SF", "SEA"],
}
TEAMS = sorted(t for v in DIVS.values() for t in v)
TIX = {t: i for i, t in enumerate(TEAMS)}
DIV_OF = np.array([[k for k, v in DIVS.items() if t in v][0] for t in TEAMS])
CONF_OF = np.array([d[:3] for d in DIV_OF])
TEAM_VALS = ["elo", "qb_rating", "qb_change", "off_epa", "def_epa", "off_sr", "def_sr",
             "pass_epa", "rush_epa", "to_margin", "pt_diff"]
TEAM_FEATS = ["elo_diff", "qb_diff", "off_epa_diff", "def_epa_diff", "off_sr_diff", "def_sr_diff",
              "pass_epa_diff", "rush_epa_diff", "to_margin_diff", "pt_diff_diff", "qb_change_diff",
              "fw_elo_diff", "fw_pt_diff_diff", "fw_qb_diff"]


# ============================================================== data
def fetch_win_totals():
    """Preseason win totals + over/under juice, 2003-2026, from nfelo's open WT-ratings file
    (github.com/greerreNFL/nfelosrs, wt_ratings.csv). Pre-2025 rows are dated 'not_tracked'
    (Aug 30 placeholder = late-preseason consensus); 2025-26 are DraftKings."""
    src = SCRATCH / "nfelosrs" / "wt_ratings.csv"
    if not src.exists():
        SCRATCH.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "-q", "https://github.com/greerreNFL/nfelosrs",
                        str(SCRATCH / "nfelosrs")], check=True)
    d = pd.read_csv(src).drop(columns=["Unnamed: 0"], errors="ignore")
    d["team"] = d["team"].replace(TEAM_FIX)
    d = d[["season", "team", "line", "over_odds", "under_odds", "hold", "over_probability",
           "under_probability", "line_adj", "source", "source_date"]]
    FUT.mkdir(parents=True, exist_ok=True)
    d.sort_values(["season", "team"]).to_csv(WT_CSV, index=False)
    return d


def load_features() -> pd.DataFrame:
    cache = SCRATCH / "feat.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from nflpred.pipeline import build
        df = build(refresh=False, today=dt.date.today(), live_news=False)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    df = df[df.game_type == "REG"].copy()
    for c in ("home_team", "away_team"):
        df[c] = df[c].replace(TEAM_FIX)
    df["gameday"] = pd.to_datetime(df["gameday"])
    df["margin"] = df["home_score"] - df["away_score"]
    df["hi"] = df.home_team.map(TIX)
    df["ai"] = df.away_team.map(TIX)
    last_wk = df.groupby("season")["week"].transform("max")
    df["final_week"] = (df.week == last_wk).astype(int)
    return df.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def actual_wins(df, season):
    g = df[(df.season == season) & df.completed]
    w = np.zeros(32)
    np.add.at(w, g.hi.values, (g.margin > 0).values.astype(float))
    np.add.at(w, g.ai.values, (g.margin < 0).values.astype(float))
    return w


# ============================================================== model version
_MODELS: dict = {}


def walk_forward_model(df_all, season):
    if season not in _MODELS:
        _MODELS[season] = M.fit(df_all, before_season=season)
    return _MODELS[season]


def team_state(df, season, asof):
    """Team-level pre-game inputs from each team's next REG game on/after `asof` (leak-free: that row's
    features only use games before its own kickoff, and the team plays no game between asof and it)."""
    g = df[(df.season == season) & (df.gameday >= asof)]
    st = {}
    for side in ("home", "away"):
        for r in g.itertuples(index=False):
            t = getattr(r, f"{side}_team")
            if t in st and st[t]["_day"] <= r.gameday:
                continue
            st[t] = {"_day": r.gameday, **{v: getattr(r, f"{side}_{v}") for v in TEAM_VALS}}
    s = pd.DataFrame(st).T.drop(columns="_day").astype(float)
    return s.reindex(TEAMS)


def synth_rows(games, st):
    h, a = st.loc[games.home_team].reset_index(drop=True), st.loc[games.away_team].reset_index(drop=True)
    X = pd.DataFrame(index=range(len(games)))
    X["elo_diff"] = h.elo - a.elo
    X["qb_diff"] = h.qb_rating - a.qb_rating
    for s in ("off_epa", "off_sr", "pass_epa", "rush_epa", "to_margin", "pt_diff"):
        X[f"{s}_diff"] = h[s] - a[s]
    X["def_epa_diff"] = a.def_epa - h.def_epa
    X["def_sr_diff"] = a.def_sr - h.def_sr
    X["qb_change_diff"] = h.qb_change - a.qb_change
    fw = games.final_week.values
    X["fw_elo_diff"] = fw * X.elo_diff
    X["fw_pt_diff_diff"] = fw * X.pt_diff_diff
    X["fw_qb_diff"] = fw * X.qb_diff
    X["rest_diff"] = (games.home_rest.values - games.away_rest.values).clip(-7, 7)
    X["home_field"] = (games.location.values != "Neutral").astype(int)
    X["div_game"] = games.div_game.fillna(0).astype(int).values
    for c in ("inj_off_diff", "inj_def_diff", "ot_diff"):
        X[c] = 0.0
    return X[FEATURES].fillna(0.0)


def model_mu_parts(mdl, games, st):
    X = synth_rows(games, st)
    full = mdl.predict_margin(X)
    X0 = X.copy()
    X0[TEAM_FEATS] = 0.0
    ctx = mdl.predict_margin(X0)
    return ctx, full - ctx


# ============================================================== market version
MKT_HALFLIFE = 6.0   # weeks (chosen on 2015-19 from {3, 6, 10}, see fit_market_halflife)
MKT_OFFSEASON = 10.0
MKT_RIDGE = 1.0


def market_ratings(df, asof, upto_next_week=True, halflife=None, exclude_games=()):
    """Ratings + HFA fitted to closing spreads posted before `asof` (+ the upcoming week's lines)."""
    hl = halflife or MKT_HALFLIFE
    m = df.spread_line.notna() & (df.gameday < asof) & (df.gameday >= asof - pd.Timedelta(days=500))
    if upto_next_week:
        m |= df.spread_line.notna() & (df.gameday >= asof) & (df.gameday < asof + pd.Timedelta(days=7))
    if len(exclude_games):
        m &= ~df.game_id.isin(exclude_games)
    d = df[m]
    cur_season = df.loc[df.gameday >= asof, "season"].min() if (df.gameday >= asof).any() else d.season.max()
    age = (asof - d.gameday).dt.days.clip(lower=0) / 7.0 + (cur_season - d.season) * MKT_OFFSEASON
    w = 0.5 ** (age.values / hl)
    k, n = len(d), 32
    X = np.zeros((k, n + 1))
    X[np.arange(k), d.hi.values] = 1
    X[np.arange(k), d.ai.values] = -1
    X[:, n] = (d.location.values != "Neutral").astype(float)
    A = X.T @ (X * w[:, None]) + MKT_RIDGE * np.diag(np.r_[np.ones(n), 0.0])
    sol = np.linalg.solve(A + 1e-9 * np.eye(n + 1), X.T @ (w * d.spread_line.values))
    return sol[:n], sol[n]


def market_mu_parts(df, games, asof, r=None, hfa=None):
    if r is None:
        r, hfa = market_ratings(df, asof)
    ctx = hfa * (games.location.values != "Neutral")
    team = r[games.hi.values] - r[games.ai.values]
    return ctx, team


# ============================================================== shrink + uncertainty (fit on dev)
START_WEEKS = [1, 4, 7, 10, 13]


def horizon_frame(df, seasons, version, starts=START_WEEKS):
    """For each season and start week: predicted parts for all games from that week on, with horizon."""
    rows = []
    for s in seasons:
        sd = df[df.season == s]
        for w0 in starts:
            asof = sd[sd.week == w0].gameday.min() - pd.Timedelta(days=1)
            g = sd[sd.week >= w0]
            if version == "model":
                ctx, team = model_mu_parts(walk_forward_model(DF_ALL, s), g, team_state(df, s, asof))
            else:
                ctx, team = market_mu_parts(df, g, asof)
            rows.append(pd.DataFrame({"season": s, "w0": w0, "h": g.week.values - w0, "hi": g.hi.values,
                                      "ai": g.ai.values, "ctx": ctx, "team": team, "margin": g.margin.values,
                                      "spread": g.spread_line.values, "game_id": g.game_id.values}))
    return pd.concat(rows, ignore_index=True)


def fit_shrink(hf):
    """k(h) = k0 + k1*h, separately for preseason starts (w0==1) and in-season starts; least squares of
    (margin - ctx) on team part."""
    out = {}
    for lab, sub in (("pre", hf[hf.w0 == 1]), ("in", hf[hf.w0 > 1])):
        sub = sub.dropna(subset=["margin"])
        y = sub.margin - sub.ctx
        X = np.column_stack([sub.team, sub.team * sub.h])
        k0, k1 = np.linalg.lstsq(X, y, rcond=None)[0]
        out[lab] = (float(k0), float(k1))
    return out


def apply_shrink(team, h, w0, shrink):
    k0, k1 = shrink["pre" if w0 == 1 else "in"]
    return team * np.clip(k0 + k1 * h, 0.05, 1.5)


def fit_uncertainty(hf, shrink):
    """Residual r = margin - mu = d_home(h) - d_away(h) + eps. For two games of the same team (sign-adjusted
    to that team's perspective, different opponents), Cov = Var(d_t at min(h1,h2)) = tau_a^2 + tau_b^2*min(h).
    Estimate by mean products binned by min horizon, then linear fit; sig_eps^2 = Var(r) - 2*mean tau^2."""
    hf = hf.dropna(subset=["margin"]).copy()
    hf["mu"] = hf.ctx + np.where(hf.w0 == 1, hf.team * np.clip(shrink["pre"][0] + shrink["pre"][1] * hf.h, .05, 1.5),
                          hf.team * np.clip(shrink["in"][0] + shrink["in"][1] * hf.h, .05, 1.5))
    hf["r"] = hf.margin - hf.mu
    res = {}
    for lab, sub in (("pre", hf[hf.w0 == 1]), ("in", hf[hf.w0 > 1])):
        long = pd.concat([sub.assign(team_i=sub.hi, opp=sub.ai, rr=sub.r),
                          sub.assign(team_i=sub.ai, opp=sub.hi, rr=-sub.r)])
        acc = {}
        for (s, w0, t), grp in long.groupby(["season", "w0", "team_i"]):
            hh, rr, oo = grp.h.values, grp.rr.values, grp.opp.values
            i, j = np.triu_indices(len(rr), 1)
            ok = oo[i] != oo[j]
            mh = np.minimum(hh[i], hh[j])[ok]
            pr = (rr[i] * rr[j])[ok]
            for m_, p_ in zip(mh, pr):
                a = acc.setdefault(int(m_), [0.0, 0])
                a[0] += p_
                a[1] += 1
        hs = np.array(sorted(acc))
        cov = np.array([acc[x][0] / acc[x][1] for x in hs])
        nn = np.array([acc[x][1] for x in hs])
        keep = nn > 200
        A = np.column_stack([np.ones(keep.sum()), hs[keep]])
        ta2, tb2 = np.linalg.lstsq(A * np.sqrt(nn[keep])[:, None], cov[keep] * np.sqrt(nn[keep]), rcond=None)[0]
        ta2, tb2 = max(ta2, 0.0), max(tb2, 0.0)
        var_r = float(np.mean(sub.r ** 2))
        mean_tau2 = float(np.mean(ta2 + tb2 * sub.h))
        sig = float(np.sqrt(max(var_r - 2 * mean_tau2, 100.0)))
        res[lab] = {"tau_a": float(np.sqrt(ta2)), "tau_b": float(np.sqrt(tb2)), "sig_eps": sig,
                    "resid_sd": float(np.sqrt(var_r)),
                    "cov_by_h": {int(x): round(float(c), 2) for x, c in zip(hs[keep], cov[keep])}}
    return res


# ============================================================== simulator
def simulate(season_games, mu, h, unc, n=N_SIMS, seed=0, standings=True, n_playoff=None):
    """season_games: all REG games of the season (completed ones keep results). mu/h: arrays for the
    rows with completed == False (same order). Returns dict of per-team arrays."""
    rng = np.random.default_rng(seed)
    G = season_games.reset_index(drop=True)
    done = G.completed.values
    rem = np.where(~done)[0]
    H = int(h.max()) + 1 if len(rem) else 1
    ta, tb, se = unc["tau_a"], unc["tau_b"], unc["sig_eps"]
    d = rng.normal(0, ta, (n, 32, 1)) + np.concatenate(
        [np.zeros((n, 32, 1)), np.cumsum(rng.normal(0, tb, (n, 32, H - 1)), axis=2)], axis=2) if H > 1 else \
        rng.normal(0, ta, (n, 32, 1))
    hi, ai = G.hi.values, G.ai.values
    res = np.zeros((n, len(G)), dtype=np.float32)   # home result: 1 win, 0 loss, .5 tie
    res[:, done] = np.where(G.margin.values[done] > 0, 1.0, np.where(G.margin.values[done] < 0, 0.0, 0.5))
    if len(rem):
        hh = h.astype(int)
        marg = mu[None, :] + d[:, hi[rem], hh] - d[:, ai[rem], hh] + rng.normal(0, se, (n, len(rem)))
        res[:, rem] = (marg > 0).astype(np.float32)
    Hm = np.zeros((len(G), 32), np.float32)
    Am = np.zeros((len(G), 32), np.float32)
    Hm[np.arange(len(G)), hi] = 1
    Am[np.arange(len(G)), ai] = 1
    wins = res @ Hm + (1 - res) @ Am
    out = {"wins": wins}
    if not standings:
        return out
    divg = (DIV_OF[hi] == DIV_OF[ai]).astype(np.float32)
    confg = (CONF_OF[hi] == CONF_OF[ai]).astype(np.float32)
    ngames_div = (Hm * divg[:, None]).sum(0) + (Am * divg[:, None]).sum(0)
    ngames_conf = (Hm * confg[:, None]).sum(0) + (Am * confg[:, None]).sum(0)
    div_pct = ((res * divg) @ Hm + ((1 - res) * divg) @ Am) / np.maximum(ngames_div, 1)
    conf_pct = ((res * confg) @ Hm + ((1 - res) * confg) @ Am) / np.maximum(ngames_conf, 1)
    h2h = np.zeros((n, 32, 32), np.float32)
    h2n = np.zeros((32, 32), np.float32)
    for gi in range(len(G)):
        h2h[:, hi[gi], ai[gi]] += res[:, gi]
        h2h[:, ai[gi], hi[gi]] += 1 - res[:, gi]
        h2n[hi[gi], ai[gi]] += 1
        h2n[ai[gi], hi[gi]] += 1
    ngames = Hm.sum(0) + Am.sum(0)
    pct = wins / ngames
    coin = rng.random((n, 32))
    n_po = n_playoff or (7 if int(season_games.season.iloc[0]) >= 2020 else 6)
    divwin = np.zeros((n, 32), bool)
    playoff = np.zeros((n, 32), bool)
    seed1 = np.zeros((n, 32), bool)
    seeds = np.zeros((n, 32), np.int8)
    div_members = {k: [TIX[t] for t in v] for k, v in DIVS.items()}
    for s in range(n):
        P, DP, CP, C, HH = pct[s].tolist(), div_pct[s].tolist(), conf_pct[s].tolist(), coin[s].tolist(), h2h[s]

        def order(ts, same_div):
            # sort by pct, then within exact-pct groups apply tiebreakers
            ts = sorted(ts, key=lambda t: -P[t])
            out, i = [], 0
            while i < len(ts):
                j = i
                while j + 1 < len(ts) and abs(P[ts[j + 1]] - P[ts[i]]) < 1e-9:
                    j += 1
                grp = ts[i:j + 1]
                if len(grp) > 1:
                    def key(t):
                        gw = sum(HH[t, o] for o in grp if o != t)
                        gn = sum(h2n[t, o] for o in grp if o != t)
                        # head-to-head counts only if every tied pair met (always true inside a division;
                        # across divisions this approximates the NFL "sweep" rule)
                        return (-(gw / gn) if (gn and all_met) else 0.0, -DP[t] if same_div else 0.0, -CP[t], C[t])
                    all_met = all(h2n[a_, b_] > 0 for ii, a_ in enumerate(grp) for b_ in grp[ii + 1:])
                    grp = sorted(grp, key=key)
                out += grp
                i = j + 1
            return out

        for conf in ("AFC", "NFC"):
            dlist = [order(m, True) for k, m in div_members.items() if k.startswith(conf)]
            winners = order([dl[0] for dl in dlist], False)
            rest = [dl[1:] for dl in dlist]
            wc = []
            while len(wc) < n_po - 4:
                cands = [r[0] for r in rest if r]
                best = order(cands, False)[0]
                wc.append(best)
                for r in rest:
                    if r and r[0] == best:
                        r.pop(0)
            for k_, t in enumerate(winners + wc):
                seeds[s, t] = k_ + 1
            divwin[s, winners] = True
            playoff[s, winners + wc] = True
            seed1[s, winners[0]] = True
    out.update(divwin=divwin, playoff=playoff, seed1=seed1, seeds=seeds)
    return out


def summarize(sim, lines=None):
    w = sim["wins"]
    rows = []
    for t in TEAMS:
        i = TIX[t]
        r = {"team": t, "division": DIV_OF[i], "mean_wins": float(w[:, i].mean()), "sd_wins": float(w[:, i].std())}
        if "playoff" in sim:
            r.update(p_division=float(sim["divwin"][:, i].mean()), p_playoffs=float(sim["playoff"][:, i].mean()),
                     p_seed1=float(sim["seed1"][:, i].mean()))
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================== betting helpers
def payout(odds):
    odds = np.asarray(odds, float)
    return np.where(odds < 0, 100 / np.abs(odds), odds / 100)


def no_vig(o_over, o_under):
    io = np.where(o_over < 0, -o_over / (-o_over + 100), 100 / (o_over + 100))
    iu = np.where(o_under < 0, -o_under / (-o_under + 100), 100 / (o_under + 100))
    return io / (io + iu)


def roi_table(b, label):
    """b: rows with profit (per 1 unit staked), season."""
    if len(b) == 0:
        return {"label": label, "n": 0}
    p = b.profit.values
    seas = b.groupby("season").profit.sum()
    # season-clustered SE of the mean profit per bet
    cl = np.sqrt(((seas - b.groupby("season").size() * p.mean()) ** 2).sum()) / len(p)
    return {"label": label, "n": int(len(p)), "win": int((p > 0).sum()), "loss": int((p < 0).sum()),
            "push": int((p == 0).sum()), "roi": float(p.mean()), "se": float(p.std(ddof=1) / np.sqrt(len(p))),
            "se_cluster": float(cl)}


# ============================================================== 1. preseason win-total test
def preseason(df, wt, shrink_m, unc_m, shrink_k, unc_k):
    rows = []
    for s in DEV + HOLD + [2026]:
        sd = df[df.season == s].copy()
        asof = sd.gameday.min() - pd.Timedelta(days=1)
        G = sd.copy()
        G["completed"] = False
        h = G.week.values - 1
        mdl = walk_forward_model(DF_ALL, s)
        ctx, team = model_mu_parts(mdl, G, team_state(df, s, asof))
        mu_model = ctx + apply_shrink(team, h, 1, shrink_m)
        sim_m = simulate(G, mu_model, h, unc_m["pre"], standings=False, seed=s)
        ctx2, team2 = market_mu_parts(df, G, asof)
        mu_mkt = ctx2 + apply_shrink(team2, h, 1, shrink_k)
        sim_k = simulate(G, mu_mkt, h, unc_k["pre"], standings=False, seed=s + 1)
        aw = actual_wins(df, s) if s < 2026 else np.full(32, np.nan)
        L = wt[wt.season == s].set_index("team")
        for t in TEAMS:
            i = TIX[t]
            line = L.loc[t, "line"]
            wm, wk = sim_m["wins"][:, i], sim_k["wins"][:, i]
            rows.append({"season": s, "team": t, "line": line, "over_odds": L.loc[t, "over_odds"],
                         "under_odds": L.loc[t, "under_odds"], "line_adj": L.loc[t, "line_adj"],
                         "nv_over": float(no_vig(L.loc[t, "over_odds"], L.loc[t, "under_odds"])),
                         "model_mean": float(wm.mean()), "model_p_over": float((wm > line).mean()),
                         "model_p_under": float((wm < line).mean()),
                         "mkt_mean": float(wk.mean()), "mkt_p_over": float((wk > line).mean()),
                         "mkt_p_under": float((wk < line).mean()), "actual": aw[i]})
    return pd.DataFrame(rows)


def bets_from(pt, mean_col, thr, p_over_col=None, p_under_col=None, ev_thr=None):
    b = pt.dropna(subset=["actual"]).copy()
    if ev_thr is None:
        diff = b[mean_col] - b.line
        b = b[diff.abs() >= thr].copy()
        b["side"] = np.where(b[mean_col] > b.line, "over", "under")
    else:
        evo = b[p_over_col] * payout(b.over_odds) - b[p_under_col]
        evu = b[p_under_col] * payout(b.under_odds) - b[p_over_col]
        b["ev"] = np.maximum(evo, evu)
        b["side"] = np.where(evo >= evu, "over", "under")
        b = b[b.ev >= ev_thr].copy()
    win = np.where(b.side == "over", b.actual > b.line, b.actual < b.line)
    push = b.actual == b.line
    odds = np.where(b.side == "over", b.over_odds, b.under_odds)
    b["profit"] = np.where(push, 0.0, np.where(win, payout(odds), -1.0))
    return b


def preseason_report(pt):
    out = {"accuracy": {}, "bets": [], "blend": {}}
    for lab, seas in (("dev_2015_19", DEV), ("holdout_2020_25", HOLD)):
        d = pt[pt.season.isin(seas)].dropna(subset=["actual"])
        out["accuracy"][lab] = {
            "n": int(len(d)),
            "rmse_line_adj": float(np.sqrt(((d.actual - d.line_adj) ** 2).mean())),
            "rmse_model": float(np.sqrt(((d.actual - d.model_mean) ** 2).mean())),
            "rmse_mkt_sim": float(np.sqrt(((d.actual - d.mkt_mean) ** 2).mean())),
            "corr_model_minus_line_vs_actual_minus_line": float(np.corrcoef(d.model_mean - d.line_adj, d.actual - d.line_adj)[0, 1]),
            "mean_abs_model_minus_line": float((d.model_mean - d.line).abs().mean()),
        }
        for ver, mc, po, pu in (("model", "model_mean", "model_p_over", "model_p_under"),
                                ("market_sim", "mkt_mean", "mkt_p_over", "mkt_p_under")):
            for thr in (0.5, 1.0, 1.5):
                r = roi_table(bets_from(d.assign(), mc, thr), f"{ver} |diff|>={thr}")
                out["bets"].append({"set": lab, **r})
            for ev in (0.03, 0.08):
                r = roi_table(bets_from(d, mc, None, po, pu, ev), f"{ver} EV>={ev}")
                out["bets"].append({"set": lab, **r})
    for lab, seas in (("dev_2015_19", DEV), ("holdout_2020_25", HOLD)):
        d = pt[pt.season.isin(seas)].dropna(subset=["actual"])
        d = d[d.actual != d.line]
        y = (d.actual > d.line).astype(float)
        out["accuracy"][lab].update({
            "brier_p_over_novig": float(((d.nv_over - y) ** 2).mean()),
            "brier_p_over_model": float(((d.model_p_over / (d.model_p_over + d.model_p_under) - y) ** 2).mean()),
            "brier_p_over_mkt_sim": float(((d.mkt_p_over / (d.mkt_p_over + d.mkt_p_under) - y) ** 2).mean()),
            "over_rate": float(y.mean())})
    # blend weight fit on dev: actual - line_adj = b*(model - line_adj)
    d = pt[pt.season.isin(DEV)].dropna(subset=["actual"])
    x, y = d.model_mean - d.line_adj, d.actual - d.line_adj
    b = float((x * y).sum() / (x * x).sum())
    se_b = float(np.sqrt(((y - b * x) ** 2).sum() / (len(x) - 1) / (x * x).sum()))
    h = pt[pt.season.isin(HOLD)].dropna(subset=["actual"])
    xh, yh = h.model_mean - h.line_adj, h.actual - h.line_adj
    bh = float((xh * yh).sum() / (xh * xh).sum())
    se_bh = float(np.sqrt(((yh - bh * xh) ** 2).sum() / (len(xh) - 1) / (xh * xh).sum()))
    out["blend"] = {"dev_weight_on_model": b, "dev_se": se_b, "holdout_weight_refit": bh, "holdout_se": se_bh,
                    "holdout_rmse_line_adj": float(np.sqrt((yh ** 2).mean())),
                    "holdout_rmse_blend_devweight": float(np.sqrt(((yh - b * xh) ** 2).mean()))}
    return out


# ============================================================== 2. QB analysis
def market_qb_points(df):
    """How many points the closing spread moves per unit of QB downgrade (qb_change, EPA/dropback):
    residual of the closing spread vs market ratings fitted on all earlier spreads (leaving the game out),
    regressed on qb_change_diff. 2015-2025, weeks 3+."""
    rows = []
    for (s, w), g in df[(df.season.between(2015, 2025)) & (df.week >= 3)].groupby(["season", "week"]):
        asof = g.gameday.min() - pd.Timedelta(days=1)
        r, hfa = market_ratings(df, asof, upto_next_week=False)
        pred = r[g.hi.values] - r[g.ai.values] + hfa * (g.location.values != "Neutral")
        rows.append(pd.DataFrame({"resid": g.spread_line.values - pred, "qcd": g.qb_change_diff.values,
                                  "season": s}))
    d = pd.concat(rows).dropna()
    x, y = d.qcd.values, d.resid.values
    b = float((x * y).sum() / (x * x).sum())
    se = float(np.sqrt(((y - b * x) ** 2).sum() / (len(x) - 1) / (x * x).sum()))
    return {"pts_per_epa_db": b, "se": se, "n": int(len(d))}


def qb_events(df):
    """Starter absences 2015-2025: the team's primary QB (most starts in its last 4 games) does not
    start for >= 3 straight games, beginning in week 4-14."""
    ev = []
    for s in range(2015, 2026):
        sd = df[(df.season == s) & df.completed]
        for t in TEAMS:
            tg = sd[(sd.home_team == t) | (sd.away_team == t)].sort_values("gameday")
            qbs = np.where(tg.home_team == t, tg.home_qb_id, tg.away_qb_id)
            weeks = tg.week.values
            k = 4
            while k < len(tg):
                prim = pd.Series(qbs[max(0, k - 4):k]).mode()
                if len(prim) == 0 or not (3 <= weeks[k] <= 14):
                    k += 1
                    continue
                p = prim.iloc[0]
                if (qbs[max(0, k - 4):k] == p).sum() < 3 or qbs[k] == p:
                    k += 1
                    continue
                L = 0
                while k + L < len(tg) and qbs[k + L] != p:
                    L += 1
                if L >= 3:
                    ev.append({"season": s, "team": t, "week": int(weeks[k]), "games_missed": int(L),
                               "rest_of_season": bool(k + L == len(tg)),
                               "first_game": tg.game_id.values[k], "gameday": tg.gameday.values[k]})
                k += max(L, 1)
    return pd.DataFrame(ev)


def qb_event_study(df, events, mq, shrink_k, unc_k, n=4000):
    """For each event: market-implied playoff odds (a) stale = ratings from spreads before the news, no QB
    adjustment; (b) informed = same ratings, minus the market-sized QB downgrade for the games actually
    missed (duration = hindsight, optimistic); (c) next-line = refit including the first missed game's line.
    Outcome: made playoffs. Tests whether the QB adjustment improves Brier."""
    out = []
    for e in events.itertuples(index=False):
        sd = df[df.season == e.season].copy()
        g0 = sd[sd.game_id == e.first_game].iloc[0]
        asof = pd.Timestamp(e.gameday) - pd.Timedelta(days=5)
        G = sd.copy()
        G.loc[G.gameday >= asof, "completed"] = False
        rem = G[~G.completed]
        cur_w = int(rem.week.min())
        h = rem.week.values - cur_w
        r, hfa = market_ratings(df, asof, upto_next_week=False)
        ctx, team = market_mu_parts(df, rem, asof, r, hfa)
        mu_stale = ctx + apply_shrink(team, h, cur_w, shrink_k)
        side = 1 if g0.home_team == e.team else -1
        qc = (g0.home_qb_change if side == 1 else g0.away_qb_change)
        drop = mq["pts_per_epa_db"] * qc   # negative = worse
        tg = rem[(rem.home_team == e.team) | (rem.away_team == e.team)].sort_values("gameday")
        miss = set(tg.game_id.values[:e.games_missed])
        adj = np.array([(drop if gid in miss else 0.0) * (1 if ht == e.team else -1)
                        for gid, ht in zip(rem.game_id, rem.home_team)])
        mu_inf = mu_stale + adj
        r2, hfa2 = market_ratings(df, asof, upto_next_week=True)
        ctx2, team2 = market_mu_parts(df, rem, asof, r2, hfa2)
        mu_next = ctx2 + apply_shrink(team2, h, cur_w, shrink_k)
        i = TIX[e.team]
        res = {}
        for lab, mu in (("stale", mu_stale), ("informed", mu_inf), ("next_line", mu_next)):
            sim = simulate(G, mu, h, unc_k["in"], n=n, seed=e.season * 1000 + TIX[e.team] * 10 + len(lab))
            res[lab] = (float(sim["playoff"][:, i].mean()), float(sim["wins"][:, i].mean()))
        # realized playoff: re-run with all games completed (n=1 deterministic up to coin flips)
        full = sd.copy()
        fin = simulate(full, np.array([]), np.array([0]), unc_k["in"], n=50, seed=1)
        made = float(fin["playoff"][:, i].mean())
        out.append({**e._asdict(), "qb_change": float(qc), "pts_drop_per_game": float(drop),
                    "p_po_stale": res["stale"][0], "p_po_informed": res["informed"][0],
                    "p_po_next_line": res["next_line"][0], "wins_stale": res["stale"][1],
                    "wins_informed": res["informed"][1], "made_playoffs": made})
    return pd.DataFrame(out)


# ============================================================== 3. current season
def current_season(df, wt, shrink_m, unc_m, shrink_k, unc_k, mq, today):
    s = int(df[df.gameday <= today].season.max())
    sd = df[df.season == s].copy()
    asof = today
    G = sd.copy()
    G.loc[G.gameday >= asof, "completed"] = False
    rem = G[~G.completed]
    cur_w = int(rem.week.min())
    h = rem.week.values - cur_w
    mdl = walk_forward_model(DF_ALL, s)
    st = team_state(df, s, asof)
    ctx, team = model_mu_parts(mdl, rem, st)
    mu_m = ctx + apply_shrink(team, h, cur_w, shrink_m)
    # market: posted lines for the upcoming week; fitted ratings for later weeks
    r, hfa = market_ratings(df, asof)
    ctx2, team2 = market_mu_parts(df, rem, asof, r, hfa)
    mu_k = ctx2 + apply_shrink(team2, h, cur_w, shrink_k)
    posted = rem.spread_line.notna().values & (rem.gameday < asof + pd.Timedelta(days=7)).values
    mu_k = np.where(posted, rem.spread_line.values, mu_k)
    sim_m = simulate(G, mu_m, h, unc_m["in"], seed=11)
    sim_k = simulate(G, mu_k, h, unc_k["in"], seed=12)
    sm, sk = summarize(sim_m), summarize(sim_k)
    W = wt[wt.season == s].set_index("team")
    rec = actual_wins(df, s)
    teams = []
    for t in TEAMS:
        i = TIX[t]
        a, b = sm.iloc[i], sk.iloc[i]
        teams.append({
            "team": t, "division": DIV_OF[i], "conference": CONF_OF[i],
            "wins_so_far": float(rec[i]), "preseason_win_total": float(W.loc[t, "line"]) if t in W.index else None,
            "model": {"mean_wins": round(a.mean_wins, 2), "p_over_preseason_total": float((sim_m["wins"][:, i] > W.loc[t, "line"]).mean()),
                      "p_division": round(a.p_division, 4), "p_playoffs": round(a.p_playoffs, 4), "p_seed1": round(a.p_seed1, 4),
                      "win_dist": np.bincount(sim_m["wins"][:, i].round().astype(int), minlength=18)[:18].tolist()},
            "market": {"mean_wins": round(b.mean_wins, 2), "p_over_preseason_total": float((sim_k["wins"][:, i] > W.loc[t, "line"]).mean()),
                       "p_division": round(b.p_division, 4), "p_playoffs": round(b.p_playoffs, 4), "p_seed1": round(b.p_seed1, 4),
                       "rating_pts": round(float(r[i]), 2)},
        })
    # QB sensitivity (market version): starter out next 4 games / rest of season
    qb = []
    base_po = sk.set_index("team").p_playoffs
    for t in TEAMS:
        i = TIX[t]
        tq = st.loc[t]
        qc = (-0.10 - tq.qb_rating)          # backup at replacement-level prior vs current starter
        drop = mq["pts_per_epa_db"] * qc
        tg = rem[(rem.home_team == t) | (rem.away_team == t)].sort_values("gameday").game_id.values
        rowres = {"team": t, "starter_qb_rating": round(float(tq.qb_rating), 3), "pts_per_game_if_out": round(float(drop), 2),
                  "p_playoffs_base": round(float(base_po[t]), 4)}
        for lab, nmiss in (("out_4", 4), ("out_season", len(tg))):
            miss = set(tg[:nmiss])
            adj = np.array([(drop if gid in miss else 0.0) * (1 if ht == t else -1)
                            for gid, ht in zip(rem.game_id, rem.home_team)])
            sim = simulate(G, mu_k + adj, h, unc_k["in"], n=4000, seed=100 + i)
            rowres[f"p_playoffs_{lab}"] = round(float(sim["playoff"][:, i].mean()), 4)
            rowres[f"mean_wins_{lab}"] = round(float(sim["wins"][:, i].mean()), 2)
        qb.append(rowres)
    return {"season": s, "as_of": str(asof.date()), "next_week": cur_w,
            "completed_games": int(G.completed.sum()), "market_hfa": round(float(hfa), 2),
            "teams": teams, "qb_out_sensitivity_market": qb}


# ============================================================== main
DF_ALL = None


def main():
    global DF_ALL
    if "--fetch" in sys.argv or not WT_CSV.exists():
        fetch_win_totals()
    wt = pd.read_csv(WT_CSV)
    df = load_features()                      # REG only, normalized teams (builds the cache if missing)
    DF_ALL = pd.read_parquet(SCRATCH / "feat.parquet")   # all game types, for walk-forward model training
    DF_ALL["gameday"] = pd.to_datetime(DF_ALL["gameday"])
    res = {"generated": str(dt.date.today()), "n_sims": N_SIMS}

    # --- market half-life choice on dev (predict future closing spreads)
    hl_res = {}
    for hl in (3.0, 6.0, 10.0):
        errs = []
        for s in DEV:
            sd = df[df.season == s]
            for w0 in START_WEEKS:
                asof = sd[sd.week == w0].gameday.min() - pd.Timedelta(days=1)
                g = sd[sd.week >= w0]
                r, hfa = market_ratings(df, asof, halflife=hl)
                pred = r[g.hi.values] - r[g.ai.values] + hfa * (g.location.values != "Neutral")
                errs.append(g.spread_line.values - pred)
        hl_res[hl] = float(np.sqrt(np.nanmean(np.concatenate(errs) ** 2)))
    globals()["MKT_HALFLIFE"] = min(hl_res, key=hl_res.get)
    res["market_halflife_rmse_vs_future_spreads_dev"] = hl_res
    res["market_halflife"] = MKT_HALFLIFE
    print("halflife", hl_res)

    # --- shrink + uncertainty on dev
    hf_m = horizon_frame(df, DEV, "model")
    hf_k = horizon_frame(df, DEV, "market")
    shrink_m, shrink_k = fit_shrink(hf_m), fit_shrink(hf_k)
    unc_m, unc_k = fit_uncertainty(hf_m, shrink_m), fit_uncertainty(hf_k, shrink_k)
    res["calibration_dev"] = {"shrink_model": shrink_m, "shrink_market": shrink_k,
                              "uncertainty_model": unc_m, "uncertainty_market": unc_k}
    print(json.dumps(res["calibration_dev"], indent=1)[:2000])

    # --- sanity: realized next-week + season-long win prob log loss (model vs market) on holdout
    hf_mh = horizon_frame(df, HOLD, "model")
    hf_kh = horizon_frame(df, HOLD, "market")
    cal = []
    from scipy.stats import norm
    for lab, hf, sh, un in (("model", hf_mh, shrink_m, unc_m), ("market", hf_kh, shrink_k, unc_k)):
        hf = hf.dropna(subset=["margin"])
        hf = hf[hf.margin != 0]
        for pre in (True, False):
            sub = hf[(hf.w0 == 1) == pre]
            key = "pre" if pre else "in"
            mu = sub.ctx + sub.team * np.clip(sh[key][0] + sh[key][1] * sub.h, .05, 1.5)
            u = un[key]
            sd_ = np.sqrt(u["sig_eps"] ** 2 + 2 * (u["tau_a"] ** 2 + u["tau_b"] ** 2 * sub.h))
            p = np.clip(norm.cdf(mu / sd_), 1e-4, 1 - 1e-4)
            y = (sub.margin > 0).astype(float)
            for hb, m_ in (("h0-3", sub.h <= 3), ("h4-9", sub.h.between(4, 9)), ("h10+", sub.h >= 10)):
                yy, pp = y[m_], p[m_]
                cal.append({"version": lab, "start": key, "horizon": hb, "n": int(m_.sum()),
                            "log_loss": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))),
                            "mean_p_fav": float(np.maximum(pp, 1 - pp).mean()),
                            "fav_win_rate": float(np.where(pp >= .5, yy, 1 - yy).mean())})
    res["holdout_game_prob_by_horizon"] = cal

    # --- 1. preseason test
    pt = preseason(df, wt, shrink_m, unc_m, shrink_k, unc_k)
    pt.to_csv(SCRATCH / "preseason_table.csv", index=False)
    res["preseason"] = preseason_report(pt)
    print(json.dumps(res["preseason"], indent=1)[:3000])

    # --- division odds sanity (partial, LLM-extracted)
    dpath = FUT / "division_odds_top2_2022_2025_partial.csv"
    if dpath.exists():
        res["division_odds_check"] = division_check(df, dpath, shrink_m, unc_m)

    # --- 2. QB analysis
    mq = market_qb_points(df)
    res["market_qb_points"] = mq
    mdl25 = walk_forward_model(DF_ALL, 2025)
    coefs = M.coefficients(mdl25)
    # model points per EPA/db of starter downgrade: qb_diff + qb_change_diff coefficients / feature sd
    sc = mdl25.pipe[0]
    sd_map = dict(zip(FEATURES, sc.scale_))
    res["model_qb_points"] = {"pts_per_epa_db_qb_diff": coefs["qb_diff"] / sd_map["qb_diff"],
                              "pts_per_epa_db_qb_change": coefs["qb_change_diff"] / sd_map["qb_change_diff"]}
    ev = qb_events(df)
    es = qb_event_study(df, ev, mq, shrink_k, unc_k)
    es.to_csv(SCRATCH / "qb_events.csv", index=False)
    res["qb_event_study"] = qb_summary(es)

    # --- 3. 2026
    cur = current_season(df, wt, shrink_m, unc_m, shrink_k, unc_k, mq, pd.Timestamp(dt.date.today()))
    res["current"] = {k: v for k, v in cur.items() if k not in ("teams", "qb_out_sensitivity_market")}
    M.save_json(cur | {"method": "scripts/research/season_sim.py", "n_sims": N_SIMS,
                       "calibration_dev": res["calibration_dev"]}, OUT / "season_sim_2026.json")
    res["preseason_rows_2026"] = pt[pt.season == 2026].to_dict("records")
    M.save_json(res, OUT / "season_sim.json")
    write_md(res, pt, es, cur)


def division_check(df, path, shrink_m, unc_m):
    d = pd.read_csv(path)
    d["team"] = d["team"].replace(TEAM_FIX)
    rows = []
    for s in sorted(d.season.unique()):
        sd = df[df.season == s].copy()
        asof = sd.gameday.min() - pd.Timedelta(days=1)
        G = sd.copy()
        G["completed"] = False
        h = G.week.values - 1
        ctx, team = model_mu_parts(walk_forward_model(DF_ALL, s), G, team_state(df, s, asof))
        sim = simulate(G, ctx + apply_shrink(team, h, 1, shrink_m), h, unc_m["pre"], n=4000, seed=s)
        fin = simulate(sd, np.array([]), np.array([0]), unc_m["pre"], n=20, seed=1)
        for r in d[d.season == s].itertuples(index=False):
            i = TIX[r.team]
            o = r.american_odds
            imp = (-o / (-o + 100)) if o < 0 else 100 / (o + 100)
            pm = float(sim["divwin"][:, i].mean())
            won = float(fin["divwin"][:, i].mean() > .5)
            ev = pm * payout(o) - (1 - pm)
            rows.append({"season": int(s), "team": r.team, "odds": int(o), "implied": imp, "model_p": pm,
                         "won": won, "ev": float(ev), "profit_if_bet": float(payout(o)) if won else -1.0})
    t = pd.DataFrame(rows)
    t.to_csv(SCRATCH / "division_check.csv", index=False)
    b = t[t.ev > 0.05]
    return {"n_listed": int(len(t)), "brier_implied_with_vig": float(((t.implied - t.won) ** 2).mean()),
            "brier_model": float(((t.model_p - t.won) ** 2).mean()),
            "bets_ev_gt_5pct": int(len(b)), "bets_won": int(b.won.sum()),
            "roi": float(b.profit_if_bet.mean()) if len(b) else None}


def qb_summary(es):
    if es.empty:
        return {}
    y = es.made_playoffs
    out = {"n_events": int(len(es)), "rest_of_season_events": int(es.rest_of_season.sum()),
           "median_games_missed": float(es.games_missed.median()),
           "mean_pts_drop_per_game": float(es.pts_drop_per_game.mean()),
           "mean_delta_po_informed_minus_stale": float((es.p_po_informed - es.p_po_stale).mean()),
           "mean_abs_delta_po": float((es.p_po_informed - es.p_po_stale).abs().mean()),
           "mean_delta_po_nextline_minus_stale": float((es.p_po_next_line - es.p_po_stale).mean()),
           "mean_delta_wins_informed_minus_stale": float((es.wins_informed - es.wins_stale).mean()),
           "realized_playoff_rate": float(y.mean()), "mean_p_stale": float(es.p_po_stale.mean()),
           "mean_p_informed": float(es.p_po_informed.mean()), "mean_p_next_line": float(es.p_po_next_line.mean()),
           "brier_stale": float(((es.p_po_stale - y) ** 2).mean()),
           "brier_informed": float(((es.p_po_informed - y) ** 2).mean()),
           "brier_next_line": float(((es.p_po_next_line - y) ** 2).mean())}
    big = es[(es.p_po_stale.between(0.15, 0.85))]
    out["contending_events(stale p 15-85%)"] = {
        "n": int(len(big)), "mean_delta_po": float((big.p_po_informed - big.p_po_stale).mean()) if len(big) else None,
        "realized": float(big.made_playoffs.mean()) if len(big) else None,
        "mean_p_stale": float(big.p_po_stale.mean()) if len(big) else None,
        "mean_p_informed": float(big.p_po_informed.mean()) if len(big) else None}
    d = es.p_po_informed - es.p_po_stale
    out["paired_brier_gain"] = float((((es.p_po_stale - y) ** 2) - ((es.p_po_informed - y) ** 2)).mean())
    out["paired_brier_gain_se"] = float(((((es.p_po_stale - y) ** 2) - ((es.p_po_informed - y) ** 2))).std(ddof=1) / np.sqrt(len(es)))
    return out


def write_md(res, pt, es, cur):
    from textwrap import dedent
    L = []
    P = res["preseason"]
    L.append("# Season simulator & futures test\n")
    L.append(f"Generated {res['generated']} by `scripts/research/season_sim.py` ({N_SIMS:,} sims per run).\n")
    L.append("## Data\n")
    L.append("* Win totals 2013-2026 with over/under juice: `data/research_futures/win_totals_2013_2026.csv`, from nfelo's "
             "open `wt_ratings.csv` (github.com/greerreNFL/nfelosrs). Pre-2025 rows are a late-preseason (~Aug 30) "
             "consensus with real juice; 2025-26 are DraftKings at the start of the season. sportsoddshistory.com "
             "(now covers.com) was blocked by the egress proxy / returned 404.\n")
    L.append("* Division odds: only the top-2 favorites per division 2022-2025, extracted by an LLM page "
             "summarizer from sportsbettingdime.com (`division_odds_top2_2022_2025_partial.csv`), unverified, "
             "single book unknown, no complete market -> informational only. No playoff-odds or in-season futures archive found.\n")
    c = res["calibration_dev"]
    L.append("## Simulator calibration (fit on 2015-2019)\n")
    L.append(f"* Market ratings half-life {res['market_halflife']} wk (RMSE vs future closing spreads: "
             + ", ".join(f"{k}: {v:.2f}" for k, v in res['market_halflife_rmse_vs_future_spreads_dev'].items()) + ")")
    for v in ("model", "market"):
        sh, un = c[f"shrink_{v}"], c[f"uncertainty_{v}"]
        L.append(f"* {v}: shrink k(h)=k0+k1*h  preseason {sh['pre'][0]:.2f}{sh['pre'][1]:+.3f}h, in-season "
                 f"{sh['in'][0]:.2f}{sh['in'][1]:+.3f}h; rating sd tau_a pre {un['pre']['tau_a']:.2f} / in {un['in']['tau_a']:.2f} pts, "
                 f"random-walk tau_b pre {un['pre']['tau_b']:.2f} / in {un['in']['tau_b']:.2f} pts/sqrt(wk), game noise "
                 f"{un['pre']['sig_eps']:.1f}/{un['in']['sig_eps']:.1f}")
    L.append("\nHoldout (2020-25) game-level log loss of the sim's per-game win probability by weeks ahead:\n")
    L.append("| version | start | horizon | n | log loss | fav win% | mean p(fav) |\n|---|---|---|---|---|---|---|")
    for r in res["holdout_game_prob_by_horizon"]:
        L.append(f"| {r['version']} | {r['start']} | {r['horizon']} | {r['n']} | {r['log_loss']:.4f} | {r['fav_win_rate']:.3f} | {r['mean_p_fav']:.3f} |")
    L.append("\n## 1. Preseason win totals: model sim vs posted totals\n")
    L.append("Model = production ridge trained on seasons < S; team inputs as of the day before week 1 (Elo with 1/3 "
             "regression, QB rating of the week-1 starter + qb_change vs the team's previous QBs, prior-season EWMAs), "
             "shrunk by k(h) fit on 2015-19. Market sim = ratings fitted to prior-season spreads (offseason-discounted) + week-1 lines. "
             "Bets at the listed juice; pushes refunded. SE = naive per-bet; SEc = season-clustered.\n")
    L.append("| set | n teams | RMSE line_adj | RMSE model | RMSE market-sim | corr(model-line, actual-line) | Brier P(over): no-vig / model / mkt-sim |\n|---|---|---|---|---|---|---|")
    for k, a in P["accuracy"].items():
        L.append(f"| {k} | {a['n']} | {a['rmse_line_adj']:.3f} | {a['rmse_model']:.3f} | {a['rmse_mkt_sim']:.3f} | {a['corr_model_minus_line_vs_actual_minus_line']:+.3f} | "
                 f"{a['brier_p_over_novig']:.4f} / {a['brier_p_over_model']:.4f} / {a['brier_p_over_mkt_sim']:.4f} |")
    L.append("\n| set | rule | bets | W-L-P | ROI | SE | SEc |\n|---|---|---|---|---|---|---|")
    for b in P["bets"]:
        if b["n"] == 0:
            L.append(f"| {b['set']} | {b['label']} | 0 | | | | |")
            continue
        L.append(f"| {b['set']} | {b['label']} | {b['n']} | {b['win']}-{b['loss']}-{b['push']} | {b['roi']:+.3f} | {b['se']:.3f} | {b['se_cluster']:.3f} |")
    bl = P["blend"]
    L.append(f"\nInformation test: (actual - line_adj) = b x (model - line_adj). Dev b = {bl['dev_weight_on_model']:.2f} ± {bl['dev_se']:.2f}; "
             f"holdout refit b = {bl['holdout_weight_refit']:.2f} ± {bl['holdout_se']:.2f}. Holdout RMSE line {bl['holdout_rmse_line_adj']:.3f} "
             f"vs blend with dev weight {bl['holdout_rmse_blend_devweight']:.3f}.\n")
    if "division_odds_check" in res:
        dc = res["division_odds_check"]
        L.append(f"Division odds sanity (partial top-2 list, n={dc['n_listed']}): Brier implied(with vig) {dc['brier_implied_with_vig']:.3f} "
                 f"vs model {dc['brier_model']:.3f}; model EV>5% bets {dc['bets_ev_gt_5pct']} (won {dc['bets_won']}), ROI {dc['roi']}.\n")
    L.append("## 2. In-season: QB injuries and futures\n")
    mq, mm = res["market_qb_points"], res["model_qb_points"]
    L.append(f"* Market: closing spread moves {mq['pts_per_epa_db']:.1f} ± {mq['se']:.1f} pts per 1.0 EPA/dropback of starter "
             f"change (n={mq['n']}). Model coefficients: qb_diff {mm['pts_per_epa_db_qb_diff']:.1f}, qb_change {mm['pts_per_epa_db_qb_change']:.1f} pts per EPA/db.")
    q = res["qb_event_study"]
    if q:
        L.append(f"* {q['n_events']} starter absences of 3+ games (2015-25, start wk 3-14; {q['rest_of_season_events']} lasted the rest of the season; "
                 f"median {q['median_games_missed']:.0f} games). Market-sized downgrade {q['mean_pts_drop_per_game']:.1f} pts/game.")
        L.append(f"* Market-implied playoff probability: stale (no QB news) {q['mean_p_stale']:.3f} -> informed {q['mean_p_informed']:.3f} "
                 f"(mean move {q['mean_delta_po_informed_minus_stale']:+.3f}, mean |move| {q['mean_abs_delta_po']:.3f}); refit with the next posted line only "
                 f"{q['mean_p_next_line']:.3f}. Realized rate {q['realized_playoff_rate']:.3f}.")
        L.append(f"* Brier: stale {q['brier_stale']:.4f}, informed {q['brier_informed']:.4f}, next-line {q['brier_next_line']:.4f}; paired gain "
                 f"{q['paired_brier_gain']:+.4f} ± {q['paired_brier_gain_se']:.4f}.")
        cc = q["contending_events(stale p 15-85%)"]
        L.append(f"* Contenders (stale p 15-85%, n={cc['n']}): stale {cc['mean_p_stale']:.3f}, informed {cc['mean_p_informed']:.3f}, realized {cc['realized']:.3f}.")
        L.append("* Caveats: absence length uses hindsight (actual games missed), events include benchings, and there are no "
                 "historical futures PRICES, so this measures how far a stale (pre-news) market-implied price is from an "
                 "informed one, not a realized betting ROI. 'next_line' shows that refitting ratings with only the first "
                 "post-news spread recovers ~1/3 of the move: rating systems built from past spreads lag QB news.\n")
    qbm_ = pd.DataFrame(cur["qb_out_sensitivity_market"])
    con = qbm_[qbm_.p_playoffs_base.between(0.2, 0.8)]
    L.append(f"2026 sensitivity (market version, backup at replacement-level QB rating -0.10): for {len(con)} teams with 20-80% "
             f"playoff odds, starter out 4 games moves P(playoffs) by median {(con.p_playoffs_out_4 - con.p_playoffs_base).median():+.3f}, "
             f"out for the season by median {(con.p_playoffs_out_season - con.p_playoffs_base).median():+.3f} "
             f"(range {(con.p_playoffs_out_season - con.p_playoffs_base).min():+.2f} to {(con.p_playoffs_out_season - con.p_playoffs_base).max():+.2f}); "
             f"full table in season_sim_2026.json.\n")
    L.append(f"## 3. {cur['season']} projections (as of {cur['as_of']}, next week {cur['next_week']}, {cur['completed_games']} games final)\n")
    L.append("| team | W | pre total | model wins | mkt wins | model P(div) | mkt P(div) | model P(PO) | mkt P(PO) | mkt P(#1) | QB out rest-of-season P(PO) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    qbm = {r["team"]: r for r in cur["qb_out_sensitivity_market"]}
    for t in sorted(cur["teams"], key=lambda r: -r["market"]["p_playoffs"]):
        L.append(f"| {t['team']} | {t['wins_so_far']:.0f} | {t['preseason_win_total']} | {t['model']['mean_wins']:.1f} | {t['market']['mean_wins']:.1f} | "
                 f"{t['model']['p_division']:.2f} | {t['market']['p_division']:.2f} | {t['model']['p_playoffs']:.2f} | {t['market']['p_playoffs']:.2f} | "
                 f"{t['market']['p_seed1']:.2f} | {qbm[t['team']]['p_playoffs_out_season']:.2f} |")
    L.append("\n## Verdict\n")
    L.append("* Preseason win totals: the model's projection carries no information beyond the posted total "
             "(regression weight ~0 on dev and holdout; RMSE and P(over) Brier worse than the no-vig line). "
             "Dev ROI looked positive; every model rule lost on the 2020-25 holdout. Do not bet model-vs-total disagreements.")
    L.append("* The only rule positive in both dev and holdout is market_sim |diff|>=1.5 (prior-season spreads + week-1 lines vs the "
             "total; 28-13-2 combined), but it is 1 of 10 rules looked at on the holdout, n is small (~1.5 SE), and week-1 CLOSING "
             "lines post-date the win-total snapshot. Treat as a lead to paper-track with timestamped look-ahead lines, not an edge.")
    L.append("* In-season: properly pricing a starting-QB loss moves playoff odds by ~2-11 pts (4 games, median 5) and ~8-41 pts (season, median 20) for "
             "contenders, and the informed price beats a stale one on realized outcomes (paired Brier gain ~2.7 SE). The edge window "
             "exists only if a book's futures lag the news; that cannot be verified without timestamped futures prices "
             "(start logging them alongside history/odds_*.json).")
    L.append("* Caveats: pre-2025 win totals are an undated late-preseason consensus (source 'not_tracked'); "
             "the model preseason state uses the actual week-1 starter (known in August in almost all cases); ties not simulated; "
             "tiebreakers approximate (common games / strength of victory omitted).")
    (OUT / "season_sim.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
