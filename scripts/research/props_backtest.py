"""RECEIVING-YARDS PROPS vs THE MARKET: line accuracy, information test, and an early-snapshot betting backtest.

Stages (cache in scratch; holdout runs once, after output/research/props_frozen.json exists):
  python scripts/research/props_backtest.py prep      # match props -> games/players, model distributions, consensus
  python scripts/research/props_backtest.py dev       # 2023 only: info test fit, rule grid (prints, writes dev json)
  python scripts/research/props_backtest.py holdout   # 2024-25 ONCE with the frozen rules -> props_backtest.json

Data: data/historical_odds/props/player_reception_yds_{2023,2024,2025}.csv.gz (Odds API, regions=us), two snapshots per
event: "early" (Fri 21:40 UTC for Sunday games, kickoff-24h otherwise) and "close" (kickoff-75 min).
Model: scripts/research/props.py walk-forward GBM mean (train seasons < S) + conditional-empirical distribution
(400 nearest out-of-sample (mu, y) pairs from the previous 4 seasons, yards rescaled). "early" mode = teammate
availability from the final injury report (Out/Doubtful) only; "late" = actual actives.
Optimism in the early model (not removable here): recorded weather, nflverse CLOSING spread/total as features.

Market conventions:
  per-book no-vig P(over) = multiplicative (1/o)/(1/o+1/u);
  per-book location shift c_b = L_b - Q_S(1 - p_b): how far the model distribution S must be shifted so that it
  prices that book's line at the book's no-vig probability. Consensus shift c = median_b c_b (all us books).
  Market-implied distribution = S + c; blend(w) = S + (1-w)c (w=1 model, w=0 market).
  Consensus line L* = modal point across books (ties -> the one nearest the median point); p* = median no-vig P(over)
  among books at L*.
CLV of a bet at point L, side s, decimal d: P_close(s at L) * d - 1 where P_close is the median no-vig probability of
the close books quoting exactly L if any ("direct"), else the model distribution shifted to the close consensus
("shift"). Bets on players who did not play are void (no CLV, no PnL).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
import props as PR  # noqa: E402

RAW = ROOT / "data" / "raw"
PROPS = ROOT / "data" / "historical_odds" / "props"
OUT = ROOT / "output" / "research"
SCR = Path("/tmp/claude-0/-home-claude-nfl-win-probability/9625cec3-ac82-57cc-b374-feea441e3f9c/scratchpad/propsbt")
PSCR = PR.SCR
SEASONS = (2023, 2024, 2025)
DEVS, HOLDS = (2023,), (2024, 2025)
ALLOWED = set(json.load(open(ROOT / "my_books.json"))["allowed_books"])
NN = PR.NN
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


# ------------------------------------------------------------------------------------------------ utils
def dec(am):
    am = np.asarray(am, float)
    return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))


def norm_name(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", str(s))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " ").replace(".", "").replace("'", ""))
    toks = [t for t in s.split() if t not in SUFFIX]
    s = " ".join(toks)
    return ALIAS.get(s, s)


ALIAS = {"hollywood brown": "marquise brown", "amon ra stbrown": "amon ra st brown"}


def first_ok(a: str, b: str) -> bool:
    """Fuzzy guard: first names must be close (Eli/Elijah, Drew/Andrew) - rejects Deonte vs Damien Harris."""
    fa, fb = (a.split() or [""])[0], (b.split() or [""])[0]
    return fa in fb or fb in fa or SequenceMatcher(None, fa, fb).ratio() >= 0.6


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------- prep
def load_props() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(PROPS / f"player_reception_yds_{s}.csv.gz").assign(season=s) for s in SEASONS],
                  ignore_index=True)
    d = d.dropna(subset=["over_price", "under_price", "point"])
    # main-line quotes only: Kambi books (betrivers/unibet) post ~25-33% of their 2023-24 rec-yds lines as
    # rounded "milestone" points at skewed prices (e.g. 29.5 at +450 when the market line is 13.5). Those price the
    # tails and break consensus/CLV conventions, so both sides must lie in [-200, +170].
    main = d.over_price.between(-200, 170) & d.under_price.between(-200, 170)
    print("dropped non-main quotes:", int((~main).sum()), "of", len(d))
    d = d[main].copy()
    d["ct"] = pd.to_datetime(d.commence_time, utc=True)
    d["rt"] = pd.to_datetime(d.requested_ts, utc=True)
    d["hours_before"] = (d.ct - d.rt).dt.total_seconds() / 3600
    d["snap"] = np.where(d.hours_before < 3, "close", "early")
    d["dec_o"], d["dec_u"] = dec(d.over_price), dec(d.under_price)
    io, iu = 1 / d.dec_o, 1 / d.dec_u
    d["p_nv"] = io / (io + iu)
    d["hold"] = io + iu - 1
    return d


def map_games(d: pd.DataFrame) -> pd.DataFrame:
    g = pd.read_parquet(RAW / "games.parquet")
    g = g[g.season.isin(SEASONS)][["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team"]]
    g["gameday"] = pd.to_datetime(g.gameday).dt.date
    ev = d[["event_id", "home", "away", "ct"]].drop_duplicates("event_id").copy()
    ev["gameday"] = ev.ct.dt.tz_convert("America/New_York").dt.date
    m = ev.merge(g, left_on=["home", "away", "gameday"], right_on=["home_team", "away_team", "gameday"], how="left")
    return m[["event_id", "game_id", "week", "game_type"]]


def match_players(d: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """(event,player name) -> player_id among those who PLAYED in the game; else a global name lookup marks
    a known player who did not play (void)."""
    pl = pd.read_parquet(RAW / "players.parquet")
    pl = pl[pl.position.isin(["WR", "TE", "RB", "FB", "QB"]) | pl.position.isna()]
    names = {}
    for c in ["display_name", "football_name", "common_first_name", "first_name"]:
        if c in ("display_name",):
            v = pl[c]
        else:
            v = pl[c].fillna("") + " " + pl.last_name.fillna("")
        for pid, nm in zip(pl.gsis_id, v):
            if isinstance(nm, str) and nm.strip():
                names.setdefault(norm_name(nm), set()).add(pid)
    pid_names = {}
    for nm, ids in names.items():
        for i in ids:
            pid_names.setdefault(i, set()).add(nm)
    played = pg.groupby("game_id").player_id.apply(set).to_dict()
    keys = d[["game_id", "player"]].drop_duplicates()
    res = []
    for gid, raw in zip(keys.game_id, keys.player):
        n = norm_name(raw)
        cand = played.get(gid, set())
        ids = names.get(n, set())
        hit = ids & cand
        how = "exact"
        if len(hit) != 1:
            # first-initial + last name, or fuzzy, among players who played
            sc = []
            for p in cand:
                best = max((SequenceMatcher(None, n, x).ratio() for x in pid_names.get(p, {""})
                            if first_ok(n, x)), default=0)
                sc.append((best, p))
            sc.sort(reverse=True)
            if sc and sc[0][0] >= 0.85 and (len(sc) == 1 or sc[1][0] < sc[0][0] - 0.05):
                hit, how = {sc[0][1]}, "fuzzy"
            else:
                toks = n.split()
                lastm = [p for p in cand if any(x.split()[-1:] == toks[-1:] and x[:1] == n[:1]
                                                for x in pid_names.get(p, set()))] \
                    if toks and len(toks[0]) == 1 else []
                if len(lastm) == 1:
                    hit, how = {lastm[0]}, "initial_last"
                else:
                    hit = set()
        if len(hit) == 1:
            res.append((gid, raw, next(iter(hit)), "played", how))
        elif len(ids) >= 1:
            res.append((gid, raw, None, "did_not_play", "known_name"))
        else:
            res.append((gid, raw, None, "unmatched", ""))
    return pd.DataFrame(res, columns=["game_id", "player", "player_id", "status", "how"])


def model_samples(mode: str) -> tuple[pd.DataFrame, np.ndarray]:
    P = pd.read_parquet(PSCR / "preds.parquet")
    P = P[(P.market == "rec_yds") & (P["mode"] == mode)]
    rows, smps = [], []
    for S in SEASONS:
        tr = P[(P.season >= S - PR.LOOKBACK) & (P.season < S) & P.in_universe].dropna(subset=["y", "gbm"])
        te = P[(P.season == S)].dropna(subset=["gbm"])
        smp = np.sort(PR.cond_samples(tr.gbm.to_numpy(float), tr.y.to_numpy(float), te.gbm.to_numpy(float),
                                      "yards"), axis=1)
        rows.append(te[["game_id", "player_id", "gbm", "in_universe"]])
        smps.append(smp)
    return pd.concat(rows, ignore_index=True), np.concatenate(smps).astype(np.float32)


def shift_for(S: np.ndarray, L: np.ndarray, p_over: np.ndarray) -> np.ndarray:
    """c such that mean(S + c > L) = p_over, per row (S sorted, rows aligned)."""
    q = np.array([np.quantile(s, 1 - p) for s, p in zip(S, np.clip(p_over, 0.0025, 0.9975))])
    return L - q


def p_over_at(S: np.ndarray, shift: np.ndarray, L: np.ndarray) -> np.ndarray:
    return ((S + shift[:, None]) > L[:, None]).mean(axis=1)


def consensus(q: pd.DataFrame) -> pd.DataFrame:
    """Per (key, snap): modal point L*, median no-vig p at L*, median shift c (needs column c), n books, med hold."""
    out = []
    for (k, sn), G in q.groupby(["key", "snap"], sort=False):
        vc = G.point.value_counts()
        top = vc[vc == vc.max()].index.to_numpy()
        L = top[np.argmin(np.abs(top - G.point.median()))]
        out.append((k, sn, L, G.loc[G.point == L, "p_nv"].median(), G.c.median() if "c" in G else np.nan,
                    G.point.median(), len(G), G.hold.median()))
    return pd.DataFrame(out, columns=["key", "snap", "L", "p", "c", "pt_med", "nbooks", "hold_med"])


def prep():
    SCR.mkdir(parents=True, exist_ok=True)
    d = load_props()
    gm = map_games(d)
    d = d.merge(gm, on="event_id", how="left")
    pg = pd.read_parquet(PSCR / "player_games.parquet", columns=["game_id", "team", "player_id", "season", "rec_yds",
                                                                 "position"])
    pg = pg[pg.season.isin(SEASONS)]
    mp = match_players(d[d.game_id.notna()], pg)
    d = d.merge(mp, on=["game_id", "player"], how="left")
    d["status"] = d.status.fillna("no_game")
    d = d.merge(pg[["game_id", "player_id", "rec_yds", "team", "position"]], on=["game_id", "player_id"], how="left")
    d["key"] = d.game_id.astype(str) + "|" + d.player_id.astype(str)
    # model distributions (early and late)
    for mode in ["early", "late"]:
        rows, smp = model_samples(mode)
        rows["key"] = rows.game_id + "|" + rows.player_id
        rows = rows.drop_duplicates("key")
        np.save(SCR / f"samples_{mode}.npy", smp[rows.index.to_numpy()])
        rows.reset_index(drop=True)[["key", "gbm", "in_universe"]].to_parquet(SCR / f"rows_{mode}.parquet")
    d.drop(columns=["ct", "rt"]).to_parquet(SCR / "props_matched.parquet")
    st = d.drop_duplicates(["event_id", "player"]).groupby(["season", "status"]).size().unstack(fill_value=0)
    print(st)
    print(d.drop_duplicates(["event_id", "player"]).how.value_counts())
    print("events without game:", d[d.game_id.isna()].event_id.nunique())


# --------------------------------------------------------------------------------------- analysis table
def table() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Quotes for played+modelled player-games, with model probabilities, shifts and consensus."""
    d = pd.read_parquet(SCR / "props_matched.parquet")
    d = d[d.status == "played"].copy()
    S, R = {}, {}
    for mode in ["early", "late"]:
        R[mode] = pd.read_parquet(SCR / f"rows_{mode}.parquet")
        S[mode] = np.load(SCR / f"samples_{mode}.npy")
    idx = pd.Series(np.arange(len(R["early"])), index=R["early"].key)
    d["si"] = d.key.map(idx)
    d["has_model"] = d.si.notna()
    q = d[d.has_model].copy()
    q["si"] = q.si.astype(int)
    Se = S["early"][q.si.to_numpy()]
    q["c"] = shift_for(Se, q.point.to_numpy(float), q.p_nv.to_numpy(float))
    cons = consensus(q)
    consall = consensus(d.assign(c=np.nan))
    return d, q, {"S": S, "R": R, "cons": cons, "consall": consall}


def build_pg(d: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """One row per played player-game with early and close consensus and model quantities."""
    ca = ctx["consall"].pivot(index="key", columns="snap", values=["L", "p", "pt_med", "nbooks", "hold_med"])
    ca.columns = [f"{a}_{b}" for a, b in ca.columns]
    cm = ctx["cons"].pivot(index="key", columns="snap", values="c")
    cm.columns = [f"c_{b}" for b in cm.columns]
    base = d.drop_duplicates("key")[["key", "game_id", "player_id", "season", "week", "game_type", "rec_yds",
                                     "player", "has_model", "position"]].set_index("key")
    pgt = base.join(ca).join(cm).reset_index()
    for mode in ["early", "late"]:
        r = ctx["R"][mode]
        ii = pd.Series(np.arange(len(r)), index=r.key)
        pgt[f"si_{mode}"] = pgt.key.map(ii)
    return pgt


def model_cols(pgt: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    for mode in ["early", "late"]:
        m = pgt[f"si_{mode}"].notna()
        Sx = ctx["S"][mode][pgt.loc[m, f"si_{mode}"].astype(int).to_numpy()]
        pgt.loc[m, f"med_{mode}"] = np.median(Sx, axis=1)
        pgt.loc[m, f"gbm_{mode}"] = ctx["R"][mode].gbm.to_numpy()[pgt.loc[m, f"si_{mode}"].astype(int)]
        for sn in ["early", "close"]:
            ok = m & pgt[f"L_{sn}"].notna()
            Sy = ctx["S"][mode][pgt.loc[ok, f"si_{mode}"].astype(int).to_numpy()]
            pgt.loc[ok, f"pm_{mode}_at_{sn}"] = (Sy > pgt.loc[ok, f"L_{sn}"].to_numpy()[:, None]).mean(axis=1)
    # market-implied median (price-adjusted) = early-model median + consensus shift
    for sn in ["early", "close"]:
        pgt[f"mkt_med_{sn}"] = pgt.med_early + pgt[f"c_{sn}"]
    return pgt


def accuracy(pgt: pd.DataFrame, seasons) -> dict:
    out = {}
    D = pgt[pgt.season.isin(seasons)]
    y = D.rec_yds
    for sn in ["early", "close"]:
        M = D[D[f"L_{sn}"].notna()]
        r = {"n_all": int(len(M)), "line_mae_all": round(float((M.rec_yds - M[f"L_{sn}"]).abs().mean()), 3)}
        Mm = M[M.med_early.notna() & M.med_late.notna()]
        e = lambda c: round(float((Mm.rec_yds - Mm[c]).abs().mean()), 3)
        r.update({"n_both": int(len(Mm)), "line_mae": e(f"L_{sn}"), "mkt_price_adj_median_mae": e(f"mkt_med_{sn}"),
                  "model_early_median_mae": e("med_early"), "model_late_median_mae": e("med_late"),
                  "line_minus_model_early_mean": round(float((Mm[f"L_{sn}"] - Mm.med_early).mean()), 2),
                  "corr_line_model": round(float(np.corrcoef(Mm[f"L_{sn}"], Mm.med_early)[0, 1]), 3),
                  "over_rate_at_line": round(float((Mm.rec_yds > Mm[f"L_{sn}"]).mean()), 4),
                  "mean_nv_p_over": round(float(Mm[f"p_{sn}"].mean()), 4),
                  "median_hold": round(float(Mm[f"hold_med_{sn}"].median()), 4),
                  "books_per_player": round(float(Mm[f"nbooks_{sn}"].mean()), 2)})
        # half-blend median
        r["half_blend_median_mae"] = round(float((Mm.rec_yds - 0.5 * (Mm[f"L_{sn}"] + Mm.med_early)).abs().mean()), 3)
        out[sn] = r
    return out


def info_fit(pgt: pd.DataFrame, sn: str, mode: str, fit_seasons, test_seasons) -> dict:
    """Logistic over-outcome on logit(market no-vig p) [+ logit(model p)] at the consensus line."""
    from sklearn.linear_model import LogisticRegression
    col = f"pm_{mode}_at_{sn}"
    D = pgt[pgt[f"L_{sn}"].notna() & pgt[col].notna() & pgt[f"p_{sn}"].notna()].copy()
    D = D[D.rec_yds != D[f"L_{sn}"]]
    D["o"] = (D.rec_yds > D[f"L_{sn}"]).astype(int)
    D["xm"], D["xo"] = logit(D[f"p_{sn}"].to_numpy()), logit(D[col].to_numpy())
    tr, te = D[D.season.isin(fit_seasons)], D[D.season.isin(test_seasons)]
    big = 1e6
    m1 = LogisticRegression(C=big).fit(tr[["xm"]], tr.o)
    m2 = LogisticRegression(C=big).fit(tr[["xm", "xo"]], tr.o)
    m0 = LogisticRegression(C=big).fit(tr[["xo"]], tr.o)

    def ll(p, o):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(o * np.log(p) + (1 - o) * np.log(1 - p))

    res = {"n_fit": int(len(tr)), "n_test": int(len(te)),
           "coef_market_only": [round(float(m1.intercept_[0]), 4), round(float(m1.coef_[0][0]), 4)],
           "coef_market_plus_model": [round(float(m2.intercept_[0]), 4)] + [round(float(x), 4) for x in m2.coef_[0]]}
    if len(te):
        o = te.o.to_numpy()
        l_raw = ll(te[f"p_{sn}"].to_numpy(), o)
        l1 = ll(m1.predict_proba(te[["xm"]])[:, 1], o)
        l2 = ll(m2.predict_proba(te[["xm", "xo"]])[:, 1], o)
        l0 = ll(m0.predict_proba(te[["xo"]])[:, 1], o)
        g = te.game_id.to_numpy()
        res.update({"logloss_market_raw": round(float(l_raw.mean()), 5), "logloss_market_recal": round(float(l1.mean()), 5),
                    "logloss_market_plus_model": round(float(l2.mean()), 5),
                    "logloss_model_only_recal": round(float(l0.mean()), 5),
                    "gain_plus_model_vs_market_recal": boot_gain(l1 - l2, g),
                    "gain_model_only_vs_market_recal": boot_gain(l1 - l0, g),
                    "over_rate": round(float(o.mean()), 4)})
    # in-sample clustered SE of model coefficient (on the fit seasons)
    res["fit_model_coef_cluster_se"] = cluster_se_logit(tr, ["xm", "xo"])
    return res


def boot_gain(x: np.ndarray, g: np.ndarray, B: int = 2000) -> dict:
    u, gi = np.unique(g, return_inverse=True)
    s = np.bincount(gi, weights=x)
    n = np.bincount(gi)
    rng = np.random.default_rng(11)
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(u), len(u))
        bs.append(s[k].sum() / n[k].sum())
    return {"mean": round(float(x.mean()), 5), "ci95": [round(float(np.percentile(bs, 2.5)), 5),
                                                         round(float(np.percentile(bs, 97.5)), 5)],
            "p_le_0": round(float(np.mean(np.array(bs) <= 0)), 4)}


def cluster_se_logit(D: pd.DataFrame, xs: list[str]) -> dict:
    """Unpenalised logit with game-clustered sandwich SEs: [coef, se]."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e6, max_iter=1000).fit(D[xs], D.o)
    X = np.c_[np.ones(len(D)), D[xs].to_numpy()]
    beta = np.r_[m.intercept_, m.coef_[0]]
    p = 1 / (1 + np.exp(-X @ beta))
    H = (X * (p * (1 - p))[:, None]).T @ X
    sc = X * (D.o.to_numpy() - p)[:, None]
    gi = pd.factorize(D.game_id)[0]
    G = np.zeros((gi.max() + 1, X.shape[1]))
    np.add.at(G, gi, sc)
    Hi = np.linalg.inv(H)
    V = Hi @ (G.T @ G) @ Hi
    se = np.sqrt(np.diag(V))
    return {n: [round(float(b), 4), round(float(s), 4)] for n, b, s in zip(["const"] + xs, beta, se)}


# ------------------------------------------------------------------------------------------------- bets
K_OFF = 0.55  # off-point valuation slope, 2023 main-line close quotes: vs outcomes 0.58 [0.21,0.94], vs books 0.54 [0.50,0.58]


def candidate_bets(d: pd.DataFrame, q: pd.DataFrame, pgt: pd.DataFrame, ctx: dict, w: float,
                   k: float = K_OFF) -> pd.DataFrame:
    """Every allowed-book early quote with blend-w probabilities, EV both sides, close valuation.
    Close valuation at the bet point L: 'direct' = median close no-vig P(over) of close books quoting L; else
    'anchored' = p*_close + k * (P_{S+c_close}(>L) - P_{S+c_close}(>L*_close)), i.e. the consensus probability at the
    modal close line moved along the shifted model distribution, damped by k (2023 dev: a pure shift, k=1,
    overstates off-point value). Sensitivity columns use k=0.3 and k=1.0."""
    E = q[(q.snap == "early") & q.book.isin(ALLOWED)].copy()
    E = E.merge(pgt[["key", "c_early", "c_close", "L_close", "p_close"]], on="key", how="left")
    E["off_close"] = E.point - E.L_close
    Se = ctx["S"]["early"][E.si.to_numpy()]
    sh = (1 - w) * E.c_early.to_numpy(float)
    L = E.point.to_numpy(float)
    X = Se + sh[:, None]
    E["p_over"] = (X > L[:, None]).mean(axis=1)
    E["p_push"] = (np.abs(X - L[:, None]) < 0.5).mean(axis=1) * (L % 1 == 0)
    E["p_under"] = 1 - E.p_over - E.p_push
    E["ev_over"] = E.p_over * E.dec_o + E.p_push - 1
    E["ev_under"] = E.p_under * E.dec_u + E.p_push - 1
    # close valuation: direct (same point, median no-vig of close books) else shift
    C = q[q.snap == "close"].groupby(["key", "point"]).p_nv.median().rename("p_close_direct").reset_index()
    E = E.merge(C, on=["key", "point"], how="left")
    cc = E.c_close.to_numpy(float)
    ok = ~np.isnan(cc)
    pL, pLs = np.full(len(E), np.nan), np.full(len(E), np.nan)
    pL[ok] = ((Se[ok] + cc[ok, None]) > L[ok, None]).mean(axis=1)
    pLs[ok] = ((Se[ok] + cc[ok, None]) > E.L_close.to_numpy(float)[ok, None]).mean(axis=1)
    E["p_close_shift"] = pL
    for lab, kk in [("", k), ("_k03", 0.3), ("_k10", 1.0)]:
        anc = np.clip(E.p_close.to_numpy(float) + kk * (pL - pLs), 0.005, 0.995)
        E[f"p_close_over{lab}"] = E.p_close_direct.fillna(pd.Series(anc, index=E.index))
    E["clv_src"] = np.where(E.p_close_direct.notna(), "direct", np.where(ok, "anchored", "none"))
    return E


def select(E: pd.DataFrame, thr: float, side: str = "both", ev_max: float = 9.0) -> pd.DataFrame:
    """Best EV side/book per player-game; keep if thr <= EV < ev_max."""
    pc = ["p_close_over", "p_close_over_k03", "p_close_over_k10"]
    cols = ["key", "game_id", "season", "week", "book", "point", "clv_src", "rec_yds", "c_early", "p_nv",
            "off_close"] + pc
    o = E[cols + ["dec_o", "p_over", "ev_over"]].rename(columns={"dec_o": "dec", "p_over": "p", "ev_over": "ev"})
    o["side"] = "over"
    u = E[cols + ["dec_u", "p_under", "ev_under"]].rename(columns={"dec_u": "dec", "p_under": "p", "ev_under": "ev"})
    u["side"] = "under"
    for c in pc + ["p_nv"]:
        u[c] = 1 - u[c]
    b = pd.concat([o, u], ignore_index=True)
    if side != "both":
        b = b[b.side == side]
    b = b.sort_values("ev", ascending=False).drop_duplicates("key")
    b = b[(b.ev >= thr) & (b.ev < ev_max)].copy()
    b["clv"] = b.p_close_over * b.dec - 1
    b["clv_k03"] = b.p_close_over_k03 * b.dec - 1
    b["clv_k10"] = b.p_close_over_k10 * b.dec - 1
    b["early_nv_ev"] = b.p_nv * b.dec - 1  # value at the bet book's own early no-vig (shopping benchmark)
    win = np.where(b.side == "over", b.rec_yds > b.point, b.rec_yds < b.point)
    push = b.rec_yds == b.point
    b["pnl"] = np.where(push, 0.0, np.where(win, b.dec - 1, -1.0))
    b["win"] = win.astype(float)
    return b


def summarize(b: pd.DataFrame) -> dict:
    if len(b) == 0:
        return {"n": 0}
    g = b.game_id.to_numpy()

    def cl(x):
        x = np.asarray(x, float)
        m = ~np.isnan(x)
        x, gg = x[m], g[m]
        u, gi = np.unique(gg, return_inverse=True)
        s = np.bincount(gi, weights=x - x.mean())
        se = float(np.sqrt((s ** 2).sum()) / len(x)) if len(x) else np.nan
        return float(x.mean()), se, int(len(x))

    from scipy.stats import norm
    cm, cse, cn = cl(b.clv)
    rm, rse, _ = cl(b.pnl)
    weeks = b.groupby(["season", "week"]).ngroups
    return {"n": int(len(b)), "n_clv": cn, "clv_mean": round(cm, 4), "clv_se": round(cse, 4),
            "clv_t": round(cm / cse, 2) if cse else None, "clv_p_one_sided": float(1 - norm.cdf(cm / cse)) if cse else None,
            "clv_direct_share": round(float((b.clv_src == "direct").mean()), 3),
            "clv_direct_only": round(float(b.loc[b.clv_src == "direct", "clv"].mean()), 4),
            "clv_anchored_only": round(float(b.loc[b.clv_src == "anchored", "clv"].mean()), 4),
            "clv_k03": round(float(b.clv_k03.mean()), 4), "clv_k10_pure_shift": round(float(b.clv_k10.mean()), 4),
            "close_move_toward_bet": round(float((b.p_close_over - b.p_nv).mean()), 4),
            "beat_close_rate": round(float((b.clv > 0).mean()), 3),
            "roi": round(rm, 4), "roi_se": round(rse, 4), "hit_rate": round(float(b.win.mean()), 4),
            "mean_ev_model": round(float(b.ev.mean()), 4), "mean_dec": round(float(b.dec.mean()), 3),
            "bets_per_week": round(len(b) / max(weeks, 1), 1), "weeks": int(weeks),
            "under_share": round(float((b.side == "under").mean()), 3),
            "by_side": {s: {"n": int(len(x)), "clv": round(float(x.clv.mean()), 4), "roi": round(float(x.pnl.mean()), 4)}
                        for s, x in b.groupby("side")},
            "by_book": {k: {"n": int(len(x)), "clv": round(float(x.clv.mean()), 4), "roi": round(float(x.pnl.mean()), 4)}
                        for k, x in b.groupby("book")}}


def best_book_stats(q: pd.DataFrame, seasons) -> dict:
    """How often each allowed book has the best over / under price at the modal point (early), and holds."""
    E = q[(q.snap == "early") & q.book.isin(ALLOWED) & q.season.isin(seasons)]
    res = {"hold_by_book": E.groupby("book").hold.median().round(4).to_dict(),
           "quotes_by_book": E.groupby("book").size().to_dict()}
    # best number for each side: over wants lowest point then best price; score by no-vig-equivalent EV vs consensus
    E = E.copy()
    best_o = E.sort_values(["key", "point", "dec_o"], ascending=[True, True, False]).drop_duplicates("key")
    best_u = E.sort_values(["key", "point", "dec_u"], ascending=[True, False, False]).drop_duplicates("key")
    res["best_over_book_share"] = best_o.book.value_counts(normalize=True).round(3).to_dict()
    res["best_under_book_share"] = best_u.book.value_counts(normalize=True).round(3).to_dict()
    return res


# --------------------------------------------------------------------------------------------- stages
GRID_W = (1.0, 0.5, 0.25, 0.1, 0.0)
GRID_T = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def dev():
    d, q, ctx = table()
    pgt = model_cols(build_pg(d, ctx), ctx)
    pgt.to_parquet(SCR / "pg_table.parquet")
    res = {"accuracy_2023": accuracy(pgt, DEVS)}
    res["info_2023_insample"] = {f"{sn}_{mode}": info_fit(pgt, sn, mode, DEVS, ()) for sn in ["early", "close"]
                                 for mode in ["early", "late"]}
    grid = {}
    for w in GRID_W:
        E = candidate_bets(d, q, pgt, ctx, w)
        E = E[E.season.isin(DEVS)]
        for side in ["both", "under", "over"]:
            for t in GRID_T:
                s = summarize(select(E, t, side))
                grid[f"w{w}_t{t}_{side}"] = {k: s.get(k) for k in ["n", "clv_mean", "clv_t", "roi", "roi_se",
                                                                   "hit_rate", "bets_per_week", "under_share",
                                                                   "mean_ev_model", "clv_direct_share",
                                                                   "clv_direct_only", "clv_anchored_only",
                                                                   "clv_k03", "clv_k10_pure_shift",
                                                                   "close_move_toward_bet"]}
    res["grid_2023"] = grid
    res["books_2023"] = best_book_stats(q, DEVS)
    json.dump(res, open(SCR / "dev.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "grid_2023"}, indent=1, default=float))
    for k, v in grid.items():
        print(k, v)


def holdout():
    fz = json.load(open(OUT / "props_frozen.json"))
    d, q, ctx = table()
    pgt = model_cols(build_pg(d, ctx), ctx)
    dmp = pd.read_parquet(SCR / "props_matched.parquet")
    u = dmp.drop_duplicates(["event_id", "player"])
    res = {"definitions": __doc__, "frozen": fz,
           "match": {str(s): u[u.season == s].status.value_counts().to_dict() for s in SEASONS},
           "match_method": u[u.status == "played"].how.value_counts().to_dict(),
           "model_coverage_of_played": {str(s): round(float(pgt[pgt.season == s].has_model.mean()), 4)
                                        for s in SEASONS},
           "events": {str(s): int(dmp[dmp.season == s].event_id.nunique()) for s in SEASONS},
           "books_present": {str(s): sorted(dmp[dmp.season == s].book.unique().tolist()) for s in SEASONS},
           "accuracy": {"2023": accuracy(pgt, DEVS), "2024_25": accuracy(pgt, HOLDS)},
           "info_test": {f"{sn}_{mode}": info_fit(pgt, sn, mode, DEVS, HOLDS) for sn in ["early", "close"]
                         for mode in ["early", "late"]},
           "books": {"2023": best_book_stats(q, DEVS), "2024_25": best_book_stats(q, HOLDS)},
           "dev_rules": {}, "holdout_rules": {}}
    for r in fz["rules"]:
        E = candidate_bets(d, q, pgt, ctx, r["w"])
        for lab, ss in [("dev_rules", DEVS), ("holdout_rules", HOLDS)]:
            b = select(E[E.season.isin(ss)], r["thr"], r["side"], r.get("ev_max", 9.0))
            s = summarize(b)
            s["by_season"] = {str(k): {kk: summarize(x)[kk] for kk in ["n", "clv_mean", "clv_t", "roi", "roi_se",
                                                                      "hit_rate"]} for k, x in b.groupby("season")}
            s["pass"] = bool(s.get("clv_p_one_sided") is not None and s["clv_mean"] > 0
                             and s["clv_p_one_sided"] < 0.05 / 3) if lab == "holdout_rules" else None
            res[lab][r["name"]] = s
            if lab == "holdout_rules":
                b.to_csv(SCR / f"bets_{r['name']}.csv", index=False)
    json.dump(res, open(OUT / "props_backtest.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "definitions"}, indent=1, default=float))


if __name__ == "__main__":
    SCR.mkdir(parents=True, exist_ok=True)
    {"prep": prep, "dev": dev, "holdout": holdout}[sys.argv[1]]()
