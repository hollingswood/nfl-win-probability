"""College shop-vs-sharp: WHERE is the validated edge strongest and how to scale it.
PRE-DECLARED 2026-10-07, written before any segment result was computed. Run once.

Background. The cfb_shop rule (cfb_shop_rules.json, research output/research/cfb/round2.md) is already validated:
Arizona soft-book quote vs Pinnacle no-vig at the same snapshot, priced at any number with the key-number
distributions (cfb_dist.json); spreads EV >= 4%, totals and moneylines EV >= 2%, price -200..+200, 1 h - 7 days
before kickoff, one bet per game per market at the first qualifying snapshot (best EV across books/sides there).
This script does NOT re-validate it. It slices it, which is where false discoveries come from, so every slice,
metric and pass rule is fixed here and ALL results are reported, significant or not.

Data / populations (2021-2025 = analysis; 2026 reported separately as out of sample, never used for verdicts)
  BETS  = the frozen-rule bets exactly as the round-2 screen made them (output/research/cfb/shop_bets.parquet,
          rules 'AZ spread EV>=4%', 'AZ total EV>=2%', 'AZ ml EV>=2%'). Moneyline devig = multiplicative (rule v1;
          live v2 uses Shin, which differs negligibly inside -200..+200). A reconstruction from the raw snapshots
          below must reproduce the same bet counts (sanity check, reported).
  OPPS  = every qualifying OPPORTUNITY: each (game, market, snapshot) where at least one Arizona quote meets the
          frozen thresholds, best-EV quote at that snapshot. Same game appears at several snapshots -> all
          standard errors are clustered by game (event_id).
  BOOKQ = per-book population: each book's OWN first qualifying quote per (game, market) (as if it were the only
          book), best side at that snapshot. Same game appears for several books -> clustered by game.
  CAL   = all three markets at EV >= 2% across Arizona books, first qualifying snapshot (round-2 rules
          'AZ spread EV>=2%', 'AZ total EV>=2%', 'AZ ml EV>=2%'), for the EV -> CLV calibration.

Metric. CLV = expected profit per 1 unit staked of the bet (same number and price) under Pinnacle's last
pre-kickoff quote of that market, key-number pricing (identical to shop_screen.grade). Realized ROI and win rate
are reported but are NOT test criteria (too noisy at these sizes). Because the three markets have different CLV
levels, every heterogeneity test is a regression of CLV on market fixed effects + segment dummies with
game-clustered (CR1) standard errors; the test is the Wald chi-square that all segment coefficients are 0.

Primary tests (13). Bonferroni: alpha = 0.05 / 13 = 0.00385.
  T1  S1a Timing, hours before kickoff  (OPPS): >120 h, 72-120, 48-72, 24-48, 1-24 h.  Heterogeneity Wald.
  T2  S1b Timing, snapshot weekday (US Eastern) for SATURDAY (ET) kickoffs only (OPPS): Sun-Mon, Tue-Wed,
          Thu-Fri, Sat.  Heterogeneity Wald.  (Snapshots are 16:10 and 23:10 UTC, i.e. ~noon and ~7 pm ET, plus
          extra Sat/Sun slots; there is NO snapshot in the 9-11 am ET Monday window, so the Pro-Spanky claim can
          only be approximated by the Monday 16:10 UTC slot. Per-slot table and Pinnacle first-posting times are
          reported descriptively.)
  T3  S2a Conference tier (BETS): P4 vs P4, P4 vs G5/independent, G5 vs G5, FCS-or-lower involved.
          P4 = SEC, Big Ten, Big 12, ACC, Pac-12 through 2023, plus Notre Dame. FBS non-P4 = G5/independent.
  T4  S2b Conference game vs non-conference game (BETS; CFBD conferenceGame).
  T5  S3  Season phase (BETS): CFBD regular weeks 1-2 (CFBD labels week-0 games week 1), weeks 3-8,
          weeks 9+ (incl. conference championships), postseason/bowls.
  T6  S4a Book (BOOKQ): 9 Arizona books.  Heterogeneity Wald.
  T7  S4b Your 3 books (DraftKings, FanDuel, ESPN Bet/theScore) vs the 6 Arizona books without an account (BOOKQ).
  T8  S5a Favorite vs underdog (BETS, spreads + moneylines): spread point < 0 / ML price < 0 = favorite;
          point > 0 / price > 0 = dog; pick'em excluded.
  T9  S5b Over vs under (BETS, totals).
  T10 S5c Home vs away (BETS, spreads + moneylines; CFBD neutral-site games excluded).
  T11 S6  EV -> CLV calibration (CAL): slope of CLV on EV at bet with market fixed effects; one-sided test
          slope > 0. The slope is the shrink factor for Kelly sizing (stake on E[CLV | EV], not raw EV).
          EV-bucket table (2-3, 3-4, 4-6, 6-10, >= 10%) reported.
  T12 S7  Persistence (OPPS): for game-markets that qualify at >= 2 snapshots, paired difference
          CLV(first qualifying snapshot) - CLV(last qualifying snapshot); two-sided. Also reported: share of
          game-markets that qualify more than once, P(still qualifies at the next snapshot), CLV of the 2nd+
          opportunities on their own (i.e. is adding at a later snapshot still +CLV).
  T13 S8  Decay (BETS): linear trend of CLV in season 2021-25 with market fixed effects; two-sided.

Verdict per test (mechanical):
  heterogeneity tests T1-T10: "DIFFERS" if Wald p < alpha AND the best level (highest market-adjusted mean CLV,
     pooled) beats the rest in >= 4 of 5 seasons; "suggestive only" if p < 0.05 uncorrected; else
     "no detectable difference" (treat the edge as uniform along this dimension).
  T11: "EV predicts CLV" if slope > 0 at p < alpha.   T12: "first vs later differ" if p < alpha.
  T13: "edge shrinking" if slope < 0 at p < alpha; "growing" if > 0 at p < alpha; else "no detectable trend".
Per level we also report n, mean EV at bet, mean CLV (+- clustered SE, one-sided p > 0), CLV/EV retention,
ROI +- SE and win rate. A level whose own CLV is not > 0 is flagged, but with ~25 cells that is informational.

Scaling estimate (descriptive, labeled as an estimate, NOT a test): bets per season and expected profit
(sum stake x CLV, CLV taken as the expected ROI of each bet) for (a) your 3 books and (b) all 9 Arizona books,
at flat $100 and at quarter-Kelly on a $10k bankroll (Kelly with pushes, capped at 2% as in the rule),
plus a calibrated-Kelly variant that stakes on E[CLV | EV] from T11, with an approximate season SD and
P(losing season) from per-bet variance. Plus a greedy "which book to open next" table on BOOK sets
(MINE + the book that adds the most expected CLV dollars, repeated). Any threshold idea that comes out of this
is a hypothesis for a NEW paper-track version, never an edit of cfb_shop_rules.json.

Outputs: output/research/round3/cfb_shop_segments.json and cfb_shop_segments_tables.md (tables + mechanical
verdicts); the narrative report output/research/round3/cfb_shop_segments.md is written from them afterwards.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research" / "cfb"))
import price_screen as PS  # noqa: E402
import shop_screen as SS  # noqa: E402

OUT = ROOT / "output" / "research" / "round3"
MINE = SS.MINE
AZ = SS.AZ
NOACCT = [b for b in AZ if b not in MINE]
TH = {"spread": 0.04, "total": 0.02, "ml": 0.02}
FROZEN = {"spread": "AZ spread EV>=4%", "total": "AZ total EV>=2%", "ml": "AZ ml EV>=2%"}
MINE_RULES = {"spread": "MINE spread EV>=4%", "total": "MINE total EV>=2%", "ml": "MINE ml EV>=2%"}
CAL_RULES = ["AZ spread EV>=2%", "AZ total EV>=2%", "AZ ml EV>=2%"]
N_TESTS = 13
ALPHA = 0.05 / N_TESTS
DEV = (2021, 2025)
P4_ALWAYS = {"SEC", "Big Ten", "Big 12", "ACC"}
BANKROLL, KF, CAP = 10_000.0, 0.25, 0.02


# ----------------------------------------------------------------------------------------------- statistics
def ols_cl(y, X, g):
    """OLS with CR1 cluster-robust covariance."""
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    sums = pd.DataFrame(X * u[:, None]).groupby(np.asarray(g)).sum().to_numpy()
    G, (n, k) = len(sums), X.shape
    c = G / (G - 1) * (n - 1) / max(1, n - k)
    return b, c * XtXi @ (sums.T @ sums) @ XtXi


def mean_cl(x, g):
    x, g = np.asarray(x, float), np.asarray(g)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan), np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum().to_numpy()
    G = len(s)
    se = math.sqrt((s ** 2).sum() * G / max(1, G - 1)) / n
    return float(m), float(se)


def level_stats(d):
    d = d.dropna(subset=["clv"])
    if not len(d):
        return {"n": 0}
    m, se = mean_cl(d.clv, d.event_id)
    roi, rse = mean_cl(d.pnl, d.event_id)
    dec = (d.res != 0).sum()
    return {"n": int(len(d)), "games": int(d.event_id.nunique()), "ev": round(float(d.ev.mean()), 4),
            "clv": round(m, 4), "se": round(se, 4) if se == se else None,
            "p_gt0": float(norm.sf(m / se)) if se and se == se and se > 0 else None,
            "retention": round(m / float(d.ev.mean()), 2) if d.ev.mean() > 0 else None,
            "roi": round(roi, 4), "roi_se": round(rse, 4) if rse == rse else None,
            "win": round(float((d.res > 0).sum() / dec), 3) if dec else None}


def het_test(d, seg, fe="market"):
    """Wald test that segment coefficients are 0 given market fixed effects; clustered by game."""
    d = d.dropna(subset=["clv", seg])
    levels = sorted(d[seg].unique(), key=str)
    if len(levels) < 2:
        return {"p": None}
    F = pd.get_dummies(d[fe]).to_numpy(float)
    S = pd.get_dummies(d[seg]).reindex(columns=levels).to_numpy(float)[:, 1:]
    X = np.hstack([F, S])
    b, V = ols_cl(d.clv.to_numpy(float), X, d.event_id.to_numpy())
    k = S.shape[1]
    bs, Vs = b[-k:], V[-k:, -k:]
    W = float(bs @ np.linalg.pinv(Vs) @ bs)
    return {"wald": round(W, 3), "df": k, "p": float(chi2.sf(W, k))}


def consistency(d, seg):
    """best level by market-adjusted CLV (pooled) vs the rest, per season 2021-25."""
    d = d.dropna(subset=["clv", seg]).copy()
    d["adj"] = d.clv - d.groupby("market").clv.transform("mean")
    lv = d.groupby(seg).adj.mean()
    best = lv.idxmax()
    wins, tot = 0, 0
    for s, g in d.groupby("season"):
        a, r = g[g[seg] == best].adj, g[g[seg] != best].adj
        if len(a) and len(r):
            tot += 1
            wins += int(a.mean() > r.mean())
    return str(best), wins, tot, {str(k): round(float(v), 4) for k, v in lv.items()}


def segment(d, seg, name, test_id, order=None):
    d = d.dropna(subset=["clv", seg])
    order = order or sorted(d[seg].unique(), key=str)
    lv = {str(k): level_stats(d[d[seg] == k]) for k in order}
    per_market = {mk: {str(k): level_stats(g[g[seg] == k]) for k in order} for mk, g in d.groupby("market")}
    ht = het_test(d, seg)
    best, wins, tot, adj = consistency(d, seg)
    p = ht.get("p")
    if p is not None and p < ALPHA and wins >= 4:
        verdict = "DIFFERS"
    elif p is not None and p < 0.05:
        verdict = "suggestive only (not significant after Bonferroni)" if p >= ALPHA else \
            f"significant but inconsistent across seasons ({wins}/{tot})"
    else:
        verdict = "no detectable difference"
    return {"test": test_id, "name": name, "levels": lv, "by_market": per_market, "het": ht,
            "best_level": best, "best_beats_rest_seasons": f"{wins}/{tot}", "market_adjusted_clv": adj,
            "verdict": verdict}


# ----------------------------------------------------------------------------------------------- data
def game_attrs(O):
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    out = []
    for s, e in ev.groupby("season"):
        g = PS.D.games([s])
        g = g[g.start.notna()]
        m = PS.team_matcher(sorted(set(g.home) | set(g.away)))
        idx = {(r.home, r.away): r for r in g.itertuples()}
        for r in e.itertuples():
            gg = idx.get((m(r.home), m(r.away)))
            if gg is None or abs((gg.start - r.ko).total_seconds()) >= 36 * 3600:
                continue

            def p4(conf, team):
                return conf in P4_ALWAYS or (conf == "Pac-12" and s <= 2023) or team == "Notre Dame"
            hp, ap = p4(gg.home_conf, gg.home), p4(gg.away_conf, gg.away)
            fcs = (gg.home_div != "fbs") or (gg.away_div != "fbs")
            tier = ("FCS/lower involved" if fcs else "P4 vs P4" if hp and ap else
                    "P4 vs G5/Ind" if hp or ap else "G5 vs G5")
            phase = ("4 postseason" if gg.season_type == "postseason" else "1 wk 0-2" if gg.week <= 2 else
                     "2 wk 3-8" if gg.week <= 8 else "3 wk 9+")
            out.append({"event_id": r.event_id, "tier": tier, "conf_game": "conference" if gg.conf_game else "non-conference",
                        "phase": phase, "neutral": bool(gg.neutral), "week": int(gg.week), "season_type": gg.season_type})
    return pd.DataFrame(out).set_index("event_id")


def qualifying_quotes(O, pin, close, R):
    """every Arizona quote meeting the frozen thresholds at any snapshot, graded."""
    parts = []
    for s in sorted(O.season.unique()):
        C = SS.candidates(O[O.season == s], pin[pin.season == s], AZ)
        C = C[C.ev >= C.market.map(TH)]
        parts.append(C)
        print("season", s, "qualifying quotes", len(C), flush=True)
    Q = pd.concat(parts, ignore_index=True)
    return SS.grade(Q.set_index("event_id"), close, R)


def select(Q, books):
    """the frozen rule restricted to a book set: first qualifying snapshot per game-market, best EV there."""
    q = Q[Q.book.isin(books)]
    ft = q.groupby(["event_id", "market"]).t.transform("min")
    q = q[q.t == ft].sort_values("ev", ascending=False)
    return q.groupby(["event_id", "market"]).head(1)


def kelly_stake(d, ev_col="ev"):
    dd = PS.dec(d.price.to_numpy(float))
    b = dd - 1
    pw = d.pwin.to_numpy(float)
    pl = np.clip(pw * b - d.ev.to_numpy(float), 0, 1)           # from EV = pw*b - pl (pushes allowed)
    if ev_col != "ev":                                          # calibrated: keep pw+pl, shift edge to E[CLV|EV]
        e = d[ev_col].to_numpy(float)
        tot = pw + pl
        pw = np.clip((e + tot) / (b + 1), 0, 1)
        pl = np.clip(tot - pw, 0, 1)
    f = (pw * b - pl) / (b * np.maximum(pw + pl, 1e-9))
    return BANKROLL * np.clip(KF * f, 0, CAP)


def scale_row(d, stake, n_seasons):
    d = d.dropna(subset=["clv"])
    stake = stake[: len(d)] if np.ndim(stake) else np.full(len(d), stake)
    dd = PS.dec(d.price.to_numpy(float))
    pw = d.pwin.to_numpy(float)
    pl = np.clip(pw * (dd - 1) - d.ev.to_numpy(float), 0, 1)
    exp = stake * d.clv.to_numpy()
    var = stake ** 2 * (pw * (dd - 1) ** 2 + pl) - (stake * d.ev.to_numpy()) ** 2
    E, SD = exp.sum() / n_seasons, math.sqrt(max(var.sum(), 0) / n_seasons)
    return {"bets_per_season": round(len(d) / n_seasons, 1), "turnover_per_season": round(float(stake.sum()) / n_seasons),
            "avg_stake": round(float(stake.mean()), 1), "exp_profit_per_season": round(E),
            "season_sd": round(SD), "p_losing_season": round(float(norm.cdf(-E / SD)), 3) if SD > 0 else None,
            "realized_profit_per_season": round(float((stake * d.pnl.to_numpy()).sum()) / n_seasons)}


# ----------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    O = PS.load("cfb")
    pin = pd.read_parquet(ROOT / "output" / "research" / "cfb" / "pin_mus.parquet")
    close = pin.sort_values("t").groupby("event_id")[["mu_m", "mu_t", "q_ml"]].agg(
        lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    close.columns = ["c_mu_m", "c_mu_t", "c_q_ml"]
    R = SS.results_full(O)
    A = game_attrs(O)
    print("events", O.event_id.nunique(), "with results", len(R), "with CFBD attrs", len(A), flush=True)
    res = {"declared": "2026-10-07, before results", "alpha_bonferroni": ALPHA, "n_tests": N_TESTS}

    SB = pd.read_parquet(ROOT / "output" / "research" / "cfb" / "shop_bets.parquet")
    BETS = pd.concat([SB[SB.rule == r] for r in FROZEN.values()]).join(A, on="event_id")
    CAL = pd.concat([SB[SB.rule == r] for r in CAL_RULES])
    MB = pd.concat([SB[SB.rule == r] for r in MINE_RULES.values()])
    Q = qualifying_quotes(O, pin, close, R).join(A, on="event_id")
    Q["hb"] = (Q.ko - Q.t).dt.total_seconds() / 3600
    REC = select(Q, AZ)
    res["sanity_reconstruction"] = {mk: {"parquet": int((BETS.market == mk).sum()), "rebuilt": int((REC.market == mk).sum())}
                                    for mk in TH}
    print("sanity", res["sanity_reconstruction"], flush=True)
    dev = lambda d: d[d.season.between(*DEV)]  # noqa: E731
    Bd = dev(BETS)
    res["overall"] = {"dev_2021_25": {mk: level_stats(g) for mk, g in Bd.groupby("market")} | {"all": level_stats(Bd)},
                      "oos_2026": {mk: level_stats(g) for mk, g in BETS[BETS.season == 2026].groupby("market")}
                      | {"all": level_stats(BETS[BETS.season == 2026])}}
    tests = {}

    # ---------------- S1 timing (OPPS)
    OPPS = Q.sort_values("ev", ascending=False).groupby(["event_id", "market", "t"]).head(1).copy()
    OPPS["hbucket"] = pd.cut(OPPS.hb, [0, 24, 48, 72, 120, 1e9], labels=["5 1-24h", "4 24-48h", "3 48-72h", "2 72-120h", "1 >120h"]).astype(str)
    et_t = OPPS.t.dt.tz_convert("America/New_York")
    et_k = OPPS.ko.dt.tz_convert("America/New_York")
    wd = et_t.dt.weekday
    OPPS["wdgroup"] = np.select([wd.isin([6, 0]), wd.isin([1, 2]), wd.isin([3, 4]), wd == 5],
                                ["1 Sun-Mon", "2 Tue-Wed", "3 Thu-Fri", "4 Sat"], "?")
    OPPS["sat_ko"] = et_k.dt.weekday == 5
    OPPS["slot"] = OPPS.t.dt.strftime("%a %H:%M UTC")
    Od = dev(OPPS)
    tests["T1"] = segment(Od, "hbucket", "S1a hours before kickoff (all qualifying opportunities)", "T1")
    tests["T2"] = segment(Od[Od.sat_ko], "wdgroup", "S1b snapshot weekday ET, Saturday kickoffs (all opportunities)", "T2")
    # descriptive: slot table (opportunities + where the frozen rule's first bets come from)
    REC = REC.copy()
    REC["slot"] = REC.t.dt.strftime("%a %H:%M UTC")
    REC["hbucket"] = pd.cut(REC.hb, [0, 24, 48, 72, 120, 1e9], labels=["5 1-24h", "4 24-48h", "3 48-72h", "2 72-120h", "1 >120h"]).astype(str)
    slot_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    slots = sorted(Od.slot.unique(), key=lambda s: (slot_order.index(s[:3]), s))
    res["slots"] = {s: {"opps": level_stats(Od[Od.slot == s]),
                        "first_bets": int((dev(REC).slot == s).sum()),
                        "first_bets_clv": level_stats(dev(REC)[dev(REC).slot == s]).get("clv")} for s in slots}
    res["first_bets_by_hbucket"] = {k: level_stats(g) for k, g in dev(REC).groupby("hbucket")}
    # when does Pinnacle first post (Saturday ET kickoffs)?
    pf = pin[pin.season.between(*DEV)].groupby("event_id").agg(t=("t", "min"), ko=("ko", "first"))
    pf = pf[pf.ko.dt.tz_convert("America/New_York").dt.weekday == 5]
    pfs = pf.t.dt.strftime("%a %H:%M UTC").value_counts()
    res["pinnacle_first_post_sat_games"] = {k: int(v) for k, v in pfs.items()}
    res["pinnacle_first_post_hours_before_ko"] = {q: round(float(((pf.ko - pf.t).dt.total_seconds() / 3600).quantile(q)), 1)
                                                  for q in (0.1, 0.25, 0.5, 0.75, 0.9)}

    # ---------------- S2 / S3 / S5 / S8 on BETS
    tests["T3"] = segment(Bd, "tier", "S2a conference tier", "T3", ["P4 vs P4", "P4 vs G5/Ind", "G5 vs G5", "FCS/lower involved"])
    tests["T4"] = segment(Bd, "conf_game", "S2b conference vs non-conference game", "T4")
    res["T4_fbs_only_info"] = {k: level_stats(g) for k, g in Bd[Bd.tier != "FCS/lower involved"].groupby("conf_game")}
    tests["T5"] = segment(Bd, "phase", "S3 season phase", "T5")
    sm = Bd[Bd.market.isin(["spread", "ml"])].copy()
    sm["favdog"] = np.where(sm.market == "spread", np.where(sm.point < 0, "favorite", np.where(sm.point > 0, "underdog", None)),
                            np.where(sm.price < 0, "favorite", np.where(sm.price > 0, "underdog", None)))
    tests["T8"] = segment(sm, "favdog", "S5a favorite vs underdog (spreads + ML)", "T8")
    tests["T9"] = segment(Bd[Bd.market == "total"], "side", "S5b over vs under (totals)", "T9")
    tests["T10"] = segment(sm[sm.neutral == False], "side", "S5c home vs away (spreads + ML, no neutral sites)", "T10")  # noqa: E712

    # ---------------- S4 books (BOOKQ)
    ft = Q.groupby(["event_id", "market", "book"]).t.transform("min")
    BOOKQ = Q[Q.t == ft].sort_values("ev", ascending=False).groupby(["event_id", "market", "book"]).head(1).copy()
    BOOKQ["acct"] = np.where(BOOKQ.book.isin(MINE), "1 your 3 books", "2 no-account AZ books")
    BOOKQ["stale"] = (BOOKQ.t - pd.to_datetime(BOOKQ.last_update, utc=True)) > pd.Timedelta(hours=3)
    Kd = dev(BOOKQ)
    tests["T6"] = segment(Kd, "book", "S4a book (each book's own first qualifying quote)", "T6", AZ)
    tests["T7"] = segment(Kd, "acct", "S4b your 3 books vs no-account Arizona books", "T7")
    res["book_share_of_frozen_bets"] = {b: int(v) for b, v in dev(REC).book.value_counts().items()}
    res["book_stale_share"] = {b: round(float(g.stale.mean()), 3) for b, g in Kd.groupby("book")}

    # ---------------- S6 calibration (CAL)
    Cd = dev(CAL).dropna(subset=["clv"])
    F = pd.get_dummies(Cd.market).to_numpy(float)
    X = np.hstack([F, Cd[["ev"]].to_numpy(float)])
    b, V = ols_cl(Cd.clv.to_numpy(float), X, Cd.event_id.to_numpy())
    slope, sse = float(b[-1]), float(math.sqrt(V[-1, -1]))
    by_mk = {}
    for mk, g in Cd.groupby("market"):
        Xm = np.column_stack([np.ones(len(g)), g.ev.to_numpy(float)])
        bm, Vm = ols_cl(g.clv.to_numpy(float), Xm, g.event_id.to_numpy())
        by_mk[mk] = {"intercept": round(float(bm[0]), 4), "slope": round(float(bm[1]), 3), "slope_se": round(float(math.sqrt(Vm[1, 1])), 3)}
    Cd = Cd.assign(evb=pd.cut(Cd.ev, [0.02, 0.03, 0.04, 0.06, 0.10, 10], right=False,
                              labels=["2-3%", "3-4%", "4-6%", "6-10%", ">=10%"]).astype(str))
    p11 = float(norm.sf(slope / sse))
    tests["T11"] = {"test": "T11", "name": "S6 EV -> CLV calibration (EV>=2%, all markets, AZ)", "slope": round(slope, 3),
                    "slope_se": round(sse, 3), "p_slope_gt0": p11, "intercepts_by_market": {k: round(float(v), 4) for k, v in zip(pd.get_dummies(Cd.market).columns, b[:-1])},
                    "by_market": by_mk,
                    "buckets": {k: level_stats(g) for k, g in Cd.groupby("evb")},
                    "buckets_by_market": {mk: {k: level_stats(gg) for k, gg in g.groupby("evb")} for mk, g in Cd.groupby("market")},
                    "verdict": "EV predicts CLV" if p11 < ALPHA else "EV does not detectably predict CLV"}

    # ---------------- S7 persistence (OPPS)
    o = Od.sort_values("t")
    o["k"] = o.groupby(["event_id", "market"]).cumcount()
    o["nq"] = o.groupby(["event_id", "market"]).t.transform("size")
    multi = o[o.nq >= 2]
    fst = multi[multi.k == 0].set_index(["event_id", "market"])
    lst = multi.groupby(["event_id", "market"]).tail(1).set_index(["event_id", "market"])
    pair = fst[["clv", "ev", "season"]].join(lst[["clv", "ev"]], rsuffix="_last").dropna()
    dif = pair.clv - pair.clv_last
    m7, se7 = mean_cl(dif, pair.index.get_level_values(0))
    p12 = float(2 * norm.sf(abs(m7 / se7)))
    # persistence to the next Pinnacle snapshot in window
    snaps = pin[pin.season.between(*DEV)][["event_id", "t", "ko"]].drop_duplicates()
    snaps = snaps[((snaps.ko - snaps.t) >= pd.Timedelta(hours=1)) & ((snaps.ko - snaps.t) <= pd.Timedelta(days=7))].sort_values("t")
    snaps["t_next"] = snaps.groupby("event_id").t.shift(-1)
    qq = o.merge(snaps[["event_id", "t", "t_next"]], on=["event_id", "t"], how="left")
    qset = set(zip(o.event_id, o.market, o.t))
    has_next = qq.t_next.notna()
    qq["persists"] = [((e, m, tn) in qset) if isinstance(tn, pd.Timestamp) else np.nan
                      for e, m, tn in zip(qq.event_id, qq.market, qq.t_next)]
    gap = (qq.t_next - qq.t).dt.total_seconds() / 3600
    best_ev = o.sort_values("ev", ascending=False).groupby(["event_id", "market"]).head(1)
    tests["T12"] = {"test": "T12", "name": "S7 persistence: first vs last qualifying snapshot (paired)",
                    "game_markets": int(o.groupby(["event_id", "market"]).ngroups),
                    "share_qualifying_2plus": round(float((o.groupby(["event_id", "market"]).nq.first() >= 2).mean()), 3),
                    "median_qualifying_snapshots": float(o.groupby(["event_id", "market"]).nq.first().median()),
                    "pairs": int(len(pair)), "clv_first": round(float(pair.clv.mean()), 4), "clv_last": round(float(pair.clv_last.mean()), 4),
                    "diff_first_minus_last": round(m7, 4), "diff_se": round(se7, 4), "p_two_sided": p12,
                    "p_persist_next_snapshot": round(float(qq.loc[has_next, "persists"].astype(float).mean()), 3),
                    "p_persist_by_market": {mk: round(float(g.loc[g.t_next.notna(), "persists"].astype(float).mean()), 3) for mk, g in qq.groupby("market")},
                    "median_gap_to_next_snapshot_h": round(float(gap.median()), 1),
                    "by_occurrence": {("1st" if k == 0 else "2nd" if k == 1 else "3rd+"): level_stats(g)
                                      for k, g in o.assign(kk=o.k.clip(upper=2)).groupby("kk")},
                    "repeat_opps_2nd_plus": level_stats(o[o.k >= 1]),
                    "max_ev_snapshot": level_stats(best_ev),
                    "first_snapshot_all": level_stats(o[o.k == 0]),
                    "verdict": ("first vs later differ" if p12 < ALPHA else "no detectable first-vs-later difference")}

    # ---------------- S8 decay (BETS)
    F = pd.get_dummies(Bd.dropna(subset=["clv"]).market).to_numpy(float)
    bb = Bd.dropna(subset=["clv"])
    X = np.hstack([F, (bb.season.to_numpy(float) - 2023)[:, None]])
    b8, V8 = ols_cl(bb.clv.to_numpy(float), X, bb.event_id.to_numpy())
    s8, se8 = float(b8[-1]), float(math.sqrt(V8[-1, -1]))
    p13 = float(2 * norm.sf(abs(s8 / se8)))
    tests["T13"] = {"test": "T13", "name": "S8 decay: CLV trend per season 2021-25", "slope_per_season": round(s8, 4),
                    "se": round(se8, 4), "p_two_sided": p13,
                    "by_season": {int(s): level_stats(g) for s, g in BETS.groupby("season")},
                    "by_season_market": {mk: {int(s): level_stats(gg) for s, gg in g.groupby("season")} for mk, g in BETS.groupby("market")},
                    "opps_by_season": {int(s): level_stats(g) for s, g in OPPS.groupby("season")},
                    "verdict": ("edge shrinking" if s8 < 0 and p13 < ALPHA else "edge growing" if s8 > 0 and p13 < ALPHA
                                else "no detectable trend")}
    res["tests"] = tests

    # ---------------- scaling estimate (descriptive)
    cal_line = {mk: (v["intercept"], v["slope"]) for mk, v in by_mk.items()}

    def calib(d):
        return np.array([max(0.0, cal_line[m][0] + cal_line[m][1] * e) for m, e in zip(d.market, d.ev)])
    n_s = DEV[1] - DEV[0] + 1
    scale = {}
    for name, d in (("your 3 books", dev(MB)), ("all 9 Arizona books", Bd)):
        d = d.dropna(subset=["clv"]).copy()
        d["ev_cal"] = calib(d)
        scale[name] = {"flat_100": scale_row(d, 100.0, n_s), "quarter_kelly_10k": scale_row(d, kelly_stake(d), n_s),
                       "calibrated_quarter_kelly_10k": scale_row(d, kelly_stake(d, "ev_cal"), n_s),
                       "by_market_flat_100": {mk: scale_row(g, 100.0, n_s) for mk, g in d.groupby("market")}}
    res["scale"] = scale
    # greedy book additions from your 3 books (flat $100 expected profit)
    cur, order = list(MINE), []
    base = dev(select(Q, cur)).dropna(subset=["clv"])
    order.append({"set": "your 3 books", "bets_per_season": round(len(base) / n_s, 1),
                  "exp_profit_flat100": round(100 * base.clv.sum() / n_s)})
    remaining = list(NOACCT)
    while remaining:
        best = None
        for bk in remaining:
            d = dev(select(Q, cur + [bk])).dropna(subset=["clv"])
            val = 100 * d.clv.sum() / n_s
            if best is None or val > best[1]:
                best = (bk, val, len(d))
        cur.append(best[0]); remaining.remove(best[0])
        order.append({"set": "+ " + best[0], "bets_per_season": round(best[2] / n_s, 1), "exp_profit_flat100": round(best[1])})
    res["greedy_books"] = order
    single = {}
    for bk in AZ:
        d = dev(select(Q, [bk])).dropna(subset=["clv"])
        single[bk] = {"bets_per_season": round(len(d) / n_s, 1), "exp_profit_flat100": round(100 * d.clv.sum() / n_s),
                      "clv": round(float(d.clv.mean()), 4)}
    res["single_book_alone"] = single

    (OUT / "cfb_shop_segments.json").write_text(json.dumps(res, indent=1, default=str))
    write_tables(res)
    print(json.dumps({k: {kk: v.get(kk) for kk in ("het", "verdict", "best_level", "best_beats_rest_seasons", "slope", "p_slope_gt0",
                                                     "diff_first_minus_last", "p_two_sided", "slope_per_season")} for k, v in tests.items()},
                     indent=1, default=str))


def fmt_p(p):
    return "-" if p is None else f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def pct(x, nd=1):
    return "-" if x is None or x != x else f"{100 * x:+.{nd}f}%"


def level_table(levels):
    rows = ["| level | n | games | EV at bet | CLV ± SE | p(CLV>0) | retention | ROI ± SE | win |", "|---|---|---|---|---|---|---|---|---|"]
    for k, s in levels.items():
        if not s.get("n"):
            rows.append(f"| {k} | 0 | | | | | | | |"); continue
        rows.append(f"| {k} | {s['n']} | {s['games']} | {pct(s['ev'])} | {pct(s['clv'])} ± {100 * (s['se'] or 0):.1f} | {fmt_p(s['p_gt0'])} | "
                    f"{s['retention']} | {pct(s['roi'])} ± {100 * (s['roi_se'] or 0):.1f} | {s['win']} |")
    return "\n".join(rows)


def write_tables(res):
    L = ["# cfb_shop segments: tables (generated by scripts/research/round3/cfb_shop_segments.py)", "",
         f"Bonferroni alpha = 0.05/{res['n_tests']} = {res['alpha_bonferroni']:.5f}. Analysis seasons 2021-25; 2026 out of sample.", "",
         f"Sanity (frozen-rule bets, parquet vs rebuilt): {res['sanity_reconstruction']}", "", "## Overall (frozen rule)", "",
         "2021-25", "", level_table(res["overall"]["dev_2021_25"]), "", "2026 (out of sample)", "", level_table(res["overall"]["oos_2026"]), ""]
    for k, t in res["tests"].items():
        L += [f"## {k}: {t['name']}", "", f"**Verdict: {t['verdict']}**", ""]
        if "levels" in t:
            L += [f"Wald {t['het'].get('wald')} (df {t['het'].get('df')}), p = {fmt_p(t['het'].get('p'))}; best level `{t['best_level']}` "
                  f"beats the rest in {t['best_beats_rest_seasons']} seasons. Market-adjusted CLV: {t['market_adjusted_clv']}", "",
                  level_table(t["levels"]), ""]
            for mk, lv in t["by_market"].items():
                L += [f"<details><summary>{mk}</summary>", "", level_table(lv), "", "</details>", ""]
        elif k == "T11":
            L += [f"slope {t['slope']} ± {t['slope_se']} (CLV per unit EV), p = {fmt_p(t['p_slope_gt0'])}; intercepts {t['intercepts_by_market']}; "
                  f"per market {t['by_market']}", "", level_table(t["buckets"]), ""]
            for mk, lv in t["buckets_by_market"].items():
                L += [f"{mk}", "", level_table(lv), ""]
        elif k == "T12":
            L += [", ".join(f"{kk}: {vv}" for kk, vv in t.items() if not isinstance(vv, dict) and kk not in ("test", "name", "verdict")), "",
                  level_table(t["by_occurrence"]), "", "repeat (2nd+) opportunities / max-EV snapshot / first snapshot:", "",
                  level_table({"2nd+": t["repeat_opps_2nd_plus"], "max EV": t["max_ev_snapshot"], "first": t["first_snapshot_all"]}), ""]
        elif k == "T13":
            L += [f"slope {pct(t['slope_per_season'], 2)} per season ± {100 * t['se']:.2f}, p = {fmt_p(t['p_two_sided'])}", "",
                  level_table({str(s): v for s, v in t["by_season"].items()}), ""]
            for mk, lv in t["by_season_market"].items():
                L += [f"{mk}", "", level_table({str(s): v for s, v in lv.items()}), ""]
            L += ["all opportunities by season", "", level_table({str(s): v for s, v in t["opps_by_season"].items()}), ""]
    L += ["## Snapshot slots (opportunities, 2021-25)", "", "| slot | opps | opp CLV | frozen first bets | their CLV |", "|---|---|---|---|---|"]
    for s, v in res["slots"].items():
        L.append(f"| {s} | {v['opps'].get('n')} | {pct(v['opps'].get('clv'))} | {v['first_bets']} | {pct(v['first_bets_clv'])} |")
    L += ["", f"Frozen first bets by hours bucket:", "", level_table(res["first_bets_by_hbucket"]), "",
          f"Pinnacle first post (Saturday games): {res['pinnacle_first_post_sat_games']}",
          f"hours before kickoff quantiles: {res['pinnacle_first_post_hours_before_ko']}", "",
          "## Scaling estimate (expected profit = sum stake x CLV; estimate)", ""]
    for name, s in res["scale"].items():
        L += [f"### {name}", "", "| sizing | bets/season | avg stake | turnover | expected profit/season | season SD | P(losing season) | realized/season (2021-25) |",
              "|---|---|---|---|---|---|---|---|"]
        for z in ("flat_100", "quarter_kelly_10k", "calibrated_quarter_kelly_10k"):
            r = s[z]
            L.append(f"| {z} | {r['bets_per_season']} | ${r['avg_stake']} | ${r['turnover_per_season']} | ${r['exp_profit_per_season']} | "
                     f"${r['season_sd']} | {r['p_losing_season']} | ${r['realized_profit_per_season']} |")
        L += ["", "by market (flat $100): " + json.dumps({mk: (v["bets_per_season"], v["exp_profit_per_season"]) for mk, v in s["by_market_flat_100"].items()}), ""]
    L += ["### Greedy book additions (flat $100)", "", "| set | bets/season | expected profit/season |", "|---|---|---|"]
    L += [f"| {r['set']} | {r['bets_per_season']} | ${r['exp_profit_flat100']} |" for r in res["greedy_books"]]
    L += ["", "### Each book alone", "", "| book | bets/season | CLV | expected profit/season |", "|---|---|---|---|"]
    L += [f"| {b} | {v['bets_per_season']} | {pct(v['clv'])} | ${v['exp_profit_flat100']} |" for b, v in res["single_book_alone"].items()]
    L += ["", f"Book share of frozen bets: {res['book_share_of_frozen_bets']}", f"Stale (>3 h) share per book: {res['book_stale_share']}"]
    (OUT / "cfb_shop_segments_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
