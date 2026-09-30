"""Does the market misprice quarterback changes -- at the close, or early in the week?

Research script (not a model input). Discipline:
  * 2012-2019 = long closing-line ATS sample, 2020-2022 = development (early-week snapshots exist).
  * 2023-2025 is held out. `--holdout` refuses to run until output/research/qb_timing_frozen.json exists,
    and it evaluates ONLY the frozen rules, once.
  * Early-week bets are graded by price-based CLV against edge_lab.closing_fair() (not nflverse spread_line).

Run (dev):      PYTHONPATH=src:scripts python scripts/research/qb_timing.py
Run (holdout):  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES PYTHONPATH=src:scripts python scripts/research/qb_timing.py --holdout
Cache for the expensive QB-rating/pbp step: $QBT_CACHE (default: system temp dir).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nflpred import features as F  # noqa: E402
from nflpred import margins as K, spread_bets as SB  # noqa: E402
from nflpred.weather import _kickoff_utc  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "research"
FROZEN = OUT / "qb_timing_frozen.json"
CACHE = Path(os.environ.get("QBT_CACHE", tempfile.gettempdir()))
TM = F.TEAM_MAP
LONG, DEV, HOLD = range(2012, 2020), range(2020, 2023), range(2023, 2026)
BE = 110 / 210  # 52.38% break-even at -110
N_RULES = 3
ALPHA = 0.05 / N_RULES
DB_PER_GAME = 38  # rough team dropbacks/game: converts an EPA/dropback QB gap into points


# ============================================================================ team-game table
def _qbr() -> tuple[pd.DataFrame, pd.DataFrame]:
    """QB ratings (post-game, EPA/dropback shrunk, same as the model) and per-game dropback shares."""
    p1, p2 = CACHE / "qbt_qbr.parquet", CACHE / "qbt_share.parquet"
    if p1.exists() and p2.exists():
        return pd.read_parquet(p1), pd.read_parquet(p2)
    frames = []
    for s in range(2012, 2026):
        x = pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=["game_id", "posteam", "qb_dropback", "qb_epa", "id"])
        frames.append(x[x.qb_dropback.eq(1)])
    pbp = pd.concat(frames, ignore_index=True)
    g = pd.read_parquet(RAW / "games.parquet", columns=["game_id", "gameday"])
    g["gameday"] = pd.to_datetime(g.gameday)
    qbr = F.qb_ratings(pbp, g)
    pbp["posteam"] = pbp.posteam.replace(TM)
    d = pbp[pbp.id.notna()].groupby(["game_id", "posteam", "id"]).size().rename("db").reset_index()
    d["share"] = d.db / d.groupby(["game_id", "posteam"]).db.transform("sum")
    d = d.rename(columns={"posteam": "team", "id": "qb_id"})
    qbr.to_parquet(p1)
    d.to_parquet(p2)
    return qbr, d


def team_games(max_season: int) -> pd.DataFrame:
    """One row per team per game (1999+ for history; analysis uses 2012+), with QB-change sub-types."""
    g = pd.read_parquet(RAW / "games.parquet")
    g = g[(g.season <= max_season) & g.home_score.notna()].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TM)
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        sgn = 1 if side == "home" else -1
        x = pd.DataFrame({
            "game_id": g.game_id, "season": g.season, "week": g.week, "game_type": g.game_type,
            "gameday": g.gameday, "gametime": g.gametime, "team": g[f"{side}_team"], "opp": g[f"{opp}_team"],
            "side": side, "qb_id": g[f"{side}_qb_id"], "qb_name": g[f"{side}_qb_name"],
            "opp_qb_id": g[f"{opp}_qb_id"],
            "margin": sgn * (g.home_score - g.away_score), "exp_margin": sgn * g.spread_line,
            "spread_line": g.spread_line,
        })
        rows.append(x)
    t = pd.concat(rows, ignore_index=True).sort_values(["team", "gameday", "game_id"]).reset_index(drop=True)
    t["resid"] = t.margin - t.exp_margin               # ATS result vs the nflverse close (+ = covered)
    t["cover"] = np.where(t.resid > 0, 1.0, np.where(t.resid < 0, 0.0, np.nan))
    grp = t.groupby("team")
    t["prev_qb"] = grp.qb_id.shift(1)
    t["prev_season"] = grp.season.shift(1)
    t["prev_game_id"] = grp.game_id.shift(1)
    t["prev_gameday"] = grp.gameday.shift(1)
    t["prev_gametime"] = grp.gametime.shift(1)
    # history of this franchise's starters (previous 17 games), career starts, established starter
    hist17, stint, est, was_est = [], [], [], []
    seen_career: set = set()
    t_sorted_idx = t.sort_values(["gameday", "game_id"]).index
    career_first = pd.Series(False, index=t.index)
    for i in t_sorted_idx:
        q = t.at[i, "qb_id"]
        career_first.at[i] = q not in seen_career   # a QB can't start twice on one day, so ties are harmless
        seen_career.add(q)
    for team, d in t.groupby("team", sort=False):
        qbs, seas = d.qb_id.tolist(), d.season.tolist()
        team_est: list = []
        for k, q in enumerate(qbs):
            prev = qbs[max(0, k - 17):k]
            hist17.append(q in prev)
            # games since this QB last started for the franchise (within 17)
            n = 0
            for pq_ in reversed(prev):
                if pq_ == q:
                    break
                n += 1
            stint.append(n if q in prev else np.nan)
            # established starter = most starts among this season's earlier games (tie -> the earlier one)
            cur = [x for x, y in zip(qbs[:k], seas[:k]) if y == seas[k]]
            est_k = max(cur, key=lambda x: (cur.count(x), -cur.index(x))) if cur else None
            est.append(est_k)
            team_est.append(est_k)
            # returning QB was the established starter when the replacement stint began
            was_est.append(bool(q in prev and n >= 1 and team_est[k - n] == q))
    t["started_for_team_last17"] = hist17
    t["games_missed"] = stint
    t["established_qb"] = est
    t["returning_established"] = was_est
    t["career_first_start"] = career_first
    pl = pd.read_parquet(RAW / "players.parquet", columns=["gsis_id", "rookie_season", "draft_year"])
    t = t.merge(pl.rename(columns={"gsis_id": "qb_id"}), on="qb_id", how="left")
    t["rookie"] = (t.rookie_season == t.season) | (t.draft_year == t.season)
    t = t[t.season >= 2012].copy()
    # ratings (pre-game, strictly earlier games) for today's starter S and last game's starter P
    qbr, share = _qbr()
    for who, col in (("S", "qb_id"), ("P", "prev_qb"), ("O", "opp_qb_id")):
        k = t[["gameday", col]].reset_index().rename(columns={col: "qb_id"}).dropna().sort_values("gameday")
        m = pd.merge_asof(k, qbr, on="gameday", by="qb_id", allow_exact_matches=False)
        t[f"r_{who}"] = m.set_index("index").qb_rating.reindex(t.index).fillna(F.QB_PRIOR)
    t["gap"] = t.r_S - t.r_P                            # EPA/dropback: new starter minus last game's starter
    t["gap_pts"] = t.gap * DB_PER_GAME
    # previous game: did last week's starter P finish it?
    sh = share.rename(columns={"game_id": "prev_game_id", "qb_id": "prev_qb", "share": "prev_share_P"})
    t = t.merge(sh[["prev_game_id", "team", "prev_qb", "prev_share_P"]], on=["prev_game_id", "team", "prev_qb"], how="left")
    t["prev_share_P"] = t.prev_share_P.fillna(0.0)
    # injury report (final weekly row) for P and for S this week
    inj = pd.concat([pd.read_parquet(RAW / f"injuries_{s}.parquet") for s in range(2012, max_season + 1)
                     if (RAW / f"injuries_{s}.parquet").exists()])
    inj = inj[inj.position == "QB"].copy()
    inj["team"] = inj.team.replace(TM)
    inj["week"] = inj.week.astype(int)
    inj["season"] = inj.season.astype(int)
    inj = inj.sort_values("date_modified").drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    ij = inj[["season", "week", "team", "gsis_id", "report_status", "practice_status", "date_modified"]]
    t = t.merge(ij.rename(columns={"gsis_id": "prev_qb", "report_status": "P_status", "practice_status": "P_practice",
                                   "date_modified": "P_modified"}), on=["season", "week", "team", "prev_qb"], how="left")
    t = t.merge(ij.rename(columns={"gsis_id": "qb_id", "report_status": "S_status", "practice_status": "S_practice",
                                   "date_modified": "S_modified"}), on=["season", "week", "team", "qb_id"], how="left")
    t = t.drop_duplicates(["game_id", "team"])
    # ---- classification
    same = t.prev_season == t.season
    chg = same & (t.qb_id != t.prev_qb) & t.qb_id.notna() & t.prev_qb.notna()
    t["qb_change"] = chg
    listed = t.P_status.isin(["Out", "Doubtful", "Questionable"])
    left = t.prev_share_P < 0.5
    ret = chg & t.started_for_team_last17 & t.returning_established      # the usual starter comes back
    back = chg & t.started_for_team_last17 & ~t.returning_established    # an earlier backup gets another turn
    new = chg & ~t.started_for_team_last17
    t["subtype"] = np.select(
        [ret & (t.games_missed == 1), ret & (t.games_missed >= 2), back,
         new & listed, new & ~listed & left, new & ~listed & ~left,
         same & ~chg & (t.qb_id != t.established_qb)],
        ["return_after_1", "return_after_2plus", "other_qb_back_in", "new_injury_listed", "new_prev_left_early_unlisted",
         "new_not_listed", "backup_continuing"], default="none")
    t["kind"] = np.select([ret, new, back, t.subtype == "backup_continuing"],
                          ["return", "new", "other_change", "backup_cont"], "none")
    t["rookie_debut"] = new & t.career_first_start & t.rookie
    t["first_career_start"] = new & t.career_first_start
    # opponent's change status (to handle games where both teams switch)
    o = t[["game_id", "team", "kind", "gap"]].rename(columns={"team": "opp", "kind": "opp_kind", "gap": "opp_gap"})
    t = t.merge(o, on=["game_id", "opp"], how="left")
    t["kick"] = pd.to_datetime([_kickoff_utc(a, b) for a, b in zip(t.gameday, t.gametime)], utc=True)
    t["prev_kick"] = pd.to_datetime([_kickoff_utc(a, b) if pd.notna(a) else pd.NaT
                                     for a, b in zip(t.prev_gameday, t.prev_gametime)], utc=True)
    return t


# ============================================================================ statistics
def cover_stats(d: pd.DataFrame) -> dict:
    c = d.cover.dropna()
    n = len(c)
    if n == 0:
        return {"n": 0}
    p = c.mean()
    r = d.resid.dropna()
    return {"n": int(n), "cover": round(float(p), 4), "se": round(float(math.sqrt(p * (1 - p) / n)), 4),
            "mean_resid": round(float(r.mean()), 2), "resid_se": round(float(r.std(ddof=1) / math.sqrt(len(r))), 2)
            if len(r) > 1 else None,
            "p_two_sided_vs_50": round(float(st.binomtest(int(c.sum()), n, 0.5).pvalue), 4)}


def binom_rule(d: pd.DataFrame) -> dict:
    c = d.cover.dropna()
    n, k = len(c), int(c.sum())
    if n == 0:
        return {"n": 0, "pass": False}
    p = float(st.binomtest(k, n, BE, alternative="greater").pvalue)
    return {"n": n, "wins": k, "cover": round(k / n, 4), "se": round(math.sqrt((k / n) * (1 - k / n) / n), 4),
            "roi_at_110": round((k * 100 / 110 - (n - k)) / n, 4), "p_one_sided_vs_524": round(p, 5),
            "pass": bool(k / n > BE and p < ALPHA)}


def ols(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    X = np.c_[np.ones_like(x), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - 2)
    # HC1 robust
    meat = (X * e[:, None] ** 2).T @ X
    bread = np.linalg.inv(X.T @ X)
    rob = bread @ meat @ bread * len(y) / (len(y) - 2)
    return {"n": int(len(y)), "intercept": round(float(b[0]), 3), "slope": round(float(b[1]), 3),
            "slope_se": round(float(math.sqrt(rob[1, 1])), 3), "slope_t": round(float(b[1] / math.sqrt(rob[1, 1])), 2),
            "ols_se": round(float(math.sqrt(cov[1, 1])), 3)}


def clv_stats(b: pd.DataFrame, col: str = "clv") -> dict:
    v = b[col].dropna().values
    n = len(v)
    if n < 3:
        return {"n": int(n), "pass": False}
    se = v.std(ddof=1) / math.sqrt(n)
    tt = v.mean() / se if se > 0 else 0.0
    p = float(st.t.sf(tt, n - 1))
    pnl = b.pnl.values if "pnl" in b else np.array([np.nan])
    return {"n": int(n), "mean_clv": round(float(v.mean()), 4), "clv_se": round(float(se), 4), "t": round(float(tt), 2),
            "p_one_sided": round(p, 5), "beat_close": round(float((v > 0).mean()), 3),
            "roi": round(float(np.nanmean(pnl)), 4), "pass": bool(v.mean() > 0 and p < ALPHA)}


# ============================================================================ close-based analysis
def close_analysis(t: pd.DataFrame, seasons) -> dict:
    d = t[t.season.isin(seasons)]
    out = {"all_teams_baseline": cover_stats(d)}
    # one row per team-game; drop games where the opponent is in the same bucket (they offset exactly)
    for k in ("new", "return", "other_change", "backup_cont"):
        x = d[(d.kind == k) & (d.opp_kind != k)]
        out[f"kind:{k}"] = cover_stats(x)
    for s in ("new_injury_listed", "new_prev_left_early_unlisted", "new_not_listed", "return_after_1",
              "return_after_2plus", "other_qb_back_in", "backup_continuing"):
        x = d[(d.subtype == s) & ~((d.opp_kind == d.kind))]
        out[f"subtype:{s}"] = cover_stats(x)
    x = d[d.rookie_debut & (d.opp_kind != "new")]
    out["rookie_debut"] = cover_stats(x)
    x = d[d.first_career_start & ~d.rookie_debut & (d.opp_kind != "new")]
    out["first_career_start_non_rookie"] = cover_stats(x)
    # QB-rating gap bins (new starter vs last week's)
    n = d[(d.kind == "new") & (d.opp_kind != "new")]
    out["new_by_gap"] = {lab: cover_stats(n[m]) for lab, m in (
        ("big_drop(gap<-0.15)", n.gap < -0.15), ("mid_drop(-0.15..-0.05)", n.gap.between(-0.15, -0.05)),
        ("small(|gap|<0.05)", n.gap.between(-0.05, 0.05, inclusive="neither")), ("upgrade(gap>0.05)", n.gap > 0.05))}
    r = d[(d.kind == "return") & (d.opp_kind != "return")]
    out["return_by_gap"] = {lab: cover_stats(r[m]) for lab, m in (
        ("big_upgrade(gap>0.15)", r.gap > 0.15), ("mid_upgrade(0.05..0.15)", r.gap.between(0.05, 0.15)),
        ("small(|gap|<=0.05)", r.gap.abs() <= 0.05), ("downgrade(gap<-0.05)", r.gap < -0.05))}
    # home rows only (each game once): home ATS residual vs net QB gap (home change minus away change), in points.
    # slope > 0 => market under-reacts to QB changes (the side that upgraded keeps covering); < 0 => over-reacts.
    h = d[d.side == "home"].assign(net_pts=lambda x: (x.gap - x.opp_gap) * DB_PER_GAME)
    chg = h.kind.isin(["new", "return", "other_change"]) | h.opp_kind.isin(["new", "return", "other_change"])
    out["home_resid_on_net_gap_pts_change_games"] = ols(h[chg].net_pts, h[chg].resid)
    out["home_resid_on_net_gap_pts_all_games"] = ols(h.net_pts, h.resid)
    return out


# ============================================================================ early-week snapshots
def _ev(mu_home, side, point, price, sig, w):
    mu_home, point, price = map(lambda a: np.asarray(a, float), (mu_home, point, price))
    home_line = np.where(side == "home", point, -point)
    hc, pu, ac = K.cover_probs(np.nan_to_num(mu_home), sig, home_line, w)
    pw = np.where(side == "home", hc, ac)
    dec = np.where(price > 0, 1 + price / 100, 1 + 100 / -price)
    ev = pw * dec + pu - 1
    return np.where(np.isfinite(mu_home), ev, np.nan)


def snapshots(holdout: bool) -> pd.DataFrame:
    import edge_lab as E
    t = E.load(holdout)
    t = t[t.point.notna() & t.sp_price.between(-200, 200)].copy()
    side_cons_pt = np.where(t.side == "home", -t.m_cons, t.m_cons)
    t = t[(t.point - side_cons_pt).abs() <= 2.5]           # drop stale/off-market lines (as edge_lab._prep)
    seasons = HOLD if holdout else DEV
    cf = E.closing_fair(list(seasons))[["game_id", "mu_close_all", "mu_close_sharp"]]
    return t.merge(cf, on="game_id", how="left")


def early_analysis(tg: pd.DataFrame, holdout: bool) -> tuple[dict, dict[str, pd.DataFrame]]:
    r = SB.load_rules()
    sig, w = r["margin"]["sigma"], r["_weights"]
    snap = snapshots(holdout)
    seasons = HOLD if holdout else DEV
    tg = tg[tg.season.isin(seasons)]
    # per team-game info at bet time
    home = tg[tg.side == "home"].set_index("game_id")
    away = tg[tg.side == "away"].set_index("game_id")
    # earliest moment both teams' previous games are over (+4h) -> "early week" first bettable snapshot
    ready = pd.concat([home.prev_kick, away.prev_kick], axis=1).max(axis=1) + pd.Timedelta(hours=4)
    snap["ready"] = snap.game_id.map(ready)
    snap = snap[snap.ready.isna() | (snap.requested_ts >= snap.ready)]
    snap = snap[snap.hours_before <= 9 * 24]
    cons = snap.drop_duplicates(["game_id", "requested_ts"]).sort_values("requested_ts")
    g_early = cons.groupby("game_id").first()[["requested_ts", "m_cons", "hours_before"]].rename(
        columns={"requested_ts": "ts_early", "m_cons": "m_early", "hours_before": "hb_early"})
    g_last = cons.groupby("game_id").last()[["m_cons"]].rename(columns={"m_cons": "m_last"})
    gm = g_early.join(g_last).join(cons.groupby("game_id")[["mu_close_all", "mu_close_sharp"]].first())
    # team-level move toward/away from each team (+ = market moved toward THIS team between early and close)
    tt = tg.merge(gm.reset_index(), on="game_id", how="inner")
    sgn = np.where(tt.side == "home", 1, -1)
    tt["move_to_team"] = sgn * (tt.mu_close_all - tt.m_early)
    # info timing of the QB news
    tt["info"] = np.select(
        [tt.prev_share_P < 0.5, tt.P_status == "Out", tt.P_status.isin(["Doubtful", "Questionable"])],
        ["prev_starter_left_prev_game", "prev_starter_listed_Out", "prev_starter_Q_or_D"], "prev_starter_not_listed")
    res = {"n_games_with_snapshots": int(len(gm)),
           "median_hours_before_early_snapshot": float(gm.hb_early.median())}
    mv = {}
    base = tt[tt.kind == "none"]
    mv["no_change_teams"] = _move(base)
    for k in ("new", "return"):
        x = tt[(tt.kind == k) & (tt.opp_kind != k)]
        mv[f"{k}:all"] = _move(x)
        for inf, xx in x.groupby("info"):
            mv[f"{k}:{inf}"] = _move(xx)
    res["line_move_early_to_close_toward_team"] = mv
    # ---- bets: best allowed-book price at a snapshot, on a side, graded vs price-based close
    snap["ev_cons"] = _ev(snap.m_cons, snap.side.values, snap.point, snap.sp_price, sig, w)
    snap["clv"] = _ev(snap.mu_close_all, snap.side.values, snap.point, snap.sp_price, sig, w)
    snap["clv_sharp"] = _ev(snap.mu_close_sharp, snap.side.values, snap.point, snap.sp_price, sig, w)
    adj = snap.result_margin + snap.point
    dec = np.where(snap.sp_price > 0, 1 + snap.sp_price / 100, 1 + 100 / -snap.sp_price)
    snap["pnl"] = np.where(adj > 0, dec - 1, np.where(adj < 0, -1.0, 0.0))

    def bet(team_rows: pd.DataFrame, against: bool, when: str) -> pd.DataFrame:
        """For each team-game row: side = that team (or its opponent), snapshot = first after `when` column."""
        if team_rows.empty:
            return pd.DataFrame()
        tr = team_rows[["game_id", "side", when]].copy()
        tr["bet_side"] = np.where(tr.side == "home", "away", "home") if against else tr.side
        s = snap.merge(tr[["game_id", "bet_side", when]], left_on=["game_id", "side"], right_on=["game_id", "bet_side"])
        s = s[s.requested_ts >= s[when]]
        if s.empty:
            return s
        s = s.sort_values(["game_id", "requested_ts", "ev_cons"], ascending=[True, True, False])
        s = s[s.requested_ts == s.groupby("game_id").requested_ts.transform("min")]
        return s.groupby("game_id").head(1)                   # best allowed book at the first qualifying snapshot

    tt["t_ready"] = tt.game_id.map(ready).fillna(pd.Timestamp("1900-01-01", tz="UTC"))
    tt["t_out"] = tt.P_modified.fillna(tt.t_ready)
    tt["t_out"] = tt[["t_out", "t_ready"]].max(axis=1)
    books = {}
    solo = tt[~((tt.kind == tt.opp_kind) & (tt.kind != "none"))]
    # Diagnostics (oracle = uses the actual starter, i.e. perfect early news)
    books["D1_oracle_early_against_new_qb_drop"] = bet(solo[(solo.kind == "new") & (solo.gap < -0.05)], True, "t_ready")
    books["D2_oracle_early_for_returning_upgrade"] = bet(solo[(solo.kind == "return") & (solo.gap > 0.05)], False, "t_ready")
    books["D3_oracle_early_against_new_qb_all"] = bet(solo[solo.kind == "new"], True, "t_ready")
    books["D4_oracle_early_for_new_qb_all"] = bet(solo[solo.kind == "new"], False, "t_ready")
    # Implementable: news known at bet time
    left = tt[(tt.prev_share_P < 0.5) & (tt.week > 1)]
    books["I1_early_against_team_whose_starter_left_prev_game"] = bet(left[left.opp_kind == "none"], True, "t_ready")
    books["I2_early_for_team_whose_starter_left_prev_game"] = bet(left[left.opp_kind == "none"], False, "t_ready")
    outq = tt[(tt.P_status == "Out") & tt.qb_change]
    books["I3_after_final_report_Out_against_team"] = bet(outq[outq.opp_kind == "none"], True, "t_out")
    books["I4_after_final_report_Out_for_team"] = bet(outq[outq.opp_kind == "none"], False, "t_out")
    qd = tt[tt.P_status.isin(["Questionable", "Doubtful"]) & (tt.week > 1)]    # last week's starter is a game-time call
    books["I5_after_final_report_starter_QD_against_team"] = bet(qd[qd.opp_kind == "none"], True, "t_out")
    books["I6_after_final_report_starter_QD_for_team"] = bet(qd[qd.opp_kind == "none"], False, "t_out")
    # null reference: every game, both sides, best allowed book at the early snapshot (= cost of vig + noise)
    allg = tt.assign(t_ready=tt.t_ready)
    books["N0_baseline_all_sides_early"] = pd.concat([bet(allg[allg.side == "home"], False, "t_ready"),
                                                      bet(allg[allg.side == "home"], True, "t_ready")])
    res["bets"] = {k: clv_stats(v) | {"clv_sharp_close": clv_stats(v, "clv_sharp").get("mean_clv")}
                   for k, v in books.items()}
    res["bets_by_season"] = {k: {int(y): clv_stats(g).get("mean_clv") for y, g in v.groupby("season")}
                             for k, v in books.items() if len(v)}
    return res, books | {"_tt": tt}


def _move(x: pd.DataFrame) -> dict:
    m = x.move_to_team.dropna()
    if len(m) < 2:
        return {"n": int(len(m))}
    return {"n": int(len(m)), "mean": round(float(m.mean()), 2), "se": round(float(m.std(ddof=1) / math.sqrt(len(m))), 2),
            "mean_abs": round(float(m.abs().mean()), 2), "share_abs_ge_1": round(float((m.abs() >= 1).mean()), 3)}


# ============================================================================ frozen rules
def apply_frozen(rules: list[dict], tg: pd.DataFrame, holdout: bool) -> dict:
    out = {}
    early = None
    for rule in rules:
        if rule["type"] == "close_ats":
            seasons = HOLD if holdout else [s for s in list(LONG) + list(DEV)]
            d = tg[tg.season.isin(seasons)]
            q = d.query(rule["filter"], engine="python")
            if rule["bet"] == "against":
                q = d.merge(q[["game_id", "opp"]].rename(columns={"opp": "team"}), on=["game_id", "team"])
            q = q.drop_duplicates("game_id")
            out[rule["id"]] = binom_rule(q) | {"by_season": {int(y): cover_stats(g).get("cover") for y, g in q.groupby("season")}}
        else:
            if early is None:
                _, early = early_analysis(tg, holdout)
            b = early[rule["book"]]
            out[rule["id"]] = clv_stats(b) | {"clv_sharp_close": clv_stats(b, "clv_sharp").get("mean_clv"),
                                              "by_season": {int(y): clv_stats(g).get("mean_clv") for y, g in b.groupby("season")}}
    return out


def main():
    hold = "--holdout" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    if hold:
        if not FROZEN.exists() or os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
            raise SystemExit("holdout locked: freeze rules in output/research/qb_timing_frozen.json first")
        fr = json.loads(FROZEN.read_text())
        tg = team_games(2025)
        res = apply_frozen(fr["rules"], tg, True)
        (OUT / "qb_timing_holdout.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res, indent=2, default=str))
        return
    tg = team_games(2022)
    res = {"close_2012_2019": close_analysis(tg, LONG), "close_2020_2022": close_analysis(tg, DEV),
           "close_2012_2022": close_analysis(tg, list(LONG) + list(DEV))}
    er, _ = early_analysis(tg, False)
    res["early_2020_2022"] = er
    counts = tg[tg.season <= 2022].groupby("subtype").size().to_dict()
    res["counts_2012_2022"] = {k: int(v) for k, v in counts.items()}
    if FROZEN.exists():
        res["frozen_rules_dev"] = apply_frozen(json.loads(FROZEN.read_text())["rules"], tg, False)
    (OUT / "qb_timing_dev.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
