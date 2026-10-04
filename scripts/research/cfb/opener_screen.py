"""College openers vs power ratings, with REAL bettable openers. PRE-DECLARED 2026-10-04 before any result was seen.

Opener = the first snapshot (data/historical_odds/cfb, 2021-26) where any of your books (DraftKings, FanDuel, ESPN Bet)
posts a spread; the bet takes the best number+price for the chosen side among those books at that snapshot.
Weeks >= 4 only (ratings need games); regular season.

O1/O2 MARKET RATINGS (what sharp bettors use to attack openers): before each week, ridge-fit team ratings to every
   earlier CLOSING spread (CFBD consensus): -close = HFA*(not neutral) + r_home - r_away; this season weight 1, last
   season weight 0.3, ridge 2.0. Gap = rating margin - opener market margin. Bet the side the rating favors if
   |gap| >= 3 (O1) or >= 5 (O2).
O3/O4 OUR MODEL (walk-forward m_pred, output/research/cfb/preds.parquet): same with |gap| >= 3 / >= 5.
O5 FADE BIG MOVES: line at your books moved >= 2.5 points from the opener to the snapshot 1-6 h before kickoff ->
   bet against the move at your best late number.
Graded like shop_screen: CLV vs Pinnacle's last pre-kickoff spread (key-number pricing at our number) and ROI.
Pass: >= 100 bets 2021-25, CLV > 0 at one-sided p < 0.01 (5 rules), ROI >= 0, CLV positive in >= 4 of 5 seasons.
2026 reported separately. Power-4 split reported (not a pass criterion).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS  # noqa: E402
import shop_screen as SS  # noqa: E402

MINE = SS.MINE
N_RULES = 5


def event_map(O):
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    out = []
    for s, e in ev.groupby("season"):
        g = PS.D.games([s])
        m = PS.team_matcher(sorted(set(g.home) | set(g.away)))
        idx = {(r.home, r.away): r for r in g.itertuples()}
        for r in e.itertuples():
            gg = idx.get((m(r.home), m(r.away)))
            if gg is not None and abs((gg.start - r.ko).total_seconds()) < 36 * 3600:
                out.append({"event_id": r.event_id, "game_id": int(gg.game_id), "week": int(gg.week), "season_type": gg.season_type})
    return pd.DataFrame(out).set_index("event_id")


def market_ratings(pred: pd.DataFrame) -> pd.Series:
    """game_id -> rating-implied home margin from earlier closing lines only."""
    out = {}
    pred = pred.dropna(subset=["start"]).copy()
    pred["start"] = pd.to_datetime(pred.start, utc=True)
    for s in sorted(pred.season.unique()):
        if s < 2021:
            continue
        cur = pred[pred.season == s]
        for wk, games in cur.groupby("week"):
            t0 = games.start.min()
            hist = pred[((pred.season == s) & (pred.start < t0 - pd.Timedelta(hours=12))) | (pred.season == s - 1)].dropna(subset=["spread_close"])
            if len(hist) < 50:
                continue
            teams = sorted(set(hist.home) | set(hist.away) | set(games.home) | set(games.away))
            ix = {t: i for i, t in enumerate(teams)}
            n = len(teams)
            X = np.zeros((len(hist), n + 1))
            X[np.arange(len(hist)), [ix[t] for t in hist.home]] = 1
            X[np.arange(len(hist)), [ix[t] for t in hist.away]] = -1
            X[:, n] = (~hist.neutral.astype(bool)).astype(float)
            y = -hist.spread_close.to_numpy(float)
            w = np.where(hist.season == s, 1.0, 0.3)
            A = X.T @ (X * w[:, None]) + 2.0 * np.diag([1.0] * n + [0.0])
            beta = np.linalg.solve(A, X.T @ (y * w))
            for r in games.itertuples():
                out[int(r.game_id)] = float(beta[ix[r.home]] - beta[ix[r.away]] + beta[n] * (0 if r.neutral else 1))
    return pd.Series(out, name="rating_margin")


def best_quote(Q, side):
    pt, pr = ("sp_home_point", "sp_home_price") if side == "home" else ("sp_away_point", "sp_away_price")
    q = Q.dropna(subset=[pt, pr]).sort_values([pt, pr], ascending=False).head(1)
    return None if q.empty else q.iloc[0]


def main():
    O = PS.load("cfb")
    R = SS.results_full(O)
    pin = pd.read_parquet(PS.OUT / "pin_mus.parquet")
    close = pin.sort_values("t").groupby("event_id")[["mu_m", "mu_t", "q_ml"]].agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    close.columns = ["c_mu_m", "c_mu_t", "c_q_ml"]
    E = event_map(O)
    pred = pd.read_parquet(PS.OUT / "preds.parquet")
    MR = market_ratings(pred)
    E = E.join(MR, on="game_id").join(pred.set_index("game_id")[["m_pred"]], on="game_id")
    E = E[(E.week >= 4) & (E.season_type == "regular")]
    mine = O[O.book.isin(MINE) & O.sp_home_point.notna()]
    mine = mine[mine.event_id.isin(E.index)]
    first_t = mine.groupby("event_id").t.min()
    rows = {k: [] for k in ("O1", "O2", "O3", "O4", "O5")}
    for eid, g in mine.groupby("event_id"):
        e = E.loc[eid]
        op = g[g.t == first_t[eid]]
        open_mkt = -op.sp_home_point.median()
        for rule, col, th in (("O1", "rating_margin", 3), ("O2", "rating_margin", 5), ("O3", "m_pred", 3), ("O4", "m_pred", 5)):
            if pd.isna(e[col]):
                continue
            gap = e[col] - open_mkt
            if abs(gap) < th:
                continue
            side = "home" if gap > 0 else "away"
            q = best_quote(op, side)
            if q is None:
                continue
            rows[rule].append({"event_id": eid, "t": q.t, "ko": q.ko, "season": q.season, "book": q.book, "last_update": q.last_update,
                               "market": "spread", "side": side, "point": q.sp_home_point if side == "home" else q.sp_away_point,
                               "price": q.sp_home_price if side == "home" else q.sp_away_price, "ev": np.nan, "gap": gap})
        h = (g.ko - g.t).dt.total_seconds() / 3600
        late = g[(h >= 1) & (h <= 6)]
        if not late.empty:
            lt = late.t.max()
            lq = late[late.t == lt]
            move = (-lq.sp_home_point.median()) - open_mkt          # + = market moved toward home
            if abs(move) >= 2.5:
                side = "away" if move > 0 else "home"
                q = best_quote(lq, side)
                if q is not None:
                    rows["O5"].append({"event_id": eid, "t": q.t, "ko": q.ko, "season": q.season, "book": q.book, "last_update": q.last_update,
                                       "market": "spread", "side": side, "point": q.sp_home_point if side == "home" else q.sp_away_point,
                                       "price": q.sp_home_price if side == "home" else q.sp_away_price, "ev": np.nan, "gap": move})
    out = {}
    for rule, rr in rows.items():
        if not rr:
            out[rule] = {"bets": 0}; continue
        B = SS.grade(pd.DataFrame(rr).set_index("event_id"), close, R)
        B["ev"] = B.ev.fillna(0)
        dev = B[B.season <= 2025]
        st = SS.stats(dev)
        per = dev.groupby("season").clv.mean()
        st["seasons_clv_positive"] = f"{int((per > 0).sum())}/{len(per)}"
        st["by_season"] = {int(s): SS.stats(g) for s, g in dev.groupby("season")}
        st["pass"] = bool(st.get("bets", 0) >= 100 and st.get("p", 1) < 0.01 and st.get("roi", -1) >= 0 and int((per > 0).sum()) >= 4)
        st["2026_out_of_sample"] = SS.stats(B[B.season == 2026])
        st["power4_game"] = SS.stats(dev[dev.p4_game])
        st["non_power4"] = SS.stats(dev[~dev.p4_game])
        st["mean_abs_gap"] = round(float(dev.gap.abs().mean()), 2)
        out[rule] = st
        print(rule, {k: st[k] for k in ("bets", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass") if k in st},
              "| 2026", {k: st["2026_out_of_sample"].get(k) for k in ("bets", "mean_clv", "roi")}, flush=True)
    (PS.OUT / "opener_screen.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
