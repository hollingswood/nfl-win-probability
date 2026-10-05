"""DEFENSIVE + MORE QB/RB PROPS screen. PRE-DECLARED 2026-10-04, written and committed before the odds were downloaded
(odds_backfill.yml plan `more_props`: 2023-25, Friday-early and close snapshots, regions=us).

Markets (nflverse stats_player column):
  defense:  player_sacks (def_sacks), player_solo_tackles (def_tackles_solo + def_tackles_with_assist),
            player_tackles_assists (solo + with_assist + def_tackle_assists; mapping fixed after the first run, see STAT), player_defensive_interceptions (def_interceptions)
  offense:  player_pass_tds (passing_tds), player_pass_interceptions (passing_interceptions),
            player_pass_completions (completions), player_rush_attempts (carries)
Snapshots: early = Fri 21:40 UTC for Sunday games, kickoff - 24 h otherwise; close = kickoff - 75 min.
Main line: both prices in [-200, +170]; if a book shows several such points for a player, keep the one priced closest
  to even. No-vig per quote: multiplicative (as the receptions rule that passed, rec_shop_early).
Settlement: stat vs point. Player with no stat line: 0 if he took >= 1 snap on his side of the ball (defense_snaps for
  defensive markets, offense_snaps otherwise) in nflverse snap counts, else void (also void if not found at all).

Two pre-declared rules per market (16 tests):
  S  SHOP (copy of rec_shop_early): at the early snapshot, each allowed-book (my_books.json) quote vs the median no-vig
     probability of the SAME side at the SAME point at >= 3 OTHER books; EV = fair x decimal - 1; bet if 2% <= EV < 50%;
     one bet per player-market (highest EV). CLV = median no-vig of >= 2 close books at the same point x decimal - 1.
     Pass: >= 100 bets with CLV, mean CLV > 0 at one-sided p < 0.05/8, CLV positive in >= 2 of 3 seasons.
  U  BLIND UNDER: at the early snapshot, every player-market quoted by >= 2 books: UNDER at the allowed book with the
     highest point, then best price. Pass: >= 150 bets, ROI > 0 at one-sided p < 0.05/8, ROI positive in >= 2 of 3 seasons.
     (CLV is reported for U but is not its test: earlier research found prop closes carry the over/under bias.)
A market that passes S or U becomes a paper track for 2026 (new rules file, frozen before any bet). Nothing else is
tuned: no thresholds, books or windows are searched. Descriptives (over rate at the consensus line, hold) are reported.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nflpred import player_stats as PS  # noqa: E402

PROPS = ROOT / "data" / "historical_odds" / "props"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "research"
SEASONS = (2023, 2024, 2025)
ALLOWED = set(json.load(open(ROOT / "my_books.json"))["allowed_books"])
FILES = {"defense": "player_sacks+player_solo_tackles+player_tackles_assists+player_defensive_interceptions",
         "offense": "player_pass_tds+player_pass_interceptions+player_pass_completions+player_rush_attempts"}
# official solo = nflverse def_tackles_solo + def_tackles_with_assist; official total adds def_tackle_assists
# (checked: Zaire Franklin 2024 official 173 total / 93 solo = 75 + 18 + 80 in nflverse)
STAT = {"player_sacks": ("def_sacks",), "player_solo_tackles": ("def_tackles_solo", "def_tackles_with_assist"),
        "player_tackles_assists": ("def_tackles_solo", "def_tackles_with_assist", "def_tackle_assists"), "player_defensive_interceptions": ("def_interceptions",),
        "player_pass_tds": ("passing_tds",), "player_pass_interceptions": ("passing_interceptions",),
        "player_pass_completions": ("completions",), "player_rush_attempts": ("carries",)}
DEF = {"player_sacks", "player_solo_tackles", "player_tackles_assists", "player_defensive_interceptions"}
K = 8
LO, HI = -200, 170


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def p_one(m, se):
    return float(0.5 * math.erfc((m / se) / math.sqrt(2))) if se and se > 0 else 1.0


def load() -> pd.DataFrame:
    fr = []
    for fam, stem in FILES.items():
        for s in SEASONS:
            f = PROPS / f"{stem}_{s}.csv.gz"
            if f.exists():
                fr.append(pd.read_csv(f).assign(season=s))
    if not fr:
        raise SystemExit("no data yet: run the odds backfill plan more_props")
    d = pd.concat(fr, ignore_index=True)
    rt, ct = pd.to_datetime(d.requested_ts, utc=True), pd.to_datetime(d.commence_time, utc=True)
    d["snap"] = np.where((ct - rt).dt.total_seconds() < 3 * 3600, "close", "early")
    d["ct"] = ct
    return d


def main_lines(d: pd.DataFrame) -> pd.DataFrame:
    q = d.dropna(subset=["point", "over_price", "under_price"])
    q = q[q.over_price.between(LO, HI) & q.under_price.between(LO, HI)].copy()
    io, iu = 1 / dec(q.over_price), 1 / dec(q.under_price)
    q["p_over"] = io / (io + iu)
    q["bal"] = np.abs(io - iu)
    q = q.sort_values("bal").drop_duplicates(["event_id", "snap", "market", "book", "player"])
    return q


def map_games(q: pd.DataFrame) -> pd.Series:
    g = pd.read_parquet(RAW / "games.parquet")
    g = g[g.season.isin(SEASONS)][["game_id", "gameday", "home_team", "away_team"]]
    g["gameday"] = pd.to_datetime(g.gameday).dt.date
    ev = q[["event_id", "home", "away", "ct"]].drop_duplicates("event_id").copy()
    ev["gameday"] = ev.ct.dt.tz_convert("America/New_York").dt.date
    m = ev.merge(g, left_on=["home", "away", "gameday"], right_on=["home_team", "away_team", "gameday"], how="left")
    return m.set_index("event_id").game_id


class Settler:
    def __init__(self):
        self.st = {s: PS.load_week_stats(s, refresh=False) for s in SEASONS}
        self.sn = {s: pd.read_parquet(RAW / f"snap_counts_{s}.parquet") for s in SEASONS}
        self.cache = {}

    def value(self, season, game_id, player, market):
        key = (game_id, player, market)
        if key in self.cache:
            return self.cache[key]
        st = self.st[season]; st = st[st.game_id == game_id]
        out = None
        if len(st):
            cands = {i: {PS.norm_name(a), PS.norm_name(b)} for i, a, b in zip(st.index, st.player_display_name, st.player_name)}
            hit = PS.match(player, cands)
            if hit is not None:
                out = float(sum(0 if pd.isna(st.loc[hit, c]) else st.loc[hit, c] for c in STAT[market]))
            else:
                sn = self.sn[season]; sn = sn[sn.game_id == game_id]
                col = "defense_snaps" if market in DEF else "offense_snaps"
                h2 = PS.match(player, {i: {PS.norm_name(n)} for i, n in zip(sn.index, sn.player)})
                if h2 is not None and sn.loc[h2, col] > 0:
                    out = 0.0
        self.cache[key] = out
        return out


def summarize(R: pd.DataFrame, test: str, n_min: int) -> dict:
    if R.empty:
        return {"bets": 0, "pass": False}
    c = R.clv.dropna()
    m_c, se_c = (c.mean(), c.std() / math.sqrt(len(c))) if len(c) > 1 else (np.nan, np.nan)
    m_r, se_r = R.pnl.mean(), R.pnl.std() / math.sqrt(len(R)) if len(R) > 1 else np.nan
    per = R.groupby("season").agg(n=("pnl", "size"), roi=("pnl", "mean"), clv=("clv", "mean"))
    st = {"bets": int(len(R)), "bets_with_clv": int(len(c)), "mean_ev_at_bet": round(float(R.ev.mean()), 4) if "ev" in R else None,
          "mean_clv": None if len(c) < 2 else round(float(m_c), 4), "clv_p": None if len(c) < 2 else p_one(m_c, se_c),
          "roi": round(float(m_r), 4), "roi_se": None if len(R) < 2 else round(float(se_r), 4), "roi_p": p_one(m_r, se_r),
          "win_rate": round(float((R.res > 0).sum() / max(1, (R.res != 0).sum())), 3),
          "by_season": {int(s): {"n": int(x.n), "roi": round(float(x.roi), 4), "clv": None if pd.isna(x.clv) else round(float(x.clv), 4)}
                        for s, x in per.iterrows()}}
    if test == "clv":
        st["pass"] = bool(len(c) >= n_min and m_c > 0 and st["clv_p"] < 0.05 / K and int((per.clv > 0).sum()) >= 2)
    else:
        st["pass"] = bool(len(R) >= n_min and m_r > 0 and st["roi_p"] < 0.05 / K and int((per.roi > 0).sum()) >= 2)
    return st


def main():
    d = load()
    print("rows", len(d), d.groupby(["market", "snap"]).size().to_dict(), flush=True)
    q = main_lines(d)
    gid = map_games(q)
    q["game_id"] = q.event_id.map(gid)
    q = q.dropna(subset=["game_id"])
    E, C = q[q.snap == "early"], q[q.snap == "close"]
    S = Settler()
    # close fair per (event, market, player, point): median no-vig over books
    cl = C.groupby(["event_id", "market", "player", "point"]).agg(p_close=("p_over", "median"), n_close=("book", "nunique"))
    cl = cl[cl.n_close >= 2].p_close.to_dict()
    out, desc = {}, {}

    def settle(r, side, point, price):
        v = S.value(int(r.season), r.game_id, r.player, r.market)
        if v is None:
            return None
        x = (point - v) if side == "under" else (v - point)
        dd = float(dec(price))
        pnl = dd - 1 if x > 0 else (-1.0 if x < 0 else 0.0)
        pc = cl.get((r.event_id, r.market, r.player, point))
        clv = np.nan if pc is None else ((1 - pc) if side == "under" else pc) * dd - 1
        return {"pnl": pnl, "res": x, "clv": clv, "stat": v}

    for mk in sorted(STAT):
        e = E[E.market == mk]
        if e.empty:
            out[mk] = {"S": {"bets": 0, "pass": False}, "U": {"bets": 0, "pass": False}, "note": "no early quotes"}
            continue
        # ---- S: shop vs leave-one-out median of >= 3 other books at the same point
        rows = []
        grp = e.groupby(["event_id", "player", "point"])
        for (eid, pl, pt), g in grp:
            if g.book.nunique() < 4:
                continue
            for r in g[g.book.isin(ALLOWED)].itertuples():
                oth = g[g.book != r.book]
                if oth.book.nunique() < 3:
                    continue
                for side, px, pf in (("over", r.over_price, oth.p_over.median()), ("under", r.under_price, 1 - oth.p_over.median())):
                    ev = pf * float(dec(px)) - 1
                    if 0.02 <= ev < 0.5:
                        rows.append((r, side, pt, px, ev))
        best = {}
        for r, side, pt, px, ev in rows:
            k = (r.event_id, r.player)
            if k not in best or ev > best[k][4]:
                best[k] = (r, side, pt, px, ev)
        RS = []
        for r, side, pt, px, ev in best.values():
            s = settle(r, side, pt, px)
            if s:
                RS.append({**s, "season": r.season, "ev": ev, "side": side, "book": r.book})
        RS = pd.DataFrame(RS)
        # ---- U: blind under at the best allowed-book number
        RU = []
        for (eid, pl), g in e.groupby(["event_id", "player"]):
            if g.book.nunique() < 2:
                continue
            mine = g[g.book.isin(ALLOWED)]
            if mine.empty:
                continue
            r = mine.assign(d=dec(mine.under_price)).sort_values(["point", "d"], ascending=False).iloc[0]
            s = settle(r, "under", r.point, r.under_price)
            if s:
                RU.append({**s, "season": r.season, "ev": np.nan, "side": "under", "book": r.book})
        RU = pd.DataFrame(RU)
        out[mk] = {"S": summarize(RS, "clv", 100), "U": summarize(RU, "roi", 150)}
        # descriptives: consensus line over rate and hold at the early snapshot
        cons = e.groupby(["event_id", "player"]).agg(point=("point", lambda s: s.mode().iloc[0]), p=("p_over", "median"),
                                                     season=("season", "first"), game_id=("game_id", "first"), market=("market", "first"))
        hits = []
        for r in cons.reset_index().itertuples():
            v = S.value(int(r.season), r.game_id, r.player, r.market)
            if v is not None and v != r.point:
                hits.append((v > r.point, r.p))
        hold = (1 / dec(e.over_price) + 1 / dec(e.under_price) - 1).mean()
        desc[mk] = {"player_markets": int(len(cons)), "graded": len(hits), "over_rate": round(float(np.mean([h[0] for h in hits])), 3) if hits else None,
                    "market_p_over": round(float(np.mean([h[1] for h in hits])), 3) if hits else None, "avg_hold": round(float(hold), 4),
                    "books_per_player": round(float(e.groupby(["event_id", "player"]).book.nunique().mean()), 2)}
        print(mk, {"S": {k: out[mk]["S"].get(k) for k in ("bets", "bets_with_clv", "mean_clv", "clv_p", "roi", "pass")},
                   "U": {k: out[mk]["U"].get(k) for k in ("bets", "roi", "roi_se", "roi_p", "win_rate", "pass")}}, desc[mk], flush=True)
    (OUT / "props_more.json").write_text(json.dumps({"rules": out, "descriptives": desc}, indent=1, default=str))


if __name__ == "__main__":
    main()
