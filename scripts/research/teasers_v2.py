"""NFL 6-point 2-team teasers, v2: every book's OWN number and juice, legs valued from the market's price-implied
fair margin (not the nflverse closing number), and teaser prices per book (fixed) or "dynamic" (leg-priced).

Stages (run in order; outputs in output/research/):
  python scripts/research/teasers_v2.py calib     # fit margin distribution on 2012-2019; leg calibration
                                                  # (LOSO 2012-2019 nflverse closes + 2020-2022 snapshots)
  python scripts/research/teasers_v2.py dev       # 2020-2022 only: windows, fair source, filters, rule grid
  python scripts/research/teasers_v2.py freeze    # writes teasers_v2_frozen.json (<=3 rules; refuses overwrite)
  EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES python scripts/research/teasers_v2.py holdout   # 2023-2025, ONCE
  python scripts/research/teasers_v2.py juice     # descriptive juice x number table, dev + holdout seasons
  python scripts/research/teasers_v2.py report    # renders teasers_v2.md
Optional env TEASERS_V2_CACHE=<dir> caches the per-season leg tables (parquet).

Model
  * Final home margin k (integer): P(k) ∝ Normal(k; mu, SIGMA) * w(|k|), w = key-number weights refit on 2012-2019
    nflverse closing spreads AND closing spread prices (price-implied mu), by iterative raking so the fitted model
    reproduces the 2012-2019 frequency of every |margin| (incl. exactly 3 and 7). SIGMA and smoothing picked by
    leave-one-season-out log likelihood on 2012-2019. Nothing after 2019 is used to fit the distribution.
  * Book-implied mu: the mu at which P(cover)/(P(cover)+P(lose)) of that book's home line equals its no-vig price.
    fair_cons = median over ALL books at the snapshot; fair_sharp = median over lowvig/betonlineag (edge_lab.SHARP),
    falling back to fair_cons when no sharp book quoted that snapshot.
  * Teased leg: side point + 6 at the book; wins iff side_margin + point + 6 > 0. P(win), P(push) from fair mu.
Pricing scenarios (2-team, 6 pt, tie_loses: a pushed leg loses the teaser -- conservative; most Wong legs are half
points so pushes are rare)
  (a) FIXED: draftkings -120, fanduel -134, betmgm -130, williamhill_us (Caesars) -120 (June-2026 BettingUSA guide;
      unverified, state-dependent). Payout ignores leg juice; juice only moves our estimate of fair mu.
  (b) DYNAMIC (assumption, no historical teaser-price data exists): teaser decimal = product of the two teased-line
      (alt-line) decimal prices; each alt price = 1 / (p_book * (1 + v)), where p_book = the book's OWN probability of
      the teased line from its own main line + juice (de-vigged), and v = that book's main-line overround.
      b_key: the book prices alt lines with the same key-number distribution as ours.
      b_norm: sensitivity -- the book prices alt lines from a plain normal (no key numbers).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from nflpred import spread_bets as SB  # noqa: E402

OUT = ROOT / "output" / "research"
JSON = OUT / "teasers_v2.json"
FROZEN = OUT / "teasers_v2_frozen.json"
MD = OUT / "teasers_v2.md"
OLD_JSON = OUT / "teasers.json"
DEV = (2020, 2021, 2022)
HOLD = (2023, 2024, 2025)
FIT = tuple(range(2012, 2020))
SHARP = {"lowvig", "betonlineag", "circasports", "bookmaker"}
FIXED = {"draftkings": -120, "fanduel": -134, "betmgm": -130, "williamhill_us": -120}
TEASE = 6.0
WONG = (1.5, 2.0, 2.5, -7.5, -8.0, -8.5)
KS = np.arange(-60, 61)
GRID = np.round(np.arange(-30, 30.0001, 0.02), 4)
CACHE = os.environ.get("TEASERS_V2_CACHE")
WINDOWS = ("tue", "fri", "sun", "kick")


# ======================================================================================= helpers
def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / -a)


def imp(a):
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def store(key, res):
    OUT.mkdir(parents=True, exist_ok=True)
    cur = json.loads(JSON.read_text()) if JSON.exists() else {}
    cur[key] = res
    JSON.write_text(json.dumps(cur, indent=1, default=_jd))


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _am(d):
    return round(-100 / (d - 1)) if d < 2 else round((d - 1) * 100)


def r4(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)


# ======================================================================================= distribution
def pmf(mu, sigma, w):
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    p = np.exp(-0.5 * ((KS[None, :] - mu) / sigma) ** 2) * w[None, :]
    return p / p.sum(1, keepdims=True)


def fit_weights(mu, m, sigma, smooth=20.0, iters=6):
    """Symmetric weights over KS (w[k] == w[-k]) by raking: model |margin| counts -> actual counts."""
    act = np.bincount(np.clip(np.abs(m), 0, 60), minlength=61).astype(float)
    w = np.ones(121)
    for _ in range(iters):
        P = pmf(mu, sigma, w).sum(0)
        exp = np.zeros(61)
        np.add.at(exp, np.abs(KS), P)
        f = np.clip((act + smooth) / (exp + smooth), 0.2, 5)
        w = np.clip(w * f[np.abs(KS)], 0.03, 4)
    return w


class Dist:
    """Fast lookups on a mu grid for one (sigma, weights)."""

    def __init__(self, sigma, w):
        self.sigma, self.w = float(sigma), np.asarray(w, float)
        P = pmf(GRID, sigma, self.w)                    # (G, 121) home-margin pmf
        self.cdf = np.cumsum(P, 1)                     # P(m <= KS[j])
        self.P = P
        self._q = {}

    def _gi(self, mu):
        return np.clip(np.rint((np.asarray(mu, float) + 30) / 0.02).astype(int), 0, len(GRID) - 1)

    def p_gt(self, mu, x):
        """P(home margin > x) for x half or whole; vectorized over mu, x."""
        gi = self._gi(mu)
        x = np.asarray(x, float)
        j = np.clip(np.floor(x).astype(int) + 60, -1, 120)     # index of largest k <= x
        c = np.where(j >= 0, self.cdf[gi, np.clip(j, 0, 120)], 0.0)
        return 1 - c

    def p_eq(self, mu, x):
        gi = self._gi(mu)
        x = np.asarray(x, float)
        isint = np.abs(x - np.round(x)) < 1e-9
        j = np.clip(np.round(x).astype(int) + 60, 0, 120)
        return np.where(isint, self.P[gi, j], 0.0)

    def side_probs(self, mu_home, side_home, point):
        """Win/push of a side with spread `point` (+ = getting points): home wins iff m + point > 0,
        away wins iff -m + point > 0 <=> m < point."""
        mu_home = np.asarray(mu_home, float)
        point = np.asarray(point, float)
        side_home = np.asarray(side_home, bool)
        pw_h = self.p_gt(mu_home, -point)
        pp = self.p_eq(mu_home, np.where(side_home, -point, point))
        pw_a = 1 - self.p_gt(mu_home, point) - self.p_eq(mu_home, point)
        return np.where(side_home, pw_h, pw_a), pp

    def implied_mu(self, home_point, q_home):
        """mu such that home no-push cover share = q_home; home_point = book's home spread (e.g. -3.5)."""
        home_point = np.asarray(home_point, float)
        q_home = np.asarray(q_home, float)
        out = np.full(len(home_point), np.nan)
        for L in np.unique(home_point[~np.isnan(home_point)]):
            if L not in self._q:
                # home covers iff m > -L
                hc = self.p_gt(GRID, np.full(len(GRID), -L))
                pu = self.p_eq(GRID, np.full(len(GRID), -L))
                q = hc / np.maximum(1 - pu, 1e-12)
                q = np.maximum.accumulate(q)  # monotone guard
                self._q[L] = q
            sel = home_point == L
            out[sel] = np.interp(q_home[sel], self._q[L], GRID)
        return out


def load_dist() -> tuple[Dist, Dist, dict]:
    J = json.loads(JSON.read_text())
    c = J["calib"]["chosen"]
    key = Dist(c["sigma"], np.array(c["weights"]))
    norm = Dist(c["sigma"], np.ones(121))
    return key, norm, c


# ======================================================================================= games / odds
def games(seasons) -> pd.DataFrame:
    from nflpred.weather import _kickoff_utc
    g = pd.read_parquet(ROOT / "data" / "raw" / "games.parquet")
    g = g[g.season.isin(list(seasons)) & g.home_score.notna()].copy()
    g["gameday"] = pd.to_datetime(g.gameday)
    g["gametime"] = g.gametime.where(g.gametime.notna(), "13:00")
    g["kick"] = pd.to_datetime([_kickoff_utc(r.gameday, r.gametime) for r in g.itertuples()], utc=True)
    g["m"] = (g.home_score - g.away_score).astype(int)
    return g


def load_odds(seasons) -> pd.DataFrame:
    fs = [ROOT / "data" / "historical_odds" / f"nfl_odds_{s}.csv.gz" for s in seasons]
    o = pd.concat([pd.read_csv(f).assign(season=s) for f, s in zip(fs, seasons)], ignore_index=True)
    o["requested_ts"] = pd.to_datetime(o.requested_ts, utc=True)
    o["commence"] = pd.to_datetime(o.commence_time, utc=True)
    return o


def load_totals(seasons) -> pd.DataFrame:
    fs = [ROOT / "data" / "historical_odds" / "totals" / f"nfl_odds_{s}.csv.gz" for s in seasons]
    t = pd.concat([pd.read_csv(f).assign(season=s) for f, s in zip(fs, seasons) if f.exists()], ignore_index=True)
    t["requested_ts"] = pd.to_datetime(t.requested_ts, utc=True)
    t["commence"] = pd.to_datetime(t.commence_time, utc=True)
    return t


def match(o, g):
    import replay_early_lines as R
    return R.match_games(o, g)


def build_legs(seasons, key: Dist, norm: Dist) -> pd.DataFrame:
    """One row per (game, snapshot, allowed book, side) with leg probabilities, plus per-snapshot fair mus."""
    tag = f"{min(seasons)}_{max(seasons)}"
    if CACHE:
        p = Path(CACHE) / f"legs_{tag}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    from nflpred import odds as O
    allowed = O.load_allowed_books()
    g = games(seasons)
    o = match(load_odds(seasons), g)
    o = o.merge(g[["game_id", "kick", "week", "game_type", "home_team", "away_team", "m", "spread_line",
                   "total_line", "gameday", "weekday"]], on="game_id")
    o = o[(o.requested_ts < o.kick) & o.sp_home_point.notna() & o.sp_home_price.notna() & o.sp_away_price.notna()]
    o = o[(o.sp_home_point + o.sp_away_point).abs() < 1e-9]
    # main lines only: alt-line entries (e.g. BetMGM 2020 rows at -250 on shifted numbers) cannot be teased at a
    # fixed price, so keep juice in [-145, +125] and numbers within 2.5 pts of the snapshot's median number
    o = o[o.sp_home_price.between(-145, 125) & o.sp_away_price.between(-145, 125) & (o.sp_home_point.abs() <= 30)]
    o = o[(o.sp_home_point - o.groupby(["event_id", "requested_ts"]).sp_home_point.transform("median")).abs() <= 2.5]
    o = o[(o.sp_home_point * 2) % 1 == 0]
    o = o.sort_values("last_update").drop_duplicates(["game_id", "requested_ts", "book"], keep="last")
    ih, ia = imp(o.sp_home_price.values), imp(o.sp_away_price.values)
    o["q_home"] = ih / (ih + ia)
    o["overround"] = ih + ia - 1
    o = o[o.overround.between(-0.01, 0.12)]
    o["mu_book"] = key.implied_mu(o.sp_home_point.values, o.q_home.values)
    o["mu_book_norm"] = norm.implied_mu(o.sp_home_point.values, o.q_home.values)
    k2 = ["game_id", "requested_ts"]
    cons = o.groupby(k2).agg(mu_cons=("mu_book", "median"), n_books=("book", "nunique"),
                             pt_cons=("sp_home_point", "median"))
    sh = o[o.book.isin(SHARP)].groupby(k2).agg(mu_sharp=("mu_book", "median"), n_sharp=("book", "nunique"))
    snap = cons.join(sh).reset_index()
    snap["sharp_fallback"] = snap.mu_sharp.isna()
    snap["mu_sharp"] = snap.mu_sharp.fillna(snap.mu_cons)
    # closing fair = last snapshot before kick
    last = snap.sort_values("requested_ts").groupby("game_id").tail(1)
    snap = snap.merge(last[["game_id", "mu_sharp", "mu_cons", "requested_ts"]].rename(
        columns={"mu_sharp": "mu_close_sharp", "mu_cons": "mu_close_cons", "requested_ts": "close_ts"}), on="game_id")
    # consensus total at (or before) the snapshot
    t = load_totals(seasons)
    t = match(t, g)
    t = t[t.tot_point.notna()].groupby(k2).tot_point.median().rename("tot_snap").reset_index()
    snap = snap.sort_values("requested_ts")
    t = t.sort_values("requested_ts")
    snap = pd.merge_asof(snap, t, on="requested_ts", by="game_id", direction="backward")
    b = o[o.book.isin(allowed)].merge(snap, on=k2)
    rows = []
    for side in ("home", "away"):
        x = b[["game_id", "season", "week", "game_type", "requested_ts", "kick", "book", "m", "spread_line",
               "total_line", "gameday", "weekday", "q_home", "overround", "mu_book", "mu_book_norm", "mu_cons",
               "mu_sharp", "sharp_fallback", "mu_close_sharp", "mu_close_cons", "close_ts", "tot_snap", "n_books",
               "pt_cons"]].copy()
        x["side"] = side
        x["point"] = b[f"sp_{side}_point"].values
        x["price"] = b[f"sp_{side}_price"].values
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    hs = (L.side == "home").values
    L["tpoint"] = L.point + TEASE
    sgn = np.where(hs, 1, -1)
    L["side_margin"] = sgn * L.m
    adj = L.side_margin + L.tpoint
    L["y"] = np.sign(adj).astype(int)          # +1 win, 0 push, -1 loss
    L["y_raw"] = np.sign(L.side_margin + L.point).astype(int)
    for c in ("cons", "sharp", "close_sharp", "close_cons"):
        pw, pp = key.side_probs(L[f"mu_{c}"].values, hs, L.tpoint.values)
        L[f"pw_{c}"], L[f"pp_{c}"] = pw, pp
    # book's own view of the teased line (dynamic pricing)
    pwb, ppb = key.side_probs(L.mu_book.values, hs, L.tpoint.values)
    pwn, ppn = norm.side_probs(L.mu_book_norm.values, hs, L.tpoint.values)
    L["pw_book"], L["pp_book"], L["pw_bnorm"], L["pp_bnorm"] = pwb, ppb, pwn, ppn
    v = L.overround.clip(lower=0).values
    L["alt_dec_key"] = 1 / np.clip(pwb / np.maximum(1 - ppb, 1e-9) * (1 + v), 1e-6, 0.995)
    L["alt_dec_norm"] = 1 / np.clip(pwn / np.maximum(1 - ppn, 1e-9) * (1 + v), 1e-6, 0.995)
    L["wong_num"] = L.point.isin(WONG)
    L["typ"] = np.where(L.point.isin(WONG[:3]), "wong_dog", np.where(L.point.isin(WONG[3:]), "wong_fav",
                        np.where(L.point > 0, "dog", np.where(L.point < 0, "fav", "pk"))))
    L["hours_before"] = (L.kick - L.requested_ts).dt.total_seconds() / 3600
    L = L.drop(columns=["m"])
    if CACHE:
        Path(CACHE).mkdir(parents=True, exist_ok=True)
        L.to_parquet(Path(CACHE) / f"legs_{tag}.parquet")
    return L


# ======================================================================================= windows
def assign_windows(L: pd.DataFrame) -> pd.DataFrame:
    """Tag each row with the decision window it belongs to (or none).
    tue  = Tuesday 14:10 UTC of the game week (5 days before the week's first Sunday)
    fri  = Friday 21:40 UTC (2 days before Sunday); sun = Sunday 14:10 UTC (~3h before the 1pm ET slate)
    kick = each game's last snapshot (~75 min pre-kick). Rows must be >= 10 min before their own kickoff."""
    L = L.copy()
    wk = L.groupby(["season", "week"]).gameday.apply(
        lambda s: s[pd.to_datetime(s).dt.dayofweek == 6].min()).rename("sunday").reset_index()
    L = L.merge(wk, on=["season", "week"], how="left")
    sun = pd.to_datetime(L.sunday).dt.tz_localize("UTC")
    ts = L.requested_ts
    L["win"] = ""
    for name, off, hhmm in (("tue", -5, (14, 10)), ("fri", -2, (21, 40)), ("sun", 0, (14, 10))):
        target = sun + pd.Timedelta(days=off) + pd.Timedelta(hours=hhmm[0], minutes=hhmm[1])
        L.loc[(ts == target), "win"] = name
    last = L.groupby("game_id").requested_ts.transform("max")
    L["is_last"] = ts == last
    L = L[L.hours_before >= 10 / 60]
    return L


# ======================================================================================= teaser selection
def teaser_candidates(L: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Rows eligible under a rule's leg filters, at its window."""
    w = rule["window"]
    X = L[L.is_last] if w == "kick" else L[L.win == w]
    if rule.get("books"):
        X = X[X.book.isin(rule["books"])]
    else:
        X = X[X.book.isin(FIXED)]
    if rule.get("legs") == "wong":
        X = X[X.wong_num]
    elif rule.get("legs") == "wong_dog":
        X = X[X.typ == "wong_dog"]
    if rule.get("half_only"):
        X = X[(X.tpoint % 1) != 0]
    if rule.get("total_max") is not None:
        X = X[X.tot_snap.fillna(99) <= rule["total_max"]]
    if rule.get("min_leg_p") is not None:
        X = X[X[f"pw_{rule['fair']}"] >= rule["min_leg_p"]]
    return X


def select(L: pd.DataFrame, rule: dict, scenario: str) -> pd.DataFrame:
    """Greedy: within each (season, week) pick the highest-EV 2-team teaser (both legs same book, same snapshot,
    different games), remove its games, repeat while EV >= ev_min and pairs < max_pairs.
    scenario: 'fixed' | 'b_key' | 'b_norm'. Returns one row per teaser."""
    X = teaser_candidates(L, rule)
    if X.empty:
        return pd.DataFrame()
    f = rule["fair"]
    X = X.assign(p=X[f"pw_{f}"].values)
    if scenario == "fixed":
        X = X.assign(score=X.p, legdec=1.0)
    else:
        col = "alt_dec_key" if scenario == "b_key" else "alt_dec_norm"
        X = X.assign(score=X.p * X[col], legdec=X[col])
    ev_min, cap = rule.get("ev_min", 0.0), rule.get("max_pairs", 1)
    out = []
    for (s, wkn), d in X.groupby(["season", "week"]):
        d = d.sort_values("score", ascending=False)
        # best side per (game, book, ts); never both sides of a game
        d = d.drop_duplicates(["game_id", "book", "requested_ts"])
        used = set()
        for _ in range(cap):
            best = None
            for (bk, ts), e in d[~d.game_id.isin(used)].groupby(["book", "requested_ts"]):
                if len(e) < 2:
                    continue
                a, b = e.iloc[0], e.iloc[1]
                if scenario == "fixed":
                    D = float(dec(FIXED[bk]))
                else:
                    D = a.legdec * b.legdec
                ev = D * a.p * b.p - 1
                if best is None or ev > best[0]:
                    best = (ev, D, a, b)
            if best is None or best[0] < ev_min:
                break
            ev, D, a, b = best
            used |= {a.game_id, b.game_id}
            win = (a.y > 0) and (b.y > 0)
            cev = D * a[f"pw_close_{'sharp' if f == 'sharp' else 'cons'}"] * b[f"pw_close_{'sharp' if f == 'sharp' else 'cons'}"] - 1
            out.append({"season": s, "week": wkn, "book": a.book, "ts": a.requested_ts, "dec": D,
                        "american": round(-100 / (D - 1)) if D < 2 else round((D - 1) * 100),
                        "ev": ev, "close_ev": cev, "win": int(win), "push": int(a.y == 0 or b.y == 0),
                        "pnl": (D - 1) if win else -1.0,
                        "g1": a.game_id, "s1": a.side, "pt1": a.point, "pr1": a.price, "p1": a.p, "y1": a.y, "t1": a.typ,
                        "g2": b.game_id, "s2": b.side, "pt2": b.point, "pr2": b.price, "p2": b.p, "y2": b.y, "t2": b.typ,
                        "hb": min(a.hours_before, b.hours_before)})
    return pd.DataFrame(out)


def summarize(T: pd.DataFrame, nseasons: int) -> dict:
    if T is None or T.empty:
        return {"teasers": 0, "per_season": 0.0}
    n = len(T)
    pnl = T.pnl.values
    legs_y = np.r_[T.y1.values, T.y2.values]
    se = float(pnl.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    cse = float(T.close_ev.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"teasers": n, "per_season": round(n / nseasons, 1), "hit_rate": r4(T.win.mean()),
            "hit_rate_pred": r4((T.p1 * T.p2).mean()),
            "leg_win": r4((legs_y > 0).mean()), "leg_pred": r4(np.r_[T.p1, T.p2].mean()), "pushes": int(T.push.sum()),
            "avg_price": _am(float(np.mean(T.dec))), "roi": r4(pnl.mean()), "roi_se": r4(se),
            "units": round(float(pnl.sum()), 2), "ev_decision": r4(T.ev.mean()), "ev_close": r4(T.close_ev.mean()),
            "ev_close_se": r4(cse), "by_book": T.book.value_counts().to_dict(),
            "leg_types": pd.Series(np.r_[T.t1, T.t2]).value_counts().to_dict()}


def eval_rule(L, rule, nseasons, keep=False):
    res = {}
    T_fixed = select(L, rule, "fixed")
    res["a_fixed"] = summarize(T_fixed, nseasons)
    # the SAME fixed-selected teasers if the book priced them dynamically
    if not T_fixed.empty:
        res["a_selected_priced_b_key"] = summarize(reprice(L, T_fixed, "alt_dec_key", rule["fair"]), nseasons)
        res["a_selected_priced_b_norm"] = summarize(reprice(L, T_fixed, "alt_dec_norm", rule["fair"]), nseasons)
        res["a_by_season"] = {int(s): summarize(d, 1) for s, d in T_fixed.groupby("season")}
    for sc in ("b_key", "b_norm"):
        res[sc] = summarize(select(L, rule, sc), nseasons)
    if keep:
        return res, T_fixed
    return res


def reprice(L, T, col, fair):
    k = L.set_index(["game_id", "requested_ts", "book", "side"])[col]
    T = T.copy()
    d1 = k.reindex(pd.MultiIndex.from_arrays([T.g1, T.ts, T.book, T.s1])).values
    d2 = k.reindex(pd.MultiIndex.from_arrays([T.g2, T.ts, T.book, T.s2])).values
    T["dec"] = d1 * d2
    T["american"] = np.where(T.dec < 2, -100 / (T.dec - 1), (T.dec - 1) * 100)
    T["pnl"] = np.where(T.win == 1, T.dec - 1, -1.0)
    cl = "sharp" if fair == "sharp" else "cons"
    kc = L.set_index(["game_id", "requested_ts", "book", "side"])[f"pw_close_{cl}"]
    c1 = kc.reindex(pd.MultiIndex.from_arrays([T.g1, T.ts, T.book, T.s1])).values
    c2 = kc.reindex(pd.MultiIndex.from_arrays([T.g2, T.ts, T.book, T.s2])).values
    T["ev"] = T.dec * T.p1 * T.p2 - 1
    T["close_ev"] = T.dec * c1 * c2 - 1
    return T


# ======================================================================================= old approach on same games
def old_approach(L: pd.DataFrame, nseasons: int) -> dict:
    """Old study R1/R2 (nflverse closing number in Wong range, R2 total <= 48.5), paired in kickoff order within the
    week, graded at the nflverse number, at each fixed book price; restricted to games present in our snapshots.
    Also: the model's leg probability for those legs at the close (fair = sharp close)."""
    G = L[L.is_last].drop_duplicates(["game_id", "side"]).copy()
    G["close_pt"] = np.where(G.side == "home", -G.spread_line, G.spread_line)
    G = G[G.close_pt.isin(WONG)]
    adj = G.side_margin + G.close_pt + TEASE
    G["yo"] = np.sign(adj)
    out = {}
    import nflpred.margins  # noqa: F401
    key, _, _ = load_dist()
    pw, _ = key.side_probs(G.mu_close_sharp.values, (G.side == "home").values, G.close_pt.values + TEASE)
    G["p_close"] = pw
    for rid, sel in (("R1_wong_classic", G), ("R2_wong_total_le48_5", G[G.total_line <= 48.5])):
        rows = []
        for (s, w), d in sel.sort_values(["kick", "game_id"]).groupby(["season", "week"]):
            d = d.drop_duplicates("game_id")
            for i in range(0, len(d) - 1, 2):
                a, b = d.iloc[i], d.iloc[i + 1]
                rows.append((s, a.yo > 0 and b.yo > 0, a.yo, b.yo, a.p_close * b.p_close))
        P = pd.DataFrame(rows, columns=["season", "win", "y1", "y2", "pp"])
        blk = {"legs": int(len(sel)), "leg_win": r4((sel.yo > 0).mean()), "leg_pred_close_model": r4(sel.p_close.mean()),
               "teasers": int(len(P)), "per_season": round(len(P) / nseasons, 1), "hit_rate": r4(P.win.mean()),
               "hit_rate_pred": r4(P.pp.mean()), "by_price": {}}
        for bk, a in sorted(set((k, v) for k, v in FIXED.items()), key=lambda z: z[1]):
            D = float(dec(a))
            pnl = np.where(P.win, D - 1, -1.0)
            blk["by_price"][str(a)] = {"roi": r4(pnl.mean()), "roi_se": r4(pnl.std(ddof=1) / math.sqrt(len(P))),
                                       "ev_model_close": r4((D * P.pp - 1).mean())}
        out[rid] = blk
    return out


# ======================================================================================= juice / number analysis
def juice_analysis(L: pd.DataFrame) -> dict:
    """How much do book-level number/juice differences matter for which legs qualify?"""
    X = L[L.book.isin(FIXED) & (L.win.isin(["tue", "fri", "sun"]) | L.is_last)].copy()
    out = {}
    grp = X.groupby(["game_id", "requested_ts", "side"])
    nums = grp.point.agg(["min", "max", "nunique"])
    out["side_snapshots"] = int(len(nums))
    out["share_books_differ_in_number"] = r4((nums["nunique"] > 1).mean())
    # posted number Wong at some book but not all
    wn = grp.wong_num.agg(["any", "all"])
    out["share_wong_at_some_not_all"] = r4((wn["any"] & ~wn["all"]).mean())
    out["share_wong_any"] = r4(wn["any"].mean())
    # spread of fair teased-leg prob among Wong-number legs that share the SAME posted number
    W = X[X.wong_num]
    out["wong_leg_p_by_number"] = [
        {"point": float(k), "legs": int(len(d)), "p_mean": r4(d.pw_sharp.mean()), "p_p10": r4(d.pw_sharp.quantile(.1)),
         "p_p90": r4(d.pw_sharp.quantile(.9)), "price_p10": float(d.price.quantile(.1)), "price_p90": float(d.price.quantile(.9))}
        for k, d in W.groupby("point")]
    # classification: number-only rule vs price-aware rule at each book's fixed break-even
    X["be"] = np.sqrt(1 / dec(X.book.map(FIXED).values))
    X["qual_ev"] = X.pw_sharp >= X.be
    t = pd.crosstab(X.wong_num, X.qual_ev)
    out["number_rule_vs_ev_rule"] = {f"wong_{a}_evqual_{b}": int(t.loc[a, b]) if (a in t.index and b in t.columns) else 0
                                     for a in (True, False) for b in (True, False)}
    out["share_wong_legs_failing_ev"] = r4(1 - X[X.wong_num].qual_ev.mean())
    out["share_ev_legs_not_wong"] = r4(1 - X[X.qual_ev].wong_num.mean()) if X.qual_ev.any() else None
    # same game-side-snapshot, two books with different numbers: e.g. +1.5 -125 vs +2.5 -105
    nn = X.groupby(["game_id", "requested_ts", "side"]).point.transform("nunique")
    D = X[nn > 1]
    if len(D):
        hi = D.loc[D.groupby(["game_id", "requested_ts", "side"]).point.idxmax()]
        lo = D.loc[D.groupby(["game_id", "requested_ts", "side"]).point.idxmin()]
        m = hi.merge(lo, on=["game_id", "requested_ts", "side"], suffixes=("_hi", "_lo"))
        m["juice_better_on_hi"] = imp(m.price_hi) <= imp(m.price_lo)
        out["number_differs"] = {"n": int(len(m)), "hi_number_also_better_or_equal_juice": r4(m.juice_better_on_hi.mean()),
                                 "teased_p_gain_hi_vs_lo": r4((m.pw_sharp_hi - m.pw_sharp_lo).mean()),
                                 "main_line_ev_gain_hi_vs_lo": None}
        # main-line (untreated) EV of each book's leg under fair: is the extra half point 'paid for' by juice?
        key, _, _ = load_dist()
        for sfx in ("hi", "lo"):
            pw, pp = key.side_probs(m.mu_sharp_hi.values, (m.side == "home").values, m[f"point_{sfx}"].values)
            m[f"ev_main_{sfx}"] = pw * dec(m[f"price_{sfx}"].values) + pp - 1
        out["number_differs"]["main_line_ev_hi_minus_lo"] = r4((m.ev_main_hi - m.ev_main_lo).mean())
        # does the teaser choice (fixed price) differ from the straight-bet choice?
        out["number_differs"]["straight_prefers_lo_number"] = r4((m.ev_main_lo > m.ev_main_hi).mean())
        mw = m[m.wong_num_hi | m.wong_num_lo]
        out["number_differs"]["wong_cases"] = int(len(mw))
        out["number_differs"]["wong_cases_straight_prefers_lo"] = r4((mw.ev_main_lo > mw.ev_main_hi).mean()) if len(mw) else None
    return out


# ======================================================================================= calibration
def leg_calibration(df: pd.DataFrame, pcol: str, ycol: str = "y") -> list[dict]:
    ppc = pcol.replace("pw_", "pp_")
    d = df[df[ycol] != 0].copy()
    d["w"] = (d[ycol] > 0).astype(float)
    d["pn"] = d[pcol] / np.maximum(1 - d[ppc], 1e-9) if ppc in d.columns else d[pcol]
    d["b"] = pd.cut(d.pn, [0, .6, .65, .7, .72, .74, .76, .78, .8, .85, 1])
    rows = []
    for b, e in d.groupby("b", observed=True):
        p, a, n = e.pn.mean(), e.w.mean(), len(e)
        rows.append({"bucket": str(b), "n": n, "pred": r4(p), "actual": r4(a),
                     "z": round((a - p) / math.sqrt(p * (1 - p) / n), 2)})
    return rows


def cal_by(df, pcol, by, ycol="y"):
    d = df[df[ycol] != 0].copy()
    d["w"] = (d[ycol] > 0).astype(float)
    ppc = pcol.replace("pw_", "pp_")
    d["pn"] = d[pcol] / np.maximum(1 - d[ppc], 1e-9) if ppc in d else d[pcol]
    rows = []
    for k, e in d.groupby(by, observed=True):
        p, a, n = e.pn.mean(), e.w.mean(), len(e)
        rows.append({**dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,))),
                     "n": n, "pred": r4(p), "actual": r4(a), "se": r4(math.sqrt(p * (1 - p) / n)),
                     "z": round((a - p) / math.sqrt(p * (1 - p) / n), 2),
                     "logloss": r4(-(e.w * np.log(e.pn) + (1 - e.w) * np.log(1 - e.pn)).mean())})
    return rows


def run_calib():
    g = games(range(1999, 2023))
    g = g[g.spread_line.notna()].copy()
    ih = imp(g.home_spread_odds.fillna(-110).values)
    ia = imp(g.away_spread_odds.fillna(-110).values)
    g["q"] = ih / (ih + ia)
    F = g[g.season.isin(FIT)].copy()
    m = F.m.values
    # price-implied closing mu needs weights; 2 rounds
    w = fit_weights(-(-F.spread_line.values), m, 13.4)
    mu = Dist(13.4, w).implied_mu(-F.spread_line.values, F.q.values)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "fit_seasons": list(FIT), "grid": []}
    best = None
    old = SB.load_rules()
    w_old = np.array([old["_weights"].get(abs(int(k)), 1.0) for k in KS])
    for sig in (12.8, 13.1, 13.4, 13.7, 14.0):
        for sm in (5.0, 20.0, 50.0):
            ll = 0.0
            for s in FIT:
                tr = F.season.values != s
                ww = fit_weights(mu[tr], m[tr], sig, smooth=sm)
                P = pmf(mu[~tr], sig, ww)
                ll += np.log(P[np.arange((~tr).sum()), m[~tr] + 60]).sum()
            ll /= len(F)
            res["grid"].append({"sigma": sig, "smooth": sm, "loso_loglik": round(ll, 5)})
            if best is None or ll > best[0]:
                best = (ll, sig, sm)
    _, sig, sm = best
    w = fit_weights(mu, m, sig, smooth=sm)
    D = Dist(sig, w)
    mu = D.implied_mu(-F.spread_line.values, F.q.values)  # refresh with chosen dist
    w = fit_weights(mu, m, sig, smooth=sm)
    D = Dist(sig, w)
    res["chosen"] = {"sigma": sig, "smooth": sm, "weights": [round(float(x), 5) for x in w],
                     "weights_0_14": {int(k): round(float(w[k + 60]), 3) for k in range(15)},
                     "old_weights_0_14": {int(k): round(float(w_old[k + 60]), 3) for k in range(15)},
                     "old_sigma": old["margin"]["sigma"]}
    # exact 3 / 7 check (LOSO) and leg calibration by era, old vs new distribution
    rows3 = []
    legs_all = []
    for lab, mk in (("new_loso", None), ("old_spread_rules", (old["margin"]["sigma"], w_old))):
        for s in range(1999, 2023):
            e = g[g.season == s]
            if mk is None:
                if s in FIT:
                    tr = F.season.values != s
                    ww = fit_weights(mu[tr], m[tr], sig, smooth=sm)
                    Ds = Dist(sig, ww)
                else:
                    Ds = D
            else:
                Ds = Dist(*mk)
            mus = Ds.implied_mu(-e.spread_line.values, e.q.values)
            P = pmf(mus, Ds.sigma, Ds.w)
            rows3.append({"model": lab, "season": s, "pred3": P[:, [57, 63]].sum(), "act3": int((np.abs(e.m) == 3).sum()),
                          "pred7": P[:, [53, 67]].sum(), "act7": int((np.abs(e.m) == 7).sum()), "n": len(e)})
            for side in ("home", "away"):
                hs = np.full(len(e), side == "home")
                pt = np.where(hs, -e.spread_line.values, e.spread_line.values)
                pw, pp = Ds.side_probs(mus, hs, pt + TEASE)
                sm_ = np.where(hs, e.m.values, -e.m.values)
                legs_all.append(pd.DataFrame({"model": lab, "season": s, "point": pt, "pw": pw, "pp": pp,
                                              "y": np.sign(sm_ + pt + TEASE), "total": e.total_line.values}))
    R3 = pd.DataFrame(rows3)
    R3["era"] = pd.cut(R3.season, [1998, 2011, 2019, 2022], labels=["1999-2011", "2012-2019", "2020-2022"])
    res["exact_3_7"] = R3.groupby(["model", "era"], observed=True)[["pred3", "act3", "pred7", "act7", "n"]].sum().round(1).reset_index().to_dict("records")
    LG = pd.concat(legs_all, ignore_index=True)
    LG["era"] = pd.cut(LG.season, [1998, 2011, 2019, 2022], labels=["1999-2011", "2012-2019 (new=LOSO)", "2020-2022"])
    LG["typ"] = np.where(LG.point.isin(WONG[:3]), "wong_dog", np.where(LG.point.isin(WONG[3:]), "wong_fav", "other"))
    LG = LG.rename(columns={"pw": "pw_x", "pp": "pp_x"})
    res["closes_leg_cal_by_type"] = cal_by(LG, "pw_x", ["model", "era", "typ"])
    LG["tb"] = pd.cut(LG.total, [0, 41, 44, 47, 49, 80], labels=["<41", "41-43.5", "44-46.5", "47-48.5", ">=49"])
    W = LG[(LG.typ != "other") & (LG.model == "new_loso") & (LG.season >= 2012)]
    res["closes_wong_cal_by_total_2012_2022"] = cal_by(W, "pw_x", ["tb"])
    res["closes_leg_cal_buckets_new_2012_2022"] = leg_calibration(LG[(LG.model == "new_loso") & (LG.season >= 2012)], "pw_x")
    store("calib", res)
    # ---- 2020-2022 snapshot calibration (dev)
    key, norm, _ = load_dist()
    L = assign_windows(build_legs(DEV, key, norm))
    snapcal = {}
    for wname in WINDOWS:
        X = L[L.is_last] if wname == "kick" else L[L.win == wname]
        X = X.drop_duplicates(["game_id", "side", "point"])  # one row per distinct posted number
        snapcal[wname] = {"sharp_by_type": cal_by(X, "pw_sharp", ["typ"]), "cons_by_type": cal_by(X, "pw_cons", ["typ"]),
                          "sharp_wong_all": cal_by(X[X.wong_num].assign(a="wong"), "pw_sharp", ["a"]),
                          "cons_wong_all": cal_by(X[X.wong_num].assign(a="wong"), "pw_cons", ["a"])}
    X = L[L.is_last].drop_duplicates(["game_id", "side", "point"])
    snapcal["kick_buckets_sharp"] = leg_calibration(X, "pw_sharp")
    res["snap_2020_2022"] = snapcal
    store("calib", res)
    print(json.dumps({k: res[k] for k in ("grid", "exact_3_7")}, indent=1, default=_jd)[:3000])


# ======================================================================================= dev
GRID_RULES = []


def dev_rule_grid():
    rules = []
    for window in WINDOWS:
        for fair in ("sharp", "cons"):
            for legs in ("any", "wong"):
                for ev_min in (-1.0, -0.02, 0.0, 0.015):
                    for cap in (1, 3):
                        for tot in (None, 48.5):
                            rid = f"{window}|{fair}|{legs}|ev{ev_min}|cap{cap}|tot{tot}"
                            rules.append({"id": rid, "window": window, "fair": fair, "legs": legs, "ev_min": ev_min,
                                          "max_pairs": cap, "total_max": tot})
    return rules


def window_clv(L: pd.DataFrame) -> list[dict]:
    """For legs at fixed-price books: mean teased-leg prob at decision vs at close (sharp), by window and leg type.
    Positive drift = the number you got was better than the closing market says (leg-level CLV)."""
    rows = []
    for w in WINDOWS:
        X = L[L.is_last] if w == "kick" else L[L.win == w]
        X = X[X.book.isin(FIXED)]
        X = X.sort_values("pw_sharp", ascending=False).drop_duplicates(["game_id", "side", "requested_ts"])  # best book
        for typ, d in [("all", X), ("wong_num", X[X.wong_num]), ("wong_dog", X[X.typ == "wong_dog"]),
                       ("wong_fav", X[X.typ == "wong_fav"]), ("dog", X[X.point > 0]), ("fav", X[X.point < 0])]:
            if len(d) == 0:
                continue
            clv = d.pw_close_sharp - d.pw_sharp
            dd = d[d.y != 0]
            rows.append({"window": w, "legs": typ, "n": len(d), "p_decision": r4(d.pw_sharp.mean()),
                         "p_close": r4(d.pw_close_sharp.mean()), "clv_pts": r4(clv.mean() * 100),
                         "clv_se_pts": r4(clv.std(ddof=1) / math.sqrt(len(d)) * 100), "actual": r4((dd.y > 0).mean()),
                         "share_positive_ev_at_-120": r4((d.pw_sharp >= math.sqrt(1 / 1.8333)).mean())})
    return rows


def run_dev():
    key, norm, _ = load_dist()
    L = assign_windows(build_legs(DEV, key, norm))
    n = len(DEV)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "seasons": list(DEV),
           "rows": int(len(L)), "sharp_fallback_share": r4(L.sharp_fallback.mean()),
           "window_rows": L.win.value_counts().to_dict(), "games": int(L.game_id.nunique())}
    res["window_clv"] = window_clv(L)
    res["juice"] = juice_analysis(L)
    res["old_approach"] = old_approach(L, n)
    grid = []
    for r in dev_rule_grid():
        e = eval_rule(L, r, n)
        a = e["a_fixed"]
        grid.append({"id": r["id"], **{k: a.get(k) for k in ("teasers", "per_season", "hit_rate", "hit_rate_pred",
                                                               "roi", "roi_se", "ev_decision", "ev_close", "ev_close_se")},
                     "b_key_teasers": e["b_key"]["teasers"], "b_key_roi": e["b_key"].get("roi"),
                     "b_norm_teasers": e["b_norm"]["teasers"], "b_norm_roi": e["b_norm"].get("roi"),
                     "b_norm_ev_close": e["b_norm"].get("ev_close"),
                     "a_as_b_key_roi": e.get("a_selected_priced_b_key", {}).get("roi"),
                     "a_as_b_key_ev_close": e.get("a_selected_priced_b_key", {}).get("ev_close")})
        print(grid[-1]["id"], grid[-1]["teasers"], grid[-1]["roi"], grid[-1]["ev_close"], flush=True)
    res["grid"] = grid
    store("dev", res)


# ======================================================================================= freeze / holdout
# Chosen after reviewing the 2020-2022 dev grid (teasers_v2.json["dev"]["grid"]); <=3 rules, frozen before any
# 2023-2025 evaluation. Fixed-price scenario (a) selects; the same rule is also run under the dynamic scenarios.
CANDIDATES: list[dict] = [
    {"id": "V1_fri_ev_pos", "window": "fri", "fair": "sharp", "legs": "any", "ev_min": 0.0, "max_pairs": 3,
     "total_max": None,
     "desc": "Friday 21:40 UTC snapshot: at DK/FD/MGM/CZR, any main-line legs teased 6 from THAT book's number; leg "
             "win prob from the sharp (lowvig/betonlineag) price-implied fair margin; take the highest-EV same-book "
             "pair if EV >= 0 at that book's fixed teaser price; up to 3 disjoint pairs per week."},
    {"id": "V2_kick_ev_pos", "window": "kick", "fair": "sharp", "legs": "any", "ev_min": 0.0, "max_pairs": 3,
     "total_max": None,
     "desc": "Same as V1 but at each game's last snapshot (~75 min pre-kick); pairs only among games sharing that "
             "snapshot (same kickoff window)."},
    {"id": "V3_fri_wong_bestbook", "window": "fri", "fair": "sharp", "legs": "wong", "ev_min": -1.0, "max_pairs": 3,
     "total_max": None,
     "desc": "Classic Wong, executed honestly: legs whose number AT THE BOOK is +1.5/+2/+2.5 or -7.5/-8/-8.5 at the "
             "Friday snapshot; best-EV same-book pairs (book number, juice and fixed price all count), NO EV "
             "threshold, up to 3 disjoint pairs per week. Baseline for what the model's pessimism costs."},
]


def run_freeze():
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; refusing to overwrite (frozen rules are final)")
    J = json.loads(JSON.read_text())
    if "dev" not in J:
        raise SystemExit("run dev first")
    if not CANDIDATES or len(CANDIDATES) > 3:
        raise SystemExit("define 1-3 CANDIDATES")
    fz = {"frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
          "developed_on": "distribution fit 2012-2019 nflverse closes; rules chosen on 2020-2022 snapshots only",
          "dev_generated": J["dev"]["generated"], "holdout": list(HOLD), "evaluate_once": True,
          "distribution": {"sigma": J["calib"]["chosen"]["sigma"], "smooth": J["calib"]["chosen"]["smooth"],
                           "weights": "teasers_v2.json calib.chosen.weights (fit 2012-2019)"},
          "fixed_prices": FIXED, "grading": "tie_loses", "candidates": CANDIDATES}
    FROZEN.write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))


def run_holdout():
    if os.environ.get("EDGE_HOLDOUT") != "I_HAVE_FROZEN_CANDIDATES":
        raise SystemExit("holdout is locked: set EDGE_HOLDOUT=I_HAVE_FROZEN_CANDIDATES after freezing")
    if not FROZEN.exists():
        raise SystemExit("freeze first")
    J = json.loads(JSON.read_text())
    if "holdout" in J:
        raise SystemExit("holdout already evaluated once; not re-running")
    fz = json.loads(FROZEN.read_text())
    key, norm, _ = load_dist()
    L = assign_windows(build_legs(HOLD, key, norm))
    n = len(HOLD)
    res = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "frozen_at": fz["frozen_at"],
           "rows": int(len(L)), "games": int(L.game_id.nunique()), "rules": {}}
    teas = {}
    for r in fz["candidates"]:
        e, T = eval_rule(L, r, n, keep=True)
        res["rules"][r["id"]] = e
        teas[r["id"]] = T.assign(ts=T.ts.astype(str)).to_dict("records") if not T.empty else []
    res["teasers_list"] = teas
    res["old_approach"] = old_approach(L, n)
    # descriptive (not used for any selection): calibration + window CLV + juice in 2023-2025
    X = L[L.is_last].drop_duplicates(["game_id", "side", "point"])
    res["cal_kick_sharp_by_type"] = cal_by(X, "pw_sharp", ["typ"])
    res["cal_kick_sharp_wong"] = cal_by(X[X.wong_num].assign(a="wong"), "pw_sharp", ["a"])
    res["window_clv"] = window_clv(L)
    res["juice"] = juice_analysis(L)
    store("holdout", res)
    print(json.dumps({k: v for k, v in res.items() if k != "teasers_list"}, indent=1, default=_jd)[:8000])


def juice_by_number(L: pd.DataFrame) -> list[dict]:
    """Wong-number legs at fixed-price books (decision windows): fair teased-leg prob by the leg's OWN juice, and the
    share that clears the -120 break-even (0.7385). Shows how juice (= where the market puts the fair margin) decides
    which legs with the same posted number are worth teasing."""
    X = L[L.book.isin(FIXED) & L.wong_num & (L.win.isin(["tue", "fri", "sun"]) | L.is_last)].copy()
    X["juice"] = pd.cut(X.price, [-200, -116, -111, -106, 200], labels=["<=-117", "-116..-112", "-111..-107", ">=-106"])
    be = float(np.sqrt(1 / dec(-120)))
    rows = []
    for (pt, j), d in X.groupby(["point", "juice"], observed=True):
        if len(d) < 15:
            continue
        rows.append({"point": float(pt), "own_juice": str(j), "legs": int(len(d)), "p_fair": r4(d.pw_sharp.mean()),
                     "share_clears_-120": r4((d.pw_sharp >= be).mean()),
                     "share_clears_-130": r4((d.pw_sharp >= float(np.sqrt(1 / dec(-130)))).mean())})
    return rows


def run_juice_extra():
    """Descriptive only (no rule evaluation): juice x number table for dev and holdout seasons."""
    key, norm, _ = load_dist()
    res = {}
    for lab, seas in (("dev_2020_2022", DEV), ("hold_2023_2025", HOLD)):
        L = assign_windows(build_legs(seas, key, norm))
        res[lab] = juice_by_number(L)
    store("juice_by_number", res)
    print(json.dumps(res, indent=1, default=_jd))


def _tbl(rows, cols=None):
    if not rows:
        return "_none_\n"
    df = pd.DataFrame(rows)
    cols = cols or list(df.columns)
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in df[cols].itertuples(index=False):
        out += "| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r) + " |\n"
    return out


def _pm(x, se, pct=True):
    if x is None:
        return "-"
    if se is None or (isinstance(se, float) and math.isnan(se)):
        return f"{x * 100:+.1f}%" if pct else f"{x:+.3f}"
    return f"{x * 100:+.1f}% ± {se * 100:.1f}" if pct else f"{x:+.3f} ± {se:.3f}"


SCEN = [("a_fixed", "(a) fixed book price, (a)-selected"),
        ("a_selected_priced_b_key", "same teasers, dynamic price b_key"),
        ("a_selected_priced_b_norm", "same teasers, dynamic price b_norm (book ignores key numbers)"),
        ("b_key", "(b) dynamic b_key, own selection"),
        ("b_norm", "(b) dynamic b_norm, own selection")]


def _rule_rows(rules: dict) -> list[dict]:
    rows = []
    for rid, e in rules.items():
        for k, lab in SCEN:
            x = e.get(k) or {}
            if not x.get("teasers"):
                rows.append({"rule": rid, "scenario": lab, "teasers/season": 0})
                continue
            rows.append({"rule": rid, "scenario": lab, "teasers/season": x["per_season"],
                         "hit rate (pred)": f"{x['hit_rate']:.3f} ({x['hit_rate_pred']:.3f})",
                         "leg win (pred)": f"{x['leg_win']:.3f} ({x['leg_pred']:.3f})", "avg price": x["avg_price"],
                         "ROI ± SE": _pm(x["roi"], x["roi_se"]), "units": x["units"],
                         "model EV at bet": _pm(x["ev_decision"], None), "EV at close ± SE": _pm(x["ev_close"], x["ev_close_se"])})
    return rows


def run_report():
    J = json.loads(JSON.read_text())
    c, d, h = J["calib"], J["dev"], J.get("holdout")
    fz = json.loads(FROZEN.read_text())
    jb = J.get("juice_by_number", {})
    old = json.loads(OLD_JSON.read_text()) if OLD_JSON.exists() else {}
    md = [NARRATIVE.strip() + "\n"]

    md.append("\n## 1. Margin distribution (fit on 2012-2019 only)\n")
    ch = c["chosen"]
    md.append(f"Normal(mu, sigma) x key-number weight w(|margin|), mu = price-implied (closing spread AND its juice). "
              f"Chosen by leave-one-season-out log likelihood: sigma = {ch['sigma']}, smoothing = {ch['smooth']} "
              f"pseudo-games; weights raked until the model reproduces 2012-2019 margin frequencies "
              f"(the old spread_rules.json weights, fit 2012-2014, sigma {ch['old_sigma']:.2f}, under-predicted exact 3s by "
              f"~9% and 7s by ~13% on 2012-2019).\n\n")
    md.append(_tbl([{"|margin|": k, "new w": v, "old w": ch["old_weights_0_14"][k]} for k, v in ch["weights_0_14"].items()]))
    md.append("\nExact-3 / exact-7 counts, predicted vs actual (new = leave-one-season-out inside 2012-2019; "
              "1999-2011 and 2020-2022 are out of sample):\n\n")
    md.append(_tbl(c["exact_3_7"]))

    md.append("\n## 2. Leg calibration (6-pt teased legs)\n")
    md.append("Pred = model P(win | no push); actual = W/(W+L). nflverse closes with closing juice:\n\n")
    md.append(_tbl([r for r in c["closes_leg_cal_by_type"]], ["model", "era", "typ", "n", "pred", "actual", "se", "z", "logloss"]))
    md.append("\nWong legs by closing total, 2012-2022 (new model):\n\n")
    md.append(_tbl(c["closes_wong_cal_by_total_2012_2022"]))
    md.append("\n2020-2022 snapshots (dev), allowed-book posted numbers, fair = sharp price-implied margin at that snapshot "
              "(one row per distinct game/side/number):\n\n")
    rows = []
    for w in WINDOWS:
        for r in c["snap_2020_2022"][w]["sharp_by_type"]:
            rows.append({"window": w, **r})
        for r in c["snap_2020_2022"][w]["sharp_wong_all"]:
            rows.append({"window": w, "typ": "ALL WONG", **{k: v for k, v in r.items() if k != "a"}})
    md.append(_tbl(rows, ["window", "typ", "n", "pred", "actual", "se", "z"]))
    md.append("\nCalibration buckets, last pre-kick snapshot 2020-2022:\n\n")
    md.append(_tbl(c["snap_2020_2022"]["kick_buckets_sharp"]))
    if h:
        md.append("\n2023-2025 (descriptive, computed with the holdout run; not used for any choice), last pre-kick snapshot:\n\n")
        md.append(_tbl(h["cal_kick_sharp_by_type"] + [{"typ": "ALL WONG", **{k: v for k, v in r.items() if k != 'a'}}
                                                      for r in h["cal_kick_sharp_wong"]], ["typ", "n", "pred", "actual", "se", "z"]))

    md.append("\n## 3. Juice and number: does it change which legs qualify?\n")
    for lab, J2 in (("2020-2022", d["juice"]), ("2023-2025", h["juice"] if h else None)):
        if not J2:
            continue
        nd = J2["number_differs"]
        md.append(f"\n**{lab}** (DK/FD/MGM/CZR, decision windows): the four books post different numbers on the same side in "
                  f"{J2['share_books_differ_in_number']:.0%} of side-snapshots; a Wong number is available at some but not all "
                  f"of them in {J2['share_wong_at_some_not_all']:.1%} (Wong at any: {J2['share_wong_any']:.1%}). "
                  f"{J2['share_wong_legs_failing_ev']:.0%} of Wong-number legs do NOT clear their book's fixed-price break-even "
                  f"under the price-implied fair margin; {J2['share_ev_legs_not_wong'] or 0:.1%} of legs that do clear are not Wong numbers. "
                  f"When two books differ on the number (n={nd['n']}), the higher number has equal-or-better juice only "
                  f"{nd['hi_number_also_better_or_equal_juice']:.0%} of the time, adds {nd['teased_p_gain_hi_vs_lo'] * 100:.1f} pts of "
                  f"teased-leg win probability, and a straight bettor would still prefer the LOWER number {nd['straight_prefers_lo_number']:.0%} "
                  f"of the time (Wong cases: {nd['wong_cases_straight_prefers_lo']:.0%}) -- for a fixed-price teaser the higher number always wins.\n")
        md.append("\nFair teased-leg probability within the SAME posted number:\n\n")
        md.append(_tbl(J2["wong_leg_p_by_number"]))
    for lab, rows in jb.items():
        md.append(f"\nWong-number legs by their OWN juice ({lab}); break-even 0.7385 at -120, 0.7518 at -130:\n\n")
        md.append(_tbl(rows))

    md.append("\n## 4. Early or late? Leg-level drift from decision snapshot to close\n")
    md.append("Best fixed-price book per leg at each window; clv_pts = (closing fair P(teased leg wins) - P at decision) x 100, "
              "same number. kick = the close itself (0 by construction).\n")
    for lab, rows in (("2020-2022 (dev)", d["window_clv"]), ("2023-2025 (descriptive)", h["window_clv"] if h else [])):
        md.append(f"\n{lab}:\n\n")
        md.append(_tbl([r for r in rows if r["window"] != "kick"],
                       ["window", "legs", "n", "p_decision", "p_close", "clv_pts", "clv_se_pts", "actual", "share_positive_ev_at_-120"]))

    md.append("\n## 5. Development grid (2020-2022) and frozen rules\n")
    G = pd.DataFrame(d["grid"])
    G = G[G.id.str.contains("\\|sharp\\|") & G.id.str.contains("cap3") & G.id.str.contains("totNone")]
    md.append("Subset shown (sharp fair, up to 3 pairs/week, no total filter); full grid (256 rules) in teasers_v2.json. "
              "ROI is realized at the fixed price; ev_close = EV of the same teaser at the closing fair margin.\n\n")
    md.append(_tbl(G[["id", "teasers", "hit_rate", "hit_rate_pred", "roi", "roi_se", "ev_decision", "ev_close",
                      "ev_close_se", "b_key_teasers", "b_key_roi", "a_as_b_key_roi"]].to_dict("records")))
    md.append("\nFrozen (" + fz["frozen_at"] + ", before any 2023-2025 rule evaluation):\n\n")
    for r in fz["candidates"]:
        md.append(f"* **{r['id']}** -- {r['desc']}\n")

    if h:
        md.append("\n## 6. Holdout 2023-2025 (frozen rules, run once)\n")
        md.append("tie_loses grading. 'hit rate (pred)' = model-predicted share of teasers won. EV at close = expected "
                  "ROI of the same teasers at the closing sharp fair margin (CLV analogue, much less noisy than ROI).\n\n")
        md.append(_tbl(_rule_rows(h["rules"])))
        md.append("\nBy season, scenario (a):\n\n")
        rows = []
        for rid, e in h["rules"].items():
            for s_, x in e.get("a_by_season", {}).items():
                rows.append({"rule": rid, "season": s_, "teasers": x["teasers"], "hit": x["hit_rate"],
                             "ROI": _pm(x["roi"], x["roi_se"]), "EV close": _pm(x["ev_close"], None)})
        md.append(_tbl(rows))
        md.append("\nOld fixed-number approach on the same 2023-2025 games (nflverse closing number, Wong range, legs paired in "
                  "kickoff order; model EV uses the closing sharp fair margin):\n\n")
        rows = []
        for rid, b in h["old_approach"].items():
            for a, x in b["by_price"].items():
                rows.append({"old rule": rid, "price": a, "teasers/season": b["per_season"],
                             "leg win (model)": f"{b['leg_win']:.3f} ({b['leg_pred_close_model']:.3f})",
                             "hit rate (model)": f"{b['hit_rate']:.3f} ({b['hit_rate_pred']:.3f})",
                             "ROI ± SE": _pm(x["roi"], x["roi_se"]), "model EV": _pm(x["ev_model_close"], None)})
        md.append(_tbl(rows))
        md.append("\nSame comparison in 2020-2022 (dev):\n\n")
        rows = []
        for rid, b in d["old_approach"].items():
            for a, x in b["by_price"].items():
                rows.append({"old rule": rid, "price": a, "teasers/season": b["per_season"],
                             "leg win (model)": f"{b['leg_win']:.3f} ({b['leg_pred_close_model']:.3f})",
                             "ROI ± SE": _pm(x["roi"], x["roi_se"]), "model EV": _pm(x["ev_model_close"], None)})
        md.append(_tbl(rows))
    md.append("\n## 7. At the betting window: checklist\n" + CHECKLIST.strip() + "\n")
    MD.write_text("".join(md))
    print(f"wrote {MD}")


NARRATIVE = """
# NFL 6-point teasers v2: each book's own number, juice and teaser price

Generated by `scripts/research/teasers_v2.py` (stages calib -> dev -> freeze -> holdout -> juice -> report).
Data: nflverse results/closing spreads + closing spread prices (distribution fit 2012-2019), per-book spreads
and prices at our live-run snapshots 2020-2025 (`data/historical_odds/`, daily 14:10 UTC, Fri 21:40 UTC, ~75 min
pre-kick), consensus totals from `data/historical_odds/totals/`. Books bet: DraftKings, FanDuel, BetMGM, Caesars
(williamhill_us) -- the four `my_books.json` books with a stated teaser price. Rules developed on 2020-2022, frozen in
`teasers_v2_frozen.json`, run once on 2023-2025.

## Verdict

**Still no demonstrated teaser edge, and under leg-priced ("dynamic") teasers a clear expected loss.**

* Pricing legs honestly (each book's own number; fair margin from the market's prices) says a typical Wong leg wins
  ~0.73-0.74 -- below the 0.7385 break-even at -120 and well below 0.752 (-130) / 0.757 (-134). Only 37-50% of
  Wong-number legs at DK/Caesars -120 clear break-even, few (<10%) at -130/-134, and those that clear do so by
  1-2 pts. Model EV of the teasers the frozen rules found: about +1.8% per teaser, ~9-13 teasers a season.
* 2023-2025 holdout, fixed prices: V1 (Friday, EV>=0) ROI -13% ± 15% (38 teasers), V2 (pre-kick, EV>=0) -1% ± 18%
  (26), V3 (classic Wong at the best book, no EV filter) -6% ± 8% (130). Expected-at-close: +1.0%, +1.8%, -1.7%.
  All within noise of the model; none is evidence of an edge. The old fixed-number approach on the same games:
  +0.6% ± 8.6% at -120, -2.9% at -130, -4.2% at -134 (model expected -1.1% / -4.6% / -5.8%).
* Dynamic pricing (assumed = product of the two teased-line prices with the book's own main-line hold, priced off
  its own key-number-aware view): every rule's teasers price around -145 to -155 and lose 8-10% in expectation
  (realized -11% to -22%). Positive only if a book's alt-line pricer ignored key numbers (b_norm, +8-11% expected) --
  an assumption we cannot verify and should not bet on without seeing the slip price.
* Calibration: the refit distribution matches exact-3/7 frequencies and is well calibrated on 1999-2019 Wong legs.
  Wong legs out-performed it in 2020-2022 (+5 pts, z~2) but only +1.3 pts in 2023-2025 (z 0.6), where dogs
  beat it and favorites badly under-performed it (0.60 vs 0.73) -- the same flip the v1 study saw.
* Juice matters, but not how one might expect: with a FIXED teaser price the payout ignores juice, so on the same game
  always take the most points (+2.5 -105 beats +1.5 -125); across games, juice tells you where the fair margin is, and
  a +1.5/+2.5 dog juiced to -115 or worse is a much better leg (p ~0.745-0.755) than one at -105 or better (~0.73).

Caveats: teaser prices are a June-2026 guide applied to all years (unverified, state-dependent); no historical teaser
price data exist, so (b) is a modelling assumption; tie_loses grading; ~10-45 teasers/season so ROI SEs are 8-18%.
"""

CHECKLIST = """
1. **The teased number AND the main-line juice at every book.** For a fixed-price book, rank legs by fair
   P(teased leg wins), not by the posted number alone: same game -> take the book with the most points regardless of
   juice; across games -> a Wong dog at +1.5/+2.5 juiced -115 or worse is worth teasing at -120, one at -105/even
   usually is not; -8/+2 legs (almost) never cleared -120; fav -7.5 cleared mainly when juiced -112 or worse
   (41-67% of those) and rarely otherwise.
2. **The teaser price on the slip, for the exact pair.** Break-even: slip decimal >= 1/(p1 x p2). Two 0.745 legs need
   about -125 or better; two 0.735 legs need -117 or better. If the slip shows the same price for a pair of
   +1.5->+7.5 Wong legs as for +4.5->+10.5 non-Wong legs, the book is pricing flat (fixed); if the price moves with
   the legs, it is dynamic -- compare it with the product of the alt-line prices and expect ~-145 to -155 on Wong
   pairs (losing). Only a dynamic price clearly BETTER than -120 on a Wong pair would be interesting.
3. **Push rule** for 2-team teasers at that book (we assumed a push loses); prefer half-point teased numbers.
4. **Early vs late.** Leg-level drift to the close is small (|clv| < 0.5 pt of leg probability, mostly within 2 SE):
   Wong dogs gained ~+0.4 pt from Tuesday in both periods (2023-25: +0.43 ± 0.23), Wong favorites and favorites in
   general lost value from Tuesday (-0.3 to -0.6 pt). So: dog legs early (Tue/Fri), favorite legs late; the
   pre-kick snapshot is never worse for favorites and lets you see inactives. Friday edges mostly survived to the
   close (EV at close +1.0% vs +1.8% at decision); Sunday-morning edges did not in dev.
"""


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"calib": run_calib, "dev": run_dev, "freeze": run_freeze, "holdout": run_holdout, "juice": run_juice_extra,
     "report": run_report}[stage]()
