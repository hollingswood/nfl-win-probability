"""College first-half and team-total screen. PRE-DECLARED 2026-10-04, written and committed before the derivative odds
(data/historical_odds/cfb_deriv, plan cfb_deriv: 2023-25, 48 h and 75 min before kickoff) were downloaded.

Quotes: first-half spreads, first-half totals and team totals at Arizona books (DraftKings, FanDuel, ESPN Bet, BetMGM,
Caesars, BetRivers, Fanatics, Hard Rock, Bally), taken at the 48-hour snapshot. Two fair references:
  same-market: Pinnacle's own quote for that market at the SAME number in the same snapshot (Shin no-vig).
  derived:     Pinnacle's full-game spread and total (latest cfb_pin snapshot at or before, within 12 h) turned into the
               derivative with parameters fit on 2014-21 (cfb_deriv_params.json, fit before any derivative odds):
               team total mean = (T +/- M)/2, sd 11.51; 1H total mean = 0.5207 T, sd 11.36; 1H margin mean = 0.6082 M,
               sd 12.12 (normal, integer outcomes, pushes counted).
Rules (one bet per game per market-side group, best EV at the 48 h snapshot, price -200..+200):
  D1 1H spread EV >= 3% vs Pinnacle same market     D4 1H spread EV >= 3% vs derived
  D2 1H total  EV >= 3% vs Pinnacle same market     D5 1H total  EV >= 3% vs derived
  D3 team total EV >= 3% vs Pinnacle same market    D6 team total EV >= 3% vs derived (one bet per team)
Graded: results from CFBD quarter scores; CLV vs Pinnacle's same-market quote at the 75-min snapshot when it is at our
number (Shin); ROI. Pass: >= 100 bets, CLV > 0 at one-sided p < 0.05/6, ROI >= 0, CLV positive in >= 2 of 3 seasons.
D4-D6 also report ROI-only significance (their CLV needs Pinnacle's derivative close).
"""
import glob
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS  # noqa: E402
from nflpred.devig import shin_from_implied  # noqa: E402

AZ = ["draftkings", "fanduel", "espnbet", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]
P = json.loads((ROOT / "cfb_deriv_params.json").read_text())
TH = 0.03
N_RULES = 6


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def probs_over(mean, sd, line):
    """(win, push, lose) for an OVER at `line` with integer outcomes ~ N(mean, sd)."""
    if float(line).is_integer():
        lo, hi = Phi((line - 0.5 - mean) / sd), Phi((line + 0.5 - mean) / sd)
        return 1 - hi, hi - lo, lo
    w = 1 - Phi((math.floor(line) + 0.5 - mean) / sd)
    return w, 0.0, 1 - w


def ev(w, p, l, price):
    d = 1 + price / 100 if price > 0 else 1 + 100 / abs(price)
    return w * (d - 1) - l


def quarter_scores():
    out = {}
    for f in glob.glob(str(ROOT / "data/cfb/raw/games_20*_*.json.gz")):
        for g in json.load(gzip.open(f)):
            h, a = g.get("homeLineScores") or [], g.get("awayLineScores") or []
            if g.get("completed") and len(h) >= 2 and len(a) >= 2:
                out[g["id"]] = (h[0] + h[1], a[0] + a[1], g["homePoints"], g["awayPoints"])
    return out


def main():
    fs = sorted(glob.glob(str(ROOT / "data/historical_odds/cfb_deriv/cfb_deriv_*.csv.gz")))
    if not fs:
        raise SystemExit("no derivative odds yet (run the cfb_deriv backfill)")
    D = pd.concat([pd.read_csv(f).assign(season=int(Path(f).name[10:14])) for f in fs], ignore_index=True)
    D["t"] = pd.to_datetime(D.requested_ts, utc=True); D["ko"] = pd.to_datetime(D.commence_time, utc=True)
    D["h"] = (D.ko - D.t).dt.total_seconds() / 3600
    early, close = D[D.h > 24], D[D.h < 3]
    # map odds events to CFBD games (same matcher as the main screens)
    print("derivative rows", len(D), flush=True)
    O = PS.load("cfb")
    import opener_screen as OS
    E = OS.event_map(O)
    Q = quarter_scores()
    pin_full = PS.load("cfb_pin")
    out, bets_all = {}, []

    def side_rows(df):
        """normalize to rows: event, market, group, side, point, price, book."""
        r = df.copy()
        r["side"] = np.where(r.market == "spreads_h1", np.where(r.name == r.home, "home", "away"), r.name.str.lower())
        r["team_side"] = np.where(r.market == "team_totals", np.where(r.description == r.home, "home", np.where(r.description == r.away, "away", "")), "")
        return r.dropna(subset=["point", "price"])

    E1 = side_rows(early); C1 = side_rows(close)
    pin_e = E1[E1.book == "pinnacle"]; pin_c = C1[C1.book == "pinnacle"]
    books = E1[E1.book.isin(AZ) & E1.price.between(-200, 200)]

    def index(pin):   # (event, market, side, point, team_side) -> price   (lookup speed only; same rule)
        return {(r.event_id, r.market, r.side, float(r.point), r.team_side or ""): r.price for r in pin.itertuples()}
    IDX = {}

    def pin_fair(pin, row):
        """Shin no-vig probability of row's side at row's point from Pinnacle's same market (same number), else None."""
        if id(pin) not in IDX:
            IDX[id(pin)] = index(pin)
        ix = IDX[id(pin)]
        ts = row.team_side or ""
        if row.market == "spreads_h1":
            other = "away" if row.side == "home" else "home"
            a = ix.get((row.event_id, row.market, row.side, float(row.point), ts))
            b = ix.get((row.event_id, row.market, other, float(-row.point), ts))
        else:
            other = "under" if row.side == "over" else "over"
            a = ix.get((row.event_id, row.market, row.side, float(row.point), ts))
            b = ix.get((row.event_id, row.market, other, float(row.point), ts))
        if a is None or b is None:
            return None
        return shin_from_implied(PS.imp(a).item(), PS.imp(b).item())

    pf = pin_full.dropna(subset=["sp_home_point", "tot_point"]).sort_values("t")

    PF = {eid: grp[["t", "sp_home_point", "tot_point"]].to_numpy() for eid, grp in pf.groupby("event_id")}

    def derived(row):
        arr = PF.get(row.event_id)
        if arr is None:
            return None
        ok = [(t, sp, tt) for t, sp, tt in arr if row.t - pd.Timedelta(hours=12) <= t <= row.t]
        if not ok:
            return None
        _, sp, tt = ok[-1]
        M, T = -sp, tt
        if row.market == "spreads_h1":
            mean, sd = P["h1_margin_share"] * M, P["h1_margin_sd"]
            # home covers if margin + home_point > 0  ->  margin > -home_point
            hp = row.point if row.side == "home" else -row.point
            w, p, l = probs_over(mean, sd, -hp)
            return (w, p, l) if row.side == "home" else (l, p, w)
        if row.market == "totals_h1":
            w, p, l = probs_over(P["h1_total_share"] * T, P["h1_total_sd"], row.point)
        else:
            mean = (T + M) / 2 if row.team_side == "home" else (T - M) / 2
            w, p, l = probs_over(mean, P["team_total_sd"], row.point)
        return (w, p, l) if row.side == "over" else (l, p, w)

    rows = []
    print("book quotes to price:", len(books), flush=True)
    for n_, r in enumerate(books.itertuples()):
        if n_ % 20000 == 0:
            print("  priced", n_, flush=True)
        f = pin_fair(pin_e, r)
        e_same = None
        if f is not None:
            e_same = f * (1 + (r.price / 100 if r.price > 0 else 100 / abs(r.price))) - 1    # no push at the same number: share-based
        dv = derived(r)
        e_der = ev(*dv, r.price) if dv else None
        rows.append({**r._asdict(), "ev_same": e_same, "ev_der": e_der})
    B = pd.DataFrame(rows)
    rules = {"D1 1H spread vs Pinnacle 1H": ("spreads_h1", "ev_same"), "D2 1H total vs Pinnacle 1H": ("totals_h1", "ev_same"),
             "D3 team total vs Pinnacle team total": ("team_totals", "ev_same"), "D4 1H spread vs derived": ("spreads_h1", "ev_der"),
             "D5 1H total vs derived": ("totals_h1", "ev_der"), "D6 team total vs derived": ("team_totals", "ev_der")}
    for name, (mk, col) in rules.items():
        q = B[(B.market == mk) & (B[col] >= TH)].copy()
        grp = ["event_id"] + (["team_side"] if mk == "team_totals" else [])
        q = q.sort_values(col, ascending=False).groupby(grp).head(1)
        res = []
        for r in q.itertuples():
            gid = E.game_id.get(r.event_id) if r.event_id in E.index else None
            sc = Q.get(gid)
            if not sc:
                continue
            h1h, a1h, hp_, ap_ = sc
            if mk == "spreads_h1":
                x = (h1h - a1h) + r.point if r.side == "home" else (a1h - h1h) + r.point
            elif mk == "totals_h1":
                x = (h1h + a1h) - r.point if r.side == "over" else r.point - (h1h + a1h)
            else:
                pts = hp_ if r.team_side == "home" else ap_
                x = pts - r.point if r.side == "over" else r.point - pts
            d = 1 + (r.price / 100 if r.price > 0 else 100 / abs(r.price))
            pnl = d - 1 if x > 0 else (-1.0 if x < 0 else 0.0)
            cf = pin_fair(pin_c, r)
            clv = cf * d - 1 if cf is not None else np.nan
            res.append({"event_id": r.event_id, "season": r.season, "book": r.book, "ev": getattr(r, col), "pnl": pnl, "res": x, "clv": clv, "rule": name})
        R = pd.DataFrame(res)
        bets_all.append(R)
        if R.empty:
            out[name] = {"bets": 0}; continue
        c = R.clv.dropna()
        m, se = (c.mean(), c.std() / math.sqrt(len(c))) if len(c) > 1 else (float("nan"), float("nan"))
        per = R.groupby("season").clv.mean()
        st = {"bets": int(len(R)), "mean_ev_at_bet": round(float(R.ev.mean()), 4), "bets_with_clv": int(len(c)),
              "mean_clv": round(float(m), 4) if len(c) > 1 else None, "p": float(0.5 * math.erfc((m / se) / math.sqrt(2))) if len(c) > 1 and se > 0 else None,
              "roi": round(float(R.pnl.mean()), 4), "roi_se": round(float(R.pnl.std() / math.sqrt(len(R))), 4) if len(R) > 1 else None,
              "win_rate": round(float((R.res > 0).sum() / max(1, (R.res != 0).sum())), 3),
              "seasons_clv_positive": f"{int((per > 0).sum())}/{len(per)}",
              "by_season": {int(s): {"n": int(len(g)), "clv": None if g.clv.dropna().empty else round(float(g.clv.mean()), 4), "roi": round(float(g.pnl.mean()), 4)} for s, g in R.groupby("season")}}
        st["pass"] = bool(st["bets"] >= 100 and st["p"] is not None and st["p"] < 0.05 / N_RULES and st["roi"] >= 0 and int((per > 0).sum()) >= 2)
        out[name] = st
        print(name, {k: st[k] for k in ("bets", "mean_ev_at_bet", "bets_with_clv", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass")}, flush=True)
    pd.concat(bets_all).to_parquet(PS.OUT / "deriv_bets.parquet")
    (PS.OUT / "deriv_screen.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
