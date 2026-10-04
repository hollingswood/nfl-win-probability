"""College football PRICE screen (pre-declared 2026-10-03, before any results), NCAAF odds 2021-2026.
Snapshots: 16:10 and 23:10 UTC daily within 8 days of games, plus Saturday 13:10/19:10 and Sunday 01:10/03:10 UTC.
"Your books" = DraftKings, FanDuel, theScore Bet (espnbet). Fair price = Pinnacle no-vig at the same snapshot.
Close = the last snapshot before kickoff that has the quote (Pinnacle for fair prices; US median for points).

P1 soft vs sharp moneyline: first snapshot (>= 1 h before kickoff) where your best ML beats Pinnacle no-vig by EV >= 2%,
   price -300..+300. Graded CLV vs Pinnacle no-vig close, and ROI.
P2 moneyline vs Pinnacle SPREAD: Pinnacle spread -> win prob (normal, sd 15.7, juice-adjusted); bet your best ML when
   EV >= 2% vs that AND EV >= 0 vs Pinnacle's own ML. Same grading as P1.
P3 early unders: at the first snapshot 96-200 h before kickoff, bet the under at your best price when its EV >= 0 vs
   Pinnacle no-vig at the same total. Graded CLV vs the Pinnacle close (normal, sd 16.5, if the total moved) and ROI.
P4 follow Pinnacle early: at the first snapshot >= 48 h before kickoff, if Pinnacle's spread differs from the US median
   by >= 1 point, bet Pinnacle's side at your best number. Graded CLV in points vs the US median close, and ATS/ROI.
One bet per game per rule. Pass: one-sided p < 0.0125 (4 rules) on mean CLV > 0 AND ROI not negative AND most seasons
with positive CLV. 2026 (partial) reported separately as extra out-of-sample data.
"""
from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from cfbpred import data as D  # noqa: E402
from cfbpred.pipeline import team_matcher  # noqa: E402

MINE = {"draftkings", "fanduel", "espnbet"}
SD_M, SD_T = 15.7, 16.5
OUT = ROOT / "output" / "research" / "cfb"


def imp(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


def Phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def Phinv(p):
    from statistics import NormalDist
    return np.vectorize(lambda q: NormalDist().inv_cdf(min(max(q, 1e-6), 1 - 1e-6)))(p)


def load(sub):
    fs = sorted(glob.glob(str(ROOT / f"data/historical_odds/{sub}/cfb_odds_*.csv.gz")))
    o = pd.concat([pd.read_csv(f).assign(season=int(re.findall(r"(\d{4})", f)[-1])) for f in fs], ignore_index=True)
    o["t"] = pd.to_datetime(o.requested_ts, utc=True)
    o["ko"] = pd.to_datetime(o.commence_time, utc=True)
    return o[o.t < o.ko]


def results(O):
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    out = []
    for s, e in ev.groupby("season"):
        g = D.games([s])
        g = g[g.completed & g.margin.notna()]
        m = team_matcher(sorted(set(g.home) | set(g.away)))
        idx = {(r.home, r.away): r for r in g.itertuples()}
        for r in e.itertuples():
            h, a = m(r.home), m(r.away)
            gg = idx.get((h, a))
            if gg is not None and abs((gg.start - r.ko).total_seconds()) < 36 * 3600:
                out.append({"event_id": r.event_id, "margin": float(gg.margin), "total": float(gg.total),
                            "fbs_both": gg.home_div == "fbs" and gg.away_div == "fbs"})
    return pd.DataFrame(out)


def summarize(name, R, clv_col, extra=""):
    R = R.dropna(subset=[clv_col])
    dev = R[R.season <= 2025]
    def stats(x):
        if not len(x):
            return {"bets": 0}
        m, se = x[clv_col].mean(), x[clv_col].std() / math.sqrt(len(x))
        roi = x.pnl.mean()
        return {"bets": int(len(x)), "mean_clv": round(float(m), 4), "se": round(float(se), 4),
                "p": round(0.5 * math.erfc((m / se) / math.sqrt(2)), 5) if se > 0 else 1.0,
                "roi": round(float(roi), 4), "roi_se": round(float(x.pnl.std() / math.sqrt(len(x))), 4),
                "win_rate": round(float((x.res > 0).sum() / max(1, (x.res != 0).sum())), 3)}
    st = stats(dev)
    per = dev.groupby("season")[clv_col].mean()
    st["seasons_clv_positive"] = f"{int((per > 0).sum())}/{len(per)}"
    st["by_season"] = {int(k): {"n": int((dev.season == k).sum()), "clv": round(float(v), 4),
                                "roi": round(float(dev[dev.season == k].pnl.mean()), 3)} for k, v in per.items()}
    st["pass"] = bool(st.get("p", 1) < 0.0125 and st.get("roi", -1) >= 0 and (per > 0).sum() > len(per) / 2)
    st["2026_out_of_sample"] = stats(R[R.season == 2026])
    st["clv_unit"] = extra
    return st


def main():
    O = load("cfb")
    P = load("cfb_pin")
    res = results(O)
    key = ["event_id", "t"]
    # ---------------- Pinnacle fair per snapshot
    pin = P.copy()
    ph, pa = imp(pin.ml_home), imp(pin.ml_away)
    pin["fair_h"] = ph / (ph + pa)
    sh, sa = imp(pin.sp_home_price), imp(pin.sp_away_price)
    pin["cover_h"] = sh / (sh + sa)
    oh, ou = imp(pin.tot_over_price), imp(pin.tot_under_price)
    pin["p_under"] = ou / (oh + ou)
    pin = pin[key + ["fair_h", "sp_home_point", "cover_h", "tot_point", "p_under", "ml_home", "ml_away"]].rename(
        columns={"sp_home_point": "pin_pt", "tot_point": "pin_tot", "ml_home": "pin_mlh", "ml_away": "pin_mla"})
    # Pinnacle close per event
    pc = pin.sort_values("t").groupby("event_id").tail(1).set_index("event_id")
    mine = O[O.book.isin(MINE)]
    us_med = O.groupby(key).agg(us_pt=("sp_home_point", "median"), us_tot=("tot_point", "median")).reset_index()
    us_close = us_med.sort_values("t").groupby("event_id").tail(1).set_index("event_id")
    base = O.drop_duplicates("event_id").set_index("event_id")[["season", "ko"]]
    out = {"events": int(O.event_id.nunique()), "matched_results": int(len(res)), "rules": {}}
    resi = res.set_index("event_id")

    # ---------------- P1 / P2 moneylines
    best = []
    for side in ("home", "away"):
        col = f"ml_{side}"
        b = mine[mine[col].notna()].assign(d=lambda x: dec(x[col]))
        b = b.sort_values("d", ascending=False).groupby(key).head(1)[key + [col, "book", "d"]].rename(columns={col: "price"})
        b["side"] = side
        best.append(b)
    B = pd.concat(best).merge(pin, on=key, how="inner")
    B = B.merge(base, left_on="event_id", right_index=True)
    B = B[(B.ko - B.t) >= pd.Timedelta(hours=1)]
    B["p_fair"] = np.where(B.side == "home", B.fair_h, 1 - B.fair_h)
    mu = -B.pin_pt + SD_M * Phinv(B.cover_h)
    pw = Phi(mu / SD_M)
    B["p_spread"] = np.where(B.side == "home", pw, 1 - pw)
    B["ev_fair"] = B.p_fair * B.d - 1
    B["ev_spread"] = B.p_spread * B.d - 1
    B = B[(B.price >= -300) & (B.price <= 300)]

    def grade_ml(sel):
        sel = sel.sort_values("t").groupby("event_id").head(1).copy()
        sel = sel.join(pc[["fair_h"]].rename(columns={"fair_h": "close_h"}), on="event_id").join(resi, on="event_id", how="inner")
        cp = np.where(sel.side == "home", sel.close_h, 1 - sel.close_h)
        sel["clv"] = sel.d * cp - 1
        won = np.where(sel.side == "home", sel.margin > 0, sel.margin < 0)
        sel["res"] = np.where(sel.margin == 0, 0, np.where(won, 1, -1))
        sel["pnl"] = np.where(sel.res > 0, sel.d - 1, np.where(sel.res < 0, -1, 0))
        return sel

    out["rules"]["P1 soft-vs-Pinnacle moneyline (EV>=2%)"] = summarize("P1", grade_ml(B[B.ev_fair >= 0.02]), "clv", "price CLV vs Pinnacle no-vig close")
    out["rules"]["P2 moneyline vs Pinnacle spread (EV>=2%, >=0 vs ML)"] = summarize("P2", grade_ml(B[(B.ev_spread >= 0.02) & (B.ev_fair >= 0)]), "clv", "price CLV vs Pinnacle no-vig close")

    # ---------------- P3 early unders
    U = mine[mine.tot_under_price.notna() & mine.tot_point.notna()].assign(d=lambda x: dec(x.tot_under_price))
    U = U.sort_values(["tot_point", "d"], ascending=[False, False]).groupby(key).head(1)[key + ["tot_point", "tot_under_price", "d"]]
    U = U.merge(pin, on=key).merge(base, left_on="event_id", right_index=True)
    hrs = (U.ko - U.t).dt.total_seconds() / 3600
    U = U[(hrs >= 96) & (hrs <= 200) & (U.tot_point == U.pin_tot)]
    U = U.sort_values("t").groupby("event_id").head(1)
    U = U[U.p_under * U.d - 1 >= 0].copy()
    U = U.join(pc[["pin_tot", "p_under"]].rename(columns={"pin_tot": "c_tot", "p_under": "c_pu"}), on="event_id").join(resi, on="event_id", how="inner")
    mu_c = U.c_tot - SD_T * Phinv(U.c_pu)            # mean total implied by the Pinnacle close (P(total < c_tot) = c_pu)
    p_close = Phi((U.tot_point - mu_c) / SD_T)
    U["clv"] = U.d * p_close - 1
    U["res"] = np.sign(U.tot_point - U.total)
    U["pnl"] = np.where(U.res > 0, U.d - 1, np.where(U.res < 0, -1, 0))
    out["rules"]["P3 early unders (96-200 h, EV>=0 vs Pinnacle)"] = summarize("P3", U, "clv", "price CLV vs Pinnacle close (normal)")

    # ---------------- P4 follow Pinnacle early (spread)
    S = us_med.merge(pin[key + ["pin_pt"]], on=key).merge(base, left_on="event_id", right_index=True)
    S = S[(S.ko - S.t) >= pd.Timedelta(hours=48)].dropna(subset=["us_pt", "pin_pt"])
    S = S.sort_values("t").groupby("event_id").head(1)
    S = S[(S.pin_pt - S.us_pt).abs() >= 1.0].copy()
    S["side"] = np.where(S.pin_pt < S.us_pt, "home", "away")     # Pinnacle more favorable to home -> home is the value at US books
    bh = mine.assign(d=dec(mine.sp_home_price)).sort_values(["sp_home_point", "d"], ascending=[False, False]).groupby(key).head(1)[key + ["sp_home_point", "sp_home_price"]]
    ba = mine.assign(d=dec(mine.sp_away_price)).sort_values(["sp_away_point", "d"], ascending=[False, False]).groupby(key).head(1)[key + ["sp_away_point", "sp_away_price"]]
    S = S.merge(bh, on=key, how="left").merge(ba, on=key, how="left")
    S["pt"] = np.where(S.side == "home", S.sp_home_point, S.sp_away_point)
    S["price"] = np.where(S.side == "home", S.sp_home_price, S.sp_away_price)
    S = S.dropna(subset=["pt", "price"]).join(us_close[["us_pt"]].rename(columns={"us_pt": "c_pt"}), on="event_id").join(resi, on="event_id", how="inner")
    S["clv"] = np.where(S.side == "home", S.pt - S.c_pt, S.pt + S.c_pt)      # points gained vs the close for our side
    S["res"] = np.where(S.side == "home", S.margin + S.pt, -S.margin + S.pt)
    S["pnl"] = np.where(S.res > 0, dec(S.price) - 1, np.where(S.res < 0, -1, 0))
    out["rules"]["P4 follow Pinnacle early (>=1 pt off US median)"] = summarize("P4", S, "clv", "points vs US median close")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "price_screen.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
