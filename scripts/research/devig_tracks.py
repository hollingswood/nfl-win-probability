"""Does Shin vig removal improve soft-book-vs-Pinnacle moneyline bets? (2026-10-04)
Same rule both ways: your-book or AZ-book moneyline with EV >= 2% vs Pinnacle's no-vig line at the same snapshot,
first qualifying snapshot per game. Fair price by multiplicative vs Shin. Graded by CLV under Pinnacle's last
pre-kickoff line using the SHIN close (better calibrated, devig_methods.md) and by realized ROI."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/research"), str(ROOT / "scripts/research/cfb")]
from nflpred import devig as DV, data as ND  # noqa
import price_screen as PS, shop_screen as SS, nfl_shop as NS  # noqa

def run(O, P, R, books, label, rng):
    P = P.dropna(subset=["ml_home", "ml_away"]).drop_duplicates(["event_id", "t"]).copy()
    for m in ("multiplicative", "shin"):
        P[m] = [DV.METHODS[m](h, a) for h, a in zip(P.ml_home, P.ml_away)]
    close = P.sort_values("t").groupby("event_id").tail(1).set_index("event_id")[["shin", "multiplicative"]].add_prefix("c_")
    Q = O[O.book.isin(books)].dropna(subset=["ml_home", "ml_away"]).merge(P[["event_id", "t", "multiplicative", "shin"]], on=["event_id", "t"])
    Q = Q[((Q.ko - Q.t) >= pd.Timedelta(hours=1)) & ((Q.ko - Q.t) <= pd.Timedelta(days=7))]
    out = {}
    rows = []
    for side in ("home", "away"):
        pr = Q.ml_home if side == "home" else Q.ml_away
        d = PS.dec(pr.to_numpy(float))
        for m in ("multiplicative", "shin"):
            p = Q[m] if side == "home" else 1 - Q[m]
            rows.append(Q[["event_id", "t", "season", "book"]].assign(side=side, price=pr.values, method=m, ev=p.values * d - 1))
    C = pd.concat(rows)
    C = C[C.price.between(*rng)]
    for m in ("multiplicative", "shin"):
        q = C[(C.method == m) & (C.ev >= 0.02)]
        ft = q.groupby("event_id").t.min()
        q = q[q.t == q.event_id.map(ft)].sort_values("ev", ascending=False).groupby("event_id").head(1)
        q = q.join(close, on="event_id", how="inner").join(R, on="event_id", how="inner")
        d = PS.dec(q.price.to_numpy(float))
        pc = np.where(q.side == "home", q.c_shin, 1 - q.c_shin)
        clv = d * pc - 1
        won = np.where(q.side == "home", q.margin > 0, q.margin < 0)
        pnl = np.where(q.margin == 0, 0, np.where(won, d - 1, -1))
        dog = q.price > 150
        out[m] = {"bets": int(len(q)), "clv_shin_close": round(float(clv.mean()), 4), "clv_se": round(float(clv.std() / math.sqrt(len(q))), 4),
                  "roi": round(float(pnl.mean()), 4), "roi_se": round(float(pnl.std() / math.sqrt(len(q))), 4),
                  "dogs_over_+150": int(dog.sum()), "dog_clv": round(float(clv[dog.to_numpy()].mean()), 4) if dog.any() else None,
                  "fav_clv": round(float(clv[~dog.to_numpy()].mean()), 4) if (~dog).any() else None}
    print(label, json.dumps(out)); return out

res = {}
sched = ND.load_schedules()
O, P = NS.load("dense"), NS.load("dense_pin"); P = P[P.book == "pinnacle"]
R = NS.results(O, sched)
for rng in ((-1000, 1000), (-200, 200)):
    res[f"NFL 2022-25 AZ books {rng}"] = run(O, P, R, NS.AZ, f"NFL {rng}", rng)
O, P = PS.load("cfb"), PS.load("cfb_pin")
R = SS.results_full(O); O = O[O.season <= 2025]
for rng in ((-1000, 1000), (-300, 300)):
    res[f"CFB 2021-25 AZ books {rng}"] = run(O, P, R, SS.AZ, f"CFB {rng}", rng)
(ROOT / "output/research/devig_tracks.json").write_text(json.dumps(res, indent=1))
