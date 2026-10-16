"""Round robin grid on the remaining pools (pre-registered: output/research/round5/round_robin_more_prereg.json, 2026-10-16).

NFL A+ legs are rebuilt with the frozen grade models (scripts/research/grade_v2.py, grade_v2_spread.py, grade_granular.py;
holdout flags set because those one-shot holdouts were already run) and cached as parquet next to this script's output.
Run: GRADEV2_HOLDOUT=I_HAVE_FROZEN GRADETOT_HOLDOUT=I_HAVE_FROZEN GRADE2SP_HOLDOUT=I_HAVE_FROZEN \
     PYTHONPATH=src:scripts:scripts/research python scripts/research/round5/round_robin_more.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "research"), str(ROOT / "scripts" / "research" / "cfb"),
                str(Path(__file__).parent)]
import parlays as PL  # noqa: E402
import round_robin_grid as RG  # noqa: E402

OUT = ROOT / "output" / "research" / "round5"
CACHE = OUT / "legs"
NY = "America/New_York"


def slate(ko: pd.Series) -> pd.Series:
    k = pd.to_datetime(ko, utc=True).dt.tz_convert(NY)
    return (k - pd.to_timedelta((k.dt.weekday - 1) % 7, unit="D")).dt.date     # Tuesday of the kickoff week


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    d = fn(); d.to_parquet(p); return d


def nfl_ml():
    import grade_v2 as G2
    fz = json.loads(G2.FROZEN.read_text()); m = G2.load_frozen_model(fz); th = fz["thresholds_pred_clv"]
    out = []
    for seas in (G2.DEV, G2.HOLD):
        C = G2.candidates(G2.table(seas)); D = G2.best_offers(C, m.predict(G2.featurize(C)), th)
        out.append(G2.first_with(D, D.pred >= th["A+"]))
    A = pd.concat(out)
    return pd.DataFrame({"src": "nfl_aplus_ml", "game": A.game_id, "fam": "side", "t": A.requested_ts, "ko": A.kick,
                         "season": A.season, "d": A.dec, "o": np.sign(A.pnl), "clv": A.clv, "edge": A.pred})


def nfl_spread():
    import grade_v2_spread as S
    out = []
    for seas in (S.DEV, S.HOLD):
        X = S.table(seas)
        X = X[X.clv.notna() & X.price.between(-145, 125)].sort_values("v1_edge", ascending=False).drop_duplicates(["game_id", "requested_ts"])
        X["v1"] = S.v1_letter(S.v1_score(X))
        out.append(S.first_with(X, X.v1 == "A+"))
    A = pd.concat(out)
    return pd.DataFrame({"src": "nfl_aplus_spread", "game": A.game_id, "fam": "side", "t": A.requested_ts, "ko": A.kick,
                         "season": A.season, "d": A.dec, "o": np.sign(A.pnl), "clv": A.clv, "edge": A.v1_edge})


def nfl_totals():
    import grade_granular as GG
    fz = json.loads(GG.FROZEN.read_text()); m = GG.load_frozen_tmodel(fz); th = fz["thresholds_pred_clv"]
    out = []
    for seas in (GG.DEV, GG.HOLD):
        C = GG.tcandidates(GG.totals_table(seas)); C["pred"] = m.predict(GG.tfeaturize(C))
        D = GG.best_per_snapshot(C, C.pred.values); out.append(GG.first_with(D, D.pred >= th["A+"]))
    A = pd.concat(out)
    ko = pd.to_datetime(A.requested_ts, utc=True) + pd.to_timedelta(A.hours_before, unit="h")
    return pd.DataFrame({"src": "nfl_aplus_totals", "game": A.game_id, "fam": "total", "t": A.requested_ts, "ko": ko,
                         "season": A.season, "d": A.dec, "o": np.sign(A.pnl), "clv": A.clv, "edge": A.pred})


def college_shop():
    d = PL.edge_legs()
    return pd.DataFrame({"src": np.where(d.ev >= 0.04, "college_aplus", "college_shop_A"), "game": d.event_id,
                         "fam": np.where(d.market == "total", "total", "side"), "t": d.t, "ko": d.ko, "season": d.season,
                         "d": d.d, "o": d.o, "clv": d.clv, "edge": d.ev})


def college_ml():
    """College moneyline track v3 on history (P1 or P2, first qualifying snapshot per game), price_screen.py logic."""
    import price_screen as PS
    O, P = PS.load("cfb"), PS.load("cfb_pin")
    res = PS.results(O).set_index("event_id")
    key = ["event_id", "t"]
    ph, pa = PS.imp(P.ml_home), PS.imp(P.ml_away)
    sh, sa = PS.imp(P.sp_home_price), PS.imp(P.sp_away_price)
    pin = P.assign(fair_h=ph / (ph + pa), cover_h=sh / (sh + sa))[key + ["fair_h", "sp_home_point", "cover_h"]].rename(columns={"sp_home_point": "pin_pt"})
    pc = pin.sort_values("t").groupby("event_id").tail(1).set_index("event_id")
    mine = O[O.book.isin(PS.MINE)]
    base = O.drop_duplicates("event_id").set_index("event_id")[["season", "ko"]]
    best = []
    for side in ("home", "away"):
        col = f"ml_{side}"
        b = mine[mine[col].notna()].assign(d=lambda x: PS.dec(x[col]))
        b = b.sort_values("d", ascending=False).groupby(key).head(1)[key + [col, "book", "d"]].rename(columns={col: "price"})
        best.append(b.assign(side=side))
    B = pd.concat(best).merge(pin, on=key).merge(base, left_on="event_id", right_index=True)
    B = B[((B.ko - B.t) >= pd.Timedelta(hours=1)) & B.price.between(-300, 300)]
    B["p_fair"] = np.where(B.side == "home", B.fair_h, 1 - B.fair_h)
    mu = -B.pin_pt + PS.SD_M * PS.Phinv(B.cover_h)
    pw = PS.Phi(mu / PS.SD_M)
    B["p_spread"] = np.where(B.side == "home", pw, 1 - pw)
    B["ev_fair"], B["ev_spread"] = B.p_fair * B.d - 1, B.p_spread * B.d - 1
    q = B[(B.ev_fair >= 0.02) | ((B.ev_spread >= 0.02) & (B.ev_fair >= 0))].sort_values("t").groupby("event_id").head(1)
    q = q.join(pc[["fair_h"]].rename(columns={"fair_h": "close_h"}), on="event_id").join(res, on="event_id", how="inner")
    cp = np.where(q.side == "home", q.close_h, 1 - q.close_h)
    won = np.where(q.side == "home", q.margin > 0, q.margin < 0)
    return pd.DataFrame({"src": "college_ml", "game": q.event_id, "fam": "side", "t": q.t, "ko": q.ko, "season": q.season,
                         "d": q.d, "o": np.where(q.margin == 0, 0, np.where(won, 1, -1)), "clv": q.d * cp - 1,
                         "edge": np.maximum(q.ev_fair, q.ev_spread)})


POOLS = {
    "1 NFL A+ (ML + spread + totals)": ["nfl_aplus_ml", "nfl_aplus_spread", "nfl_aplus_totals"],
    "2 College A+ only": ["college_aplus"],
    "3 College moneyline track": ["college_ml"],
    "4 All college tracks": ["college_aplus", "college_shop_A", "college_ml"],
    "5 All A+ (college + NFL)": ["college_aplus", "nfl_aplus_ml", "nfl_aplus_spread", "nfl_aplus_totals"],
    "6 Everything (all college + NFL A+)": ["college_aplus", "college_shop_A", "college_ml", "nfl_aplus_ml", "nfl_aplus_spread", "nfl_aplus_totals"],
}


def pool(L: pd.DataFrame, srcs) -> pd.DataFrame:
    x = L[L.src.isin(srcs) & L.season.between(2021, 2025)].copy()
    x["t"] = pd.to_datetime(x.t, utc=True)
    x = x.sort_values("t").drop_duplicates(["game", "fam"])            # one leg per game per market family, earliest
    x["slate"], x["rank"], x["ev_id"] = slate(x.ko), -x.edge, x.game
    return x.dropna(subset=["d", "clv"])


def ordinary_nfl() -> pd.DataFrame:
    c = PL.closing_legs()["nfl"].copy()
    rng = np.random.default_rng(7)
    c["o"] = np.sign(c.res.values * (rng.integers(0, 2, len(c)) * 2 - 1)).astype(int)
    c["d"], c["clv"], c["rank"] = PL.DEC110, -0.0455, rng.random(len(c))
    c["slate"], c["ev_id"] = c.season.astype(str) + "-" + c.wk.astype(str), c.game_id
    return c


def main():
    L = pd.concat([cached("nfl_aplus_ml", nfl_ml), cached("nfl_aplus_spread", nfl_spread), cached("nfl_aplus_totals", nfl_totals),
                   cached("college_shop", college_shop), cached("college_ml", college_ml)], ignore_index=True)
    out = {"prereg": "output/research/round5/round_robin_more_prereg.json", "pools": {}}
    for name, srcs in POOLS.items():
        x = pool(L, srcs)
        info = {"legs": int(len(x)), "by_source": x.src.value_counts().to_dict(),
                "legs_per_week_median": float(x.groupby("slate").size().median()),
                "singles_all_legs": {lab: {"bets": int(x.season.isin(y).sum()), "roi": round(float(np.where(x.o > 0, x.d - 1, np.where(x.o < 0, -1, 0))[x.season.isin(y)].mean()), 4),
                                           "clv": round(float(x.clv[x.season.isin(y)].mean()), 4)} for lab, y in RG.PERIODS.items()}}
        out["pools"][name] = {"info": info, **RG.grid(x, RG.NS)}
        print(name, info, "passes:", out["pools"][name]["passes"])
    out["pools"]["7 Ordinary NFL legs"] = RG.grid(ordinary_nfl(), (4, 6, 8, 10))
    print("7 ordinary NFL passes:", out["pools"]["7 Ordinary NFL legs"]["passes"])
    (OUT / "round_robin_more.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
