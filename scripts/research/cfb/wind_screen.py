"""College wind-under screen. PRE-DECLARED 2026-10-03, written before any weather data was downloaded.

Weather = GFS forecast for the kickoff window (kickoff hour + 3 h, mean wind at 10 m, mph) AS IT STOOD 1 or 2 days
before (Open-Meteo Previous Runs API; data/cfb/weather/game_weather.json from cfbpred.weather). Recorded weather is
never used to pick bets (it leaks: the NFL wind lead died on exactly that).
Odds = our hourly historical snapshots (data/historical_odds/cfb + cfb_pin, 2021-2026).

W1  2-day-old forecast wind >= 15 mph, open-air venue -> UNDER at your best allowed book (highest number, then price)
    at the snapshot closest to 48 h before kickoff (window 36-60 h).
W2  1-day-old forecast wind >= 15 mph -> same, snapshot closest to 24 h before (window 12-36 h).
Graded: CLV = our decimal price x Pinnacle no-vig UNDER probability at the close, at our number (normal, sd 16.5 around
the close-implied mean; same method as price_screen2 P6), and ROI against the actual final total.
Pass (both must hold, 2 rules -> one-sided p < 0.025): >= 40 bets 2021-25, mean CLV > 0 at p < 0.025, ROI >= 0,
CLV positive in most seasons. 2026 reported separately as out-of-sample.
Information only (not rules): under rate at the Pinnacle closing total by forecast-wind bucket (does the close price wind?).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS  # noqa: E402

WIND = 15.0
RULES = {"W1 2-day forecast >= 15 mph, bet ~48 h out": ("d2", 36, 60, 48),
         "W2 1-day forecast >= 15 mph, bet ~24 h out": ("d1", 12, 36, 24)}


def event_games(O):
    """event_id -> CFBD game_id, final total (same matching as price_screen.results)."""
    ev = O.drop_duplicates("event_id")[["event_id", "season", "home", "away", "ko"]]
    out = []
    for s, e in ev.groupby("season"):
        g = PS.D.games([s])
        g = g[g.completed & g.total.notna()]
        m = PS.team_matcher(sorted(set(g.home) | set(g.away)))
        idx = {(r.home, r.away): r for r in g.itertuples()}
        for r in e.itertuples():
            gg = idx.get((m(r.home), m(r.away)))
            if gg is not None and abs((gg.start - r.ko).total_seconds()) < 36 * 3600:
                out.append({"event_id": r.event_id, "game_id": str(int(gg.game_id)), "season": int(s), "total": float(gg.total)})
    return pd.DataFrame(out).set_index("event_id")


def main():
    W = json.loads((PS.ROOT / "data" / "cfb" / "weather" / "game_weather.json").read_text())
    O, P = PS.load("cfb"), PS.load("cfb_pin")
    eg = event_games(O)
    wx = pd.DataFrame([{"game_id": k, **{f"{t}_wind": (v.get(t) or {}).get("wind_mph") for t in ("d0", "d1", "d2")}}
                       for k, v in W.items() if not v.get("dome")]).set_index("game_id")
    eg = eg.join(wx, on="game_id", how="inner")
    pin = P[["event_id", "t", "tot_point", "tot_over_price", "tot_under_price"]].dropna().copy()
    a, b = PS.imp(pin.tot_over_price), PS.imp(pin.tot_under_price)
    pin["q_over"] = a / (a + b)
    close = pin.sort_values("t").groupby("event_id").tail(1).set_index("event_id")[["tot_point", "q_over"]]
    close.columns = ["c_pt", "c_qo"]
    mine = O[O.book.isin(PS.MINE)][["event_id", "t", "ko", "season", "tot_point", "tot_under_price"]].dropna()
    mine = mine.assign(h=(mine.ko - mine.t).dt.total_seconds() / 3600)
    out = {"n_games_with_weather_and_odds": int(len(eg)), "wind_mph": WIND}
    for name, (vint, lo, hi, target) in RULES.items():
        q = eg[eg[f"{vint}_wind"] >= WIND]
        m = mine[mine.event_id.isin(q.index) & mine.h.between(lo, hi)].copy()
        m["dist"] = (m.h - target).abs()
        snap = m.sort_values("dist").groupby("event_id").t.first()
        m = m[m.t == m.event_id.map(snap)]
        m = m.sort_values(["tot_point", "tot_under_price"], ascending=False).groupby("event_id").head(1)
        M = m.join(close, on="event_id", how="inner").join(eg.drop(columns="season"), on="event_id", how="inner")
        M = M.rename(columns={"tot_point": "pt", "tot_under_price": "price"})
        M["d"] = PS.dec(M.price)
        mu_c = M.c_pt + PS.SD_T * PS.Phinv(M.c_qo)
        p_under_close = PS.Phi((M.pt - mu_c) / PS.SD_T)
        M["clv"] = M.d * p_under_close - 1
        M["res"] = M.pt - M.total
        M["pnl"] = np.where(M.res > 0, M.d - 1, np.where(M.res < 0, -1, 0))
        st = PS.summarize(name, M, "clv", "price CLV vs the Pinnacle close at our number")
        st["pass"] = bool(st.get("bets", 0) >= 40 and st.get("p", 1) < 0.025 and st.get("roi", -1) >= 0
                          and int(st["seasons_clv_positive"].split("/")[0]) > int(st["seasons_clv_positive"].split("/")[1]) / 2)
        st["avg_point_move_to_close"] = round(float((M.c_pt - M.pt).mean()), 2) if len(M) else None
        out[name] = st
    # information: does the closing total already price the forecast wind?
    info = eg.join(close, how="inner")
    info = info[info.season <= 2025]
    info["under"] = np.where(info.total < info.c_pt, 1.0, np.where(info.total > info.c_pt, 0.0, np.nan))
    buckets = pd.cut(info.d1_wind, [-1, 8, 12, 15, 20, 99], labels=["<8", "8-12", "12-15", "15-20", "20+"])
    out["info_close_under_rate_by_1day_wind"] = {str(k): {"games": int(g.under.notna().sum()), "under_rate": round(float(g.under.mean()), 3)}
                                                 for k, g in info.groupby(buckets, observed=True)}
    out["info_forecast_error_mph"] = {"d2_vs_latest_mean_abs": round(float((eg.d2_wind - eg.d0_wind).abs().mean()), 2),
                                      "d1_vs_latest_mean_abs": round(float((eg.d1_wind - eg.d0_wind).abs().mean()), 2)}
    (PS.OUT / "wind_screen.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: (v if not isinstance(v, dict) or "bets" not in v else
                          {x: v[x] for x in ("bets", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass") if x in v})
                      for k, v in out.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
