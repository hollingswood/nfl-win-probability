"""College price screen part 2 (pre-declared 2026-10-03 before results): the P1 idea on SPREADS and TOTALS.
P5 spreads: first snapshot (>= 1 h before kickoff) where your best spread price beats Pinnacle's no-vig cover
   probability AT THE SAME NUMBER by EV >= 2%. Graded CLV = price x Pinnacle no-vig cover prob at the close (same
   number; if Pinnacle moved, the normal approximation with sd 15.7 around the close-implied mean), and ATS/ROI.
P6 totals: same for overs/unders (sd 16.5).
Pass: one-sided p < 0.025 (2 rules) on mean CLV > 0, ROI not negative, most seasons positive. 2026 reported separately."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import price_screen as PS

def main():
    O, P = PS.load("cfb"), PS.load("cfb_pin")
    res = PS.results(O).set_index("event_id")
    key = ["event_id", "t"]
    mine = O[O.book.isin(PS.MINE)]
    base = O.drop_duplicates("event_id").set_index("event_id")[["season", "ko"]]
    out = {}
    for name, (pt_h, pr_h, pr_a, sd) in {"P5 spreads": ("sp_home_point", "sp_home_price", "sp_away_price", PS.SD_M),
                                         "P6 totals": ("tot_point", "tot_over_price", "tot_under_price", PS.SD_T)}.items():
        pin = P[key + [pt_h, pr_h, pr_a]].dropna().copy()
        a, b = PS.imp(pin[pr_h]), PS.imp(pin[pr_a])
        pin["q1"] = a / (a + b)                                    # no-vig prob of side 1 (home cover / over)
        pin = pin.rename(columns={pt_h: "pin_pt"})[key + ["pin_pt", "q1"]]
        close = pin.sort_values("t").groupby("event_id").tail(1).set_index("event_id").drop(columns="t")
        rows = []
        for side, price_col in ((1, pr_h), (2, pr_a)):
            m = mine[mine[pt_h].notna() & mine[price_col].notna()][key + [pt_h, price_col]].rename(columns={pt_h: "pt", price_col: "price"})
            m["d"] = PS.dec(m.price)
            m = m.merge(pin, on=key)
            m = m[m.pt == m.pin_pt]
            m["p"] = m.q1 if side == 1 else 1 - m.q1
            m["ev"] = m.p * m.d - 1
            m["side"] = side
            rows.append(m)
        M = pd.concat(rows).merge(base, left_on="event_id", right_index=True)
        M = M[((M.ko - M.t) >= pd.Timedelta(hours=1)) & (M.ev >= 0.02) & (M.price.between(-200, 200))]
        M = M.sort_values(["t", "ev"], ascending=[True, False]).groupby("event_id").head(1)
        M = M.join(close.rename(columns={"pin_pt": "c_pt", "q1": "c_q1"}), on="event_id").join(res, on="event_id", how="inner")
        # closing probability for our side at OUR number
        if name.startswith("P5"):
            mu_c = -M.c_pt + sd * PS.Phinv(M.c_q1)                  # home margin mean implied by the close
            p_home_cover = PS.Phi((mu_c + M.pt) / sd)                # P(margin + pt > 0)
            pc = np.where(M.side == 1, p_home_cover, 1 - PS.Phi((mu_c + M.pt) / sd))
            # away side: our pt is the HOME point stored? -> away rows used sp_home_point as the number; away covers if margin < -pt
            M["res"] = np.where(M.side == 1, M.margin + M.pt, -(M.margin + M.pt))
        else:
            mu_c = M.c_pt + sd * PS.Phinv(M.c_q1)                   # P(total > c_pt) = c_q1
            p_over = 1 - PS.Phi((M.pt - mu_c) / sd)
            pc = np.where(M.side == 1, p_over, 1 - p_over)
            M["res"] = np.where(M.side == 1, M.total - M.pt, M.pt - M.total)
        M["clv"] = M.d * pc - 1
        M["pnl"] = np.where(M.res > 0, M.d - 1, np.where(M.res < 0, -1, 0))
        st = PS.summarize(name, M, "clv", "price CLV vs the Pinnacle close at our number")
        st["pass"] = bool(st.get("p", 1) < 0.025 and st.get("roi", -1) >= 0 and int(st["seasons_clv_positive"].split("/")[0]) > int(st["seasons_clv_positive"].split("/")[1]) / 2)
        out[name] = st
    (PS.OUT / "price_screen2.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out.items():
        print(k, {x: v[x] for x in ("bets", "mean_clv", "p", "roi", "roi_se", "win_rate", "seasons_clv_positive", "pass")}, "2026:", v["2026_out_of_sample"])
        print("   ", v["by_season"])

if __name__ == "__main__":
    main()
