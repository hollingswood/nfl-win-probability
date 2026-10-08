"""EXPLORATORY (post-hoc, written 2026-10-07 AFTER seeing t1-t5 results). Nothing here can pass; it only describes
slices that stood out, so a future pre-registered test has something specific to check.

X1 Academy vs academy (Army/Navy/Air Force playing each other) UNDER the close: game list, by era, 2026 to date, and
   the Pinnacle closing total + under price 2021-2026 where the cfb_pin snapshots have the game.
X2 NFL 2024-2025 OVER in weeks 5-18 (the kickoff-rule "lag" may have shown up after week 4, not in weeks 1-4).
X3 College weeks 0-2 UNDER at the close, ALL totals, FBS v FBS only, by era and 2026 to date.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "cfb"))
import common as C  # noqa: E402


def x1(md, out):
    c = C.cfb_frame()
    s = c[c.home.isin(C.ACADEMIES) & c.away.isin(C.ACADEMIES) & c.total_close.notna()].copy()
    s["res"] = C.ou_result(s.total, s.total_close, "under")
    s = s.sort_values("start")
    # Pinnacle close
    pin = {}
    try:
        import price_screen as PS
        O = PS.load("cfb_pin")
        O = O[(O.book == "pinnacle") & O.tot_point.notna()]
        names = {"Army": "Army", "Navy": "Navy", "Air Force": "Air Force"}
        for t in names:
            pass
        O = O[O.home.str.startswith(("Army", "Navy", "Air Force")) & O.away.str.startswith(("Army", "Navy", "Air Force"))]
        last = O.sort_values("t").groupby("event_id").tail(1)
        for r in last.itertuples():
            pin[(r.ko.date(), r.home.split()[0])] = (r.tot_point, r.tot_under_price)
    except Exception as e:  # noqa: BLE001
        md.append(f"(Pinnacle load failed: {e})")
    rows = []
    for r in s.itertuples():
        k = None
        for dd in (0, -1, 1):
            key = ((r.start + pd.Timedelta(days=dd)).date(), r.home.split()[0])
            if key in pin:
                k = pin[key]
                break
        rows.append({"season": r.season, "week": r.week, "game": f"{r.away} @ {r.home}", "close": r.total_close,
                     "open": r.total_open, "actual": r.total, "res": int(r.res),
                     "pin_close": None if k is None else k[0], "pin_under_price": None if k is None else k[1]})
    df = pd.DataFrame(rows)
    out["X1_games"] = df.to_dict("records")
    hist = df[df.season <= 2025]
    md.append("### X1 academy vs academy UNDER (exploratory)\n")
    md.append(C.table([("2014-2025", C.grade(hist.res.values)),
                       *[(e, C.grade(d.res.values)) for e, d in hist.groupby(hist.season.map(C.era))],
                       ("2026 to date", C.grade(df[df.season == 2026].res.values))]))
    md.append(f"\nMean close {hist.close.mean():.1f}, mean actual {hist.actual.mean():.1f}, mean (actual - close) "
              f"{(hist.actual - hist.close).mean():+.1f}.")
    p = df[df.pin_close.notna()].copy()
    if len(p):
        p["pres"] = np.sign(p.pin_close - p.actual)
        g = C.grade(p.pres.values, p.pin_under_price.values)
        md.append(f"\nAt the Pinnacle close (actual under price), games found {len(p)}: {C.fmt(g)}; "
                  f"mean Pinnacle close {p.pin_close.mean():.1f} vs CFBD close {p.close.mean():.1f}.")
        out["X1_pinnacle"] = g
    md.append("\n| season | wk | game | open | close | Pinnacle close | actual | under |\n|---|---|---|---|---|---|---|---|")
    for r in df.itertuples():
        md.append(f"| {r.season} | {r.week} | {r.game} | {r.open if pd.notna(r.open) else ''} | {r.close} | "
                  f"{'' if pd.isna(r.pin_close) else r.pin_close} | {r.actual:.0f} | {'W' if r.res > 0 else 'L' if r.res < 0 else 'P'} |")


def x2(md, out):
    g = pd.read_parquet(C.ROOT / "data/raw/games.parquet")
    g = g[(g.game_type == "REG") & g.total_line.notna() & g.home_score.notna()].copy()
    g["res"] = C.ou_result(g.home_score + g.away_score, g.total_line, "over")
    rows = []
    for lab, m in [("2024 wk 5-18", (g.season == 2024) & (g.week >= 5)), ("2025 wk 5-18", (g.season == 2025) & (g.week >= 5)),
                   ("2024-25 wk 5-18", g.season.isin([2024, 2025]) & (g.week >= 5)),
                   ("2018-23 wk 5-18 (context)", g.season.between(2018, 2023) & (g.week >= 5)),
                   ("2026 wk 5 to date", (g.season == 2026) & (g.week >= 5))]:
        d = g[m]
        rows.append((lab, C.grade(d.res.values, d.over_odds.values)))
    out["X2"] = dict(rows)
    md.append("\n### X2 NFL OVER after week 4, 2024-25 (exploratory)\n")
    md.append(C.table(rows))


def x3(md, out):
    c = C.cfb_frame()
    c = c[(c.season_type == "regular") & (c.week <= 2) & c.total_close.notna() & (c.home_div == "fbs") & (c.away_div == "fbs")].copy()
    c["res"] = C.ou_result(c.total, c.total_close, "under")
    rows = [("2014-2025", C.grade(c[c.season <= 2025].res.values))]
    rows += [(e, C.grade(d.res.values)) for e, d in c.groupby(c.season.map(C.era))]
    rows += [(str(y), C.grade(d.res.values)) for y, d in c.groupby("season")]
    out["X3"] = dict(rows)
    md.append("\n### X3 college weeks 0-2 UNDER, all totals, FBS v FBS (exploratory)\n")
    md.append(C.table(rows))


def main():
    md, out = ["EXPLORATORY: written after seeing the pre-declared results; not evidence on its own.\n"], {}
    x1(md, out)
    x2(md, out)
    x3(md, out)
    C.save("x_exploratory", out)
    (C.OUT / "x_exploratory.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
