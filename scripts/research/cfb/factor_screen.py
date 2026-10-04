"""College football situational factor screen (pre-declared 2026-10-03 before results), FBS vs FBS, 2014-2025.
Each factor is a fixed betting rule vs the CLOSING spread (CFBD median of providers; -110 assumed: no prices).
Pass: one-sided p < 0.005 (10 rules, Bonferroni) on cover rate > 52.38%, AND positive (> 52.38%) in most seasons,
AND not negative in 2022-25 (recent years).

F1 long trip: away team travels >= 1,500 miles to a non-neutral game -> bet HOME
F2 body clock: away team's home time zone is >= 2 hours WEST of the game and kickoff is before 1:00 pm local -> bet HOME
F3 altitude: home stadium >= 1,500 m and the away team's stadium < 600 m -> bet HOME
F4 rest edge: one team has >= 4 more days of rest than the other (bye vs normal week) -> bet the RESTED team
F5 letdown: team that won outright as a 10+ point underdog in its previous game -> bet AGAINST it
F6 in-state conference game -> bet the UNDERDOG
F7 huge favorites: closing spread >= 21 -> bet the UNDERDOG
F8 home underdogs -> bet the home dog
F9 late-season small time-zone disadvantage (Coleman 2017): weeks 9+, underdog whose home is exactly 1 hour WEST of
   the game -> bet that UNDERDOG
F10 preseason prior (NFL S6 analog): weeks 2-6, preseason rating (0.69 x last season + 3.35 x talent z-score, as in
   the ratings model) differs from the closing spread by >= 7 points -> bet the side the prior favors
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
from cfbpred import data as D  # noqa: E402

YEARS = range(2014, 2026)
OUT = ROOT / "output" / "research" / "cfb"
TZ_OFF = {"America/New_York": -5, "America/Detroit": -5, "America/Indiana/Indianapolis": -5, "America/Kentucky/Louisville": -5,
          "America/Chicago": -6, "America/Denver": -7, "America/Boise": -7, "America/Phoenix": -7, "America/Los_Angeles": -8,
          "Pacific/Honolulu": -10, "America/Indiana/Knox": -6, "America/Indiana/Tell_City": -6, "America/Menominee": -6,
          "America/North_Dakota/Center": -6, "America/Anchorage": -9}


def hav(lat1, lon1, lat2, lon2):
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def locations():
    loc = {}
    for y in YEARS:
        for t in D._load(f"teams_fbs_{y}.json.gz"):
            L = t.get("location") or {}
            if L.get("latitude") is not None:
                loc[(y, t["school"])] = {"lat": float(L["latitude"]), "lon": float(L["longitude"]),
                                         "elev": float(L["elevation"]) if L.get("elevation") not in (None, "") else np.nan,
                                         "tz": TZ_OFF.get(L.get("timezone"), np.nan), "state": L.get("state")}
    return loc


def main():
    g = D.games(YEARS)
    lines = D.lines(YEARS)
    g = g.merge(lines[["game_id", "spread_close"]], on="game_id", how="left")
    g = g.sort_values("start")
    # rest days and previous-game upset flags per team
    long = pd.concat([g[["game_id", "season", "start", "home", "margin", "spread_close"]].rename(columns={"home": "team"}).assign(sign=1),
                      g[["game_id", "season", "start", "away", "margin", "spread_close"]].rename(columns={"away": "team"}).assign(sign=-1)])
    long = long.sort_values("start")
    long["prev_start"] = long.groupby(["season", "team"]).start.shift(1)
    long["team_margin"] = long.sign * long.margin
    long["team_line"] = long.sign * -long.spread_close         # expected team margin (spread_close: + = home underdog)
    long["upset_win"] = (long.team_margin > 0) & (long.team_line <= -10)
    long["prev_upset"] = long.groupby(["season", "team"]).upset_win.shift(1).fillna(False)
    rest = long.set_index(["game_id", "team"])
    loc = locations()
    F = g[(g.home_div == "fbs") & (g.away_div == "fbs") & g.margin.notna() & g.spread_close.notna()].copy()
    F["v"] = -F.spread_close                                   # market home margin
    F["res"] = F.margin - F.v                                  # > 0 = home covered
    def L(y, t, k):
        return (loc.get((y, t)) or {}).get(k, np.nan)
    for k in ("lat", "lon", "elev", "tz"):
        F[f"h_{k}"] = [L(y, t, k) for y, t in zip(F.season, F.home)]
        F[f"a_{k}"] = [L(y, t, k) for y, t in zip(F.season, F.away)]
    F["h_state"] = [(loc.get((y, t)) or {}).get("state") for y, t in zip(F.season, F.home)]
    F["a_state"] = [(loc.get((y, t)) or {}).get("state") for y, t in zip(F.season, F.away)]
    F["miles"] = hav(F.a_lat, F.a_lon, F.h_lat, F.h_lon)
    F["local_hour"] = (F.start.dt.hour + F.h_tz.fillna(-5)) % 24
    F["rest_h"] = [(s - rest.loc[(gid, t), "prev_start"]).days if pd.notna(rest.loc[(gid, t), "prev_start"]) else np.nan
                   for gid, t, s in zip(F.game_id, F.home, F.start)]
    F["rest_a"] = [(s - rest.loc[(gid, t), "prev_start"]).days if pd.notna(rest.loc[(gid, t), "prev_start"]) else np.nan
                   for gid, t, s in zip(F.game_id, F.away, F.start)]
    F["prev_upset_h"] = [bool(rest.loc[(gid, t), "prev_upset"]) for gid, t in zip(F.game_id, F.home)]
    F["prev_upset_a"] = [bool(rest.loc[(gid, t), "prev_upset"]) for gid, t in zip(F.game_id, F.away)]
    # preseason prior (F10): last season's final ridge rating + talent, as in fast.py
    import fast
    gg, tal, ret = fast.prep()
    final = {}
    pr = fast.run(gg, tal, ret, last_seasons=(2013 if 2013 in set(gg.season) else 2014, 2025))  # warms the chain
    tz = tal.copy(); tz["z"] = tz.groupby("season").talent.transform(lambda x: (x - x.mean()) / x.std())
    tzd = {(r.season, r.team): r.z for r in tz.itertuples()}
    # rebuild end-of-season ratings from week-1 predictions is not exposed; use the model's week-1 numbers as the prior
    wk1 = F[F.week <= 1]
    pr = pr.set_index("game_id").m_pred
    F["prior_margin"] = F.game_id.map(pr)
    first_slot = F.groupby("season").week.transform("min")
    # prior = the model's preseason-only prediction is only available for week-1 games; approximate per-game prior with
    # the walk-forward prediction made BEFORE any games of that season are known is not stored, so F10 uses the week-1
    # prediction machinery: rating difference at the start of the season.
    rules = {}

    def bet(mask, side_home):
        """side_home: boolean Series, True = bet home."""
        d = F[mask].copy(); sh = side_home[mask]
        r = np.where(sh, d.res, -d.res)
        d["r"] = r
        c, l = int((r > 0).sum()), int((r < 0).sum()); n = c + l
        z = (c - n * 0.5238) / math.sqrt(n * 0.5238 * 0.4762) if n else 0
        per = d.groupby("season").r.apply(lambda x: (x > 0).sum() / max(1, (x != 0).sum()))
        recent = d[d.season >= 2022].r
        rc = (recent > 0).sum() / max(1, (recent != 0).sum())
        out = {"bets": n, "cover": round(c / max(1, n), 3), "p": round(0.5 * math.erfc(z / math.sqrt(2)), 5),
               "roi_at_-110": round((c * 0.9091 - l) / max(1, n), 4),
               "seasons_above_52.4": f"{int((per > 0.5238).sum())}/{len(per)}", "cover_2022_25": round(float(rc), 3),
               "by_season": {int(k): round(float(v), 3) for k, v in per.items()}}
        out["pass"] = bool(out["p"] < 0.005 and (per > 0.5238).sum() > len(per) / 2 and rc >= 0.5238)
        return out
    home = pd.Series(True, index=F.index)
    nn = ~F.neutral
    rules["F1 long trip >= 1500 mi -> home"] = bet(nn & (F.miles >= 1500), home)
    rules["F2 body clock: away >= 2 h west, kickoff < 1pm local -> home"] = bet(nn & ((F.h_tz - F.a_tz) >= 2) & (F.local_hour < 13), home)
    rules["F3 altitude >= 1500 m vs away < 600 m -> home"] = bet(nn & (F.h_elev >= 1500) & (F.a_elev < 600), home)
    rd = F.rest_h - F.rest_a
    rules["F4 rest edge >= 4 days -> rested team"] = bet(rd.abs() >= 4, rd > 0)
    up = F.prev_upset_h ^ F.prev_upset_a
    rules["F5 letdown after 10+ pt upset win -> fade"] = bet(up, F.prev_upset_a)
    instate = F.conf_game & (F.h_state == F.a_state) & F.h_state.notna()
    rules["F6 in-state conference game -> underdog"] = bet(instate & (F.v != 0), F.v < 0)
    rules["F7 favorite by 21+ -> underdog"] = bet(F.v.abs() >= 21, F.v < 0)
    rules["F8 home underdog"] = bet(F.v < 0, home)
    late = F.week >= 9
    dog_h = F.v < 0
    one_w_away = nn & ((F.h_tz - F.a_tz) == 1)       # away team's home is 1 h west of the game
    one_w_home = pd.Series(False, index=F.index)      # home team plays at home: no disadvantage
    rules["F9 late-season dog 1 h west of home -> that dog"] = bet(late & one_w_away & ~dog_h & (F.v != 0), pd.Series(False, index=F.index))
    # F10: preseason prior: use the margin model's prediction for weeks 2-6 built only from the preseason prior is not
    # stored separately; we use the season's FIRST walk-forward prediction for each team pair? Not available -> computed
    # directly: prior rating = 0.69 * last season final rating diff + 3.35 * talent z diff (+ HFA 2.5).
    lastfinal = {}
    allp = fast.run(gg, tal, ret, last_seasons=(2014, 2025))
    gx = gg.merge(allp, on="game_id")
    # season-end rating proxy: average of each team's last 3 walk-forward margins vs opponents is noisy; use the
    # ridge-fit end ratings by refitting quickly per season below
    for s in range(2014, 2025):
        S = gg[(gg.season == s) & gg.completed & gg.margin.notna()]
        teams = sorted(set(S.home) | set(S.away)); ix = {t: i for i, t in enumerate(teams)}
        X = np.zeros((len(S), len(teams) + 1))
        for i, r in enumerate(S.itertuples()):
            X[i, ix[r.home]] = 1; X[i, ix[r.away]] = -1; X[i, -1] = 0 if r.neutral else 1
        A = X.T @ X + 4 * np.eye(X.shape[1]); A[-1, -1] -= 4
        b = np.linalg.solve(A, X.T @ S.margin.clip(-28, 28).to_numpy())
        for t, i in ix.items():
            lastfinal[(s, t)] = b[i]
    def prior(y, t):
        return 0.69 * lastfinal.get((y - 1, t), np.nan) + 3.35 * tzd.get((y, t), 0.0)
    F["prior_m"] = [2.5 * (0 if n else 1) + prior(y, h) - prior(y, a) for y, h, a, n in zip(F.season, F.home, F.away, F.neutral)]
    gap = F.prior_m - F.v
    rules["F10 preseason prior wk2-6 |prior-close|>=7"] = bet(F.week.between(2, 6) & (gap.abs() >= 7) & gap.notna(), gap > 0)
    out = {"games": int(len(F)), "rules": rules,
           "note": "CFBD closing spreads have no prices: -110 assumed (break-even 52.38%)."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "factor_screen.json").write_text(json.dumps(out, indent=1))
    for k, v in rules.items():
        print(f"{k:58s} {v['bets']:5d} bets  cover {v['cover']:.3f}  p {v['p']:.4f}  seasons {v['seasons_above_52.4']}  2022-25 {v['cover_2022_25']:.3f}  {'PASS' if v['pass'] else ''}")


if __name__ == "__main__":
    main()
