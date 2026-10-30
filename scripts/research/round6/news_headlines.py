"""Round 6: does the tone of a team's headlines in the week before a game predict the result beyond the closing line?

Pre-registered in output/research/round6/news_headlines_prereg.json (committed before any headline was scored).
Stages:
  probe  (Actions) check that Google News RSS honours after:/before: date ranges
  fetch  (Actions) for every 2023-2025 game and both teams: Google News RSS search "<team name> NFL" between
         kickoff-7d and kickoff -> data/research/news_headlines/<season>.jsonl.gz (title, source, published)
  score  (local) VADER compound score of each title; per team-game: mean, share negative, count, using only items
         published >= 2 h before kickoff
  test   (local) dev 2023, holdout 2024-25, as pre-registered
No Claude credits are used.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "research" / "news_headlines"
UA = {"User-Agent": "Mozilla/5.0 (research; nfl-win-probability)"}
NAMES = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
         "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
         "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
         "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
         "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
         "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
         "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}


def search(q: str, after: str, before: str) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f'"{q}" after:{after} before:{before}', "hl": "en-US", "gl": "US", "ceid": "US:en"})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        root = ET.fromstring(r.read())
    out = []
    for it in root.iter("item"):
        pub = it.findtext("pubDate") or ""
        try:
            p = parsedate_to_datetime(pub).isoformat()
        except Exception:
            p = None
        out.append({"title": (it.findtext("title") or "").strip(), "source": it.findtext("source"), "published": p})
    return out


def games(seasons):
    from nflpred import data as D
    import pandas as pd
    g = D.load_schedules()
    g = g[g.season.isin(seasons) & g.home_score.notna()].copy()
    day = pd.to_datetime(g.gameday).dt.strftime("%Y-%m-%d")
    g["kick"] = pd.to_datetime(day + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g


def probe():
    res = {}
    for q, a, b in (("Kansas City Chiefs", "2024-10-01", "2024-10-08"), ("Detroit Lions", "2023-12-10", "2023-12-17")):
        items = search(q, a, b)
        pubs = sorted(i["published"] for i in items if i["published"])
        res[q] = {"n": len(items), "first": pubs[:1], "last": pubs[-1:], "sample": [i["title"] for i in items[:3]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def fetch(seasons, pause: float = 1.2):
    g = games(seasons)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        path = OUT / f"{s}.jsonl.gz"
        done = set()
        if path.exists():
            with gzip.open(path, "rt") as f:
                done = {(r["game_id"], r["team"]) for r in map(json.loads, f)}
        with gzip.open(path, "at") as f:
            for r in g[g.season == s].itertuples():
                for team in (r.home_team, r.away_team):
                    if (r.game_id, team) in done or team not in NAMES:
                        continue
                    a = (r.kick - timedelta(days=7)).strftime("%Y-%m-%d")
                    b = (r.kick + timedelta(days=1)).strftime("%Y-%m-%d")
                    for attempt in range(4):
                        try:
                            items = search(NAMES[team], a, b)
                            break
                        except Exception as e:
                            print("retry", r.game_id, team, e)
                            time.sleep(10 * (attempt + 1))
                    else:
                        items = None
                    if items is not None:
                        f.write(json.dumps({"game_id": r.game_id, "team": team, "kick": r.kick.isoformat(), "items": items}) + "\n")
                        f.flush()
                    time.sleep(pause)
        print("season", s, "done")


def score():
    """Per team-game tone features from titles published >= 2 h before kickoff (duplicates once)."""
    import pandas as pd
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import math
    va = SentimentIntensityAnalyzer()
    rows = []
    for f in sorted(OUT.glob("20*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                kick = pd.Timestamp(r["kick"])
                seen, comp = set(), []
                for it in r["items"]:
                    t = it.get("title") or ""
                    if not t or t in seen or not it.get("published"):
                        continue
                    if pd.Timestamp(it["published"]) > kick - pd.Timedelta(hours=2):
                        continue
                    seen.add(t)
                    comp.append(va.polarity_scores(t.rsplit(" - ", 1)[0])["compound"])   # drop the trailing source name
                rows.append({"game_id": r["game_id"], "team": r["team"], "n": len(comp),
                             "tone": sum(comp) / len(comp) if comp else 0.0,
                             "neg_share": sum(c <= -0.05 for c in comp) / len(comp) if comp else 0.0,
                             "volume": math.log1p(len(comp))})
    df = pd.DataFrame(rows).drop_duplicates(["game_id", "team"], keep="last")
    df.to_parquet(ROOT / "output" / "research" / "round6" / "news_headlines_scores.parquet")
    return df


def test():
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from nflpred import data as D
    pre = json.loads((ROOT / "output" / "research" / "round6" / "news_headlines_prereg.json").read_text())
    sc = pd.read_parquet(ROOT / "output" / "research" / "round6" / "news_headlines_scores.parquet")
    g = D.load_schedules()
    g = g[g.season.between(2023, 2025) & g.home_score.notna() & g.spread_line.notna()].copy()
    h = sc.rename(columns={c: f"h_{c}" for c in ("n", "tone", "neg_share", "volume")}).rename(columns={"team": "home_team"})
    a = sc.rename(columns={c: f"a_{c}" for c in ("n", "tone", "neg_share", "volume")}).rename(columns={"team": "away_team"})
    d = g.merge(h, on=["game_id", "home_team"]).merge(a, on=["game_id", "away_team"])
    d["cover"] = (d.home_score - d.away_score) - d.spread_line      # nflverse spread_line: + = home favoured by that many
    d["tot_res"] = (d.home_score + d.away_score) - d.total_line
    for k in ("tone", "neg_share", "volume"):
        d[f"diff_{k}"] = d[f"h_{k}"] - d[f"a_{k}"]
    d["sum_volume"] = d.h_volume + d.a_volume
    out = {"n_games": {str(s): int((d.season == s).sum()) for s in (2023, 2024, 2025)},
           "headlines_per_team_game": {"median": float(np.median(pd.concat([d.h_n, d.a_n]))), "share_zero": float((pd.concat([d.h_n, d.a_n]) == 0).mean())}}
    def ols(df, y, x):
        X = sm.add_constant(df[[x]].astype(float))
        m = sm.OLS(df[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": df["week"].astype(str) + df["season"].astype(str)})
        return {"n": int(len(df)), "slope": float(m.params[x]), "se": float(m.bse[x]), "t": float(m.tvalues[x]),
                "sd_x": float(df[x].std()), "effect_pts_per_sd": float(m.params[x] * df[x].std())}
    tests = {"T1_cover": ("cover", "diff_tone"), "T2_neg": ("cover", "diff_neg_share"), "T3_volume_total": ("tot_res", "sum_volume")}
    from math import erfc, sqrt
    res = {}
    for name, (y, x) in tests.items():
        dev, hold = ols(d[d.season == 2023], y, x), ols(d[d.season.isin([2024, 2025])], y, x)
        sign = 1 if dev["slope"] > 0 else -1
        p1 = 0.5 * erfc(sign * hold["t"] / sqrt(2))     # one-sided in the dev direction
        res[name] = {"dev_2023": dev, "holdout_2024_25": hold, "holdout_one_sided_p": p1,
                     "pass": bool(p1 < 0.025 and abs(hold["effect_pts_per_sd"]) >= 0.25 and np.sign(hold["slope"]) == sign)}
    out["tests"] = res
    out["T4_rule"] = "not run (only if T1 or T2 pass)" if not (res["T1_cover"]["pass"] or res["T2_neg"]["pass"]) else "see T4"
    (ROOT / "output" / "research" / "round6" / "news_headlines.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "probe":
        probe()
    elif stage == "fetch":
        fetch([int(x) for x in sys.argv[2].split(",")])
    elif stage == "score":
        print(score().describe())
    elif stage == "test":
        test()
