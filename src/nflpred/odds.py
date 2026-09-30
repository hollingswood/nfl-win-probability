"""Live multi-sportsbook odds from The Odds API (https://the-odds-api.com).

Needs env var ODDS_API_KEY (free tier: 500 requests/month; one call covers every NFL game,
so three runs a week uses ~15/month). Each run:
  * saves the raw snapshot to history/odds_<UTC timestamp>.json (the line at publish time, for CLV)
  * computes, per game, the consensus no-vig home win probability (median across books), the
    consensus spread, and the BEST available moneyline for each side and which book has it.
"""
from __future__ import annotations

import json
import os
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",  # 2020-21 names (historical odds)
}


def _implied(american: float) -> float:
    return -american / (-american + 100) if american < 0 else 100 / (american + 100)


def fetch(api_key: str | None = None, timeout: float = 20) -> list[dict]:
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set")
    # us + us2 covers the regulated US books (us2 adds e.g. ESPN BET, Fanatics, Hard Rock);
    # cost = markets x regions = 4 credits per call.
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us,us2", "markets": "h2h,spreads",
                                "oddsFormat": "american"})
    with urllib.request.urlopen(f"{URL}?{q}", timeout=timeout) as r:
        return json.load(r)


def load_allowed_books(path=None) -> set | None:
    from pathlib import Path
    path = path or Path(__file__).resolve().parents[2] / "my_books.json"
    try:
        return set(json.loads(Path(path).read_text())["allowed_books"])
    except Exception:
        return None


SHARP_BOOKS = {"lowvig", "betonlineag", "circasports", "bookmaker"}


def summarize(events: list[dict], allowed: set | None = None) -> dict[tuple[str, str, str], dict]:
    """(home_abbr, away_abbr, kickoff date UTC) -> consensus + best prices.
    Consensus uses every book; best prices and per-book spreads only books in `allowed` (if given)."""
    out = {}
    for ev in events:
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        probs, spreads, best = [], [], {"home": None, "away": None}
        sharp_probs = []
        book_spreads = []
        for bk in ev.get("bookmakers", []):
            mk = {m["key"]: m for m in bk.get("markets", [])}
            if "h2h" in mk:
                px = {o["name"]: o["price"] for o in mk["h2h"]["outcomes"]}
                if ev["home_team"] in px and ev["away_team"] in px:
                    h, a = _implied(px[ev["home_team"]]), _implied(px[ev["away_team"]])
                    probs.append(h / (h + a))
                    if bk.get("key") in SHARP_BOOKS:
                        sharp_probs.append(h / (h + a))
                    if allowed is not None and bk.get("key") not in allowed:
                        continue
                    for side, name in (("home", ev["home_team"]), ("away", ev["away_team"])):
                        if best[side] is None or px[name] > best[side]["price"]:
                            best[side] = {"price": px[name], "book": bk.get("title", bk.get("key"))}
            if "spreads" in mk:
                sp = {o["name"]: o for o in mk["spreads"]["outcomes"]}
                ho, ao = sp.get(ev["home_team"]), sp.get(ev["away_team"])
                if ho and ho.get("point") is not None:
                    spreads.append(-ho["point"])  # home -3.5 => home margin +3.5
                if (ho and ao and ho.get("point") is not None and ao.get("point") is not None
                        and (allowed is None or bk.get("key") in allowed)):
                    book_spreads.append({"book": bk.get("title", bk.get("key")),
                                         "home_point": ho["point"], "home_price": ho["price"],
                                         "away_point": ao["point"], "away_price": ao["price"]})
        if not probs:
            continue
        out[(home, away, ev["commence_time"][:10])] = {
            "books": len(probs),
            "consensus_home_prob": round(statistics.median(probs), 4),
            "sharp_home_prob": round(statistics.median(sharp_probs), 4) if sharp_probs else None,
            "sharp_books": len(sharp_probs),
            "consensus_home_margin": round(statistics.median(spreads), 1) if spreads else None,
            "best_home_ml": best["home"], "best_away_ml": best["away"],
            "spreads_by_book": book_spreads,
            "commence_time": ev["commence_time"],
        }
    return out


def snapshot(history_dir: Path) -> dict | None:
    """Fetch, save raw snapshot, return summary. Returns None if no key / request fails."""
    try:
        events = fetch()
    except Exception as e:
        print(f"odds: skipped ({e})")
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    (history_dir / f"odds_{ts}.json").write_text(json.dumps(events))
    return summarize(events, load_allowed_books())


def match(summary: dict, home: str, away: str, gameday) -> dict | None:
    """Find a game's odds; commence_time is UTC so a late ET kickoff can be the next UTC day."""
    import datetime as dt
    for d in (gameday, gameday + dt.timedelta(days=1)):
        k = (home, away, d.isoformat())
        if k in summary:
            return summary[k]
    return None
