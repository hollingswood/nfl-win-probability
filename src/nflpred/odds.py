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


# Pinnacle (sharpest book; via its public site, may lag slightly) and US exchanges / prediction
# markets. Requested by bookmaker key: up to 10 books cost the same as one region.
EXTRA_BOOKS = "pinnacle,kalshi,prophetx,polymarket,novig,betopenly"
EXCHANGES = {"kalshi", "prophetx", "polymarket", "novig", "betopenly"}


CREDITS: dict = {}


def _get(params: dict, timeout: float, url: str = URL):
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout) as r:
        for h in ("x-requests-remaining", "x-requests-used", "x-requests-last"):
            if r.headers.get(h) is not None:
                CREDITS[h] = r.headers.get(h)
        return json.load(r)


def fetch(api_key: str | None = None, timeout: float = 20) -> list[dict]:
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set")
    base = {"apiKey": key, "markets": "h2h,spreads,totals", "oddsFormat": "american"}
    # us + us2 = regulated US books (us2 adds e.g. ESPN BET, Fanatics, Hard Rock): 6 credits.
    events = _get({**base, "regions": "us,us2"}, timeout)
    # + Pinnacle and exchanges: 3 credits. Merged into the same events so snapshots keep everything.
    try:
        extra = {e["id"]: e for e in _get({**base, "bookmakers": EXTRA_BOOKS}, timeout)}
        for e in events:
            if e.get("id") in extra:
                e["bookmakers"] = e.get("bookmakers", []) + extra[e["id"]].get("bookmakers", [])
    except Exception as ex:
        print(f"odds: pinnacle/exchanges skipped ({ex})")
    return events


def load_allowed_books(path=None) -> set | None:
    from pathlib import Path
    path = path or Path(__file__).resolve().parents[2] / "my_books.json"
    try:
        return set(json.loads(Path(path).read_text())["allowed_books"])
    except Exception:
        return None


SHARP_BOOKS = {"lowvig", "betonlineag", "circasports", "bookmaker"}


def _book_detail(mk: dict, home: str, away: str) -> dict:
    """One book's moneyline [home, away] and main spread [home_point, home_price, away_price] (None if missing)."""
    out = {"ml": None, "sp": None}
    try:
        if "h2h" in mk:
            px = {o["name"]: o["price"] for o in mk["h2h"]["outcomes"]}
            if px.get(home) is not None and px.get(away) is not None:
                out["ml"] = [px[home], px[away]]
        if "spreads" in mk:
            sp = {o["name"]: o for o in mk["spreads"]["outcomes"]}
            ho, ao = sp.get(home), sp.get(away)
            if (ho and ao and ho.get("point") is not None and ho.get("price") is not None
                    and ao.get("price") is not None):
                out["sp"] = [ho["point"], ho["price"], ao["price"]]
    except Exception:
        pass
    return out


def summarize(events: list[dict], allowed: set | None = None) -> dict[tuple[str, str, str], dict]:
    """(home_abbr, away_abbr, kickoff date UTC) -> consensus + best prices.
    Consensus uses every book; best prices and per-book spreads only books in `allowed` (if given)."""
    out = {}
    for ev in events:
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        probs, spreads, best = [], [], {"home": None, "away": None}
        sharp_probs, pin_probs, best_ex = [], [], {"home": None, "away": None}
        ref3 = []  # moneyline v3 reference: Pinnacle + LowVig + BetOnline
        sharp_spreads = []  # moneyline v4: sharp books' spread + juice
        book_spreads = []
        by_book = {}  # grade v2 (grade_v2.py): raw ML / spread per book (exchanges left out, as in the research feed)
        for bk in ev.get("bookmakers", []):
            mk = {m["key"]: m for m in bk.get("markets", [])}
            if bk.get("key") and bk.get("key") not in EXCHANGES:
                by_book[bk["key"]] = _book_detail(mk, ev["home_team"], ev["away_team"])
            if "h2h" in mk:
                px = {o["name"]: o["price"] for o in mk["h2h"]["outcomes"]}
                if ev["home_team"] in px and ev["away_team"] in px:
                    h, a = _implied(px[ev["home_team"]]), _implied(px[ev["away_team"]])
                    probs.append(h / (h + a))
                    if bk.get("key") in SHARP_BOOKS:
                        sharp_probs.append(h / (h + a))
                    if bk.get("key") == "pinnacle":
                        pin_probs.append(h / (h + a))
                    if bk.get("key") in ("pinnacle", "lowvig", "betonlineag"):
                        ref3.append(h / (h + a))
                    if bk.get("key") in EXCHANGES:
                        for side, name in (("home", ev["home_team"]), ("away", ev["away_team"])):
                            if best_ex[side] is None or px[name] > best_ex[side]["price"]:
                                best_ex[side] = {"price": px[name], "book": bk.get("title", bk.get("key"))}
                    if allowed is None or bk.get("key") in allowed:  # (was a `continue` that also
                        # dropped non-allowed books from the consensus SPREAD; consensus = all books)
                        for side, name in (("home", ev["home_team"]), ("away", ev["away_team"])):
                            if best[side] is None or px[name] > best[side]["price"]:
                                best[side] = {"price": px[name], "book": bk.get("title", bk.get("key"))}
                                if bk.get("key"):
                                    best[side]["key"] = bk["key"]  # Odds API key (grade v2 book feature)
            if "spreads" in mk:
                sp = {o["name"]: o for o in mk["spreads"]["outcomes"]}
                ho, ao = sp.get(ev["home_team"]), sp.get(ev["away_team"])
                if ho and ho.get("point") is not None:
                    spreads.append(-ho["point"])  # home -3.5 => home margin +3.5
                if (ho and ao and ho.get("point") is not None and bk.get("key") in ("lowvig", "betonlineag")
                        and ho.get("price") is not None and ao.get("price") is not None):
                    sharp_spreads.append([ho["point"], ho["price"], ao["price"]])
                if (ho and ao and ho.get("point") is not None and ao.get("point") is not None
                        and (allowed is None or bk.get("key") in allowed)):
                    book_spreads.append({"book": bk.get("title", bk.get("key")),
                                         "home_point": ho["point"], "home_price": ho["price"],
                                         "away_point": ao["point"], "away_price": ao["price"]})
        if not probs:
            continue
        try:
            from . import totals as _T
            tot = _T.summarize_event(ev, allowed, SHARP_BOOKS, _T.default_dist())
        except Exception:
            tot = None
        out[(home, away, ev["commence_time"][:10])] = {
            "event_id": ev.get("id"),  # Odds API event id (per-event player-prop calls)
            "totals": tot,
            "books": len(probs),
            "consensus_home_prob": round(statistics.median(probs), 4),
            "sharp_home_prob": round(statistics.median(sharp_probs), 4) if sharp_probs else None,
            "sharp_books": len(sharp_probs),
            "pinnacle_home_prob": round(pin_probs[0], 4) if pin_probs else None,
            "sharp_spreads": sharp_spreads,
            "pin_sharp_home_prob": round(statistics.median(ref3), 4) if ref3 else None,
            "best_exchange_home_ml": best_ex["home"], "best_exchange_away_ml": best_ex["away"],
            "consensus_home_margin": round(statistics.median(spreads), 1) if spreads else None,
            "best_home_ml": best["home"], "best_away_ml": best["away"],
            "spreads_by_book": book_spreads,
            "by_book": by_book,
            "commence_time": ev["commence_time"],
        }
    return out


# ------------------------------------------------------------------------------- player props (per event)
PROPS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"


def fetch_event_props(event_id: str, markets: str = "player_receptions", api_key: str | None = None,
                      timeout: float = 20) -> dict:
    """One event's player props (regions=us, the research feed). Cost: markets x regions = ~1 credit per call
    for player_receptions. Raises if no key / the request fails."""
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set")
    params = {"apiKey": key, "regions": "us", "markets": markets, "oddsFormat": "american"}
    return _get(params, timeout, PROPS_URL.format(event_id=urllib.parse.quote(str(event_id))))


def _write_credits(history_dir: Path) -> None:
    if CREDITS:  # Odds API balance after the last call, readable in the repo
        (history_dir / "odds_credits.json").write_text(json.dumps(
            {"checked_at": datetime.now(timezone.utc).isoformat(timespec="minutes"), **CREDITS}, indent=1))


def save_props_snapshot(history_dir: Path, entries: list[dict], now: datetime | None = None) -> Path | None:
    """Raw per-event prop responses of one run -> history/props_<UTC timestamp>.json.gz (list of
    {event_id, game_id, kind, fetched_at, commence_time, response})."""
    if not entries:
        return None
    import gzip
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H%M")
    path = history_dir / f"props_{ts}.json.gz"
    old = []
    if path.exists():  # two runs in the same minute: keep both
        try:
            old = json.loads(gzip.open(path, "rt").read())
        except Exception:
            old = []
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(old + entries))
    _write_credits(history_dir)
    return path


def snapshot(history_dir: Path) -> dict | None:
    """Fetch, save raw snapshot, return summary. Returns None if no key / request fails."""
    try:
        events = fetch()
    except Exception as e:
        print(f"odds: skipped ({e})")
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    _write_credits(history_dir)
    import gzip  # compressed: hourly snapshots would otherwise bloat the repo
    with gzip.open(history_dir / f"odds_{ts}.json.gz", "wt") as f:
        f.write(json.dumps(events))
    return summarize(events, load_allowed_books())


def match(summary: dict, home: str, away: str, gameday) -> dict | None:
    """Find a game's odds; commence_time is UTC so a late ET kickoff can be the next UTC day."""
    import datetime as dt
    for d in (gameday, gameday + dt.timedelta(days=1)):
        k = (home, away, d.isoformat())
        if k in summary:
            return summary[k]
    return None
