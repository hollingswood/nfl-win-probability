"""Score logged published picks (history/picks_log.jsonl) and run the pre-registered hot-pickers test
(picks_hot_rules.json). Logging and scoring only; nothing bets.

Each pick seen before kickoff is matched to its game (NFL: nflverse schedule; college: CFBD) and graded at the CLOSING
line: spreads and totals 1 unit at -110 (pushes don't count), moneylines straight up and in units at the closing price.
'Line value' = points the market moved toward the pick between the picker's stated number and the close.
Writes history/picks_scores.json: per-picker records, the hot list for the coming week, and the hot test so far.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import picks_check as PC

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
RULES = ROOT / "picks_hot_rules.json"
NICK = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers", "CHI": "Bears",
        "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
        "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs", "LV": "Raiders", "LAC": "Chargers", "LA": "Rams",
        "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
        "PHI": "Eagles", "PIT": "Steelers", "SF": "49ers", "SEA": "Seahawks", "TB": "Buccaneers", "TEN": "Titans",
        "WAS": "Commanders"}
WIN_110 = 100 / 110


def load_rules() -> dict:
    return json.loads(RULES.read_text())


def nfl_abbr(name: str) -> str | None:
    n = (name or "").lower()
    hits = [a for a, nick in NICK.items() if nick.lower() in n]
    if len(hits) == 1:
        return hits[0]
    if (name or "").upper() in NICK:
        return name.upper()
    return None


NFL_CITY = {"arizona": "ARI", "atlanta": "ATL", "baltimore": "BAL", "buffalo": "BUF", "carolina": "CAR", "chicago": "CHI",
            "cincinnati": "CIN", "cleveland": "CLE", "dallas": "DAL", "denver": "DEN", "detroit": "DET", "green bay": "GB",
            "houston": "HOU", "indianapolis": "IND", "jacksonville": "JAX", "kansas city": "KC", "las vegas": "LV",
            "miami": "MIA", "minnesota": "MIN", "new england": "NE", "new orleans": "NO", "philadelphia": "PHI",
            "pittsburgh": "PIT", "san francisco": "SF", "seattle": "SEA", "tampa bay": "TB", "tennessee": "TEN",
            "washington": "WAS"}


def nfl_team(name: str) -> str | None:
    """NFL team from a nickname ('Chiefs'), full name, abbreviation, or a one-team city ('Washington')."""
    a = nfl_abbr(name)
    if a:
        return a
    n = re.sub(r"[^a-z ]", "", (name or "").lower()).strip()
    return NFL_CITY.get(n)


# college names as writers use them -> the schedule's (CFBD) spelling, checked before prefix matching
CFB_ALIAS = {"mizzou": "missouri", "pitt": "pittsburgh", "brighamyoung": "byu", "brighamyounguniversity": "byu",
             "southerncal": "usc", "southerncalifornia": "usc", "cal": "california", "fiu": "floridainternational",
             "fau": "floridaatlantic", "mississippi": "olemiss", "olemissrebels": "olemiss", "uconn": "uconn",
             "umass": "massachusetts", "louisianamonroe": "ulmonroe", "ulm": "ulmonroe", "appalachianstate": "appstate",
             "southernmississippi": "southernmiss", "miamiflorida": "miami", "miamifl": "miami", "miamiohio": "miamioh",
             "hawaii": "hawaii", "sanjosestate": "sanjosestate", "texasamaggies": "texasam", "utep": "utep",
             "louisianaragincajuns": "louisiana", "ncstate": "ncstate", "northcarolinastate": "ncstate"}
# what may follow a school's name when the rest is a mascot; anything that makes it a DIFFERENT school is refused
OTHER_SCHOOL = ("state", "st", "tech", "am", "international", "atlantic", "christian", "southern", "northern", "eastern",
                "western", "central", "oh", "fl", "ohio", "florida", "carolina", "kentucky", "illinois", "michigan",
                "texas", "georgia", "colorado", "virginia", "washington", "mississippi", "tennessee", "alabama", "arkansas",
                "lafayette", "monroe", "dakota", "methodist", "baptist", "intl")


def cfb_matcher(teams: list[str]):
    """Strict college matcher for picks. The odds-feed matcher (cfbpred.pipeline.team_matcher) takes the first schedule
    name that prefixes the input, which turned 'Michigan St.' into Michigan, 'Ohio St.' into Ohio and 'Southern Cal' into
    Southern. Here 'St.' means State, writer nicknames are mapped, and a prefix only counts when the rest of the name is
    not the start of another school's name."""
    import unicodedata
    def norm(s):
        s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
        s = re.sub(r"\bst\.?(?=\s|$)", "state", s)
        s = re.sub(r"\(([a-z]+)\)", r" \1", s)
        return re.sub(r"[^a-z0-9]", "", s.replace("&", "a"))
    keys = sorted(((norm(t), t) for t in teams), key=lambda x: -len(x[0]))
    exact = {k: t for k, t in keys}
    def match(name: str) -> str | None:
        n = norm(name)
        if not n or n in ("unknown", "tbd", "none"):
            return None
        for a in sorted(CFB_ALIAS, key=len, reverse=True):
            if n == a or (n.startswith(a) and len(n) > len(a) and not n[len(a):].startswith(OTHER_SCHOOL)
                          and a not in exact):
                n = CFB_ALIAS[a] + n[len(a):]
                break
        if n in exact:
            return exact[n]
        for k, t in keys:
            if n.startswith(k) and not n[len(k):].startswith(OTHER_SCHOOL):
                return t
        return None
    return match


def _dec(a) -> float | None:
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    if math.isnan(a) or a == 0:
        return None
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def nfl_games() -> pd.DataFrame:
    from . import data
    try:
        g = data.load_schedules(refresh=True)      # nflverse schedule with this week's results
    except Exception:
        g = data.load_schedules()
    g = g[g.season >= 2026].copy()
    day = pd.to_datetime(g.gameday).dt.strftime("%Y-%m-%d")
    g["kick"] = pd.to_datetime(day + " " + g.gametime.fillna("13:00")).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    return g


def cfb_games() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from cfbpred import data as D
    yrs = sorted({datetime.now(timezone.utc).year - 1, datetime.now(timezone.utc).year})
    g = D.games(yrs)
    L = D.lines(yrs, prefer=D.PREFERRED_PROVIDERS)
    g = g.merge(L, on="game_id", how="left")
    g["kick"] = pd.to_datetime(g.start, utc=True)
    return g


def _game(df: pd.DataFrame, hc: str, ac: str, h: str | None, a: str | None, seen: datetime, published=None):
    """The game a pick is about.
    Both teams known: their meeting from 4 days before we saw the pick (post-game re-reads then come out as 'first seen
    after kickoff', not 'no game') to 10 days after.
    One team known (the writer left the opponent out): that team's games from 12 h before the ARTICLE was published to
    6 days after; used only when exactly one game fits, so a re-read after the game can't land on next week's game.
    -> schedule row, or the reason there is none."""
    t = pd.Timestamp(seen)
    if h and a:
        win = df[(df.kick > t - timedelta(days=4)) & (df.kick < t + timedelta(days=10))]
        g = win[((win[hc] == h) & (win[ac] == a)) | ((win[hc] == a) & (win[ac] == h))]
        if g.empty:
            return "no game between these teams near the pick date"
        return g.sort_values("kick").iloc[0]
    if not (h or a):
        return "teams not recognized"
    try:
        t0 = pd.Timestamp(published) if published else t
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
    except (ValueError, TypeError):
        t0 = t
    t0 = min(t0, t)
    x = h or a
    g = df[((df[hc] == x) | (df[ac] == x)) & (df.kick > t0 - timedelta(hours=12)) & (df.kick < t0 + timedelta(days=6))]
    if len(g) != 1:
        return "opponent not named and the game can't be pinned down" if len(g) else "no game for this team near the pick date"
    return g.iloc[0]


def _utc(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _utc_seen(p: dict) -> bool:
    t = _utc(p.get("seen_at"))
    if t is None:
        return False
    p["seen_at"] = t.isoformat(timespec="minutes")
    return True


def grade_pick(p: dict, nfl: pd.DataFrame, cfb: pd.DataFrame, cmatch) -> dict:
    """Match a logged pick to its real game and grade it at the close. Every outcome names the game and the canonical side
    (team as the schedule spells it, or over/under), so the same pick written two ways is one pick:
    {skip: why} | {late: True, ...} | {excluded: why, ...} | {pending: True, ...} | graded {result, units, close_line, ...}."""
    seen = _utc(p["seen_at"])
    mk, side = p.get("market"), str(p.get("pick") or "")
    if p.get("sport") == "nfl":
        h, a = nfl_team(p.get("home_team")), nfl_team(p.get("away_team"))
        g = _game(nfl, "home_team", "away_team", h, a, seen, p.get("published"))
        if isinstance(g, str):
            return {"skip": g}
        r = g
        home, away, gid, week = r.home_team, r.away_team, str(r.game_id), int(r.week)
        picked = nfl_team(side)
        final = not pd.isna(r.home_score)
        if final:
            hs, as_ = float(r.home_score), float(r.away_score)
            home_margin_line = float(r.spread_line) if pd.notna(r.spread_line) else None     # + = home favored
            total = float(r.total_line) if pd.notna(r.total_line) else None
            ml = {home: _dec(r.home_moneyline), away: _dec(r.away_moneyline)}
    else:
        h, a = cmatch(p.get("home_team") or ""), cmatch(p.get("away_team") or "")
        g = _game(cfb, "home", "away", h, a, seen, p.get("published"))
        if isinstance(g, str):
            return {"skip": g}
        r = g
        home, away, gid, week = r.home, r.away, str(r.game_id), int(r.week)
        picked = cmatch(side)
        final = bool(r.completed) and not pd.isna(r.home_pts)
        if final:
            hs, as_ = float(r.home_pts), float(r.away_pts)
            home_margin_line = -float(r.spread_close) if pd.notna(r.spread_close) else None
            total = float(r.total_close) if pd.notna(r.total_close) else None
            ml = {home: _dec(r.home_ml), away: _dec(r.away_ml)}
    if mk == "total":
        cs = side.lower()
        if cs not in ("over", "under"):
            return {"skip": "total pick is not over/under"}
    else:
        if picked not in (home, away):
            return {"skip": "picked team isn't in this game"}
        cs = picked
    base = {"game_id": gid, "week": week, "kick": r.kick.isoformat(), "away_actual": away, "home_actual": home, "side": cs}
    if r.kick <= pd.Timestamp(seen):
        return dict(base, late=True)
    if not final:
        return dict(base, pending=True)
    out = dict(base, final=f"{int(as_)}-{int(hs)}")
    if mk == "total":
        if total is None:
            return dict(base, excluded="no closing total for this game")
        why = PC.grade_check("total", p.get("line"), total)
        if why:
            return dict(base, excluded=why)
        d = (hs + as_ - total) * (1 if cs == "over" else -1)
        out["close_line"] = total
        out["result"] = "push" if d == 0 else "win" if d > 0 else "loss"
        out["units"] = 0.0 if d == 0 else (WIN_110 if d > 0 else -1.0)
        st = p.get("line")
        if isinstance(st, (int, float)) and abs(st - total) <= 7:
            out["line_value"] = round((total - st) * (1 if cs == "over" else -1), 2)
        return out
    out["pick_is_home"] = picked == home
    pm = (hs - as_) if picked == home else (as_ - hs)
    if mk == "spread":
        if home_margin_line is None:
            return dict(base, excluded="no closing spread for this game")
        close_pick = -home_margin_line if picked == home else home_margin_line    # picked team's spread at the close
        why = PC.grade_check("spread", p.get("line"), close_pick)
        if why:
            return dict(base, excluded=why)
        d = pm + close_pick
        out["close_line"] = close_pick
        out["result"] = "push" if d == 0 else "win" if d > 0 else "loss"
        out["units"] = 0.0 if d == 0 else (WIN_110 if d > 0 else -1.0)
        st = p.get("line")
        if isinstance(st, (int, float)) and abs(st - close_pick) <= 7:
            out["line_value"] = round(st - close_pick, 2)    # + = picker's number was better than the close
        return out
    if mk == "moneyline":
        d0 = ml.get(picked)
        if not d0:
            return dict(base, excluded="no closing moneyline for this game")
        out["close_line"] = round((d0 - 1) * 100) if d0 >= 2 else round(-100 / (d0 - 1))
        out["result"] = "win" if pm > 0 else "loss" if pm < 0 else "push"
        out["units"] = round(d0 - 1, 3) if pm > 0 else -1.0 if pm < 0 else 0.0
        return out
    return {"skip": "not a game line"}


# Display/cleaning only (frozen hot-picker rule v1 counts "named analysts or accounts"): one name per person or
# outlet, and items that are not anyone's pick (market prices quoted in an article, fan-forum predictions) are skipped.
NAME_ALIASES = {"zach cohen": "Zachary Cohen", "sportsline": "SportsLine Projection Model"}
OUTLET_ALIASES = {"yahoo_nfl": "Yahoo Sports", "yahoo_cfb": "Yahoo Sports", "cbs_nfl": "CBS Sports", "cbs_cfb": "CBS Sports",
                  "CBS Sports (SportsLine)": "SportsLine", "vsin": "VSiN", "sdsouth": "Saturday Down South", "fox_nfl": "FOX Sports",
                  "pfn": "Pro Football Network"}
MARKET_OUTLETS = {"polymarket", "kalshi", "betmgm", "caesars sports", "draftkings", "fanduel", "fanatics", "espn bet", "thescore bet"}


def outlet_of(p: dict) -> str:
    o = p.get("outlet") or p.get("source") or ""
    if str(o).startswith("bsky:"):
        return "Bluesky"
    return OUTLET_ALIASES.get(o, o)


def not_a_pick(p: dict) -> str | None:
    """Why a logged item isn't anyone's pick (None = it is): odds/market prices quoted in an article, fan-forum posts."""
    words = f"{p.get('confidence_words') or ''}".lower()
    if not (p.get("analyst") or "").strip() and str(p.get("outlet") or "").lower() in MARKET_OUTLETS:
        return "market price, not a pick"
    if "per share" in words or "trading at $" in words:
        return "market price, not a pick"
    if "system match" in words or "betting splits" in words:
        return "betting system or splits, not a pick"
    if " fans:" in f" {(p.get('title') or '').lower()}" or "fans divid" in (p.get("title") or "").lower():
        return "fan-forum prediction"
    return None


def picker_of(p: dict) -> str:
    a = (p.get("analyst") or "").strip()
    if p.get("source") == "bluesky":
        h = (p.get("outlet") or "bluesky").split(" (")[0]
        return "@" + h[5:].replace(".bsky.social", "") if h.startswith("bsky:") else h
    if a:
        if a.lower() in NAME_ALIASES:
            return NAME_ALIASES[a.lower()]
        o = outlet_of(p)
        # a lone first or last name ("Jon") is different people at different outlets: keep them apart
        return f"{a} ({o})" if len(a.split()) == 1 and o and o.lower() != a.lower() else a
    o = outlet_of(p)
    return o if "model" in o.lower() else f"{o} staff"


def is_named(picker: str) -> bool:
    return not picker.endswith(" staff")


def _p_cover(w: int, n: int, p0: float) -> float:
    if n == 0:
        return 1.0
    mu, sd = n * p0, math.sqrt(n * p0 * (1 - p0))
    return 0.5 * math.erfc(((w - 0.5) - mu) / sd / math.sqrt(2))


CAP_GRADED, CAP_NOT_COUNTED = 40, 30


def _why(p: dict) -> dict:
    """Why the picker made it: the grounded AI summary and their own words (picks_reasons.json fills in reasons found on a
    later read of the article)."""
    return {"reason": (p.get("reason") or "")[:400], "quote": (p.get("confidence_words") or "")[:400]}


def _row(p: dict, g: dict) -> dict:
    """What the page shows for one pick (pending, graded or not counted)."""
    return {"picker": p["_picker"], "sport": p.get("sport"), "seen_at": p.get("seen_at"), "market": p.get("market"),
            "pick": p.get("pick"), "line": p.get("line"), "link": p.get("link"), "outlet": outlet_of(p),
            "away": g.get("away_actual") or p.get("away_team"), "home": g.get("home_actual") or p.get("home_team"),
            "kick": g.get("kick"), "game_id": g.get("game_id"), "side": g.get("side"), **_why(p)}


def _last(n: str) -> str:
    w = PC._norm(n).split()
    return w[-1] if w else ""


def reread_verdicts(items: list[dict], reads: dict, team) -> dict:
    """picks_check layer 4, by SIDE: a pick is dropped when the article's latest read (after the pick was logged) shows the
    same analyst on the same game and market with the OTHER side only, or when that read's picks table rejected it.
    A pick merely missing from a re-read is kept: the AI skips things, and spellings differ (before 2026-10-18 a missing
    exact string dropped 355 picks, most of them real). -> {id(item): reason}"""
    latest = {}
    for r in reads.values():
        if r.get("link") and ("picks" in r or "keys" in r):
            if r.get("last", "") >= latest.get(r["link"], {}).get("last", ""):
                latest[r["link"]] = r
    def entries(r):
        if "picks" in r:
            return r["picks"], r.get("rejected") or []
        out = []
        for k in r.get("keys") or []:
            f = k.split("|")
            if len(f) == 5:
                out.append({"analyst": f[0], "away_team": f[1], "home_team": f[2], "market": f[3], "pick": f[4]})
        return out, []
    def canon(e, it):
        g = it["_g"]
        teams = {g["away_actual"], g["home_actual"]}
        if {team(e.get("away_team")), team(e.get("home_team"))} != teams:
            return None
        if e.get("market") == "total":
            return str(e.get("pick") or "").lower()
        return team(e.get("pick"))
    out = {}
    for it in items:
        r = latest.get(it.get("link"))
        if not r or r.get("last", "") <= it.get("seen_at", ""):
            continue
        kept, rej = entries(r)
        mine = lambda e: _last(e.get("analyst") or "") == _last(it.get("analyst") or "") and e.get("market") == it.get("market")
        sides = {canon(e, it) for e in kept if mine(e)} - {None}
        if sides and it["_g"]["side"] not in sides:
            out[id(it)] = "the article's latest read shows the other side"
            continue
        for e in rej:
            if mine(e) and canon(e, it) == it["_g"]["side"] and "table" in (e.get("why") or ""):
                out[id(it)] = "the article's picks table disagrees"
                break
    return out


def contradictions(items: list[dict]) -> dict:
    """One analyst's picks on one real game must agree: not both sides of a market, and not a moneyline on one team while
    laying points with the other (Tennessee ML + Alabama -8.5). Nobody can tell which read is right, so neither counts.
    Picks are matched by game id, so rematches are separate games. -> {id(item): reason}"""
    by = {}
    for it in items:
        if is_named(it["_picker"]):        # "<outlet> staff" pools different unnamed writers: they may disagree
            by.setdefault((it["_picker"], it["_g"]["game_id"]), []).append(it)
    out = {}
    for ps in by.values():
        mk = {}
        for it in ps:
            mk.setdefault(it.get("market"), set()).add(it["_g"]["side"])
        bad = {m for m, ss in mk.items() if len(ss) > 1}
        ml = mk.get("moneyline", set())
        fav = {it["_g"]["side"] for it in ps if it.get("market") == "spread"
               and isinstance(it.get("line"), (int, float)) and it["line"] < 0}
        if len(ml) == 1 and fav and not fav <= ml:
            bad |= {"moneyline", "spread"}
        for it in ps:
            if it.get("market") in bad:
                out[id(it)] = "this analyst's picks on the game contradict each other"
    return out


def regrade(x: dict) -> tuple[str, float] | None:
    """Recompute a graded pick's result and units from its stored final score and closing line (consistency gate)."""
    try:
        a, h = (float(v) for v in str(x["final"]).split("-"))
    except (KeyError, ValueError):
        return None
    if x["market"] == "total":
        d = (a + h - x["close_line"]) * (1 if x["side"] == "over" else -1)
    else:
        pm = (h - a) if x.get("pick_is_home") else (a - h)
        d = pm + x["close_line"] if x["market"] == "spread" else pm
    res = "push" if d == 0 else "win" if d > 0 else "loss"
    if x["market"] == "moneyline":
        c = x["close_line"]
        units = 0.0 if res == "push" else -1.0 if res == "loss" else round((c / 100) if c > 0 else (100 / abs(c)), 3)
    else:
        units = 0.0 if res == "push" else WIN_110 if res == "win" else -1.0
    return res, units


def hot_list(graded: list[dict]) -> list[dict]:
    hot_now = []
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        if not rows:
            continue
        nxt = max(x["week"] for x in rows) + 1
        rec = {}
        for x in rows:
            if nxt - 4 <= x["week"] < nxt:
                t = rec.setdefault(x["picker"], [0, 0]); t[0] += x["result"] == "win"; t[1] += x["result"] == "loss"
        for k, (w, l) in sorted(rec.items()):
            if is_named(k) and w + l >= 8 and w / (w + l) >= 0.60:
                hot_now.append({"picker": k, "sport": sport, "week": nxt, "record": f"{w}-{l}"})
    return hot_now


def records(graded: list[dict], pending: list[dict], not_counted: list[dict]) -> tuple[list[dict], dict]:
    """Per-picker table rows and pop-up lists, built from the SAME lists so they always agree."""
    by: dict = {}
    def b_of(name, outlet):
        return by.setdefault(name, {"picker": name, "outlet": outlet, "graded": [], "pending": [], "nc": []})
    for x in graded:
        b_of(x["picker"], x["outlet"])["graded"].append(x)
    for x in pending:
        b_of(x["picker"], x["outlet"])["pending"].append(x)
    for x in not_counted:
        if x["picker"] in by:            # shown in the pop-up of pickers who have counted picks
            by[x["picker"]]["nc"].append(x)
    rows, detail = [], {}
    keep = ("kick", "week", "away", "home", "market", "pick", "line", "close_line", "result", "units", "line_value", "final",
            "link", "reason", "quote", "seen_at")
    for b in by.values():
        g = b["graded"]
        w, l, pu = (sum(x["result"] == k for x in g) for k in ("win", "loss", "push"))
        lv = [x["line_value"] for x in g if x.get("line_value") is not None]
        rows.append({"picker": b["picker"], "outlet": b["outlet"], "sports": sorted({x["sport"] for x in g + b["pending"]}),
                     "graded": len(g), "wins": w, "losses": l, "pushes": pu, "win_pct": round(w / (w + l), 3) if w + l else None,
                     "units": round(sum(x["units"] for x in g), 2), "avg_line_value": round(sum(lv) / len(lv), 2) if lv else None,
                     "n_line_value": len(lv), "pending": len(b["pending"]), "picks": len(g) + len(b["pending"]),
                     "not_counted": len(b["nc"])})
        detail[b["picker"]] = {
            "pending": sorted(({k: x.get(k) for k in keep} for x in b["pending"]), key=lambda x: (x["kick"] or "", x["seen_at"])),
            "graded": sorted(({k: x.get(k) for k in keep} for x in g), key=lambda x: x["kick"], reverse=True)[:CAP_GRADED],
            "not_counted": sorted(({**{k: x.get(k) for k in keep}, "why": x["why"]} for x in b["nc"]),
                                  key=lambda x: x["seen_at"] or "", reverse=True)[:CAP_NOT_COUNTED]}
    rows.sort(key=lambda x: (-(x["graded"] > 0), -x["units"], -x["graded"], -x["pending"], x["picker"]))
    return rows, detail


def _overall(gs: list[dict]) -> dict:
    lv = [x["line_value"] for x in gs if x.get("line_value") is not None]
    return {"w": sum(x["result"] == "win" for x in gs), "l": sum(x["result"] == "loss" for x in gs),
            "p": sum(x["result"] == "push" for x in gs), "units": round(sum(x["units"] for x in gs), 2),
            "avg_line_value": round(sum(lv) / len(lv), 2) if lv else None}


def audit(out: dict, graded: list[dict], pending: list[dict], now: datetime) -> tuple[list[str], list[str], int]:
    """Consistency gate (poka-yoke): every number on the page is re-derived and compared. -> (errors, warnings, checks).
    Any error blocks publishing: the previous good picks_scores.json stays and a phone alert goes out."""
    err, warn, n = [], [], 0
    def chk(ok, msg):
        nonlocal n
        n += 1
        if not ok:
            err.append(msg)
    ids = [(x["picker"], x["sport"], x["game_id"], x["market"], x["side"]) for x in graded + pending]
    chk(len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} picks counted twice")
    for x in graded:
        rg = regrade(x)
        chk(rg is not None and rg[0] == x["result"] and abs(rg[1] - x["units"]) < 0.002,
            f"{x['picker']} {x['pick']} {x['market']} ({x['away']} @ {x['home']}): stored {x['result']} {x['units']}, recomputed {rg}")
        chk(_utc(x["seen_at"]) < _utc(x["kick"]), f"{x['picker']} {x['pick']}: counted although first seen after kickoff")
    games = {}
    for x in graded + pending:
        if is_named(x["picker"]):
            games.setdefault((x["picker"], x["game_id"], x["market"]), set()).add(x["side"])
    for k, ss in games.items():
        chk(len(ss) == 1, f"{k[0]} has both sides of game {k[1]} {k[2]}")
    for sport, v in out["by_sport"].items():
        tot = {"w": 0, "l": 0}
        for r in v["pickers"]:
            d = v["picks_by_picker"].get(r["picker"], {})
            chk(r["graded"] == r["wins"] + r["losses"] + r["pushes"], f"{sport} {r['picker']}: graded != W+L+P")
            chk(r["picks"] == r["graded"] + r["pending"], f"{sport} {r['picker']}: picks != graded + pending")
            chk(len(d.get("pending", [])) == r["pending"], f"{sport} {r['picker']}: table says {r['pending']} pending, pop-up lists {len(d.get('pending', []))}")
            chk(len(d.get("graded", [])) == min(r["graded"], CAP_GRADED), f"{sport} {r['picker']}: table says {r['graded']} graded, pop-up lists {len(d.get('graded', []))}")
            chk(len(d.get("not_counted", [])) == min(r["not_counted"], CAP_NOT_COUNTED), f"{sport} {r['picker']}: not-counted list mismatch")
            tot["w"] += r["wins"]; tot["l"] += r["losses"]
        chk(len(v["pickers"]) == v["n_pickers"], f"{sport}: picker table truncated")
        chk((tot["w"], tot["l"]) == (v["overall"]["w"], v["overall"]["l"]), f"{sport}: overall record != sum of pickers")
    chk(sorted(map(str, out["hot_now"])) == sorted(map(str, hot_list(graded))), "hot list doesn't match the graded picks")
    for x in pending:
        if x["kick"] and _utc(x["kick"]) < now - timedelta(hours=36):
            warn.append(f"{x['picker']} {x['pick']} ({x['away']} @ {x['home']}): no final score 36 h after kickoff")
    return err, warn, n


def record_failure(msg: str, hist: Path = HIST, notify: bool = True, now: datetime | None = None) -> None:
    """Scoring could not finish: keep the last good picks_scores.json, mark picks_audit.json failed, alert once."""
    now = now or datetime.now(timezone.utc)
    p = hist / "picks_audit.json"
    try:
        prev = json.loads(p.read_text())
    except (OSError, ValueError):
        prev = {}
    state = {"ok": False, "at": now.isoformat(timespec="minutes"), "errors": [msg[:500]], "n_errors": 1,
             "last_good": prev.get("last_good"), "alerted": prev.get("alerted")}
    if notify and prev.get("alerted") != msg[:200]:
        try:
            from . import notify as N
            N.send("Picker records failed a check", f"{msg[:300]}. The page keeps the last good records.", priority=4,
                   tags=["warning"])
        except Exception as e:
            print("picks failure alert failed:", e)
        state["alerted"] = msg[:200]
    p.write_text(json.dumps(state, indent=1))


def run(hist: Path = HIST, now: datetime | None = None, nfl=None, cfb=None, notify: bool = True) -> dict:
    now = now or datetime.now(timezone.utc)
    r = load_rules()
    path = hist / "picks_log.jsonl"
    picks = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    picks = [p for p in picks if _utc_seen(p)]          # every timestamp compared below is UTC, same format
    try:
        reads = json.loads((hist / "picks_reads.json").read_text())
    except (OSError, ValueError):
        reads = {}
    wp = hist / "picks_reasons.json"
    why = json.loads(wp.read_text()) if wp.exists() else {}
    if why:
        from .picks_log import pick_key
        for p in picks:                      # reasons found on a later read of the article (grounded at read time)
            w = why.get("|".join(map(str, pick_key(p))))
            if w:
                if len(w.get("reason") or "") > len(p.get("reason") or ""):
                    p["reason"] = w["reason"]
                if len(w.get("quote") or "") > len(p.get("confidence_words") or ""):
                    p["confidence_words"] = w["quote"]
    nfl = nfl if nfl is not None else nfl_games()
    try:
        cfb = cfb if cfb is not None else cfb_games()
    except Exception as e:
        print("picks: college games unavailable:", e)
        cfb = pd.DataFrame(columns=["home", "away", "kick", "completed", "home_pts", "game_id", "week"])
    cmatch = cfb_matcher(sorted(set(cfb.home) | set(cfb.away))) if len(cfb) else (lambda s: None)
    def team(t):
        return nfl_team(t) or cmatch(t or "")
    # last-name-only bylines ("Hummer") -> the full name the same outlet uses elsewhere ("Chris Hummer")
    full: dict = {}
    for p in picks:
        n = (p.get("analyst") or "").strip()
        if len(n.split()) >= 2:
            full.setdefault((p.get("source"), n.split()[-1].lower()), set()).add(n)
    for p in picks:
        n = (p.get("analyst") or "").strip()
        if n and len(n.split()) == 1 and len(full.get((p.get("source"), n.lower()), ())) == 1:
            p["analyst"] = next(iter(full[(p.get("source"), n.lower())]))
    skipped: dict = {}
    not_counted: list = []
    left_out = {"nfl": {}, "cfb": {}}
    def leave_out(p, g, why):
        skipped[why] = skipped.get(why, 0) + 1
        if p.get("sport") in ("nfl", "cfb"):
            lo = left_out[p["sport"]]; lo[why] = lo.get(why, 0) + 1
            not_counted.append(dict(_row(p, g or {}), why=why))
    # 1) text checks, 2) match each pick to its real game, 3) one pick per (picker, game, market, side): earliest sighting
    items, seen_ids = [], {}
    for p, w in zip(picks, PC.offline_reasons(picks, norm=team)):
        p["_picker"] = picker_of(p)
        if p.get("sport") not in ("nfl", "cfb"):
            leave_out(p, None, "not NFL or college football"); continue
        w = w or not_a_pick(p)
        if w:
            leave_out(p, None, w); continue
        try:
            g = grade_pick(p, nfl, cfb, cmatch)
        except Exception as e:
            g = {"skip": f"could not be graded ({type(e).__name__})"}
        if g.get("skip"):
            leave_out(p, None, g["skip"]); continue
        p["_g"] = g
        cid = (p["_picker"], p.get("sport"), g["game_id"], p.get("market"), g["side"])
        prev = seen_ids.get(cid)
        if prev is None:
            seen_ids[cid] = p; items.append(p)
        else:                                   # same pick again (another article, a re-read, another spelling)
            first, other = (p, prev) if p.get("seen_at", "") < prev.get("seen_at", "") else (prev, p)
            for k in ("reason", "confidence_words"):
                if len(other.get(k) or "") > len(first.get(k) or ""):
                    first[k] = other[k]
            if first is p:
                items[items.index(prev)] = p; seen_ids[cid] = p
    # 4) re-reads that show the other side, 5) contradictions across articles
    drop = reread_verdicts(items, reads, team)
    drop.update({k: v for k, v in contradictions([it for it in items if id(it) not in drop]).items()})
    graded, pending, late = [], [], 0
    for it in items:
        g = it["_g"]
        if id(it) in drop:
            leave_out(it, g, drop[id(it)]); continue
        if g.get("late"):
            late += 1; leave_out(it, g, "first seen after kickoff"); continue
        if g.get("excluded"):
            leave_out(it, g, g["excluded"]); continue
        if g.get("pending"):
            pending.append(_row(it, g)); continue
        graded.append({**_row(it, g), **{k: g.get(k) for k in ("week", "result", "units", "close_line", "final", "line_value",
                                                                 "pick_is_home")}})
    # hot test: per sport and week, hot = >= 60% with >= 8 graded over the 4 previous weeks; follow next week's picks
    test = {"followed": 0, "w": 0, "l": 0, "units": 0.0, "others_w": 0, "others_l": 0}
    for sport in ("nfl", "cfb"):
        rows = [x for x in graded if x["sport"] == sport]
        for wk in sorted({x["week"] for x in rows}):
            rec: dict = {}
            for x in rows:
                if wk - 4 <= x["week"] < wk:
                    t = rec.setdefault(x["picker"], [0, 0]); t[0] += x["result"] == "win"; t[1] += x["result"] == "loss"
            hot = {k for k, (w, l) in rec.items() if is_named(k) and w + l >= 8 and w / (w + l) >= 0.60}
            for x in rows:
                if x["week"] != wk or x["result"] == "push":
                    continue
                if x["picker"] in hot:
                    test["followed"] += 1; test["w"] += x["result"] == "win"; test["l"] += x["result"] == "loss"; test["units"] += x["units"]
                else:
                    test["others_w"] += x["result"] == "win"; test["others_l"] += x["result"] == "loss"
    n = test["w"] + test["l"]
    test["cover_rate"] = round(test["w"] / n, 4) if n else None
    no = test["others_w"] + test["others_l"]
    test["others_rate"] = round(test["others_w"] / no, 4) if no else None
    test["p_value"] = round(_p_cover(test["w"], n, r["validation"]["cover_rate_vs_close_above"]), 4) if n else None
    v = r["validation"]
    test["passed"] = bool(n >= v["min_followed_picks"] and test["cover_rate"] > v["cover_rate_vs_close_above"]
                          and test["p_value"] < v["one_sided_p_below"] and (test["others_rate"] is None or test["cover_rate"] > test["others_rate"]))
    test["units"] = round(test["units"], 2)
    hot_now = hot_list(graded)
    by_sport = {}
    for sport in ("nfl", "cfb"):   # each page shows its own league
        gs, ps = [x for x in graded if x["sport"] == sport], [x for x in pending if x["sport"] == sport]
        tb, detail = records(gs, ps, [x for x in not_counted if x["sport"] == sport])
        by_sport[sport] = {"pickers": tb, "picks_by_picker": detail, "n_pickers": len(tb), "picks_graded": len(gs),
                           "picks_pending": len(ps), "hot_now": [h for h in hot_now if h["sport"] == sport],
                           "left_out": dict(sorted(left_out[sport].items(), key=lambda kv: -kv[1])),
                           "overall": _overall(gs)}
    pickers, _ = records(graded, pending, [])
    out = {"generated_at": now.isoformat(timespec="minutes"), "rules_version": r["version"], "picks_logged": len(picks),
           "not_picks_skipped": dict(sorted(skipped.items(), key=lambda kv: -kv[1])), "by_sport": by_sport,
           "picks_graded": len(graded), "picks_pending": len(pending), "late_picks_dropped": late, "pickers": pickers[:250],
           "n_pickers": len(pickers), "hot_now": hot_now, "hot_test": test, "overall": _overall(graded)}
    errors, warnings, checks = audit(out, graded, pending, now)
    out["audit"] = {"ok": not errors, "checks": checks, "at": out["generated_at"], "warnings": warnings[:20]}
    state_p = hist / "picks_audit.json"
    try:
        prev = json.loads(state_p.read_text())
    except (OSError, ValueError):
        prev = {}
    if errors:
        sig = str(sorted(errors))[:4000]
        state = {"ok": False, "at": out["generated_at"], "errors": errors[:50], "n_errors": len(errors), "checks": checks,
                 "last_good": prev.get("last_good"), "alerted": prev.get("alerted")}
        if notify and prev.get("alerted") != sig:
            try:
                from . import notify as N
                N.send("Picker records failed a check", f"{len(errors)} problem(s); the page keeps the last good records. "
                       f"First: {errors[0][:200]}", priority=4, tags=["warning"])
            except Exception as e:
                print("picks audit alert failed:", e)
            state["alerted"] = sig
        state_p.write_text(json.dumps(state, indent=1))
        print(f"PICKS AUDIT FAILED ({len(errors)}): not publishing.", *errors[:10], sep="\n  ")
        return out
    state_p.write_text(json.dumps({"ok": True, "at": out["generated_at"], "checks": checks, "last_good": out["generated_at"],
                                   "warnings": warnings[:20]}, indent=1))
    (hist / "picks_scores.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    res = run()
    print(json.dumps({k: res[k] for k in ("picks_logged", "picks_graded", "late_picks_dropped", "overall", "hot_test")}, indent=1))
    for p in res["pickers"][:15]:
        print(p)
