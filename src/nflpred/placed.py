"""Real bets you placed, logged through the 'I placed a bet' GitHub issue form.

Flow: an alert (ntfy button) or the dashboard's "Log bet" link opens the issue form pre-filled with
the paper bet's id, pick, book and price -> you fix the odds/stake you actually got and submit ->
the placed_bet workflow runs `python -m nflpred.placed ingest`, which appends the bet to
history/placed_bets.json and closes the issue. The dashboard scores each placed bet against the
paper bet it came from: your price vs the track's price, closing line value at your price, result
and profit in dollars. Only issues opened by the repo owner are accepted.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
LOG = HIST / "placed_bets.json"
REPO_URL = "https://github.com/hollingswood/nfl-win-probability"
FIELDS = {"Bet ID": "bet_id", "Pick": "pick", "Book": "book", "Odds": "odds", "Stake ($)": "stake", "Notes": "notes",
          "Different book? (optional)": "alt_book", "Different odds? (optional)": "alt_odds",
          "Different line? (optional)": "point", "Notes (optional)": "notes",
          "Odds you got": "odds", "Line you got": "point"}    # last two: labels of the first version of the form


# ------------------------------------------------------------------------------------- links
def pick_text(b: dict) -> str:
    if b.get("player"):
        return f"{b['player']} {b['side'].title()} {b['point']} {b.get('stat_label') or 'rec'}"
    team = b.get("team") or b.get("side", "")
    if b.get("market") == "h2h" or (b.get("point") is None and b.get("side") in ("home", "away")):
        return f"{team} ML"
    if b.get("side") in ("over", "under") and str(b.get("point")) not in team:
        return f"{team} {b['point']}"
    if b.get("point") is not None and str(abs(b["point"])) not in team:
        return f"{team} {b['point']:+g}"
    return team


def log_url(b: dict) -> str:
    """Pre-filled 'I placed a bet' form for a paper bet."""
    price = b.get("price")
    # Pre-filled fields can't be edited reliably (GitHub's form puts the URL value back), so they are never meant
    # to be edited: a different book / odds / line goes in the empty 'Different ...' boxes, which override them.
    q = {"template": "placed_bet.yml", "title": f"Placed: {pick_text(b)}", "bet_id": b.get("id", ""),
         "pick": pick_text(b), "book": b.get("book", ""), "odds": f"{price:+d}" if isinstance(price, int) else str(price or "")}
    return f"{REPO_URL}/issues/new?" + urllib.parse.urlencode(q)


# ------------------------------------------------------------------------------------- parsing
def parse_form(body: str) -> dict:
    out, cur = {}, None
    for line in (body or "").splitlines():
        m = re.match(r"^###\s+(.*)$", line.strip())
        if m:
            cur = FIELDS.get(m.group(1).strip())
            if cur:
                out[cur] = ""
            continue
        if cur:
            out[cur] = (out[cur] + "\n" + line).strip()
    return {k: ("" if v == "_No response_" else v) for k, v in out.items()}


def to_decimal(odds: str) -> float | None:
    s = str(odds).strip().lower().replace("−", "-").replace("¢", "c").replace(" ", "")
    try:
        if s.endswith("c"):
            c = float(s[:-1])
            return 100 / c if 0 < c < 100 else None
        v = float(s)
    except ValueError:
        return None
    if 0 < v < 1:  # a contract price given as 0.35
        return 1 / v
    if v >= 100:
        return 1 + v / 100
    if v <= -100:
        return 1 + 100 / -v
    if 1 < v < 100:  # decimal odds
        return v
    return None


def _num(s: str) -> float | None:
    try:
        return float(re.sub(r"[^\d.\-+]", "", str(s).replace("−", "-")))
    except ValueError:
        return None


def normalize(rec: dict) -> dict:
    """Odds typed into the 'Line you got' field (the form wouldn't let the odds field change): a 'line' of 100+ in
    absolute value is American odds, not a spread/total number."""
    p = rec.get("point")
    if p is not None and abs(p) >= 100:
        odds = f"{p:+.0f}"
        dec = to_decimal(odds)
        rec = dict(rec, odds=odds, decimal=round(dec, 4) if dec else rec.get("decimal"), point=None,
                   notes=((rec.get("notes") or "") + " (odds taken from the line field)").strip())
    return rec


def record_from_issue(issue: dict) -> dict:
    return normalize(_record_from_issue(issue))


def _record_from_issue(issue: dict) -> dict:
    f = parse_form(issue.get("body", ""))
    if f.get("alt_book"):   # 'Different ...' boxes override the pre-filled values
        f["book"] = f["alt_book"]
    if f.get("alt_odds"):
        f["odds"] = f["alt_odds"]
    dec = to_decimal(f.get("odds", ""))
    return {"issue": issue.get("number"), "logged_at": issue.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "bet_id": f.get("bet_id") or None, "pick": f.get("pick", ""), "book": f.get("book", ""),
            "odds": f.get("odds", ""), "decimal": round(dec, 4) if dec else None,
            "point": _num(f["point"]) if f.get("point") else None, "stake": _num(f.get("stake", "")),
            "notes": f.get("notes", "")}


def ingest(event_path: str, log: Path = LOG) -> dict:
    ev = json.loads(Path(event_path).read_text())
    issue = ev["issue"]
    rec = record_from_issue(issue)
    rows = json.loads(log.read_text()) if log.exists() else []
    rows = [r for r in rows if r.get("issue") != rec["issue"]] + [rec]  # an edited issue replaces its entry
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(rows, indent=1))
    return rec


# ------------------------------------------------------------------------------------- scoring
def load_ledgers(hist: Path = HIST) -> dict:
    out = {}
    for p in [*hist.glob("paper_bets*.json"), *hist.glob("cfb/paper_bets*.json")]:   # NFL + college ledgers
        try:
            for b in json.loads(p.read_text()):
                if b.get("id"):
                    out[b["id"]] = b
        except (OSError, ValueError):
            continue
    return out


SPREAD_TRACKS = ("spread", "night_west", "preseason", "tuesday", "qb_news", "clock", "aplus_sp")


def bet_key(b: dict) -> tuple | None:
    """Same real-world bet across tracks: game, market, side (and player for props). Different numbers or prices of
    the same side still count as the same bet (you shouldn't bet it twice)."""
    game = b.get("game_id") or b.get("event_id")
    if not game:
        return None
    m, t, side = str(b.get("market") or "").lower(), str(b.get("track") or "").lower(), str(b.get("side") or "").lower()
    if b.get("player"):
        cls = "prop:" + m
    elif side in ("over", "under") or m in ("total", "totals") or "total" in t:
        cls = "total"
    elif m in ("spread", "spreads") or any(k in t for k in SPREAD_TRACKS):
        cls = "spread"
    else:
        cls = "ml"
    who = side if side in ("home", "away", "over", "under") else str(b.get("team") or "").lower()
    return (str(game), cls, who, str(b.get("player") or "").lower())


def placed_index(hist: Path = HIST) -> dict:
    """paper bet id -> what you placed, for every paper bet (any track, NFL or college) that is the same bet as one
    you logged, so the page can show 'placed' instead of 'I placed it' and you don't bet it twice."""
    rows = json.loads((hist / "placed_bets.json").read_text()) if (hist / "placed_bets.json").exists() else []
    led = load_ledgers(hist)
    keyed: dict = {}
    for b in led.values():
        k = bet_key(b)
        if k:
            keyed.setdefault(k, []).append(b["id"])
    out = {}
    for r in map(normalize, rows):
        src = led.get(r.get("bet_id") or "")
        k = bet_key(src) if src else None
        info = {"stake": r.get("stake"), "odds": r.get("odds"), "book": r.get("book"), "logged_at": r.get("logged_at"),
                "via": src.get("track") if src else None, "pick": r.get("pick")}
        if r.get("bet_id"):
            out[r["bet_id"]] = info
        for bid in keyed.get(k, []) if k else []:
            out.setdefault(bid, dict(info, same_bet=True))
    return out


def _paper_decimal(b: dict) -> float | None:
    if b.get("all_in_cost"):
        return 1 / b["all_in_cost"]
    p = b.get("price")
    return to_decimal(str(p)) if p is not None else None


def _result_at_point(b: dict, point: float) -> str | None:
    """Re-grade a spread/total at a different number from the final score ('away-home')."""
    try:
        a, h = (float(x) for x in b["final"].split("-"))
    except (KeyError, ValueError):
        return None
    side = b.get("side")
    if side == "over":
        d = a + h - point
    elif side == "under":
        d = point - a - h
    elif side == "home":
        d = h + point - a
    elif side == "away":
        d = a + point - h
    else:
        return None
    return "push" if d == 0 else "win" if d > 0 else "loss"


def kickoff_of(b: dict, schedule: dict | None = None) -> str | None:
    """Game start (UTC ISO) of a paper bet: its own kickoff_utc, else the nflverse schedule (gameday + gametime ET)."""
    if b.get("kickoff_utc"):
        return b["kickoff_utc"]
    gt = (schedule or {}).get(b.get("game_id"))
    if gt:
        from zoneinfo import ZoneInfo
        try:
            t = datetime.fromisoformat(f"{gt[0]}T{gt[1] or '13:00'}").replace(tzinfo=ZoneInfo("America/New_York"))
            return t.astimezone(timezone.utc).isoformat(timespec="minutes")
        except ValueError:
            return None
    return None


def _schedule(root: Path = ROOT) -> dict:
    try:
        import pandas as pd
        g = pd.read_parquet(root / "data" / "raw" / "games.parquet", columns=["game_id", "gameday", "gametime"])
        return {r.game_id: (str(r.gameday)[:10], r.gametime if isinstance(r.gametime, str) else None) for r in g.itertuples()}
    except Exception:
        return {}


TRACK_LABELS = {None: "Moneyline v1", "spread": "Spread v1", "moneyline_v2": "Moneyline v2", "moneyline_v3": "Moneyline v3",
                "moneyline_v4": "Moneyline v4", "aplus_ml": "A+ moneyline", "aplus_spread": "A+ spread", "aplus_totals": "A+ totals",
                "totals_wind": "Wind unders", "totals_early_under": "Early unders", "night_west": "Night games",
                "props_receptions": "Receptions props", "props_unders_tue": "Tuesday star unders", "preseason_prior": "Preseason prior",
                "tuesday_move": "Tuesday move", "qb_news": "QB news", "exchange_value": "Exchange value",
                "kalshi_maker": "Kalshi resting orders", "futures_shop": "Futures", "cfb_shop": "Shop vs sharp",
                "cfb_shop_v3": "Stricter overs (test)", "cfb_aplus": "College A+", "cfb_ml": "College moneyline",
                "cfb_body_clock": "Body clock"}


def score(rec: dict, ledgers: dict, schedule: dict | None = None) -> dict:
    r = dict(rec)
    b = ledgers.get(rec.get("bet_id") or "")
    if not b:
        r["status"] = "manual" if not rec.get("bet_id") else "unmatched"
        return r
    away, home = b.get("away_team") or b.get("away"), b.get("home_team") or b.get("home")
    if not (away and home) and str(b.get("game_id", "")).count("_") >= 3:
        away, home = str(b["game_id"]).split("_")[2:4]
    if away and home:
        r["matchup"] = f"{away} @ {home}"
    r["track_name"] = TRACK_LABELS.get(b.get("track"), str(b.get("track") or "").replace("_", " "))
    r.update(track=b.get("track"), paper_price=b.get("price"), paper_book=b.get("book"), game_id=b.get("game_id"),
             gameday=b.get("gameday"), paper_point=b.get("point"), kickoff_utc=kickoff_of(b, schedule))
    pd_, ud = _paper_decimal(b), rec.get("decimal")
    if pd_ and ud:
        r["price_vs_paper"] = round(ud / pd_ - 1, 4)  # +2% = you got 2% more payout than the track logged
    if b.get("status") != "graded":
        r["status"] = "open"
        return r
    same = rec.get("point") is None or b.get("point") is None or float(rec["point"]) == float(b["point"])
    res = b.get("result") if same else (None if b.get("player") else _result_at_point(b, float(rec["point"])))
    if res is None:
        r["status"] = "check"
        return r
    r["status"], r["result"] = "graded", res
    if ud and rec.get("stake"):
        r["profit"] = round(0.0 if res == "push" else rec["stake"] * (ud - 1) if res == "win" else -rec["stake"], 2)
    if same and ud and pd_ and b.get("clv") is not None:
        close_p = (1 + b["clv"]) / pd_
        r["clv"] = round(ud * close_p - 1, 4)
    return r


def sport_of(rec: dict, paper: dict | None) -> str:
    """'cfb' for college bets (college ledgers, cfb* ids, Kalshi NCAAF tickers or a college note), else 'nfl'."""
    bid = str(rec.get("bet_id") or "")
    if paper and (paper.get("sport") == "cfb" or str(paper.get("track") or "").startswith("cfb")):
        return "cfb"
    if bid.startswith("cfb") or "NCAAF" in bid or re.search(r"\b(college|ncaa|cfb)\b", f"{rec.get('notes', '')} {rec.get('pick', '')}", re.I):
        return "cfb"
    return "nfl"


def summary(hist: Path = HIST, sport: str | None = None) -> dict:
    """Your logged bets scored; sport='nfl' / 'cfb' keeps one league (each page shows its own)."""
    rows = json.loads((hist / "placed_bets.json").read_text()) if (hist / "placed_bets.json").exists() else []
    led = load_ledgers(hist)
    sched = _schedule(hist.parent)
    bets = [dict(score(normalize(x), led, sched), sport=sport_of(x, led.get(x.get("bet_id") or ""))) for x in rows]
    if sport:
        bets = [b for b in bets if b["sport"] == sport]
    g = [b for b in bets if b.get("status") == "graded"]
    staked = sum(b.get("stake") or 0 for b in g if b.get("result") != "push")
    profit = sum(b.get("profit") or 0 for b in g)
    clvs = [b["clv"] for b in g if b.get("clv") is not None]
    pv = [b["price_vs_paper"] for b in bets if b.get("price_vs_paper") is not None]
    return {"bets": bets[::-1], "graded": len(g), "wins": sum(b["result"] == "win" for b in g),
            "losses": sum(b["result"] == "loss" for b in g), "staked": round(staked, 2), "profit": round(profit, 2),
            "roi": round(profit / staked, 4) if staked else None, "avg_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
            "avg_price_vs_paper": round(sum(pv) / len(pv), 4) if pv else None,
            "open_stake": round(sum(b.get("stake") or 0 for b in bets if b.get("status") == "open"), 2)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        rec = ingest(os.environ.get("GITHUB_EVENT_PATH") or sys.argv[2])
        led = load_ledgers()
        s = score(rec, led)
        msg = (f"Logged: **{rec['pick']}** at {rec['book']} {rec['odds']}, ${rec['stake']:g}"
               if rec.get("stake") is not None else f"Logged: {rec['pick']}")
        if s.get("track"):
            msg += f"\n\nMatched paper bet `{rec['bet_id']}` ({s['track']}, logged at {s.get('paper_price')} {s.get('paper_book')})."
            if s.get("price_vs_paper") is not None:
                msg += f" Your price is {s['price_vs_paper']:+.1%} vs the track's."
        elif rec.get("bet_id"):
            msg += f"\n\n⚠️ No paper bet with id `{rec['bet_id']}`; kept as a manual bet."
        if rec.get("decimal") is None:
            msg += "\n\n⚠️ Couldn't read the odds; edit the issue to fix (it will re-log)."
        Path(os.environ.get("RUNNER_TEMP", "/tmp"), "placed_comment.md").write_text(msg)
        print(msg)
