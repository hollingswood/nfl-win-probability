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
FIELDS = {"Bet ID": "bet_id", "Pick": "pick", "Book": "book", "Odds you got": "odds", "Line you got": "point",
          "Stake ($)": "stake", "Notes": "notes"}


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


def record_from_issue(issue: dict) -> dict:
    f = parse_form(issue.get("body", ""))
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


def score(rec: dict, ledgers: dict) -> dict:
    r = dict(rec)
    b = ledgers.get(rec.get("bet_id") or "")
    if not b:
        r["status"] = "manual" if not rec.get("bet_id") else "unmatched"
        return r
    r.update(track=b.get("track"), paper_price=b.get("price"), paper_book=b.get("book"), game_id=b.get("game_id"),
             gameday=b.get("gameday"), paper_point=b.get("point"))
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


def summary(hist: Path = HIST) -> dict:
    rows = json.loads((hist / "placed_bets.json").read_text()) if (hist / "placed_bets.json").exists() else []
    led = load_ledgers(hist)
    bets = [score(x, led) for x in rows]
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
