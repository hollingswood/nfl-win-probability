"""Phone alerts through ntfy (https://ntfy.sh): free app, no account.

Set the GitHub secret NTFY_TOPIC to a long private topic name and subscribe to the same topic in the
ntfy app. Anyone who knows the topic can read it, so treat the name like a password.

* new_bets(pred)  -> one push per run listing every paper bet logged in that run
* python -m nflpred.notify "title" "message" [priority]  -> ad-hoc push (used for failed runs)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

DASHBOARD = "https://hollingswood.github.io/nfl-win-probability/"
AZ = timezone(timedelta(hours=-7))  # America/Phoenix, no DST
TRACK_NAMES = {
    "bets": "Moneyline v1", "spread_bets": "Spread v1", "ml_v2_bets": "Moneyline v2", "ml_v3_bets": "Moneyline v3",
    "ml_v4_bets": "Moneyline v4", "night_west_bets": "Night west", "totals_wind_bets": "Wind under",
    "totals_early_under_bets": "Early under", "props_receptions_bets": "Receptions prop",
    "aplus_ml_bets": "A+ moneyline", "aplus_spread_bets": "A+ spread", "aplus_totals_bets": "A+ total",
    "exchange_value_bets": "Exchange value",
}
PRIORITY_TRACKS = {"aplus_ml_bets", "aplus_spread_bets", "aplus_totals_bets", "ml_v4_bets", "props_receptions_bets"}


def send(title: str, message: str, priority: int = 3, tags: list[str] | None = None,
         click: str = DASHBOARD, topic: str | None = None) -> bool:
    topic = topic or os.environ.get("NTFY_TOPIC")
    if not topic:
        return False
    body = json.dumps({"topic": topic, "title": title, "message": message, "priority": priority,
                       "tags": tags or [], "click": click}).encode()
    req = urllib.request.Request("https://ntfy.sh/", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return 200 <= r.status < 300


def _kick(b: dict, pred: dict) -> str:
    k = b.get("kickoff_utc")
    if not k:
        g = next((g for g in pred.get("upcoming", []) if g.get("game_id") == b.get("game_id")), {})
        k = g.get("kickoff_utc")
    if not k:
        return ""
    t = datetime.fromisoformat(k).astimezone(AZ)
    return t.strftime("%a %-I:%M%p").replace("AM", "am").replace("PM", "pm")


def bet_line(key: str, b: dict, pred: dict) -> str:
    price = b.get("price")
    px = f"{price:+d}" if isinstance(price, int) else str(price)
    what = b.get("team") or b.get("side", "")
    if b.get("market") == "h2h":
        what += " ML"
    if key in ("bets", "ml_v2_bets", "ml_v3_bets", "ml_v4_bets", "aplus_ml_bets", "night_west_bets") and "ML" not in what:
        what = f"{what} ML" if key != "night_west_bets" else what
    parts = [f"{TRACK_NAMES.get(key, key)}: {what} {px} @ {b.get('book', '?')}"]
    if b.get("edge") is not None:
        parts.append(f"edge {b['edge']:+.1%}")
    if b.get("units") is not None:
        parts.append(f"{b['units']:g}u")
    kk = _kick(b, pred)
    if kk:
        parts.append(kk)
    return " · ".join(parts)


def collect(pred: dict) -> list[tuple[str, dict]]:
    out = []
    for key, block in pred.items():
        if key.endswith("_bets") and isinstance(block, dict):
            for b in block.get("new") or []:
                out.append((key, b))
    return out


def new_ids(pred: dict) -> set:
    return {b.get("id") for _, b in collect(pred)}


def new_bets(pred: dict, exclude_ids: set | None = None) -> bool:
    """exclude_ids: 'new' bets already present before this run (blocks a watch run did not refresh)."""
    items = [(k, b) for k, b in collect(pred) if b.get("id") not in (exclude_ids or set())]
    if not items:
        return False
    lines = [bet_line(k, b, pred) for k, b in items]
    hot = any(k in PRIORITY_TRACKS for k, _ in items)
    title = f"{len(items)} new paper bet{'s' if len(items) > 1 else ''}" + (" (A+/top track)" if hot else "")
    msg = "\n".join(lines) + "\n\nPrices move within hours. Paper tracks, not yet validated. Not financial advice."
    ok = send(title, msg, priority=4 if hot else 3, tags=["football"])
    print(f"notify: {'sent' if ok else 'skipped (no NTFY_TOPIC)'} {len(items)} bet(s)")
    return ok


if __name__ == "__main__":
    a = sys.argv[1:]
    ok = send(a[0] if a else "NFL predictor", a[1] if len(a) > 1 else "test", int(a[2]) if len(a) > 2 else 3,
              tags=["warning"] if len(a) > 2 and int(a[2]) >= 4 else ["football"])
    print("sent" if ok else "no NTFY_TOPIC set")
