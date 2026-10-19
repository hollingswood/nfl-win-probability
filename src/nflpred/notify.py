"""Phone alerts through ntfy (https://ntfy.sh): free app, no account.

Channels (any or all, as GitHub secrets): Pushover (PUSHOVER_USER + PUSHOVER_TOKEN), Telegram (TELEGRAM_BOT_TOKEN +
TELEGRAM_CHAT_ID), ntfy (NTFY_TOPIC). For ntfy: set NTFY_TOPIC to a long private topic name and subscribe to the same topic in the
ntfy app. Anyone who knows the topic can read it, so treat the name like a password.

* new_bets(pred)  -> one push per run listing every paper bet logged in that run
* python -m nflpred.notify "title" "message" [priority]  -> ad-hoc push (used for failed runs)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

DASHBOARD = "https://hollingswood.github.io/nfl-win-probability/"
AZ = timezone(timedelta(hours=-7))  # America/Phoenix, no DST
TRACK_NAMES = {
    "bets": "Moneyline v1", "spread_bets": "Spread v1", "ml_v2_bets": "Moneyline v2", "ml_v3_bets": "Moneyline v3",
    "ml_v4_bets": "Moneyline v4", "night_west_bets": "Night west", "totals_wind_bets": "Wind under",
    "totals_early_under_bets": "Early under", "props_receptions_bets": "Receptions prop", "props_unders_bets": "Tuesday under",
    "aplus_ml_bets": "A+ moneyline", "aplus_spread_bets": "A+ spread", "aplus_totals_bets": "A+ total",
    "exchange_value_bets": "Exchange value", "preseason_prior_bets": "Preseason prior", "tuesday_move_bets": "Tuesday move", "cfb_ml_bets": "College ML", "cfb_shop_bets": "College shop vs sharp", "cfb_shop_v3_bets": "College shop, stricter overs (test)", "cfb_aplus_bets": "College A+", "qb_news_bets": "QB news (speed)", "cfb_body_clock_bets": "College body clock (unvalidated)",
}
QUIET_TRACKS = {"props_unders_bets"}
PRIORITY_TRACKS = {"qb_news_bets", "cfb_shop_bets", "cfb_aplus_bets", "preseason_prior_bets", "tuesday_move_bets", "aplus_ml_bets", "aplus_spread_bets", "aplus_totals_bets", "ml_v4_bets", "props_receptions_bets"}


def _post(url: str, data: bytes, headers: dict) -> bool:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return 200 <= r.status < 300


def channels() -> list[str]:
    """Configured push channels (GitHub secrets): ntfy (NTFY_TOPIC), Pushover (PUSHOVER_USER + PUSHOVER_TOKEN),
    Telegram (TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID). Every configured channel gets every alert."""
    e = os.environ
    return [c for c, ok in (("ntfy", e.get("NTFY_TOPIC")), ("pushover", e.get("PUSHOVER_USER") and e.get("PUSHOVER_TOKEN")),
                            ("telegram", e.get("TELEGRAM_BOT_TOKEN") and e.get("TELEGRAM_CHAT_ID"))) if ok]


def send(title: str, message: str, priority: int = 3, tags: list[str] | None = None,
         click: str = DASHBOARD, topic: str | None = None, actions: list[dict] | None = None) -> bool:
    """Push to every configured channel; True if at least one accepted it. Failures on one channel never block another."""
    e, ok = os.environ, False
    if topic or e.get("NTFY_TOPIC"):
        payload = {"topic": topic or e["NTFY_TOPIC"], "title": title, "message": message, "priority": priority,
                   "tags": tags or [], "click": click}
        if actions:
            payload["actions"] = actions[:3]
        try:
            ok |= _post("https://ntfy.sh/", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        except Exception as ex:
            print("ntfy failed:", ex)
    link = next((a for a in (actions or []) if a.get("url")), None)
    if e.get("PUSHOVER_USER") and e.get("PUSHOVER_TOKEN"):
        form = {"token": e["PUSHOVER_TOKEN"], "user": e["PUSHOVER_USER"], "title": title[:250], "message": message[:1024],
                "priority": 1 if priority >= 4 else 0, "url": (link or {}).get("url", click),
                "url_title": (link or {}).get("label", "Dashboard")}
        try:
            ok |= _post("https://api.pushover.net/1/messages.json", urllib.parse.urlencode(form).encode(),
                        {"Content-Type": "application/x-www-form-urlencoded"})
        except Exception as ex:
            print("pushover failed:", ex)
    if e.get("TELEGRAM_BOT_TOKEN") and e.get("TELEGRAM_CHAT_ID"):
        buttons = [[{"text": a.get("label", "Open"), "url": a["url"]}] for a in (actions or [])[:3] if a.get("url")] or \
            [[{"text": "Dashboard", "url": click}]]
        body = {"chat_id": e["TELEGRAM_CHAT_ID"], "text": f"{title}\n\n{message}"[:4000], "disable_web_page_preview": True,
                "disable_notification": priority <= 2, "reply_markup": {"inline_keyboard": buttons}}
        try:
            ok |= _post(f"https://api.telegram.org/bot{e['TELEGRAM_BOT_TOKEN']}/sendMessage", json.dumps(body).encode(),
                        {"Content-Type": "application/json"})
        except Exception as ex:
            print("telegram failed:", ex)
    return bool(ok)


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def account_names() -> set | None:
    try:
        from pathlib import Path
        d = json.loads((Path(__file__).resolve().parents[2] / "my_books.json").read_text())
        return {_norm(x) for x in d.get("account_names", [])} or None
    except Exception:
        return None


def _kick(b: dict, pred: dict) -> str:
    k = b.get("kickoff_utc")
    if not k:
        g = next((g for g in pred.get("upcoming", []) if g.get("game_id") == b.get("game_id")), {})
        k = g.get("kickoff_utc")
    if not k:
        return ""
    t = datetime.fromisoformat(str(k).replace("Z", "+00:00")).astimezone(AZ)
    return t.strftime("%a %b %-d, %-I:%M%p").replace("AM", "am").replace("PM", "pm") + " AZ"


def is_college(key: str, b: dict) -> bool:
    return key.startswith("cfb_") or b.get("sport") == "cfb" or "NCAAF" in str(b.get("ticker", ""))


def matchup(b: dict, pred: dict) -> str:
    """'Away @ Home' for any track's bet (NFL abbreviations, college names, props' 'BAL@ATL')."""
    a, h = b.get("away_team") or b.get("away"), b.get("home_team") or b.get("home")
    if not (a and h):
        g = next((g for g in pred.get("upcoming", []) if g.get("game_id") == b.get("game_id")), None)
        if g:
            a, h = g.get("away_team"), g.get("home_team")
    if not (a and h) and "@" in str(b.get("opponent") or ""):
        a, h = str(b["opponent"]).split("@", 1)
    if not (a and h) and str(b.get("game_id", "")).count("_") >= 3:
        _, _, a, h = str(b["game_id"]).split("_")[:4]
    return f"{a} @ {h}" if a and h else ""


def grade_of(key: str, b: dict) -> str | None:
    g = b.get("grade_v2") or b.get("totals_grade") or b.get("grade")
    if g:
        return str(g)
    if key.startswith("cfb_") and b.get("edge") is not None and b.get("market") in ("spread", "total", "ml"):
        e = b["edge"]
        return "A+" if e >= 0.04 else "A" if e >= 0.02 else "B" if e >= 0 else "C"
    return None


def pick_of(key: str, b: dict) -> str:
    """'Over 69.5', 'WAS +3.5', 'LSU ML', 'Bijan Robinson Under 3.5' (no price, no book)."""
    if b.get("player"):
        return f"{b['player']} {str(b.get('side', '')).title()} {b.get('point')} {b.get('stat_label') or 'rec'}"
    side = str(b.get("side") or "")
    if side in ("over", "under"):
        return f"{side.title()} {b.get('point'):g}" if b.get("point") is not None else side.title()
    team = b.get("team") or side
    if b.get("market") in ("h2h", "ml") or key in ("bets", "ml_v2_bets", "ml_v3_bets", "ml_v4_bets", "aplus_ml_bets", "exchange_value_bets", "cfb_ml_bets", "kalshi_maker_bets") \
            or (b.get("point") is None and side in ("home", "away")):
        return f"{team} ML"
    if b.get("point") is not None:
        return f"{team} {b['point']:+g}"
    return str(team)


def alert(key: str, b: dict, pred: dict, also_keys: list[str] | None = None) -> tuple[str, str]:
    """(title, body) for one new paper bet: what to bet, which game and when, where, how good, how much."""
    price = b.get("price")
    px = f"{price:+d}" if isinstance(price, int) else str(price or "")
    sport = "College" if is_college(key, b) else "NFL"
    pick = pick_of(key, b)
    title = f"{sport} · {pick} {px}".strip()
    lines = [f"{matchup(b, pred)}{' · ' + _kick(b, pred) if _kick(b, pred) else ''}".strip(" ·"),
             f"Bet: {pick} at {px} on {b.get('book', '?')}"]
    gr, extra = grade_of(key, b), []
    if gr:
        extra.append(f"Grade {gr}")
    if b.get("edge") is not None:
        e = b["edge"]
        extra.append(f"edge {e:+.1%}" if e >= 0 else f"price value {e:+.1%} (this rule bets regardless)")
    if b.get("kelly_pct") is not None:
        extra.append(f"stake {b['kelly_pct']:g}% of bankroll")
    elif b.get("units") is not None:
        extra.append(f"stake {b['units']:g} unit{'' if b['units'] == 1 else 's'}")
    if extra:
        lines.append(" · ".join(extra))
    if b.get("why"):
        lines.append(str(b["why"]))
    g = next((g for g in pred.get("upcoming", []) if g.get("game_id") and g.get("game_id") == b.get("game_id")), {})
    for h in g.get("heads_up") or []:   # heads_up.py: QB reported out, books split
        lines.append(f"⚠ {h['text']}")
    names = [TRACK_NAMES.get(k, k) for k in (also_keys or [key])]
    lines.append(f"Track{'s' if len(names) > 1 else ''}: {', '.join(dict.fromkeys(names))} (paper, not yet validated)")
    return title, "\n".join(l for l in lines if l)


def bet_line(key: str, b: dict, pred: dict) -> str:
    price = b.get("price")
    px = f"{price:+d}" if isinstance(price, int) else str(price)
    what = b.get("team") or b.get("side", "")
    if b.get("market") in ("h2h", "ml"):
        what += " ML"
    elif b.get("point") is not None:
        pt = b["point"]
        what += f" {pt:+g}" if b.get("market") == "spread" else f" {pt:g}"
    if key in ("bets", "ml_v2_bets", "ml_v3_bets", "ml_v4_bets", "aplus_ml_bets", "night_west_bets") and "ML" not in what:
        what = f"{what} ML" if key != "night_west_bets" else what
    parts = [f"{TRACK_NAMES.get(key, key)}: {what} {px} @ {b.get('book', '?')}"]
    if b.get("edge") is not None:
        parts.append(f"edge {b['edge']:+.1%}")
    if b.get("kelly_pct") is not None:
        parts.append(f"stake {b['kelly_pct']:g}% of bankroll")
    elif b.get("units") is not None:
        parts.append(f"{b['units']:g}u")
    if key == "qb_news_bets" and b.get("why"):
        parts.append(b["why"])
    mu = matchup(b, pred)
    if mu:
        parts.append(mu)
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


def _dec(price) -> float:
    try:
        p = float(price)
    except (TypeError, ValueError):
        return 0.0
    return 1 + (p / 100 if p > 0 else 100 / -p) if p else 0.0


def group_same_bet(items: list[tuple[str, dict]]) -> list[tuple[str, dict, list[str]]]:
    """One alert per real-world bet: tracks that picked the same game, market and side in this run are merged
    (the best price is shown; the alert lists every track)."""
    from .placed import bet_key
    groups: dict = {}
    for k, b in items:
        gk = bet_key(b) or ("id", b.get("id"))
        groups.setdefault(gk, []).append((k, b))
    out = []
    for g in groups.values():
        k, b = max(g, key=lambda kb: (_dec(kb[1].get("price")), kb[0] in PRIORITY_TRACKS))
        out.append((k, b, [k] + [x for x, _ in g if x != k]))
    return out


from . import betslip  # noqa: E402


def new_bets(pred: dict, exclude_ids: set | None = None, max_single: int = 5) -> bool:
    """exclude_ids: 'new' bets already present before this run (blocks a watch run did not refresh).
    Up to `max_single` bets: one push each with an "I placed it" button (pre-filled log form).
    More than that: one combined push; log bets from the dashboard."""
    from .placed import log_url
    items = [(k, b) for k, b in collect(pred) if b.get("id") not in (exclude_ids or set())]
    quiet = [(k, b) for k, b in items if k in QUIET_TRACKS]
    items = [(k, b) for k, b in items if k not in QUIET_TRACKS]
    if quiet:  # blind every-player tracks: one summary push, not one per bet
        n = {}
        for k, _ in quiet:
            n[k] = n.get(k, 0) + 1
        send("New paper bets (summary)", "\n".join(f"{TRACK_NAMES.get(k, k)}: {c} paper bets" for k, c in n.items())
             + "\n\nList on the dashboard (Paper bets). Paper track, not yet validated. Not financial advice.", priority=2, tags=["football"])
    if not items:
        return bool(quiet)
    items = group_same_bet(items)
    accts = account_names()
    note = "Paper track, not yet validated. Not financial advice."
    sent = 0
    if len(items) <= max_single:
        for k, b, keys in items:
            hot = any(x in PRIORITY_TRACKS for x in keys)
            title, body = alert(k, b, pred, keys)
            if accts is not None and _norm(b.get("book", "")) not in accts:
                body += "\n(no account at this book)"
            slip = betslip.for_bet(b, pred, college=is_college(k, b))
            acts = [{"action": "view", "label": "I placed it", "url": log_url(b)},
                    {"action": "view", "label": "Dashboard", "url": DASHBOARD}]
            how = ""
            if slip:   # main button: the bet on the book's slip (enter the stake); tapping the alert opens it too
                acts = [{"action": "view", "label": betslip.label(slip), "url": slip["url"]}] + acts
                how = ("\nTap to open it on the " + slip["short"] + " bet slip; enter your stake and check the price."
                       if slip["kind"] == "slip" else f"\nOpens {slip['short']} on this game ({slip['note'] or 'no bet-slip link from this book'}); "
                       "pick the bet there.")
            ok = send(title, f"{body}{how}\n\nPrices move within hours. Not financial advice.",
                      priority=4 if hot else 3, tags=["football"], click=slip["url"] if slip else DASHBOARD,
                      actions=acts)
            sent += bool(ok)
    else:
        hot = any(x in PRIORITY_TRACKS for _, _, keys in items for x in keys)
        lines = [bet_line(k, b, pred) + (f" (also {', '.join(TRACK_NAMES.get(x, x) for x in keys if x != k)})" if len(keys) > 1 else "")
                 for k, b, keys in items]
        ok = send(f"{len(items)} new paper bets",
                  "\n".join(lines) + f"\n\nLog the ones you place from the dashboard (Log bet links). {note}",
                  priority=4 if hot else 3, tags=["football"])
        sent += bool(ok)
    print(f"notify: {'sent' if sent else 'not sent (no channel configured or all failed)'} {len(items)} bet(s) via {channels()}")
    return bool(sent)


if __name__ == "__main__":
    a = sys.argv[1:]
    ok = send(a[0] if a else "NFL predictor", a[1] if len(a) > 1 else "test", int(a[2]) if len(a) > 2 else 3,
              tags=["warning"] if len(a) > 2 and int(a[2]) >= 4 else ["football"])
    print(f"channels configured: {channels() or 'none'}; " + ("sent" if ok else "NOT sent"))
    sys.exit(0 if ok or "--soft" in a else 1)
