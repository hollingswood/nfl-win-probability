"""Odds API credit budget (2026-11-02: plan downgraded to 20,000 credits a month; normal use ~550-650 a day).

Reads the balance the last call reported (history/odds_credits.json: x-requests-used / -remaining; plan = their sum)
and the billing cycle (state history/odds_budget_state.json: the cycle starts when `used` drops, i.e. the plan
renewed; it renews on the same day each month).

Phone alerts, once per cycle each: 50 / 75 / 90% of the plan used, and "on pace to run out" (projected use at the
current daily rate > the plan, checked after 3 days of the cycle).

Throttle (`allow`), so a tight month degrades gracefully instead of every odds track stopping at once:
  ok        everything runs as scheduled
  tight     projected to exceed the plan, or < 15% left: skip the futures snapshots (logging only) and college
            odds outside its busy / Pinnacle-posting windows
  critical  < 4% left: also NFL odds only every other hour
"""
from __future__ import annotations

import calendar
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
THRESHOLDS = (0.5, 0.75, 0.9)


def _t(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def next_renewal(start: datetime) -> datetime:
    y, m = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    return start.replace(year=y, month=m, day=min(start.day, calendar.monthrange(y, m)[1]))


def status(hist: Path = HIST, now: datetime | None = None, update: bool = False) -> dict | None:
    now = now or datetime.now(timezone.utc)
    c = _read(hist / "odds_credits.json")
    try:
        used, left = int(float(c["x-requests-used"])), int(float(c["x-requests-remaining"]))
    except (KeyError, TypeError, ValueError):
        return None
    plan = used + left
    st = _read(hist / "odds_budget_state.json")
    start = _t(st.get("cycle_start"))
    if start is None:                                                   # first run: assume it renewed a day ago
        start = now - timedelta(days=1)
        st = {"cycle_start": start.isoformat(timespec="minutes"), "alerted": []}
    elif used < int(st.get("last_used", used)):                         # renewed (or the plan changed)
        due = next_renewal(start)
        start = due if abs((now - due).total_seconds()) < 3 * 86400 else (_t(c.get("checked_at")) or now)
        st = {"cycle_start": start.isoformat(timespec="minutes"), "alerted": []}
    while next_renewal(start) <= now:                                  # missed a renewal while idle
        start = next_renewal(start)
    end = next_renewal(start)
    days = max(0.25, (now - start).total_seconds() / 86400)
    per_day = used / days
    proj = used + per_day * max(0.0, (end - now).total_seconds() / 86400)
    tier = "critical" if left < 0.04 * plan else "tight" if (left < 0.15 * plan or (days >= 3 and proj > plan)) else "ok"
    out = {"plan": plan, "used": used, "left": left, "used_pct": round(used / plan, 3) if plan else None,
           "per_day": round(per_day), "projected": round(proj), "cycle_start": start.isoformat(timespec="minutes"),
           "renews": end.isoformat(timespec="minutes"), "tier": tier, "checked_at": now.isoformat(timespec="minutes")}
    if update:
        st.update(cycle_start=out["cycle_start"], last_used=used, last=out)
        (hist / "odds_budget_state.json").write_text(json.dumps(st, indent=1))
    return out


def allow(kind: str, hist: Path = HIST, now: datetime | None = None, peak: bool = False) -> bool:
    """kind: 'futures' | 'cfb' (peak=True inside its busy / posting windows) | 'nfl'. Unknown balance -> allowed."""
    now = now or datetime.now(timezone.utc)
    s = status(hist, now)
    tier = s["tier"] if s else "ok"
    if tier == "ok":
        return True
    if kind == "futures":
        return False
    if kind == "cfb":
        return peak and tier == "tight"
    if kind == "nfl":
        return tier == "tight" or now.hour % 2 == 0
    return True


def check(hist: Path = HIST, now: datetime | None = None, send=None) -> dict | None:
    """Update the cycle state and send each alert once per cycle."""
    s = status(hist, now, update=True)
    if not s or not s["plan"]:
        return s
    st = _read(hist / "odds_budget_state.json")
    done = set(st.get("alerted") or [])
    msgs = []
    for th in THRESHOLDS:
        if s["used"] >= th * s["plan"] and str(th) not in done:
            done.add(str(th))
            msgs.append(f"{th:.0%} of this month's {s['plan']:,} Odds API credits used ({s['used']:,}); "
                        f"on pace for {s['projected']:,} by {s['renews'][:10]}.")
    days = ((now or datetime.now(timezone.utc)) - _t(s["cycle_start"])).total_seconds() / 86400
    if days >= 3 and s["projected"] > s["plan"] and "pace" not in done:
        done.add("pace")
        msgs.append(f"Odds API on pace to run out: {s['per_day']:,}/day, {s['projected']:,} projected vs the "
                    f"{s['plan']:,} plan. Low-priority fetches (futures, off-peak college) are paused until it renews "
                    f"{s['renews'][:10]}.")
    if msgs:
        try:
            if send is None:
                from .notify import send
            send("Odds API credits", "\n".join(msgs), priority=4, tags=["moneybag"])
        except Exception as e:
            print("odds budget: alert failed:", e)
        st = _read(hist / "odds_budget_state.json")
        st["alerted"] = sorted(done)
        (hist / "odds_budget_state.json").write_text(json.dumps(st, indent=1))
    return s
