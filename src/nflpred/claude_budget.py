"""Claude credit budget: how much of the last top-up the automation has spent, how long the rest should last, and
phone alerts before (and when) it runs out.

claude_budget.json (repo root, edit when you add credits): {"topups": [{"at": ISO time, "usd": amount}, ...]}; the
latest top-up is the one tracked. Spend = the USD logged per call in history/claude_usage.jsonl (news_llm.log_usage,
priced at Haiku 4.5 list prices), so it can differ a little from Anthropic's bill.
Alerts (once per top-up each): 50%, 75% and 90% of the top-up used, and Claude refusing calls for lack of credits
(`out_of_credits`, raised by news_llm.claude_request). State: history/claude_budget_state.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "claude_budget.json"
THRESHOLDS = (0.5, 0.75, 0.9)


def _t(s) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _usage(hist: Path) -> list[dict]:
    p = hist / "claude_usage.jsonl"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def status(hist: Path = ROOT / "history", now: datetime | None = None, config: Path | None = None) -> dict | None:
    """{topup_at, topup_usd, spent, left, per_day, days_left, empty_at, by_component} for the latest top-up."""
    now = now or datetime.now(timezone.utc)
    try:
        tops = json.loads((config or CONFIG).read_text()).get("topups") or []
    except (OSError, ValueError):
        return None
    if not tops:
        return None
    top = max(tops, key=lambda x: x["at"])
    t0, usd = _t(top["at"]), float(top["usd"])
    rows = [r for r in _usage(hist) if (_t(r.get("at")) or t0) >= t0]
    spent = sum(float(r.get("usd") or 0) for r in rows)
    by: dict = {}
    for r in rows:
        by[r.get("component", "?")] = by.get(r.get("component", "?"), 0) + float(r.get("usd") or 0)
    # daily rate: the last 3 days of the log (whole log if shorter), so one busy hour doesn't swing it
    recent = [r for r in _usage(hist) if (_t(r.get("at")) or now) >= now - timedelta(days=3)]
    span_h = max(6.0, (now - min((_t(r["at"]) for r in recent), default=now)).total_seconds() / 3600) if recent else 0
    per_day = sum(float(r.get("usd") or 0) for r in recent) / span_h * 24 if span_h else None
    left = max(0.0, usd - spent)
    days_left = left / per_day if per_day else None
    return {"topup_at": top["at"], "topup_usd": usd, "spent": round(spent, 2), "left": round(left, 2),
            "used_pct": round(spent / usd, 3) if usd else None, "per_day": round(per_day, 2) if per_day else None,
            "days_left": round(days_left, 1) if days_left is not None else None,
            "empty_at": (now + timedelta(days=days_left)).isoformat(timespec="minutes") if days_left is not None else None,
            "by_component": {k: round(v, 2) for k, v in sorted(by.items(), key=lambda kv: -kv[1])},
            "checked_at": now.isoformat(timespec="minutes")}


def _state(hist: Path) -> dict:
    try:
        return json.loads((hist / "claude_budget_state.json").read_text())
    except (OSError, ValueError):
        return {}


def _az(iso: str) -> str:
    t = _t(iso)
    return (t - timedelta(hours=7)).strftime("%a %b %-d, %-I:%M%p").replace("AM", "am").replace("PM", "pm") + " AZ" if t else "?"


def check(hist: Path = ROOT / "history", now: datetime | None = None, send=None) -> dict | None:
    """Phone alert the first time spend crosses 50/75/90% of the latest top-up."""
    s = status(hist, now)
    if not s:
        return None
    st = _state(hist)
    if st.get("topup_at") != s["topup_at"]:
        st = {"topup_at": s["topup_at"], "alerted": []}
    for th in THRESHOLDS:
        if s["used_pct"] is not None and s["used_pct"] >= th and th not in st["alerted"]:
            if send is None:
                from .notify import send
            send(f"Claude credits {int(th * 100)}% used", f"${s['spent']:.2f} of ${s['topup_usd']:.0f} spent since "
                 f"{_az(s['topup_at'])}. About ${s['per_day']:.2f}/day, so roughly {s['days_left']} days left "
                 f"(empty around {_az(s['empty_at'])}). Top up at console.anthropic.com, then add it to claude_budget.json.",
                 priority=4 if th >= 0.9 else 3, tags=["warning"])
            st["alerted"].append(th)
    st["last"] = s
    (hist / "claude_budget_state.json").write_text(json.dumps(st, indent=1))
    return s


def out_of_credits(component: str, detail: str, hist: Path = ROOT / "history", send=None) -> None:
    """Claude refused a call for lack of credits: one phone alert per top-up, and the time it happened is kept."""
    st = _state(hist)
    s0 = status(hist)
    if s0 and st.get("topup_at") != s0["topup_at"]:      # a new top-up: forget the last one's alerts
        st = {"topup_at": s0["topup_at"], "alerted": []}
    if st.get("empty_at"):
        return
    st["empty_at"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    st["empty_detail"] = f"{component}: {detail[:200]}"
    s = s0
    if send is None:
        from .notify import send
    send("Claude credits ran out", f"The AI news and picks readers stopped at {_az(st['empty_at'])}"
         + (f", ${s['spent']:.2f} after the ${s['topup_usd']:.0f} top-up on {_az(s['topup_at'])}" if s else "")
         + ". Everything else keeps running. Top up at console.anthropic.com, then add it to claude_budget.json.",
         priority=4, tags=["warning"])
    (hist / "claude_budget_state.json").write_text(json.dumps(st, indent=1))
