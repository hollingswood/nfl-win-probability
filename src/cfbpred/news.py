"""College football AI news reader (logging only, like the NFL one): free feeds -> Claude Haiku -> structured
QB / availability signals and team context, logged with the time we first saw them.

history/cfb/news_llm.jsonl (availability), history/cfb/news_context.jsonl (team context), history/cfb/news_seen.json.
Team names must be CFBD school names (validated against the FBS list). Runs from the hourly odds watch on the same
cadence as the college odds (hourly Thu-Sat, every 3 h otherwise) to keep the API cost to a few dollars a month.
Nothing here places or changes a bet.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from nflpred import news_llm as N

from . import data as D

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history" / "cfb"
Q = lambda s: "https://news.google.com/rss/search?q=" + urllib.parse.quote(s) + "&hl=en-US&gl=US&ceid=US:en"
FEEDS = {
    "espn_cfb": "https://www.espn.com/espn/rss/ncf/news",
    "cbs_cfb": "https://www.cbssports.com/rss/headlines/college-football/",
    "yahoo_cfb": "https://sports.yahoo.com/college-football/rss/",
    "google_cfb_qb": Q("college football quarterback (start OR benched OR injury OR \"ruled out\" OR \"out for\" OR suspended) when:1d"),
    "google_cfb_injury": Q("college football (injury OR \"ruled out\" OR questionable OR \"opt out\" OR suspended OR \"availability report\") when:1d"),
    "google_cfb_context": Q("college football (\"offensive coordinator\" OR \"play-caller\" OR fired OR \"interim coach\" OR illness OR flu) when:1d"),
}


def fbs_teams() -> list[str]:
    now = datetime.now(timezone.utc)
    y = now.year if now.month >= 7 else now.year - 1
    for yy in (y, y - 1):
        t = D._load(f"teams_fbs_{yy}.json.gz")
        if t:
            return sorted(x["school"] for x in t)
    return []


def due(now: datetime) -> bool:
    from .odds_live import due as odds_due
    return odds_due(now)


def call_claude(items: list[dict], api_key: str, teams: list[str], timeout: float = 90):
    system = ("You extract COLLEGE FOOTBALL (FBS) injury, quarterback and availability news for a betting model. "
              "Only report facts stated in the items; never guess. The team field must be EXACTLY one of these school "
              "names: " + "; ".join(teams) + ". Skip NFL, recruiting and high-school items.")
    instr = N.INSTRUCTIONS.replace("NFL", "college football")
    body = "\n\n".join(f"[{i}] ({it['source']}, {it.get('published') or 'time unknown'}) {it['title']}\n{it['summary']}"
                       for i, it in enumerate(items))
    payload = {"model": N.MODEL, "max_tokens": 4000, "system": system,
               "messages": [{"role": "user", "content": instr + "\n\nITEMS:\n" + body}]}
    req = urllib.request.Request(N.API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0)) if m else {}
    except Exception:
        obj = {}
    ts = set(teams)
    av = [x for x in (obj.get("availability") or []) if isinstance(x, dict) and x.get("team") in ts and x.get("player")]
    cx = [x for x in (obj.get("context") or []) if isinstance(x, dict) and x.get("team") in ts and x.get("category") in N.CONTEXT_CATEGORIES]
    return av, cx


def scan(now: datetime | None = None, api_key: str | None = None, fetcher=N.fetch_feed, llm=call_claude,
         max_items: int = 120, force: bool = False) -> dict:
    now = now or datetime.now(timezone.utc)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    rep = {"feeds": {}, "new_items": 0, "signals": 0, "context_items": 0}
    if not api_key:
        rep["skipped"] = "ANTHROPIC_API_KEY not set"; return rep
    if not (force or due(now)):
        rep["skipped"] = "not due this hour"; return rep
    teams = fbs_teams()
    HIST.mkdir(parents=True, exist_ok=True)
    seen_p = HIST / "news_seen.json"
    seen = set(json.loads(seen_p.read_text())) if seen_p.exists() else set()
    new = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url); rep["feeds"][name] = len(items)
        except Exception as e:
            rep["feeds"][name] = f"failed: {str(e)[:80]}"; continue
        for it in items:
            k = N._key(it)
            if k not in seen and k not in {x["key"] for x in new}:
                new.append(dict(it, source=name, key=k))
    rep["new_items"] = len(new)
    new.sort(key=lambda x: x.get("published") or "", reverse=True)
    sig, ctx = [], []
    for i in range(0, min(len(new), max_items), 30):
        batch = new[i:i + 30]
        try:
            av, cx = llm(batch, api_key, teams)
            for kind, rows, out in (("av", av, sig), ("cx", cx, ctx)):
                for s in rows:
                    idx = s.get("item")
                    src = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else {}
                    keep = ("player", "team", "position", "signal", "is_starting_qb_news", "game_week_relevant", "certainty", "quote") if kind == "av" \
                        else ("team", "category", "summary", "player", "direction", "game_week_relevant", "certainty", "quote")
                    out.append({"seen_at": now.isoformat(timespec="minutes"), "published": src.get("published"),
                                "source": src.get("source"), "link": src.get("link"), "title": src.get("title"),
                                **{k: s.get(k) for k in keep}})
            seen.update(x["key"] for x in batch)
        except Exception as e:
            rep.setdefault("errors", []).append(str(e)[:160])
    for path, rows in ((HIST / "news_llm.jsonl", sig), (HIST / "news_context.jsonl", ctx)):
        if rows:
            with path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
    rep["signals"], rep["context_items"] = len(sig), len(ctx)
    rep["qb_signals"] = [f"{s['player']} ({s['team']}): {s['signal']}" for s in sig if s.get("is_starting_qb_news")][:20]
    return rep


def recent_by_team(days: int = 7, now: datetime | None = None) -> dict:
    """team -> recent availability/context items (newest first), for the college page."""
    now = now or datetime.now(timezone.utc)
    out: dict = {}
    for name, kind in (("news_llm.jsonl", "av"), ("news_context.jsonl", "cx")):
        p = HIST / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines()[-3000:]:
            try:
                r = json.loads(line)
                t = datetime.fromisoformat(r["seen_at"])
            except Exception:
                continue
            if (now - t).days > days or not r.get("game_week_relevant", True):
                continue
            out.setdefault(r["team"], []).append({**r, "kind": kind})
    for t in out:
        out[t] = sorted(out[t], key=lambda r: r["seen_at"], reverse=True)[:6]
    return out


if __name__ == "__main__":
    print(json.dumps(scan(force="--force" in sys.argv), indent=2))
