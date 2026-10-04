"""Analyst-picks logger (logging only): explicit published picks for NFL and college games, with the time we first saw
them, so they can be scored against the closing line at season's end (picks_rules.json).

Prior evidence is against it (tout and expert picks, and fading the public, have not beaten the close in published
studies; reports/NFL and college betting edges.md), so this is a cheap test, not a signal. Nothing here bets.

Sources: public RSS feeds; articles whose titles look like picks ("picks", "best bets", "predictions", "against the
spread") are fetched in full (publisher pages only) and read by Claude Haiku, which returns only picks stated in the text
with their line. Saved to history/picks_log.jsonl (one row per pick) and history/picks_seen.json.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import news_llm as N

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "history"
FEEDS = {
    "cbs_nfl": "https://www.cbssports.com/rss/headlines/nfl/",
    "cbs_cfb": "https://www.cbssports.com/rss/headlines/college-football/",
    "yahoo_nfl": "https://sports.yahoo.com/nfl/rss/",
    "yahoo_cfb": "https://sports.yahoo.com/college-football/rss/",
    "espn_nfl": "https://www.espn.com/espn/rss/nfl/news",
    "espn_cfb": "https://www.espn.com/espn/rss/ncf/news",
    "pft": "https://www.nbcsports.com/profootballtalk.rss",
}
PICKY = re.compile(r"\b(picks?|best bets?|predictions?|against the spread|ATS|locks?|upset picks|expert)\b", re.I)
INSTR = """Below is the text of a sports article. Extract every EXPLICIT pick the author or a named analyst makes on an NFL or
college football game: a side against the spread, a moneyline winner pick, or an over/under. Only picks stated in the text;
skip straight-up win predictions that give no betting line unless they say "moneyline". Return ONLY JSON:
{"picks": [{"analyst": str or null, "outlet": str or null, "sport": "nfl" or "cfb", "away_team": str, "home_team": str,
  "market": "spread" | "moneyline" | "total", "pick": team name for spread/moneyline, or "over"/"under",
  "line": number or null (spread from the picked team's side, e.g. -3.5; or the total), "confidence_words": short quote}]}
Use full team names. Return {"picks": []} if there are none."""


def _text(url: str, timeout: float = 20, limit: int = 15000) -> str:
    req = urllib.request.Request(url, headers=N.UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(600_000).decode("utf-8", "ignore")
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    t = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", t)[:limit]


def call_claude(text: str, api_key: str, timeout: float = 90) -> list[dict]:
    payload = {"model": N.MODEL, "max_tokens": 3000, "system": "You extract published betting picks. Never invent picks.",
               "messages": [{"role": "user", "content": INSTR + "\n\nARTICLE:\n" + text}]}
    req = urllib.request.Request(N.API_URL, data=json.dumps(payload).encode(), method="POST", headers={
        "x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    out = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    m = re.search(r"\{.*\}", out, re.S)
    try:
        picks = json.loads(m.group(0)).get("picks", []) if m else []
    except Exception:
        picks = []
    return [p for p in picks if isinstance(p, dict) and p.get("market") in ("spread", "moneyline", "total") and p.get("pick")]


def scan(now: datetime | None = None, api_key: str | None = None, fetcher=N.fetch_feed, texter=_text, llm=call_claude,
         max_articles: int = 12) -> dict:
    now = now or datetime.now(timezone.utc)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    rep = {"feeds": {}, "articles": 0, "picks": 0}
    if not api_key:
        rep["skipped"] = "ANTHROPIC_API_KEY not set"; return rep
    if now.hour % 3 != 0 and texter is _text:      # live runs: every 3 hours (picks articles change slowly)
        rep["skipped"] = "not due this hour"; return rep
    seen_p = HIST / "picks_seen.json"
    seen = set(json.loads(seen_p.read_text())) if seen_p.exists() else set()
    cands = []
    for name, url in FEEDS.items():
        try:
            items = fetcher(url); rep["feeds"][name] = len(items)
        except Exception as e:
            rep["feeds"][name] = f"failed: {str(e)[:60]}"; continue
        for it in items:
            k = hashlib.sha1((it.get("link") or it.get("title", "")).encode()).hexdigest()[:16]
            if k in seen or k in {c["key"] for c in cands} or not PICKY.search(it.get("title", "")) or "news.google.com" in (it.get("link") or ""):
                continue
            cands.append(dict(it, source=name, key=k))
    rows = []
    for it in cands[:max_articles]:
        seen.add(it["key"])
        try:
            picks = llm(texter(it["link"]), api_key)
        except Exception as e:
            rep.setdefault("errors", []).append(str(e)[:100]); continue
        rep["articles"] += 1
        for p in picks:
            rows.append({"seen_at": now.isoformat(timespec="minutes"), "published": it.get("published"), "source": it["source"],
                         "title": it.get("title"), "link": it.get("link"), **p})
    if rows:
        with (HIST / "picks_log.jsonl").open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    seen_p.write_text(json.dumps(sorted(seen)[-20000:]))
    rep["picks"] = len(rows)
    return rep


if __name__ == "__main__":
    print(json.dumps(scan(), indent=1))
    sys.exit(0)
