"""Render output/*.json into a static dashboard.

Writes:
  site/index.html      full HTML document (GitHub Pages)
  site/artifact.html   body-only version (for publishing as a Claude artifact)

    python scripts/build_dashboard.py [--predictions PATH] [--news PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nflpred.model import clean_json  # noqa: E402

NEWS_DAYS = 7
NEWS_PER_GAME = 3
NEWS_FIELDS = ("seen_at", "published", "source", "link", "title", "player", "team", "position",
               "signal", "is_starting_qb_news", "game_week_relevant", "certainty")


def _ts(s: str | None) -> dt.datetime | None:
    try:
        t = dt.datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def load_news_signals(path: Path) -> list[dict]:
    """AI-read news signals (history/news_llm.jsonl, one JSON object per line). Bad lines are skipped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("team") and _ts(r.get("seen_at")):
            out.append(r)
    return out


def news_by_game(signals: list[dict], upcoming: list[dict], now: dt.datetime | None = None,
                 days: int = NEWS_DAYS, per_game: int = NEWS_PER_GAME) -> dict[str, list[dict]]:
    """For each upcoming game, up to 3 signals about either team seen in the last `days`,
    preferring starting-QB news, then game-week relevance, certainty and recency. One entry per
    (team, player): the most recently seen one (higher certainty breaks ties)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=days)
    recent = [s for s in signals if cutoff <= _ts(s["seen_at"]) <= now + dt.timedelta(hours=1)]
    out = {}
    for g in upcoming:
        teams = {g.get("home_team"), g.get("away_team")}
        latest: dict[tuple, dict] = {}
        for s in recent:
            if s.get("team") not in teams:
                continue
            k = (s.get("team"), s.get("player") or s.get("title"))
            rank = (_ts(s["seen_at"]), float(s.get("certainty") or 0))
            if k not in latest or rank > (_ts(latest[k]["seen_at"]), float(latest[k].get("certainty") or 0)):
                latest[k] = s
        ranked = sorted(latest.values(), key=lambda s: (
            bool(s.get("is_starting_qb_news")), bool(s.get("game_week_relevant")),
            float(s.get("certainty") or 0), _ts(s["seen_at"])), reverse=True)[:per_game]
        if ranked:
            out[g["game_id"]] = [{k: s.get(k) for k in NEWS_FIELDS} for s in ranked]
    return out


RULE_FILES = {"ml_v1": "betting_rules.json", "spread": "spread_rules.json", "ml_v2": "moneyline_v2_rules.json",
              "ml_v3": "moneyline_v3_rules.json", "ml_v4": "moneyline_v4_rules.json",
              "totals": "totals_wind_rules.json", "night": "night_west_rules.json",
              "totals_eu": "totals_early_under_rules.json", "props_rec": "props_receptions_rules.json"}


def load_rule_limits(root: Path = ROOT) -> dict:
    """Display-only facts from the (read-only) rules files: each track's allowed price range and,
    for window-based tracks, their betting windows (UTC) and hours-before-kickoff limits. Missing/bad files
    are skipped."""
    out = {}
    for key, name in RULE_FILES.items():
        try:
            q = json.loads((root / name).read_text()).get("qualify") or {}
        except (OSError, json.JSONDecodeError):
            continue
        r = {"min_odds": q.get("min_american_odds"), "max_odds": q.get("max_american_odds")}
        if q.get("windows_utc"):
            r["windows_utc"] = q["windows_utc"]
            r["window_tolerance_minutes"] = q.get("window_tolerance_minutes", 0)
        if q.get("bet_within_hours_of_kickoff") is not None:
            r["bet_within_hours"] = q["bet_within_hours_of_kickoff"]
        if q.get("min_hours_before_kickoff") is not None:
            r["min_hours_before"] = q["min_hours_before_kickoff"]
            r["max_hours_before"] = q.get("max_hours_before_kickoff")
        if q.get("friday_window_utc"):
            r["window_tolerance_minutes"] = q.get("window_tolerance_minutes", 0)
        out[key] = r
    return out


def build_payload(predictions: dict, backtest: dict, vs_vegas: dict | None,
                  news_signals: list[dict] | None, now: dt.datetime | None = None,
                  rules: dict | None = None) -> dict:
    return {"predictions": predictions, "backtest": backtest, "vs_vegas": vs_vegas,
            "news_ai": news_by_game(news_signals or [], predictions.get("upcoming", []), now),
            "rules": load_rule_limits() if rules is None else rules}


def render(payload: dict, template: str | None = None) -> str:
    tpl = template if template is not None else (ROOT / "scripts" / "dashboard_template.html").read_text()
    return tpl.replace("__DATA__", json.dumps(clean_json(payload), allow_nan=False).replace("</", "<\\/"))


def write_site(body: str, site: Path) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    (site / "artifact.html").write_text(body)
    (site / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>'
        + body + "</body></html>")
    return site / "index.html"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(ROOT / "output" / "predictions.json"))
    ap.add_argument("--news", default=str(ROOT / "history" / "news_llm.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    a = ap.parse_args(argv)
    vv = ROOT / "output" / "vs_vegas.json"
    payload = build_payload(
        json.loads(Path(a.predictions).read_text()),
        json.loads((ROOT / "output" / "backtest.json").read_text()),
        json.loads(vv.read_text()) if vv.exists() else None,
        load_news_signals(Path(a.news)))
    print("wrote", write_site(render(payload), Path(a.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
