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
NEWS_FIELDS = ("seen_at", "published", "source", "outlet", "link", "title", "player", "team", "position",
               "signal", "is_starting_qb_news", "game_week_relevant", "certainty", "source_weight")


def news_team(s: dict) -> str | None:
    """Team a signal belongs to on the dashboard: the roster-checked team when there is one (roster.py),
    else the AI's team (unchecked, or team unclear)."""
    return s.get("team_verified") or s.get("team")


def news_check(s: dict) -> str:
    """'verified' (roster match), 'unclear' (team mismatch / not found / ambiguous) or 'unchecked'."""
    st = (s.get("verify") or {}).get("roster")
    return "unchecked" if st is None else "verified" if st == "match" else "unclear"


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
        if isinstance(r, dict) and r.get("team") and _ts(r.get("seen_at")) and not r.get("source_muted"):  # muted: news_sources.py
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
            if news_team(s) not in teams:
                continue
            k = (news_team(s), (s.get("verify") or {}).get("gsis_id") or s.get("player") or s.get("title"))
            rank = (_ts(s["seen_at"]), float(s.get("certainty") or 0))
            if k not in latest or rank > (_ts(latest[k]["seen_at"]), float(latest[k].get("certainty") or 0)):
                latest[k] = s
        ranked = sorted(latest.values(), key=lambda s: (
            news_check(s) != "unclear", bool(s.get("is_starting_qb_news")), bool(s.get("game_week_relevant")),
            float(s.get("certainty") or 0), _ts(s["seen_at"])), reverse=True)[:per_game]
        if ranked:
            out[g["game_id"]] = [{**{k: s.get(k) for k in NEWS_FIELDS}, "team": news_team(s), "check": news_check(s),
                                  "team_ai": s.get("team") if news_team(s) != s.get("team") else None}
                                 for s in ranked]
    return out


def context_by_game(items: list[dict], upcoming: list[dict], now: dt.datetime | None = None,
                    days: int = 7, per_game: int = 2) -> dict[str, list[dict]]:
    """Team-level AI context (history/news_context.jsonl): play-caller changes, snap limits, illness...
    Up to 2 per game from the last 7 days, game-week relevant and most certain first."""
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=days)
    recent = [c for c in items if c.get("team") and _ts(c.get("seen_at")) and cutoff <= _ts(c["seen_at"]) <= now + dt.timedelta(hours=1)]
    out = {}
    for g in upcoming:
        teams = {g.get("home_team"), g.get("away_team")}
        cs = sorted((c for c in recent if c["team"] in teams), key=lambda c: (
            bool(c.get("game_week_relevant")), float(c.get("certainty") or 0), _ts(c["seen_at"])), reverse=True)
        seen, keep = set(), []
        for c in cs:
            k = (c["team"], c.get("category"), c.get("player"))
            if k not in seen:
                seen.add(k)
                keep.append({k2: c.get(k2) for k2 in ("team", "category", "summary", "player", "direction", "source",
                                                       "link", "seen_at", "certainty")})
        if keep:
            out[g["game_id"]] = keep[:per_game]
    return out


def load_news_audit(path: Path) -> dict | None:
    """Compact accuracy summary of the AI news reader (history/news_audit.json, written by full runs) plus the
    per-source record (history/news_sources.json, news_sources.py)."""
    try:
        a = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    out = {k: a.get(k) for k in ("generated_at", "overall", "qb", "rule_population", "promotion",
                                 "lead_vs_report_hours", "roster_check")}
    try:
        src = json.loads(path.with_name("news_sources.json").read_text()).get("sources", {})
        out["sources"] = [dict(v, name=k) for k, v in list(src.items())[:15]]
    except (OSError, json.JSONDecodeError):
        pass
    return out


RULE_FILES = {"ml_v1": "betting_rules.json", "spread": "spread_rules.json", "ml_v2": "moneyline_v2_rules.json",
              "ml_v3": "moneyline_v3_rules.json", "ml_v4": "moneyline_v4_rules.json",
              "totals": "totals_wind_rules.json", "night": "night_west_rules.json",
              "totals_eu": "totals_early_under_rules.json", "props_rec": "props_receptions_rules.json", "props_un": "props_unders_rules.json",
              "aplus_ml": "grade_aplus_ml_rules.json", "aplus_sp": "grade_aplus_spread_rules.json",
              "aplus_tot": "grade_aplus_totals_rules.json", "exch": "exchange_value_rules.json",
              "pre": "preseason_prior_rules.json", "tue": "tuesday_move_rules.json"}


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
                  rules: dict | None = None, news_audit: dict | None = None,
                  news_context: list[dict] | None = None) -> dict:
    return {"predictions": predictions, "backtest": backtest, "vs_vegas": vs_vegas,
            "news_ai": news_by_game(news_signals or [], predictions.get("upcoming", []), now),
            "news_audit": news_audit,
            "news_context": context_by_game(news_context or [], predictions.get("upcoming", []), now),
            "rules": load_rule_limits() if rules is None else rules,
            "placed": load_placed(), "log_urls": log_urls(predictions), "my_books": load_my_books()}


def load_my_books() -> list:
    try:
        return json.loads((ROOT / "my_books.json").read_text()).get("allowed_books", [])
    except (OSError, ValueError):
        return []


def load_placed() -> dict | None:
    """Real bets you logged (history/placed_bets.json), scored against their paper bets."""
    try:
        from nflpred import placed
        return placed.summary(ROOT / "history")
    except Exception as e:
        print("placed bets: skipped:", e)
        return None


def log_urls(predictions: dict) -> dict:
    """bet id -> pre-filled 'I placed a bet' form, for every open paper bet."""
    try:
        from nflpred.placed import log_url
    except Exception:
        return {}
    out = {}
    for k, block in predictions.items():
        if k.endswith("_bets") and isinstance(block, dict):
            for b in block.get("open") or []:
                if b.get("id"):
                    out[b["id"]] = log_url(b)
    return out


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


def add_prior_versions(pred: dict, history: Path = ROOT / "history") -> dict:
    """For each paper track, summarize bets placed under EARLIER rules versions (they are not counted toward the
    current version's validation, so the track record would otherwise look empty after a version bump).
    Ledger = history/paper_bets_<key without _bets>.json (paper_bets.json for the v1 moneyline 'bets')."""
    for key, B in pred.items():
        if not (key.endswith("_bets") and isinstance(B, dict) and B.get("rules_version") is not None):
            continue
        path = history / ("paper_bets.json" if key == "bets" else f"paper_bets_{key[:-5]}.json")
        try:
            ledger = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cur, out = B["rules_version"], {}
        for b in ledger:
            v = b.get("rules_version")
            if v is None or v == cur:
                continue
            x = out.setdefault(v, {"version": v, "bets": 0, "open": 0, "graded": 0, "wins": 0, "losses": 0, "pushes": 0,
                                   "profit_units": 0.0, "clv": []})
            x["bets"] += 1
            if b.get("status") == "open":
                x["open"] += 1
            elif b.get("status") == "graded":
                x["graded"] += 1
                res = b.get("result")
                x["wins"] += res == "win"; x["losses"] += res == "loss"; x["pushes"] += res == "push"
                x["profit_units"] += float(b.get("profit_units") or 0)
                if isinstance(b.get("clv"), (int, float)):
                    x["clv"].append(b["clv"])
        if out:
            B["prior_versions"] = [dict({k: v for k, v in x.items() if k != "clv"}, profit_units=round(x["profit_units"], 2),
                                        avg_clv=round(sum(x["clv"]) / len(x["clv"]), 4) if x["clv"] else None)
                                   for _, x in sorted(out.items())]
    return pred


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(ROOT / "output" / "predictions.json"))
    ap.add_argument("--news", default=str(ROOT / "history" / "news_llm.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "site"))
    a = ap.parse_args(argv)
    vv = ROOT / "output" / "vs_vegas.json"
    payload = build_payload(
        add_prior_versions(json.loads(Path(a.predictions).read_text())),
        json.loads((ROOT / "output" / "backtest.json").read_text()),
        json.loads(vv.read_text()) if vv.exists() else None,
        load_news_signals(Path(a.news)),
        news_audit=load_news_audit(Path(a.news).with_name("news_audit.json")),
        news_context=load_news_signals(Path(a.news).with_name("news_context.jsonl")))
    print("wrote", write_site(render(payload), Path(a.out)))
    cfb = build_cfb(Path(a.out))
    if cfb:
        print("wrote", cfb)
    return 0


def build_cfb(site: Path) -> Path | None:
    """College football page (site/cfb/index.html) from output/cfb_predictions.json; shares the NFL page's styles."""
    src = ROOT / "output" / "cfb_predictions.json"
    if not src.exists():
        return None
    nfl = (ROOT / "scripts" / "dashboard_template.html").read_text()
    style = nfl.split("<style>", 1)[1].split("</style>", 1)[0]
    tpl = (ROOT / "scripts" / "cfb_template.html").read_text().replace("__STYLE__", style)
    data = json.loads(src.read_text())
    try:  # pre-filled "I placed it" forms for open college paper bets (scored on the NFL page's My bets)
        from nflpred.placed import log_url
        data["log_urls"] = {b["id"]: log_url(b) for k in ("cfb_shop_bets", "cfb_ml_bets", "cfb_body_clock_bets")
                            for b in ((data.get(k) or {}).get("open") or []) if b.get("id")}
    except Exception as e:
        print("cfb log links skipped:", e)
    body = tpl.replace("__DATA__", json.dumps(clean_json(data), allow_nan=False).replace("</", "<\\/"))
    return write_site(body, site / "cfb")


if __name__ == "__main__":
    sys.exit(main())
