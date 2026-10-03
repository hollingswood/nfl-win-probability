import gzip
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from nflpred import exchanges as X
from nflpred import futures as F
from nflpred import notify as N

R = X.load_rules()


def _ev(kalshi_home=-120, kalshi_away=110, px_home=-105, px_away=125, pin_home=-120, pin_away=104, upd="2026-10-04T15:00:00Z"):
    def bk(key, h, a):
        return {"key": key, "markets": [{"key": "h2h", "last_update": upd, "outcomes": [
            {"name": "Philadelphia Eagles", "price": h}, {"name": "Los Angeles Rams", "price": a}]}]}
    return {"id": "e1", "home_team": "Philadelphia Eagles", "away_team": "Los Angeles Rams",
            "commence_time": "2026-10-04T17:00:00Z",
            "bookmakers": [bk("kalshi", kalshi_home, kalshi_away), bk("prophetx", px_home, px_away),
                           bk("pinnacle", pin_home, pin_away), bk("fanduel", -125, 105)]}


def test_fee_models():
    v = R["venues"]
    assert abs(X.all_in_cost(0.5, v["kalshi"]) - 0.5175) < 1e-9
    assert abs(X.all_in_cost(0.5, v["robinhood"]) - 0.52) < 1e-9   # 1c commission + exchange fee capped at 1c
    assert abs(X.all_in_cost(0.5, v["polymarket"]) - 0.5125) < 1e-9
    d = 1 / X.all_in_cost(0.5, v["prophetx"])                       # 2% of winnings
    assert abs(d - 1.98) < 1e-9
    assert X.all_in_cost(0.4, v["novig"]) == 0.4


def test_quotes_and_bet(tmp_path):
    ev = _ev()
    snap = datetime(2026, 10, 4, 15, 10, tzinfo=timezone.utc)
    q = X.quotes(ev, R, snap)
    rh = [r for r in q["h2h"] if r["venue"] == "robinhood" and r["side"] == "home"][0]
    k = [r for r in q["h2h"] if r["venue"] == "kalshi" and r["side"] == "home"][0]
    assert rh["all_in_cents"] > k["all_in_cents"]                    # Robinhood = Kalshi + extra fee
    game = {"game_id": "2026_04_LA_PHI", "home_team": "PHI", "away_team": "LA", "season": 2026, "week": 4, "gameday": "2026-10-04"}
    bets = X.evaluate(game, ev, R, snap, snap, set())
    assert len(bets) == 1 and bets[0]["venue"] == "prophetx" and bets[0]["side"] == "away"
    assert bets[0]["edge"] >= R["qualify"]["min_ev"]
    # stale exchange quote -> no bet
    assert X.evaluate(game, _ev(upd="2026-10-04T10:00:00Z"), R, snap, snap, set()) == []
    # grading: write a pre-kickoff snapshot for the closing price, then grade a win
    with gzip.open(tmp_path / "odds_2026-10-04T1650.json.gz", "wt") as f:
        f.write(json.dumps([_ev(pin_home=-130, pin_away=112)]))
    games = pd.DataFrame([{"game_id": "2026_04_LA_PHI", "completed": True, "home_score": 20, "away_score": 24}])
    g = X.grade(bets[0], games, X.snapshot_files(tmp_path), {})
    assert g["result"] == "win" and g["profit_units"] > 0 and "clv" in g


def test_spread_and_total_grading():
    base = {"units": 1.0, "all_in_cost": 0.5, "game_id": "g", "kickoff_utc": "2026-10-04T17:00:00+00:00", "event_id": "x"}
    games = pd.DataFrame([{"game_id": "g", "completed": True, "home_score": 24, "away_score": 21}])
    assert X.grade({**base, "market": "spreads", "side": "home", "point": -3.0}, games, [], {})["result"] == "push"
    assert X.grade({**base, "market": "spreads", "side": "away", "point": 3.5}, games, [], {})["result"] == "win"
    assert X.grade({**base, "market": "totals", "side": "under", "point": 44.5}, games, [], {})["result"] == "loss"


def test_notify_noop_without_topic(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    pred = {"upcoming": [], "ml_v4_bets": {"new": [{"id": "a", "team": "WAS", "price": 188, "book": "FanDuel", "edge": 0.024, "units": 0.32}]}}
    assert N.new_bets(pred) is False
    line = N.bet_line("ml_v4_bets", pred["ml_v4_bets"]["new"][0], pred)
    assert "WAS ML +188 @ FanDuel" in line and "+2.4%" in line
    assert N.collect(pred) and not [b for _, b in N.collect(pred) if b["id"] not in N.new_ids(pred)]


def test_notify_sends_only_unseen(monkeypatch):
    sent = []
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.setattr(N, "send", lambda *a, **k: sent.append((a, k)) or True)
    pred = {"upcoming": [], "aplus_ml_bets": {"new": [{"id": "a", "team": "X", "price": 120, "book": "B"}]}}
    assert N.new_bets(pred, exclude_ids={"a"}) is False and not sent
    assert N.new_bets(pred) is True and sent[0][1]["priority"] == 4


def test_futures_due(tmp_path, monkeypatch):
    now = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(F, "HIST", tmp_path)   # never read the real news log
    assert F.due(now, {}) == "scheduled"
    assert F.due(now, {"last": (now - timedelta(hours=2)).isoformat()}) is None
    log = tmp_path / "news_llm.jsonl"
    log.write_text(json.dumps({"seen_at": (now - timedelta(minutes=30)).isoformat(), "player": "Caleb Williams", "team": "CHI",
                               "signal": "out", "is_starting_qb_news": True}) + "\n")
    monkeypatch.setattr(F, "HIST", tmp_path)
    assert F.due(now, {"last": (now - timedelta(hours=2)).isoformat()}).startswith("news: Caleb Williams")
    assert F.due(now, {"last": (now - timedelta(minutes=20)).isoformat()}) is None   # at most one extra per hour


def test_backfill_stops_at_deadline(tmp_path, monkeypatch):
    from nflpred import odds_history as OH
    monkeypatch.setenv("BACKFILL_DEADLINE", "1")   # long past
    calls = []
    res = OH.backfill(None, [2025], "k", out_dir=tmp_path, planner=lambda g, s: [datetime(2025, 9, 7, tzinfo=timezone.utc)],
                      fetcher=lambda *a: calls.append(a) or ({}, 100.0))
    assert res["stopped"].startswith("time limit") and not calls


def test_cfb_plan_and_cadence():
    from cfbpred import odds_history as CH, odds_live as CL
    t = CH.plan(2025)
    assert 300 < len(t) < 500 and all(x.minute == 10 for x in t)
    assert CL.due(datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc))          # Saturday: hourly
    assert not CL.due(datetime(2026, 10, 6, 17, 0, tzinfo=timezone.utc))      # Tuesday 17:00: skip
    assert CL.due(datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc))          # every 3rd hour
