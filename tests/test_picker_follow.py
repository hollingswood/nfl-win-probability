import gzip
import json
from datetime import datetime, timezone

from cfbpred import picker_follow as PF

U = timezone.utc
R = {"track": "cfb_picker_follow", "version": 1, "start": "2026-11-16T00:00+00:00", "pickers": ["Brandon Marcello"],
     "sports": ["cfb"], "books": ["draftkings", "fanduel"], "units": 1,
     "validation": {"min_bets": 100, "roi_p_below": 0.05}, "ledger": "cfb/l.json", "queue": "cfb/q.json"}
PICK = {"picker": "Brandon Marcello", "sport": "cfb", "game_id": "401", "market": "spread", "side": "Texas", "line": 3.5,
        "seen_at": "2026-11-20T15:10+00:00", "kick": "2026-11-22T00:30:00+00:00", "home": "LSU", "away": "Texas", "week": 12}
EV = {"id": "e1", "home_team": "LSU Tigers", "away_team": "Texas Longhorns", "commence_time": "2026-11-22T00:30:00Z", "bookmakers": [
    {"key": "draftkings", "title": "DraftKings", "markets": [{"key": "spreads", "outcomes": [
        {"name": "Texas Longhorns", "point": 3.5, "price": -110}, {"name": "LSU Tigers", "point": -3.5, "price": -110}]}]},
    {"key": "fanduel", "title": "FanDuel", "markets": [{"key": "spreads", "outcomes": [
        {"name": "Texas Longhorns", "point": 4.0, "price": -115}, {"name": "LSU Tigers", "point": -4.0, "price": -105}]}]},
    {"key": "betmgm", "title": "BetMGM", "markets": [{"key": "spreads", "outcomes": [
        {"name": "Texas Longhorns", "point": 5.0, "price": -110}]}]}]}


def test_queue_only_followed_pickers_after_start(tmp_path):
    rows = [PICK, {**PICK, "picker": "Someone Else"}, {**PICK, "seen_at": "2026-11-10T00:00+00:00", "game_id": "400"}]
    assert PF.queue_picks(rows, tmp_path, R) == 1
    assert PF.queue_picks(rows, tmp_path, R) == 0          # idempotent


def test_bet_at_best_number_at_your_books(tmp_path):
    f = tmp_path / "odds_2026-11-20T1520.json.gz"
    with gzip.open(f, "wt") as fh:
        json.dump([EV], fh)
    q = {PF.pick_id(PICK): PICK}
    led: list = []
    new = PF.make_bets(q, led, [f], R, datetime(2026, 11, 20, 16, tzinfo=U))
    assert len(new) == 1
    b = new[0]
    assert (b["team"], b["point"], b["price"], b["book"], b["side"]) == ("Texas Longhorns", 4.0, -115, "FanDuel", "away")   # BetMGM ignored
    assert PF.make_bets(q, led, [f], R, datetime(2026, 11, 20, 17, tzinfo=U)) == []   # once per pick


def test_record_counts_results():
    led = [{"status": "graded", "rules_version": 1, "result": "win", "profit_units": 0.909, "clv": 0.02},
           {"status": "graded", "rules_version": 1, "result": "loss", "profit_units": -1.0, "clv": -0.01},
           {"status": "skipped", "rules_version": 1}]
    r = PF.record(led, R)
    assert (r["graded"], r["wins"], r["losses"], r["skipped"]) == (2, 1, 1, 1) and not r["passed"]
