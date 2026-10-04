import json
from urllib.parse import parse_qs, urlparse

from nflpred import placed as PL

BODY = """### Bet ID

2026_04_IND_WAS:ml_v4:home

### Pick

WAS ML

### Book

FanDuel

### Odds you got

+180

### Line you got

_No response_

### Stake ($)

$50

### Notes

_No response_
"""


def test_parse_and_odds():
    f = PL.parse_form(BODY)
    assert f["bet_id"] == "2026_04_IND_WAS:ml_v4:home" and f["odds"] == "+180" and f["point"] == "" and f["notes"] == ""
    assert abs(PL.to_decimal("+180") - 2.8) < 1e-9 and abs(PL.to_decimal("−110") - 1.9091) < 1e-3
    assert abs(PL.to_decimal("35c") - 100 / 35) < 1e-9 and abs(PL.to_decimal("0.35") - 1 / 0.35) < 1e-9
    assert PL.to_decimal("abc") is None


def test_ingest_and_score(tmp_path):
    ev = tmp_path / "ev.json"
    ev.write_text(json.dumps({"issue": {"number": 7, "created_at": "2026-10-03T03:00:00Z", "body": BODY}}))
    log = tmp_path / "placed_bets.json"
    rec = PL.ingest(str(ev), log)
    assert rec["stake"] == 50 and rec["decimal"] == 2.8
    PL.ingest(str(ev), log)                       # edited issue replaces, not duplicates
    assert len(json.loads(log.read_text())) == 1
    paper = {"id": rec["bet_id"], "track": "moneyline_v4", "price": 188, "book": "FanDuel", "side": "home",
             "status": "graded", "result": "win", "clv": 0.03, "final": "17-24"}
    s = PL.score(rec, {paper["id"]: paper})
    assert s["status"] == "graded" and s["profit"] == 90.0
    assert s["price_vs_paper"] < 0                # +180 is worse than the logged +188
    assert abs(s["clv"] - (2.8 * (1.03 / 2.88) - 1)) < 1e-3


def test_regrade_at_other_point():
    rec = {"bet_id": "x", "decimal": 1.91, "stake": 10, "point": -3.5}
    paper = {"id": "x", "side": "home", "point": -3.0, "price": -110, "status": "graded", "result": "push", "final": "21-24"}
    s = PL.score(rec, {"x": paper})
    assert s["result"] == "loss"                  # 24-3.5 < 21 is false -> 20.5 < 21 loss
    assert PL.score({"bet_id": None, "pick": "x"}, {})["status"] == "manual"


def test_log_url_prefills_form():
    u = PL.log_url({"id": "g:ml_v4:home", "team": "WAS", "price": 188, "book": "FanDuel", "side": "home"})
    q = parse_qs(urlparse(u).query)
    assert q["template"] == ["placed_bet.yml"] and q["bet_id"] == ["g:ml_v4:home"] and q["odds"] == ["+188"]
    assert q["pick"] == ["WAS ML"]


def test_spread_tracks_preseason_and_tuesday():
    from datetime import datetime, timedelta, timezone
    from nflpred import spread_tracks as ST
    now = datetime(2026, 10, 4, 15, 0, tzinfo=timezone.utc)
    lo = {"consensus_home_margin": -2.0,
          "spreads_by_book": [{"book": "FanDuel", "home_point": 2.0, "home_price": -110, "away_point": -2.0, "away_price": -110},
                              {"book": "DraftKings", "home_point": 2.5, "home_price": -115, "away_point": -2.5, "away_price": -105}],
          "by_book": {"fanduel": {"ml": [110, -130], "sp": [2.0, -110, -110]}, "draftkings": {"ml": [115, -135], "sp": [2.5, -115, -105]},
                      "pinnacle": {"ml": [112, -125], "sp": [1.5, -105, -105]}}, "totals": {"consensus_total": 44.5}}
    wt = ST.win_totals(2026)
    home, away = sorted(wt, key=lambda t: -wt[t])[0], sorted(wt, key=lambda t: wt[t])[0]   # best preseason team at home vs worst
    g = {"game_id": "2026_05_X_Y", "season": 2026, "week": 5, "gameday": "2026-10-04", "home_team": home, "away_team": away,
         "kickoff_utc": (now + timedelta(hours=2)).isoformat(), "neutral_site": False, "context": {"live_odds": lo},
         "model_home_margin_no_inj": 1.0, "qb_change": {"home": 0, "away": 0}, "qb_status": {}}
    r = ST.load_rules("preseason_prior")
    bet = ST.preseason_evaluate(g, r, now)
    assert bet and bet["side"] == "home" and bet["point"] == 2.5 and bet["price"] == -115   # best number at your books
    g["week"] = 10
    assert ST.preseason_evaluate(g, r, now) is None
    rt = ST.load_rules("tuesday_move")
    f = ST.tuesday_features(g, lo)
    assert f["pt"] == 2.25 and abs(f["f_pin"] - (1.5 - 2.25)) < 1e-9
    tue = datetime(2026, 10, 6, 14, 15, tzinfo=timezone.utc)
    assert ST.in_window(tue, rt) and not ST.in_window(now, rt)
