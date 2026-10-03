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
