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


def test_cfb_ml_candidates():
    from cfbpred import tracks as T
    r = T.load_rules()
    ev = {"id": "e", "home_team": "Texas Longhorns", "away_team": "Baylor Bears", "commence_time": "2026-10-10T20:00:00Z",
          "bookmakers": [
              {"key": "pinnacle", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Texas Longhorns", "price": -300}, {"name": "Baylor Bears", "price": 250}]},
                  {"key": "spreads", "outcomes": [{"name": "Texas Longhorns", "point": -7.5, "price": -105}, {"name": "Baylor Bears", "point": 7.5, "price": -105}]}]},
              {"key": "fanduel", "title": "FanDuel", "markets": [
                  {"key": "h2h", "outcomes": [{"name": "Texas Longhorns", "price": -320}, {"name": "Baylor Bears", "price": 300}]}]}]}
    c = T.candidates(ev, r)
    rules = {x["rule"] for x in c if x["team"] == "Baylor Bears"}
    assert "P1" in rules and all(x["price"] == 300 and x["book"] == "FanDuel" for x in c)
    assert not [x for x in c if x["team"] == "Texas Longhorns"]


def test_cfb_news_scan_logs_valid_teams(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from cfbpred import news as CN
    monkeypatch.setattr(CN, "HIST", tmp_path)
    monkeypatch.setattr(CN, "fbs_teams", lambda: ["Texas", "Alabama"])
    items = [{"title": "Texas QB ruled out", "link": "http://x/1", "published": "2026-10-08T12:00+00:00", "summary": "..."}]
    def llm(batch, key, teams):
        return ([{"item": 0, "player": "QB One", "team": "Texas", "signal": "out", "is_starting_qb_news": True,
                  "game_week_relevant": True, "certainty": 1.0, "quote": "ruled out"},
                 {"item": 0, "player": "X", "team": "Nowhere State", "signal": "out"}][:1], [])
    rep = CN.scan(now=datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc), api_key="k", fetcher=lambda u: items, llm=llm, force=True)
    assert rep["signals"] == 1 and rep["qb_signals"] == ["QB One (Texas): out"]
    rec = CN.recent_by_team(7, datetime(2026, 10, 9, tzinfo=timezone.utc))
    assert rec["Texas"][0]["player"] == "QB One"
    assert CN.scan(now=datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc), api_key="k", fetcher=lambda u: items, llm=llm, force=True)["new_items"] == 0


def test_cfb_weather_window_and_backfill(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import pandas as pd
    from cfbpred import weather as CW
    ko = datetime(2024, 10, 5, 19, 30, tzinfo=timezone.utc)
    hrs = [f"2024-10-05T{h:02d}:00" for h in range(24)]
    pay = {"hourly": {"time": hrs, "wind_speed_10m": [10.0] * 24, "wind_speed_10m_previous_day2": [h * 1.0 for h in range(24)],
                      "wind_gusts_10m_previous_day1": [30.0] * 24, "temperature_2m": [50.0] * 24, "precipitation": [0.01] * 24}}
    w = CW.window(pay, ko)
    assert w["d0"]["wind_mph"] == 10.0 and w["d2"]["wind_mph"] == 20.5     # hours 19..22
    assert w["d1"]["gust_mph"] == 30.0 and w["d0"]["precip_in"] == 0.04 and w["d1"]["wind_mph"] is None
    monkeypatch.setattr(CW, "WDIR", tmp_path)
    monkeypatch.setattr(CW, "venues", lambda: {1: {"lat": 40, "lon": -80, "dome": False}, 2: {"lat": 40, "lon": -80, "dome": True}})
    monkeypatch.setattr(CW, "_tbd", lambda s: set())
    G = pd.DataFrame([{"game_id": 11, "start": pd.Timestamp(ko), "completed": True, "home_div": "fbs", "away_div": "fbs", "venue_id": 1},
                      {"game_id": 12, "start": pd.Timestamp(ko), "completed": True, "home_div": "fbs", "away_div": "fbs", "venue_id": 2}])
    monkeypatch.setattr(CW.D, "games", lambda ys: G)
    calls = []
    rep = CW.backfill([2024], sleep=0, getter=lambda u, p: calls.append(p) or pay)
    assert rep["done"] and len(calls) == 1 and rep["skipped"]["dome"] == 1
    assert len(calls[0]["hourly"].split(",")) == 10
    store = __import__("json").loads((tmp_path / "game_weather.json").read_text())
    assert store["11"]["d2"]["wind_mph"] == 20.5 and store["12"] == {"dome": True}
    assert CW.backfill([2024], sleep=0, getter=lambda u, p: 1 / 0)["calls"] == 0   # resumable: nothing left


def test_cfb_weather_skips_weeks_without_old_forecasts(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import pandas as pd
    from cfbpred import weather as CW
    ko = datetime(2021, 9, 4, 19, 0, tzinfo=timezone.utc)
    pay = {"hourly": {"time": [f"2021-09-04T{h:02d}:00" for h in range(24)], "wind_speed_10m": [9.0] * 24}}
    monkeypatch.setattr(CW, "WDIR", tmp_path)
    monkeypatch.setattr(CW, "venues", lambda: {1: {"lat": 40, "lon": -80, "dome": False}})
    monkeypatch.setattr(CW, "_tbd", lambda s: set())
    G = pd.DataFrame([{"game_id": i, "start": pd.Timestamp(ko), "completed": True, "home_div": "fbs", "away_div": "fbs",
                       "venue_id": 1, "season_type": "regular", "week": 1} for i in range(5)])
    monkeypatch.setattr(CW.D, "games", lambda ys: G)
    calls = []
    rep = CW.backfill([2021], sleep=0, getter=lambda u, p: calls.append(1) or pay)
    assert len(calls) == 1 and rep["skipped"]["no_old_forecast"] == 4


def test_cfb_shop_candidates_and_grading(tmp_path, monkeypatch):
    from cfbpred import shop as SH
    r = SH.load_rules()
    def bk(key, sp_h, sp_hp, sp_ap, tot, op, up, mlh, mla):
        return {"key": key, "title": key, "markets": [
            {"key": "spreads", "outcomes": [{"name": "Home U", "point": sp_h, "price": sp_hp}, {"name": "Away U", "point": -sp_h, "price": sp_ap}]},
            {"key": "totals", "outcomes": [{"name": "Over", "point": tot, "price": op}, {"name": "Under", "point": tot, "price": up}]},
            {"key": "h2h", "outcomes": [{"name": "Home U", "price": mlh}, {"name": "Away U", "price": mla}]}]}
    ev = {"id": "e1", "home_team": "Home U", "away_team": "Away U", "commence_time": "2026-10-10T19:00:00Z",
          "bookmakers": [bk("pinnacle", -7, -105, -105, 50.5, -105, -105, -280, 240),
                         bk("fanduel", -5.5, -110, -110, 50.5, -110, -110, -300, 230),     # home -5.5 is well off the sharp -7
                         bk("draftkings", -7, -110, -110, 54.5, -110, -110, -300, 230)]}  # under 54.5 vs sharp 50.5
    c = {x["market"]: x for x in SH.candidates(ev, r)}
    assert c["spread"]["side"] == "home" and c["spread"]["point"] == -5.5 and c["spread"]["book_key"] == "fanduel"
    assert c["total"]["side"] == "under" and c["total"]["point"] == 54.5 and c["total"]["edge"] > 0.04
    assert "ml" not in c and 0 < c["spread"]["kelly_pct"] <= 2.0
    b = {"market": "spread", "side": "home", "point": -5.5}
    assert SH.settle(b, 6, 50) > 0 and SH.settle(b, 5, 50) < 0
    assert SH.settle({"market": "total", "side": "under", "point": 54.5}, 3, 50) > 0
    assert SH.settle({"market": "ml", "side": "away"}, -3, 50) > 0


def test_kalshi_maker_match_post_settle(tmp_path, monkeypatch):
    import json, gzip
    from datetime import datetime, timezone, timedelta
    from nflpred import kalshi_maker as K
    monkeypatch.setattr(K, "HIST", tmp_path)
    now = datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc)
    ev = {"id": "e1", "home_team": "Los Angeles Chargers", "away_team": "Los Angeles Rams", "commence_time": "2026-10-11T20:05:00Z",
          "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [{"name": "Los Angeles Chargers", "price": -150}, {"name": "Los Angeles Rams", "price": 135}]}]}]}
    with gzip.open(tmp_path / "odds_2026-10-09T1730.json.gz", "wt") as f:
        json.dump([ev], f)
    mk = [{"ticker": "KXNFLGAME-X-LAC", "title": "Los Angeles R at Los Angeles C", "yes_sub_title": "Los Angeles C", "yes_bid": 50, "yes_ask": 60,
           "expected_expiration_time": "2026-10-11T23:30:00Z"},
          {"ticker": "KXNFLGAME-X-LAR", "title": "Los Angeles R at Los Angeles C", "yes_sub_title": "Los Angeles R", "yes_bid_dollars": "0.38", "yes_ask_dollars": "0.45",
           "expected_expiration_time": "2026-10-11T23:30:00Z"}]
    assert K.match_market(mk[0], [ev])[1] == "home" and K.match_market(mk[1], [ev])[1] == "away"
    def getter(url):
        if "/markets?" in url:
            return {"markets": mk if "KXNFLGAME" in url else []}
        return {"trades": [{"yes_price": 52}, {"yes_price": 50}]}
    out = K.process(now=now, getter=getter)
    act = out["active"]
    lac = next(o for o in act if o["side"] == "home")
    fair = K.shin(-150, 135)
    assert lac["bid_cents"] == int(fair * 100) - 3 and 50 < lac["bid_cents"] < 60
    # next run: a trade at 50 printed below a bid of ~54 -> filled (through)
    out2 = K.process(now=now + timedelta(hours=1), getter=getter)
    assert out2["report"]["filled"] >= 1
    assert any(b["status"] == "filled" and b["fill"] == "through" for b in out2["open"])
