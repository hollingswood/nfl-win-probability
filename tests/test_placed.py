import json
from datetime import datetime, timedelta, timezone

import pytest
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
    assert q["template"] == ["placed_bet.yml"] and q["bet_id"] == ["g:ml_v4:home"] and q["pick"] == ["WAS ML"]
    assert q["odds"] == ["+188"] and q["book"] == ["FanDuel"]
    body = ("### Stake ($)\n\n25\n\n### Bet ID\n\ng:ml_v4:home\n\n### Pick\n\nWAS ML\n\n### Book\n\nFanDuel\n\n### Odds\n\n+188\n\n"
            "### Different book? (optional)\n\nDraftKings\n\n### Different odds? (optional)\n\n+180\n\n### Different line? (optional)\n\n_No response_\n\n"
            "### Notes (optional)\n\n_No response_")
    rec = PL.record_from_issue({"number": 7, "body": body})
    assert rec["book"] == "DraftKings" and rec["odds"] == "+180" and rec["decimal"] == 2.8 and rec["stake"] == 25 and rec["point"] is None
    plain = body.replace("DraftKings", "_No response_").replace("+180", "_No response_")
    rec = PL.record_from_issue({"number": 8, "body": plain})
    assert rec["book"] == "FanDuel" and rec["odds"] == "+188"
    r = PL.normalize({"odds": "-142", "decimal": 1.704, "point": -140.0, "notes": ""})   # odds typed in the line field
    assert r["odds"] == "-140" and r["point"] is None and abs(r["decimal"] - 1.7143) < 1e-3
    assert PL.normalize({"odds": "-105", "point": 53.5})["point"] == 53.5


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
    idx = {"qb one": {"played": "Texas", "roster": set(), "portal": None}}
    rec = CN.recent_by_team(7, datetime(2026, 10, 9, tzinfo=timezone.utc), idx=idx)
    assert rec["Texas"][0]["player"] == "QB One"
    assert CN.recent_by_team(7, datetime(2026, 10, 9, tzinfo=timezone.utc), idx={}) == {}   # player not on any roster: left off
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
    mk = [{"ticker": "KXNFLGAME-X-LAC", "event_ticker": "KXNFLGAME-X", "title": "Los Angeles C wins", "yes_sub_title": "Los Angeles C", "yes_bid": 50, "yes_ask": 60,
           "expected_expiration_time": "2026-10-11T23:30:00Z"},
          {"ticker": "KXNFLGAME-X-LAR", "event_ticker": "KXNFLGAME-X", "title": "Los Angeles R wins", "yes_sub_title": "Los Angeles R", "yes_bid_dollars": "0.38", "yes_ask_dollars": "0.45",
           "expected_expiration_time": "2026-10-11T23:30:00Z"}]
    assert K.match_market(mk[0], [ev], mk[1])[1] == "home" and K.match_market(mk[1], [ev], mk[0])[1] == "away"
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


def test_picks_logger_and_futures_view(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from nflpred import picks_log as PL, futures_value as FV
    monkeypatch.setattr(PL, "HIST", tmp_path)
    feed = [{"title": "Week 6 NFL picks against the spread", "link": "https://x.com/a", "published": None, "summary": ""},
            {"title": "Injury update", "link": "https://x.com/b", "published": None, "summary": ""}]
    calls = []
    def llm(text, key):
        calls.append(text)
        return [{"analyst": "A", "sport": "nfl", "away_team": "Dallas Cowboys", "home_team": "Houston Texans", "market": "spread", "pick": "Houston Texans", "line": -3.5}]
    rep = PL.scan(now=datetime(2026, 10, 8, tzinfo=timezone.utc), api_key="k", fetcher=lambda u: feed, texter=lambda u: "text", llm=llm)
    assert rep["picks"] == 1 and len(calls) == 1          # same article across feeds is read once
    assert PL.scan(api_key="k", fetcher=lambda u: feed, texter=lambda u: "t", llm=llm)["picks"] == 0
    snap = {"markets": {"m": [{"bookmakers": [
        {"key": k, "title": k, "markets": [{"key": "outrights", "outcomes": [{"name": "A", "price": p}, {"name": "B", "price": -120}]}]}
        for k, p in (("draftkings", 300), ("fanduel", 140), ("betmgm", 150), ("pinnacle", 150))]}]}}
    v = FV.snapshot_view(snap, "m")
    assert v["A"]["book"] == "draftkings" and v["A"]["ev"] > 0.3 and v["A"]["n_books"] == 4


def test_cfb_lines_prefer_real_book(monkeypatch):
    from cfbpred import data as D
    games = [{"id": 1, "lines": [{"provider": "DraftKings", "spread": -2.5, "overUnder": 51.5},
                                 {"provider": "Bovada", "spread": -3.0, "overUnder": 51.0}]},
             {"id": 2, "lines": [{"provider": "Bovada", "spread": 7.0, "overUnder": 44.0}, {"provider": "teamrankings", "spread": 6.0}]}]
    monkeypatch.setattr(D, "_load", lambda name: games if "regular" in name else [])
    avg = D.lines([2026]).set_index("game_id")
    assert avg.loc[1, "spread_close"] == -2.75                       # two-provider median = a number no book posted
    L = D.lines([2026], prefer=D.PREFERRED_PROVIDERS).set_index("game_id")
    assert L.loc[1, "spread_close"] == -2.5 and L.loc[1, "total_close"] == 51.5 and L.loc[1, "line_source"] == "DraftKings"
    assert L.loc[2, "spread_close"] == 7.0 and L.loc[2, "line_source"] == "Bovada"


def test_cfb_pinnacle_prob_uses_shin():
    from cfbpred import pipeline as P
    from nflpred.devig import shin
    ev = {"home_team": "Troy Trojans", "away_team": "Southern Miss Golden Eagles", "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [{"name": "Troy Trojans", "price": -450}, {"name": "Southern Miss Golden Eagles", "price": 340}]}]}]}
    o = P.odds_view(ev, set())
    assert o["pinnacle"]["home_prob"] == round(shin(-450, 340), 4)


def test_picks_score_grades_at_close_and_hot_test(tmp_path):
    import pandas as pd
    from nflpred import picks_score as PS
    kick = pd.Timestamp("2026-10-11T17:00", tz="UTC")
    nfl = pd.DataFrame([{"game_id": f"2026_{w:02d}_DET_KC", "week": w, "home_team": "KC", "away_team": "DET", "kick": kick + pd.Timedelta(days=7 * (w - 5)),
                         "home_score": 27, "away_score": 20, "spread_line": 3.0, "total_line": 50.5, "home_moneyline": -150, "away_moneyline": 130}
                        for w in range(1, 7)])
    cfb = pd.DataFrame(columns=["home", "away", "kick", "completed", "home_pts"])
    seen = lambda w: (kick + pd.Timedelta(days=7 * (w - 5)) - pd.Timedelta(days=2)).isoformat()
    p = {"sport": "nfl", "home_team": "Kansas City Chiefs", "away_team": "Detroit Lions", "market": "spread", "pick": "Kansas City Chiefs",
         "line": -2.5, "seen_at": seen(5)}
    g = PS.grade_pick(p, nfl, cfb, lambda s: None)
    assert g["result"] == "win" and g["units"] == pytest.approx(100 / 110) and g["line_value"] == 0.5    # KC -3 at close, won by 7
    assert PS.grade_pick(dict(p, market="total", pick="under", line=51.5), nfl, cfb, lambda s: None)["result"] == "win"
    ml = PS.grade_pick(dict(p, market="moneyline", pick="Detroit Lions", line=None), nfl, cfb, lambda s: None)
    assert ml["result"] == "loss" and ml["units"] == -1.0
    assert PS.grade_pick(dict(p, seen_at=(kick + pd.Timedelta(minutes=5)).isoformat()), nfl, cfb, lambda s: None) == {"late": True}
    # hot test: 'Hot Hand' goes 9-1 over weeks 1-4, then picks week 5; 'Cold' does the opposite
    rows = []
    for w in range(1, 6):
        for i in range(2 if w < 5 else 1):
            win = not (w == 1 and i == 0)
            rows.append({**p, "analyst": "Hot Hand", "seen_at": seen(w), "pick": "Kansas City Chiefs" if win else "Detroit Lions", "line": None, "x": i})
            rows.append({**p, "analyst": "Cold", "seen_at": seen(w), "pick": "Detroit Lions", "line": None, "x": i})
    (tmp_path / "picks_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = PS.run(tmp_path, nfl=nfl, cfb=cfb)
    # repeated identical picks are de-duplicated per picker+game+market+side; one row per week per side survives
    t = out["hot_test"]
    assert out["picks_graded"] > 0 and t["passed"] is False
    assert {x["picker"] for x in out["pickers"]} == {"Hot Hand", "Cold"}


def test_picks_log_bluesky_posts_become_picks(tmp_path, monkeypatch):
    from nflpred import picks_log as PL, news_sources as NS
    monkeypatch.setattr(PL, "HIST", tmp_path)
    items = [{"title": "KC -3 (2u)", "link": "https://bsky.app/profile/capper.bsky.social/post/1", "published": "2026-10-08T15:00+00:00",
              "summary": "KC -3 (2u) vs DET, love it", "outlet": "bsky:capper.bsky.social"},
             {"title": "nice weather", "link": "https://bsky.app/profile/capper.bsky.social/post/2", "published": "2026-10-08T15:00+00:00",
              "summary": "nice weather today", "outlet": "bsky:capper.bsky.social"}]
    monkeypatch.setattr(NS, "bluesky_items", lambda acc, per_account=20: (items, {"ok": 1}))
    seen_text = []
    def llm(text, key):
        seen_text.append(text)
        return [{"analyst": "@capper.bsky.social", "sport": "nfl", "away_team": "Detroit Lions", "home_team": "Kansas City Chiefs",
                 "market": "spread", "pick": "Kansas City Chiefs", "line": -3}]
    rep = PL.scan(now=datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc), api_key="k", fetcher=lambda u: [], texter=lambda u: "",
                  llm=llm, bluesky={"capper.bsky.social": {}})
    row = json.loads((tmp_path / "picks_log.jsonl").read_text().splitlines()[0])
    assert rep["picks"] == 1 and row["source"] == "bluesky" and row["analyst"] == "@capper.bsky.social"
    assert "nice weather" not in seen_text[0] and "KC -3" in seen_text[0]
    acc = PL.bluesky_pickers(tmp_path, get=lambda m, p: {"actors": [{"handle": "a.bsky.social", "description": "NFL picks, 5u max"},
                                                                     {"handle": "b.bsky.social", "description": "dad, Bears fan"}]}
                             if m.endswith("searchActors") else {"followersCount": 900}, now=datetime(2026, 10, 8, tzinfo=timezone.utc))
    assert list(acc) == ["a.bsky.social"]


def test_cfb_odds_poll_hourly_when_pinnacle_posts():
    from cfbpred import odds_live as OL
    sun = lambda h: datetime(2026, 10, 11, h, 23, tzinfo=timezone.utc)      # Sunday
    mon = lambda h: datetime(2026, 10, 12, h, 23, tzinfo=timezone.utc)
    assert all(OL.due(sun(h)) for h in range(16, 24)) and all(OL.due(mon(h)) for h in range(13, 17))
    assert not OL.due(mon(10)) and not OL.due(datetime(2026, 10, 13, 13, 23, tzinfo=timezone.utc))   # Tue 13 UTC: every 3 h only


def test_cfb_shop_v3_over_bar_and_haircut():
    from cfbpred import shop as SH
    r2, r3 = SH.load_rules(), SH.load_rules(SH.RULES_V3)
    assert r3["track"] == "cfb_shop_v3" and r3["qualify"]["min_ev_side"]["over"] == 0.04 and r3["ledger"] != r2["ledger"]
    ev = {"home_team": "A", "away_team": "B", "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 50.5, "price": -110}, {"name": "Under", "point": 50.5, "price": -110}]}]},
        {"key": "draftkings", "markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": 49.5, "price": -110}, {"name": "Under", "point": 49.5, "price": -110}]}]}]}
    c2 = {c["side"]: c for c in SH.candidates(ev, r2)}
    c3 = {c["side"]: c for c in SH.candidates(ev, r3)}
    if "over" in c2 and c2["over"]["edge"] < 0.04:
        assert "over" not in c3                       # v3: overs need 4%+
    w, l = 0.55, 0.45
    hw, hl = SH._haircut(w, l, -110, "total", r3)
    assert hw < w and SH._haircut(w, l, -110, "total", r2) == (w, l)
    assert SH.kelly_pct(hw, hl, -110, r3) < SH.kelly_pct(w, l, -110, r3)


def test_dedupe_counts_only_runs_that_did_work_and_late_copies_skip():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("dd", Path(__file__).resolve().parents[1] / "scripts" / "dedupe_run.py")
    dd = importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
    now = datetime(2026, 10, 5, 2, 53, tzinfo=timezone.utc)                       # GitHub copy of '5 0 * * 1' arriving 2h48m late
    since = dd.slot_start("schedule", "5 0 * * 1", now)
    assert since == datetime(2026, 10, 4, 23, 50, tzinfo=timezone.utc)
    runs = [{"id": 1, "created_at": "2026-10-05T00:05:02Z", "conclusion": "success"}]
    jobs = {1: [{"name": "check", "conclusion": "success"}, {"name": "predict", "conclusion": "success"}]}
    assert dd.decide(runs, jobs.get, 9, since) is True                            # the 17:05 AZ dispatch did the slot
    jobs[1][1]["conclusion"] = "skipped"
    assert dd.decide(runs, jobs.get, 9, since) is False                           # a skipped run doesn't count (no chains)
    assert dd.decide([], jobs.get, 9, since) is False                             # nothing covered it: run late
    # outside-scheduler dispatch: 40-minute window as before
    assert dd.slot_start("workflow_dispatch", "", now) == now - timedelta(minutes=40)
    # cron later in the UTC day than now -> yesterday's slot
    assert dd.slot_start("schedule", "5 23 * * 0,1,4", datetime(2026, 10, 9, 2, 54, tzinfo=timezone.utc)).day == 8


def test_notify_sends_to_every_configured_channel(monkeypatch):
    from nflpred import notify as N
    sent = []
    monkeypatch.setattr(N, "_post", lambda url, data, headers: sent.append(url) or True)
    for k in ("NTFY_TOPIC", "PUSHOVER_USER", "PUSHOVER_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(k, raising=False)
    assert N.send("t", "m") is False and N.channels() == []
    monkeypatch.setenv("PUSHOVER_USER", "u"); monkeypatch.setenv("PUSHOVER_TOKEN", "a")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "b"); monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    assert N.send("t", "m", actions=[{"action": "view", "label": "I placed it", "url": "https://x"}]) is True
    assert any("pushover" in u for u in sent) and any("telegram" in u for u in sent) and N.channels() == ["pushover", "telegram"]


def test_cfb_tracks_use_only_your_books_and_aplus_track(tmp_path, monkeypatch):
    import json
    from cfbpred import shop as SH
    mine = set(json.loads((SH.ROOT / "my_books.json").read_text())["allowed_books"])
    r, r3, ra = SH.load_rules(), SH.load_rules(SH.RULES_V3), SH.load_rules(SH.RULES_APLUS)
    for x in (r, r3, ra):
        assert x["books"] == "my_books" and set(SH.books_of(x)) == mine
    assert r["version"] == 4 and r3["version"] == 5 and r.get("count_earlier_bets_at_my_books") and ra["track"] == "cfb_aplus" and ra["ledger"] not in (r["ledger"], r3["ledger"])
    assert all(v == 0.04 for v in ra["qualify"]["min_ev"].values()) and ra["qualify"]["price_range"] == [-200, 200]

    def bk(key, tot):
        return {"key": key, "title": key, "markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": tot, "price": -110}, {"name": "Under", "point": tot, "price": -110}]}]}
    ev = {"id": "e9", "home_team": "H", "away_team": "A", "commence_time": "2026-10-12T19:00:00Z",
          "bookmakers": [bk("pinnacle", 50.5), bk("betmgm", 56.5), bk("draftkings", 52.5)]}
    c = {x["side"]: x for x in SH.candidates(ev, r)}
    assert c["under"]["book_key"] == "draftkings"                          # BetMGM's better number is ignored (no account)
    e = c["under"]["edge"]
    ca = {x["side"]: x for x in SH.candidates(ev, ra)}
    assert ("under" in ca) == (e >= 0.04)                                   # A+ track: 4%+ only

    # since the start, your books only: an earlier-version bet at a book you don't have is neither shown nor counted;
    # one at your book counts (record, open list, one bet per game per market)
    now = datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(SH, "HIST", tmp_path)
    monkeypatch.setattr(SH, "snapshots", lambda: ["s"])
    monkeypatch.setattr(SH, "snap_time", lambda f: now)
    monkeypatch.setattr(SH, "load", lambda f: [ev])
    monkeypatch.setattr(SH, "grade", lambda ledger, files: ledger)
    other = {"id": "cfbshop:e9:total:under", "rules_version": 2, "event_id": "e9", "market": "total", "side": "under", "status": "open",
             "book_key": "betmgm", "your_book": False, "units": 1}
    won = {"id": "cfbshop:e8:spread:home", "rules_version": 2, "event_id": "e8", "market": "spread", "side": "home", "status": "graded",
           "book_key": "draftkings", "your_book": True, "units": 1, "result": "win", "profit_units": 0.91, "clv": 0.03}
    (tmp_path / r["ledger"]).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / r["ledger"]).write_text(json.dumps([other, won]))
    out = SH.process(now)
    assert [b["book_key"] for b in out["open"]] == ["draftkings"] and out["open"][0]["id"].endswith(f":v{r['version']}")
    assert out["record"]["graded"] == 1 and out["record"]["wins"] == 1 and [b["id"] for b in out["recent_graded"]] == [won["id"]]
    assert "betmgm" not in json.dumps({k: out[k] for k in ("open", "recent_graded", "record")})
    led = json.loads((tmp_path / r["ledger"]).read_text())
    assert len(led) == 3                                                    # nothing deleted from the file
    won2 = dict(won, id="x", event_id="e9", market="total", status="open")   # an earlier bet at your book blocks a second bet on that market
    (tmp_path / r["ledger"]).write_text(json.dumps([won2]))
    assert SH.process(now)["new"] == []


def test_cfb_results_refresh_every_2h_in_game_windows():
    from datetime import timedelta
    from cfbpred import pipeline as P
    sat = datetime(2026, 10, 10, 20, 0, tzinfo=timezone.utc)
    assert P.refresh_interval(sat) == timedelta(hours=2)
    assert P.refresh_interval(datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc)) == timedelta(hours=2)    # Sunday early (late West Coast finals)
    assert P.refresh_interval(datetime(2026, 10, 13, 20, 0, tzinfo=timezone.utc)) == timedelta(hours=12)  # Tuesday


def test_cfb_ml_counts_one_bet_per_game():
    from cfbpred import tracks as T
    r = T.load_rules()
    assert r["version"] >= 3
    g = lambda **k: {"status": "graded", "units": 1, **k}
    led = [g(event_id="a", rule="P1", side="home", placed_at="2026-10-05T00:24", result="win", profit_units=0.76, clv=0.03, edge=0.034),
           g(event_id="a", rule="P2", side="home", placed_at="2026-10-05T00:24", result="win", profit_units=0.76, clv=0.03, edge=0.027),
           g(event_id="b", rule="P1", side="home", placed_at="2026-10-04T18:25", result="win", profit_units=0.57, clv=0.08),
           g(event_id="b", rule="P2", side="away", placed_at="2026-10-05T15:25", result="loss", profit_units=-1, clv=-0.03)]
    one = T.one_per_game(led)
    assert len(one) == 2
    a = next(b for b in one if b["event_id"] == "a")
    assert a["rule"] == "P1+P2" and a["edge"] == 0.034
    assert next(b for b in one if b["event_id"] == "b")["side"] == "home"      # first logged counts, the later other-side bet does not
    rec = T.record(led, r)
    assert rec["graded"] == 2 and rec["wins"] == 2 and rec["by_rule"]["P2"]["graded"] == 1


def test_kalshi_maker_record_by_league():
    from nflpred import kalshi_maker as K
    r = K.load_rules()
    g = lambda sport, ticker, res: {"status": "graded", "rules_version": r["version"], "sport": sport, "ticker": ticker, "result": res,
                                    "profit": 0.4 if res == "win" else -0.6, "cost": 0.6, "clv": 0.01}
    led = [g("nfl", "KXNFLGAME-A", "win"), g("cfb", "KXNCAAFGAME-B", "loss"), {**g(None, "KXNCAAFGAME-C", "win"), "sport": None}]
    assert K.record(led, r, "nfl")["graded"] == 1 and K.record(led, r, "cfb")["graded"] == 2 and K.record(led, r)["graded"] == 3


def test_cfb_news_roster_check(tmp_path):
    import gzip, json
    from cfbpred import roster as R
    wk = [{"teams": [{"team": "Oregon", "categories": [{"types": [{"name": "C/ATT", "athletes": [{"name": "Dylan Raiola", "stat": "11/11"}]}]}]}]}]
    with gzip.open(tmp_path / "games_players_passing_2026_regular_4.json.gz", "wt") as f:
        json.dump(wk, f)
    with gzip.open(tmp_path / "roster_2026.json.gz", "wt") as f:
        json.dump([{"firstName": "Tyler", "lastName": "Brown", "team": "Iowa"}, {"firstName": "Tyler", "lastName": "Brown", "team": "James Madison"},
                   {"firstName": "Dante", "lastName": "Moore", "team": "Oregon"}], f)
    with gzip.open(tmp_path / "player_portal_2026.json.gz", "wt") as f:
        json.dump([{"firstName": "Nico", "lastName": "Iamaleava", "origin": "Tennessee", "destination": "UCLA"}], f)
    idx = R.index(2026, str(tmp_path))
    assert R.verify("Dylan Raiola", "Nebraska", 2026, idx) == {"roster": "corrected", "team_verified": "Oregon"}   # transferred
    assert R.verify("Dante Moore", "Nebraska", 2026, idx)["team_verified"] == "Oregon"                          # matchup story
    assert R.verify("Nico Iamaleava Jr.", "Tennessee", 2026, idx)["team_verified"] == "UCLA"                    # portal
    assert R.verify("Tyler Brown", "Iowa", 2026, idx)["roster"] == "match"                                      # same name, two schools
    assert R.verify("Tyler Brown", "Ohio State", 2026, idx)["team_verified"] is None
    assert R.verify("Nobody Here", "Iowa", 2026, idx) == {"roster": "not_found", "team_verified": None}
    assert R.verify(None, "Iowa", 2026, idx)["team_verified"] == "Iowa"                                         # team-level context


def test_qb_news_track(tmp_path):
    import gzip, json
    import pandas as pd
    from datetime import datetime, timezone
    from nflpred import qb_news as Q
    r = Q.load_rules()
    now = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)
    g = {"game_id": "2026_07_BUF_KC", "season": 2026, "week": 7, "gameday": "2026-10-18", "kickoff_utc": "2026-10-18T20:25:00+00:00",
         "home_team": "KC", "away_team": "BUF", "home_qb": "Patrick Mahomes", "away_qb": "Josh Allen", "qb_status": {}}
    sig = {"seen_at": "2026-10-14T17:30+00:00", "published": "2026-10-14T17:10+00:00", "player": "Patrick Mahomes", "team": "KC",
           "team_verified": "KC", "verify": {"roster": "match", "roster_source": "sleeper"}, "signal": "out", "is_starting_qb_news": True,
           "game_week_relevant": True, "certainty": 1.0, "source_weight": 0.93, "outlet": "PFT"}
    assert Q.classify(sig, g, r)[0] == "away"                                                       # starter out: bet against KC
    assert Q.classify({**sig, "player": "Mahomes"}, g, r)[0] is None                                # surname only
    assert Q.classify({**sig, "certainty": 0.7}, g, r)[0] is None
    assert Q.classify({**sig, "player": "Gardner Minshew", "signal": "will_start"}, g, r)[0] == "away"  # backup named starter
    assert Q.classify({**sig, "signal": "expected_to_play"}, g, r)[0] is None                        # not listed: no bet
    assert Q.classify({**sig, "signal": "expected_to_play"}, {**g, "qb_status": {"home": {"status": "Questionable"}}}, r)[0] == "home"

    def ev(home_pt, book_price=-110):
        bk = lambda key, pt, pr: {"key": key, "title": key, "markets": [{"key": "spreads", "outcomes": [
            {"name": "Kansas City Chiefs", "point": pt, "price": pr}, {"name": "Buffalo Bills", "point": -pt, "price": pr}]}]}
        return [{"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
                 "bookmakers": [bk("fanduel", home_pt, book_price), bk("draftkings", home_pt, book_price), bk("betmgm", home_pt, -110)]}]
    with gzip.open(tmp_path / "odds_2026-10-14T1623.json.gz", "wt") as f:
        json.dump(ev(-3.0), f)
    (tmp_path / "news_llm.jsonl").write_text(json.dumps(sig) + "\n")
    sched = pd.DataFrame([{"home_team": "KC", "away_team": "LV", "gameday": "2026-10-11", "gametime": "13:00"}])
    out = Q.process({"upcoming": [g]}, pd.DataFrame(columns=["game_id", "completed"]), tmp_path, now=now, fetch=lambda: ev(-2.8), schedule=sched)
    assert len(out["new"]) == 1 and out["new"][0]["side"] == "away" and out["new"][0]["point"] == 2.8 and out["new"][0]["book"] in ("fanduel", "draftkings")
    # the line already moved 1 point toward Buffalo: no bet
    (tmp_path / r["ledger"]).unlink()
    out = Q.process({"upcoming": [g]}, pd.DataFrame(columns=["game_id", "completed"]), tmp_path, now=now, fetch=lambda: ev(-2.0), schedule=sched)
    assert out["new"] == [] and any("already moved" in c["reason"] for c in out["checks"])
    # news published before the team's previous game ended does not count
    (tmp_path / r["ledger"]).unlink(missing_ok=True)
    old = dict(sig, published="2026-10-11T18:00+00:00")
    (tmp_path / "news_llm.jsonl").write_text(json.dumps(old) + "\n")
    out = Q.process({"upcoming": [g]}, pd.DataFrame(columns=["game_id", "completed"]), tmp_path, now=now, fetch=lambda: ev(-2.8), schedule=sched)
    assert out["new"] == []


def test_context_audit(tmp_path):
    import gzip, json
    import pandas as pd
    from datetime import datetime, timezone
    from nflpred import context_audit as CA
    sched = pd.DataFrame([{"season": 2026, "game_id": "g1", "home_team": "KC", "away_team": "BUF", "gameday": "2026-10-18",
                           "gametime": "16:25", "home_score": 20.0, "away_score": 27.0}])
    item = {"seen_at": "2026-10-14T17:30+00:00", "published": "2026-10-14T17:00+00:00", "team": "KC", "category": "offensive_line_shuffle",
            "direction": "hurts", "certainty": 0.9, "game_week_relevant": True, "summary": "two starters out"}
    (tmp_path / "news_context.jsonl").write_text(json.dumps(item) + "\n" + json.dumps({**item, "seen_at": "2026-10-15T10:00+00:00"}) + "\n")
    def snap(name, pt):
        with gzip.open(tmp_path / name, "wt") as f:
            json.dump([{"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "bookmakers": [{"key": "fanduel", "markets": [
                {"key": "spreads", "outcomes": [{"name": "Kansas City Chiefs", "point": pt, "price": -110}]}]}]}], f)
    snap("odds_2026-10-14T1623.json.gz", -3.0); snap("odds_2026-10-18T1923.json.gz", -1.5)
    out = CA.run(tmp_path, sched, now=datetime(2026, 10, 20, tzinfo=timezone.utc))
    c = out["categories"]["offensive_line_shuffle"]
    assert out["items"] == 1 and c["avg_line_move_toward"] == 1.5 and c["graded"] == 1 and c["cover_rate_vs_close"] == 1.0  # BUF +1.5 won by 7


def test_watch_results_only_when_a_game_ended(tmp_path):
    import json
    import datetime as dt
    import pandas as pd
    from nflpred import pipeline as P
    now = dt.datetime(2026, 10, 12, 4, 30, tzinfo=dt.timezone.utc)
    sched = pd.DataFrame([{"game_id": "g1", "season": 2026, "week": 5, "gameday": "2026-10-11", "home_team": "ATL", "away_team": "BAL",
                           "home_score": 23.0, "away_score": 30.0}])
    (tmp_path / "paper_bets_x.json").write_text(json.dumps([{"status": "open", "kickoff_utc": "2026-10-12T02:00:00+00:00"}]))
    assert len(P.watch_results(tmp_path, now, loader=lambda: sched)) == 0          # kicked off 2.5 h ago: not yet
    (tmp_path / "paper_bets_x.json").write_text(json.dumps([{"status": "open", "kickoff_utc": "2026-10-12T00:20:00+00:00"}]))
    g = P.watch_results(tmp_path, now, loader=lambda: sched)
    assert bool(g.completed.iloc[0]) and g.home_win.iloc[0] == 0.0


def test_placed_index_marks_same_bet_across_tracks(tmp_path):
    import json
    from nflpred import placed as PL
    (tmp_path / "paper_bets_aplus_ml.json").write_text(json.dumps([{"id": "G1:aplus_ml:away", "track": "aplus_ml", "game_id": "G1", "side": "away", "team": "WAS"}]))
    (tmp_path / "paper_bets_ml_v4.json").write_text(json.dumps([{"id": "G1:ml_v4:away", "track": "ml_v4", "game_id": "G1", "side": "away", "team": "WAS"},
                                                               {"id": "G1:ml_v4:home", "track": "ml_v4", "game_id": "G1", "side": "home", "team": "SF"}]))
    (tmp_path / "paper_bets_spread.json").write_text(json.dumps([{"id": "G1:spread:away", "track": "spread", "game_id": "G1", "side": "away", "point": 7.5}]))
    (tmp_path / "placed_bets.json").write_text(json.dumps([{"bet_id": "G1:aplus_ml:away", "pick": "WAS ML", "book": "FanDuel", "odds": "+350", "stake": 25}]))
    ix = PL.placed_index(tmp_path)
    assert set(ix) == {"G1:aplus_ml:away", "G1:ml_v4:away"}           # same game/market/side; not the other side or the spread
    assert ix["G1:ml_v4:away"]["same_bet"] and ix["G1:ml_v4:away"]["stake"] == 25


def test_picks_long_articles_read_in_chunks():
    from nflpred import picks_log as PL
    text = "x" * 40000
    parts = PL.chunks(text)
    assert len(parts) == 3 and all(len(p) <= PL.CHUNK for p in parts)
    calls = []
    def llm(part, key):
        calls.append(len(part))
        return [{"analyst": "Chris Hummer", "away_team": "Notre Dame", "home_team": "BYU", "market": "spread", "pick": "BYU", "line": 11.5}]
    got = PL.extract(text, "k", llm)
    assert len(calls) == 3 and len(got) == 1          # same pick seen in overlapping pieces is kept once


def test_alert_names_the_game():
    from nflpred import notify as N
    b = {"market": "total", "side": "over", "team": "Over", "point": 69.5, "price": -105, "book": "theScore Bet", "edge": 0.022,
         "kelly_pct": 0.57, "home": "Memphis Tigers", "away": "Navy Midshipmen", "kickoff_utc": "2026-10-16T00:00:00+00:00"}
    t, body = N.alert("cfb_shop_bets", b, {})
    assert t == "College · Over 69.5 -105" and "Navy Midshipmen @ Memphis Tigers · Thu Oct 15, 5:00pm AZ" in body
    assert "Grade A" in body and "theScore Bet" in body and "A+/top" not in t
    t2, body2 = N.alert("aplus_ml_bets", {"game_id": "2026_06_WAS_SF", "side": "away", "team": "WAS", "price": 350, "book": "FanDuel"}, {})
    assert t2 == "NFL · WAS ML +350" and body2.startswith("WAS @ SF")


def test_moneyline_clv_filled_after_early_grading():
    import pandas as pd
    from nflpred import bets as ML
    bet = {"game_id": "g1", "side": "away", "price": 170, "units": 1.0, "status": "graded", "result": "loss", "profit_units": -1.0}
    assert ML.needs_grade(bet) and not ML.needs_grade({**bet, "clv": 0.1})
    games = pd.DataFrame([{"game_id": "g1", "completed": True, "home_score": 20, "away_score": 23, "vegas_home_prob": 0.4}])
    out = ML.grade(bet, games)
    assert out["result"] == "win" and abs(out["clv"] - (2.7 * 0.6 - 1)) < 1e-3


def test_watch_results_has_closing_moneyline(tmp_path):
    import datetime as dt
    import json
    import pandas as pd
    from nflpred import pipeline as P
    (tmp_path / "paper_bets.json").write_text(json.dumps([{"status": "open", "kickoff_utc": "2026-10-11T17:00:00+00:00"}]))
    sched = pd.DataFrame([{"season": 2026, "game_id": "g1", "gameday": "2026-10-11", "home_score": 20, "away_score": 23,
                           "home_moneyline": -175, "away_moneyline": 145}])
    g = P.watch_results(tmp_path, dt.datetime(2026, 10, 12, tzinfo=dt.timezone.utc), loader=lambda: sched)
    assert 0.6 < float(g.vegas_home_prob.iloc[0]) < 0.65


def test_same_bet_from_two_tracks_is_one_alert(monkeypatch):
    from nflpred import notify as N
    sent = []
    monkeypatch.setattr(N, "send", lambda t, m, *a, **k: sent.append((t, m)) or True)
    monkeypatch.setattr(N, "account_names", lambda: None)
    b1 = {"id": "a", "game_id": "2026_07_ATL_NO", "side": "away", "team": "ATL", "price": 112, "book": "FanDuel", "units": 0.8}
    b2 = {**b1, "id": "b", "price": 110, "book": "DraftKings"}
    N.new_bets({"upcoming": [], "ml_v2_bets": {"new": [b1]}, "ml_v3_bets": {"new": [b2]}})
    assert len(sent) == 1
    t, m = sent[0]
    assert "+112" in t and "Moneyline v2, Moneyline v3" in m and "0.8 units" in m


def test_heads_up_flags_qb_out_and_book_split(tmp_path):
    import json
    from datetime import datetime, timezone
    from nflpred import heads_up as H
    (tmp_path / "news_llm.jsonl").write_text(json.dumps({
        "seen_at": "2026-10-15T17:11+00:00", "published": "2026-10-15T17:00+00:00", "player": "Jalen Hurts", "team": "PHI",
        "team_verified": "PHI", "is_starting_qb_news": True, "game_week_relevant": True, "certainty": 1.0, "signal": "out",
        "verify": {"roster": "match"}, "source": "pft", "link": "https://x"}) + "\n")
    g = {"game_id": "g", "home_team": "PHI", "away_team": "CAR", "home_qb": "Jalen Hurts", "away_qb": "Bryce Young",
         "home_prev_end": "2026-10-11T20:30+00:00", "away_prev_end": "2026-10-11T20:30+00:00",
         "context": {"live_odds": {"consensus_home_margin": -3.5, "sharp_spreads": [[2.0, -105, -105], [1.0, -105, -115]]}}}
    pred = {"upcoming": [g]}
    H.annotate(pred, tmp_path, datetime(2026, 10, 15, 18, tzinfo=timezone.utc))
    kinds = [h["kind"] for h in g["heads_up"]]
    assert kinds == ["qb_out", "split"] and "Hurts" in g["heads_up"][0]["text"] and "CAR −3.5" in g["heads_up"][1]["text"]
    g["home_prev_end"] = "2026-10-15T18:00+00:00"   # news from before the team's last game is ignored
    H.annotate(pred, tmp_path, datetime(2026, 10, 15, 18, tzinfo=timezone.utc))
    assert [h["kind"] for h in g["heads_up"]] == ["split"]
