"""Totals grade, early-week under track and receptions-props track (all frozen 2026-10-02)."""
import gzip
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


# ------------------------------------------------------------------------------------------- fixtures
def _tot_event(quotes, commence="2026-10-11T17:00:00Z", eid="ev1"):
    """quotes: {book_key: (point, over, under)} -> one Odds API event with totals only."""
    return {"id": eid, "home_team": "Chicago Bears", "away_team": "Green Bay Packers", "commence_time": commence,
            "bookmakers": [{"key": k, "title": k.title(), "markets": [
                {"key": "h2h", "outcomes": [{"name": "Chicago Bears", "price": -120},
                                            {"name": "Green Bay Packers", "price": 100}]},
                {"key": "totals", "outcomes": [{"name": "Over", "point": pt, "price": o},
                                               {"name": "Under", "point": pt, "price": u}]}]}
                for k, (pt, o, u) in quotes.items()]}


QUOTES = {"lowvig": (45.5, -105, -105), "betonlineag": (45.5, -102, -108), "draftkings": (46.5, -110, -110),
          "fanduel": (45.5, -112, -108), "betmgm": (46.0, -110, -110), "pinnacle": (40.5, -105, -105),
          "williamhill_us": (45.5, -110, -110)}


def _tot_game(lo, kickoff="2026-10-11T17:00:00+00:00"):
    return {"game_id": "2026_05_GB_CHI", "season": 2026, "week": 5, "gameday": kickoff[:10], "home_team": "CHI",
            "away_team": "GB", "kickoff_utc": kickoff, "context": {"live_odds": lo}}


def _live(quotes, **kw):
    from nflpred import odds as O
    return next(iter(O.summarize([_tot_event(quotes, **kw)], {"draftkings", "fanduel", "betmgm", "williamhill_us"}).values()))


# ------------------------------------------------------------------------------------------- odds summary
def test_odds_summary_keeps_event_id_and_every_books_total_quote():
    lo = _live(QUOTES)
    assert lo["event_id"] == "ev1"
    q = {r[0]: r for r in lo["totals"]["all_quotes"]}
    assert set(q) == set(QUOTES) and q["draftkings"][2:] == [46.5, -110, -110]      # all books, incl. Pinnacle
    assert {b["key"] for b in lo["totals"]["totals_by_book"]} == {"draftkings", "fanduel", "betmgm", "williamhill_us"}


def test_fetch_event_props_is_one_market_one_region(monkeypatch):
    from nflpred import odds as O
    seen = {}

    def fake_get(params, timeout, url=O.URL):
        seen.update(params=params, url=url)
        return {"id": "abc"}
    monkeypatch.setattr(O, "_get", fake_get)
    assert O.fetch_event_props("abc", api_key="k") == {"id": "abc"}
    assert seen["url"].endswith("/events/abc/odds")
    assert seen["params"]["regions"] == "us" and seen["params"]["markets"] == "player_receptions"   # ~1 credit
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        O.fetch_event_props("abc")


# ------------------------------------------------------------------------------------------- totals grade
def test_totals_grade_frozen_copy_and_letters():
    from nflpred import grade_totals as GT
    fz = json.loads(GT.MODEL_PATH.read_text())
    research = ROOT / "output" / "research" / "grade_totals_frozen.json"
    if research.exists():   # verbatim copy of the frozen research model
        assert json.loads(research.read_text()) == fz
    th = fz["thresholds_pred_clv"]
    assert th == {"A+": 0.0, "A": -0.01, "B": -0.02}
    assert [GT.letter(x, th) for x in (0.01, 0.0, -0.005, -0.01, -0.015, -0.02, -0.05)] == \
        ["A+", "A+", "A", "A", "B", "B", "C"]
    assert GT.letter(None, th) is None and GT.letter(float("nan"), th) is None
    assert fz["feature_order"] == ["ev_sharp", "ev_cons", "sharp_missing", "pt_adv_cons", "pt_adv_sharp", "key_right",
                                   "on_key", "key_cross", "p_imp", "move_mu", "move_pts", "disp", "log_hours",
                                   "is_over", "tot_level", "best_gap"]


def test_totals_grade_research_filters_and_features():
    from nflpred import grade_totals as GT, totals as T
    d = T.default_dist()
    raw = [["pinnacle", "Pin", 45.5, -105, -105],          # left out (not in the research feed)
           ["lowvig", "LowVig", 45.5, -105, -105], ["betonlineag", "BOL", 46.5, -110, -110],
           ["draftkings", "DK", 45.5, -110, -110], ["fanduel", "FD", 45.5, -300, 220],   # price out of range
           ["betmgm", "MGM", 52.5, -110, -110],             # > 5 from the median point
           ["bovada", "Bov", 45.5, -110, -130]]             # overround 1.089 (<= 1.12): kept
    rows = GT.research_rows(raw)
    assert [r["key"] for r in rows] == ["lowvig", "betonlineag", "draftkings", "bovada"]
    R = GT.refs(rows, d)
    mus = [GT.implied_mu(d, r["point"], r["nv_over"]) for r in rows]
    assert R["mu_cons"] == pytest.approx(statistics.median(mus))
    assert R["mu_sharp"] == pytest.approx(statistics.median(mus[:2])) and R["pt_sharp"] == 46.0
    assert R["disp"] == pytest.approx(statistics.stdev(mus)) and R["pt_cons"] == 45.5
    assert abs(GT.implied_mu(d, 45.5, 0.5) - d.implied_mu(45.5, 0.5)) < 0.05      # grid == bisection inversion
    first = {"mu_cons": R["mu_cons"] + 1.0, "pt_cons": 46.5}
    F, raw_ev = GT.offer_features("under", 46.5, -110, R, 130.0, first, best_ev_cons=0.05, dist=d)
    assert F["ev_cons"] == pytest.approx(d.ev(R["mu_cons"], 46.5, -110, "under"))
    assert F["ev_sharp"] == pytest.approx(d.ev(R["mu_sharp"], 46.5, -110, "under"))
    assert F["pt_adv_cons"] == pytest.approx(1.0) and F["pt_adv_sharp"] == pytest.approx(0.5)   # under: higher is better
    assert F["move_mu"] == pytest.approx(1.0) and F["move_pts"] == pytest.approx(1.0)    # total fell = toward under
    assert F["is_over"] == 0.0 and F["log_hours"] == pytest.approx(math.log1p(130))
    assert F["tot_level"] == pytest.approx(R["mu_cons"] - 44) and F["p_imp"] == pytest.approx(110 / 210)
    assert F["best_gap"] == pytest.approx(min(max(0.05 - raw_ev["ev_cons_raw"], 0), 0.1))
    assert GT.key_feats(43.5, "over", 44.5) == (1.0, 0.0, 1.0)     # over at 44-0.5, crosses 44 in our favour
    assert GT.key_feats(44.5, "over", 43.5) == (-1.0, 0.0, -1.0)
    assert GT.key_feats(44.0, "under", 44.0)[1] == 1.0
    F2, _ = GT.offer_features("over", 45.5, -110, dict(R, mu_sharp=None, pt_sharp=None, disp=None), 2.0, None, None, d)
    assert F2["sharp_missing"] == 1.0 and F2["ev_sharp"] == F2["ev_cons"] and F2["disp"] == 0.5
    assert F2["move_mu"] == 0.0 and F2["best_gap"] == 0.0


def test_totals_grade_attach_first_seen_and_bet_label(tmp_path):
    pytest.importorskip("lightgbm")
    from nflpred import grade_totals as GT
    # an earlier snapshot 6 days out (within 7 days) with a higher total -> first seen; one 8 days out is ignored
    for ts, pt in (("2026-10-03T1500", 49.5), ("2026-10-05T1410", 47.5), ("2026-10-06T1410", 46.5)):
        q = {k: (pt, -110, -110) for k in ("lowvig", "draftkings", "fanduel")}
        with gzip.open(tmp_path / f"odds_{ts}.json.gz", "wt") as f:
            f.write(json.dumps([_tot_event(q)]))
    lo = _live(QUOTES)
    g = _tot_game(lo)
    first = GT.first_seen(tmp_path, [g])
    assert first[g["game_id"]]["ts"].startswith("2026-10-05T14:10") and first[g["game_id"]]["pt_cons"] == 47.5
    now = datetime(2026, 10, 6, 14, 15, tzinfo=UTC)
    s = GT.attach({"upcoming": [g]}, tmp_path, now)
    assert s["available"] and s["graded_games"] == 1 and s["errors"] == 0
    G = g["totals_grade"]
    assert G["grade"] == GT.letter(G["predicted_clv"], s["thresholds"]) and G["book_key"] in GT.SHARP | {
        "draftkings", "fanduel", "betmgm", "williamhill_us"} and G["book_key"] not in GT.SHARP
    assert len(g["totals_grade_offers"]) == 2 * 4 and all(o[4] <= G["predicted_clv"] for o in g["totals_grade_offers"])
    assert set(g["totals_grade_sides"]) == {"over", "under"}
    lab = GT.bet_fields(g, G["side"], G["book_key"], G["point"], G["price"])
    assert lab == {"totals_grade": G["grade"], "predicted_clv": G["predicted_clv"]}
    assert GT.bet_fields(g, "under", "draftkings", 99.5, -110) == {"totals_grade": None, "predicted_clv": None}
    far = _tot_game(lo, kickoff="2026-10-20T17:00:00+00:00")                    # > 7 days out: not graded
    GT.attach({"upcoming": [far]}, tmp_path, now)
    assert far["totals_grade"] is None and far["totals_grade_check"]
    off = _tot_game(lo)
    assert GT.attach({"upcoming": [off]}, tmp_path, now, model=False)["available"] is False and off["totals_grade"] is None
    rows = GT.by_grade([{"status": "graded", "totals_grade": "A+", "units": 1, "result": "loss", "profit_units": -1,
                         "clv": 0.02, "predicted_clv": 0.004}])
    assert rows[0]["grade"] == "A+" and rows[0]["roi"] == -1 and rows[0]["avg_clv"] == 0.02


# ------------------------------------------------------------------------------------------- early-week unders
def test_early_under_rule_window_and_grading(tmp_path):
    from nflpred import totals_early_under as EU, totals as T
    r = EU.load_rules()
    assert r["version"] == 1 and r["qualify"]["windows_utc"] == [["Tue", "14:10"]]
    d = T.default_dist()
    lo = _live(QUOTES)   # sharp fair ~45.5; DraftKings under 46.5 -110 beats it, BetMGM 46.0 less so
    tue = datetime(2026, 10, 6, 14, 30, tzinfo=UTC)                              # 122.5 h before kickoff
    g = _tot_game(lo)
    bet = EU.evaluate(g, r, d, tue)
    assert bet and bet["book_key"] == "draftkings" and bet["point"] == 46.5 and bet["side"] == "under"
    assert bet["edge"] > 0 and bet["units"] == 1.0 and 96 <= bet["hours_before"] <= 168
    assert bet["edge"] == g["early_under"]["ev_sharp"] and abs(bet["sharp_total"] - 45.5) < 0.6
    wed = datetime(2026, 10, 7, 14, 30, tzinfo=UTC)
    g2 = _tot_game(lo)
    assert EU.evaluate(g2, r, d, wed) is None and any("only bets at the Tue" in x for x in g2["early_under_check"])
    late = datetime(2026, 10, 6, 15, 5, tzinfo=UTC)                              # 55 min after 14:10: outside
    assert EU.evaluate(_tot_game(lo), r, d, late) is None
    thu = _tot_game(lo, kickoff="2026-10-09T00:15:00+00:00")                     # Thursday game: 58 h at Tue
    assert EU.evaluate(thu, r, d, tue) is None and any("less than 96 h" in x for x in thu["early_under_check"])
    flat = {k: (45.5, -110, -110) for k in ("lowvig", "betonlineag", "draftkings", "fanduel")}
    g3 = _tot_game(_live(flat))
    assert EU.evaluate(g3, r, d, tue) is None and any("vs the sharp fair total" in x for x in g3["early_under_check"])
    nosharp = _tot_game(_live({"draftkings": (46.5, -110, -110), "fanduel": (45.5, -110, -110)}))
    assert EU.evaluate(nosharp, r, d, tue) is None and "no sharp" in nosharp["early_under_check"][0]
    # process: one bet per game, ledger + verdicts; grading vs our own closing snapshot
    pred = {"upcoming": [_tot_game(lo)]}
    no_games = pd.DataFrame(columns=["game_id", "completed"])
    res = EU.process(pred, no_games, tmp_path, r, now=tue)
    assert len(res["new"]) == 1 and pred["upcoming"][0]["early_under"]["verdict"] == "bet"
    assert EU.process({"upcoming": [_tot_game(lo)]}, no_games, tmp_path, r, now=tue)["new"] == []
    close = {k: (44.5, -110, -110) for k in ("lowvig", "betonlineag", "draftkings")}
    with gzip.open(tmp_path / "odds_2026-10-11T1545.json.gz", "wt") as f:
        f.write(json.dumps([_tot_event(close)]))
    games = pd.DataFrame([{"game_id": "2026_05_GB_CHI", "completed": True, "home_score": 28, "away_score": 21,
                           "home_team": "CHI", "away_team": "GB", "gameday": pd.Timestamp("2026-10-11")}])
    res = EU.process({"upcoming": []}, games, tmp_path, r, now=tue)
    b = res["recent_graded"][0]
    assert b["result"] == "loss" and b["final"] == "49 pts" and b["clv"] > 0 and b["clv_sharp_close"] > 0
    assert res["record"]["graded"] == 1 and res["mode"] == "shadow"


# ------------------------------------------------------------------------------------------- receptions props
def _props_resp(rows, eid="ev1"):
    """rows: [(book, player, point, over, under)]"""
    bks = {}
    for b, pl, pt, o, u in rows:
        bks.setdefault(b, []).extend([{"name": "Over", "description": pl, "price": o, "point": pt},
                                      {"name": "Under", "description": pl, "price": u, "point": pt}])
    return {"id": eid, "bookmakers": [{"key": b, "title": b.title(), "markets": [
        {"key": "player_receptions", "outcomes": oc}]} for b, oc in bks.items()]}


PROPS = [("draftkings", "Amon-Ra St. Brown", 6.5, 120, -150),     # over priced well above the others
         ("fanduel", "Amon-Ra St. Brown", 6.5, -110, -120), ("betmgm", "Amon-Ra St. Brown", 6.5, -115, -115),
         ("bovada", "Amon-Ra St. Brown", 6.5, -110, -120), ("betonlineag", "Amon-Ra St. Brown", 6.5, -112, -118),
         ("williamhill_us", "Amon-Ra St. Brown", 7.5, 300, -450),    # milestone/alt quote: dropped
         ("draftkings", "Sam LaPorta", 4.5, -115, -115), ("fanduel", "Sam LaPorta", 4.5, -115, -115),
         ("betmgm", "Sam LaPorta", 4.5, -115, -115),                  # only 2 others: no fair price
         ("draftkings", "Jameson Williams", 3.5, 400, -150), ("fanduel", "Jameson Williams", 3.5, -110, -110),
         ("betmgm", "Jameson Williams", 3.5, -110, -110), ("bovada", "Jameson Williams", 3.5, -110, -110)]
ALLOWED = {"draftkings", "fanduel", "betmgm", "williamhill_us"}


def test_props_windows():
    from nflpred import props_receptions as PR
    r = PR.load_rules()
    sun = datetime(2026, 10, 11, 17, 0, tzinfo=UTC)
    assert PR.early_time(sun, r) == datetime(2026, 10, 9, 21, 40, tzinfo=UTC)              # Friday 21:40 UTC
    assert PR.early_time(datetime(2026, 10, 11, 13, 30, tzinfo=UTC), r).weekday() == 4      # London game too
    snf = datetime(2026, 10, 12, 0, 20, tzinfo=UTC)                                          # Sunday night = Mon UTC
    assert PR.early_time(snf, r) == snf - timedelta(hours=24)
    tnf = datetime(2026, 10, 9, 0, 15, tzinfo=UTC)
    assert PR.early_time(tnf, r) == tnf - timedelta(hours=24)
    assert PR.in_early_window(datetime(2026, 10, 9, 21, 23, tzinfo=UTC), sun, r)            # hourly watch :23
    assert PR.in_early_window(datetime(2026, 10, 9, 22, 23, tzinfo=UTC), sun, r)
    assert not PR.in_early_window(datetime(2026, 10, 9, 20, 37, tzinfo=UTC), sun, r)        # Friday full run
    assert PR.in_early_window(datetime(2026, 10, 8, 0, 23, tzinfo=UTC), tnf, r)
    assert PR.in_close_window(sun - timedelta(minutes=37), sun) and not PR.in_close_window(sun - timedelta(minutes=97), sun)
    assert not PR.in_close_window(sun - timedelta(minutes=5), sun)


def test_props_leave_one_out_rule():
    from nflpred import props_receptions as PR
    r = PR.load_rules()
    q = PR.parse_event(_props_resp(PROPS), r)
    assert not any(x["book"] == "williamhill_us" for x in q)                                # main lines only
    c = PR.candidates(q, ALLOWED, r)
    dk = [x for x in c if x["player"] == "Amon-Ra St. Brown" and x["book_key"] == "draftkings" and x["side"] == "over"][0]
    others = [x["p_nv"] for x in q if x["player"] == "Amon-Ra St. Brown" and x["book"] != "draftkings"]
    assert dk["n_other"] == 4 and dk["p_fair"] == pytest.approx(statistics.median(others), abs=1e-4)
    assert dk["ev"] == pytest.approx(statistics.median(others) * 2.2 - 1)
    assert not any(x["player"] == "Sam LaPorta" for x in c)                                 # < 3 other books
    picks = PR.select(c, r)
    assert [p["player"] for p in picks] == ["Amon-Ra St. Brown"]                            # one per player
    assert picks[0]["book_key"] == "draftkings" and picks[0]["side"] == "over" and picks[0]["ev"] >= 0.02
    # Jameson Williams' best quote (+400 over) is >= 50% EV: treated as stale/erroneous, no fallback bet
    assert all(x["ev"] >= 0.5 for x in c if x["player"] == "Jameson Williams" and x["side"] == "over" and x["book_key"] == "draftkings")


def _props_game():
    return {"game_id": "2026_05_DET_KC", "season": 2026, "week": 5, "gameday": "2026-10-11", "home_team": "KC",
            "away_team": "DET", "kickoff_utc": "2026-10-11T20:25:00+00:00",
            "context": {"live_odds": {"event_id": "ev1"}}}


def test_props_process_fetches_once_then_closes_and_grades(tmp_path):
    from nflpred import props_receptions as PR
    r = PR.load_rules()
    calls = []

    def fetch(eid):
        calls.append(eid)
        return _props_resp(PROPS if len(calls) == 1 else [  # close: DK over 6.5 now -140 everywhere
            (b, "Amon-Ra St. Brown", 6.5, -140, 110) for b in ("draftkings", "fanduel", "betmgm", "bovada")])
    no_games = pd.DataFrame(columns=["game_id", "completed"])
    fri = datetime(2026, 10, 9, 21, 23, tzinfo=UTC)
    pred = {"upcoming": [_props_game()]}
    res = PR.process(pred, no_games, tmp_path, r, now=fri, fetch=fetch, allowed=ALLOWED)
    assert calls == ["ev1"] and len(res["new"]) == 1 and res["api_calls"] == 1
    b = res["new"][0]
    assert b["player"] == "Amon-Ra St. Brown" and b["side"] == "over" and b["price"] == 120 and b["units"] == 1.0
    assert pred["upcoming"][0]["props_receptions"]["bets"] == 1 and list(tmp_path.glob("props_*.json.gz"))
    # second run in the same window: no new call, no duplicate bet
    res = PR.process({"upcoming": [_props_game()]}, no_games, tmp_path, r, now=fri + timedelta(hours=1), fetch=fetch,
                     allowed=ALLOWED)
    assert calls == ["ev1"] and res["new"] == []
    # outside any window: nothing; offline (fetch=None): nothing
    PR.process({"upcoming": [_props_game()]}, no_games, tmp_path, r, now=fri + timedelta(hours=10), fetch=fetch, allowed=ALLOWED)
    assert calls == ["ev1"]
    # close: ~60 min before kickoff, only because there is an open bet
    PR.process({"upcoming": [_props_game()]}, no_games, tmp_path, r, now=datetime(2026, 10, 11, 19, 23, tzinfo=UTC),
               fetch=fetch, allowed=ALLOWED)
    assert calls == ["ev1", "ev1"]
    games = pd.DataFrame([{"game_id": "2026_05_DET_KC", "completed": True}])

    def stats_for(season, rec=8):
        st = pd.DataFrame([{"game_id": "2026_05_DET_KC", "player_display_name": "Amon-Ra St. Brown",
                            "player_name": "A.St. Brown", "receptions": rec}])
        return st, pd.DataFrame(columns=["game_id", "player", "offense_snaps"])
    res = PR.process({"upcoming": []}, games, tmp_path, r, now=datetime(2026, 10, 13, 14, 17, tzinfo=UTC),
                     stats_for=stats_for, allowed=ALLOWED)
    g = res["recent_graded"][0]
    assert g["result"] == "win" and g["receptions"] == 8 and g["profit_units"] == 1.2
    p_close = 1 / 1.7142857 / (1 / 1.7142857 + 100 / 210)                         # over -140 / under +110 no-vig
    assert g["closing_prob"] == pytest.approx(p_close, abs=1e-3) and g["clv"] == pytest.approx(p_close * 2.2 - 1, abs=2e-3)
    assert res["record"]["graded"] == 1


def test_player_stats_matching_and_settlement():
    from nflpred import player_stats as PS, props_receptions as PR
    assert PS.norm_name("Amon-Ra St. Brown") == "amon ra st brown" == PS.norm_name("Amon-Ra St. Brown Jr.")
    assert PS.norm_name("Marvin Harrison Jr.") == "marvin harrison"
    cands = {1: {"deonte harty"}, 2: {"damien harris"}, 3: {"kenneth walker"}}
    assert PS.match("Kenneth Walker III", cands) == 3 and PS.match("Deonte Harris", cands) is None
    assert PS.match("K. Walker", {3: {"kenneth walker"}, 4: {"travis kelce"}}) == 3
    stats = pd.DataFrame([{"game_id": "g", "player_display_name": "Travis Kelce", "player_name": "T.Kelce", "receptions": 5}])
    snaps = pd.DataFrame([{"game_id": "g", "player": "Noah Gray", "offense_snaps": 30},
                          {"game_id": "g", "player": "Jared Wiley", "offense_snaps": 0}])
    assert PS.receptions("g", "Travis Kelce", stats, snaps) == ("graded", 5)
    assert PS.receptions("g", "Noah Gray", stats, snaps) == ("graded", 0)          # played, no stat line
    assert PS.receptions("g", "Jared Wiley", stats, snaps)[0] == "void"             # no offensive snap
    assert PS.receptions("g", "Rashee Rice", stats, snaps)[0] == "void"
    assert PS.receptions("g", "Noah Gray", stats, snaps.iloc[0:0])[0] == "open"     # snap counts not out yet
    assert PS.receptions("h", "Travis Kelce", stats, snaps)[0] == "open"            # game not in stats yet
    r = PR.load_rules()
    bet = {"game_id": "g", "season": 2026, "gameday": "2026-10-11", "player": "Rashee Rice", "side": "under",
           "point": 4.5, "price": -110, "units": 1.0, "status": "open"}
    games = pd.DataFrame([{"game_id": "g", "completed": True}])
    out = PR.grade(bet, games, [], r, lambda s: (stats, snaps), datetime(2026, 10, 13, tzinfo=UTC))
    assert out["status"] == "void" and "snap" in out["void_reason"]
    out = PR.grade(dict(bet, player="Travis Kelce"), games, [], r, lambda s: (stats, snaps), datetime(2026, 10, 13, tzinfo=UTC))
    assert out["result"] == "loss" and "clv" not in out and out["clv_source"] == "no closing props snapshot"


# ------------------------------------------------------------------------------------------- pipeline glue
def test_alert_lines_only_for_live_tracks():
    from nflpred import pipeline as P
    eu = {"team": "GB@CHI UNDER", "point": 46.5, "price": -110, "book": "DraftKings", "gameday": "2026-10-11",
          "edge": 0.012, "units": 1.0}
    pr = {"player": "Amon-Ra St. Brown", "side": "over", "point": 6.5, "price": 120, "book": "DraftKings",
          "opponent": "DET@KC", "gameday": "2026-10-11", "edge": 0.05, "units": 1.0}
    res = {"totals_early_under_bets": {"mode": "shadow", "new": [eu]}, "props_receptions_bets": {"mode": "shadow", "new": [pr]}}
    assert P.alert_lines(res) == []
    res["totals_early_under_bets"]["mode"] = res["props_receptions_bets"]["mode"] = "live"
    lines = P.alert_lines(res)
    assert len(lines) == 2 and "early-week under" in lines[0] and "Amon-Ra St. Brown over 6.5 receptions" in lines[1]


def test_new_rules_files_are_frozen_and_complete():
    for name, track in (("totals_early_under_rules.json", "totals_early_under"),
                        ("props_receptions_rules.json", "props_receptions")):
        r = json.loads((ROOT / name).read_text())
        assert r["track"] == track and r["version"] == 1 and r["frozen_on"] == "2026-10-02"
        assert {"min_bets", "avg_clv_positive_with_p_below", "realized_roi_above"} <= set(r["validation"])
        assert r["ledger"].startswith("paper_bets_") and r["sizing"]["units"] == 1.0
    assert "new pre-registered hypothesis" in json.loads((ROOT / "totals_early_under_rules.json").read_text())["note"].lower()


def test_dashboard_renders_totals_grade_and_new_tracks():
    import re
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "scripts" / "build_dashboard.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    lim = bd.load_rule_limits()
    assert lim["totals_eu"]["windows_utc"] == [["Tue", "14:10"]] and lim["totals_eu"]["min_hours_before"] == 96
    assert lim["props_rec"]["window_tolerance_minutes"] == 50
    tg = {"side": "under", "point": 46.5, "price": -110, "book": "DraftKings", "book_key": "draftkings",
          "predicted_clv": 0.004, "grade": "A+", "grading_version": 2}
    up = {"game_id": "2026_05_GB_CHI", "season": 2026, "week": 5, "gameday": "2026-10-11", "gametime": "13:00",
          "kickoff_utc": "2026-10-11T17:00:00+00:00", "home_team": "CHI", "away_team": "GB", "home_win_prob": 0.55,
          "away_win_prob": 0.45, "model_home_margin": 1.0, "vegas_home_margin": 1.5, "vegas_home_prob": 0.55,
          "injury_report": True, "top_factors": [], "totals_grade": tg, "totals_grade_sides": {"under": tg},
          "early_under": {"book": "DraftKings", "book_key": "draftkings", "point": 46.5, "price": -110, "ev_sharp": 0.01,
                          "sharp_total": 45.5, "verdict": "bet", "reasons": []},
          "props_receptions": {"early_check_at": "2026-10-09T21:40+00:00", "checked_at": None, "bets": 1},
          "context": {"live_odds": {"consensus_home_prob": 0.55, "totals": {"consensus_total": 45.8, "totals_by_book": []}}}}
    rec = {"graded": 0, "min_bets": 40, "wins": 0, "losses": 0, "profit_units": 0.0, "roi": 0.0, "avg_clv": 0.0,
           "clv_p_value": 1.0, "checks": {"enough_bets": False, "roi_positive": False, "clv_positive_and_significant": False}}
    eu_bet = {"game_id": up["game_id"], "gameday": "2026-10-11", "team": "GB@CHI UNDER", "opponent": "", "side": "under",
              "point": 46.5, "price": -110, "book": "DraftKings", "edge": 0.01, "units": 1.0,
              "placed_at": "2026-10-06T14:17+00:00", "totals_grade": "A+", "predicted_clv": 0.004}
    pr_bet = {"game_id": up["game_id"], "gameday": "2026-10-11", "team": "Amon-Ra St. Brown Over 6.5", "player": "Amon-Ra St. Brown",
              "side": "over", "point": 6.5, "price": 120, "book": "DraftKings", "edge": 0.08, "n_other": 4, "units": 1.0,
              "opponent": "GB@CHI", "placed_at": "2026-10-09T21:23+00:00"}
    pred = {"generated_at": "2026-10-06T14:17:00+00:00", "season": 2026, "upcoming": [up],
            "season_to_date": {"model": {}, "vegas": {}, "games": []}, "clv": {"games": 0}, "trained_on_games": 3000,
            "live_odds_available": True, "totals_grade": {"version": 2, "available": True, "record": {"all": [], "by_track": []}},
            "totals_early_under_bets": {"mode": "shadow", "record": rec, "open": [eu_bet], "recent_graded": [], "by_totals_grade": []},
            "props_receptions_bets": {"mode": "shadow", "record": dict(rec, min_bets=150), "open": [pr_bet], "recent_graded": []}}
    backtest = {"overall": {}, "seasons": "2018-2025", "calibration": [], "by_season": []}
    body = bd.render(bd.build_payload(pred, backtest, None, []))
    for s in ('id="tgrec"', "totals_early_under_bets", "props_receptions_bets", "Totals grade record", "Rec props"):
        assert s in body
    assert "NO GRADE YET" not in body and "no grade yet" not in body.lower()
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', body, re.S).group(1))
    assert data["predictions"]["upcoming"][0]["totals_grade"]["grade"] == "A+"
