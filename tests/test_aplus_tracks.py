"""A+ grade paper tracks (grade_aplus.py, frozen 2026-10-02): selection, locking, grading, validation, alerts."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from nflpred import grade_aplus as GA

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
KICK = "2026-10-11T20:25:00+00:00"
NOW = datetime(2026, 10, 9, 14, 10, tzinfo=UTC)          # ~54 h before kickoff
NO_GAMES = pd.DataFrame(columns=["game_id", "completed"])


def _game(**kw):
    g = {"game_id": "2026_05_SF_SEA", "season": 2026, "week": 5, "gameday": "2026-10-11", "home_team": "SEA",
         "away_team": "SF", "kickoff_utc": KICK, "home_win_prob": 0.62, "model_home_margin": 9.5, "context": {}}
    g.update(kw)
    return g


def _ml_grade(grade="A+", price=-120, side="home", pc=0.031):
    return {"side": side, "team": "SEA" if side == "home" else "SF", "price": price, "book": "FanDuel",
            "predicted_clv": pc, "grade": grade, "grading_version": 2}


def _done(home=27, away=20, vegas_home_prob=0.60, spread_line=3.0):
    return pd.DataFrame([{"game_id": "2026_05_SF_SEA", "completed": True, "home_score": home, "away_score": away,
                          "home_team": "SEA", "away_team": "SF", "gameday": "2026-10-11",
                          "vegas_home_prob": vegas_home_prob, "spread_line": spread_line}])


# ------------------------------------------------------------------------------------------------ rules
def test_rules_frozen_with_basis_and_validation():
    for market, n in (("ml", 100), ("spread", 60), ("totals", 100)):
        r = GA.load_rules(market)
        assert r["version"] == 1 and r["frozen_on"] == "2026-10-02" and r["track"] == f"aplus_{market}"
        assert r["ledger"] == f"paper_bets_aplus_{market}.json" and r["sizing"] == {"units": 1.0}
        v = r["validation"]
        assert v["min_bets"] == n and v["avg_clv_positive_with_p_below"] == 0.05 and v["realized_roi_above"] == 0.0
        assert "output/research/" in r["note"] and "2026" in r["note"] and "clean test" in r["note"]
        assert "hourly" in r["qualify"]["lock"] and "FIRST snapshot" in r["qualify"]["lock"]
        assert r["qualify"]["grade"] == "A+" and r["qualify"]["max_hours_before_kickoff"] == 168


# ------------------------------------------------------------------------------------------------ moneyline
def test_ml_bets_first_aplus_snapshot_and_locks(tmp_path):
    g = _game(grade_v2=_ml_grade("A", -115))
    pred = {"upcoming": [g]}
    res = GA.process("ml", pred, NO_GAMES, tmp_path, now=NOW)
    assert res["new"] == [] and g["aplus_ml"]["verdict"] == "pass" and g["aplus_ml"]["reasons"] == ["grade A (needs A+)"]
    g["grade_v2"] = _ml_grade("A+", -120)                                         # next snapshot: A+
    res = GA.process("ml", pred, NO_GAMES, tmp_path, now=NOW + timedelta(hours=1))
    (b,) = res["new"]
    assert (b["team"], b["opponent"], b["price"], b["book"], b["units"]) == ("SEA", "SF", -120, "FanDuel", 1.0)
    assert b["grade_v2"] == "A+" and b["predicted_clv"] == 0.031 and b["track"] == "aplus_ml" and b["rules_version"] == 1
    g["grade_v2"] = _ml_grade("A+", -105, pc=0.05)                                 # better price later: locked
    res = GA.process("ml", pred, NO_GAMES, tmp_path, now=NOW + timedelta(hours=2))
    assert res["new"] == [] and len(res["open"]) == 1 and res["open"][0]["price"] == -120
    v = g["aplus_ml"]
    assert v["verdict"] == "bet" and v["price"] == -120 and v["placed_at"].startswith("2026-10-09T15:10")
    assert len(json.loads((tmp_path / "paper_bets_aplus_ml.json").read_text())) == 1


def test_ml_window_and_ungraded(tmp_path):
    far = _game(game_id="far", kickoff_utc=(NOW + timedelta(days=8)).isoformat(), grade_v2=_ml_grade())
    none = _game(game_id="none", grade_v2=None)
    res = GA.process("ml", {"upcoming": [far, none]}, NO_GAMES, tmp_path, now=NOW)
    assert res["new"] == []
    assert any("7 days" in x for x in far["aplus_ml"]["reasons"]) and none["aplus_ml"] is None


def test_ml_grading_clv_vs_closing_no_vig_and_validation(tmp_path):
    g = _game(grade_v2=_ml_grade("A+", 110))
    GA.process("ml", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    res = GA.process("ml", {"upcoming": []}, _done(vegas_home_prob=0.5), tmp_path, now=NOW)
    (b,) = res["recent_graded"]
    assert b["result"] == "win" and b["profit_units"] == 1.1 and b["clv"] == pytest.approx(2.1 * 0.5 - 1)
    assert res["record"]["graded"] == 1 and res["mode"] == "shadow" and not res["record"]["checks"]["enough_bets"]
    assert res["by_grade_v2"][0]["grade"] == "A+"


# ------------------------------------------------------------------------------------------------ spread
def _spread_lo():
    rows = [("DraftKings", "draftkings", -3.0, -110, -110), ("FanDuel", "fanduel", -3.0, -105, -115),
            ("BetMGM", "betmgm", -3.0, 130, -150)]                            # BetMGM fails the price filter
    return {"consensus_home_margin": 3.0,
            "spreads_by_book": [{"book": t, "home_point": p, "home_price": h, "away_point": -p, "away_price": a}
                                for t, _, p, h, a in rows],
            "by_book": {k: {"ml": None, "sp": [p, h, a]} for _, k, p, h, a in rows}}


def _first_seen(history: Path, margin=2.0):
    history.mkdir(exist_ok=True)
    (history / "predictions_2026-10-06T1410.json").write_text(json.dumps(
        {"upcoming": [{"game_id": "2026_05_SF_SEA", "context": {"live_odds": {"consensus_home_margin": margin}}}]}))


def test_spread_research_filter_and_v1_aplus_without_eligibility(tmp_path):
    from nflpred import spread_bets as SB
    r = GA.load_rules("spread")
    lo = _spread_lo()
    assert [b["book"] for b in GA.research_spread_rows(lo, r)] == ["DraftKings", "FanDuel"]
    # production view (unfiltered) picks the out-of-range +130; the A+ candidate is the best in-range offer
    g = _game(context={"live_odds": lo})
    SB.evaluate(g, SB.load_rules(), 2.0)
    assert g["spread"]["best"]["price"] == 130
    o, reasons = GA.spread_offer(g, r, SB.load_rules(), 2.0)
    assert (o["book"], o["point"], o["price"]) == ("FanDuel", -3.0, -105)
    assert o["grade"] == "A+" and o["score"] >= 4 and reasons == []
    assert g["spread"]["best"]["price"] == 130                                   # game's own view untouched
    # no injury report / unconfirmed QB do not matter (research applied v1 at any snapshot)
    _first_seen(tmp_path)
    g = _game(context={"live_odds": lo}, injury_report=False, qb_status={"home": {"play_prob": 0.5}})
    res = GA.process("spread", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    (b,) = res["new"]
    assert (b["team"], b["point"], b["price"], b["book"], b["grade"], b["units"]) == ("SEA", -3.0, -105, "FanDuel", "A+", 1.0)
    assert g["aplus_spread"]["verdict"] == "bet"
    # line moves against us later: grade drops, but the bet stays locked and no second bet is made
    _first_seen(tmp_path, 2.0)
    g2 = _game(context={"live_odds": dict(lo, consensus_home_margin=1.5)})
    res = GA.process("spread", {"upcoming": [g2]}, NO_GAMES, tmp_path, now=NOW + timedelta(hours=1))
    assert res["new"] == [] and len(res["open"]) == 1
    assert g2["aplus_spread"]["verdict"] == "bet" and g2["aplus_spread"]["current_grade"] != "A+"


def test_spread_not_aplus_is_a_pass(tmp_path):
    _first_seen(tmp_path, 3.0)                                                   # no line move: score 3 = A
    g = _game(context={"live_odds": _spread_lo()})
    res = GA.process("spread", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    assert res["new"] == [] and g["aplus_spread"]["grade"] == "A" and g["aplus_spread"]["verdict"] == "pass"


def _odds_snapshot(history: Path, ts: str, spread=None, total=None):
    bks = []
    if spread:
        bks.append({"key": "lowvig", "title": "LowVig", "markets": [{"key": "spreads", "outcomes": [
            {"name": "Seattle Seahawks", "point": spread[0], "price": spread[1]},
            {"name": "San Francisco 49ers", "point": -spread[0], "price": spread[2]}]}]})
    if total:
        bks.append({"key": "draftkings", "title": "DraftKings", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "point": total[0], "price": total[1]}, {"name": "Under", "point": total[0], "price": total[2]}]}]})
    ev = {"id": "e1", "home_team": "Seattle Seahawks", "away_team": "San Francisco 49ers",
          "commence_time": "2026-10-11T20:25:00Z", "bookmakers": bks}
    (history / f"odds_{ts}.json").write_text(json.dumps([ev]))


def test_spread_grading_uses_closing_prices(tmp_path):
    _first_seen(tmp_path)
    g = _game(context={"live_odds": _spread_lo()})
    GA.process("spread", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    _odds_snapshot(tmp_path, "2026-10-11T1900", spread=(-4.0, -110, -110))       # closed at SEA -4
    res = GA.process("spread", {"upcoming": []}, _done(home=24, away=20), tmp_path, now=NOW)
    (b,) = res["recent_graded"]
    assert b["result"] == "win" and b["profit_units"] == pytest.approx(100 / 105, abs=1e-3)
    assert b["clv"] > 0 and "own closing snapshot" in b["clv_source"]           # -3 at -105 beat a -4 close
    assert res["by_grade"][0]["grade"] == "A+"


# ------------------------------------------------------------------------------------------------ totals
def _tg(side="under", grade="A+", point=45.5, price=-105):
    return {"side": side, "point": point, "price": price, "book": "DraftKings", "book_key": "draftkings",
            "predicted_clv": 0.004, "grade": grade, "grading_version": 2, "consensus_total": 46.0, "sharp_total": 45.8}


def test_totals_bets_aplus_over_and_grades_side_aware(tmp_path):
    g = _game(totals_grade=_tg("over", point=44.5))
    res = GA.process("totals", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    (b,) = res["new"]
    assert (b["side"], b["team"], b["point"], b["price"], b["totals_grade"]) == ("over", "SF@SEA OVER", 44.5, -105, "A+")
    assert g["aplus_totals"]["team"] == "Over" and g["aplus_totals"]["ledger_team"] == "SF@SEA OVER"
    g["totals_grade"] = _tg("under")                                             # later A+ under: same game, locked
    assert GA.process("totals", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)["new"] == []
    _odds_snapshot(tmp_path, "2026-10-11T1900", total=(46.5, -110, -110))       # close moved up: over beat it
    res = GA.process("totals", {"upcoming": []}, _done(home=27, away=20), tmp_path, now=NOW)
    (gb,) = res["recent_graded"]
    assert gb["final"] == "47 pts"
    assert gb["result"] == "win" and gb["profit_units"] == pytest.approx(100 / 105, abs=1e-3)   # 47 > 44.5
    assert gb["clv"] > 0 and gb["closing_total"] > 44.5


def test_totals_under_grading_unchanged():
    from nflpred import totals as T
    dist = T.default_dist()
    bet = {"game_id": "2026_05_SF_SEA", "point": 45.5, "price": -110, "units": 1.0, "status": "open"}  # legacy: no side
    closes = {("SEA", "SF"): [{"commence": "2026-10-11T20:25:00Z", "mu": 44.0, "mu_sharp": None, "ts": "x"}]}
    out = T.grade(bet, _done(home=24, away=20), dist, closes)
    assert out["result"] == "win" and out["clv"] == round(dist.ev(44.0, 45.5, -110, "under"), 4)
    out = T.grade(dict(bet, side="over"), _done(home=24, away=20), dist, closes)
    assert out["result"] == "loss" and out["clv"] == round(dist.ev(44.0, 45.5, -110, "over"), 4)


def test_totals_not_graded_shows_reason(tmp_path):
    g = _game(totals_grade=None, totals_grade_check=["graded only within 7 days of kickoff"])
    GA.process("totals", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)
    assert g["aplus_totals"] is None
    g = _game(totals_grade=_tg(grade="B"))
    assert GA.process("totals", {"upcoming": [g]}, NO_GAMES, tmp_path, now=NOW)["new"] == []
    assert g["aplus_totals"]["reasons"] == ["grade B (needs A+)"]


# ------------------------------------------------------------------------------------------------ validation / alerts
def test_validation_goes_live_only_with_enough_significant_clv(tmp_path):
    from nflpred import bets as ML
    r = GA.load_rules("ml")
    mk = lambda i, clv, pr: {"game_id": str(i), "status": "graded", "rules_version": 1, "units": 1.0,  # noqa: E731
                             "result": "win" if pr > 0 else "loss", "profit_units": pr, "clv": clv}
    led = [mk(i, 0.02 + 0.01 * (i % 3 - 1), 1.1 if i % 2 else -1.0) for i in range(99)]
    assert not ML.record(led, r)["passed"]                                       # 99 < 100 bets
    led.append(mk(99, 0.02, 1.1))
    rec = ML.record(led, r)
    assert rec["checks"]["enough_bets"] and rec["checks"]["clv_positive_and_significant"] and rec["passed"]


def test_alert_lines_only_for_live_tracks():
    from nflpred import pipeline
    b = {"team": "SEA", "opponent": "SF", "price": -120, "book": "FanDuel", "gameday": "2026-10-11",
         "predicted_clv": 0.03, "units": 1.0, "point": -3.0}
    res = {"aplus_ml_bets": {"mode": "shadow", "new": [b]}, "aplus_spread_bets": {"mode": "live", "new": [b]}}
    lines = pipeline.alert_lines(res)
    assert len(lines) == 1 and "spread grade A+" in lines[0] and "SEA -3" in lines[0]
    res["aplus_ml_bets"]["mode"] = "live"
    assert any("moneyline grade A+" in x for x in pipeline.alert_lines(res))


def test_pipeline_helper_never_raises_and_keeps_graded_list(tmp_path, monkeypatch):
    from nflpred import pipeline
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    (tmp_path / "history").mkdir()
    for f in GA.RULE_FILES.values():
        (tmp_path / f).write_text((ROOT / f).read_text())
    monkeypatch.setattr(GA, "ROOT", tmp_path)
    pred = {"upcoming": [_game(grade_v2=_ml_grade())], "aplus_ml_bets": {"recent_graded": [{"id": "old"}]}}
    pipeline._aplus_tracks(pred, NO_GAMES, NOW)
    assert pred["aplus_ml_bets"]["new"] and pred["aplus_ml_bets"]["recent_graded"] == [{"id": "old"}]
    assert "error" not in pred["aplus_spread_bets"] and "error" not in pred["aplus_totals_bets"]
    (tmp_path / GA.RULE_FILES["totals"]).unlink()
    pipeline._aplus_tracks(pred, NO_GAMES, NOW)
    assert "error" in pred["aplus_totals_bets"]


def test_dashboard_lists_aplus_tracks(tmp_path):
    import importlib.util
    import re
    spec = importlib.util.spec_from_file_location("build_dashboard", ROOT / "scripts" / "build_dashboard.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    lim = bd.load_rule_limits()
    assert lim["aplus_sp"]["min_odds"] == -145 and lim["aplus_sp"]["max_odds"] == 125
    tpl = (ROOT / "scripts" / "dashboard_template.html").read_text()
    for s in ("aplus_ml_bets", "aplus_spread_bets", "aplus_totals_bets", "A+ moneyline", "A+ spread", "A+ totals",
              "+2.3% CLV · 163 bets 2023–25", "+2.2% CLV · 49 bets 2023–25 · borderline", "+1.5% CLV · 243 bets 2023–25"):
        assert s in tpl
    # tracks are listed in a dynamic ranking (live score, then research evidence), not a fixed order
    for k in ("aplus_ml", "aplus_sp", "aplus_tot"):
        assert re.search(rf"\n  {k}:{{name:", tpl) and re.search(rf"\n  {k}:{{es:[0-9.]+, label:", tpl)
    assert "Object.keys(TRACKS)" in tpl and "liveScore" in tpl
