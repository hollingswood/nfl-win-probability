from nflpred import placed as P


def _rec(market, side, point, stake=54.0, cents=54):
    return {"bet_id": None, "pick": "x", "book": "Robinhood", "odds": f"{cents}c", "decimal": round(100 / cents, 4),
            "stake": stake, "game": {"sport": "cfb", "game_id": "1", "away": "Texas", "home": "Missouri",
                                     "market": market, "side": side, "point": point}}


def test_manual_bets_graded_from_final(monkeypatch):
    monkeypatch.setitem(P._FINALS, "cfb", {"1": (24.0, 21.0)})          # Texas 24 @ Missouri 21
    r = P.score(_rec("spread", "away", -2.5), {})
    assert r["status"] == "graded" and r["result"] == "win" and r["profit"] == 46.0 and r["final"] == "24-21"
    assert P.score(_rec("spread", "away", -3.5), {})["result"] == "loss"
    assert P.score(_rec("spread", "home", 3.0), {})["result"] == "push"
    assert P.score(_rec("ml", "away", None), {})["result"] == "win"
    assert P.score(_rec("total", "under", 45.5), {})["result"] == "win"     # 45 points
    assert P.sport_of(_rec("ml", "away", None), None) == "cfb"


def test_manual_bet_open_until_final(monkeypatch):
    monkeypatch.setitem(P._FINALS, "cfb", {})
    r = P.score(_rec("ml", "home", None), {})
    assert r["status"] == "open" and r["matchup"] == "Texas @ Missouri"
