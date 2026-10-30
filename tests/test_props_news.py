"""Props + news track: triggers only on strict, fresh 'regular is out' news; bets the next men up's overs; grades CLV + line move."""
import json
from datetime import datetime, timezone

import pandas as pd

from nflpred import props_news as PN

NOW = datetime(2026, 10, 29, 18, 0, tzinfo=timezone.utc)
GAME = {"game_id": "2026_09_KC_DEN", "season": 2026, "week": 9, "gameday": "2026-11-01", "kickoff_utc": "2026-11-01T21:05:00+00:00",
        "home_team": "DEN", "away_team": "KC", "home_prev_end": "2026-10-27T03:00+00:00", "away_prev_end": "2026-10-26T23:00+00:00",
        "context": {"live_odds": {"event_id": "ev1"}}}
SNAPS = pd.DataFrame([{"team": "DEN", "game_id": f"2026_0{w}_X", "week": w, "player": p, "position": pos, "offense_pct": v}
                      for w in (6, 7, 8) for p, pos, v in (("Courtland Sutton", "WR", 0.9), ("Troy Franklin", "WR", 0.7),
                                                             ("Marvin Mims", "WR", 0.4), ("Devaughn Vele", "WR", 0.1),
                                                             ("Evan Engram", "TE", 0.6))])


def sig(**k):
    base = {"player": "Courtland Sutton", "team": "DEN", "position": "WR", "signal": "out", "certainty": 1.0,
            "game_week_relevant": True, "verify": {"roster": "match"}, "seen_at": "2026-10-29T16:00+00:00",
            "published": "2026-10-29T15:50+00:00", "link": "L"}
    return {**base, **k}


def test_trigger_rules():
    r = PN.load_rules()
    sh = lambda team: PN.snap_shares(SNAPS, team)
    t = PN.triggers([sig()], GAME, sh, r, NOW, set())
    assert len(t) == 1 and [b["player"] for b in t[0]["beneficiaries"]] == ["Troy Franklin", "Marvin Mims"]
    assert PN.triggers([sig(certainty=0.7)], GAME, sh, r, NOW, set()) == []                       # not strict
    assert PN.triggers([sig(player="Marvin Mims")], GAME, sh, r, NOW, set()) == []                # not a regular (40%)
    assert PN.triggers([sig(seen_at="2026-10-29T08:00+00:00")], GAME, sh, r, NOW, set()) == []   # older than 6 h
    assert PN.triggers([sig(published="2026-10-26T12:00+00:00")], GAME, sh, r, NOW, set()) == []  # before his last game
    assert PN.triggers([sig(verify={"roster": "not_found"})], GAME, sh, r, NOW, set()) == []
    done = set(); PN.triggers([sig()], GAME, sh, r, NOW, done)
    assert PN.triggers([sig()], GAME, sh, r, NOW, done) == []                                       # once per player-game
    # a beneficiary also reported out is skipped
    t2 = PN.triggers([sig(), sig(player="Troy Franklin", certainty=0.5)], GAME, sh, r, NOW, set())
    assert [b["player"] for b in t2[0]["beneficiaries"]] == ["Marvin Mims", "Devaughn Vele"][:1] or \
        [b["player"] for b in t2[0]["beneficiaries"]][0] == "Marvin Mims"


def resp(pt=3.5, over=-110, under=-110, extra=()):
    bms = []
    for bk in ("draftkings", "fanduel", "betmgm", "williamhill_us"):
        outs = []
        for name in ("Troy Franklin", "Marvin Mims"):
            outs += [{"name": "Over", "description": name, "point": pt, "price": over},
                     {"name": "Under", "description": name, "point": pt, "price": under}]
        bms.append({"key": bk, "title": bk, "markets": [{"key": "player_receptions", "outcomes": outs},
                                                        {"key": "player_reception_yds", "outcomes": [
                                                            {"name": "Over", "description": "Troy Franklin", "point": 40.5, "price": -115},
                                                            {"name": "Under", "description": "Troy Franklin", "point": 40.5, "price": -105}]}]})
    return {"id": "ev1", "bookmakers": bms}


def test_bets_and_grading(tmp_path):
    r = PN.load_rules()
    tr = PN.triggers([sig()], GAME, lambda t: PN.snap_shares(SNAPS, t), r, NOW, set())[0]
    bets = PN.bets_for(tr, GAME, resp(), r, {"draftkings", "fanduel"}, NOW)
    assert {(b["player"], b["market"]) for b in bets} == {("Troy Franklin", "player_receptions"), ("Troy Franklin", "player_reception_yds"),
                                                         ("Marvin Mims", "player_receptions")}
    assert all(b["side"] == "over" and b["book_key"] in ("draftkings", "fanduel") for b in bets)
    b = next(x for x in bets if x["market"] == "player_receptions" and x["player"] == "Troy Franklin")
    close = [{"event_id": "ev1", "fetched_at": "2026-11-01T20:00+00:00", "response": resp(pt=3.5, over=-140, under=110)}]
    games = pd.DataFrame([{"game_id": GAME["game_id"], "completed": True}])
    stats = pd.DataFrame([{"game_id": GAME["game_id"], "player_display_name": "Troy Franklin", "player_name": "T.Franklin",
                           "receptions": 5, "receiving_yards": 61}])
    g = PN.grade(b, games, close, lambda s: (stats, pd.DataFrame()), NOW)
    assert g["result"] == "win" and g["clv"] > 0 and g["points_moved"] == 0.0
