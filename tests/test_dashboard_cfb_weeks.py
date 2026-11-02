import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_dashboard as B  # noqa: E402

U = dt.timezone.utc
GAMES = [
    {"id": "1", "t": dt.datetime(2026, 11, 7, 21, tzinfo=U), "home": "Mississippi State", "away": "Vanderbilt", "week": 11, "post": False},
    {"id": "2", "t": dt.datetime(2026, 11, 7, 20, tzinfo=U), "home": "Miami (OH)", "away": "Ohio", "week": 11, "post": False},
    {"id": "3", "t": dt.datetime(2026, 11, 7, 19, tzinfo=U), "home": "Miami", "away": "Ohio State", "week": 11, "post": False},
    {"id": "4", "t": dt.datetime(2026, 10, 31, 19, tzinfo=U), "home": "Hawai'i", "away": "Nevada", "week": 10, "post": False},
    {"id": "5", "t": dt.datetime(2026, 12, 30, 19, tzinfo=U), "home": "Utah", "away": "Iowa", "week": 1, "post": True},
]


def test_college_bets_get_their_week():
    d = {"cfb_shop_bets": {"open": [
        {"home": "Mississippi State Bulldogs", "away": "Vanderbilt Commodores", "kickoff_utc": "2026-11-07T21:15:00+00:00"},
        {"home": "Miami (OH) RedHawks", "away": "Ohio Bobcats", "kickoff_utc": "2026-11-07T20:00:00+00:00"},
        {"home": "Hawaii Rainbow Warriors", "away": "Nevada Wolf Pack", "kickoff_utc": "2026-10-31T19:00:00+00:00"},   # spelled differently
        {"home": "Utah Utes", "away": "Iowa Hawkeyes", "kickoff_utc": "2026-12-30T19:00:00+00:00"}]},
         "cfb_body_clock_bets": {"recent_graded": [{"game_id": "3", "home": "x", "away": "y"}]}}
    B.add_cfb_weeks(d, GAMES)
    assert [b["week"] for b in d["cfb_shop_bets"]["open"]] == [11, 11, 10, "Bowl"]
    assert d["cfb_body_clock_bets"]["recent_graded"][0]["week"] == 11
