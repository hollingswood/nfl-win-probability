"""betslip.py: the alert/dashboard link opens the exact bet on the book's slip, never a different number."""
from nflpred import betslip as BS

EV = [{"id": "e1", "home_team": "San Francisco 49ers", "away_team": "Washington Commanders", "bookmakers": [
    {"key": "fanduel", "title": "FanDuel", "link": "https://sportsbook.fanduel.com/x/e1", "sid": "111", "markets": [
        {"key": "h2h", "link": None, "outcomes": [
            {"name": "Washington Commanders", "price": 340, "link": "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=2", "sid": "2"},
            {"name": "San Francisco 49ers", "price": -420, "link": "https://sportsbook.fanduel.com/addToBetslip?marketId=1&selectionId=3", "sid": "3"}]},
        {"key": "spreads", "link": None, "outcomes": [
            {"name": "Washington Commanders", "price": -110, "point": 8.5, "link": "https://sportsbook.fanduel.com/addToBetslip?marketId=4&selectionId=5"}]}]},
    {"key": "draftkings", "title": "DraftKings", "link": "https://sportsbook.draftkings.com/event/999", "sid": "999", "markets": [
        {"key": "totals", "outcomes": [{"name": "Under", "price": 100, "point": 50.5, "sid": "u505"},
                                       {"name": "Over", "price": -120, "point": 50.5, "sid": "o505"}]}]},
    {"key": "espnbet", "title": "theScore Bet", "link": "https://example.thescore/game", "markets": [
        {"key": "h2h", "outcomes": [{"name": "Washington Commanders", "price": 330}]}]}]}]


def test_moneyline_goes_on_the_slip():
    s = BS.link_for({"track": "aplus_ml", "side": "away", "team": "WAS", "book": "FanDuel", "price": 350}, EV, "e1")
    assert s["kind"] == "slip" and "selectionId=2" in s["url"] and BS.label(s) == "Bet at FanDuel"


def test_draftkings_slip_built_from_ids_and_totals_side():
    s = BS.link_for({"track": "aplus_totals", "side": "under", "point": 50.5, "book": "DraftKings", "book_key": "draftkings"}, EV, "e1")
    assert s["kind"] == "slip" and s["url"] == "https://sportsbook.draftkings.com/event/999?outcomes=u505" and BS.label(s) == "Bet at DK"


def test_moved_number_never_puts_a_different_line_on_the_slip():
    s = BS.link_for({"market": "spread", "track": "spread", "side": "away", "team": "WAS", "point": 7.5, "book": "FanDuel"}, EV, "e1")
    assert s["kind"] == "game" and s["url"].endswith("/e1") and "+8.5" in s["note"] and BS.label(s) == "Open FanDuel"


def test_thescore_has_no_slip_link_so_it_opens_the_game():
    s = BS.link_for({"track": "cfb_aplus", "market": "ml", "side": "away", "team": "Washington Commanders",
                     "book": "theScore Bet", "book_key": "espnbet"}, EV, "e1")
    assert s["kind"] == "game" and s["short"] == "theScore"


def test_unknown_event_or_book_gives_no_link():
    assert BS.link_for({"side": "away", "book": "FanDuel"}, EV, "nope") is None
    assert BS.link_for({"side": "away", "book": "Kalshi"}, EV, "e1") is None


def test_alert_uses_the_slip_as_main_button(monkeypatch):
    from nflpred import notify as N
    sent = []
    monkeypatch.setattr(N, "send", lambda title, msg, **k: sent.append((title, msg, k)) or True)
    monkeypatch.setattr(N.betslip, "for_bet", lambda b, pred, college=None: BS.link_for(b, EV, "e1"))
    monkeypatch.setattr(N, "account_names", lambda: None)
    b = {"id": "2026_06_WAS_SF:aplus_ml:away", "track": "aplus_ml", "game_id": "2026_06_WAS_SF", "side": "away", "team": "WAS",
         "opponent": "SF", "price": 350, "book": "FanDuel", "units": 1.0}
    N.new_bets({"upcoming": [], "aplus_ml_bets": {"new": [b]}}, set())
    title, msg, k = sent[-1]
    assert k["click"].startswith("https://sportsbook.fanduel.com/addToBetslip") and k["actions"][0]["label"] == "Bet at FanDuel"
    assert [a["label"] for a in k["actions"]] == ["Bet at FanDuel", "I placed it", "Dashboard"] and "enter your stake" in msg
