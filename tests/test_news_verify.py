"""AI news verification: roster check (roster.py), outcome audit and the pre-registered promotion rule
(news_audit.py, news_rules.json), and the dashboard's use of the checked team."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from nflpred import news_audit as NA
from nflpred import roster as R

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def _roster(source="sleeper"):
    rows = [("Jalon Daniels", "Jalon", "Daniels", "TB", "QB", "TBQB2", 2),
            ("Baker Mayfield", "Baker", "Mayfield", "TB", "QB", "TBQB1", 1),
            ("Jayden Daniels", "Jayden", "Daniels", "WAS", "QB", "WASQB1", 1),
            ("Josh Allen", "Josh", "Allen", "BUF", "QB", "BUFQB1", 1),
            ("Josh Allen", "Josh", "Allen", "JAX", "LB", "JAXLB1", None),
            ("Michael Penix Jr.", "Michael", "Penix", "ATL", "QB", "ATLQB1", 1),
            ("J.J. McCarthy", "J.J.", "McCarthy", "NYG", "QB", "NYGQB2", 2),
            ("Amon-Ra St. Brown", "Amon-Ra", "St. Brown", "DET", "WR", "DETWR1", 1),
            ("Mike Evans", "Mike", "Evans", "TB", "WR", "TBWR1", 1),
            ("Bucky Irving", "Bucky", "Irving", "TB", "RB", "TBRB1", 1)]
    df = pd.DataFrame(rows, columns=["full_name", "first_name", "last_name", "team", "position", "gsis_id", "depth_order"])
    return R.build(df, source=source, as_of=datetime(2026, 10, 1, 14, tzinfo=UTC))


def _sig(player, team, **kw):
    s = {"seen_at": "2026-10-01T18:23+00:00", "player": player, "team": team, "position": None, "signal": "out",
         "is_starting_qb_news": False, "game_week_relevant": True, "certainty": 0.9}
    s.update(kw)
    return s


# ------------------------------------------------------------------------------------------- roster check
@pytest.mark.parametrize("name,team", [
    ("Michael Penix Jr.", "ATL"), ("Michael Penix", "ATL"), ("Mike Penix", "ATL"),    # suffix, nickname
    ("J.J. McCarthy", "NYG"), ("J. J. McCarthy", "NYG"), ("JJ McCarthy", "NYG"),     # initials punctuation
    ("Amon-Ra St. Brown", "DET"), ("Amon Ra St Brown", "DET"),                        # hyphen, period
    ("Jalon Daniels", "TB"), ("Daniels", "TB")])                                      # last name only + team
def test_roster_name_variants_match(name, team):
    v, tv = R.verify(_sig(name, team, position="QB" if team in ("ATL", "NYG", "TB") else None), _roster())
    assert v["roster"] == "match" and tv == team


def test_roster_team_mismatch_corrects_only_from_sleeper():
    s = _sig("Jalon Daniels", "KC", position="QB", signal="will_start", is_starting_qb_news=True)
    v, tv = R.verify(s, _roster())
    assert v["roster"] == "team_mismatch" and v["roster_team"] == "TB" and v["gsis_id"] == "TBQB2" and tv == "TB"
    v2, tv2 = R.verify(s, _roster("nflverse"))      # fallback roster can lag trades: flag, don't move
    assert v2["roster"] == "team_mismatch" and tv2 is None


def test_roster_not_found_and_ambiguous():
    assert R.verify(_sig("Jack Keenum", "CHI", position="QB"), _roster())[0]["roster"] == "not_found"
    v, tv = R.verify(_sig("Josh Allen", "NYJ"), _roster())          # two Josh Allens, neither on NYJ
    assert v["roster"] == "ambiguous" and set(v["roster_teams"]) == {"BUF", "JAX"} and tv is None
    assert R.verify(_sig("Josh Allen", "JAX"), _roster())[0]["gsis_id"] == "JAXLB1"   # claimed team disambiguates
    v, tv = R.verify(_sig("Josh Allen", "NYJ", position="QB"), _roster())             # position disambiguates
    assert v["roster"] == "team_mismatch" and tv == "BUF"
    assert R.verify(_sig("Jalon Daniels", "TB"), {"source": "sleeper", "players": {}})[0]["roster"] == "not_found"


def test_roster_save_load_and_backfill(tmp_path):
    R.save(_roster("nflverse"), tmp_path)
    assert R.load(tmp_path)["source"] == "nflverse"
    log = tmp_path / "news_llm.jsonl"
    rows = [_sig("Jalon Daniels", "KC", position="QB"), _sig("Mike Evans", "TB")]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    c = R.backfill(tmp_path, R.load(tmp_path))
    assert c["checked"] == 2 and c["team_mismatch"] == 1 and c["match"] == 1
    lines = log.read_text().splitlines()
    assert lines[-1] == "not json"                                  # bad lines kept as-is
    first = json.loads(lines[0])
    assert first["team"] == "KC" and first["player"] == "Jalon Daniels"   # AI fields untouched
    assert first["verify"]["roster_source"] == "nflverse" and first["team_verified"] is None
    assert R.backfill(tmp_path, R.load(tmp_path))["checked"] == 0   # nothing left to check
    c = R.backfill(tmp_path, _roster("sleeper"))                    # fresh roster re-checks fallback rows
    assert c["checked"] == 2
    assert json.loads(log.read_text().splitlines()[0])["team_verified"] == "TB"


def test_news_llm_scan_attaches_roster_check(tmp_path):
    from nflpred import news_llm as NL
    items = [{"title": "Jalon Daniels to start", "link": "u1", "published": None, "summary": "first start"}]

    def fake_llm(batch, key):
        return [{"item": 0, "player": "Jalon Daniels", "team": "KC", "position": "QB", "signal": "will_start",
                 "is_starting_qb_news": True, "game_week_relevant": True, "certainty": 0.85, "quote": "start"}]
    r = NL.scan(tmp_path, "k", datetime(2026, 10, 1, 18, tzinfo=UTC), fetcher=lambda u: items, llm=fake_llm,
                roster=_roster())
    assert r["roster_check"]["team_mismatch"] == 1
    row = json.loads((tmp_path / "news_llm.jsonl").read_text().splitlines()[0])
    assert row["team"] == "KC" and row["team_verified"] == "TB" and row["verify"]["roster"] == "team_mismatch"


def test_from_nflverse_fallback():
    p = pd.DataFrame({"display_name": ["Jalon Daniels", "Old Guy", "Cut Guy"], "first_name": ["Jalon", "Old", "Cut"],
                      "football_name": [None, "Old", "Cut"], "last_name": ["Daniels", "Guy", "Guy"],
                      "latest_team": ["TB", "LAR", "NYJ"], "position": ["QB", "QB", "QB"],
                      "gsis_id": ["a", "b", "c"], "status": ["ACT", "ACT", "CUT"], "last_season": [2026, 2010, 2026]})
    r = R.from_nflverse(p, 2026)
    assert r["source"] == "nflverse" and set(r["players"]) == {"jalon daniels"}


# ------------------------------------------------------------------------------------------- outcome audit
def _audit_fixture():
    games = pd.DataFrame([
        {"game_id": "2026_04_GB_TB", "season": 2026, "week": 4, "gameday": "2026-10-04", "gametime": "13:00",
         "home_team": "TB", "away_team": "GB", "home_score": 20, "away_score": 24,
         "home_qb_id": "TBQB2", "away_qb_id": "GBQB1", "home_qb_name": "Jalon Daniels", "away_qb_name": "Jordan Love"},
        {"game_id": "2026_04_IND_WAS", "season": 2026, "week": 4, "gameday": "2026-10-04", "gametime": "09:30",
         "home_team": "WAS", "away_team": "IND", "home_score": 17, "away_score": 21,
         "home_qb_id": "WASQB1", "away_qb_id": "INDQB1", "home_qb_name": "Jayden Daniels", "away_qb_name": "Daniel Jones"},
        {"game_id": "2026_04_ATL_NO", "season": 2026, "week": 4, "gameday": "2026-10-05", "gametime": "20:15",
         "home_team": "NO", "away_team": "ATL", "home_score": None, "away_score": None,
         "home_qb_id": None, "away_qb_id": None, "home_qb_name": None, "away_qb_name": None},
    ])
    snaps = pd.DataFrame([
        {"game_id": "2026_04_GB_TB", "team": "TB", "player": "Jalon Daniels", "pfr_player_id": "DaniJa00",
         "offense_snaps": 60, "defense_snaps": 0, "st_snaps": 0},
        {"game_id": "2026_04_GB_TB", "team": "TB", "player": "Mike Evans", "pfr_player_id": "EvanMi00",
         "offense_snaps": 50, "defense_snaps": 0, "st_snaps": 0},
    ])
    stats = pd.DataFrame(columns=["game_id", "player_id", "player_display_name", "player_name"])
    injuries = pd.DataFrame([
        {"season": 2026, "week": 4, "team": "TB", "gsis_id": "TBWR1", "full_name": "Mike Evans", "report_status": "Out"},
        {"season": 2026, "week": 4, "team": "TB", "gsis_id": "TBQB1", "full_name": "Baker Mayfield", "report_status": "Out"},
    ])
    players = pd.DataFrame({"gsis_id": ["TBQB2", "TBWR1"], "pfr_id": ["DaniJa00", "EvanMi00"],
                            "display_name": ["Jalon Daniels", "Mike Evans"]})
    t0 = datetime(2026, 10, 1, 18, 23, tzinfo=UTC)
    spreads = {("TB", "GB", "2026-10-04"): [(datetime(2026, 10, 1, 12, 23, tzinfo=UTC), 3.0), (t0, 3.0),
                                           (datetime(2026, 10, 2, 0, 23, tzinfo=UTC), 1.5)]}
    ros = _roster()

    def chk(s):
        return R.attach(s, ros)
    sigs = [
        chk(_sig("Jalon Daniels", "KC", position="QB", signal="will_start", is_starting_qb_news=True, certainty=0.85)),
        chk(_sig("Jalon Daniels", "TB", position="QB", signal="will_start", is_starting_qb_news=True, certainty=1.0,
                 seen_at="2026-10-01T20:23+00:00")),                    # same unit, later: deduped
        chk(_sig("Baker Mayfield", "TB", position="QB", signal="out", is_starting_qb_news=True, certainty=1.0)),
        chk(_sig("Mike Evans", "TB", position="WR", signal="out")),     # played anyway: wrong
        chk(_sig("Bucky Irving", "TB", position="RB", signal="out")),   # no snaps: correct
        chk(_sig("Jayden Daniels", "WAS", position="QB", signal="expected_to_play", is_starting_qb_news=True,
                 certainty=0.65)),                                      # correct, but below rule certainty
        chk(_sig("Mike Evans", "TB", position="WR", signal="questionable")),   # ungradable
        chk(_sig("Jack Keenum", "CHI", position="QB", signal="will_start", is_starting_qb_news=True)),  # team unclear
        chk(_sig("Michael Penix Jr.", "ATL", position="QB", signal="will_start", is_starting_qb_news=True)),  # pending
        chk(_sig("Mike Evans", "TB", signal="out", game_week_relevant=False)),
    ]
    return sigs, games, snaps, stats, injuries, players, spreads


def test_audit_grades_signals_against_outcomes():
    sigs, games, snaps, stats, injuries, players, spreads = _audit_fixture()
    rules = NA.load_rules()
    a = NA.audit(sigs, games, snaps, stats, injuries, players, spreads, now=datetime(2026, 10, 7, tzinfo=UTC),
                 rules=rules)
    assert a["skipped"] == {"team_unclear": 1, "no_game": 0, "not_game_week": 1}
    assert a["by_signal"]["out"] == {"n": 3, "correct": 2, "pct": 66.7}
    assert a["by_signal"]["will_start"]["n"] == 1                      # duplicate Jalon signal counted once
    assert a["qb"] == {"n": 3, "correct": 3, "pct": 100.0, "pending": 1}
    assert a["overall"]["n"] == 5 and a["overall"]["correct"] == 4
    assert a["rule_population"]["n"] == 2 and a["rule_population"]["pending"] == 1
    rows = {(r["player"], r["signal"]): r for r in a["rows"]}
    assert rows[("Jalon Daniels", "will_start")]["team"] == "TB"       # game from team_verified, not the AI's KC
    assert rows[("Jalon Daniels", "will_start")]["truth"] == "started"
    assert rows[("Mike Evans", "out")]["truth"] == "played" and rows[("Mike Evans", "out")]["report_agrees"] is True
    assert rows[("Bucky Irving", "out")]["verdict"] == "correct"
    assert rows[("Michael Penix Jr.", "will_start")]["verdict"] == "pending"
    # final report proxy: Friday 2026-10-02 16:00 ET = 20:00 UTC; first seen Thu 18:23 UTC
    assert rows[("Baker Mayfield", "out")]["lead_vs_report_h"] == pytest.approx(25.6)
    assert rows[("Baker Mayfield", "out")]["line_move"] == {"after": -1.5, "before": 0.0}
    assert a["qb_line_move_6h"]["avg_after_negative_signals"] == -1.5
    assert a["promotion"]["eligible"] is False


def test_audit_participation_fallbacks():
    sigs, games, snaps, stats, injuries, players, spreads = _audit_fixture()
    s = sigs[4]  # Bucky Irving
    g = games.iloc[0].to_dict()
    assert NA.played(s, g, "TB", snaps.iloc[0:0], stats, {}) is None          # nothing published yet
    st = pd.DataFrame([{"game_id": "2026_04_GB_TB", "player_id": "TBRB1", "player_display_name": "Bucky Irving",
                        "player_name": "B.Irving"}])
    assert NA.played(s, g, "TB", snaps.iloc[0:0], st, {}) is True             # stat line = played
    st2 = st.assign(player_id="X", player_display_name="Someone Else", player_name="S.Else")
    assert NA.played(s, g, "TB", snaps.iloc[0:0], st2, {}) is None            # no stat line is not proof


def test_report_time_proxy():
    sun = NA.report_time({"gameday": "2026-10-04"})
    thu = NA.report_time({"gameday": "2026-10-08"})
    mon = NA.report_time({"gameday": "2026-10-05"})
    assert sun == datetime(2026, 10, 2, 20, tzinfo=UTC)                       # Fri 4pm EDT
    assert thu == datetime(2026, 10, 7, 20, tzinfo=UTC)                       # Wed for Thursday games
    assert mon == datetime(2026, 10, 3, 20, tzinfo=UTC)                       # Sat for Monday games


def test_spread_series_reads_snapshots(tmp_path):
    import gzip
    ev = {"home_team": "Tampa Bay Buccaneers", "away_team": "Green Bay Packers", "commence_time": "2026-10-04T17:00:00Z",
          "bookmakers": [{"key": k, "markets": [{"key": "spreads", "outcomes": [
              {"name": "Tampa Bay Buccaneers", "point": p, "price": -110},
              {"name": "Green Bay Packers", "point": -p, "price": -110}]}]} for k, p in (("a", -3.0), ("b", -2.5), ("c", -3.0))]}
    (tmp_path / "odds_2026-10-01T1823.json").write_text(json.dumps([ev]))
    with gzip.open(tmp_path / "odds_2026-10-02T0023.json.gz", "wt") as f:
        f.write(json.dumps([dict(ev, bookmakers=ev["bookmakers"][:1])]))
    s = NA.spread_series(tmp_path)[("TB", "GB", "2026-10-04")]
    assert s == [(datetime(2026, 10, 1, 18, 23, tzinfo=UTC), 3.0), (datetime(2026, 10, 2, 0, 23, tzinfo=UTC), 3.0)]


# ------------------------------------------------------------------------------------------- promotion rule
def test_promotion_rule_frozen():
    r = json.loads((ROOT / "news_rules.json").read_text())
    assert r["version"] == 1 and r["frozen_on"] == "2026-10-02"
    assert r["promotion"] == {**r["promotion"], "min_audited_qb_signals": 25, "min_precision": 0.95}
    assert r["population"]["min_certainty"] == 0.8 and r["population"]["roster"] == "match"
    assert r["population"]["game_week_relevant"] is True


@pytest.mark.parametrize("n,correct,eligible", [(0, 0, False), (24, 24, False), (25, 23, False), (25, 24, True),
                                                (40, 38, True), (40, 37, False)])
def test_promotion_check(n, correct, eligible):
    res = NA.promotion_check({"rule_population": {"n": n, "correct": correct}}, NA.load_rules())
    assert res["eligible"] is eligible and res["reasons"]
    assert res["rules_version"] == 1


def test_ai_news_not_wired_into_predictions():
    """Display only until the rule passes: QB availability and feature code never read AI signals."""
    for f in ("qb_availability.py", "features.py", "model.py"):
        src = (ROOT / "src" / "nflpred" / f).read_text()
        assert "news_llm" not in src and "news_audit" not in src and "news_rules" not in src


# ------------------------------------------------------------------------------------------- dashboard
def test_dashboard_uses_checked_team_and_shows_accuracy(tmp_path):
    import datetime as dt
    import importlib.util
    path = ROOT / "scripts" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("build_dashboard", path)
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    ros = _roster()
    sigs = [R.attach(_sig("Jalon Daniels", "KC", position="QB", signal="will_start", is_starting_qb_news=True), ros),
            R.attach(_sig("Mike Evans", "TB", position="WR", signal="questionable"), ros),
            R.attach(_sig("Jack Keenum", "GB", position="QB", signal="will_start", is_starting_qb_news=True), ros),
            _sig("Unchecked Guy", "GB", signal="out")]
    up = [{"game_id": "2026_04_GB_TB", "home_team": "TB", "away_team": "GB"},
          {"game_id": "2026_04_KC_LV", "home_team": "LV", "away_team": "KC"}]
    by = bd.news_by_game(sigs, up, now=dt.datetime(2026, 10, 2, tzinfo=dt.timezone.utc))
    assert "2026_04_KC_LV" not in by                                     # moved off the AI's (wrong) team
    got = {n["player"]: n for n in by["2026_04_GB_TB"]}
    assert got["Jalon Daniels"]["team"] == "TB" and got["Jalon Daniels"]["team_ai"] == "KC"
    assert got["Jalon Daniels"]["check"] == "unclear" and got["Mike Evans"]["check"] == "verified"
    assert "Jack Keenum" not in got                                      # unclear ranked below the 3 shown
    assert got["Unchecked Guy"]["check"] == "unchecked"
    (tmp_path / "news_audit.json").write_text(json.dumps({"overall": {"n": 45, "correct": 42}, "qb": {"n": 8, "correct": 8},
                                                          "promotion": {"eligible": False, "n": 3}, "rows": [1] * 5}))
    audit = bd.load_news_audit(tmp_path / "news_audit.json")
    assert audit["overall"]["correct"] == 42 and "rows" not in audit
    assert bd.load_news_audit(tmp_path / "missing.json") is None
    payload = {"predictions": {"upcoming": []}, "news_ai": by, "news_audit": audit}
    body = bd.render(payload)
    assert "ainewsacc" in body and '"correct": 42' in body


def test_pipeline_saves_sleeper_roster_or_falls_back(tmp_path):
    from nflpred import pipeline
    (tmp_path / "news_llm.jsonl").write_text(json.dumps(_sig("Jalon Daniels", "KC", position="QB")) + "\n")
    nflv = pd.DataFrame({"display_name": ["Jalon Daniels"], "first_name": ["Jalon"], "football_name": ["Jalon"],
                         "last_name": ["Daniels"], "latest_team": ["TB"], "position": ["QB"], "gsis_id": ["a"],
                         "status": ["ACT"], "last_season": [2026]})
    r = pipeline.save_roster_and_check(pd.DataFrame(columns=["source"]), {"sleeper": "skipped this run"}, nflv, 2026,
                                       tmp_path)
    assert r["source"] == "nflverse" and r["news_check"]["team_mismatch"] == 1
    live = pd.DataFrame({"full_name": [f"P{i} X{i}" for i in range(999)] + ["Jalon Daniels"],
                         "first_name": None, "last_name": None, "team": ["NE"] * 999 + ["TB"], "position": "QB",
                         "gsis_id": None, "depth_order": None, "source": "Sleeper"})
    r = pipeline.save_roster_and_check(live, {"sleeper": "ok (1000 rostered players)"}, nflv, 2026, tmp_path)
    assert r["source"] == "sleeper" and R.load(tmp_path)["source"] == "sleeper" and r["news_check"]["checked"] == 1
    row = json.loads((tmp_path / "news_llm.jsonl").read_text())
    assert row["team_verified"] == "TB" and row["verify"]["roster_source"] == "sleeper"


def test_news_context_extraction_logged(tmp_path):
    from datetime import datetime, timezone
    from nflpred import news_llm as NL
    items = [{"title": "Bears hand play-calling to OC", "link": "u9", "published": None, "summary": "..."}]
    def fake(batch, key):
        NL.LAST_CONTEXT[:] = [{"item": 0, "team": "CHI", "category": "play_caller_change", "summary": "OC calls plays",
                               "direction": "unclear", "game_week_relevant": True, "certainty": 0.9, "quote": "hand play-calling"},
                              {"item": 0, "team": "XXX", "category": "other"}]
        NL.LAST_CONTEXT[:] = [c for c in NL.LAST_CONTEXT if c.get("team") in NL.TEAMS]
        return []
    r = NL.scan(tmp_path, "k", datetime(2026, 10, 2, tzinfo=timezone.utc), fetcher=lambda u: items, llm=fake, roster={})
    assert r["context_items"] == 1
    import json
    row = json.loads((tmp_path / "news_context.jsonl").read_text().splitlines()[0])
    assert row["category"] == "play_caller_change" and row["team"] == "CHI"


# ------------------------------------------------------------------------------------------- sources: Bluesky, team sites, accuracy
def test_bluesky_items_discover_and_source_accuracy(tmp_path):
    from nflpred import news_sources as NS, news_llm as NL
    assert len(NS.team_site_feeds()) == 4 and all("site%3Achiefs.com" in u or "chiefs" not in u for u in NS.team_site_feeds().values())
    assert any("team_sites_1" == k for k in NL.FEEDS)

    def get(method, params):
        if method == "app.bsky.actor.getProfile":
            if params["actor"] == "chiefs.com":
                return {"handle": "chiefs.com", "displayName": "Kansas City Chiefs", "followersCount": 90000}
            if params["actor"] == "kcbeat.bsky.social":
                return {"handle": "kcbeat.bsky.social", "followersCount": 5000}
            if params["actor"] == "tiny.bsky.social":
                return {"handle": "tiny.bsky.social", "followersCount": 20}
            raise OSError("not found")
        if method == "app.bsky.actor.searchActors":
            if params["q"] == "Chiefs reporter":
                return {"actors": [{"handle": "kcbeat.bsky.social", "displayName": "KC Beat", "description": "I cover the Chiefs for the Star"},
                                   {"handle": "tiny.bsky.social", "description": "Chiefs beat writer"},
                                   {"handle": "fan.bsky.social", "description": "Chiefs fan, dad"}]}
            return {"actors": []}
        if method == "app.bsky.feed.getAuthorFeed":
            return {"feed": [{"post": {"uri": "at://did:x/app.bsky.feed.post/abc", "record": {"text": "Patrick Mahomes  (ankle) will start Sunday.", "createdAt": "2026-10-03T18:00:00.000Z"}}},
                             {"post": {"uri": "at://did:x/app.bsky.feed.post/r", "record": {"text": "repost"}}, "reason": {"$type": "app.bsky.feed.defs#reasonRepost"}}]}
    acc = NS.discover(tmp_path, get=get, now=datetime(2026, 10, 4, tzinfo=timezone.utc))
    assert acc["chiefs.com"]["how"] == "team_domain" and "kcbeat.bsky.social" in acc
    assert "tiny.bsky.social" not in acc and "fan.bsky.social" not in acc and "jordanraanan.bsky.social" in acc
    assert NS.discover(tmp_path, get=lambda *a: 1 / 0, now=datetime(2026, 10, 5, tzinfo=timezone.utc)) == acc   # cached for a week
    items, rep = NS.bluesky_items({"kcbeat.bsky.social": {"team": "KC"}}, get=get)
    assert len(items) == 1 and items[0]["outlet"] == "bsky:kcbeat.bsky.social (KC)" and items[0]["title"].startswith("Patrick Mahomes (ankle)")
    assert items[0]["link"] == "https://bsky.app/profile/kcbeat.bsky.social/post/abc" and items[0]["published"] == "2026-10-03T18:00+00:00"
    assert NS.source_key({"outlet": "bsky:kcbeat.bsky.social (KC)"}) == "bsky:kcbeat.bsky.social"
    assert NS.source_key({"source": "google_injury", "title": "Mahomes to start - Arrowhead Pride"}) == "Arrowhead Pride"
    assert NS.source_key({"source": "pft", "title": "x - y"}) == "pft"
    # accuracy: misreads are not held against the source; muting needs >= 10 judged
    ev = lambda src, i: {"source": src, "title": f"t{i}", "player": "P", "signal": "out", "link": f"l{src}{i}"}
    rows = [{"verdict": "correct", "evidence": [ev("pft", i), ev("pft", i)]} for i in range(3)]
    rows += [{"verdict": "wrong", "evidence": [ev("pft", 9)]}] + [{"verdict": "wrong", "evidence": [ev("cbs", i)]} for i in range(12)]
    errors = {NS.error_id(ev("pft", 9)): {"why": "misread"}, **{NS.error_id(ev("cbs", i)): {"why": "source_wrong"} for i in range(12)}}
    t = NS.source_table({"rows": rows}, errors)
    assert t["pft"]["n"] == 4 and t["pft"]["correct"] == 3 and t["pft"]["misread"] == 1 and t["pft"]["accuracy"] == 1.0
    assert t["pft"]["weight"] == round(7 / 8, 3) and not t["pft"]["muted"]
    assert t["cbs"]["source_wrong"] == 12 and t["cbs"]["muted"] and t["cbs"]["weight"] == round(4 / 17, 3)


def test_classify_errors_once(tmp_path):
    from nflpred import news_sources as NS
    row = {"verdict": "wrong", "team": "CHI", "truth": "did not start",
           "evidence": [{"seen_at": "2026-10-03T00:23+00:00", "source": "pft", "title": "Bagent to start", "quote": "will start",
                         "link": "L", "player": "Tyson Bagent", "signal": "will_start"}]}
    (tmp_path / "news_audit.json").write_text(json.dumps({"rows": [row, {"verdict": "correct", "evidence": []}]}))
    calls = []
    errs = NS.classify_errors(tmp_path, "k", llm=lambda p: calls.append(p) or {"0": "source_wrong"})
    assert list(errs.values())[0]["why"] == "source_wrong" and "Tyson Bagent" in calls[0]
    NS.classify_errors(tmp_path, "k", llm=lambda p: calls.append(p) or {})
    assert len(calls) == 1                                                   # cached
    out = NS.refresh(tmp_path, None)
    assert out["sources"]["pft"]["source_wrong"] == 1 and (tmp_path / "news_sources.json").exists()
