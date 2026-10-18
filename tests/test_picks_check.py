"""Mistake-proofing for published picks (src/nflpred/picks_check.py). The CBS Alabama-Tennessee table of 2026-10-16 is the
regression case: the AI shifted picks one column over and read the against-the-spread row as moneyline."""
import json

from nflpred import picks_check as PC

NAMES = ["Tom Fornelli", "Chip Patterson", "Brandon Marcello", "John Talty", "Richard Johnson", "Shehan Jeyarajah", "David Cobb"]
ATS = ["Alabama", "Tennessee", "Alabama", "Alabama", "Tennessee", "Alabama", "Alabama"]
LOGO = '<img src="x.png" alt="{0}"><span>{0}</span>'
CBS = ("<p>Look for Bama to pull away in the fourth quarter. Pick: Alabama -8.5</p><table><tr><th></th>"
       + "".join(f'<th><img alt="{n} headshot"><span>{n}</span></th>' for n in NAMES) + "</tr>"
       + "<tr><td>BAMA -8.5</td>" + "".join(f"<td>{LOGO.format(t)}</td>" for t in ATS) + "</tr>"
       + "<tr><td>SU</td>" + "".join(f"<td>{LOGO.format('Alabama')}</td>" for _ in NAMES) + "</tr></table>")
GAME = {"sport": "cfb", "away_team": "Alabama", "home_team": "Tennessee"}


def test_table_becomes_explicit_rows_per_analyst():
    t = PC.html_to_text(CBS)
    rows = PC.records(t)
    assert [r["kind"] for r in rows] == ["against the spread", "straight up"]
    assert rows[0]["picks"]["Brandon Marcello"] == "Alabama" and rows[0]["picks"]["Chip Patterson"] == "Tennessee"
    assert rows[0]["picks"]["Richard Johnson"] == "Tennessee" and set(rows[1]["picks"].values()) == {"Alabama"}
    assert "Tom Fornelli headshot" not in t or "Tom Fornelli picks Alabama" in t


def test_validate_rejects_the_shifted_and_mislabeled_picks():
    t = PC.html_to_text(CBS)
    ai = [dict(GAME, analyst="Brandon Marcello", market="moneyline", pick="Tennessee", line=None, confidence_words=""),
          dict(GAME, analyst="Shehan Jeyarajah", market="moneyline", pick="Tennessee", line=None, confidence_words=""),
          dict(GAME, analyst="Chip Patterson", market="moneyline", pick="Tennessee", line=None, confidence_words=""),
          dict(GAME, analyst="Chip Patterson", market="spread", pick="Tennessee", line=8.5, confidence_words=""),
          dict(GAME, analyst="Brandon Marcello", market="spread", pick="Alabama", line=-8.5, confidence_words=""),
          dict(GAME, analyst="John Talty", market="moneyline", pick="Alabama", line=None, confidence_words="")]
    kept, rej = PC.validate(ai, t)
    assert {(p["analyst"], p["market"], p["pick"]) for p in kept} == {
        ("Chip Patterson", "spread", "Tennessee"), ("Brandon Marcello", "spread", "Alabama"), ("John Talty", "moneyline", "Alabama")}
    assert all("table" in r["reject_reason"] for r in rej) and len(rej) == 3


def test_validate_rejects_props_quotes_and_both_sides():
    t = "Bet: Alabama -8.5. I like the Tide. Wan'Dale Robinson over 4.5 receptions. Jane Doe likes the over 51.5."
    ai = [dict(GAME, analyst=None, market="moneyline", pick="Wan'Dale Robinson", confidence_words="Over 4.5 receptions"),
          dict(GAME, analyst=None, market="spread", pick="Alabama", line=-8.5, confidence_words="Bama is a lock today"),
          dict(GAME, analyst="Jane Doe", market="total", pick="over", line=51.5, confidence_words="likes the over 51.5"),
          dict(GAME, analyst="Jane Doe", market="total", pick="under", line=51.5, confidence_words=""),
          dict(GAME, analyst="Mystery Man", market="spread", pick="Alabama", line=-8.5, confidence_words="Bet: Alabama -8.5"),
          dict(GAME, analyst=None, market="spread", pick="Alabama Crimson Tide", line=-8.5, confidence_words="Bet: Alabama -8.5")]
    kept, rej = PC.validate(ai, t)
    reasons = sorted(r["reject_reason"] for r in rej)
    assert [p["pick"] for p in kept] == ["Alabama Crimson Tide"]
    assert reasons == sorted(["picked side is not one of the two teams", "quote not found in the article",
                              "the article has this analyst on both sides", "the article has this analyst on both sides",
                              "analyst not named in the article"])


def test_same_team_matching():
    assert PC.same_team("Alabama", "Alabama Crimson Tide") and PC.same_team("Colts", "Indianapolis Colts")
    assert not PC.same_team("Ohio State", "Michigan State") and not PC.same_team("Wan'Dale Robinson", "New York Giants")


def test_grade_check_catches_team_totals_and_wrong_spreads():
    assert PC.grade_check("total", 21.5, 51.5) and PC.grade_check("total", 38.5, 51.0)
    assert PC.grade_check("total", 49.5, 51.5) is None and PC.grade_check("total", None, 51.5) is None
    assert PC.grade_check("spread", -20, 8.5) and PC.grade_check("spread", 9, 8.5) is None


def test_reread_drops_only_picks_the_article_now_shows_on_the_other_side():
    from nflpred import picks_score as S
    team = lambda t: {"alabama": "Alabama", "alabama crimson tide": "Alabama", "tennessee": "Tennessee",
                      "tennessee volunteers": "Tennessee"}.get(str(t or "").lower())
    g = {"away_actual": "Alabama", "home_actual": "Tennessee", "game_id": "1"}
    def item(market, side, pick, analyst="Brandon Marcello"):
        return dict(GAME, analyst=analyst, market=market, pick=pick, link="L", seen_at="2026-10-16T09:57", _g=dict(g, side=side))
    wrong = item("moneyline", "Tennessee", "Tennessee")
    right = item("spread", "Alabama", "Alabama")
    missing = item("spread", "Alabama", "Alabama", analyst="John Talty")      # absent from the re-read: kept
    reads = {"k": {"link": "L", "last": "2026-10-17T12:00", "picks": [
        {"analyst": "Brandon Marcello", "away_team": "Alabama Crimson Tide", "home_team": "Tennessee Volunteers",
         "market": "moneyline", "pick": "Alabama Crimson Tide"},
        {"analyst": "Marcello", "away_team": "Tennessee", "home_team": "Alabama", "market": "spread", "pick": "Alabama"}]}}
    v = S.reread_verdicts([wrong, right, missing], reads, team)
    assert v == {id(wrong): "the article's latest read shows the other side"}
    # the read's picks table rejected it
    reads["k"]["picks"] = []
    reads["k"]["rejected"] = [{"analyst": "Brandon Marcello", "away_team": "Alabama", "home_team": "Tennessee",
                               "market": "moneyline", "pick": "Tennessee", "why": 'contradicts the picks table (row "SU": Alabama)'}]
    assert S.reread_verdicts([wrong, right], reads, team) == {id(wrong): "the article's picks table disagrees"}


def test_articles_read_before_the_checks_are_reread_once(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from nflpred import picks_log as PL
    monkeypatch.setattr(PL, "HIST", tmp_path)
    (tmp_path / "picks_seen.json").write_text(json.dumps(["k1"]))
    (tmp_path / "picks_reads.json").write_text(json.dumps({"k1": {
        "first": "2026-10-15T21:26+00:00", "last": "2026-10-17T12:26+00:00", "n": 4, "link": "L", "picks_found": 1}}))
    calls = []
    def llm(text, key):
        calls.append(text)
        return [dict(GAME, analyst="Brandon Marcello", market="moneyline", pick="Tennessee", line=None, confidence_words="")]
    now = datetime(2026, 10, 17, 18, tzinfo=timezone.utc)
    rep = PL.scan(now=now, api_key="k", fetcher=lambda u: [], texter=lambda u: PC.html_to_text(CBS), llm=llm)
    reads = json.loads((tmp_path / "picks_reads.json").read_text())
    assert len(calls) == 1 and rep["picks"] == 0 and reads["k1"]["checks"] == PC.VERSION and reads["k1"]["keys"] == []
    assert reads["k1"]["picks"] == [] and reads["k1"]["rejected"][0]["pick"] == "Tennessee" and "table" in reads["k1"]["rejected"][0]["why"]
    PL.scan(now=now, api_key="k", fetcher=lambda u: [], texter=lambda u: PC.html_to_text(CBS), llm=llm)
    assert len(calls) == 1                                   # once per checks version


def test_reason_is_kept_only_when_the_article_says_it():
    t = ("Fornelli: Tennessee's secondary has allowed explosive plays all season and Alabama's receivers should feast. "
         "Pick: Alabama -8.5")
    ok = dict(GAME, analyst="Tom Fornelli", market="spread", pick="Alabama", line=-8.5, confidence_words="Pick: Alabama -8.5",
              reason="Tennessee's secondary keeps allowing explosive plays, so Alabama's receivers should feast.")
    made_up = dict(ok, pick="Alabama", market="moneyline", reason="Nico Iamaleava returns from injury and the weather favors the defense.")
    kept, rej = PC.validate([ok, made_up], t)
    assert not rej and kept[0]["reason"].startswith("Tennessee's secondary") and kept[1]["reason"] == ""


def test_scores_carry_reasons_found_on_a_later_read(tmp_path):
    from nflpred import picks_score as S
    from nflpred.picks_log import pick_key
    p = dict(GAME, analyst="Tom Fornelli", market="spread", pick="Alabama", line=-8.5, confidence_words="")
    (tmp_path / "picks_reasons.json").write_text(json.dumps({"|".join(map(str, pick_key(p))): {
        "reason": "Bama's receivers should feast.", "quote": "Pick: Alabama -8.5", "at": "2026-10-17T00:00"}}))
    assert S._why(dict(p, reason="x"))["reason"] == "x"
    import inspect
    assert "picks_reasons.json" in inspect.getsource(S.run)


def test_contradictions_and_same_team_games():
    from nflpred import picks_score as S
    def it(game, market, side, line=None, who="Brandon Marcello"):
        return {"_picker": who, "market": market, "line": line, "_g": {"game_id": game, "side": side}}
    rows = [it("bama-tenn", "moneyline", "Tennessee"), it("bama-tenn", "spread", "Alabama", -8.5),
            it("osu-ind", "spread", "Indiana", 3.5), it("osu-ind", "spread", "Ohio State", -3.5),
            it("uga-bama", "spread", "Georgia", 1.5), it("uga-bama", "moneyline", "Alabama"),       # both can be right
            it("osu-ind-2", "spread", "Ohio State", -3)]                                          # rematch = other game id
    bad = S.contradictions(rows)
    assert set(bad) == {id(r) for r in rows[:4]}
    why = PC.offline_reasons([dict(GAME, away_team="LSU", home_team="LSU", market="spread", pick="LSU", analyst="X", link="L")])
    assert why == ["the two teams are the same"]


def _fixture(tmp_path, rows, reads=None):
    import pandas as pd
    kick = pd.Timestamp("2026-10-17T19:30", tz="UTC")
    cfb = pd.DataFrame([
        {"game_id": "1", "week": 8, "home": "Tennessee", "away": "Alabama", "kick": kick, "completed": True, "home_pts": 24,
         "away_pts": 20, "spread_close": 6.5, "total_close": 51.5, "home_ml": 210, "away_ml": -260},     # Bama -6.5 at close
        {"game_id": "2", "week": 8, "home": "Indiana", "away": "Ohio State", "kick": kick, "completed": True, "home_pts": 17,
         "away_pts": 27, "spread_close": 3.5, "total_close": 49.5, "home_ml": 140, "away_ml": -165},
        {"game_id": "3", "week": 9, "home": "Michigan State", "away": "Ohio State", "kick": kick + pd.Timedelta(days=7),
         "completed": False, "home_pts": None, "away_pts": None, "spread_close": None, "total_close": None, "home_ml": None,
         "away_ml": None}])
    nfl = pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "kick", "home_score", "away_score", "spread_line",
                                "total_line", "home_moneyline", "away_moneyline", "gametime", "gameday"])
    (tmp_path / "picks_log.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if reads:
        (tmp_path / "picks_reads.json").write_text(json.dumps(reads))
    return nfl, cfb


def test_scoring_end_to_end_counts_each_pick_once_and_passes_the_gate(tmp_path):
    from datetime import datetime, timezone
    from nflpred import picks_score as S
    b = dict(sport="cfb", source="cbs_cfb", seen_at="2026-10-15T21:26")
    rows = [
        dict(b, analyst="Chip Patterson", away_team="Alabama", home_team="Tennessee", market="spread", pick="Tennessee", line=6.5, link="A"),
        # same pick, another article, other spelling and home/away swapped: counted once, earliest kept
        dict(b, analyst="Chip Patterson", away_team="Tennessee Volunteers", home_team="Alabama Crimson Tide", market="spread",
             pick="Tennessee Volunteers", line=7, link="B", seen_at="2026-10-16T10:00"),
        # contradiction across articles (named analyst): neither counts
        dict(b, analyst="Brandon Marcello", away_team="Alabama", home_team="Tennessee", market="moneyline", pick="Tennessee", link="A"),
        dict(b, analyst="Brandon Marcello", away_team="Alabama", home_team="Tennessee", market="spread", pick="Alabama", line=-8.5, link="C"),
        # 'Ohio St.' must be Ohio State, not Ohio; one-team pick resolves to the only game
        dict(b, analyst="John Talty", away_team="Ohio St.", home_team="Indiana", market="spread", pick="Ohio St.", line=-3.5, link="A"),
        dict(b, analyst="David Cobb", away_team="Ohio State", home_team="Unknown", market="moneyline", pick="Ohio State", link="A",
             published="2026-10-15T20:00+00:00"),
        # pending next week; seen after kickoff -> not counted
        dict(b, analyst="John Talty", away_team="Ohio State", home_team="Michigan St.", market="spread", pick="Ohio State", line=-20,
             link="D", seen_at="2026-10-19T12:00"),
        dict(b, analyst="Late Larry", away_team="Alabama", home_team="Tennessee", market="spread", pick="Alabama", line=-6.5, link="E",
             seen_at="2026-10-17T20:00"),
        # staff = different writers: opposite sides both count
        dict(b, analyst=None, away_team="Alabama", home_team="Tennessee", market="total", pick="over", line=51.5, link="F"),
        dict(b, analyst=None, away_team="Alabama", home_team="Tennessee", market="total", pick="under", line=51.5, link="G"),
    ]
    nfl, cfb = _fixture(tmp_path, rows)
    out = S.run(tmp_path, now=datetime(2026, 10, 20, tzinfo=timezone.utc), nfl=nfl, cfb=cfb, notify=False)
    assert out["audit"]["ok"], json.loads((tmp_path / "picks_audit.json").read_text())
    c = out["by_sport"]["cfb"]
    rec = {r["picker"]: r for r in c["pickers"]}
    assert rec["Chip Patterson"]["graded"] == 1 and rec["Chip Patterson"]["wins"] == 1      # TEN +6.5, lost by 4: covers
    assert c["picks_by_picker"]["Chip Patterson"]["graded"][0]["line"] == 6.5              # the earliest sighting
    assert "Brandon Marcello" not in rec
    assert rec["John Talty"]["graded"] == 1 and rec["John Talty"]["wins"] == 1 and rec["John Talty"]["pending"] == 1
    assert rec["David Cobb"]["wins"] == 1
    assert "Late Larry" not in rec and out["late_picks_dropped"] == 1
    assert rec["CBS Sports staff"]["graded"] == 2
    for r in c["pickers"]:
        d = c["picks_by_picker"][r["picker"]]
        assert len(d["pending"]) == r["pending"] and len(d["graded"]) == r["graded"]


def test_gate_blocks_publishing_when_numbers_dont_add_up(tmp_path):
    from datetime import datetime, timezone
    from nflpred import picks_score as S
    b = dict(sport="cfb", source="cbs_cfb", seen_at="2026-10-15T21:26", link="A")
    rows = [dict(b, analyst="Chip Patterson", away_team="Alabama", home_team="Tennessee", market="spread", pick="Tennessee", line=6.5)]
    nfl, cfb = _fixture(tmp_path, rows)
    now = datetime(2026, 10, 20, tzinfo=timezone.utc)
    good = S.run(tmp_path, now=now, nfl=nfl, cfb=cfb, notify=False)
    assert good["audit"]["ok"] and (tmp_path / "picks_scores.json").exists()
    before = (tmp_path / "picks_scores.json").read_text()
    # each kind of inconsistency is caught
    g = [dict(x) for x in [{"picker": "P", "sport": "cfb", "game_id": "1", "market": "spread", "side": "Tennessee", "pick": "Tennessee",
                            "away": "Alabama", "home": "Tennessee", "final": "20-24", "close_line": 6.5, "pick_is_home": True,
                            "result": "loss", "units": -1.0, "seen_at": "2026-10-15", "kick": "2026-10-17T19:30+00:00", "week": 8}]]
    err, _, _ = S.audit({"by_sport": {}, "hot_now": []}, g, [], now)
    assert any("recomputed" in e for e in err)                                   # wrong result
    err, _, _ = S.audit({"by_sport": {}, "hot_now": []}, [dict(g[0], result="win", units=S.WIN_110)] * 2, [], now)
    assert any("counted twice" in e for e in err)                                # duplicate
    err, _, _ = S.audit({"by_sport": {}, "hot_now": []}, [dict(g[0], result="win", units=S.WIN_110, seen_at="2026-10-18")], [], now)
    assert any("after kickoff" in e for e in err)
    bad = json.loads(json.dumps(good)); bad["by_sport"]["cfb"]["pickers"][0]["pending"] = 4
    gs = [dict(g[0], picker="Chip Patterson", result="win", units=S.WIN_110, outlet="CBS Sports", line=6.5)]
    err, _, _ = S.audit(bad, gs, [], now)
    assert any("pop-up lists" in e or "graded + pending" in e for e in err)      # table vs pop-up
    # a failing run keeps the last good file and records why
    orig = S.audit
    try:
        S.audit = lambda *a, **k: (["boom"], [], 1)
        S.run(tmp_path, now=now, nfl=nfl, cfb=cfb, notify=False)
    finally:
        S.audit = orig
    assert (tmp_path / "picks_scores.json").read_text() == before
    st = json.loads((tmp_path / "picks_audit.json").read_text())
    assert st["ok"] is False and st["errors"] == ["boom"] and st["last_good"]


def test_strict_college_names():
    from nflpred import picks_score as S
    m = S.cfb_matcher(["Michigan", "Michigan State", "Ohio", "Ohio State", "Florida", "Florida State", "USC", "Southern Miss",
                       "Ole Miss", "Mississippi State", "Missouri", "Pittsburgh", "BYU", "California", "Miami", "Miami (OH)"])
    assert [m(x) for x in ["Michigan St.", "Ohio St.", "Florida St.", "Southern Cal", "Mississippi", "Mizzou", "Pitt",
                           "Brigham Young Cougars", "Cal", "Miami (FL)", "Miami (OH)", "Ohio Bobcats", "Michigan Wolverines"]] == \
        ["Michigan State", "Ohio State", "Florida State", "USC", "Ole Miss", "Missouri", "Pittsburgh", "BYU", "California",
         "Miami", "Miami (OH)", "Ohio", "Michigan"]
    assert m("Ohio Northern") is None and m("Unknown") is None


def test_quote_match_tolerates_small_ai_drift_but_not_invention():
    art = PC._norm("The Chargers averaged 3.8 yards per play and gave up five sacks in what should have been an all-in spot at "
                   "home last week. Now they have to go to Arrowhead.")
    assert PC.quote_found("The Chargers averaged 3.8 yards per play and gave up five sacks in what should've been an all-in spot", art)
    assert not PC.quote_found("Kansas City's defense has been dominant and Mahomes is back to MVP form this season", art)


def test_scoring_crash_is_recorded_and_keeps_last_good(tmp_path):
    from nflpred import picks_score as S
    (tmp_path / "picks_audit.json").write_text(json.dumps({"ok": True, "last_good": "2026-10-18T10:00+00:00"}))
    S.record_failure("scoring crashed: KeyError: 'x'", hist=tmp_path, notify=False)
    st = json.loads((tmp_path / "picks_audit.json").read_text())
    assert st["ok"] is False and st["last_good"] == "2026-10-18T10:00+00:00" and "KeyError" in st["errors"][0]


def test_single_names_stay_separate_per_outlet():
    from nflpred import picks_score as S
    assert S.picker_of({"analyst": "Jon", "source": "yahoo_nfl"}) != S.picker_of({"analyst": "Jon", "outlet": "Corn Nation Staff"})
    assert S.picker_of({"analyst": "Chris Hummer", "source": "cbs_cfb"}) == "Chris Hummer"
