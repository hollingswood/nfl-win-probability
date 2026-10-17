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


def test_reread_drops_picks_the_article_no_longer_shows(tmp_path):
    from nflpred import picks_score as S
    from nflpred.picks_log import pick_key
    good = dict(GAME, analyst="Brandon Marcello", market="spread", pick="Alabama", link="L", seen_at="2026-10-16T09:57+00:00")
    bad = dict(GAME, analyst="Brandon Marcello", market="moneyline", pick="Tennessee", link="L", seen_at="2026-10-16T09:57+00:00")
    other = dict(GAME, analyst="X Y", market="spread", pick="Alabama", link="OTHER", seen_at="2026-10-16T09:57+00:00")
    (tmp_path / "picks_reads.json").write_text(json.dumps({"k": {"link": "L", "last": "2026-10-17T18:00+00:00",
                                                                 "keys": ["|".join(map(str, pick_key(good)))]}}))
    kept, n = S.drop_superseded([good, bad, other], tmp_path)
    assert n == 1 and kept == [good, other]


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
