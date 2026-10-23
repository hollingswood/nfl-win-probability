"""'Already read' lists (news_llm.Seen): trimmed by age, never by id order (2026-10-22 credit leak)."""
import json

from nflpred.news_llm import Seen


def test_new_ids_are_kept_whatever_they_sort_like(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text(json.dumps([f"{i:016x}" for i in range(10, 25000)]))      # old-format list, past the old cap
    s = Seen(p, "2026-10-22")
    s.update(["0000000000000001", "ffffffffffffffff"])
    s.save(p)
    again = Seen(p, "2026-10-23")
    assert "0000000000000001" in again and "ffffffffffffffff" in again and len(again) == 25000 - 10 + 2


def test_trimmed_by_age_and_merged(tmp_path):
    p = tmp_path / "seen.json"
    p.write_text(json.dumps({"old": "2026-08-01", "recent": "2026-10-20"}))
    s = Seen(p, "2026-10-22")
    s.add("today"); s.add("recent")
    assert s["recent"] == "2026-10-20"            # first sighting kept
    s.save(p, keep_days=30)
    assert set(json.loads(p.read_text())) == {"recent", "today"}
