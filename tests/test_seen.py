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


def test_budget_status_alerts_and_out_of_credits(tmp_path):
    from datetime import datetime, timezone
    from nflpred import claude_budget as B
    cfg = tmp_path / "budget.json"
    cfg.write_text(json.dumps({"topups": [{"at": "2026-10-23T22:35+00:00", "usd": 20}]}))
    rows = [{"at": f"2026-10-24T{h:02d}:00+00:00", "component": "picks", "usd": 0.6} for h in range(24)] + \
           [{"at": "2026-10-23T10:00+00:00", "component": "picks", "usd": 5.0}]      # before the top-up: not counted
    (tmp_path / "claude_usage.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    now = datetime(2026, 10, 25, 0, 0, tzinfo=timezone.utc)
    orig = B.CONFIG
    B.CONFIG = cfg
    try:
        s = B.status(tmp_path, now, cfg)
        assert s["spent"] == 14.4 and s["left"] == 5.6 and s["by_component"] == {"picks": 14.4}
        sent = []
        B.check(tmp_path, now, send=lambda t, m, **k: sent.append(t))
        assert sent == ["Claude credits 50% used"]                       # 72%: only the 50% line crossed, once
        B.check(tmp_path, now, send=lambda t, m, **k: sent.append(t))
        assert sent == ["Claude credits 50% used"]
        B.out_of_credits("picks", "Your credit balance is too low", tmp_path, send=lambda t, m, **k: sent.append(t))
        B.out_of_credits("picks", "Your credit balance is too low", tmp_path, send=lambda t, m, **k: sent.append(t))
        assert sent[-1] == "Claude credits ran out" and sent.count("Claude credits ran out") == 1
    finally:
        B.CONFIG = orig
