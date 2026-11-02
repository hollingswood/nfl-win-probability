import json
from datetime import datetime, timezone

from nflpred import odds_budget as OB

U = timezone.utc


def _setup(tmp, used, left, start="2026-11-01T00:00+00:00", last=None):
    (tmp / "odds_credits.json").write_text(json.dumps({"checked_at": "2026-11-10T00:00+00:00",
                                                       "x-requests-used": str(used), "x-requests-remaining": str(left)}))
    (tmp / "odds_budget_state.json").write_text(json.dumps({"cycle_start": start, "last_used": last if last is not None else used, "alerted": []}))


def test_normal_pace_is_ok(tmp_path):
    _setup(tmp_path, 5000, 15000)                     # 9 days in, 556/day -> ~17.2k projected
    s = OB.status(tmp_path, datetime(2026, 11, 10, tzinfo=U))
    assert s["plan"] == 20000 and s["tier"] == "ok" and s["renews"].startswith("2026-12-01")
    assert all(OB.allow(k, tmp_path, datetime(2026, 11, 10, tzinfo=U)) for k in ("futures", "cfb", "nfl"))


def test_fast_pace_pauses_low_priority(tmp_path):
    now = datetime(2026, 11, 10, 3, tzinfo=U)
    _setup(tmp_path, 9000, 11000)                     # 1000/day -> 30k projected
    assert OB.status(tmp_path, now)["tier"] == "tight"
    assert not OB.allow("futures", tmp_path, now) and not OB.allow("cfb", tmp_path, now)
    assert OB.allow("cfb", tmp_path, now, peak=True) and OB.allow("nfl", tmp_path, now)
    sent = []
    OB.check(tmp_path, now, send=lambda *a, **k: sent.append(a[1]))
    OB.check(tmp_path, now, send=lambda *a, **k: sent.append(a[1]))
    assert len(sent) == 1 and "on pace to run out" in sent[0]       # once per cycle


def test_critical_nfl_every_other_hour(tmp_path):
    _setup(tmp_path, 19500, 500)
    assert OB.status(tmp_path, datetime(2026, 11, 28, tzinfo=U))["tier"] == "critical"
    assert OB.allow("nfl", tmp_path, datetime(2026, 11, 28, 4, tzinfo=U))
    assert not OB.allow("nfl", tmp_path, datetime(2026, 11, 28, 5, tzinfo=U))
    assert not OB.allow("cfb", tmp_path, datetime(2026, 11, 28, 5, tzinfo=U), peak=True)


def test_renewal_starts_a_new_cycle(tmp_path):
    _setup(tmp_path, 40, 19960, start="2026-11-01T00:00+00:00", last=18000)
    s = OB.check(tmp_path, datetime(2026, 12, 1, 1, tzinfo=U), send=lambda *a, **k: None)
    st = json.loads((tmp_path / "odds_budget_state.json").read_text())
    assert s["tier"] == "ok" and st["alerted"] == [] and st["cycle_start"] > "2026-11-30"
