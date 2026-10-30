"""News features use only items published AND seen before the cutoff, after the team's previous game."""
import json
from datetime import datetime, timezone

from nflpred import news_features as NF


def test_no_hindsight_and_windows(tmp_path):
    rows = [
        {"team": "KC", "link": "a", "published": "2026-10-28T12:00+00:00", "seen_at": "2026-10-28T12:30+00:00",
         "impact_pts": -3, "certainty": 1.0, "novelty": "new", "sentiment": -2},
        {"team": "KC", "link": "b", "published": "2026-10-28T12:00+00:00", "seen_at": "2026-11-02T12:30+00:00",   # seen too late
         "impact_pts": 5, "certainty": 1.0, "novelty": "new", "sentiment": 2},
        {"team": "KC", "link": "c", "published": "2026-10-20T12:00+00:00", "seen_at": "2026-10-20T12:30+00:00",   # before last game
         "impact_pts": 5, "certainty": 1.0, "novelty": "new", "sentiment": 2},
        {"team": "KC", "link": "a", "published": "2026-10-28T12:00+00:00", "seen_at": "2026-10-28T13:00+00:00",   # duplicate
         "impact_pts": -3, "certainty": 1.0, "novelty": "new", "sentiment": -2},
        {"team": "DEN", "link": "d", "published": "2026-10-29T12:00+00:00", "seen_at": "2026-10-29T12:30+00:00",
         "impact_pts": 1, "certainty": 0.5, "novelty": "expected", "sentiment": 1}]
    (tmp_path / "news_outlook.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cut = datetime(2026, 11, 1, 18, tzinfo=timezone.utc)
    f = NF.game_features("KC", "DEN", cut, "2026-10-26T03:00+00:00", None, tmp_path)
    assert f["home_impact_new"] == -3 and f["home_n_items"] == 1 and f["away_impact_new"] == 0
    assert f["away_impact_all"] == 0.5 and f["diff_impact_all"] == -3.5 and f["home_sentiment"] == -2
