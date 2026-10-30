"""Blind evaluation: the vault stays sealed, each model gets one look, prospective scores stay hidden until a checkpoint."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflpred import blind as B, data as D


def test_vault_seasons_refused_by_the_loaders():
    with pytest.raises(B.VaultSealed):
        D.load_pbp([2008])
    with pytest.raises(B.VaultSealed):
        D.load_injuries([2010, 2020])
    B.guard([2012, 2025])                      # production seasons are fine


def _frame():
    from nflpred import model as M
    rng = np.random.default_rng(0)
    rows = []
    for s in range(2002, 2012):
        for i in range(60):
            x = {f: rng.normal() for f in M.FEATURES}
            margin = 3 * x[M.FEATURES[0]] + rng.normal(0, 13)
            rows.append({**x, "season": s, "home_score": 24 + margin / 2, "away_score": 24 - margin / 2,
                         "home_win": float(margin > 0), "vegas_home_prob": 0.5})
    return pd.DataFrame(rows)


def test_one_look_per_model(tmp_path):
    reg = tmp_path / "registry.json"
    pre = tmp_path / "prereg.json"; pre.write_text("{}")
    with pytest.raises(ValueError):
        B.open_vault("x", lambda tr, te: np.full(len(te), 0.5), path=reg, data=_frame())
    B.register("coin", "always 50%", pre, path=reg)
    with pytest.raises(ValueError):
        B.register("coin", "again", pre, path=reg)
    r = B.open_vault("coin", lambda tr, te: np.full(len(te), 0.5), path=reg, data=_frame())
    assert r["vs_production"]["n"] == 7 * 60 and r["vs_production"]["se"] is not None
    with pytest.raises(ValueError):
        B.open_vault("coin", lambda tr, te: np.full(len(te), 0.5), path=reg, data=_frame())
    assert json.loads(reg.read_text())["models"]["coin"]["vault_result"]["seasons_scored"] == "2005-2011"


def test_prospective_scores_stay_sealed_until_the_checkpoint(tmp_path):
    folder = tmp_path / "prospective"
    cps = tmp_path / "cp.json"; cps.write_text(json.dumps({"checkpoints": ["2026-12-01"]}))
    t0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = [{"game_id": "g1", "kickoff": "2026-11-02T18:00:00+00:00", "home_win_prob": 0.7}]
    assert B.log_prospective("production", [dict(rows[0], home_win_prob=0.6)], t0, folder) == 1
    assert B.log_prospective("news_v1", rows, t0, folder) == 1
    assert B.log_prospective("news_v1", [dict(rows[0], kickoff="2026-10-31T18:00:00+00:00")], t0, folder) == 0   # started
    res = {"g1": {"home_win": 1.0, "vegas_home_prob": 0.65}}
    early = B.reveal(res, datetime(2026, 11, 20, tzinfo=timezone.utc), folder, cps)
    assert early["checkpoints"]["2026-12-01"] == "sealed until this date"
    late = B.reveal(res, datetime(2026, 12, 2, tzinfo=timezone.utc), folder, cps)
    assert late["checkpoints"]["2026-12-01"]["news_v1"]["vs_production"]["improvement"] > 0


def test_live_candidates_are_registered():
    from nflpred import blind_candidates as BC
    reg = json.loads((B.DIR / "registry.json").read_text())["models"] if (B.DIR / "registry.json").exists() else {}
    assert all(n in reg and "prospective" in reg[n]["kinds"] for n in BC.LIVE)
