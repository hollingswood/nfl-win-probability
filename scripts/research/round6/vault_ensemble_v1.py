"""One blind-vault look for ensemble_v1 (output/research/round6/ensemble_v1_prereg.json). Do not re-run: blind.open_vault
refuses a second look and the result is in blind/registry.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from nflpred import blind as B, model as M   # noqa: E402


def fit_predict(tr, te):
    ps = []
    for kind in ("margin", "logistic", "gbm"):
        m = M.make_model(kind)
        m = m.fit(tr[M.FEATURES], (tr.home_score - tr.away_score).values) if kind == "margin" \
            else m.fit(tr[M.FEATURES], tr.home_win.astype(int))
        ps.append(np.asarray(M.predict(m, te), float))
    return np.mean(ps, axis=0)


if __name__ == "__main__":
    print(json.dumps(B.open_vault("ensemble_v1", fit_predict), indent=1))
