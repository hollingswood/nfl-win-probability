"""Transfer-aware preseason prior for the college ratings. Written 2026-10-04 before running.

Research (reports/NFL and college betting edges.md): Bill Connelly credits a transfer's production from his old team to his
new one and says recruiting matters much less since the portal, so priors fit on pre-2021 seasons may overweight talent.
Test: the walk-forward ratings in fast.py with prior = a*last + b*talent_z + c*last*(returning - mean) + d*transfer_z, where
transfer_z = season z-score of (offensive PPA of incoming transfers last season - offensive PPA of outgoing transfers)
(CFBD player/portal + ppa/players/season; offense only, since PPA is offensive).
Fit (a, b, c, d) and, separately, the baseline (a, b, c) on 2021-2023 (portal era) by MAE of FBS-vs-FBS games in weeks
1-6, where the prior matters most. Score both once on 2024-2025 weeks 1-6, plus the pre-portal defaults. Adopt only if
the transfer version beats the refit baseline by >= 0.10 points of MAE on 2024-25.
"""
import gzip
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import fast as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "cfb" / "raw"


def transfer_z() -> dict:
    out = {}
    for s in range(2021, 2027):
        portal = json.load(gzip.open(RAW / f"player_portal_{s}.json.gz"))
        ppa = json.load(gzip.open(RAW / f"ppa_players_season_{s - 1}.json.gz"))
        prod = {}
        for p in ppa:
            prod[(p["name"].lower(), p["team"])] = float((p.get("totalPPA") or {}).get("all") or 0.0)
        tin, tout = {}, {}
        for t in portal:
            name = f"{t.get('firstName', '')} {t.get('lastName', '')}".strip().lower()
            v = prod.get((name, t.get("origin")), 0.0)
            if v == 0.0:
                continue
            if t.get("destination"):
                tin[t["destination"]] = tin.get(t["destination"], 0.0) + v
            tout[t["origin"]] = tout.get(t["origin"], 0.0) + v
        teams = set(tin) | set(tout)
        net = pd.Series({k: tin.get(k, 0.0) - tout.get(k, 0.0) for k in teams})
        z = (net - net.mean()) / net.std()
        for k, v in z.items():
            out[(s, k)] = float(v)
    return out


def run(g, tal, ret, tz_map, a, b, c, d, seasons, cap=28.0, lam=4.0, fcs=-20.0):
    """fast.run with one extra prior term (d * transfer_z)."""
    teams = sorted(set(g.home) | set(g.away)); ix = {t: i for i, t in enumerate(teams)}; k = len(teams)
    div = dict(zip(g.home, g.home_div)); div.update(dict(zip(g.away, g.away_div)))
    final, out = {}, []
    for s in range(seasons[0], seasons[1] + 1):
        S = g[g.season == s].sort_values("start")
        tz = tal[tal.season == s].set_index("team").talent; tz = (tz - tz.mean()) / tz.std() if len(tz) else tz
        rp = ret[ret.season == s].set_index("team").percentPPA; rpm = rp.mean() if len(rp) else 0.5
        last = final.get(s - 1, {})
        fm = np.mean([v for t, v in last.items() if div.get(t) == "fcs"]) if last else fcs
        prior = np.zeros(k + 1)
        for t, i in ix.items():
            if div.get(t) != "fbs":
                prior[i] = last.get(t, fm) * 0.8 + 0.2 * fm; continue
            lm = last.get(t, 0.0)
            prior[i] = (a * lm + b * float(tz.get(t, 0.0) if len(tz) else 0) + c * lm * ((float(rp.get(t, rpm)) - rpm) if len(rp) else 0)
                        + d * tz_map.get((s, t), 0.0))
        A = np.zeros((k + 1, k + 1)); A[np.arange(k), np.arange(k)] = lam; A[k, k] = 1e-6
        bv = np.zeros(k + 1); R = prior.copy()
        for sl in sorted(S.slot.unique()):
            cur = S[S.slot == sl]
            h = cur.home.map(ix).to_numpy(); aw = cur.away.map(ix).to_numpy(); nh = (~cur.neutral).to_numpy(float)
            out.append(pd.DataFrame({"game_id": cur.game_id.to_numpy(), "m_pred": R[h] - R[aw] + nh * R[k]}))
            done = cur[cur.completed & cur.margin.notna()]
            if done.empty:
                continue
            h = done.home.map(ix).to_numpy(); aw = done.away.map(ix).to_numpy(); nh = (~done.neutral).to_numpy(float)
            y = done.margin.clip(-cap, cap).to_numpy(float) - (prior[h] - prior[aw] + nh * prior[k])
            for hi, ai, n_, yi in zip(h, aw, nh, y):
                idx = [hi, ai, k]; x = np.array([1.0, -1.0, n_])
                A[np.ix_(idx, idx)] += np.outer(x, x); bv[idx] += x * yi
            R = prior + np.linalg.solve(A, bv)
        final[s] = {t: R[i] for t, i in ix.items()}
    return pd.concat(out)


def early_mae(pred, G, seasons):
    Q = G[G.season.between(*seasons) & (G.week <= 6)].merge(pred, on="game_id")
    return float((Q.margin - Q.m_pred).abs().mean()), int(len(Q)), Q


def main():
    g, tal, ret = F.prep()
    tzm = transfer_z()
    G = g[(g.home_div == "fbs") & (g.away_div == "fbs") & g.margin.notna() & (g.season_type == "regular")][["game_id", "season", "week", "margin"]]
    grid_a, grid_b, grid_c = (0.5, 0.6, 0.69, 0.8), (1.5, 2.5, 3.35, 4.5), (0.0, 0.3, 0.6)
    res = {}
    def best(dgrid):
        top = None
        for a, b, c, d in itertools.product(grid_a, grid_b, grid_c, dgrid):
            pr = run(g, tal, ret, tzm, a, b, c, d, (2014, 2023))
            m, n, _ = early_mae(pr, G, (2021, 2023))
            if top is None or m < top[0]:
                top = (m, a, b, c, d)
        return top
    base = best((0.0,))
    trans = best((0.5, 1.0, 2.0, 3.0))
    for name, (m_fit, a, b, c, d) in (("pre-portal defaults", (None, 0.69, 3.35, 0.30, 0.0)), ("baseline refit 2021-23", base), ("transfer refit 2021-23", trans)):
        pr = run(g, tal, ret, tzm, a, b, c, d, (2014, 2025))
        m, n, Q = early_mae(pr, G, (2024, 2025))
        res[name] = {"a": a, "b": b, "c": c, "d": d, "mae_fit_2021_23": None if m_fit is None else round(m_fit, 3),
                     "mae_2024_25_weeks_1_6": round(m, 3), "games": n}
        print(name, res[name], flush=True)
    res["adopt"] = bool(res["baseline refit 2021-23"]["mae_2024_25_weeks_1_6"] - res["transfer refit 2021-23"]["mae_2024_25_weeks_1_6"] >= 0.10)
    (ROOT / "output/research/cfb/transfer_prior.json").write_text(json.dumps(res, indent=1))
    print("adopt:", res["adopt"])


if __name__ == "__main__":
    main()
