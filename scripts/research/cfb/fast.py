"""Fast margin-only walk-forward ratings for parameter checks (dev seasons only). Incremental normal equations."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
from cfbpred import data as D

def prep():
    g = D.games(range(2014, 2027)); g = g[g.home.notna() & g.away.notna()].copy()
    g["slot"] = g.week + np.where(g.season_type == "postseason", 30, 0)
    tal = D.yearly("talent", range(2014, 2027))[["year", "team", "talent"]].rename(columns={"year": "season"})
    ret = D.yearly("player_returning", range(2014, 2027))[["season", "team", "percentPPA"]]
    return g, tal, ret

def run(g, tal, ret, cap=28.0, lam=4.0, a=0.69, b=3.35, c=0.30, fcs=-20.0, last_seasons=(2014, 2026), d=0.0, tmap=None):
    """d * tmap[(season, team)]: transfer-production term (transfer_prior.py); 0 keeps the original prior."""
    tmap = tmap or {}
    teams = sorted(set(g.home) | set(g.away)); ix = {t: i for i, t in enumerate(teams)}; k = len(teams)
    div = dict(zip(g.home, g.home_div)); div.update(dict(zip(g.away, g.away_div)))
    final = {}; out = []
    for s in range(last_seasons[0], last_seasons[1] + 1):
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
                        + d * tmap.get((s, t), 0.0))
        A = np.zeros((k + 1, k + 1)); A[np.arange(k), np.arange(k)] = lam; A[k, k] = 1e-6
        bv = np.zeros(k + 1); R = prior.copy()
        for sl in sorted(S.slot.unique()):
            cur = S[S.slot == sl]
            h = cur.home.map(ix).to_numpy(); aw = cur.away.map(ix).to_numpy(); nh = (~cur.neutral).to_numpy(float)
            pred = R[h] - R[aw] + nh * R[k]
            out.append(pd.DataFrame({"game_id": cur.game_id.to_numpy(), "m_pred": pred}))
            done = cur[cur.completed & cur.margin.notna()]
            if done.empty: continue
            h = done.home.map(ix).to_numpy(); aw = done.away.map(ix).to_numpy(); nh = (~done.neutral).to_numpy(float)
            y = done.margin.clip(-cap, cap).to_numpy(float) - (prior[h] - prior[aw] + nh * prior[k])
            for hi, ai, n_, yi in zip(h, aw, nh, y):
                idx = [hi, ai, k]; x = np.array([1.0, -1.0, n_])
                A[np.ix_(idx, idx)] += np.outer(x, x); bv[idx] += x * yi
            R = prior + np.linalg.solve(A, bv)
        final[s] = {t: R[i] for t, i in ix.items()}
    return pd.concat(out)

def score(pred, P, seasons):
    Q = P[P.season.between(*seasons)].merge(pred, on="game_id", suffixes=("_old", ""))
    Q = Q[Q.v.notna()]
    fit = Q[Q.season.between(seasons[0], seasons[0] + 4)]
    cb = np.polyfit(fit.m_pred, fit.margin, 1)          # calibrate scale on the first 5 seasons
    val = Q[Q.season > seasons[0] + 4].copy(); val["cal"] = np.polyval(cb, val.m_pred)
    d = val.cal - val.v; res = np.sign(d) * (val.margin - val.v); m = np.abs(d) >= 3
    cov = (res[m] > 0).sum() / max(1, (res[m] != 0).sum())
    return dict(mae=np.abs(val.margin - val.cal).mean(), mae_close=np.abs(val.margin - val.v).mean(),
                slope=cb[0], n3=int(m.sum()), cover3=round(cov, 3),
                coef=np.linalg.lstsq(np.column_stack([np.ones(len(val)), val.v, val.cal - val.v]), val.margin, rcond=None)[0][2])

if __name__ == "__main__":
    g, tal, ret = prep()
    P = pd.read_parquet(ROOT / "output/research/cfb/preds.parquet")
    P = P[(P.home_div == "fbs") & (P.away_div == "fbs") & P.margin.notna()].copy(); P["v"] = -P.spread_close
    for cap in (28, 40, 100):
        for lam in (2.0, 4.0, 8.0):
            pr = run(g, tal, ret, cap=cap, lam=lam, last_seasons=(2014, 2021))
            r = score(pr, P, (2014, 2021))
            print(f"cap {cap:>3} lam {lam:>3}: MAE {r['mae']:.2f} (close {r['mae_close']:.2f}) slope {r['slope']:.2f} coef(model-close) {r['coef']:+.3f} | |cal-close|>=3: {r['n3']} cover {r['cover3']}", flush=True)
