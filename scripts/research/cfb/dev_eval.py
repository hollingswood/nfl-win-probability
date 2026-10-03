"""Dev-only evaluation (never reads holdout seasons): close tests on 2014-2021, opener tests on 2021-2023."""
import numpy as np, pandas as pd
P = pd.read_parquet("output/research/cfb/preds.parquet")
P = P[(P.home_div == "fbs") & (P.away_div == "fbs") & P.margin.notna()].copy()
P["v_close"] = -P.spread_close; P["v_open"] = -P.spread_open
P["gp"] = P[["games_played_h", "games_played_a"]].min(axis=1)
def block(D, name):
    D = D.dropna(subset=["v_close"])
    e = lambda x: np.abs(D.margin - x).mean()
    print(f"\n== {name}: n={len(D)}  MAE model {e(D.m_pred):.2f}  vegas close {e(D.v_close):.2f}  elo-ish n/a")
    X = np.column_stack([np.ones(len(D)), D.v_close, D.m_pred - D.v_close])
    b = np.linalg.lstsq(X, D.margin, rcond=None)[0]
    print(f"   margin ~ {b[1]:.3f}*close + {b[2]:.3f}*(model-close)")
    for k in (3, 5, 7, 10):
        d = D.m_pred - D.v_close; sel = D[np.abs(d) >= k]; dd = d[np.abs(d) >= k]
        res = np.sign(dd) * (sel.margin - sel.v_close)
        cov = (res > 0).sum(); los = (res < 0).sum()
        print(f"   |model-close|>={k:>2}: {len(sel):5d} bets, cover {cov/(cov+los):.3f}  ({cov}-{los})")
dev = P[P.season.between(2014, 2021)]
block(dev, "dev 2014-21, all weeks")
block(dev[dev.gp >= 4], "dev 2014-21, both teams >=4 games played")
block(dev[dev.gp < 4], "dev 2014-21, early season (<4 games)")
# PPA net: scale to points on 2014-2018, check 2019-2021
a = P[P.season.between(2014, 2018) & P.v_close.notna()]
b = np.linalg.lstsq(np.column_stack([np.ones(len(a)), a.v_close, a.m_pred - a.v_close, a.ppa_net]), a.margin, rcond=None)[0]
print("\ncombined fit 2014-18: margin ~ %.2f + %.3f close + %.3f (model-close) + %.2f ppa_net" % tuple(b))
v = P[P.season.between(2019, 2021) & P.v_close.notna()].copy()
v["c_pred"] = b[0] + b[1]*v.v_close + b[2]*(v.m_pred - v.v_close) + b[3]*v.ppa_net
for k in (1, 2, 3, 4):
    d = v.c_pred - v.v_close; sel = v[np.abs(d) >= k]; dd = d[np.abs(d) >= k]
    res = np.sign(dd)*(sel.margin - sel.v_close); cov=(res>0).sum(); los=(res<0).sum()
    print(f"   2019-21 combined |pred-close|>={k}: {len(sel)} bets cover {cov/max(1,cov+los):.3f} ({cov}-{los})")
# opener tests 2021-2023
O = P[P.season.between(2021, 2023) & P.v_open.notna() & P.v_close.notna()].copy()
O["move"] = O.v_close - O.v_open; O["gap"] = O.m_pred - O.v_open
print(f"\n== openers 2021-23: n={len(O)}  MAE open {np.abs(O.margin-O.v_open).mean():.2f} close {np.abs(O.margin-O.v_close).mean():.2f}  mean|move| {O.move.abs().mean():.2f}")
print("   corr(model-open, close-open) = %.3f" % np.corrcoef(O.gap, O.move)[0,1])
for k in (3, 5, 7, 10):
    sel = O[np.abs(O.gap) >= k]; s = np.sign(sel.gap)
    clv = (s*sel.move); res = s*(sel.margin - sel.v_open); cov=(res>0).sum(); los=(res<0).sum()
    res2 = s*(sel.margin - sel.v_close); c2=(res2>0).sum(); l2=(res2<0).sum()
    print(f"   |model-open|>={k:>2}: {len(sel):4d} bets, avg line move our way {clv.mean():+.2f} pts (moved our way {np.mean(clv>0):.0%}, against {np.mean(clv<0):.0%}); cover vs open {cov/(cov+los):.3f}, vs close {c2/(c2+l2):.3f}")
T = P[P.season.between(2014, 2021) & P.total_close.notna()]
print(f"\n== totals dev 2014-21: MAE model {np.abs(T.total - T.t_pred).mean():.2f} close {np.abs(T.total - T.total_close).mean():.2f}")
for k in (3, 5, 7, 10):
    d = T.t_pred - T.total_close; sel = T[np.abs(d) >= k]; s = np.sign(d[np.abs(d) >= k])
    res = s*(sel.total - sel.total_close); c=(res>0).sum(); l=(res<0).sum()
    print(f"   |model-close|>={k:>2}: {len(sel)} bets, hit {c/(c+l):.3f} ({c}-{l})")
TO = P[P.season.between(2021, 2023) & P.total_open.notna()]
TO = TO.assign(gap=TO.t_pred - TO.total_open, move=TO.total_close - TO.total_open)
print("   totals openers 2021-23 corr(model-open, move) = %.3f" % np.corrcoef(TO.gap, TO.move)[0,1])
for k in (3, 5, 7):
    sel = TO[np.abs(TO.gap) >= k]; s = np.sign(sel.gap)
    print(f"   |model-open|>={k}: {len(sel)} bets, avg move our way {(s*sel.move).mean():+.2f}, hit vs open {np.mean(s*(sel.total-sel.total_open)>0)/max(1e-9,np.mean(s*(sel.total-sel.total_open)!=0)):.3f}")
