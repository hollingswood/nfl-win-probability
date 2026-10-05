"""Post-hoc check (2026-10-04, not pre-registered): college moneyline dogs at +300 or longer where an Arizona book
beats the Pinnacle Shin fair price by 2%+, 1-168 h before kickoff (first such quote per game/side), 2021-25.
Run from scripts/research/cfb. Result: output/research/cfb/longshots.md."""
import sys, glob, gzip, json, math
sys.path.insert(0, '/home/claude/nfl/src'); sys.path.insert(0, '.')
import numpy as np, pandas as pd
import price_screen as PS, opener_screen as OS
from nflpred.devig import shin_from_implied
AZ = ["draftkings", "fanduel", "espnbet", "betmgm", "williamhill_us", "betrivers", "fanatics", "hardrockbet", "ballybet"]
res = {}
for f in glob.glob('/home/claude/nfl/data/cfb/raw/games_20*_*.json.gz'):
    for g in json.load(gzip.open(f)):
        if g.get('completed') and g.get('homePoints') is not None:
            res[g['id']] = (g['homePoints'], g['awayPoints'])
O = PS.load('cfb'); E = OS.event_map(O); Pn = PS.load('cfb_pin')
imp = lambda a: np.where(a > 0, 100 / (a + 100), -a / (-a + 100))
Pn = Pn.dropna(subset=['ml_home', 'ml_away']).copy()
ih, ia = imp(Pn.ml_home.values), imp(Pn.ml_away.values)
Pn['fh'] = [shin_from_implied(a, b) for a, b in zip(ih, ia)]
Pn['h'] = (Pn.ko - Pn.t).dt.total_seconds() / 3600
pin_snap = Pn.set_index(['event_id', 'requested_ts']).fh.to_dict()
close = Pn[Pn.h.between(0.2, 3)].sort_values('t').groupby('event_id').fh.last().to_dict()
B = O[O.book.isin(AZ)].dropna(subset=['ml_home', 'ml_away']).copy()
B['h'] = (B.ko - B.t).dt.total_seconds() / 3600
B = B[B.h.between(1, 168)]
rows = []
for r in B.itertuples():
    fh = pin_snap.get((r.event_id, r.requested_ts))
    if fh is None: continue
    for side, px, p in (('home', r.ml_home, fh), ('away', r.ml_away, 1 - fh)):
        if px < 300: continue
        d = 1 + px / 100
        rows.append((r.event_id, r.t, r.season, side, px, p, p * d - 1, r.book))
D = pd.DataFrame(rows, columns=['event_id', 't', 'season', 'side', 'price', 'p', 'ev', 'book'])
D = D[D.ev >= 0.02].sort_values('t').groupby(['event_id', 'side']).head(1)
gid = E.game_id if hasattr(E, 'game_id') else None
out = []
for r in D.itertuples():
    g = E.game_id.get(r.event_id) if r.event_id in E.index else None
    sc = res.get(g)
    if not sc: continue
    win = (sc[0] > sc[1]) if r.side == 'home' else (sc[1] > sc[0])
    c = close.get(r.event_id)
    cp = None if c is None else (c if r.side == 'home' else 1 - c)
    d = 1 + r.price / 100
    out.append({**r._asdict(), 'win': win, 'pnl': d - 1 if win else -1, 'clv': np.nan if cp is None else cp * d - 1})
R = pd.DataFrame(out)
R['bucket'] = pd.cut(R.price, [299, 500, 1000, 2000, 100000], labels=['+300-500', '+500-1000', '+1000-2000', '+2000+'])
f = lambda x: x.std() / math.sqrt(len(x))
print(R.groupby('bucket', observed=True).agg(n=('pnl', 'size'), ev=('ev', 'mean'), p_fair=('p', 'mean'), win=('win', 'mean'),
      roi=('pnl', 'mean'), roi_se=('pnl', f), clv=('clv', 'mean'), clv_se=('clv', f)).round(3).to_string())
# calibration of Pinnacle Shin fair for big dogs (24-72h, last snapshot)
C = Pn[Pn.h.between(24, 72)].sort_values('t').groupby('event_id').last()
cal = []
for eid, r in C.iterrows():
    g = E.game_id.get(eid) if eid in E.index else None
    sc = res.get(g)
    if not sc or sc[0] == sc[1]: continue
    dog_home = r.fh < 0.5
    p = r.fh if dog_home else 1 - r.fh
    cal.append((p, (sc[0] > sc[1]) if dog_home else (sc[1] > sc[0])))
K = pd.DataFrame(cal, columns=['p', 'win'])
K['b'] = pd.cut(K.p, [0, .03, .06, .10, .15, .25])
print(K.groupby('b', observed=True).agg(n=('p', 'size'), fair=('p', 'mean'), actual=('win', 'mean')).round(3).to_string())
