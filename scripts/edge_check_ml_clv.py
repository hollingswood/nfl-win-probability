import sys,os; sys.path.insert(0,'/home/claude/nfl/scripts'); sys.path.insert(0,'/home/claude/nfl/src')
import warnings; warnings.filterwarnings('ignore')
os.environ['EDGE_HOLDOUT']='I_HAVE_FROZEN_CANDIDATES'
import pandas as pd, numpy as np
import edge_lab as E, replay_early_lines as R
from nflpred.weather import _kickoff_utc
o=R.load_odds(); wf=R.walk_forward(); wf['gameday']=pd.to_datetime(wf.gameday)
wf['kick']=pd.to_datetime([_kickoff_utc(x.gameday,x.gametime) for x in wf.itertuples()],utc=True)
o=R.match_games(o,wf).merge(wf[['game_id','kick']],on='game_id'); o=o[o.requested_ts<o.kick]
last=o[o.requested_ts==o.groupby('game_id').requested_ts.transform('max')]
last=last[last.ml_home.notna()&last.ml_away.notna()]
ih=np.where(last.ml_home<0,-last.ml_home/(-last.ml_home+100),100/(last.ml_home+100)); ia=np.where(last.ml_away<0,-last.ml_away/(-last.ml_away+100),100/(last.ml_away+100))
last=last.assign(p=ih/(ih+ia))
cs=last[last.book.isin(E.SHARP)].groupby('game_id').p.median().rename('pc_sharp'); ca=last.groupby('game_id').p.median().rename('pc_all')
for hold in (False,True):
    t=E.load(hold); C=E.candidates(t)
    for cid in ('C2_ml_sharp_dog','C4_ml_sharp_and_model','C6_ml_longshot_model'):
        b,_=C[cid]; b=b.join(cs,on='game_id').join(ca,on='game_id')
        s=[]
        for col in ('pc_sharp','pc_all'):
            p=np.where(b.side=='home',b[col],1-b[col]); v=b.ml_dec*p-1; v=v[~np.isnan(v)]
            s.append(f"{col}: clv={v.mean():+.4f} t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):+.2f} beat={(v>0).mean():.2f}")
        print(f"{'HOLD' if hold else 'DEV '} {cid:24s} n={len(b)} nflverse={b.ml_clv.mean():+.4f} | "+" | ".join(s))
