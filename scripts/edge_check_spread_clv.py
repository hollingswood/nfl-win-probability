import sys,os; sys.path.insert(0,'/home/claude/nfl/scripts'); sys.path.insert(0,'/home/claude/nfl/src')
import warnings; warnings.filterwarnings('ignore')
os.environ['EDGE_HOLDOUT']='I_HAVE_FROZEN_CANDIDATES'
import pandas as pd, numpy as np
import edge_lab as E, replay_early_lines as R
from nflpred import margins as K, spread_bets as SB
from nflpred.weather import _kickoff_utc
r=SB.load_rules(); w=r['_weights']; sig=r['margin']['sigma']
o=R.load_odds(); wf=R.walk_forward(); wf['gameday']=pd.to_datetime(wf.gameday)
wf['kick']=pd.to_datetime([_kickoff_utc(x.gameday,x.gametime) for x in wf.itertuples()],utc=True)
o=R.match_games(o,wf).merge(wf[['game_id','kick']],on='game_id'); o=o[o.requested_ts<o.kick]
last=o[o.requested_ts==o.groupby('game_id').requested_ts.transform('max')]
last=last[last.sp_home_price.notna()&last.sp_away_price.notna()&last.sp_home_point.notna()]
def nvp(a,b):
    ia=np.where(a<0,-a/(-a+100),100/(a+100)); ib=np.where(b<0,-b/(-b+100),100/(b+100)); return ia/(ia+ib)
last=last.assign(p=nvp(last.sp_home_price,last.sp_away_price))
grid=np.arange(-30,30.01,0.05)
def implied_mu(point,p):
    # home line = point; find mu with P(home cover)/(1-P(push)) = p
    hc,pu,ac=K.cover_probs(grid,sig,np.full_like(grid,point),w)
    q=hc/np.maximum(hc+ac,1e-9)
    return float(np.interp(p,q,grid))  # q increasing in mu
def fair_mu(d):
    return np.median([implied_mu(pt,p) for pt,p in zip(d.sp_home_point,d.p)])
for label,books in (('sharp',E.SHARP),('all',None)):
    L=last if books is None else last[last.book.isin(books)]
    mu_close=L.groupby('game_id').apply(fair_mu).rename('mu_close_'+label)
    wf=wf.merge(mu_close,left_on='game_id',right_index=True,how='left')
print('mu_close_all vs spread_line: mean diff',(wf.mu_close_all-wf.spread_line).mean().round(3),'MAE',(wf.mu_close_all-wf.spread_line).abs().mean().round(3))
for hold in (False,True):
    t=E.load(hold); C=E.candidates(t)
    for cid in ('C1_spread_sharp_dog','C3_spread_sharp_and_model','C5_spread_model_dog','C7_spread_sharp_gated'):
        b,_=C[cid]; b=b.merge(wf[['game_id','mu_close_sharp','mu_close_all']],on='game_id',how='left')
        out=[]
        for col in ('mu_close_sharp','mu_close_all'):
            v=[SB.side_ev(mc,pt,int(pr),sd,r)[0] if not pd.isna(mc) else np.nan for mc,pt,pr,sd in zip(b[col],b.point,b.sp_price,b.side)]
            v=np.array(v); v=v[~np.isnan(v)]
            tt=v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))
            out.append(f"{col[9:]}: clv={v.mean():+.4f} t={tt:+.2f} beat={(v>0).mean():.2f}")
        print(f"{'HOLD' if hold else 'DEV '} {cid:28s} n={len(b)} old={b.sp_clv.mean():+.4f} | "+" | ".join(out))
