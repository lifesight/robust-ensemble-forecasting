"""Cell counts and the constant-rule setup behind Section 5.3.
"""

import pandas as pd, numpy as np, glob, os
from scipy import stats
import os as _os  # path fix: resolve inputs relative to this file, not a dead absolute path
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

P = _os.path.dirname(_os.path.abspath(__file__)) + _os.sep
B=['sarimax','bayesianpymc','prophet','neuralprophet','catboost']
val=pd.read_pickle(str(CACHE / "val.pkl")); tst=pd.read_pickle(str(CACHE / "tst.pkl")); tmape=pd.read_pickle(str(CACHE / "tmape.pkl"))
pc=pd.read_pickle(str(CACHE / "pc.pkl")); nt=pd.read_pickle(str(CACHE / "nt.pkl")); ntr=pd.read_pickle(str(CACHE / "ntr.pkl"))
man=pd.read_csv(str(MANIFEST)); man['sid']=man.series_id.str[:10]
omap=dict(zip(man.sid,man.org)); vmap=dict(zip(man.sid,man.vertical))
idx=val.index; ser=np.array([i[0] for i in idx]); org=np.array([omap[s] for s in ser])
print('=== N cells with complete validation:',len(idx),'(main leaderboard N=293; one cell lacks a Prophet validation row in summary1-99_clean.csv)')

# --- 1.3 constant rules -------------------------------------------------
print('\n=== 1.3 CONSTANT-RULE BASELINES ===')
best=tst[B].idxmin(axis=1)
hits={m:(best==m).mean() for m in B}
rel={m:tst[m].mean() for m in B}
for m in sorted(rel,key=rel.get): print(f'  {m:14s} RelMAE {rel[m]:.3f}  hit-rate {hits[m]*100:.1f}%')
print(f'  AVERAGE constant rule: RelMAE {np.mean(list(rel.values())):.3f}  hit-rate {np.mean(list(hits.values()))*100:.1f}%')
print(f'  per-series selection : RelMAE {pc.loc[idx,"sel_val"].mean():.3f}')
print(f'  beats how many of 5 constant rules: {sum(pc.loc[idx,"sel_val"].mean()<v for v in rel.values())}')

# --- de-duplication + rules ---------------------------------------------
addrep=np.where(val['prophet']<=val['neuralprophet'],'prophet','neuralprophet')
print('\n=== de-dup representative: prophet %.1f%% neuralprophet %.1f%%'%((addrep=='prophet').mean()*100,(addrep=='neuralprophet').mean()*100))
M4=[[ 'sarimax','bayesianpymc','catboost',a] for a in addrep]

# --- 1.2 four-member per-series selection -------------------------------
sel4=[];sel5=[]
for k,(i,mem) in enumerate(zip(idx,M4)):
    sel4.append(tst.loc[i,val.loc[i,mem].idxmin()])
    sel5.append(tst.loc[i,val.loc[i,B].idxmin()])
sel4=np.array(sel4); sel5=np.array(sel5)
print('\n=== 1.2 SELECTION BASELINES ===')
print('  per-series sel, 5 candidates: mean %.3f geo %.3f'%(sel5.mean(),np.exp(np.log(sel5).mean())))
print('  per-series sel, 4 de-dup    : mean %.3f geo %.3f'%(sel4.mean(),np.exp(np.log(sel4).mean())))
ref=pc.loc[idx,'ens_trimmed'].values
w=stats.wilcoxon(ref,sel4); print('  trimmed vs 4-member selection: wins %d/%d  p=%.3g  mean diff %.4f'%((ref<sel4).sum(),len(ref),w.pvalue,(ref-sel4).mean()))

# --- 1.5 rho with clustered CI + 1.1-substitute subgroup ----------------
print('\n=== 1.5 WITHIN-CELL RANK CORRELATION, series-clustered CI ===')
rho=np.array([stats.spearmanr(val.loc[i,B],tmape.loc[i,B]).statistic for i in idx])
rhoX=np.array([stats.spearmanr(val.loc[i,B],tst.loc[i,B]).statistic for i in idx])
hit=np.array([val.loc[i,B].idxmin()==tmape.loc[i,B].idxmin() for i in idx])
hitX=np.array([val.loc[i,B].idxmin()==tst.loc[i,B].idxmin() for i in idx])
def bootci(x,series,fn=np.mean,B_=4000,seed=0):
    rng=np.random.default_rng(seed); u,inv=np.unique(series,return_inverse=True)
    mem=[np.where(inv==i)[0] for i in range(len(u))]; out=np.empty(B_)
    for b in range(B_):
        p=rng.integers(0,len(u),len(u)); out[b]=fn(x[np.concatenate([mem[i] for i in p])])
    return np.percentile(out,2.5),np.percentile(out,97.5)
lo,hi=bootci(rho,ser)
print('  matched  rho mean %.3f  95%% CI [%.3f, %.3f]  median %.2f  frac>0 %.2f  hit %.1f%%'%(rho.mean(),lo,hi,np.median(rho),(rho>0).mean(),hit.mean()*100))
lo2,hi2=bootci(rhoX,ser)
print('  mismatch rho mean %.3f  95%% CI [%.3f, %.3f]  hit %.1f%%'%(rhoX.mean(),lo2,hi2,hitX.mean()*100))
# subgroup by daily count
D=pd.read_csv(str(CORPUS),low_memory=False)
cnt=D.groupby('SERIES_ID').ALL_PURCHASES.mean(); cnt.index=cnt.index.str[:10]
cser=np.array([cnt.get(s,np.nan) for s in ser]); med=np.nanmedian(cnt)
lowm=cser<med
print('\n=== 1.1-SUBSTITUTE: is the weak signal a MAPE-at-low-counts artefact? (median split at %.1f/day)'%med)
for nm,msk in [('low-count half',lowm),('high-count half',~lowm)]:
    l,h=bootci(rho[msk],ser[msk]); print('  %-16s rho %.3f [%.3f,%.3f]  hit %.1f%%  n=%d'%(nm,rho[msk].mean(),l,h,hit[msk].mean()*100,msk.sum()))
