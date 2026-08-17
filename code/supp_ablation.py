"""Table 5, the family de-duplication ablation, and the tail differences of Section 5.5.
"""

import pandas as pd, numpy as np, glob, os, re
from scipy import stats
import os as _os  # path fix: resolve inputs relative to this file, not a dead absolute path
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

P = _os.path.dirname(_os.path.abspath(__file__)) + _os.sep
B=['sarimax','bayesianpymc','prophet','neuralprophet','catboost']
val=pd.read_pickle(str(CACHE / "val.pkl")); pc=pd.read_pickle(str(CACHE / "pc.pkl"))
man=pd.read_csv(str(MANIFEST)); man['sid']=man.series_id.str[:10]
omap=dict(zip(man.sid,man.org))
rows={}
for f in sorted(glob.glob(str(FORECASTS / "*.csv"))):
    s,o=re.match(r'(.+)_origin(\d)\.csv',os.path.basename(f)).groups()
    d=pd.read_csv(f); c=[m+'__yhat' for m in B]
    if any(x not in d.columns for x in c) or d[c].isna().any().any(): continue
    y=d['actual'].values; nm=np.mean(np.abs(y-d['naive'].values))
    X=np.maximum(d[c].values,0)
    t5=np.sort(X,axis=1)[:,1:4].mean(axis=1)   # trim hi+lo of 5 -> mean of middle 3
    m5=X.mean(axis=1)
    rows[(s,int(o))]=dict(t5=np.mean(np.abs(y-t5))/nm, m5=np.mean(np.abs(y-m5))/nm,
                          start=d['ds'].min(), end=d['ds'].max(), n=len(d))
A=pd.DataFrame(rows).T.astype({'t5':float,'m5':float,'n':int})
A.index=pd.MultiIndex.from_tuples(A.index,names=['series','origin'])
A=A.reindex(pc.index)
print('reconstructed trimmed-all-5 mean %.3f (paper 0.747)   mean-all-5 %.3f (paper 0.764)'%(A.t5.mean(),A.m5.mean()))
ser=np.array([i[0] for i in pc.index])
def bootdiff(x,y,series,fn,B_=4000,seed=0):
    rng=np.random.default_rng(seed); u,inv=np.unique(series,return_inverse=True)
    mem=[np.where(inv==i)[0] for i in range(len(u))]; out=np.empty(B_); pt=fn(x)-fn(y)
    for b in range(B_):
        p=rng.integers(0,len(u),len(u)); k=np.concatenate([mem[i] for i in p]); out[b]=fn(x[k])-fn(y[k])
    return pt, 2*pt-np.percentile(out,97.5), 2*pt-np.percentile(out,2.5)
Q={'P90':lambda z:np.percentile(z,90),'P95':lambda z:np.percentile(z,95),'P99':lambda z:np.percentile(z,99),'Max':np.max}
print('\n=== 1.4 TAIL DIFFERENCES, point estimate + basic (reverse-percentile) series-bootstrap CI ===')
for lab,x,y in [('trimmed - weighted',pc.ens_trimmed.values,pc.ens_wtd.values),
                ('trimmed dedup - trimmed all-5',pc.ens_trimmed.values,A.t5.values),
                ('mean dedup - mean all-5',pc.ens_mean.values,A.m5.values),
                ('trimmed - catboost',pc.ens_trimmed.values,pc.catboost.values),
                ('trimmed - sel_val',pc.ens_trimmed.values,pc.sel_val.values)]:
    print(' '+lab)
    for q,fn in Q.items():
        pt,lo,hi=bootdiff(x,y,ser,fn)
        print('   %-4s %7.3f vs %7.3f  diff %+7.3f  [%+.3f, %+.3f]%s'%(q,fn(x),fn(y),pt,lo,hi,'  *' if lo*hi>0 else ''))
print('\n=== ablation accuracy (with log-mean and Wilcoxon) ===')
for lab,y in [('trimmed all-5',A.t5.values),('mean all-5',A.m5.values)]:
    x=pc.ens_trimmed.values if 'trimmed' in lab else pc.ens_mean.values
    print('  %-14s ref %.3f vs %.3f  wins %d/293  p=%.3g'%(lab,x.mean(),y.mean(),(x<y).sum(),stats.wilcoxon(x,y).pvalue))
A[['start','end','n']].to_pickle(str(CACHE / "dates.pkl"))
