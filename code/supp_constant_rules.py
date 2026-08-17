"""Table 3: constant rules against per-series selection.
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
val=pd.read_pickle(str(CACHE / "val.pkl")); pc=pd.read_pickle(str(CACHE / "pc.pkl"))
man=pd.read_csv(str(MANIFEST)); man['sid']=man.series_id.str[:10]
omap=dict(zip(man.sid,man.org))
idx=pc.index; ser=np.array([i[0] for i in idx]); org=np.array([omap[s] for s in ser])
def bootci(x,series,fn=np.mean,B_=4000,seed=0):
    rng=np.random.default_rng(seed); u,inv=np.unique(series,return_inverse=True)
    mem=[np.where(inv==i)[0] for i in range(len(u))]; out=np.empty(B_)
    for b in range(B_):
        p=rng.integers(0,len(u),len(u)); out[b]=fn(x[np.concatenate([mem[i] for i in p])])
    return np.percentile(out,2.5),np.percentile(out,97.5)
def bootdiff(x,y,series,fn,B_=4000,seed=0):
    rng=np.random.default_rng(seed); u,inv=np.unique(series,return_inverse=True)
    mem=[np.where(inv==i)[0] for i in range(len(u))]; out=np.empty(B_)
    pt=fn(x)-fn(y)
    for b in range(B_):
        p=rng.integers(0,len(u),len(u)); k=np.concatenate([mem[i] for i in p]); out[b]=fn(x[k])-fn(y[k])
    # basic (reverse-percentile) bootstrap, bias-corrected for the quantile estimator
    lo=2*pt-np.percentile(out,97.5); hi=2*pt-np.percentile(out,2.5)
    return pt,lo,hi

print('=== 1.3 CONSTANT-RULE BASELINES (clipped test RelMAE, N=293) ===')
rel={m:pc[m].mean() for m in B}
best=pc[B].idxmin(axis=1); hits={m:(best==m).mean() for m in B}
for m in sorted(rel,key=rel.get): print(f'  {m:14s} RelMAE {rel[m]:.3f}  hit {hits[m]*100:.1f}%')
print('  AVERAGE constant rule : RelMAE %.3f   hit %.1f%%'%(np.mean(list(rel.values())),np.mean(list(hits.values()))*100))
print('  per-series selection  : RelMAE %.3f   hit 21.8%%'%pc['sel_val'].mean())
print('  beats N of 5 constant rules:',sum(pc['sel_val'].mean()<v for v in rel.values()))

print('\n=== 1.7 GEOMETRIC MEAN with series-bootstrap CI, and paired tests on log RelMAE ===')
for m in ['ens_trimmed','catboost','ens_wtd','ens_mean','ens_screened','sel_val']:
    g=np.exp(np.log(pc[m].values).mean()); lo,hi=bootci(np.log(pc[m].values),ser,np.mean)
    print('  %-13s geo %.3f  95%% CI [%.3f, %.3f]'%(m,g,np.exp(lo),np.exp(hi)))
lr=np.log(pc['ens_trimmed'].values)
for m in ['sel_val','catboost']:
    d=lr-np.log(pc[m].values); w=stats.wilcoxon(d)
    print('  log-diff trimmed vs %-9s mean %.4f  p=%.3g'%(m,d.mean(),w.pvalue))

print('\n=== 1.9 SIGN TESTS (headline comparisons) ===')
for m in ['sel_val','catboost']:
    k=int((pc['ens_trimmed']<pc[m]).sum()); n=len(pc)
    print('  trimmed better in %d/%d vs %-9s  sign-test p=%.3g'%(k,n,m,stats.binomtest(k,n,0.5).pvalue))

print('\n=== 1.11 RANK-1 SHARE over the 11 ranked methods ===')
ORD=['sarimax','bayesianpymc','prophet','neuralprophet','catboost','theta','sel_val','ens_trimmed','ens_mean','ens_wtd','ens_screened']
R=np.array([stats.rankdata(r) for r in pc[ORD].values])
for j,m in enumerate(ORD):
    print('  %-13s rank-1 share %5.1f%%   share in ranks 9-11 %5.1f%%   mean rank %.2f'%(m,(R[:,j]<1.5).mean()*100,(R[:,j]>8.5).mean()*100,R[:,j].mean()))

print('\n=== 1.10 CORPUS FLOW ===')
m2=pd.read_csv(str(MANIFEST))
print('  candidate series:',len(m2),' qualifying:',int((m2.qualifies==True).sum()))
ex=m2[m2.qualifies!=True].excluded_reason.fillna('')
print('  excluded for length only :',int(ex.str.contains('short').sum()-ex.str.contains('short').mul(ex.str.contains('low_cov')).sum()))
print('  excluded for coverage only:',int((~ex.str.contains('short')&ex.str.contains('low_cov')).sum()))
print('  excluded for both        :',int((ex.str.contains('short')&ex.str.contains('low_cov')).sum()))
print('  total excluded           :',len(ex))
