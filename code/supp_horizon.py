"""Section 5.6: horizon against training length, with series-clustered standard errors.
"""

import pandas as pd, numpy as np
from scipy import stats
import os as _os  # path fix: resolve inputs relative to this file, not a dead absolute path
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

P = _os.path.dirname(_os.path.abspath(__file__)) + _os.sep
B=['sarimax','bayesianpymc','prophet','neuralprophet','catboost']
val=pd.read_pickle(str(CACHE / "val.pkl")); pc=pd.read_pickle(str(CACHE / "pc.pkl")); D=pd.read_pickle(str(CACHE / "dates.pkl"))
nt=pd.read_pickle(str(CACHE / "nt.pkl")); ntr=pd.read_pickle(str(CACHE / "ntr.pkl"))
ser=np.array([i[0] for i in pc.index])

print('=== 1.6 HORIZON vs TRAINING LENGTH, series-clustered SEs ===')
ix=pc.index.intersection(ntr.index)
y=(pc.loc[ix,'ens_trimmed']-pc.loc[ix,'catboost']).values
x1=np.log(nt.loc[ix].values); x2=np.log(ntr.loc[ix].values)
X=np.column_stack([np.ones(len(y)),x1,x2])
b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b
XtXi=np.linalg.inv(X.T@X); g=np.array([i[0] for i in ix])
S=np.zeros((3,3))
for u in np.unique(g):
    m=g==u; Xu=X[m]; ru=r[m]; S+=np.outer(Xu.T@ru,Xu.T@ru)
nc=len(np.unique(g)); adj=nc/(nc-1)*(len(y)-1)/(len(y)-3)
V=XtXi@S@XtXi*adj; se=np.sqrt(np.diag(V))
for nm,c,s in zip(['intercept','log n_test','log n_train'],b,se):
    print('  %-12s %+.4f  clustered se %.4f  t %+.2f'%(nm,c,s,c/s))
print('  corr(log n_test, log n_train) = %.3f'%np.corrcoef(x1,x2)[0,1])
R2=1-np.corrcoef(x1,x2)[0,1]**2; print('  VIF on both length terms = %.1f'%(1/R2))
W=np.linalg.inv(V[1:,1:]); F=(b[1:]@W@b[1:])/2; print('  joint Wald F(2, %d clusters) = %.2f'%(nc,F))

print('\n=== 1.8 CALENDAR-RESTRICTED GLOBAL SELECTION ===')
d=D.reindex(pc.index); tstart=pd.to_datetime(d['start']); vend=tstart-pd.Timedelta(days=1)
vi=val.reindex(pc.index)
picks=[]; nopool=0
for i in pc.index:
    mask=(vend.reindex(pc.index)<tstart[i]) & vi.notna().all(axis=1)
    mask[i]=False
    if mask.sum()<10: picks.append(None); nopool+=1; continue
    picks.append(vi[mask][B].mean().idxmin())
pk=pd.Series(picks,index=pc.index)
print('  cells with an eligible prior pool (>=10 cells): %d of %d'%(pk.notna().sum(),len(pk)))
print('  pick distribution:',pk.value_counts().to_dict())
sub=pk.dropna().index
rr=np.array([pc.loc[i,pk[i]] for i in sub])
print('  restricted global selection mean RelMAE %.3f (unrestricted CatBoost on same cells %.3f)'%(rr.mean(),pc.loc[sub,'catboost'].mean()))
print('  trimmed on same cells %.3f ; wins %d/%d ; wilcoxon p=%.3g'%(pc.loc[sub,'ens_trimmed'].mean(),(pc.loc[sub,'ens_trimmed'].values<rr).sum(),len(sub),stats.wilcoxon(pc.loc[sub,'ens_trimmed'].values,rr).pvalue))
