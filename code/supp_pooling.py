"""Table 7: how quickly portfolio selection becomes reliable as evidence accumulates.
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
d=D.reindex(pc.index); tstart=pd.to_datetime(d['start']); vend=tstart-pd.Timedelta(days=1)
vi=val.reindex(pc.index); has=vi.notna().all(axis=1)
res=[]
for i in pc.index:
    mask=(vend<tstart[i]) & has          # a deployer at this origin: every validation window already finished
    n=int(mask.sum())
    pick=vi[mask][B].mean().idxmin() if n>=1 else None
    res.append((i,n,pick))
R=pd.DataFrame(res,columns=['i','pool','pick']).set_index('i')
R['rel']=[pc.loc[i,p] if p else np.nan for i,p in zip(R.index,R['pick'])]
print('pool size quartiles:',np.percentile(R['pool'],[0,25,50,75,100]).round(0))
print('pick distribution:',R['pick'].value_counts().to_dict())
ok=R['rel'].notna()
print('\nALL %d cells: restricted-global %.3f | unrestricted CatBoost %.3f | trimmed %.3f | per-series %.3f'%(
    ok.sum(),R.rel[ok].mean(),pc.catboost[ok].mean(),pc.ens_trimmed[ok].mean(),pc.sel_val[ok].mean()))
w=stats.wilcoxon(pc.ens_trimmed[ok].values,R.rel[ok].values)
print('  trimmed vs restricted-global: wins %d/%d p=%.3g'%((pc.ens_trimmed[ok]<R.rel[ok]).sum(),ok.sum(),w.pvalue))
for lo,hi in [(1,50),(50,150),(150,1000)]:
    m=ok&(R['pool']>=lo)&(R['pool']<hi)
    if m.sum()<5: continue
    print('  pool %3d-%4d (n=%3d): restricted-global %.3f  trimmed %.3f  picks %s'%(
        lo,hi,m.sum(),R.rel[m].mean(),pc.ens_trimmed[m].mean(),R['pick'][m].value_counts().to_dict()))
big=ok&(R['pool']>=150)
w2=stats.wilcoxon(pc.ens_trimmed[big].values,R.rel[big].values)
print('\n  mature-pool cells only (>=150 prior cells, n=%d): trimmed %.3f vs restricted-global %.3f, wins %d/%d p=%.3g'%(
    big.sum(),pc.ens_trimmed[big].mean(),R.rel[big].mean(),(pc.ens_trimmed[big]<R.rel[big]).sum(),big.sum(),w2.pvalue))
