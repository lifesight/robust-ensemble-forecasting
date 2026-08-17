"""Tercile breakdown behind Table 4.
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
val=pd.read_pickle(str(CACHE / "val.pkl")); pc=pd.read_pickle(str(CACHE / "pc.pkl")); D=pd.read_pickle(str(CACHE / "dates.pkl")); nt=pd.read_pickle(str(CACHE / "nt.pkl"))
man=pd.read_csv(str(MANIFEST)); man['sid']=man.series_id.str[:10]; omap=dict(zip(man.sid,man.org))
d=D.reindex(pc.index); ts=pd.to_datetime(d['start']); ve=ts-pd.Timedelta(days=1)
vi=val.reindex(pc.index); has=vi.notna().all(axis=1)
pick=[vi[(ve<ts[i])&has][B].mean().idxmin() for i in pc.index]
rel=np.array([pc.loc[i,p] for i,p in zip(pc.index,pick)])
df=pd.DataFrame({'ser':[i[0] for i in pc.index],'trim':pc.ens_trimmed.values,'dep':rel})
df['org']=[omap[s] for s in df.ser]
for lvl,key in [('cells',None),('series','ser'),('orgs','org')]:
    g=df if key is None else df.groupby(key)[['trim','dep']].mean().reset_index()
    w=stats.wilcoxon(g.trim,g.dep)
    print('  %-7s trimmed better in %d/%d  p=%.3g'%(lvl,(g.trim<g.dep).sum(),len(g),w.pvalue))
print('\nTERCILE CELL COUNTS')
q=pd.qcut(nt.reindex(pc.index),3,labels=['short','mid','long'])
print(q.value_counts().to_dict())
