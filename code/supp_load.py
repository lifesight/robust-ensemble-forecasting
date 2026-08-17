"""Shared data load for the supplementary analyses.

Caches the frames the other supp_*.py scripts read. Must run before them.
"""

import pandas as pd, numpy as np, glob, os
import os as _os  # path fix: resolve inputs relative to this file, not a dead absolute path
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

P = _os.path.dirname(_os.path.abspath(__file__)) + _os.sep
BASE=['sarimax','bayesianpymc','prophet','neuralprophet','catboost']
S=pd.read_csv(str(SUMMARY_CLEAN))
S=S[S.is_ensemble==0]
val=S.pivot_table(index=['series','origin'],columns='model',values='val_mape')
tst=S.pivot_table(index=['series','origin'],columns='model',values='skill_mase')   # RelMAE
tmape=S.pivot_table(index=['series','origin'],columns='model',values='mape')
nt=S.pivot_table(index=['series','origin'],columns='model',values='n_test').mean(axis=1)
ntr=S.pivot_table(index=['series','origin'],columns='model',values='n_train').mean(axis=1)
ok=val[BASE].notna().all(axis=1)&tst[BASE].notna().all(axis=1)
val,tst,tmape,nt,ntr=[x.loc[ok] if hasattr(x,'loc') else x for x in (val,tst,tmape,nt,ntr)]
val=val[BASE]; tst=tst[BASE]; tmape=tmape[BASE]
print('cells',len(val))
# cross-check vs percell_test_metrics
pc=pd.read_csv(str(RESULTS / "percell_test_metrics.csv")).pivot_table(index=['series','origin'],columns='method',values='skill')
common=tst.index.intersection(pc.index)
print('catboost match vs percell_test_metrics:', np.allclose(tst.loc[common,'catboost'],pc.loc[common,'catboost'],atol=1e-9))
val.to_pickle(str(CACHE / "val.pkl")); tst.to_pickle(str(CACHE / "tst.pkl")); tmape.to_pickle(str(CACHE / "tmape.pkl"))
nt.to_pickle(str(CACHE / "nt.pkl")); ntr.to_pickle(str(CACHE / "ntr.pkl")); pc.to_pickle(str(CACHE / "pc.pkl"))
