"""Section 5.7: member error diversity and the distribution of trimmed slots.
"""

import pandas as pd, numpy as np, glob, os, re
import os as _os  # path fix: resolve inputs relative to this file, not a dead absolute path
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paths import CODE, ROOT, DATA, FORECASTS, RESULTS, FIGURES, CORPUS, MANIFEST, \
    VALIDATION_SUMMARY, SUMMARY_CLEAN, CACHE

P = _os.path.dirname(_os.path.abspath(__file__)) + _os.sep
B=['sarimax','bayesianpymc','prophet','neuralprophet','catboost']
val=pd.read_pickle(str(CACHE / "val.pkl")); pc=pd.read_pickle(str(CACHE / "pc.pkl"))
slots={m:0 for m in B}; chg=0; pairs=0; mse=[]; Dl=[]; nobs=0; run=[]
for f in sorted(glob.glob(str(FORECASTS / "*.csv"))):
    s,o=re.match(r'(.+)_origin(\d)\.csv',os.path.basename(f)).groups(); key=(s,int(o))
    if key not in pc.index: continue
    d=pd.read_csv(f); c=[m+'__yhat' for m in B]
    if any(x not in d.columns for x in c) or d[c].isna().any().any(): continue
    if key in val.index and not val.loc[key,B].isna().any():
        rep='prophet' if val.loc[key,'prophet']<=val.loc[key,'neuralprophet'] else 'neuralprophet'
    else: rep='prophet'
    M=['sarimax','bayesianpymc','catboost',rep]
    y=d['actual'].values; nm=np.mean(np.abs(y-d['naive'].values)); nobs+=len(d)
    X=np.maximum(d[[m+'__yhat' for m in M]].values,0)
    hi=np.argmax(X,axis=1); lo=np.argmin(X,axis=1)
    for i in hi: slots[M[i]]+=1
    for i in lo: slots[M[i]]+=1
    chg+=(hi[1:]!=hi[:-1]).sum(); pairs+=len(hi)-1
    # longest run of the same trimmed-high member
    r=1; mx=1
    for k in range(1,len(hi)):
        r=r+1 if hi[k]==hi[k-1] else 1; mx=max(mx,r)
    run.append(mx)
    e=(X-y[:,None])/nm
    mse.append((e**2).mean(axis=0).mean()); av=e.mean(axis=1); Dl.append(((e-av[:,None])**2).mean())
tot=sum(slots.values())
print('=== 5.7 REGENERATED (4 de-duplicated members, %d cells, %d test observations) ==='%(len(mse),nobs))
print('  share of trimmed slots (25%% under uniformity):')
for m in sorted(slots,key=slots.get,reverse=True): print('    %-14s %5.1f%%'%(m,slots[m]/tot*100))
print('  additive family combined: %.1f%%'%((slots['prophet']+slots['neuralprophet'])/tot*100))
print('  trimmed-high member CHANGES day-to-day in %.1f%% of consecutive pairs (=> unchanged %.1f%%)'%(chg/pairs*100,100-chg/pairs*100))
print('  median longest unbroken run of the same trimmed-high member: %.0f days (mean %.0f)'%(np.median(run),np.mean(run)))
mm=np.mean(mse); DD=np.mean(Dl)
print('  average member MSE %.2f   diversity term D %.2f   removes %.0f%%   combination MSE %.2f'%(mm,DD,DD/mm*100,mm-DD))
