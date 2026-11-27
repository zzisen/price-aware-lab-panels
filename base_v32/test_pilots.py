#!/usr/bin/env python3
"""Pilot integrity checks using synthetic data and frozen aggregate outputs."""
from pathlib import Path
import sys
import numpy as np,pandas as pd
import multitask_lasso_pilot as mt
import multitask_sparse_path_pilot as path
import upstream_train_only_pilot as up
from target_resolution import group_folds
HERE=Path(__file__).resolve().parent

rng=np.random.default_rng(90210);x=rng.normal(size=(90,12));y=rng.normal(size=(90,7));groups=np.arange(90)
held=group_folds(groups,5,20261027)[0];train=np.ones(90,bool);train[held]=False
r1=mt.fit_fold(x,y,groups,20261027,held,0)
r2=path.fold(x,y,groups,20261027,held,0)
x2=x.copy();y2=y.copy();x2[held]*=111.;y2[held]=rng.normal(30,7,size=y2[held].shape)
s1=mt.fit_fold(x2,y2,groups,20261027,held,0)
s2=path.fold(x2,y2,groups,20261027,held,0)
for a,b in [(r1,s1),(r2,s2)]:
 for z,q in zip(a,b):
  for k in ['chosen','inner_risk']:
   assert z[k]==q[k],(k,z[k],q[k])
  assert z['n_selected']<=3
print('PASS both multi-task algorithms: held-out perturbations do not alter training choices')

y[0,0]=np.nan;y[3,4]=np.nan
sums,base,n=up.loss_sums(x[train],y[train],x[held],y[held])
for t in range(7):
 obs=np.isfinite(y[train,t]);val=np.isfinite(y[held,t]);tr=x[train][obs];Y=y[train,t][obs];Xv=x[held][val]
 ym=Y.mean();ys=Y.std(ddof=1);ys=1 if ys==0 else ys
 for j in [0,4,10]:
  Z=np.column_stack([np.ones(len(tr)),tr[:,j]]);ZV=np.column_stack([np.ones(len(Xv)),Xv[:,j]])
  beta=np.linalg.lstsq(Z,(Y-ym)/ys,rcond=None)[0]
  expected=((ZV@beta-(y[held,t][val]-ym)/ys)**2).sum()
  assert abs(expected-sums[j,t])<1e-9,(j,t,expected,sums[j,t])
print('PASS source pilot losses: direct OLS, missing test labels excluded')
for folder in ['multitask_lasso','multitask_sparse_path']:
 d=pd.read_csv(HERE/'results'/folder/'folds.csv')
 assert len(d)==76 and (d.n_selected<=3).all()
 col='warnings' if 'warnings' in d else 'convergence_warnings'
 assert (d[col]==0).all()
 print('PASS',folder, '38 folds, both variants, no convergence warnings')
base=HERE if (HERE/'target_resolution.py').exists() else HERE.parent/'P2_V3.1_EXACT_2026-10-28'
old=pd.read_csv(base/'results/final_budget3/summary.csv')
new=pd.read_csv(HERE/'results/upstream_rebuild/summary.csv')
for mode in ['strict_intersection','inner_fallback']:
 for a,b in [('random_5fold','repeated_random'),('held_out_tissue','held_out_tissue')]:
  v=new[(new.cohort=='original_499')&(new['mode']==mode)&(new.evaluation==a)].iloc[0]
  z=old[(old.evaluation==b)&(old.selector=='inner_cv')].iloc[0]
  assert abs(v.C0-z.C0)<1e-10
  assert v.total_scored_observations==(17465 if a=='random_5fold' else 3493)
print('PASS source reconstruction preserves original 499 rows and mean-prediction risks')
