#!/usr/bin/env python3
"""Joint l2,1 multi-task Lasso paths that actually obey the 3-feature cap.

For each training set, a fixed alpha grid is scanned. For each cap k=1,2,3,
select the least-penalized path solution with 1..k nonzero feature rows. This
uses the model's natural support (no post-hoc top-three truncation). k and
refit variant are assessed using inner folds; outer data never tune them.
This is a finite path, not an exact best-subset optimization.
"""
from pathlib import Path
import sys,time,json,warnings
import numpy as np,pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.exceptions import ConvergenceWarning
HERE=Path(__file__).resolve().parent;PREV=HERE if (HERE/'target_resolution.py').exists() else HERE.parent/'P2_V3.1_EXACT_2026-10-28'
sys.path.insert(0,str(PREV));from target_resolution import group_folds
from evaluate_gdsc import SEEDS,TARGETS
from multitask_lasso_pilot import scale,alpha_max
RATIOS=np.linspace(.999,.5,21)
METHODS=('sparse_group_lasso','sparse_group_lasso_post_ols')

def path_candidates(xt,yt,xv,yv):
    amax=alpha_max(xt,yt); model=MultiTaskLasso(fit_intercept=False,warm_start=True,max_iter=5000,tol=1e-5)
    choices={};warn_count=0
    for ratio in RATIOS:
        model.set_params(alpha=float(amax*ratio))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always',ConvergenceWarning);model.fit(xt,yt)
        warn_count+=sum(issubclass(z.category,ConvergenceWarning) for z in w)
        selected=np.flatnonzero(np.linalg.norm(model.coef_,axis=0)>1e-9)
        if 1<=len(selected)<=3:
            preds={'sparse_group_lasso':xv@model.coef_.T}
            beta=np.linalg.lstsq(xt[:,selected],yt,rcond=None)[0]
            preds['sparse_group_lasso_post_ols']=xv[:,selected]@beta
            result=(selected.copy(),float(ratio),{m:float(np.sum((preds[m]-yv)**2)) for m in METHODS})
            for k in range(len(selected),4):choices[k]=result
    for k in range(1,4):
        if k not in choices:choices[k]=(np.array([],int),1.0,{m:float(np.sum(yv**2)) for m in METHODS})
        assert len(choices[k][0])<=k
    return choices,warn_count

def fold(x,y,groups,seed,held,kfold):
    keep=np.ones(len(x),bool);keep[held]=False;tr=np.flatnonzero(keep)
    assert not set(groups[keep])&set(groups[held])
    inners=group_folds(groups[keep],5,seed+kfold+1)
    scores={m:np.zeros(3) for m in METHODS};nw=0
    for val in inners:
        m=np.ones(len(tr),bool);m[val]=False
        xt,yt,xv,yv=scale(x[tr[m]],y[tr[m]],x[tr[val]],y[tr[val]])
        paths,w=path_candidates(xt,yt,xv,yv);nw+=w
        for k,(feat,ratio,risks) in paths.items():
            for method in METHODS:scores[method][k-1]+=risks[method]
    xt,yt,xv,yv=scale(x[tr],y[tr],x[held],y[held]);paths,w=path_candidates(xt,yt,xv,yv);nw+=w
    out=[]
    for method in METHODS:
        k=int(np.argmin(scores[method]))+1;feat,ratio,risks=paths[k]
        out.append(dict(method=method,n_cell_lines=len(held),risk=risks[method]/(len(held)*y.shape[1]),
                        chosen_cap=k,n_selected=len(feat),alpha_fraction=ratio,
                        chosen='|'.join(map(str,feat)),inner_risk=float(scores[method][k-1]/(len(tr)*7)),warnings=nw))
    return out

def main():
    d=pd.read_csv(PREV/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv');assert d.shape==(499,857)
    x=d.iloc[:,20:].to_numpy(float);y=d.iloc[:,:7].to_numpy(float);t=np.argmax(d.iloc[:,7:20].to_numpy(float),axis=1)
    scenarios=[('repeated_random',SEEDS,np.arange(len(x)),5),('held_out_tissue',[SEEDS[0]],t,13)]
    out=HERE/'results/multitask_sparse_path';out.mkdir(parents=True,exist_ok=True)
    rows=[];t0=time.perf_counter()
    for ev,seeds,groups,nfold in scenarios:
        for seed in seeds:
            for k,held in enumerate(group_folds(groups,nfold,seed)):
                a=time.perf_counter();current=fold(x,y,groups,seed,held,k)
                for r in current:r.update(evaluation=ev,seed=seed,fold=k,seconds=time.perf_counter()-a)
                rows+=current;pd.DataFrame(rows).to_csv(out/'folds.csv',index=False,float_format='%.17g')
                print(ev,seed,k+1,'/',nfold,'seconds',round(time.perf_counter()-a,2),flush=True)
    df=pd.DataFrame(rows);summ=[]
    for (ev,m),g in df.groupby(['evaluation','method']):
        summ.append(dict(evaluation=ev,method=m,n_folds=len(g),risk=float(np.average(g.risk,weights=g.n_cell_lines)),
                         n_selected_min=int(g.n_selected.min()),n_selected_mean=float(g.n_selected.mean()),
                         n_selected_max=int(g.n_selected.max()),warning_folds=int((g.warnings>0).sum())))
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False,float_format='%.17g')
    (out/'metadata.json').write_text(json.dumps({'seconds':time.perf_counter()-t0,'alphas':RATIOS.tolist(),'selection':'natural group-lasso support on finite alpha grid, <=3; no truncation','caution':'same preprocessed example; post-hoc exploratory'},indent=2),encoding='utf-8')
    print(pd.DataFrame(summ).to_string(index=False),flush=True)
    print('TOTAL_SECONDS',round(time.perf_counter()-t0,2),flush=True)
if __name__=='__main__':main()
