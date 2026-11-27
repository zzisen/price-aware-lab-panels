#!/usr/bin/env python3
"""Same-budget multitask group-lasso pilot; source data remain local.

The model uses MultiTaskLasso's l2,1 penalty. A full-feature training fit ranks
feature rows by l2 coefficient norm, at most three are retained, and then a
penalized or an OLS model is refit on those features. Alpha fractions are
selected solely in the same grouped inner folds as the original analysis.
This top-three projection is an explicit pilot comparator, not exact group
best-subset lasso. Candidate preprocessors are trained within each fold.
"""
from pathlib import Path
import argparse,csv,json,sys,time,warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.exceptions import ConvergenceWarning

HERE=Path(__file__).resolve().parent
PREV=HERE if (HERE/'target_resolution.py').exists() else HERE.parent/'P2_V3.1_EXACT_2026-10-28'
sys.path.insert(0,str(PREV))
from evaluate_gdsc import SEEDS,TARGETS
from target_resolution import group_folds

RATIOS=(0.95,0.7,0.5,0.35,0.2,0.1,0.05,0.02)
METHODS=('mtlasso_penalized','mtlasso_post_ols')


def scale(x,y,xt,yt):
    xm=x.mean(0); xs=x.std(0,ddof=1); xs=np.where(xs==0,1.,xs)
    ym=y.mean(0); ys=y.std(0,ddof=1); ys=np.where(ys==0,1.,ys)
    return np.asfortranarray((x-xm)/xs), (y-ym)/ys, np.asfortranarray((xt-xm)/xs), (yt-ym)/ys


def alpha_max(x,y):return np.max(np.linalg.norm(x.T@y/len(x),axis=1))


def candidates(xt,yt,xv,yv,ratios=RATIOS):
    amax=alpha_max(xt,yt)
    model=MultiTaskLasso(fit_intercept=False,warm_start=True,max_iter=5000,tol=1e-5,selection='cyclic')
    result=[]; warning_count=0
    for fraction in ratios:
        alpha=float(amax*fraction)
        model.set_params(alpha=alpha)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(xt,yt)
        warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in records)
        norms=np.linalg.norm(model.coef_,axis=0)
        active=np.flatnonzero(norms>1e-10)
        ordered=active[np.argsort(-norms[active],kind='stable')][:3]
        pred={}
        if len(ordered)==0:
            pred={m:np.zeros_like(yv) for m in METHODS}
        else:
            q=MultiTaskLasso(alpha=alpha,fit_intercept=False,max_iter=5000,tol=1e-5,selection='cyclic')
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter('always',ConvergenceWarning)
                q.fit(xt[:,ordered],yt)
            warning_count+=sum(issubclass(w.category,ConvergenceWarning) for w in records)
            pred['mtlasso_penalized']=q.predict(xv[:,ordered])
            beta=np.linalg.lstsq(xt[:,ordered],yt,rcond=None)[0]
            pred['mtlasso_post_ols']=xv[:,ordered]@beta
        result.append((float(fraction),ordered.tolist(),{m:float(((pred[m]-yv)**2).sum()) for m in METHODS},
                       {m:pred[m] for m in METHODS},int(len(active))))
    return result,warning_count


def fit_fold(x,y,groups,seed,held,fold):
    keep=np.ones(len(x),dtype=bool);keep[held]=False
    train_ids=np.flatnonzero(keep);trgroups=groups[keep]
    assert not set(trgroups)&set(groups[held])
    inner=group_folds(trgroups,5,seed+fold+1)
    scores={m:np.zeros(len(RATIOS)) for m in METHODS};warnings_seen=0
    for innerheld in inner:
        mask=np.ones(len(train_ids),dtype=bool);mask[innerheld]=False
        xt,yt,xv,yv=scale(x[train_ids[mask]],y[train_ids[mask]],x[train_ids[innerheld]],y[train_ids[innerheld]])
        paths,w=candidates(xt,yt,xv,yv);warnings_seen+=w
        for k,(_,_,risks,_,_) in enumerate(paths):
            for method in METHODS:scores[method][k]+=risks[method]
    indices={m:int(np.argmin(scores[m])) for m in METHODS}
    xt,yt,xv,yv=scale(x[train_ids],y[train_ids],x[held],y[held])
    paths,w=candidates(xt,yt,xv,yv);warnings_seen+=w
    out=[]
    for method in METHODS:
        fraction,selected,risks,pred,active=paths[indices[method]]
        assert len(selected)<=3 and np.isfinite(risks[method])
        out.append(dict(method=method,n_cell_lines=len(held),risk=risks[method]/(len(held)*y.shape[1]),
                        fraction=fraction,chosen='|'.join(str(j) for j in selected),
                        n_selected=len(selected),n_active_uncapped=active,
                        inner_risk=float(scores[method][indices[method]]/(len(train_ids)*y.shape[1])),
                        convergence_warnings=warnings_seen))
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true');args=parser.parse_args()
    frame=pd.read_csv(PREV/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv')
    assert frame.shape==(499,857) and frame.columns[:7].tolist()==TARGETS
    x=frame.iloc[:,20:].to_numpy(float);y=frame.iloc[:,:7].to_numpy(float)
    tissue=np.argmax(frame.iloc[:,7:20].to_numpy(float),axis=1)
    scenarios=[('repeated_random',SEEDS if args.full else SEEDS[:1],np.arange(len(x)),5),
               ('held_out_tissue',[SEEDS[0]],tissue,13)]
    out=HERE/'results'/'multitask_lasso';out.mkdir(parents=True,exist_ok=True)
    rows=(pd.read_csv(out/'folds.csv').to_dict('records') if (out/'folds.csv').exists() else [])
    completed={(r['evaluation'],int(r['seed']),int(r['fold']),r['method']) for r in rows}
    t0=time.perf_counter()
    for ev,seeds,groups,nfold in scenarios:
        for seed in seeds:
            for fold,held in enumerate(group_folds(groups,nfold,seed)):
                if all((ev,int(seed),fold,m) in completed for m in METHODS):
                    continue
                assert not any((ev,int(seed),fold,m) in completed for m in METHODS), 'Partial fold; inspect before resuming'
                ts=time.perf_counter();r=fit_fold(x,y,groups,seed,held,fold)
                for row in r:row.update(evaluation=ev,seed=seed,fold=fold,seconds=time.perf_counter()-ts)
                rows+=r
                pd.DataFrame(rows).to_csv(out/'folds.csv',index=False,float_format='%.17g')
                print(ev,seed,fold+1,'/',nfold,'seconds',round(time.perf_counter()-ts,2),'risks',[round(v['risk'],4) for v in r],flush=True)
    d=pd.DataFrame(rows);summ=[]
    for (ev,method),g in d.groupby(['evaluation','method']):
        summ.append(dict(evaluation=ev,method=method,n_outer_folds=len(g),
                         risk=float(np.average(g.risk,weights=g.n_cell_lines)),
                         mean_selected=float(np.mean(g.n_selected)),
                         fractions='|'.join(f'{v:g}' for v in sorted({round(float(v),10) for v in g.fraction})),
                         convergence_warning_folds=int((g.convergence_warnings>0).sum())))
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False,float_format='%.17g')
    meta={'full_random_repeats':args.full,'alpha_fractions':RATIOS,'runtime_seconds':time.perf_counter()-t0,
          'selection':'multi-task group Lasso, top 3 active coefficient norms, refit penalized/OLS',
          'caution':'post-hoc pilot; preprocessed example; no independently frozen dataset',
          'sklearn_version':__import__('sklearn').__version__,'numpy_version':np.__version__}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(pd.DataFrame(summ).to_string(index=False),flush=True)
    print('TOTAL_SECONDS',round(time.perf_counter()-t0,2),flush=True)

if __name__=='__main__':main()
