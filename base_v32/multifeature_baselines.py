#!/usr/bin/env python3
"""Post-hoc three-feature multivariable linear baselines on fixed GDSC folds.

Both baselines acquire at most three distinct features and may use all three in
all seven response models. They are deliberately simple comparators, not a
claim to cover all sparse or nonlinear multivariate prediction methods.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_gdsc import SEEDS, TARGETS
from target_resolution import group_folds

ROOT = Path(__file__).resolve().parent


def standardize(x_train, y_train, x_test, y_test):
    xm=x_train.mean(0); xs=x_train.std(0, ddof=1); xs=np.where(xs==0, 1., xs)
    ym=y_train.mean(0); ys=y_train.std(0, ddof=1); ys=np.where(ys==0, 1., ys)
    return (x_train-xm)/xs, (y_train-ym)/ys, (x_test-xm)/xs, (y_test-ym)/ys


def forward_candidate_scores(xt, yt, xv, yv, chosen):
    """Sum held-out squared errors for each possible next predictor.

    Arrays are standardized using the current training subset. Residualization
    gives the exact unpenalized OLS prediction for each single added feature.
    Candidates collinear with the already selected columns are ineligible.
    """
    z=np.column_stack([np.ones(len(xt)), xt[:, chosen]])
    zv=np.column_stack([np.ones(len(xv)), xv[:, chosen]])
    by=np.linalg.lstsq(z, yt, rcond=None)[0]
    bx=np.linalg.lstsq(z, xt, rcond=None)[0]
    rx=xt-z@bx; ry=yt-z@by
    den=np.sum(rx*rx, axis=0)
    b=np.divide(rx.T@ry, den[:, None], out=np.zeros((xt.shape[1],yt.shape[1])), where=den[:, None]>1e-12)
    pred=(zv@by)[:, None, :]+(xv-zv@bx)[:, :, None]*b[None, :, :]
    risk=np.sum((pred-yv[:, None, :])**2, axis=(0, 2))
    risk[den<=1e-12]=np.inf
    return risk


def choose_forward(x, y, groups, seed, budget=3):
    inner=group_folds(groups, 5, seed)
    chosen=[]; scores=[]
    for _ in range(budget):
        risk=np.zeros(x.shape[1])
        for held in inner:
            keep=np.ones(len(x), dtype=bool); keep[held]=False
            xt,yt,xv,yv=standardize(x[keep],y[keep],x[held],y[held])
            risk+=forward_candidate_scores(xt,yt,xv,yv,chosen)
        risk[chosen]=np.inf
        i=int(np.argmin(risk))
        if not np.isfinite(risk[i]):
            raise RuntimeError('No candidate remains non-collinear in all inner folds')
        chosen.append(i)
        scores.append(float(risk[i]/(len(x)*y.shape[1])))
    return chosen,scores


def choose_correlation(x, y, budget=3):
    xt,yt,_,_=standardize(x,y,x[:1],y[:1])
    r=xt.T@yt/(len(x)-1)
    return np.argsort(-np.mean(r*r, axis=1),kind='stable')[:budget].tolist()


def predict_loss(x_train,y_train,x_test,y_test,chosen):
    xt,yt,xv,yv=standardize(x_train,y_train,x_test,y_test)
    z=np.column_stack([np.ones(len(xt)),xt[:,chosen]])
    zv=np.column_stack([np.ones(len(xv)),xv[:,chosen]])
    beta=np.linalg.lstsq(z,yt,rcond=None)[0]
    return (zv@beta-yv)**2


def main():
    frame=pd.read_csv(ROOT/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv')
    assert frame.shape==(499,857) and frame.columns[:7].tolist()==TARGETS
    x=frame.iloc[:,20:].to_numpy(float); y=frame.iloc[:,:7].to_numpy(float)
    tissue=np.argmax(frame.iloc[:,7:20].to_numpy(float),axis=1)
    folds=[]
    for ev,seeds,groups,nfold in [('repeated_random',SEEDS,np.arange(len(x)),5),('held_out_tissue',[SEEDS[0]],tissue,13)]:
        for seed in seeds:
            for fold,held in enumerate(group_folds(groups,nfold,seed)):
                keep=np.ones(len(x),dtype=bool);keep[held]=False
                assert not set(groups[keep]) & set(groups[held])
                train_x,train_y=x[keep],y[keep]
                forward,_=choose_forward(train_x,train_y,groups[keep],seed+fold+1)
                corr=choose_correlation(train_x,train_y)
                for method,chosen in [('correlation_top3_ols',corr),('forward_inner_cv_ols',forward)]:
                    loss=predict_loss(train_x,train_y,x[held],y[held],chosen)
                    folds.append(dict(evaluation=ev,method=method,seed=seed,fold=fold,
                        n_cell_lines=len(held),mean_risk=float(loss.mean()),
                        feature_indices='|'.join(str(i) for i in chosen)))
                print(ev,seed,fold+1,'/',nfold,flush=True)
    fold_df=pd.DataFrame(folds)
    records=[]
    for (ev,method),g in fold_df.groupby(['evaluation','method'],sort=True):
        records.append(dict(evaluation=ev,method=method,n_outer_folds=len(g),
            feature_cap=3,risk=float(np.average(g.mean_risk,weights=g.n_cell_lines))))
    out=ROOT/'results/multifeature_baselines';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out/'summary.csv',index=False,float_format='%.17g')
    fold_df.to_csv(out/'folds.csv',index=False,float_format='%.17g')
    print(pd.DataFrame(records).to_string(index=False),flush=True)

if __name__=='__main__':main()
