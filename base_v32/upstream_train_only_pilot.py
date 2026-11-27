#!/usr/bin/env python3
"""Fold-local mutation filtering and missing-outcome evaluation on release-5.

Two source-cohort scenarios: the previously selected 499 examples, and all
624 source lines with each observed response evaluated without outcome
imputation. Drug selection, upstream expression/IC50 processing and the
source's genomic matrix construction still predate our splits.
"""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PREV=HERE if (HERE/'target_resolution.py').exists() else HERE.parent/'P2_V3.1_EXACT_2026-10-28'
sys.path.insert(0,str(PREV));from target_resolution import group_folds,select_actions
from exact_budget import globally_optimal_actions
from evaluate_gdsc import SEEDS,TARGETS,FAMILIES
FAM=np.array(FAMILIES);POLICIES=['C0','Shared','Family','Global3','Exact']


def eligible_features(x,mut_start):
    result=np.ones(x.shape[1],dtype=bool)
    result[mut_start:]=(x[:,mut_start:]!=0).sum(axis=0)>1
    return result


def loss_sums(xtr,ytr,xv,yv):
    """Training-only standardization and univariate OLS per observed target."""
    sums=np.zeros((xtr.shape[1],ytr.shape[1]));base=np.zeros(ytr.shape[1]);count=np.zeros(ytr.shape[1],int)
    for t in range(ytr.shape[1]):
        train=np.isfinite(ytr[:,t]); valid=np.isfinite(yv[:,t]);n=int(train.sum())
        if n<3:raise ValueError('Fewer than three observed training outcomes')
        if not valid.any():continue
        xt=xtr[train]; yt=ytr[train,t];xm=xt.mean(0);xs=xt.std(0,ddof=1);xs=np.where(xs==0,1.,xs)
        ym=yt.mean();ys=yt.std(ddof=1);ys=1. if ys==0 else ys
        z=(xt-xm)/xs;zy=(yt-ym)/ys;den=(z*z).sum(0)
        slope=np.divide(z.T@zy,den,out=np.zeros(xtr.shape[1]),where=den>0)
        obs=(yv[valid,t]-ym)/ys;pred=((xv[valid]-xm)/xs)*slope
        sums[:,t]=np.sum((pred-obs[:,None])**2,axis=0)
        base[t]=np.sum(obs**2);count[t]=len(obs)
    return sums,base,count


def choose(x,y,groups,seed,mut_start,filter_mut=True,mode='strict_intersection'):
    folds=group_folds(groups,5,seed);sums=np.zeros((x.shape[1],y.shape[1]));counts=np.zeros(y.shape[1],int)
    allowed=eligible_features(x,mut_start) if filter_mut else np.ones(x.shape[1],bool)
    for held in folds:
        keep=np.ones(len(x),bool);keep[held]=False
        fold_allowed=eligible_features(x[keep],mut_start) if filter_mut else np.ones(x.shape[1],bool)
        if mode=='strict_intersection':allowed &= fold_allowed
        a,c,n=loss_sums(x[keep],y[keep],x[held],y[held])
        if mode=='inner_fallback':
            # A filtered feature predicts the inner-training response mean in
            # folds where it is unavailable; it must still be eligible on the
            # full outer-training set to be selected for outer evaluation.
            a[~fold_allowed,:]=c[None,:]
        sums+=a;counts+=n
    assert np.all(counts>0) and allowed.any()
    risks=sums[allowed]/counts[None,:];idx=np.flatnonzero(allowed)
    sh=np.repeat(idx[int(np.argmin(risks.mean(1)))],y.shape[1]);fa=np.empty(y.shape[1],int)
    for label in np.unique(FAM):
        t=np.flatnonzero(FAM==label);fa[t]=idx[int(np.argmin(risks[:,t].mean(1)))]
    ex=idx[np.argmin(risks,axis=0)];glob,cert=globally_optimal_actions(risks,3);glob=idx[glob]
    return {'Shared':sh,'Family':fa,'Global3':glob,'Exact':ex},allowed,counts,cert


def check_against_previous():
    d=pd.read_csv(PREV/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv');x=d.iloc[:,20:].to_numpy(float);y=d.iloc[:,:7].to_numpy(float)
    group=np.arange(len(x));held=group_folds(group,5,SEEDS[0])[0]
    keep=np.ones(len(x),bool);keep[held]=False
    new,_,_,_=choose(x[keep],y[keep],group[keep],SEEDS[0]+1,mut_start=x.shape[1],filter_mut=False)
    old,_=select_actions(x[keep],y[keep],FAMILIES,group[keep],inner_splits=5,seed=SEEDS[0]+1)
    for policy in ['Shared','Family','Exact']:assert np.array_equal(new[policy],old[policy]),policy
    print('PASS complete-case selection matches prior implementation',flush=True)


def main():
    check_against_previous()
    data=np.load(HERE/'source_local_only/release5_local_624x840.npz')
    name_map={'MS-HL.MUT':'MS.HL.MUT','CDKN2a(p14).MUT':'CDKN2a.p14..MUT'}
    names=[name_map.get(str(a),str(a)) for a in data['features']]
    ex=pd.read_csv(PREV/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv')
    map_path=PREV/'inputs/GDSC/exampleGDSC_2.3-1_row_to_source_cell_line.csv'
    if map_path.exists():
        mapping=pd.read_csv(map_path)
        idx=pd.Index(data['cells']).get_indexer(mapping.gdsc_cell_line_name)
    else:
        anchor=pd.Index(names).get_indexer(ex.columns[20:30]);assert (anchor>=0).all()
        all_anchors=data['X'][:,anchor];idx=[]
        for row in ex.iloc[:,20:30].to_numpy(float):
            matches=np.flatnonzero(np.max(np.abs(all_anchors-row),axis=1)<1e-10)
            assert len(matches)==1, 'Ambiguous source identity; inspect locally'
            idx.append(int(matches[0]))
        idx=np.asarray(idx,dtype=int)
    assert (idx>=0).all() and len(set(idx))==499
    extras=sorted(set(names)-set(ex.columns[20:]));feature_order=list(ex.columns[20:])+extras
    cols=pd.Index(names).get_indexer(feature_order);assert (cols>=0).all() and len(cols)==840
    X=data['X'][:,cols];Y=data['Y'];tissue=np.argmax(data['tissue'],axis=1)
    assert np.max(np.abs(X[idx,:837]-ex.iloc[:,20:].to_numpy(float)))<1e-10
    assert np.max(np.abs(Y[idx]-ex.iloc[:,:7].to_numpy(float)))<1e-10
    assert np.array_equal(tissue[idx],np.argmax(ex.iloc[:,7:20].to_numpy(float),axis=1))
    source_idx=np.arange(len(X));cohorts=[('original_499',idx),('source_624',source_idx)]
    records=[];targets=[];selections=[];out=HERE/'results/upstream_rebuild';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    for cohort,rows in cohorts:
        x=X[rows];y=Y[rows];ts=tissue[rows]
        for mode in ['strict_intersection','inner_fallback']:
          for ev,groups,nfold,seeds in [('random_5fold',np.arange(len(x)),5,SEEDS),('held_out_tissue',ts,13,[SEEDS[0]])]:
           for seed in seeds:
            for fold,held in enumerate(group_folds(groups,nfold,seed)):
                keep=np.ones(len(x),bool);keep[held]=False
                assert not set(groups[keep])&set(groups[held])
                choice,allowed,innercounts,cert=choose(x[keep],y[keep],groups[keep],seed+fold+1,mut_start=769,mode=mode)
                s,c,n=loss_sums(x[keep],y[keep],x[held],y[held]);risks={'C0':c}
                for policy in POLICIES[1:]:risks[policy]=s[choice[policy],np.arange(7)]
                row={'cohort':cohort,'mode':mode,'evaluation':ev,'seed':seed,'fold':fold,'n_test_lines':len(held),
                     'n_observed_outcomes':int(n.sum()),'n_eligible_features':int(allowed.sum()),
                     'inner_observed_min':int(innercounts.min()),'global_partition_count':cert['n_partitions']}
                for policy in POLICIES:row[policy]=float(risks[policy].sum()/n.sum()) if n.sum() else np.nan
                records.append(row)
                for t,name in enumerate(TARGETS):
                    one={'cohort':cohort,'mode':mode,'evaluation':ev,'seed':seed,'fold':fold,'target':name,'n_observed':int(n[t])}
                    for policy in POLICIES:one[policy]=float(risks[policy][t]/n[t]) if n[t] else np.nan
                    targets.append(one)
                    for policy in POLICIES[1:]:
                        j=int(choice[policy][t])
                        selections.append({'cohort':cohort,'mode':mode,'evaluation':ev,'seed':seed,'fold':fold,
                            'target':name,'policy':policy,'feature':feature_order[j],
                            'outer_training_nonzero':int((x[keep,j]!=0).sum()) if j>=769 else None})
                print(cohort,mode,ev,seed,fold+1,'/',nfold,'eligible',allowed.sum(),'observed',n.sum(),flush=True)
    folds=pd.DataFrame(records);detail=pd.DataFrame(targets)
    folds.to_csv(out/'folds.csv',index=False,float_format='%.17g')
    detail.to_csv(out/'target_fold_aggregates.csv',index=False,float_format='%.17g')
    pd.DataFrame(selections).to_csv(out/'action_selections.csv',index=False)
    summaries=[]
    for (cohort,mode,ev),g in folds.groupby(['cohort','mode','evaluation']):
        row={'cohort':cohort,'mode':mode,'evaluation':ev,'n_outer_folds':len(g),'total_scored_observations':int(g.n_observed_outcomes.sum()),
             'min_eligible_features':int(g.n_eligible_features.min()),'max_eligible_features':int(g.n_eligible_features.max())}
        for policy in POLICIES:row[policy]=float(np.average(g[policy],weights=g.n_observed_outcomes))
        summaries.append(row)
    result=pd.DataFrame(summaries);result.to_csv(out/'summary.csv',index=False,float_format='%.17g')
    meta={'runtime_seconds':time.perf_counter()-start,'seeds':SEEDS,'target_selection':'seven drugs fixed from prior source example',
          'filter':'mutation >1 in outer training; either require >1 in every inner training fold, or use inner-training-mean fallback when inner-ineligible. No target-based complete-case restriction in 624 scenario.',
          'missing':'fit on observed train outcomes; score observed test outcomes only; no outcome imputation',
          'source_limit':'upstream processed expression, fitted IC50, source genomic matrix and target choice predate folds',
          'aggregation':'mean over observed cell-target losses; target training selection gives equal target weight'}
    (out/'pilot_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(result.to_string(index=False),flush=True)
    print('TOTAL_SECONDS',round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':main()
