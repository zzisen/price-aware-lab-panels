#!/usr/bin/env python3
"""Exploratory repeated patient-holdout splits with complete retraining.

200 splits partition the same 44 public-data patients into 35 training and nine
test patients. Each split repeats training-only inner action selection, scaling
and fitting. Split quantiles describe this dataset's resampling variability;
they are not a confidence interval and the splits are not independent cohorts.
"""
from pathlib import Path
import csv
import json
import numpy as np
from phospho_sensitivity import (load_matrices, recorded_loss_tensor, choose_actions,
                                 predictions, TARGETS, CHANNELS)

ROOT=Path(__file__).resolve().parent
CONFIGS=[('ols','standardized',False),('bounded_ols','standardized',True),
         ('ols_native','native',False),('bounded_ols_native','native',True)]


def heldout(x,y,train,test,scale,clip):
    assert not set(train) & set(test) and len(np.unique(train))==len(train)
    shared,exact=choose_actions(x,y,train,0.0,clip,scale)
    losses=np.empty((len(test),len(TARGETS),3))
    for pos,i in enumerate(test):
        pred,ym,ys=predictions(x[train],y[train],x[i],0.0,clip)
        yz=(y[i]-ym)/ys
        loss_scale=ys if scale=='native' else 1.0
        losses[pos,:,0]=np.mean((yz*loss_scale)**2,axis=1)
        losses[pos,:,1]=np.mean(((pred[shared]-yz)*loss_scale)**2,axis=1)
        losses[pos,:,2]=np.mean(((pred[exact,np.arange(len(TARGETS))]-yz)*loss_scale)**2,axis=1)
    return losses,shared,exact


def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    patients,x,y=load_matrices(ROOT/'inputs'/'B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv',TARGETS)
    expected=recorded_loss_tensor(ROOT/'inputs'/'B1_OUTER_PATIENT_TARGET_LOSSES.csv',patients,TARGETS)
    assert len(patients)==44
    test=np.array([0]); train=np.arange(1,44)
    replay,_,_=heldout(x,y,train,test,'standardized',False)
    assert np.allclose(replay[0],expected[0],rtol=1e-9,atol=1e-8)
    rng=np.random.default_rng(20261027); splits=[]; records=[]
    for split in range(200):
        order=rng.permutation(44); train=np.sort(order[:35]); test=np.sort(order[35:])
        assert len(set(train)&set(test))==0 and len(train)==35 and len(test)==9
        splits.append({'split':split+1,'training_patients':'|'.join(patients[i] for i in train),
                       'test_patients':'|'.join(patients[i] for i in test)})
        for config,scale,clip in CONFIGS:
            losses,shared,exact=heldout(x,y,train,test,scale,clip)
            mean=losses.mean(axis=(0,1))
            records.append({'split':split+1,'config':config,'n_train':35,'n_test':9,
                            'C0':mean[0],'Shared':mean[1],'Exact':mean[2],
                            'Shared_minus_Exact':mean[1]-mean[2],
                            'selected_shared_channel':CHANNELS[shared],
                            'selected_exact_channels':'|'.join(CHANNELS[j] for j in exact),
                            'UPN21_in_test':int('UPN21' in [patients[i] for i in test]),
                            'UPN68_in_test':int('UPN68' in [patients[i] for i in test])})
    summary=[]
    for config,scale,clip in CONFIGS:
        rr=[r for r in records if r['config']==config]
        d=np.array([r['Shared_minus_Exact'] for r in rr])
        summary.append({'config':config,'n_splits':len(d),'n_Shared_lower':int((d<0).sum()),
                        'n_Exact_lower':int((d>0).sum()),'n_ties':int((d==0).sum()),
                        'difference_2.5pct':float(np.percentile(d,2.5)),
                        'difference_median':float(np.median(d)),
                        'difference_97.5pct':float(np.percentile(d,97.5))})
    out=ROOT/'results'/'repeated_holdout';out.mkdir(parents=True,exist_ok=True)
    write(out/'split_membership.csv',splits);write(out/'split_results.csv',records);write(out/'split_summary.csv',summary)
    meta={'seed':20261027,'n_splits':200,'n_train':35,'n_test':9,
          'unit':'patient','inner':'leave-one-training-patient-out',
          'warning':'post-hoc repeated splits of the same cohort; percentiles describe split variability, not a confidence interval',
          'single_fold_reproduction_max_abs_error':float(np.max(np.abs(replay[0]-expected[0])))}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
