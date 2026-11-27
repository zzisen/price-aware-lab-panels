#!/usr/bin/env python3
"""Repeat all matching outer splits with an exact 3-feature selection score."""
from pathlib import Path
import csv,json
import numpy as np
import pandas as pd
from target_resolution import group_folds,action_losses,select_actions,select_by_correlation
from evaluate_gdsc import TARGETS,FAMILIES,SEEDS
from exact_budget import globally_optimal_actions

ROOT=Path(__file__).resolve().parent
METHODS=['inner_cv','correlation_filter']
POLICIES=['C0','Shared','Family','Global3','Exact_unconstrained']

def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    frame=pd.read_csv(ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_readable.csv')
    assert frame.shape==(499,857) and frame.columns[:7].tolist()==TARGETS
    x=frame.iloc[:,20:].to_numpy(float);y=frame.iloc[:,:7].to_numpy(float)
    tissue=frame.iloc[:,7:20].to_numpy(float);groups_tissue=np.argmax(tissue,axis=1)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.all(tissue.sum(axis=1)==1)
    map_path=ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_row_to_source_cell_line.csv'
    if map_path.exists():
        source=pd.read_csv(map_path)
        assert source.package_row_1based.tolist()==list(range(1,500))
        source_ids=source.gdsc_cell_line_name.tolist(); id_mode='provided source-identity mapping'
    else:
        source_ids=[f'package_row_{i+1:03d}' for i in range(499)]
        id_mode='package row identifiers only; source identities require separate audit'
    features=frame.columns[20:].tolist();records=[];actions=[];certificates=[];summaries=[]
    for ev,seeds,groups,nfold in [('repeated_random',SEEDS,np.arange(499),5),('held_out_tissue',[SEEDS[0]],groups_tissue,13)]:
        repeated=[]
        for seed in seeds:
            result=np.full((2,len(x),7,len(POLICIES)),np.nan)
            for fold,held in enumerate(group_folds(groups,nfold,seed)):
                train=np.ones(len(x),dtype=bool);train[held]=False
                assert not set(groups[train]) & set(groups[held])
                cv,rv=select_actions(x[train],y[train],FAMILIES,groups[train],inner_splits=5,seed=seed+fold+1)
                cf,rf=select_by_correlation(x[train],y[train],FAMILIES,groups[train])
                candidate,c0=action_losses(x[train],y[train,:,None],x[held],y[held,:,None])
                for m,(method,choice,risk) in enumerate(zip(METHODS,[cv,cf],[rv,rf])):
                    optimum,cert=globally_optimal_actions(risk,3)
                    assert cert['n_partitions']==365
                    family_score=float(risk[choice['Family'],np.arange(7)].mean())
                    assert cert['score']<=family_score+1e-10
                    selected={'Shared':choice['Shared'],'Family':choice['Family'],
                              'Global3':optimum,'Exact_unconstrained':choice['Exact']}
                    result[m,held,:,0]=c0
                    for j,p in enumerate(POLICIES[1:],1):
                        a=selected[p]
                        result[m,held,:,j]=candidate[np.arange(len(held))[:,None],a[None,:],np.arange(7)[None,:]]
                        for t,target in enumerate(TARGETS):
                            actions.append({'evaluation':ev,'seed':seed,'fold':fold,'selector':method,'policy':p,
                                'target':target,'feature_index':int(a[t]),'feature':features[a[t]],
                                'distinct_features_this_fold':len(np.unique(a))})
                    certificates.append({'evaluation':ev,'seed':seed,'fold':fold,'selector':method,
                        'n_partitions':cert['n_partitions'],'global_training_score':cert['score'],
                        'exhaustive_lower_bound':cert['exhaustive_lower_bound'],
                        'family_training_score':family_score,
                        'global_distinct_features':cert['distinct_features'],'global_partition_masks':str(cert['partition_masks'])})
                for m,method in enumerate(METHODS):
                    for i in held:
                        for t,target in enumerate(TARGETS):
                            records.append({'evaluation':ev,'seed':seed,'fold':fold,'selector':method,'row_id':int(i+1),
                                'cell_line':source_ids[i],'group':int(groups[i]),'target':target,
                                **{p:float(result[m,i,t,j]) for j,p in enumerate(POLICIES)}})
                print(ev,seed,fold+1,'/',nfold,flush=True)
            assert np.isfinite(result).all();repeated.append(result)
        avg=np.stack(repeated).mean(axis=0)
        for m,method in enumerate(METHODS):
            r={'evaluation':ev,'selector':method,'n_cell_lines':len(x),'n_outer_folds':len(seeds)*nfold}
            for j,p in enumerate(POLICIES):r[p]=float(avg[m,:,:,j].mean())
            for p in POLICIES[1:]:
                q=[a['distinct_features_this_fold'] for a in actions if a['evaluation']==ev and a['selector']==method and a['policy']==p and a['target']==TARGETS[0]]
                r[p+'_mean_features']=float(np.mean(q));r[p+'_min_features']=int(min(q));r[p+'_max_features']=int(max(q))
            summaries.append(r)
    out=ROOT/'results'/'final_budget3';out.mkdir(parents=True,exist_ok=True)
    write(out/'summary.csv',summaries);write(out/'held_out_losses.csv',records)
    write(out/'action_selections.csv',actions);write(out/'certificates.csv',certificates)
    selection=pd.DataFrame(actions).pivot_table(index=['evaluation','selector','seed','fold','target'],
                  columns='policy',values='feature_index',aggfunc='first')
    same=(selection['Global3']==selection['Family']).groupby(level=['evaluation','selector','seed','fold']).all()
    matches=[]
    for (ev,selector),values in same.groupby(level=[0,1]):
        matches.append({'evaluation':ev,'selector':selector,'same_folds':int(values.sum()),'total_folds':int(len(values))})
    write(out/'figure6_match_counts.csv',matches)
    meta={'feature_cap':3,'targets':7,'target_partitions_enumerated':365,
          'meaning':'global minimizer of the finite additive training selection score for each outer fold',
          'numerics':'exhaustive partition objective and returned assignment agree to 1e-10',
          'not_a_claim':'not global optimization of held-out or population risk; not equal actual financial cost',
          'post_hoc':True,'source_id_mode':id_mode,'upstream':'source dataset globally preprocessed before downstream folds'}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Completed',out,flush=True)

if __name__=='__main__':main()
