"""Exploratory training-only selection under five distinct-feature caps."""
from pathlib import Path
import sys,csv
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from target_resolution import group_folds,select_actions,action_losses
from exact_budget import globally_optimal_actions
from evaluate_gdsc import TARGETS,FAMILIES,SEEDS

def main():
    f=pd.read_csv(ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_readable.csv')
    x=f.iloc[:,20:].to_numpy(float);y=f.iloc[:,:7].to_numpy(float)
    tissue=np.argmax(f.iloc[:,7:20].to_numpy(float),axis=1)
    budgets=[1,2,3,4,7];result=[]
    for ev,seeds,groups,nfold in [('repeated_random',SEEDS,np.arange(len(x)),5),('held_out_tissue',[SEEDS[0]],tissue,13)]:
        res=[];counts={k:[] for k in budgets}
        for seed in seeds:
            losses=np.full((len(x),7,len(budgets)),np.nan)
            for fold,held in enumerate(group_folds(groups,nfold,seed)):
                train=np.ones(len(x),bool);train[held]=False
                _,risk=select_actions(x[train],y[train],FAMILIES,groups[train],inner_splits=5,seed=seed+fold+1)
                candidate,_=action_losses(x[train],y[train,:,None],x[held],y[held,:,None])
                for j,k in enumerate(budgets):
                    choice,cert=globally_optimal_actions(risk,k)
                    losses[held,:,j]=candidate[np.arange(len(held))[:,None],choice[None,:],np.arange(7)[None,:]]
                    counts[k].append(len(np.unique(choice)))
            res.append(losses)
        mean=np.mean(np.stack(res),axis=0)
        for j,k in enumerate(budgets):
            result.append({'evaluation':ev,'budget':k,'risk':float(mean[:,:,j].mean()),
                           'mean_features':float(np.mean(counts[k])),'n_outer_folds':len(counts[k])})
    out=ROOT/'results'/'budget_curve'/'summary.csv';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print(pd.DataFrame(result).to_string(index=False),flush=True)
if __name__=='__main__':main()
