#!/usr/bin/env python3
"""Descriptive per-tissue held-out risks for the inner-CV selector."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def main():
    original=pd.read_csv(ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_readable.csv',nrows=1)
    labels=original.columns[7:20].tolist()
    losses=pd.read_csv(ROOT/'results'/'final_budget3'/'held_out_losses.csv')
    losses=losses[(losses.evaluation=='held_out_tissue')&(losses.selector=='inner_cv')]
    if len(losses)!=499*7:raise RuntimeError('Incomplete tissue evaluation')
    rows=[]
    for group,g in losses.groupby('group',sort=True):
        assert g.fold.nunique()==1 and g.row_id.nunique()*7==len(g)
        n=int(g.row_id.nunique())
        row={'tissue':labels[int(group)],'n_cell_lines':n,'outer_fold':int(g.fold.iloc[0])}
        for policy in ['C0','Shared','Family','Global3','Exact_unconstrained']:
            row[policy]=float(g[policy].mean())
        row['Family_minus_Global3']=row['Family']-row['Global3'];rows.append(row)
    output=pd.DataFrame(rows)
    assert len(output)==13 and output.n_cell_lines.sum()==499
    summary=pd.read_csv(ROOT/'results'/'final_budget3'/'summary.csv')
    summary=summary[(summary.evaluation=='held_out_tissue')&(summary.selector=='inner_cv')].iloc[0]
    for policy in ['C0','Shared','Family','Global3','Exact_unconstrained']:
        assert np.isclose(np.average(output[policy],weights=output.n_cell_lines),summary[policy],atol=1e-12)
    out=ROOT/'results'/'tissue_breakdown'/'summary.csv';out.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(out,index=False);print(output.round(4).to_string(index=False))

if __name__=='__main__':main()
