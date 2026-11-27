#!/usr/bin/env python3
"""Independent checks of stored losses, fold isolation, and regression code."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from target_resolution import action_losses, group_folds, nested_evaluate, select_by_correlation
from phospho_sensitivity import load_matrices, recorded_loss_tensor, TARGETS

ROOT = Path(__file__).resolve().parent


def main():
    report = {}
    patients, x, y = load_matrices(ROOT/'inputs'/'B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv', TARGETS)
    result = nested_evaluate(x, y, ['one_family']*len(TARGETS), np.arange(len(patients)),
                             outer_splits=44, inner_splits=43)
    expected = recorded_loss_tensor(ROOT/'inputs'/'B1_OUTER_PATIENT_TARGET_LOSSES.csv', patients, TARGETS)
    error = float(np.max(np.abs(result['losses'][:,:,[0,1,3]]-expected)))
    assert error < 1e-8
    report['original_v3_replay_max_absolute_loss_error'] = error

    rng = np.random.default_rng(2030)
    xt = rng.normal(size=(30,5)); yt = rng.normal(size=(30,3,2))
    xv = rng.normal(size=(6,5)); yv = rng.normal(size=(6,3,2))
    test_losses, c0 = action_losses(xt,yt,xv,yv)
    independently = np.empty_like(test_losses)
    for a in range(5):
        for t in range(3):
            for o in range(2):
                design = np.column_stack((np.ones(len(xt)),xt[:,a]))
                coeff = np.linalg.lstsq(design,yt[:,t,o],rcond=None)[0]
                pred = np.column_stack((np.ones(len(xv)),xv[:,a])) @ coeff
                std = yt[:,t,o].std(ddof=1)
                if o == 0: accumulated = np.zeros(len(xv))
                accumulated += ((pred-yv[:,t,o])/std)**2/2
            independently[:,a,t] = accumulated
    report['independent_ols_max_absolute_loss_error'] = float(np.max(np.abs(test_losses-independently)))
    assert report['independent_ols_max_absolute_loss_error'] < 1e-11

    _, scores = select_by_correlation(xt,yt,['a','b','b'],np.arange(30))
    independent_scores = np.empty_like(scores)
    for a in range(5):
        for t in range(3):
            vals = [np.corrcoef(xt[:,a],yt[:,t,o])[0,1]**2 for o in range(2)]
            independent_scores[a,t] = 1-np.mean(vals)
    report['correlation_filter_max_absolute_score_error'] = float(np.max(np.abs(scores-independent_scores)))
    assert report['correlation_filter_max_absolute_score_error'] < 1e-12

    groups = np.repeat(np.arange(13),[5,6,9,11,8,4,7,12,3,5,6,7,8])
    folds = group_folds(groups,5,42)
    for held in folds:
        mask=np.ones(len(groups),dtype=bool); mask[held]=False
        assert not set(groups[mask]) & set(groups[held])
    assert np.array_equal(np.sort(np.concatenate(folds)),np.arange(len(groups)))
    report['grouped_split_keeps_groups_together'] = True

    base = ROOT/'results'/'gdsc'
    rows = pd.read_csv(base/'held_out_losses.csv')
    summaries = pd.read_csv(base/'contrast_summary.csv')
    checks=[]
    for ev in summaries.evaluation.unique():
        if ev == 'random_5_splits_row_average': continue
        r=rows[rows.evaluation==ev]
        assert len(r)==499*7 and r.row_id.nunique()==499
        for _,s in summaries[summaries.evaluation==ev].iterrows():
            a,b=s['contrast'].split(' minus ')
            actual=(r[f'loss_{a}']-r[f'loss_{b}']).mean()
            checks.append(abs(actual-s['mean_difference']))
    report['gdsc_summary_recomputed_max_absolute_error'] = float(max(checks))
    assert report['gdsc_summary_recomputed_max_absolute_error'] < 1e-12
    report['gdsc_unique_source_cell_lines'] = int(rows[rows.evaluation=='leave_one_tissue_out'].cell_line.nunique())
    assert report['gdsc_unique_source_cell_lines'] == 499
    report['status'] = 'PASS'
    out=ROOT/'results'/'independent_quality_checks.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
