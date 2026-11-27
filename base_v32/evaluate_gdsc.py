#!/usr/bin/env python3
"""Exploratory independent-data application of fixed target resolutions.

The published BayesSUR exampleGDSC object is a globally preprocessed GDSC
subset; this is not patient-model transport, a prospective experiment, or a
pre-registered replication. All downstream fitting/selection occurs within
the training folds. The released object does not contain cell-line IDs.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from target_resolution import nested_evaluate

ROOT = Path(__file__).resolve().parent
TARGETS = ['Methotrexate', 'RDEA119', 'PD-0325901', 'CI-1040', 'AZD6244', 'Nilotinib', 'Axitinib']
FAMILIES = ['single', 'MEK', 'MEK', 'MEK', 'MEK', 'two_drug', 'two_drug']
SEEDS = [20261027, 20261028, 20261029, 20261030, 20261031]
PAIRS = [('C0', 'Shared'), ('Shared', 'Family'), ('Family', 'Exact'), ('Shared', 'Exact')]


def write_rows(path, rows):
    if not rows: raise ValueError(path)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def interval(contrast, groups, draws=50000, seed=20261027):
    """Resample whole groups from fixed held-out losses; descriptive only."""
    labels, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=contrast); sizes = np.bincount(inv)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(labels), size=(draws, len(labels)))
    means = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    return np.percentile(means, [2.5, 97.5])


def summarize(losses, policies, groups, evaluation):
    result = []
    for a, b in PAIRS:
        i, j = policies.index(a), policies.index(b)
        d = (losses[:, :, i] - losses[:, :, j]).mean(axis=1)
        lo, hi = interval(d, groups)
        result.append({'evaluation': evaluation, 'contrast': f'{a} minus {b}',
                       'mean_A': float(losses[:, :, i].mean()),
                       'mean_B': float(losses[:, :, j].mean()), 'mean_difference': float(d.mean()),
                       'bootstrap_2.5pct': float(lo), 'bootstrap_97.5pct': float(hi),
                       'n_records': len(d), 'n_resampled_groups': len(np.unique(groups))})
    return result


def compare_selection(cv_losses, filter_losses, groups, evaluation):
    """Same outer folds and same per-block measurement count, paired by row."""
    result = []
    for j, policy in enumerate(['C0','Shared','Family','Exact']):
        if policy == 'C0': continue
        d = (cv_losses[:,:,j] - filter_losses[:,:,j]).mean(axis=1)
        lo, hi = interval(d, groups)
        result.append({'evaluation':evaluation, 'policy':policy,
                       'inner_cv_risk':float(cv_losses[:,:,j].mean()),
                       'correlation_filter_risk':float(filter_losses[:,:,j].mean()),
                       'cv_minus_filter':float(d.mean()), 'bootstrap_2.5pct':float(lo),
                       'bootstrap_97.5pct':float(hi), 'n_records':len(d),
                       'n_resampled_groups':len(np.unique(groups))})
    return result


def extract_tables(result, evaluation, features, groups, source_ids):
    selection_rows = []; row_loss = []
    for f in result['selections']:
        for policy, actions in f['choices'].items():
            for t, name in enumerate(TARGETS):
                selection_rows.append({'evaluation':evaluation, 'fold':f['fold'], 'policy':policy,
                                       'target':name, 'family':FAMILIES[t],
                                       'selected_feature':features[int(actions[t])],
                                       'selected_feature_index_0based':int(actions[t])})
    for i in range(len(groups)):
        for t, name in enumerate(TARGETS):
            row_loss.append({'evaluation':evaluation, 'row_id':i+1, 'cell_line':source_ids[i], 'split_group':int(groups[i]),
                             'fold':int(result['fold_id'][i]), 'target':name,
                             **{f'loss_{policy}':float(result['losses'][i,t,j])
                                for j,policy in enumerate(result['policies'])}})
    return row_loss, selection_rows


def main():
    data_path = ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_readable.csv'
    out = ROOT/'results'/'gdsc'; out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    assert frame.shape == (499,857) and frame.columns[:7].tolist() == TARGETS
    y = frame.iloc[:, :7].to_numpy(dtype=float)
    tissues = frame.iloc[:, 7:20].to_numpy(dtype=float)
    assert np.all(np.isin(tissues, [0,1])) and np.all(tissues.sum(axis=1) == 1)
    tissue_group = np.argmax(tissues, axis=1)
    x = frame.iloc[:, 20:].to_numpy(dtype=float); features = frame.columns[20:].tolist()
    assert x.shape[1] == 837 and np.isfinite(x).all() and np.isfinite(y).all()
    map_path = ROOT/'inputs'/'GDSC'/'exampleGDSC_2.3-1_row_to_source_cell_line.csv'
    mapping = pd.read_csv(map_path, dtype={'cosmic_id_if_unique':str})
    assert mapping['package_row_1based'].tolist() == list(range(1,500))
    assert mapping['gdsc_cell_line_name'].is_unique and mapping['cosmic_id_if_unique'].is_unique
    assert mapping['tissue'].tolist() == [frame.columns[7:20][k] for k in tissue_group]
    assert (mapping['max_abs_error_on_343_log_expression_features'] <= 1e-12).all()
    assert mapping['tissue_indicators_equal'].all()
    source_ids = mapping['gdsc_cell_line_name'].tolist()
    random_results = []; random_filter_results = []; all_rows = []; all_choices = []; stats = []; method_stats = []
    for seed in SEEDS:
        result = nested_evaluate(x, y, FAMILIES, np.arange(len(x)), outer_splits=5, inner_splits=5, seed=seed)
        name = f'random_row_cv_seed_{seed}'; random_results.append(result['losses'])
        a,b = extract_tables(result, name, features, np.arange(len(x)), source_ids); all_rows += a; all_choices += b
        stats += summarize(result['losses'], result['policies'], np.arange(len(x)), name)
        baseline = nested_evaluate(x, y, FAMILIES, np.arange(len(x)), outer_splits=5, inner_splits=5, seed=seed, selection='in_sample_r2')
        assert np.array_equal(result['fold_id'], baseline['fold_id'])
        random_filter_results.append(baseline['losses'])
        a,b = extract_tables(baseline, name+'_correlation_filter', features, np.arange(len(x)), source_ids); all_rows += a; all_choices += b
        method_stats += compare_selection(result['losses'], baseline['losses'], np.arange(len(x)), name)
        print(name, 'complete', flush=True)
    avg_losses = np.mean(random_results, axis=0)
    stats += summarize(avg_losses, ['C0','Shared','Family','Exact'], np.arange(len(x)), 'random_5_splits_row_average')
    method_stats += compare_selection(avg_losses, np.mean(random_filter_results, axis=0), np.arange(len(x)), 'random_5_splits_row_average')
    tissue_result = nested_evaluate(x, y, FAMILIES, tissue_group, outer_splits=13, inner_splits=5, seed=SEEDS[0])
    a,b = extract_tables(tissue_result, 'leave_one_tissue_out', features, tissue_group, source_ids); all_rows += a; all_choices += b
    stats += summarize(tissue_result['losses'], tissue_result['policies'], tissue_group, 'leave_one_tissue_out')
    tissue_filter = nested_evaluate(x, y, FAMILIES, tissue_group, outer_splits=13, inner_splits=5, seed=SEEDS[0], selection='in_sample_r2')
    assert np.array_equal(tissue_result['fold_id'], tissue_filter['fold_id'])
    a,b = extract_tables(tissue_filter, 'leave_one_tissue_out_correlation_filter', features, tissue_group, source_ids); all_rows += a; all_choices += b
    method_stats += compare_selection(tissue_result['losses'], tissue_filter['losses'], tissue_group, 'leave_one_tissue_out')
    write_rows(out/'contrast_summary.csv', stats)
    write_rows(out/'feature_filter_comparison.csv', method_stats)
    write_rows(out/'held_out_losses.csv', all_rows)
    write_rows(out/'fold_action_selections.csv', all_choices)
    metadata = {'dataset':'BayesSUR 2.3-1 exampleGDSC, processed GDSC release 5',
        'source_url':'https://raw.githubusercontent.com/cran/BayesSUR/2.3-1/data/exampleGDSC.rda',
        'rda_sha256':'7b812ae9267322cc57997dc09821c228fb5610f59e1d88b33865ab8025b9163f',
        'csv_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'id_map_sha256':hashlib.sha256(map_path.read_bytes()).hexdigest(),
        'source_expression_url':'https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-5.0/gdsc_en_input_w5.csv',
        'source_expression_sha256':'89e97de69e776668f94156fdd4163eb83be4161c63775d8dcebab6579126ff0e',
        'source_cell_list_sha256':'eb8445b649cecd35d4af3a6e1dd87c902ee986694c05988da405535dd4116604',
        'reference_doi':'10.18637/jss.v100.i11',
        'n_rows':len(x), 'n_features':len(features), 'tissue_labels':frame.columns[7:20].tolist(),
        'tissue_counts':tissues.sum(axis=0).astype(int).tolist(), 'targets':TARGETS, 'family_labels':FAMILIES,
        'random_seeds':SEEDS, 'outer_random_folds':5, 'outer_tissue_folds':13, 'inner_folds':5,
        'policy':'one feature per Shared / each prespecified target family / each target; fixed OLS with intercept',
        'feature_selection_comparator':'outer-training squared-correlation filter, aggregated over targets in each block; fitted and scored on the same outer folds; post-hoc exploratory',
        'risk':'mean squared error in outer-training-standardized response coordinates, equal weight per row and target',
        'uncertainty':'50000 resamples of frozen held-out losses; tissue clusters for tissue CV; descriptive only',
        'limitations':['post-hoc exploratory application','released example lacked IDs; 499 IDs recovered by matching all 343 expression features and tissue to the source; 499 unique COSMIC IDs',
                       'unique cell-line names do not establish unrelated donors or independent biological lineages',
                       'global complete-case and feature filtering, and upstream GDSC preprocessing occurred before downstream splits',
                       'rows from the same tissue need not be independent','no prospective acquisition or equal total measurement budget',
                       'CV loss resampling does not repeat policy selection or model fitting',
                       'the seven drugs and family grouping were selected in the source study']}
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('leave_one_tissue_out complete', flush=True)


if __name__ == '__main__': main()
