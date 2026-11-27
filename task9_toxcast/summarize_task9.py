#!/usr/bin/env python3
"""Task 9 summary (plan_toxcast.md). Run with .env312 after run_all_allonly.sh and per_target9.py.
Only n = all is summarised (approved deviation; see DEVIATIONS.md). Writes summary/*.csv and summary/task9_endpoints.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OUT = HERE / 'summary'
FIXED = ['Shared', 'Family', 'Global3', 'Exact']
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
COLS = (['C0'] + FIXED + ['R4', 'Ridge', 'MultiTaskLasso', 'Lasso', 'ElasticNet', 'g_MTLrelaxed_matched']
        + ['g_Ridge', 'g_MultiTaskLasso', 'g_Lasso'])
TASKS = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
ASSAY_PRICE_USD = 36.0        # EPA phase I, 2008-2009 vintage; order of magnitude only (plan section 6)


def joined(name):
    v = pd.read_csv(HERE / 'validation' / f'{name}.csv')
    s = pd.read_csv(HERE / 'standard' / f'{name}.csv')
    b = pd.read_csv(HERE / 'budget' / f'{name}.csv')
    for f in (v, s, b):
        f['n_label'] = f.n_label.astype(str)
    assert (v[KEYS].to_numpy() == s[KEYS].to_numpy()).all() and (v[KEYS].to_numpy() == b[KEYS].to_numpy()).all()
    return pd.concat([v, s[[c for c in s if c not in v]], b[[c for c in b if c not in v and c not in s]]], axis=1)


def aid_of_endpoint():
    """aeid -> parent assay id, read straight from the annotation table the plan froze."""
    a = pd.read_excel(HERE.parent / 'external_data' / 'toxcast_v4_3' / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx',
                      usecols=['aeid', 'aid'])
    return dict(zip(a.aeid.astype(int), a.aid.astype(int)))


def assay_counts(aid_of, feats_per_subsample):
    """Plan section 6: cost is the number of distinct parent assays (aid), not endpoints."""
    n_assays = [len({aid_of[int(f.removeprefix('cand_'))] for f in fs}) for fs in feats_per_subsample]
    return float(np.mean(n_assays)), n_assays


def main():
    OUT.mkdir(exist_ok=True)
    aid_of = aid_of_endpoint()
    rows, per_target_all = [], []
    for t in TASKS:
        name = t['task']
        f = joined(name)
        h = f[f.n_label == 'all']
        assert len(h) == 25, f'{name}: expected 25 full-size subsamples, got {len(h)}'
        risk = {k: h[k].sum() / h.cells.sum() for k in COLS}
        ev = {k: 1 - risk[k] / risk['C0'] for k in COLS}
        best_fixed = min(risk[k] for k in FIXED)
        feats = json.loads((HERE / 'per_target' / f'{name}_r4_features.json').read_text(encoding='utf-8'))
        mean_assays, _ = assay_counts(aid_of, feats)
        n_cand_endpoints = len(t['candidate_aeids'])
        n_cand_assays = len({aid_of[a] for a in t['candidate_aeids']})
        ev4, evr = ev['R4'], ev['Ridge']
        ratio_ok = (ev4 >= 0.5 * evr) if evr > 0 else (ev4 >= evr)
        # The unguarded multi-measurement methods collapse numerically here (candidate endpoints that are nearly
        # constant inside an outer training fold), so the criterion against plain Ridge passes for the wrong reason.
        # The guarded Ridge is the meaningful comparison and is the guard ablation the plan already asked for
        # (section 8); the criterion is reported against both. Both give the same verdict.
        evr_g = ev['g_Ridge']
        ratio_ok_g = (ev4 >= 0.5 * evr_g) if evr_g > 0 else (ev4 >= evr_g)
        frac = mean_assays / n_cand_assays
        rows.append({'task': name, 'families': '|'.join(dict.fromkeys(t['families'])),
                     'units': int(h.n_train.iloc[0] + h.cells.iloc[0] / len(t['targets'])),
                     'cand_endpoints': n_cand_endpoints, 'cand_assays': n_cand_assays,
                     'ev_R4': ev4, 'ev_Ridge': evr, 'ev_MTL': ev['MultiTaskLasso'], 'ev_Lasso': ev['Lasso'],
                     'ev_Ridge_guarded': evr_g, 'ev_MTL_guarded': ev['g_MultiTaskLasso'],
                     'ev_matched_relaxed_MTL': ev['g_MTLrelaxed_matched'],
                     'ev_best_fixed': 1 - best_fixed / risk['C0'],
                     'k_R4_endpoints': float(h.k_R4.mean()), 'k_R4_assays': mean_assays,
                     'k_MTL_endpoints': float(h.nmeas_MultiTaskLasso.mean()),
                     'ev_ratio': ev4 / evr if evr > 0 else np.nan,
                     'ev_ratio_guarded': ev4 / evr_g if evr_g > 0 else np.nan, 'assay_fraction': frac,
                     'criterion_ev': bool(ratio_ok), 'criterion_ev_guarded': bool(ratio_ok_g),
                     'criterion_cost': bool(mean_assays <= 0.1 * n_cand_assays),
                     'value_met': bool(ratio_ok and mean_assays <= 0.1 * n_cand_assays),
                     'value_met_guarded': bool(ratio_ok_g and mean_assays <= 0.1 * n_cand_assays),
                     'regret_R4': risk['R4'] - best_fixed,
                     'regret_matched_MTL': risk['g_MTLrelaxed_matched'] - best_fixed,
                     'choice_R4_modal': h.choice_R4.mode().iloc[0],
                     'usd_R4': mean_assays * ASSAY_PRICE_USD, 'usd_all': n_cand_assays * ASSAY_PRICE_USD})
        pt = pd.read_csv(HERE / 'per_target' / f'{name}.csv'); pt['task'] = name
        per_target_all.append(pt)

    crit = pd.DataFrame(rows)
    crit.to_csv(OUT / 'task9_criterion.csv', index=False)
    pd.concat(per_target_all).to_csv(OUT / 'task9_per_target.csv', index=False)

    d = crit.regret_R4 - crit.regret_matched_MTL
    d = d[np.abs(d) > 1e-12]
    wins = int((d < 0).sum())
    sign = {'wins_R4': wins, 'losses_R4': int((d > 0).sum()),
            'p_two_sided': float(binomtest(wins, len(d), 0.5).pvalue) if len(d) else None}
    needed = int(np.ceil(2 / 3 * len(crit)))
    endpoints = {'tasks': len(crit), 'tasks_value_met': int(crit.value_met.sum()), 'needed': needed,
                 'value_established': bool(crit.value_met.sum() >= needed),
                 'tasks_value_met_guarded': int(crit.value_met_guarded.sum()),
                 'value_established_guarded': bool(crit.value_met_guarded.sum() >= needed),
                 'tasks_failing_only_on_cost': int((crit.criterion_ev_guarded & ~crit.criterion_cost).sum()),
                 'mean_ev_R4': float(crit.ev_R4.mean()), 'mean_ev_Ridge': float(crit.ev_Ridge.mean()),
                 'mean_ev_Ridge_guarded': float(crit.ev_Ridge_guarded.mean()),
                 'mean_ev_ratio_guarded': float(crit.ev_ratio_guarded.mean()),
                 'mean_assay_fraction': float(crit.assay_fraction.mean()),
                 'mean_k_R4_assays': float(crit.k_R4_assays.mean()), 'mean_cand_assays': float(crit.cand_assays.mean()),
                 'R4_vs_matched_relaxed_MTL_sign_test': sign,
                 'usd_note': 'EPA phase I 2008-2009 vintage, order of magnitude only',
                 'sizes_reported': 'n = all only (approved deviation, DEVIATIONS.md)'}
    (OUT / 'task9_endpoints.json').write_text(json.dumps(endpoints, indent=1), encoding='utf-8')
    print(crit.round(4).to_string())
    print(json.dumps(endpoints, indent=1))


if __name__ == '__main__':
    main()
