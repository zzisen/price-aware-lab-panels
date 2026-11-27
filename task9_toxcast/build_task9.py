#!/usr/bin/env python3
"""Task 9 (plan_toxcast.md): build the 6 Tox21 tasks in the Task 5 task format.
Writes tasks.json, task_counts.csv and matrices/<task>.csv into this folder. Run with .env312."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
D = WORK / 'external_data' / 'toxcast_v4_3'
GRID = [25, 50, 100, 200, 500, 1000, None]
DROP_FAMILIES = ['channel 1', 'channel 2', 'background measurement']
MIN_CHEMICALS = 6000
GROUP_SIZE = 3
# plan section 4: family group index per task, 3 distinct families each
TASK_GROUPS = [
    [('nuclear receptor', 0), ('dna binding', 0), ('gpcr', 0)],
    [('nuclear receptor', 1), ('dna binding', 1), ('gpcr', 1)],
    [('nuclear receptor', 2), ('dna binding', 2), ('gpcr', 2)],
    [('nuclear receptor', 3), ('dna binding', 3), ('cyp', 0)],
    [('nuclear receptor', 4), ('dna binding', 4), ('cyp', 1)],
    [('nuclear receptor', 5), ('dna binding', 5), ('esterase', 0)],
]


def endpoint_pool():
    """Plan section 3: the Tox21 endpoints kept, with their family and parent assay."""
    a = pd.read_excel(D / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx')
    t = a[a.assay_source_name == 'TOX21']
    keep = t[~(t.intended_target_family.isin(DROP_FAMILIES) | (t.cell_viability_assay == 1))]
    return keep.set_index('aeid')[['aid', 'intended_target_family', 'assay_component_endpoint_name']]


def wide_matrix(pool):
    """Chemicals x endpoints of median continuous hit call, complete cases only."""
    m = pd.read_csv(D / 'mc5_6_compact.csv.gz', usecols=['dsstox_substance_id', 'aeid', 'hitc'], low_memory=False)
    m = m[m.aeid.isin(pool.index) & (m.hitc != -1)].dropna(subset=['dsstox_substance_id', 'hitc'])
    w = m.groupby(['dsstox_substance_id', 'aeid']).hitc.median().unstack().clip(0.0, 1.0)
    w = w[w.columns[w.notna().sum() >= MIN_CHEMICALS]].dropna()
    return w


def family_groups(pool, columns):
    """Plan section 4: within each family, endpoints sorted by aeid, cut into disjoint groups of 3."""
    out = {}
    fam = pool.loc[columns, 'intended_target_family']
    for name, g in fam.groupby(fam):
        eps = sorted(g.index)
        out[name] = [eps[i:i + GROUP_SIZE] for i in range(0, len(eps) - GROUP_SIZE + 1, GROUP_SIZE)]
    return out


def main():
    (HERE / 'matrices').mkdir(exist_ok=True)
    pool = endpoint_pool()
    w = wide_matrix(pool)
    print('matrix', w.shape, flush=True)
    groups = family_groups(pool, list(w.columns))
    print('groups per family', {k: len(v) for k, v in sorted(groups.items())}, flush=True)

    tasks, counts = [], []
    for i, spec in enumerate(TASK_GROUPS):
        name = f'TOX21_T{i + 1}'
        targets, families = [], []
        for family, gi in spec:
            for aeid in groups[family][gi]:
                targets.append(aeid); families.append(family)
        target_aids = set(pool.loc[targets, 'aid'])
        cand = [a for a in w.columns if a not in targets and pool.at[a, 'aid'] not in target_aids]
        frame = pd.concat([w[cand].rename(columns=lambda a: f'cand_{a}'),
                           w[targets].rename(columns=lambda a: f'ep_{a}')], axis=1)
        frame.index.name = 'unit'
        frame.reset_index().to_csv(HERE / 'matrices' / f'{name}.csv', index=False)
        tasks.append({'task': name, 'targets': [f'ep_{a}' for a in targets], 'families': families,
                      'code': 931 + i, 'grid': GRID, 'cost_unit': 'assay',
                      'target_aeids': [int(a) for a in targets],
                      'candidate_aeids': [int(a) for a in cand],
                      'candidate_aids': sorted(int(x) for x in set(pool.loc[cand, 'aid']))})
        counts.append({'task': name, 'units': len(frame), 'candidates': len(cand),
                       'candidate_assays': len(set(pool.loc[cand, 'aid'])), 'targets': len(targets),
                       'target_families': '|'.join(dict.fromkeys(families)), 'included': True})
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1), encoding='utf-8')
    pd.DataFrame(counts).to_csv(HERE / 'task_counts.csv', index=False)
    print(pd.DataFrame(counts).to_string())
    names = pool.loc[[a for t in tasks for a in t['target_aeids']], 'assay_component_endpoint_name']
    names.to_csv(HERE / 'target_endpoint_names.csv')
    print('activity rate of target columns (descriptive, after freeze):')
    print(np.round((w[[a for t in tasks for a in t['target_aeids']]] >= 0.9).mean(), 3).to_string())


if __name__ == '__main__':
    main()
