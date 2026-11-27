#!/usr/bin/env python3
"""CiPA-28 addendum (plan_cipa28.md, frozen 2026-11-01): build CIPA28, its 12-training-drug sensitivity task and the
row-permuted copies, then run the frozen Task 5/6/7 pipelines unchanged. Run with .env312:
  python cipa28.py build | run <task> | perm <k0> <k1> | summary"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T8 = HERE.parent
EXT = T8.parent / 'external_data' / 'cipa'
sys.path.insert(0, str(T8)); sys.path.insert(0, str(T8.parent / 'task5_validation'))
from build_task8 import log_ic50  # noqa: E402
import rulelib as L  # noqa: E402

CHANNELS = ['hERG', 'ICaL', 'INa', 'INaL']
TRAINING = ['bepridil', 'chlorpromazine', 'cisapride', 'diltiazem', 'dofetilide', 'mexiletine', 'ondansetron',
            'quinidine', 'ranolazine', 'sotalol', 'terfenadine', 'verapamil']
N_PERM, PERM_SEED = 200, 20261101


def build():
    b = pd.read_csv(EXT / 'fda_cipa_branches' / 'Lab_Specific_Validation_Calibration_2020_chantest_Hill_fitting_data_drug_block.csv')
    b = b[b.channel.isin(CHANNELS)].dropna(subset=['block'])
    x = b.groupby(['drug', 'channel']).apply(log_ic50, include_groups=False).unstack()[CHANNELS]
    x.columns = [f'log10IC50_{c}' for c in x.columns]
    d = pd.read_excel(EXT / 'blinova2018' / 'Blinova_etal_2018_data.xlsx')
    d['drug'] = d.Drug_Name.str.lower().str.replace(r'^d,\s*l[ ,]*', '', regex=True).str.strip()
    y = d.groupby(['drug', 'Cell_type', 'conc']).ddFPDc.median().unstack(['Cell_type', 'conc'])
    y.columns = [f'ddFPDc_{ct}_c{c}' for ct, c in y.columns]
    f = x.join(y, how='inner')
    targets = sorted(c for c in y.columns if f[c].notna().all())
    f = f[list(x.columns) + targets]
    print('drugs', len(f), 'targets', targets, 'missing features', int(f[list(x.columns)].isna().sum().sum()))
    f.index.name = 'unit'
    fam = [t.split('_')[1] for t in targets]
    base = {'targets': targets, 'families': fam, 'design': 'leave_one_out', 'cost_unit': 'channel'}
    tasks = [{'task': 'CIPA28', 'code': 921, **base}, {'task': 'CIPA28_T12', 'code': 922, **base}]
    f.reset_index().to_csv(HERE / 'matrices' / 'CIPA28.csv', index=False)
    f.loc[TRAINING].reset_index().to_csv(HERE / 'matrices' / 'CIPA28_T12.csv', index=False)
    rng = np.random.default_rng(PERM_SEED)
    for k in range(N_PERM):
        g = f.copy(); g[targets] = f[targets].to_numpy()[rng.permutation(len(f))]
        g.reset_index().to_csv(HERE / 'matrices' / f'CIPA28_P{k:03d}.csv', index=False)
        tasks.append({'task': f'CIPA28_P{k:03d}', 'code': 921, **base})
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1), encoding='utf-8')
    f.to_csv(HERE / 'cipa28_matrix_view.csv')


def run(name, full=True):
    import validate_generic, validate_standard, validate_budget
    L.HERE = HERE
    t0 = time.time()
    validate_generic.main(name)
    if full:
        validate_standard.main(name); validate_budget.main(name)
    return time.time() - t0


def ev_r4(name):
    v = pd.read_csv(HERE / 'validation' / f'{name}.csv')
    return 1 - v.R4.sum() / v.C0.sum()


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build':
        build()
    elif cmd == 'run':
        print(sys.argv[2], 'seconds', run(sys.argv[2]))
    elif cmd == 'perm':
        for k in range(int(sys.argv[2]), int(sys.argv[3])):
            print(k, 'seconds', run(f'CIPA28_P{k:03d}', full=False), flush=True)
