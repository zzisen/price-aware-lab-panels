"""Guarded vs unguarded multi-measurement methods at n = all (plan section 8 asks for the guard ablation)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')):
    n = t['task']
    v = pd.read_csv(HERE / 'validation' / f'{n}.csv'); s = pd.read_csv(HERE / 'standard' / f'{n}.csv')
    b = pd.read_csv(HERE / 'budget' / f'{n}.csv')
    f = pd.concat([v, s[[c for c in s if c not in v]], b[[c for c in b if c not in v and c not in s]]], axis=1)
    c0 = f.C0.sum() / f.cells.sum()
    ev = lambda k: 1 - (f[k].sum() / f.cells.sum()) / c0
    print(n, 'guard_fraction %.4f' % f.guard_fraction.mean())
    print('   R4 %.3f | Shared %.3f Family %.3f Global3 %.3f Exact %.3f'
          % (ev('R4'), ev('Shared'), ev('Family'), ev('Global3'), ev('Exact')))
    print('   plain   Ridge %12.2e MTL %12.2e Lasso %12.2e ElasticNet %12.2e'
          % (ev('Ridge'), ev('MultiTaskLasso'), ev('Lasso'), ev('ElasticNet')))
    print('   guarded Ridge %12.3f MTL %12.3f Lasso %12.3f ElasticNet %12.3f'
          % (ev('g_Ridge'), ev('g_MultiTaskLasso'), ev('g_Lasso'), ev('g_ElasticNet')))
    print('   guarded fixed: Shared %.3f Family %.3f Global3 %.3f Exact %.3f | matched relaxed MTL %.3f'
          % (ev('g_Shared'), ev('g_Family'), ev('g_Global3'), ev('g_Exact'), ev('g_MTLrelaxed_matched')))
