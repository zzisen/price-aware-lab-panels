"""POST HOC: how does the BEST sparsity level change with sample size? The frozen budget stage already ran
relaxed MTL at fixed budgets k=1, 3, 9, so the optimal k per training size can be read off directly.
If the best k grows with n, an adaptive complexity ladder (rather than always one measurement per target)
is the fix; if the best k stays at 1, R4's restriction is not what costs it the large-sample comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
W = HERE.parent
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']


def load(base, name):
    v = pd.read_csv(base / 'validation' / f'{name}.csv')
    b = pd.read_csv(base / 'budget' / f'{name}.csv')
    f = pd.concat([v, b[[c for c in b if c not in v]]], axis=1)
    f['n_label'] = f.n_label.astype(str)
    return f


def table(base, names, label):
    out = []
    for n_label in ['25', '50', '100', '200', '500', '1000', 'all']:
        acc = {}
        for name in names:
            f = load(base, name)
            h = f[f.n_label == n_label]
            if not len(h):
                continue
            c0 = h.C0.sum() / h.cells.sum()
            for k in ['k1', 'k3', 'k9']:
                col = f'g_MTLrelaxed_{k}'
                acc.setdefault(k, []).append(1 - (h[col].sum() / h.cells.sum()) / c0)
            acc.setdefault('R4', []).append(1 - (h.R4.sum() / h.cells.sum()) / c0)
            acc.setdefault('matched', []).append(1 - (h.g_MTLrelaxed_matched.sum() / h.cells.sum()) / c0)
            acc.setdefault('k_R4', []).append(h.k_R4.mean())
        if not acc:
            continue
        row = {'dataset': label, 'n': n_label, **{k: float(np.mean(v)) for k, v in acc.items()}}
        row['best_fixed_k'] = max(['k1', 'k3', 'k9'], key=lambda k: row[k])
        out.append(row)
    return out


rows = table(W / 'task8_real_cost', [t['task'] for t in json.loads((W / 'task8_real_cost' / 'tasks.json').read_text(encoding='utf-8')) if t['task'].startswith('DRUG_')], 'drug')
rows += table(W / 'task8_real_cost', ['NHANES'], 'nhanes')
rows += table(W / 'task9_toxcast', [t['task'] for t in json.loads((W / 'task9_toxcast' / 'tasks.json').read_text(encoding='utf-8'))], 'tox21')
d = pd.DataFrame(rows)
print('Explained variance of relaxed MTL at FIXED budgets, vs R4, by training size:')
print(d[['dataset', 'n', 'k_R4', 'R4', 'matched', 'k1', 'k3', 'k9', 'best_fixed_k']].round(3).to_string(index=False))
d.to_csv(HERE / 'optimal_k.csv', index=False)
