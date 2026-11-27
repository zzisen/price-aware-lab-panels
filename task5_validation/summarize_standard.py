#!/usr/bin/env python3
"""Task 6 summary (plan_standard_methods.md): R4 versus Lasso, Elastic net, Multi-task lasso and Ridge on the 19 generic
validation tasks. Pairs standard/<task>.csv with validation/<task>.csv row by row. Run with .env312."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import rulelib as L

FIXED = ['Shared', 'Family', 'Global3', 'Exact']
STD = ['Lasso', 'ElasticNet', 'MultiTaskLasso', 'Ridge']
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']


def r4_measurements(choice, n_targets, n_families):
    return {'C0': 0, 'Shared': 1, 'Family': n_families, 'Global3': 3, 'Exact': n_targets}[choice]


def task_rows(task):
    v = pd.read_csv(L.HERE / 'validation' / f'{task}.csv'); s = pd.read_csv(L.HERE / 'standard' / f'{task}.csv')
    v['n_label'] = v.n_label.astype(str); s['n_label'] = s.n_label.astype(str)
    assert (v[KEYS].to_numpy() == s[KEYS].to_numpy()).all() and (v.cells == s.cells).all()
    t, _, y, fam = L.load_task(task)
    f = pd.concat([v, s[[c for c in s if c not in v]]], axis=1)
    f['nmeas_R4'] = [min(r4_measurements(c, y.shape[1], len(set(fam))), y.shape[1]) for c in f.choice_R4]
    cols = FIXED + ['R1', 'R3', 'R4'] + STD + [f'g_{k}' for k in STD]
    rows = []
    for n_label, h in f.groupby('n_label'):
        risk = {k: h[k].sum() / h.cells.sum() for k in cols}
        best = min(risk[k] for k in FIXED)
        rows.append({'task': task, 'n_label': n_label, 'n_mean': h.n_train.mean(), **{f'risk_{k}': v for k, v in risk.items()},
                     **{f'regret_{k}': risk[k] - best for k in cols},
                     **{f'nmeas_{k}': h[f'nmeas_{k}'].mean() for k in STD + ['R4']}})
    return rows


def main():
    tasks = (L.HERE / 'validation_tasks.txt').read_text().split()
    per_n = pd.DataFrame([r for t in tasks for r in task_rows(t)])
    per_n.to_csv(L.HERE / 'standard_per_n.csv', index=False)
    mx = per_n.groupby('task', sort=False)[[c for c in per_n if c.startswith('regret_')]].max()
    mx.to_csv(L.HERE / 'standard_max_regret.csv')

    def sign(a, b):
        d = mx[f'regret_{a}'] - mx[f'regret_{b}']; d = d[np.abs(d) > 1e-12]
        w = int((d < 0).sum())
        return {'wins': w, 'losses': int((d > 0).sum()), 'ties': int(len(mx) - len(d)), 'p_two_sided': float(binomtest(w, len(d), 0.5).pvalue)}
    mean = mx.mean()
    out = {'n_tasks': len(mx),
           'primary: R4 vs plain': {k: sign('R4', k) for k in STD},
           'secondary: R4 vs guarded': {f'g_{k}': sign('R4', f'g_{k}') for k in STD},
           'mean max regret': mean.round(4).to_dict(), 'median max regret': mx.median().round(4).to_dict()}
    out['clear advantage'] = bool(all(mean['regret_R4'] < mean[f'regret_{k}'] for k in STD)
                                  and all(out['primary: R4 vs plain'][k]['p_two_sided'] < 0.05 and
                                          out['primary: R4 vs plain'][k]['wins'] > out['primary: R4 vs plain'][k]['losses']
                                          for k in ['Lasso', 'ElasticNet', 'MultiTaskLasso']))
    out['mean regret by n'] = per_n.groupby('n_label')[[f'regret_{k}' for k in ['R4'] + STD + [f'g_{k}' for k in STD]]].mean().round(4).to_dict()
    out['mean measurements by n'] = per_n.groupby('n_label')[[f'nmeas_{k}' for k in ['R4'] + STD]].mean().round(1).to_dict()
    (L.HERE / 'standard_endpoints.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(mx[[f'regret_{k}' for k in ['R1', 'R4'] + STD + [f'g_{k}' for k in STD]]].round(3))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
