#!/usr/bin/env python3
"""Task 7 summary (plan_budget_matched.md): R4 versus budget-matched multi-task lasso on the 19 generic validation
tasks. Pairs budget/<task>.csv with validation/<task>.csv row by row. Run with .env312."""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import rulelib as L

FIXED = ['Shared', 'Family', 'Global3', 'Exact']
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
BUD = ['matched', 'k1', 'k3', 'k9']
COMP = [f'{g}{v}_{b}' for b in BUD for v in ['MTL', 'MTLrelaxed'] for g in ['', 'g_']]


def task_rows(task):
    v = pd.read_csv(L.HERE / 'validation' / f'{task}.csv'); s = pd.read_csv(L.HERE / 'budget' / f'{task}.csv')
    v['n_label'] = v.n_label.astype(str); s['n_label'] = s.n_label.astype(str)
    assert (v[KEYS].to_numpy() == s[KEYS].to_numpy()).all() and (v.cells == s.cells).all()
    f = pd.concat([v, s[[c for c in s if c not in v]]], axis=1)
    cols = FIXED + ['R4', 'g_Shared', 'g_Global3', 'g_Exact'] + COMP
    rows = []
    for n_label, h in f.groupby('n_label'):
        risk = {k: h[k].sum() / h.cells.sum() for k in cols}
        best = min(risk[k] for k in FIXED)
        rows.append({'task': task, 'n_label': n_label, 'n_mean': h.n_train.mean(), 'k_R4': h.k_R4.mean(),
                     **{f'risk_{k}': x for k, x in risk.items()}, **{f'regret_{k}': risk[k] - best for k in cols},
                     **{f'nmeas_{v}_{b}': h[f'nmeas_{v}_{b}'].mean() for b in BUD for v in ['MTL', 'MTLrelaxed']}})
    return rows


def main():
    tasks = (L.HERE / 'validation_tasks.txt').read_text().split()
    per_n = pd.DataFrame([r for t in tasks for r in task_rows(t)])
    per_n.to_csv(L.HERE / 'budget_per_n.csv', index=False)
    mx = per_n.groupby('task', sort=False)[[c for c in per_n if c.startswith('regret_')]].max()
    mx.to_csv(L.HERE / 'budget_max_regret.csv')

    def sign(a, b):
        """a versus b: wins = tasks where a has lower max regret."""
        d = mx[f'regret_{a}'] - mx[f'regret_{b}']; d = d[np.abs(d) > 1e-12]
        w = int((d < 0).sum())
        return {'wins': w, 'losses': int((d > 0).sum()), 'ties': int(len(mx) - len(d)),
                'p_two_sided': float(binomtest(w, len(d), 0.5).pvalue) if len(d) else None}
    prim = {c: sign('R4', c) for c in ['g_MTL_matched', 'g_MTLrelaxed_matched']}
    need = any(p['losses'] > p['wins'] and p['p_two_sided'] is not None and p['p_two_sided'] < 0.05 for p in prim.values())
    out = {'n_tasks': len(mx), 'primary: R4 vs guarded matched': prim, 'R5A needed': need,
           'secondary: R4 vs plain matched': {c: sign('R4', c) for c in ['MTL_matched', 'MTLrelaxed_matched']},
           'secondary: fixed budgets (R4 resolution vs guarded comparator)': {
               f'{a} vs {c}': sign(a, c) for a, k in [('g_Shared', 'k1'), ('g_Global3', 'k3'), ('g_Exact', 'k9')]
               for c in [f'g_MTL_{k}', f'g_MTLrelaxed_{k}']},
           'mean max regret': mx.mean().round(4).to_dict(), 'median max regret': mx.median().round(4).to_dict(),
           'mean regret by n': per_n.groupby('n_label')[[f'regret_{k}' for k in ['R4', 'g_MTL_matched', 'g_MTLrelaxed_matched']]].mean().round(4).to_dict(),
           'mean measurements by n': per_n.groupby('n_label')[['k_R4', 'nmeas_MTL_matched', 'nmeas_MTLrelaxed_matched']].mean().round(2).to_dict()}
    (L.HERE / 'budget_endpoints.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(mx[[f'regret_{k}' for k in ['R4', 'g_MTL_matched', 'g_MTLrelaxed_matched', 'MTL_matched', 'MTLrelaxed_matched']]].round(3))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
