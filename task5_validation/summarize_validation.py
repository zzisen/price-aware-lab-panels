#!/usr/bin/env python3
"""Task 5: validation summary with the frozen endpoints (plan_validation_suite.md, plan_r4_frozen.md).
Run with .env312 after validate_generic.py (19 tasks) and validate_cardiac.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
V = HERE / 'validation'
FIXED = ['Shared', 'Family', 'Global3', 'Exact']
RULES = ['R1', 'R3', 'R4']


def generic(task):
    f = pd.read_csv(V / f'{task}.csv')
    f['n_label'] = f.n_label.astype(str)
    cols = ['C0'] + FIXED + [f'g_{k}' for k in FIXED] + RULES
    rows = []
    for n_label, h in f.groupby('n_label'):
        risk = {k: h[k].sum() / h.cells.sum() for k in cols}
        best = min(risk[k] for k in FIXED)
        rows.append({'task': task, 'n_label': n_label, 'n_mean': h.n_train.mean(), 'best_fixed': min(FIXED, key=risk.get),
                     **{f'risk_{k}': v for k, v in risk.items()}, **{f'regret_{k}': risk[k] - best for k in cols},
                     **{f'{r}_choice_share_{c}': float((h[f'choice_{r}'] == c).mean()) for r in RULES for c in ['C0'] + FIXED}})
    return pd.DataFrame(rows).sort_values('n_mean')


def native(system, g):
    if system == 'AWINDA_MOUSE':
        return float(g.groupby(['target', 'unit']).loss.mean().groupby('target').mean().mean())
    return float(g.loss.mean())


def cardiac(system):
    f = pd.read_csv(V / f'cardiac_{system}.csv')
    pols = [p for p in ['C0', 'SHARED', 'FAMILY', 'EXACT'] if p in set(f.policy)]
    risk = {p.title() if p != 'C0' else 'C0': native(system, f[f.policy == p]) for p in pols}
    for r in RULES:
        risk[r] = native(system, f[f[r]])
    fixed = [k for k in FIXED if k in risk]
    best = min(risk[k] for k in fixed)
    return pd.DataFrame([{'task': system, 'n_label': 'all', 'best_fixed': min(fixed, key=risk.get),
                          **{f'risk_{k}': v for k, v in risk.items()}, **{f'regret_{k}': v - best for k, v in risk.items()},
                          **{f'relative_{k}': v / best for k, v in risk.items()}}])


def main():
    tasks = [t for t in (HERE / 'validation_tasks.txt').read_text().split()]
    per_n = pd.concat([generic(t) for t in tasks] + [cardiac(s) for s in ['RLC1_RAT', 'TANNER_RAT_TRABECULA', 'AWINDA_MOUSE']])
    per_n.to_csv(HERE / 'validation_per_n.csv', index=False)
    mx = per_n.groupby('task', sort=False)[[c for c in per_n if c.startswith('regret_')]].max()
    mx.to_csv(HERE / 'validation_max_regret.csv')

    def sign(a, b):
        d = mx[f'regret_{a}'] - mx[f'regret_{b}']
        d = d[np.abs(d) > 1e-12]
        wins = int((d < 0).sum())
        return {'wins': wins, 'losses': int((d > 0).sum()), 'ties': int(len(mx) - len(d)),
                'p_two_sided': float(binomtest(wins, len(d), 0.5).pvalue) if len(d) else None}
    out = {'n_tasks': len(mx),
           'primary R4 vs R1': sign('R4', 'R1'),
           'R4 vs R3': sign('R4', 'R3'), 'R3 vs R1': sign('R3', 'R1'),
           'R4 vs Family (prespecified single fixed)': sign('R4', 'Family'),
           'R4 vs Shared': sign('R4', 'Shared'),
           'ablation R4 vs guarded fixed Global3': sign('R4', 'g_Global3') if 'regret_g_Global3' in mx else None,
           'mean max regret': mx.mean().round(4).to_dict(),
           'median max regret': mx.median().round(4).to_dict()}
    gen = per_n[per_n.n_label != 'all'].copy()
    out['mean regret by n (generic tasks)'] = gen.groupby('n_label')[[f'regret_{k}' for k in ['C0'] + FIXED + RULES]].mean().round(4).to_dict()
    (HERE / 'validation_endpoints.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(mx.round(3))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
