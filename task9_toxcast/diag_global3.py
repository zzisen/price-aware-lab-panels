"""Diagnose why the frozen Global3 self-check fails on TOX21_T1."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L
L.HERE = HERE
import gdsc_learning_curve as G
from target_resolution import group_folds
import exact_budget as EB

task = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))[0]
L.GRID = task['grid']
task, x, y, fam = L.load_task(task['task'])
groups = np.arange(len(x))
held = group_folds(groups, 5, L.SEEDS[0])[0]
train_idx = np.setdiff1d(groups, held)
sub = G.draw(train_idx, 25, groups, np.random.default_rng([20261029, task['code'], L.SEEDS[0], 0, 25, 0]), 3)[0]
Lp, _ = L.unit_losses(x[sub], y[sub], groups[sub], 5, L.SEEDS[0] + 1)
risk = Lp.mean(axis=0)
print('risk shape', risk.shape)
print('risk max', float(risk.max()), 'min', float(risk.min()))
print('risk column max per target', np.round(risk.max(axis=0), 3).tolist())
bad = np.argsort(-risk.max(axis=1))[:5]
print('candidates with largest risk (index, max risk, train sd):')
for i in bad:
    print('  ', int(i), f'{risk[i].max():.3e}', f'sd={x[sub][:, i].std(ddof=1):.3e}')
try:
    EB.globally_optimal_actions(risk, 3)
    print('frozen solver OK')
except RuntimeError as e:
    import math
    parts = EB.target_partitions(risk.shape[1], 3)
    sa, sc = {}, {}
    for mask in range(1, 1 << risk.shape[1]):
        members = [i for i in range(risk.shape[1]) if (mask >> i) & 1]
        costs = risk[:, members].sum(axis=1)
        c = int(np.argmin(costs)); sa[mask] = c; sc[mask] = float(costs[c])
    costs = np.array([math.fsum(sc[m] for m in p) for p in parts])
    bi = int(np.argmin(costs)); best = parts[bi]
    act = np.empty(risk.shape[1], int)
    for mask in best:
        for t in range(risk.shape[1]):
            if (mask >> t) & 1: act[t] = sa[mask]
    obj = float(risk[act, np.arange(risk.shape[1])].mean()); low = float(costs[bi] / risk.shape[1])
    print('FAILS:', e)
    print(f'objective {obj!r}  lower {low!r}  abs diff {abs(obj-low):.3e}  distinct {len(np.unique(act))}')
