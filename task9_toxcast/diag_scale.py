"""How large do the single-feature risks get at each training size, and does the frozen Global3 check survive?"""
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
print('candidate sd on all rows: min %.2e, 5th pct %.2e, median %.2e' % (
    x.std(0, ddof=1).min(), np.percentile(x.std(0, ddof=1), 5), np.median(x.std(0, ddof=1))))
print('candidates with sd < 0.01:', int((x.std(0, ddof=1) < 0.01).sum()), 'of', x.shape[1])
for seed in L.SEEDS[:2]:
    folds = group_folds(groups, 5, seed)
    for fold in [0, 1]:
        held = folds[fold]
        train_idx = np.setdiff1d(groups, held)
        for n in [25, 100, 1000, None]:
            reps = range(3) if n else [0]
            worst, fails = 0.0, 0
            for r in reps:
                sub = train_idx if n is None else G.draw(
                    train_idx, n, groups, np.random.default_rng([20261029, task['code'], seed, fold, n, r]), 3)[0]
                Lp, c0 = L.unit_losses(x[sub], y[sub], groups[sub], 5, seed + fold + 1)
                risk = Lp.mean(axis=0)
                worst = max(worst, float(risk.max()))
                try:
                    EB.globally_optimal_actions(risk, 3)
                except RuntimeError:
                    fails += 1
            print(f'seed {seed} fold {fold} n={n if n else "all"}: max risk {worst:.2e}, '
                  f'C0 risk {float(c0.mean()):.4f}, Global3 check failures {fails}/{len(list(reps))}', flush=True)
