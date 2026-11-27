"""Timing probe (plan section 5): cost of one subsample at the largest and a middle size, using the frozen functions."""
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L
L.HERE = HERE
import validate_generic as VG
import validate_standard as VS
from target_resolution import action_losses, group_folds

task = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))[0]
L.GRID = task['grid']
task, x, y, fam = L.load_task(task['task'])
print('x', x.shape, 'y', y.shape, flush=True)
subs = list(VG.subsamples(task, x))
print('total subsamples', len(subs), 'sizes', sorted({len(s[5]) for s in subs}), flush=True)

import gdsc_learning_curve as G
groups = np.arange(len(x))
folds = group_folds(groups, 5, L.SEEDS[0])
held = folds[0]
train_idx = np.setdiff1d(groups, held)
for n in [None]:
    sub = train_idx if n is None else G.draw(train_idx, n, groups, np.random.default_rng([20261029, task['code'], L.SEEDS[0], 0, n, 0]), 3)[0]
    inner = L.SEEDS[0] + 1
    t0 = time.time()
    Lp, _ = L.unit_losses(x[sub], y[sub], groups[sub], 5, inner)
    ch = L.choices(Lp.mean(axis=0), fam)
    cand, c0 = action_losses(x[sub], y[sub][:, :, None], x[held], y[held, :, None])
    mid = L.middle_rows(x[sub], y[sub], groups[sub], fam, inner, L.MIDDLE_OFFSETS[0])
    t1 = time.time()
    g = L.cache_subsample_g(x, y, groups, fam, sub, held, inner)
    t2 = time.time()
    print(f'n={n if n else "all"} ({len(sub)} rows): plain {t1-t0:.1f}s  guarded {t2-t1:.1f}s', flush=True)
    xs, ys = x[sub], y[sub]
    xm, xsd = xs.mean(0), np.where(xs.std(0, ddof=1) == 0, 1.0, xs.std(0, ddof=1))
    ym, ysd = ys.mean(0), np.where(ys.std(0, ddof=1) == 0, 1.0, ys.std(0, ddof=1))
    xz, yz = (xs - xm) / xsd, (ys - ym) / ysd
    xt = (x[held] - xm) / xsd
    fl = group_folds(groups[sub], 5, inner)
    idx = np.arange(len(xs))
    cv = [(np.setdiff1d(idx, h), h) for h in fl]
    t3 = time.time()
    VS.fit_predict(xz, yz, xt, xt, cv)
    print(f'   standard methods {time.time()-t3:.1f}s', flush=True)
