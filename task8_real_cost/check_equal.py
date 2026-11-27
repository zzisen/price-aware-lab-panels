#!/usr/bin/env python3
"""Check global3_fast against the frozen exact_budget: random score matrices (sizes the frozen code can run) and the
real CIPA training-score matrices."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L
from exact_budget import globally_optimal_actions as slow
from global3_fast import globally_optimal_actions as fast
rng = np.random.default_rng(1)
n = bad = 0
for f, t in [(9, 7), (9, 10), (20, 8), (50, 12), (7, 7), (116, 9)]:
    for rep in range(30):
        r = rng.gamma(2.0, 1.0, size=(f, t))
        n += 1; bad += not np.array_equal(slow(r, 3)[0], fast(r, 3)[0])
L.HERE = HERE
_, x, y, fam = L.load_task('CIPA')
for rep in range(30):
    sub = rng.choice(len(x), size=11, replace=False)
    Lg, _ = L.unit_losses_g(x[sub], y[sub], np.arange(11), 5, rep)
    r = Lg.mean(axis=0)
    n += 1; bad += not np.array_equal(slow(r, 3)[0], fast(r, 3)[0])
print('checked', n, 'identical', n - bad, flush=True)
