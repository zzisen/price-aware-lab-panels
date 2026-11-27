#!/usr/bin/env python3
"""Task 43 (b), post hoc: Cox ridge-penalty sensitivity of the US all-cause selection. Re-runs task19_dm2/final_select.py 'all' for the US
schedule only (the US matched and aware panels were computed after the US beams and before AU/JP, so a US-only run reproduces them), with
the ridge penalty lam in gain_ll and beam. Training layer only (NHANES 1999-2010). Usage: b_select.py <lam>. Output b_lambda/select_<lam>.json"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
os.chdir(W / 'task21_final'); sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
M.L.low_priority()
LAM = float(sys.argv[1])
x, counts = M.load(list('ABCDEF')); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
t, e = M.outcome(x)
cache = {}
sc = lambda items: cache.setdefault(tuple(sorted(M.features_of(items))), M.gain_ll(Z, t, e, sorted(M.features_of(items)), LAM, folds))  # noqa: E731
for v in list(M.STANDARD.values()) + [[], list(M.ITEMS)]:
    sc(v)
s = 'US'
std = {k: M.cost(v, s) for k, v in M.STANDARD.items()}
grid = sorted(set(list(std.values()) + list(np.round(np.quantile(list(std.values()), [0.1, 0.3, 0.5, 0.7, 0.9]), 2))))
for B in grid:
    t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, s, lam=LAM, cache=cache)
    print(LAM, s, B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
r = {'lam': LAM, 'n': int(len(x)), 'events': int(e.sum()),
     'standard': {k: dict(items=v, cost=std[k], cv=cache[tuple(sorted(M.features_of(v)))]) for k, v in M.STANDARD.items()}, 'aware': {}}
for k in M.STANDARD:
    it, c, v = M.best_affordable(cache, std[k], s); r['aware'][k] = dict(items=it, cost=c, cv=v)
thr = r['standard']['CMP+CBC+LIPID+A1C']['cv']
ok = sorted(((M.SCHEDULES[s](k), -v), k) for k, v in cache.items() if v >= thr)
k = ok[0][1]; r['matched'] = dict(items=M.describe(k), cost=M.cost(M.describe(k), s), cv=cache[k])
r['cache_size'] = len(cache)
json.dump(r, open(HERE / 'b_lambda' / f'select_{LAM:g}.json', 'w'), indent=1, default=str)
print('matched', r['matched'], 'aware@routine', r['aware']['CMP+CBC+LIPID+A1C'], flush=True)
