#!/usr/bin/env python3
"""Independent check of the Task 21 contrasts on Understanding Society (imports no project library). Reads t27_UKHLS_person.csv and
../task21_final/t21_UKHLS_endpoints.json, recomputes AUC (Mann-Whitney with mid-ranks), the AUC contrasts and their bootstrap intervals with its
own resampling. Usage: verify_t27.py [--dry]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent; pre = 'dry_' if '--dry' in sys.argv else ''
d = pd.read_csv(HERE / f'{pre}t27_UKHLS_person.csv'); E = json.load(open(HERE.parent / 'task21_final' / f'{pre}t21_UKHLS_endpoints.json'))
y = d.y.to_numpy().astype(int)


def auc(s, yy):
    r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


bad = 0
for k, v in E['point'].items():
    if not k.startswith('dAUC '):
        continue
    a, b = k[5:].split(' - '); pa, pb = d['cox|' + a].to_numpy(), d['cox|' + b].to_numpy()
    est = auc(pa, y) - auc(pb, y); rng = np.random.default_rng(7); bs = []
    for _ in range(300):
        i = rng.integers(0, len(d), len(d)); bs.append(auc(pa[i], y[i]) - auc(pb[i], y[i]))
    lo, hi = np.quantile(bs, [.025, .975]); ok = abs(est - v) < 1e-6; bad += not ok
    print(f'{k:50s} point {est:+.5f} (run {v:+.5f}) {"OK" if ok else "MISMATCH"} | own bootstrap [{lo:+.4f}, {hi:+.4f}] vs run {E["boot"][k]}')
print('ALL POINT ESTIMATES MATCH' if bad == 0 else f'{bad} MISMATCHES')
