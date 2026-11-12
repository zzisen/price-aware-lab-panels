#!/usr/bin/env python3
"""Independent check of the Task 21 contrasts on CLHLS (imports no project library). Reads t25_<tag>_person.csv and
../task21_final/t21_<tag>_endpoints.json, recomputes Harrell's C (ties in score count 1/2), the C contrasts and their bootstrap intervals with its own
resampling. Usage: verify_t25.py <CLHLS|CLHLS80> [--dry]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; tag = sys.argv[1]; pre = 'dry_' if '--dry' in sys.argv else ''
d = pd.read_csv(HERE / f'{pre}t25_{tag}_person.csv'); E = json.load(open(HERE.parent / 'task21_final' / f'{pre}t21_{tag}_endpoints.json'))
t, e = d.t.to_numpy(), d.e.to_numpy().astype(bool)


def harrell(s, t, e):
    o = np.argsort(t, kind='stable'); s, t, e = s[o], t[o], e[o]; num = den = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        den += later.sum(); num += (s[i] > s[later]).sum() + 0.5 * (s[i] == s[later]).sum()
    return num / den


bad = 0
for k, v in E['point'].items():
    if not k.startswith('dC '):
        continue
    a, b = k[3:].split(' - '); pa, pb = d['cox|' + a].to_numpy(), d['cox|' + b].to_numpy()
    est = harrell(pa, t, e) - harrell(pb, t, e); rng = np.random.default_rng(7); bs = []
    for _ in range(300):
        i = rng.integers(0, len(d), len(d)); bs.append(harrell(pa[i], t[i], e[i]) - harrell(pb[i], t[i], e[i]))
    lo, hi = np.quantile(bs, [.025, .975]); ok = abs(est - v) < 1e-6; bad += not ok
    print(f'{k:40s} point {est:+.5f} (run {v:+.5f}) {"OK" if ok else "MISMATCH"} | own bootstrap [{lo:+.4f}, {hi:+.4f}] vs run {E["boot"][k]}')
print('ALL POINT ESTIMATES MATCH' if bad == 0 else f'{bad} MISMATCHES')
