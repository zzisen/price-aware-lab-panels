#!/usr/bin/env python3
"""Independent check of Task 24 (imports no project library). Reads t24_<c>_person.csv, recomputes C (Harrell, ties in risk count 1/2) or
AUC (rank-based), the primary contrasts and their bootstrap intervals with its own resampling, and compares with t24_<c>_endpoints.json.
Usage: verify_t24.py <cohort> [--dry]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; c = sys.argv[1]; pre = ('dry_' if '--dry' in sys.argv else '') + f't24_{c}_'
d = pd.read_csv(HERE / f'{pre}person.csv'); E = json.load(open(HERE / f'{pre}endpoints.json')); BIN = 'y' in d


def harrell(s, t, e):
    o = np.argsort(t, kind='stable'); s, t, e = s[o], t[o], e[o]; num = den = 0.0
    for i in np.flatnonzero(e):
        later = t > t[i]
        den += later.sum(); num += (s[i] > s[later]).sum() + 0.5 * (s[i] == s[later]).sum()
    return num / den


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


sc = (lambda s, i: auc(s[i], d.y.to_numpy()[i])) if BIN else (lambda s, i: harrell(s[i], d.t.to_numpy()[i], d.e.to_numpy()[i]))
allidx = np.arange(len(d)); bad = 0
for k, v in E['point'].items():
    a, b = k[2:].split(' - ')
    pa, pb = d['cox|' + a].to_numpy(), d['cox|' + b].to_numpy()
    est = sc(pa, allidx) - sc(pb, allidx)
    rng = np.random.default_rng(7); bs = []
    for _ in range(300):
        i = rng.integers(0, len(d), len(d)); bs.append(sc(pa, i) - sc(pb, i))
    lo, hi = np.quantile(bs, [.025, .975]); ok = abs(est - v) < 1e-6
    bad += not ok
    print(f'{k:55s} point {est:+.5f} (run {v:+.5f}) {"OK" if ok else "MISMATCH"} | own bootstrap [{lo:+.4f}, {hi:+.4f}] vs run {E["boot"][k]}')
print('ALL POINT ESTIMATES MATCH' if bad == 0 else f'{bad} MISMATCHES')
