#!/usr/bin/env python3
"""Task 21 selection stability on the training layer (NHANES 1999-2010): 20 random 80% subsamples (without replacement). In each,
the same procedure (beam width 4, lambda 1, 5-fold CV likelihood, US prices) at the routine price and at the Task 19 matched / CMP+CBC /
CMP+CBC+acr prices; matched = cheapest evaluated set with CV likelihood >= routine's; aware = best evaluated set within the routine price.
Output: stability.csv"""
import numpy as np
import pandas as pd
import dm2lib as M
M.L.low_priority()
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
ROUT = M.STANDARD['CMP+CBC+LIPID+A1C']; RP = M.cost(ROUT)
rng = np.random.default_rng(20261108); rows = []
for b in range(20):
    s = np.sort(rng.choice(len(t), int(0.8 * len(t)), replace=False))
    Zs, ts, es = Z.iloc[s].reset_index(drop=True), t[s], e[s]
    folds = np.array_split(np.random.default_rng(b).permutation(len(s)), 5); cache = {}
    thr = cache.setdefault(tuple(sorted(M.features_of(ROUT))), M.gain_ll(Zs, ts, es, sorted(M.features_of(ROUT)), 1.0, folds))
    for B in (27.67, 32.41, 38.63, RP):
        M.beam(Zs, ts, es, B, folds, 'US', cache=cache)
    mk = sorted(((M.cost_us(k), -v), k) for k, v in cache.items() if v >= thr)[0][1]
    ak = max((v, k) for k, v in cache.items() if M.cost_us(k) <= RP + 1e-9)[1]
    rows.append(dict(sample=b, matched=' '.join(M.describe(mk)), matched_cost=M.cost_us(mk), aware=' '.join(M.describe(ak)), aware_cost=M.cost_us(ak)))
    print(rows[-1], flush=True)
d = pd.DataFrame(rows); d.to_csv('stability.csv', index=False)
print(d.matched.value_counts(normalize=True).to_string()); print(d.aware.value_counts(normalize=True).to_string())
print('matched cost <= 80% of routine in', float((d.matched_cost <= 0.8 * RP).mean()))
