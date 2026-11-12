#!/usr/bin/env python3
"""Task 20 selection on the TRAINING layer only (NHANES 1999-2010, A-F), restricted to the 15 analytes CHNS also measured, priced with the
Chinese schedule. Same procedure as Task 19 (beam width 4, lambda 1, 5-fold CV Cox log-likelihood, all-cause death, shared score cache).
Output: selected_cn.json"""
import json
import time
import numpy as np
import pandas as pd
import dm2lib as M
import t20lib as T
M.L.low_priority()
T.install_cn()
x, counts = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
pool = list(T.CN_ITEMS)
cache = {}
sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds))  # noqa: E731
routine_cv = sc(T.CN_ROUTINE_SHARED); sc([]); sc(T.SHARED)
routine_price_shared = T.cost_cn(T.CN_ROUTINE_SHARED); routine_price_full = T.cost_cn(T.CN_ROUTINE_FULL)
grid = sorted(set([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, routine_price_shared]))
for B in grid:
    t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, 'CN', pool=pool, cache=cache)
    print(B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
ok = sorted(((T.cost_cn(k), -v), k) for k, v in cache.items() if v >= routine_cv)
mk = ok[0][1]
av, af = max((v, k) for k, v in cache.items() if T.cost_cn(k) <= routine_price_shared + 1e-9)   # best affordable (dm2lib's version assumes US item names)
ac = T.cost_cn(af)
out = dict(counts=counts, n=int(len(x)), events=int(e.sum()), routine_shared=dict(features=T.CN_ROUTINE_SHARED, price=routine_price_shared, cv=routine_cv),
           routine_full_price=routine_price_full, matched=dict(features=list(mk), items=T.describe_cn(mk), price=T.cost_cn(mk), cv=cache[mk]),
           aware_at_routine=dict(features=list(af), items=T.describe_cn(af), price=ac, cv=av), none_cv=cache[()], all_shared_cv=cache[tuple(sorted(T.SHARED))])
print(json.dumps({k: out[k] for k in ['routine_shared', 'routine_full_price', 'matched', 'aware_at_routine']}, indent=1))
pd.DataFrame([dict(features=' '.join(k), cv=v, CN=T.cost_cn(k)) for k, v in cache.items()]).to_csv('selected_cn_cache.csv', index=False)
json.dump(out, open('selected_cn.json', 'w'), indent=1, default=str)
print('done')
