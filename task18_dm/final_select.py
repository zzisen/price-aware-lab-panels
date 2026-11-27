#!/usr/bin/env python3
"""Task 18 final selection on the TRAINING cycles only (C+D, development layer). Run before the held-out plan is frozen; the output
selected_panels.json is hashed with the plan and then used unchanged on the held-out cycles.
- aware panel at the price of every standard bundle (beam width 4, lambda 1, 5-fold CV Cox log-likelihood, all-cause 9-year death);
- grid of budgets for the cost-matched panel: the cheapest grid panel whose CV log-likelihood is at least that of the routine bundle;
- sequential stages: stage 1 = aware within $20, stage 2 = stage 1 extended within $40."""
import json
import time
import numpy as np
import dmlib as L
L.low_priority()
x, counts = L.load(['C', 'D']); t, e = L.outcome(x); Z = L.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(t)), 5)
cache = {}
out = {'counts': counts, 'n': int(len(t)), 'events': int(e.sum()), 'standard': {}, 'grid': {}}
score = lambda items: cache.setdefault(tuple(sorted(L.features_of(items))), L.gain_ll(Z, t, e, sorted(L.features_of(items)), 1.0, folds))  # noqa: E731
for s, v in L.STANDARD.items():
    out['standard'][s] = dict(items=v, cost=L.cost(v), cv_ll=score(v))
out['none_cv_ll'] = score([])
out['all_cv_ll'] = score(list(L.ITEMS))
grid = sorted(set([L.cost(v) for v in L.STANDARD.values()] + [20.0, 25.0, 30.0, 33.0, 35.0, 38.0, 40.0, 45.0]))
for B in grid:
    t0 = time.time()
    sel, c, sc = L.beam_budget(Z, t, e, B, 1.0, folds, cache=cache)
    out['grid'][f'{B:.2f}'] = dict(items=sel, cost=c, cv_ll=sc)
    print(B, sel, c, round(sc, 2), round(time.time() - t0), 's', flush=True)
s1 = out['grid']['20.00']['items']
s2, c2, sc2 = L.beam_budget(Z, t, e, 40.0, 1.0, folds, cache=cache, start=tuple(s1))
out['stage1'] = dict(items=s1, cost=L.cost(s1)); out['stage2'] = dict(items=s2, cost=c2, cv_ll=sc2)
routine = out['standard']['CMP+CBC+LIPID+A1C']
ok = [(v['cost'], k) for k, v in out['grid'].items() if v['cv_ll'] >= routine['cv_ll']]
out['matched'] = out['grid'][min(ok)[1]] if ok else None
out['at_routine_price'] = out['grid'][f"{routine['cost']:.2f}"]
print(json.dumps({k: out[k] for k in ['stage1', 'stage2', 'matched', 'at_routine_price']}, indent=1))
json.dump(out, open('selected_panels.json', 'w'), indent=1)
