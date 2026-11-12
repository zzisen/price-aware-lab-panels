#!/usr/bin/env python3
"""Task 27 selection for Understanding Society on the TRAINING layer only (NHANES 1999-2010), restricted to the 13 analytes measured in the
UKHLS nurse blood sample, US Medicare prices. Same procedure as Tasks 19/20/25 (beam width 4, lambda 1, 5-fold CV Cox log-likelihood).
Also CoAI (Task 24 implementation): as published (single-test prices summed) and with the billed bundles as groups.
Output: selected_ukhls.json"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
for d in ['task21_final', 'task24_coai']:
    sys.path.insert(0, str(W / d))
sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import ukhls_menu as U  # noqa: E402
import coai_select as CO  # noqa: E402
M.L.low_priority(); U.install()
x, _ = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
pool = list(U.ITEMS_UK); cache = {}
sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds))  # noqa: E731
routine_cv = sc(U.ROUTINE_UK); sc([]); sc(U.MENU)
routine_price = M.cost_us(U.ROUTINE_UK)
for B in sorted(set([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, routine_price])):
    it, c, v = M.beam(Z, t, e, B, folds, 'US', pool=pool, cache=cache); print(B, it, c, round(v, 2), len(cache), flush=True)
ok = sorted(((M.cost_us(k), -v), k) for k, v in cache.items() if v >= routine_cv); mk = ok[0][1]
av, af = max((v, k) for k, v in cache.items() if M.cost_us(k) <= routine_price + 1e-9)
out = dict(menu=U.MENU, routine=dict(features=U.ROUTINE_UK, price=routine_price, cv=routine_cv),
           matched=dict(features=list(mk), price=M.cost_us(mk), cv=cache[mk]), aware_at_routine=dict(features=list(af), price=M.cost_us(af), cv=av),
           none_cv=cache[()], all_menu_cv=cache[tuple(sorted(U.MENU))])
known = e | (t >= M.HORIZON); y = (e & (t <= M.HORIZON))[known].astype(int)
L = M.L
pub = {a: ([a], L.CHEM[a][2]) for a in U.CMP_UK + ['ggt', 'tc', 'hdl', 'tg']}
pub.update({'CBC_UK': (['hgb'], 7.77), 'hba1c': (['hba1c'], 9.71), 'crp': (['crp'], 12.95)})
bun = {'CMP_UK': (U.CMP_UK, L.PANELS['CMP'][1]), 'LIPID': (['tc', 'hdl', 'tg'], L.PANELS['LIPID'][1]), 'ggt': (['ggt'], L.CHEM['ggt'][2]),
       'CBC_UK': (['hgb'], 7.77), 'hba1c': (['hba1c'], 9.71), 'crp': (['crp'], 12.95)}
budgets = [out['matched']['price'], routine_price]
out['coai_published'] = CO.run(Z[known], y, pub, budgets, 9.34, M.cost_us)
out['coai_bundled'] = CO.run(Z[known], y, bun, budgets, 9.34, M.cost_us)
json.dump(out, open(HERE / 'selected_ukhls.json', 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ['routine', 'matched', 'aware_at_routine']}, indent=1))
for v in ['coai_published', 'coai_bundled']:
    for p in out[v]['panels']:
        print(v, p['budget'], p['features'], 'billed', p['billed_cost'])
