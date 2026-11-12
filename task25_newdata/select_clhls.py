#!/usr/bin/env python3
"""Task 25 selection for CLHLS on the TRAINING layer only (NHANES 1999-2010, A-F), restricted to the 14 analytes CLHLS measured in all three blood
waves, priced with the Chinese schedule. Same procedure as Task 20 for CHNS (beam width 4, lambda 1, 5-fold CV Cox log-likelihood, all-cause death).
Also runs CoAI (Task 24 implementation, additive Chinese prices) on the same menu and budgets. Output: selected_clhls.json"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(W / 'task24_coai')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import clhls_lib as C  # noqa: E402
import coai_select as CO  # noqa: E402
M.L.low_priority(); C.install()
x, counts = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
pool = list(C.ITEMS_CL); cache = {}
sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds))  # noqa: E731
routine_cv = sc(C.ROUTINE_CL); sc([]); sc(C.MENU)
routine_price = T.cost_cn(C.ROUTINE_CL)
for B in sorted(set([15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, routine_price])):
    it, c, v = M.beam(Z, t, e, B, folds, 'CN', pool=pool, cache=cache); print(B, it, c, round(v, 2), len(cache), flush=True)
ok = sorted(((T.cost_cn(k), -v), k) for k, v in cache.items() if v >= routine_cv); mk = ok[0][1]
av, af = max((v, k) for k, v in cache.items() if T.cost_cn(k) <= routine_price + 1e-9)
out = dict(menu=C.MENU, routine=dict(features=C.ROUTINE_CL, price=routine_price, cv=routine_cv),
           matched=dict(features=list(mk), price=T.cost_cn(mk), cv=cache[mk]), aware_at_routine=dict(features=list(af), price=T.cost_cn(af), cv=av),
           none_cv=cache[()], all_menu_cv=cache[tuple(sorted(C.MENU))])
# CoAI on the same menu (additive Chinese prices; age and sex free), budgets = prices of our two panels
known = e | (t >= M.HORIZON); y = (e & (t <= M.HORIZON))[known].astype(int)
groups = {a: ([a], T.CN_PRICE[a]) for a in C.CHEM_CL}; groups['CBC_CL'] = (C.CBC_CL, T.CN_CBC)
out['coai'] = CO.run(Z[known], y, groups, [out['matched']['price'], routine_price], T.CN_VENI, T.cost_cn)
json.dump(out, open(HERE / 'selected_clhls.json', 'w'), indent=1, default=str)
print(json.dumps({k: out[k] for k in ['routine', 'matched', 'aware_at_routine']}, indent=1))
for p in out['coai']['panels']:
    print('CoAI', p['budget'], p['features'], p['billed_cost'])
