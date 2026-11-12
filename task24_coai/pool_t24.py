#!/usr/bin/env python3
"""Task 24 pooling and decision (plan_t24_FROZEN.md). Usage: pool_t24.py [--dry] [--final]
H5: ours matched - CoAI (as published) at the matched-panel price; H6: ours target-aware - CoAI at the routine price.
Pooled by DerSimonian-Laird random effects (Task 21 code) over cohorts priced with the US fee schedule (NHANES III, NHANES 2011-14; with --final
also Understanding Society if analysed with US prices). Pass: pooled 95% CI lower bound > 0; H6 is claimed only if H5 passes."""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'task21_final'))
import t21lib as X  # noqa: E402
P = 'dry_' if '--dry' in sys.argv else ''; FINAL = '--final' in sys.argv
coh = ['NH3', 'GH'] + (['UKHLS'] if FINAL else [])
E = {c: json.load(open(HERE / f'{P}t24_{c}_endpoints.json')) for c in coh if (HERE / f'{P}t24_{c}_endpoints.json').exists()}
out = {'cohorts': list(E)}
for h, k in [('H5', 'd ours matched - CoAI @matched price'), ('H6', 'd ours target-aware - CoAI @routine price')]:
    est = [E[c]['point'][k] for c in E]; lo = [E[c]['boot'][k][0] for c in E]; hi = [E[c]['boot'][k][1] for c in E]
    out[h] = dict(per_cohort={c: dict(est=a, lo=b, hi=g) for c, a, b, g in zip(E, est, lo, hi)}, pooled=X.pool(est, lo, hi))
out['H5']['pass'] = bool(out['H5']['pooled']['lo'] > 0); out['H6']['pass'] = bool(out['H5']['pass'] and out['H6']['pooled']['lo'] > 0)
for c in ['I', 'CHNS', 'CLHLS'] + ([] if FINAL else ['UKHLS']):
    f = HERE / f'{P}t24_{c}_endpoints.json'
    if f.exists():
        d = json.load(open(f)); out[f'reported_not_pooled_{c}'] = dict(point=d['point'], boot=d['boot'], identical_panels=d['identical_panels'])
print(json.dumps(out, indent=1, default=float)); json.dump(out, open(HERE / f'{P}t24_pooled.json', 'w'), indent=1, default=float)
