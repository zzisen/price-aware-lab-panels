#!/usr/bin/env python3
"""Task 19 final selection on the TRAINING layer only (NHANES 1999-2010, cycles A-F). Run before the plan is frozen; NHANES III is not read.
Usage: final_select.py <all|targets>
all:     all-cause 9-year death. One shared cache of cross-validated scores (the score does not depend on prices); for each schedule
         (US, AU, JP) beam searches at every standard-bundle price and at budget quantiles; then
         - aware@<bundle>: best evaluated panel affordable at that bundle's price (monotone in budget),
         - matched: cheapest evaluated panel whose CV log-likelihood is at least the routine bundle's (ties: higher score);
         sequential policy fixed as CMP > CBC > urine ACR with age, sex, BMI, SBP, smoking and diabetes as stage 0; its go-on risk intervals.
targets: the same 'matched' and 'aware@routine' panels (US prices) for CVD, cancer and other deaths (report only).
Output: selected_<mode>.json"""
import json
import sys
import time
import numpy as np
import pandas as pd
import dm2lib as M
M.L.low_priority()
MODE = sys.argv[1]
x, counts = M.load(list('ABCDEF')); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
ROUT = M.STANDARD['CMP+CBC+LIPID+A1C']
out = {'counts': counts, 'n': int(len(x))}


def select(t, e, schedules):
    cache = {}
    sc = lambda items: cache.setdefault(tuple(sorted(M.features_of(items))), M.gain_ll(Z, t, e, sorted(M.features_of(items)), 1.0, folds))  # noqa: E731
    for v in list(M.STANDARD.values()) + [[], list(M.ITEMS)]:
        sc(v)
    res = {}
    for s in schedules:
        std = {k: M.cost(v, s) for k, v in M.STANDARD.items()}
        grid = sorted(set(list(std.values()) + list(np.round(np.quantile(list(std.values()), [0.1, 0.3, 0.5, 0.7, 0.9]), 2))))
        for B in grid:
            t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, s, cache=cache)
            print(s, B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
        r = {'standard': {k: dict(items=v, cost=std[k], cv=cache[tuple(sorted(M.features_of(v)))]) for k, v in M.STANDARD.items()}, 'aware': {}}
        for k in M.STANDARD:
            it, c, v = M.best_affordable(cache, std[k], s); r['aware'][k] = dict(items=it, cost=c, cv=v)
        thr = r['standard']['CMP+CBC+LIPID+A1C']['cv']
        ok = sorted(((M.SCHEDULES[s](k), -v), k) for k, v in cache.items() if v >= thr)
        k = ok[0][1]; r['matched'] = dict(items=M.describe(k), cost=M.cost(M.describe(k), s), cv=cache[k])
        res[s] = r
        print(s, 'matched', r['matched'], 'aware@routine', r['aware']['CMP+CBC+LIPID+A1C'], flush=True)
    return res, cache


if MODE == 'all':
    t, e = M.outcome(x); out['events'] = int(e.sum())
    res, cache = select(t, e, ['US', 'AU', 'JP'])
    out.update(res)
    out['cache_size'] = len(cache)
    pd.DataFrame([dict(features=' '.join(k), cv=v, US=M.cost_us(k), AU=M.cost_au(k), JP=M.cost_jp(k)) for k, v in cache.items()]).to_csv('selected_all_cache.csv', index=False)
    # sequential policy: trained on people with complete clinical variables
    cl = M.clinical(x); ok = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
    tc, ec = t[ok], e[ok]
    fc = np.array_split(np.random.default_rng(1).permutation(len(tc)), 5)
    out['sequential'] = {'blocks': ['CMP', 'CBC', 'acr'], 'base': M.BASE + M.CLIN, 'n_complete_clinical': int(ok.sum())}
    for tau in (0.10, 0.20):
        P = M.FixedPath(['CMP', 'CBC', 'acr'], tau, base=M.BASE + M.CLIN).fit(Zc, tc, ec, fc)
        out['sequential'][f'tau={tau}'] = {'+'.join(sorted(s)) or 'start': {f'eps={eps}': M.interval(f, eps) for eps in (0.01, 0.02, 0.05)} for s, f in P.stop.items()}
        print('sequential', tau, json.dumps(out['sequential'][f'tau={tau}']), flush=True)
else:
    for cause in ['cvd', 'cancer', 'other']:
        t, e = M.outcome(x, cause)
        res, _ = select(t, e, ['US'])
        out[cause] = dict(events=int(e.sum()), **res['US'])
json.dump(out, open(f'selected_{MODE}.json', 'w'), indent=1, default=str)
print('done')
