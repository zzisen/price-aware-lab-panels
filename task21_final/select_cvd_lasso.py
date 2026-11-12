#!/usr/bin/env python3
"""Task 21 training-layer selections (NHANES 1999-2010 only). Usage: select_cvd_lasso.py <cvd|lasso>
cvd:   cardiovascular death (heart disease or stroke), base = age, sex, BMI, SBP, smoking, diabetes (complete clinical). Beam search under US
       prices; panels: matched to routine, aware within the routine price, aware within the price of the PREVENT laboratory set. Guideline
       laboratory sets scored with the same cross-validated likelihood: PCE/SCORE2 (total and HDL cholesterol), PREVENT base (+ creatinine),
       PREVENT full (+ urine ACR, HbA1c).
lasso: all-cause death, base = age, sex. Price-weighted and unweighted lasso Cox paths over all analytes; every distinct active set is scored
       with the same cross-validated likelihood as the beam search (comparator for the selection step)."""
import json
import sys
import time
import numpy as np
import pandas as pd
import dm2lib as M
import t21lib as X
M.L.low_priority()
MODE = sys.argv[1]
x, counts = M.load(list('ABCDEF')); Z = M.transform(x)
ROUT = M.STANDARD['CMP+CBC+LIPID+A1C']
if MODE == 'cvd':
    t, e = M.outcome(x, 'cvd')
    cl = M.clinical(x); ok = cl.notna().all(1).to_numpy()
    Z = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)].reset_index(drop=True); t, e = t[ok], e[ok]
    base = M.BASE + M.CLIN
    folds = np.array_split(np.random.default_rng(1).permutation(len(t)), 5); cache = {}
    sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds, base))  # noqa: E731
    G = {'none': [], 'PCE/SCORE2 lipids': ['tc', 'hdl'], 'PREVENT base': ['tc', 'hdl', 'creatinine'], 'PREVENT full': ['tc', 'hdl', 'creatinine', 'acr', 'hba1c'],
         'routine': M.features_of(ROUT), 'CMP+CBC': M.features_of(['CMP', 'CBC'])}
    gs = {k: dict(features=v, cost=M.cost_us(v) if v else 0.0, cv=sc(v)) for k, v in G.items()}
    for k, v in gs.items():
        print('guideline/standard', k, v['cost'], round(v['cv'], 2), flush=True)
    grid = sorted(set([gs['PCE/SCORE2 lipids']['cost'], gs['PREVENT base']['cost'], gs['PREVENT full']['cost'], M.cost(ROUT), 17.11, 19.9, 27.67, 35.0, 41.06]))
    for B in grid:
        t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, 'US', cache=cache, base=base)
        print('beam', B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
    mk = X.cheapest_matching(cache, gs['routine']['cv'], M.cost_us)
    mp = X.cheapest_matching(cache, gs['PREVENT full']['cv'], M.cost_us)
    ar = X.best_within(cache, M.cost(ROUT), M.cost_us); ap = X.best_within(cache, gs['PREVENT full']['cost'], M.cost_us)
    out = dict(counts=counts, n=int(len(t)), events=int(e.sum()), base=base, guideline=gs,
               matched_routine=dict(features=list(mk), items=M.describe(mk), cost=M.cost_us(mk), cv=cache[mk]),
               matched_prevent_full=dict(features=list(mp), items=M.describe(mp), cost=M.cost_us(mp), cv=cache[mp]),
               aware_at_routine=dict(features=list(ar), items=M.describe(ar), cost=M.cost_us(ar), cv=cache[ar]),
               aware_at_prevent_full=dict(features=list(ap), items=M.describe(ap), cost=M.cost_us(ap), cv=cache[ap]))
    pd.DataFrame([dict(features=' '.join(k), cv=v, US=M.cost_us(k)) for k, v in cache.items()]).to_csv('selected_cvd_cache.csv', index=False)
    json.dump(out, open('selected_cvd.json', 'w'), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ['matched_routine', 'matched_prevent_full', 'aware_at_routine', 'aware_at_prevent_full']}, indent=1, default=str))
else:
    t, e = M.outcome(x)
    folds = np.array_split(np.random.default_rng(1).permutation(len(t)), 5)
    feats = M.FEATURES
    X_ = Z[M.BASE + feats].to_numpy(); X_ = (X_ - X_.mean(0)) / X_.std(0)
    rows, seen = [], {}
    for kind, w in [('price-weighted', np.r_[0, 0, X.feature_price_weights(feats)]), ('unweighted', np.r_[0, 0, np.ones(len(feats))])]:
        t0 = time.time(); path = X.lasso_path(X_, t, e, w, n_lambda=40)
        for lam, b in path:
            act = tuple(sorted(f for f, bj in zip(feats, b[2:]) if abs(bj) > 0))
            if act not in seen:
                seen[act] = M.gain_ll(Z, t, e, list(act), 1.0, folds)
            rows.append(dict(kind=kind, lam=lam, n_active=len(act), features=' '.join(act), US=M.cost_us(act) if act else 0.0, cv=seen[act]))
        print(kind, 'path done', round(time.time() - t0), 's', len(seen), 'distinct sets', flush=True)
    d = pd.DataFrame(rows); d.to_csv('lasso_candidates.csv', index=False)
    rt = M.gain_ll(Z, t, e, M.features_of(ROUT), 1.0, folds)
    for kind in ['price-weighted', 'unweighted']:
        s = d[d.kind == kind]
        mt = s[s.cv >= rt].sort_values(['US', 'cv'], ascending=[True, False]).head(1)
        aw = s[s.US <= M.cost(ROUT) + 1e-9].sort_values('cv', ascending=False).head(1)
        print(kind, 'matched:', mt[['features', 'US', 'cv']].to_dict('records'), '| best within routine price:', aw[['features', 'US', 'cv']].to_dict('records'), flush=True)
    print('routine cv', round(rt, 2))
