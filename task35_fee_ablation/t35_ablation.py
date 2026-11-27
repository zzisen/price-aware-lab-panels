#!/usr/bin/env python3
"""Task 35 (additional analysis; written before running): fee representation x selection algorithm, all-cause death, US prices.
Representations
  single : every analyte priced as a single test (Medicare single-test fees), the blood count as one group ($7.77), HbA1c $9.71, CRP $12.95,
           urine ACR $10.96, plus one draw ($9.34) - the representation used by CoAI as published;
  bundle : the real 2026 Medicare schedule as ordered (panels performed in full, cheapest billable combination, one draw) - our representation.
Algorithms
  beam   : our beam search (width 4, ridge Cox lambda 1, 5-fold CV partial likelihood, folds seed 1) with a fresh cache per representation;
  lasso  : price-weighted lasso Cox path; weights = single-test prices (single) or panel price shared over the panel's analytes (bundle);
  CoAI   : as run in Task 24 (US_published = single, US_bundled = bundle), taken from task24_coai/coai_panels.json.
Selection in NHANES 1999-2010 within the budgets $32.41 and $50.77 measured in the algorithm's own representation; every selection is then
billed by the real schedule and validated in NHANES III (9-year C) and NHANES 2011-14 (5-year C) with the same ridge Cox refit.
Paired bootstrap (1000, seed 20261129) for bundle minus single within each algorithm. Output: t35_results.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
L = M.L
M.L.low_priority()
DRAW = 9.34


def cost_single(f):
    f = set(f)
    if not f:
        return 0.0
    tot = sum(L.CHEM[a][2] for a in f if a in L.CHEM) + (7.77 if f & set(M.CBC2) else 0) + (9.71 if 'hba1c' in f else 0) \
        + (12.95 if 'crp' in f else 0) + (10.96 if 'acr' in f else 0)
    return round(tot + DRAW, 2)


M.SCHEDULES['USS'] = cost_single
x, _ = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(t)), 5)
ROUT = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']); rcv = M.gain_ll(Z, t, e, ROUT, 1.0, folds)
SEL = json.load(open(W / 'task19_dm2' / 'selected_all.json'))
P = {'beam bundle, matched': M.features_of(SEL['US']['matched']['items']),
     'beam bundle, aware': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
# beam, single representation (fresh cache)
cache = {}
for B in [32.41, 50.77]:
    it, c, v = M.beam(Z, t, e, B, folds, 'USS', cache=cache)
    print('beam single', B, it, c, round(v, 2), len(cache), flush=True)
mk = X.cheapest_matching(cache, rcv, cost_single); aw = X.best_within(cache, 50.77, cost_single)
P['beam single, matched'] = list(mk) if mk else None
P['beam single, aware'] = list(aw)
# lasso, both weightings
feats = M.FEATURES
X_ = Z[M.BASE + feats].to_numpy(); X_ = (X_ - X_.mean(0)) / X_.std(0)


def bundle_weights(fs):
    w = []
    for f in fs:
        if f in L.PANELS['CMP'][0]:
            w.append(L.PANELS['CMP'][1] / len(L.PANELS['CMP'][0]))
        elif f in L.PANELS['LIPID'][0]:
            w.append(L.PANELS['LIPID'][1] / len(L.PANELS['LIPID'][0]))
        elif f in L.CHEM:
            w.append(L.CHEM[f][2])
        elif f in M.CBC2:
            w.append(7.77 / len(M.CBC2))
        else:
            w.append({'hba1c': 9.71, 'crp': 12.95, 'acr': 10.96}[f])
    w = np.array(w); return w / w.mean()


lasso_sets = {}
for rep, wts, price in [('single', X.feature_price_weights(feats), cost_single), ('bundle', bundle_weights(feats), M.cost_us)]:
    seen = {}
    for lam, b in X.lasso_path(X_, t, e, np.r_[0, 0, wts], n_lambda=40):
        act = tuple(sorted(f for f, bj in zip(feats, b[2:]) if abs(bj) > 0))
        if act not in seen:
            seen[act] = M.gain_ll(Z, t, e, list(act), 1.0, folds)
    m_ = X.cheapest_matching(seen, rcv, price); a_ = X.best_within(seen, 50.77, price)
    P[f'lasso {rep}, matched'] = list(m_) if m_ else None; P[f'lasso {rep}, aware'] = list(a_) if a_ else None
    lasso_sets[rep] = len(seen); print('lasso', rep, len(seen), 'sets', flush=True)
CO = json.load(open(W / 'task24_coai' / 'coai_panels.json'))
for rep, key in [('single', 'US_published'), ('bundle', 'US_bundled')]:
    pb = {p['budget']: p['features'] for p in CO[key]['panels']}
    P[f'CoAI {rep}, matched'] = pb[32.41]; P[f'CoAI {rep}, aware'] = pb[50.77]
P['routine'] = ROUT; P['CMP+CBC'] = M.features_of(['CMP', 'CBC']); P['age+sex only'] = []
P = {k: v for k, v in P.items() if v is not None}
info = {k: dict(features=sorted(v), items=M.describe(v) if v else [], billed=(M.cost_us(v) if v else 0.0), single=(cost_single(v) if v else 0.0),
                cv=M.gain_ll(Z, t, e, list(v), 1.0, folds)) for k, v in P.items()}
out = dict(routine_cv=rcv, lasso_distinct_sets=lasso_sets, panels=info, cohorts={})
pairs = [(f'{a} bundle, {s}', f'{a} single, {s}') for a in ('beam', 'lasso', 'CoAI') for s in ('matched', 'aware')]
pairs = [p for p in pairs if p[0] in P and p[1] in P]
for COH, H in [('NH3', 108), ('GH', 60)]:
    xte = M.load_nh3()[0] if COH == 'NH3' else pd.concat([T.load_gh(c)[0] for c in 'GH'])
    Zt = M.transform(xte, ref=x); tv, ev = T.outcome_at(xte, H)
    Pc = {k: v for k, v in P.items() if xte.crp.notna().all() or 'crp' not in v}
    LP = {k: M.Cox(1.0).fit(Z[M.BASE + list(f)].to_numpy(), t, e).lp(Zt[M.BASE + list(f)].to_numpy()) for k, f in Pc.items()}
    C = {k: M.cindex(v, tv, ev) for k, v in LP.items()}
    rng = np.random.default_rng(20261129); n = len(tv); bs = {f'{a} - {b}': [] for a, b in pairs if a in LP and b in LP}
    for _ in range(1000):
        j = rng.integers(0, n, n)
        for a, b in pairs:
            if a in LP and b in LP:
                bs[f'{a} - {b}'].append(M.cindex(LP[a][j], tv[j], ev[j]) - M.cindex(LP[b][j], tv[j], ev[j]))
    ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
    out['cohorts'][COH] = dict(n=n, deaths=int(ev.sum()), C=C, contrasts={k: dict(point=C[k.split(' - ')[0]] - C[k.split(' - ')[1]], ci=ci(v)) for k, v in bs.items()})
json.dump(out, open(HERE / 't35_results.json', 'w'), indent=1, default=float)
for k, v in info.items():
    print(f"{k:24s} billed ${v['billed']:6.2f}  single-priced ${v['single']:6.2f}  cv {v['cv']:.1f}  {v['items']}")
for COH, r in out['cohorts'].items():
    print(COH, 'n', r['n'], 'deaths', r['deaths'])
    for k, v in r['C'].items():
        print(f'   {k:24s} C {v:.4f}')
    for k, v in r['contrasts'].items():
        print(f"   {k:50s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
