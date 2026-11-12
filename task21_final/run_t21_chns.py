#!/usr/bin/env python3
"""Task 21 unified final analysis on CHNS (China, 2009 blood, deaths by the 2015 survey). Usage: run_t21_chns.py [--dry]
Training: NHANES 1999-2010 restricted to the 15 shared analytes; Chinese prices; panels from ../task20_chns/selected_cn.json (training-only)."""
import json
import sys
import numpy as np
import pandas as pd
import dm2lib as M
import t20lib as T
import t21lib as X
M.L.low_priority()
DRY = '--dry' in sys.argv; PRE = f"{'dry_' if DRY else ''}t21_CHNS_"
NBOOT, NSUB = (200, 100) if DRY else (1000, 200)
SEL = json.load(open('../task20_chns/selected_cn.json'))
if DRY:
    xtr, ctr = M.load(['C']); d0, _ = M.load(['D']); td, ed = T.outcome_at(d0, 72); keep = ed | (td >= 72)
    xte = d0[keep].copy(); xte['died'] = ed[keep].astype(int); cte = {'pseudo': 'NHANES D'}
else:
    xtr, ctr = M.load(list('ABCDEF')); xte, cte = T.load_chns()
ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, T.SHARED); Z = T.transform_sub(xte, xtr, T.SHARED)
y = xte.died.to_numpy().astype(bool); n = len(y)
P = {'age+sex only': [], 'CBC only': T.CBC3, 'routine (shared items)': T.CN_ROUTINE_SHARED, 'matched': SEL['matched']['features'],
     'aware@routine': SEL['aware_at_routine']['features'], 'all 15 shared': T.SHARED}
S, R6, R9 = {}, {}, {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    S[k] = m.lp(Z[cols].to_numpy()); R9[k] = m.risk(Z[cols].to_numpy()); R6[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, 72), Z[cols].to_numpy())
rows = []
for k in P:
    hi = R9[k] >= .1
    rows.append(dict(panel=k, items=' '.join(T.describe_cn(P[k])), CN_yuan=T.cost_cn(P[k]), AUC=T.auc(S[k], y), OE_6y=float(y.sum() / R6[k].sum()),
                     per1000_cost=1000 * T.cost_cn(P[k]), per1000_flagged=1000 * float(hi.mean()), per1000_deaths_flagged=1000 * float((hi & y).mean()),
                     per1000_deaths_missed=1000 * float((~hi & y).mean())))
panels = pd.DataFrame(rows)
rng = np.random.default_rng(20261107)
pairs = [('matched', 'routine (shared items)'), ('aware@routine', 'routine (shared items)'), ('routine (shared items)', 'age+sex only')]
B = {f'dAUC {a} - {b}': [] for a, b in pairs}
for _ in range(NBOOT):
    i = rng.integers(0, n, n)
    if y[i].all() or (~y[i]).all():
        continue
    A = {k: T.auc(S[k][i], y[i]) for k in set(x_ for p in pairs for x_ in p)}
    for a, b in pairs:
        B[f'dAUC {a} - {b}'].append(A[a] - A[b])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
boot = {k: ci(v) for k, v in B.items()}
pc = panels.set_index('panel'); point = {f'dAUC {a} - {b}': float(pc.AUC[a] - pc.AUC[b]) for a, b in pairs}
# spline sensitivity
spl = {}
for k in ['routine (shared items)', 'matched', 'aware@routine']:
    A_, B_ = X.spline_design(Ztr, Z, M.BASE + list(P[k])); spl[k] = M.Cox(1.0).fit(A_, ttr, etr).lp(B_)
sp = {k: T.auc(v, y) for k, v in spl.items()}
d1, d2 = [], []
for _ in range(NSUB):
    i = rng.integers(0, n, n); ar = T.auc(spl['routine (shared items)'][i], y[i])
    d1.append(T.auc(spl['matched'][i], y[i]) - ar); d2.append(T.auc(spl['aware@routine'][i], y[i]) - ar)
sp.update({'dAUC matched - routine': sp['matched'] - sp['routine (shared items)'], 'dAUC matched CI': ci(d1), 'dAUC aware - routine': sp['aware@routine'] - sp['routine (shared items)'], 'dAUC aware CI': ci(d2)})
# subgroups: sex and age
sub = []
for g, msk in {'female': xte.sex.to_numpy() == 2, 'male': xte.sex.to_numpy() == 1, 'age40-59': xte.age.to_numpy() < 60, 'age60-79': xte.age.to_numpy() >= 60}.items():
    idx = np.flatnonzero(msk); row = dict(group=g, n=len(idx), deaths=int(y[idx].sum()))
    if y[idx].sum() < 20:
        sub.append(dict(row, note='fewer than 20 deaths')); continue
    for k in ['routine (shared items)', 'matched', 'aware@routine']:
        row['AUC ' + k] = T.auc(S[k][idx], y[idx])
    sub.append(row)
subd = pd.DataFrame(sub)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print('CHNS | TEST', cte, 'n', n, 'deaths', int(y.sum())); print(panels.round(4).to_string())
for k in boot:
    print(f'{k:50s} point {point[k]:+.4f}  95% [{boot[k][0]:+.4f}, {boot[k][1]:+.4f}]')
print('SPLINE', json.dumps(sp)); print(subd.round(4).to_string())
panels.to_csv(PRE + 'panels.csv', index=False)
json.dump(dict(cohort='CHNS', test=cte, n=n, deaths=int(y.sum()), boot=boot, point=point, spline=sp, subgroups=sub, prices={k: T.cost_cn(v) for k, v in P.items()},
               routine_full_price=SEL['routine_full_price']), open(PRE + 'endpoints.json', 'w'), indent=1, default=float)
