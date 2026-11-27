#!/usr/bin/env python3
"""Task 32a (post hoc, PLAN.md): NBPHS 2025 programme reference (2017 subset measured in CHNS plus HbA1c) in CHNS.
Selection on NHANES 1999-2010 exactly as Task 29 cn mode (beam width 4, lambda 1, 5-fold CV Cox log-likelihood, seed 1, Chinese prices);
validation in CHNS exactly as t29_valid.py (AUC; 72-month risk, threshold 6%; 1000 paired bootstrap resamples, seed 20261125).
The Task 29 cache is read but never written. Output: t32a_CHNS.json"""
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task29_revision'))
import t29lib as Q  # noqa: E402
M, T = Q.M, Q.T
M.L.low_priority()
T.install_cn()

x, _ = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
cache = Q.load_cache(); n0 = len(cache)
sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds))  # noqa: E731

menu = T.SHARED; pool = list(T.CN_ITEMS)
cbc = [a for a in menu if a in M.CBC2]
ref17 = [a for a in Q.NBPHS if a in menu and a not in M.CBC2] + cbc
ref25 = ref17 + ['hba1c']
pr17, pr25 = T.cost_cn(ref17), T.cost_cn(ref25)
cv17, cv25 = sc(ref17), sc(ref25)
print('ref17', pr17, round(cv17, 2), 'ref25', pr25, round(cv25, 2), flush=True)
for B in [60.0, 70.0, 80.0, pr25]:
    t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, 'CN', pool=pool, cache=cache)
    print('CN', B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
ok = [k for k in cache if set(k) <= set(menu)]
m_ = sorted(((T.cost_cn(k), -cache[k]), k) for k in ok if cache[k] >= cv25 - 1e-9)[0][1]
a_ = max(((cache[k], -T.cost_cn(k)), k) for k in ok if T.cost_cn(k) <= pr25 + 1e-9)[1]
sel = dict(ref17=dict(features=ref17, price=pr17, cv=cv17), ref25=dict(features=ref25, price=pr25, cv=cv25),
           matched25=dict(features=list(m_), price=T.cost_cn(m_), cv=cache[m_]), budget25=dict(features=list(a_), price=T.cost_cn(a_), cv=cache[a_]),
           cache_start=n0, cache_end=len(cache))
print(json.dumps(sel), flush=True)

# ---------------- validation in CHNS (as t29_valid.py) ----------------
S0 = json.load(open(Q.W / 'task20_chns' / 'selected_cn.json'))
xte, _ = T.load_chns(); y = xte.died.to_numpy().astype(bool)
Ztr, Zt = T.transform_sub(x, x, menu), T.transform_sub(xte, x, menu)
H, TAU = 72, .06
P = {'age+sex only': [], 'NBPHS 2017': ref17, 'NBPHS 2025': ref25, 'matched to NBPHS 2025': sel['matched25']['features'],
     'NBPHS 2025 budget': sel['budget25']['features'], 'aware@routine': S0['aware_at_routine']['features']}
LP, R = {}, {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), t, e)
    LP[k] = m.lp(Zt[cols].to_numpy()); R[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), t, e, H), Zt[cols].to_numpy())
disc = lambda s, i: T.auc(s[i], y[i])  # noqa: E731
nbf = lambda r, i: Q.nb_ipcw(r[i], np.ones(len(i)), y[i], TAU)  # noqa: E731
n = len(y); allidx = np.arange(n)
pairs = [('NBPHS 2025', 'age+sex only'), ('NBPHS 2025', 'NBPHS 2017'), ('NBPHS 2025 budget', 'NBPHS 2025'), ('matched to NBPHS 2025', 'NBPHS 2025'),
         ('aware@routine', 'NBPHS 2025')]
out = dict(cohort='CHNS', n=n, deaths=int(y.sum()), horizon=H, tau=TAU, selection=sel,
           panels={k: dict(features=list(f), price=(T.cost_cn(f) if f else 0.0), order=T.describe_cn(f), disc=disc(LP[k], allidx),
                           NB=nbf(R[k], allidx), mean_risk=float(R[k].mean())) for k, f in P.items()})
rng = np.random.default_rng(20261125); bc = {f'{a} - {b}': [] for a, b in pairs}; bn = {f'{a} - {b}': [] for a, b in pairs}
for _ in range(1000):
    i = rng.integers(0, n, n)
    if y[i].all() or (~y[i]).all():
        continue
    for a, b in pairs:
        bc[f'{a} - {b}'].append(disc(LP[a], i) - disc(LP[b], i)); bn[f'{a} - {b}'].append(nbf(R[a], i) - nbf(R[b], i))
pp = out['panels']
out['contrasts'] = {f'{a} - {b}': dict(point=pp[a]['disc'] - pp[b]['disc'], ci=Q.ci(bc[f'{a} - {b}'])) for a, b in pairs}
out['nb_contrasts'] = {f'{a} - {b}': dict(point=pp[a]['NB'] - pp[b]['NB'], ci=Q.ci(bn[f'{a} - {b}'])) for a, b in pairs}
json.dump(out, open(HERE / 't32a_CHNS.json', 'w'), indent=1, default=float)
print('n', n, 'deaths', int(y.sum()))
for k, v in pp.items():
    print(f"{k:24s} {v['price']:6.1f} AUC {v['disc']:.4f} NB {v['NB']:.4f} {v['order']}")
for k, v in out['contrasts'].items():
    w = out['nb_contrasts'][k]
    print(f"{k:44s} AUC {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  NB {w['point']:+.4f} [{w['ci'][0]:+.4f}, {w['ci'][1]:+.4f}]")
