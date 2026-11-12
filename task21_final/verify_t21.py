#!/usr/bin/env python3
"""Independent check of Task 21 primary results: re-implemented Cox (scipy L-BFGS), Harrell's C, AUC (sklearn) and DerSimonian-Laird pooling
(own code); compares H1/H2 point estimates per cohort and the pooled decisions with the main pipeline. Usage: verify_t21.py [--dry]"""
import json
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import dm2lib as M
import t20lib as T
M.L.low_priority()
DRY = '--dry' in sys.argv; P = 'dry_' if DRY else ''
ok = True


def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, detail); ok &= bool(cond)


class Cox2:
    def fit(self, X, t, e, lam=1.0):
        self.m, self.s = X.mean(0), X.std(0); self.s[self.s == 0] = 1
        Z = (X - self.m) / self.s; o = np.argsort(t); Z, t, e = Z[o], t[o], e[o]; first = np.searchsorted(t, t, side='left')

        def f(b):
            eta = Z @ b; c = eta.max(); r = np.exp(eta - c)
            S0 = np.cumsum(r[::-1])[::-1][first]; S1 = np.cumsum((r[:, None] * Z)[::-1], 0)[::-1][first]
            return -(e * (eta - c - np.log(S0))).sum() + 0.5 * lam * b @ b, -(e[:, None] * (Z - S1 / S0[:, None])).sum(0) + lam * b
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=3000, gtol=1e-8)).x
        return self

    def lp(self, X):
        return ((X - self.m) / self.s) @ self.b


def cidx(s, t, e):
    o = np.argsort(t, kind='stable'); s, t, e = s[o], t[o], e[o]; num = den = 0.0
    for i in np.flatnonzero(e):
        later = s[np.searchsorted(t, t[i], side='right'):]; den += len(later); num += (later < s[i]).sum() + 0.5 * (later == s[i]).sum()
    return num / den


def dl(est, lo, hi):
    est = np.array(est); se = (np.array(hi) - np.array(lo)) / 3.92; w = 1 / se ** 2; fe = (w * est).sum() / w.sum()
    Q = (w * (est - fe) ** 2).sum(); k = len(est); tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2); m = (ws * est).sum() / ws.sum(); s = np.sqrt(1 / ws.sum())
    return m, m - 1.96 * s, m + 1.96 * s


SEL = json.load(open('../task19_dm2/selected_all.json')); SC = json.load(open('../task20_chns/selected_cn.json'))
F = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'matched': M.features_of(SEL['US']['matched']['items']),
     'aware@routine': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
xtr, _ = (M.load(['C']) if DRY else M.load(list('ABCDEF'))); ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr)
for coh, H in [('NH3', 108), ('GH', 60)]:
    if DRY:
        xte, _ = M.load(['D'])
    else:
        xte = M.load_nh3()[0] if coh == 'NH3' else pd.concat([T.load_gh(c)[0] for c in 'GH'])
    t, e = T.outcome_at(xte, H); Z = M.transform(xte, ref=xtr); E_ = json.load(open(f'{P}t21_{coh}_endpoints.json'))
    S = {k: Cox2().fit(Ztr[M.BASE + f].to_numpy(), ttr, etr).lp(Z[M.BASE + f].to_numpy()) for k, f in F.items()}
    for k in ['matched', 'aware@routine']:
        d = cidx(S[k], t, e) - cidx(S['routine'], t, e); key = f'dC {k} - routine'
        check(f'{coh} point {key}', abs(d - E_['point'][key]) < 0.002, f'{d:+.4f} vs {E_["point"][key]:+.4f}')
if DRY:
    xtr2, _ = M.load(['C']); d0, _ = M.load(['D']); td, ed = T.outcome_at(d0, 72); keep = ed | (td >= 72); xc = d0[keep].copy(); xc['died'] = ed[keep].astype(int)
else:
    xtr2, _ = M.load(list('ABCDEF')); xc, _ = T.load_chns()
t2, e2 = M.outcome(xtr2); Ztr2 = T.transform_sub(xtr2, xtr2, T.SHARED); Zc = T.transform_sub(xc, xtr2, T.SHARED); y = xc.died.to_numpy().astype(bool)
EC = json.load(open(f'{P}t21_CHNS_endpoints.json'))
G = {'routine': T.CN_ROUTINE_SHARED, 'matched': SC['matched']['features'], 'aware@routine': SC['aware_at_routine']['features']}
A = {k: roc_auc_score(y, Cox2().fit(Ztr2[M.BASE + list(f)].to_numpy(), t2, e2).lp(Zc[M.BASE + list(f)].to_numpy())) for k, f in G.items()}
for k in ['matched', 'aware@routine']:
    key = f'dAUC {k} - routine (shared items)'; d = A[k] - A['routine']
    check(f'CHNS point {key}', abs(d - EC['point'][key]) < 0.003, f'{d:+.4f} vs {EC["point"][key]:+.4f}')
PO = json.load(open(f'{P}t21_pooled_interim.json'))
for h in ['H1', 'H2']:
    pc = PO[h]['per_cohort']; m, lo, hi = dl([v['est'] for v in pc.values()], [v['lo'] for v in pc.values()], [v['hi'] for v in pc.values()])
    check(f'{h} pooled estimate', abs(m - PO[h]['pooled']['est']) < 1e-6, f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]')
check('H1 decision', (lo if False else dl([v['est'] for v in PO['H1']['per_cohort'].values()], [v['lo'] for v in PO['H1']['per_cohort'].values()],
                                          [v['hi'] for v in PO['H1']['per_cohort'].values()])[1] > -0.01) == PO['H1']['pass'])
print('VERIFY PASSED' if ok else 'VERIFY FAILED')
