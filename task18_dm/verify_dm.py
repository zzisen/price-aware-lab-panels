#!/usr/bin/env python3
"""Independent check of the Task 18 primary results (written before the held-out run). Reuses only dmlib's file loading and price table;
re-implements the Cox fit (scipy L-BFGS on the penalised Breslow partial likelihood), the 9-year risk, Harrell's C, net benefit, the
flip models (sklearn logistic regression) and the bootstrap (different seed). Usage: verify_dm.py [--dry]"""
import json
import sys
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
import dmlib as L
L.low_priority()
DRY = '--dry' in sys.argv
PRE = 'dry_' if DRY else ''
SEL = json.load(open('selected_panels.json'))
END = json.load(open(f'{PRE}endpoints.json'))
TRAIN, TEST = (['C'], ('D', ['D'])) if DRY else (['C', 'D'], ('EF', ['E', 'F']))
H = 108.0
ok = True


def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, detail); ok &= bool(cond)


# hand-computed Medicare bills (CLFS 2026 Q4) for the bundles used in the criteria
hand = {('CMP', 'CBC', 'LIPID', 'hba1c'): 10.56 + 7.77 + 13.39 + 9.71 + 9.34, ('BMP',): 8.46 + 9.34, ('LIPID',): 13.39 + 9.34, ('CBC', 'CMP'): 10.56 + 7.77 + 9.34}
for k, v in hand.items():
    check(f'price {k}', abs(L.cost(list(k)) - round(v, 2)) < 1e-9, L.cost(list(k)))


class Cox2:
    def fit(self, X, t, e, lam=1.0):
        self.m, self.s = X.mean(0), X.std(0); self.s[self.s == 0] = 1
        Z = (X - self.m) / self.s
        o = np.argsort(t); Z, t, e = Z[o], t[o], e[o]
        first = np.searchsorted(t, t, side='left')      # risk set {j: t_j >= t_i}

        def f(b):
            eta = Z @ b; c = eta.max(); r = np.exp(eta - c)
            S0 = np.cumsum(r[::-1])[::-1][first]; S1 = np.cumsum((r[:, None] * Z)[::-1], 0)[::-1][first]
            nll = -(e * (eta - c - np.log(S0))).sum() + 0.5 * lam * b @ b
            g = -(e[:, None] * (Z - S1 / S0[:, None])).sum(0) + lam * b
            return nll, g
        self.b = minimize(f, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B', options=dict(maxiter=2000, gtol=1e-8)).x
        eta = Z @ self.b; r = np.exp(eta); S0 = np.cumsum(r[::-1])[::-1][first]
        self.H0 = float((e / S0)[t <= H].sum())
        return self

    def lp(self, X):
        return ((X - self.m) / self.s) @ self.b

    def risk(self, X):
        return 1 - np.exp(-self.H0 * np.exp(self.lp(X)))


def cidx(s, t, e):
    o = np.argsort(t, kind='stable'); s, t, e = s[o], t[o], e[o]
    num = den = 0.0
    for i in np.flatnonzero(e):
        j = np.searchsorted(t, t[i], side='right')      # strictly later times
        later = s[j:]
        den += len(later); num += (later < s[i]).sum() + 0.5 * (later == s[i]).sum()
    return num / den


def nb(r, t, e, tau):
    kn = e | (t >= H); n = kn.sum()
    return ((r >= tau) & e & kn).sum() / n - ((r >= tau) & ~e & kn).sum() / n * tau / (1 - tau)


xtr, _ = L.load(TRAIN); ttr, etr = L.outcome(xtr); Ztr = L.transform(xtr)
x, _ = L.load(TEST[1]); t, e = L.outcome(x); Z = L.transform(x, ref=xtr)
P = {'routine': L.STANDARD['CMP+CBC+LIPID+A1C'], 'matched': SEL['matched']['items'], 'BMP': ['BMP'], 'LIPID': ['LIPID'],
     'aBMP': SEL['grid'][f"{L.cost(['BMP']):.2f}"]['items'], 'aLIPID': SEL['grid'][f"{L.cost(['LIPID']):.2f}"]['items'],
     'none': [], 's1': SEL['stage1']['items'], 's2': SEL['stage2']['items']}
cols = lambda k: ['age', 'female'] + L.features_of(P[k])  # noqa: E731
M = {k: Cox2().fit(Ztr[cols(k)].to_numpy(), ttr, etr) for k in P}
S = {k: M[k].lp(Z[cols(k)].to_numpy()) for k in P}
R = {k: M[k].risk(Z[cols(k)].to_numpy()) for k in P}
pt = END[TEST[0]]['point']; bt = END[TEST[0]]['boot']
for a, b, key in [('matched', 'routine', 'dC matched - std:CMP+CBC+LIPID+A1C'), ('aBMP', 'BMP', 'dC aware@BMP - std:BMP'), ('aLIPID', 'LIPID', 'dC aware@LIPID - std:LIPID')]:
    d = cidx(S[a], t, e) - cidx(S[b], t, e)
    check(f'point {key}', abs(d - pt[key]) < 0.002, f'{d:+.4f} vs {pt[key]:+.4f}')

# sequential policy at tau = 0.1, eps = 0.02 with independently fitted flip models
folds = np.array_split(np.random.default_rng(1).permutation(len(ttr)), 5)
rcv = np.zeros((3, len(ttr)))
for te in folds:
    tr = np.setdiff1d(np.arange(len(ttr)), te)
    for s, k in enumerate(['none', 's1', 's2']):
        rcv[s, te] = Cox2().fit(Ztr[cols(k)].to_numpy()[tr], ttr[tr], etr[tr]).risk(Ztr[cols(k)].to_numpy()[te])
tau, eps = 0.10, 0.02
feat = lambda r: (lambda d: np.column_stack([d, d ** 2, np.abs(d)]))(np.log(np.clip(r, 1e-4, 1 - 1e-4) / (1 - np.clip(r, 1e-4, 1 - 1e-4))) - np.log(tau / (1 - tau)))  # noqa: E731
fl = [LogisticRegression(C=1e6, max_iter=5000).fit(feat(rcv[s]), (rcv[s] >= tau) != (rcv[2] >= tau)) for s in range(2)]
c = [0.0, L.cost(P['s1']), L.cost(P['s2'])]
go1 = fl[0].predict_proba(feat(R['none']))[:, 1] >= eps
go2 = go1 & (fl[1].predict_proba(feat(R['s1']))[:, 1] >= eps)
stage = go1.astype(int) + go2
risk = np.choose(stage, [R['none'], R['s1'], R['s2']]); cp = np.array(c)[stage]
share = cp.mean() / L.cost(P['routine'])
check('point cost_share seq0.1', abs(share - pt['cost_share seq0.1']) < 0.02, f'{share:.4f} vs {pt["cost_share seq0.1"]:.4f}')
dnb = nb(risk, t, e, tau) - nb(R['routine'], t, e, tau)
check('point dNB seq0.1 - routine', abs(dnb - pt['dNB seq0.1 - routine']) < 0.003, f'{dnb:+.4f} vs {pt["dNB seq0.1 - routine"]:+.4f}')
agree = ((risk >= tau) == (R['s2'] >= tau)).mean()
print('agreement with stage 2 for everyone', round(float(agree), 4))

# bootstrap with a different seed
rng = np.random.default_rng(777); D = {'P1': [], 'BMP': [], 'LIPID': [], 'share': [], 'dnb': []}
for _ in range(300 if not DRY else 100):
    i = rng.integers(0, len(t), len(t)); ti, ei = t[i], e[i]
    cr = cidx(S['routine'][i], ti, ei)
    D['P1'].append(cidx(S['matched'][i], ti, ei) - cr)
    D['BMP'].append(cidx(S['aBMP'][i], ti, ei) - cidx(S['BMP'][i], ti, ei))
    D['LIPID'].append(cidx(S['aLIPID'][i], ti, ei) - cidx(S['LIPID'][i], ti, ei))
    D['share'].append(cp[i].mean() / L.cost(P['routine'])); D['dnb'].append(nb(risk[i], ti, ei, tau) - nb(R['routine'][i], ti, ei, tau))
q = {k: (np.quantile(v, .025), np.quantile(v, .975)) for k, v in D.items()}
for k, v in q.items():
    print(f'{k:6s} 95% [{v[0]:+.4f}, {v[1]:+.4f}]')
P1 = L.cost(P['matched']) <= 0.8 * L.cost(P['routine']) and q['P1'][0] > -0.01
P2 = q['share'][1] <= 0.60 and q['dnb'][0] > -0.005 and agree >= 0.97
P3 = P['aBMP'] != P['BMP'] and P['aLIPID'] != P['LIPID'] and q['BMP'][0] > 0 and q['LIPID'][0] > 0
cr = END[TEST[0]]['crit']
for k, v in [('P1', P1), ('P2', P2), ('P3', P3)]:
    check(f'criterion {k} same verdict', v == cr[k], f'verify {v} / run {cr[k]}')
print('VERIFY PASSED' if ok else 'VERIFY FAILED')
