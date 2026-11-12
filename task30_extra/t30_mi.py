#!/usr/bin/env python3
"""Task 30 A (post hoc, PLAN.md): paired differences after multiple imputation in NHANES III, pooled by Rubin's rules."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy import stats
W = Path(__file__).resolve().parent.parent
os.chdir(W / 'task21_final'); sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
M.L.low_priority()
H, TAU, NB_BOOT = 108, 0.10, 200
SEL = json.load(open(W / 'task19_dm2/selected_all.json'))
P = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'matched': M.features_of(SEL['US']['matched']['items']),
     'aware': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
BASE = M.BASE
REF = {r['imp']: r for r in json.load(open(W / 'task21_final/t21_NH3_endpoints.json'))['mi']}

# sample accounting
x_all, cnt = M.load_nh3(keep_missing=True)
print('NHANES III counts', cnt, '| not imputed (fewer than half of analytes measured):', cnt['mortality_eligible'] - cnt['half_measured'], flush=True)

xm_tr, _ = M.load(list('ABCDEF'), keep_missing=True); xm, _ = M.load_nh3(keep_missing=True)
Zm_tr, Zm = M.transform(xm_tr, ref=xm_tr), M.transform(xm, ref=xm_tr); tm_tr, em_tr = M.outcome(xm_tr); tm, em = M.outcome(xm)
frac = xm[M.FEATURES].notna().mean(axis=1)
print('imputed sample', len(xm), 'deaths', int(em.sum()), '| share of analytes measured: median', round(float(frac.median()), 3), '| complete', int((frac == 1).sum()), flush=True)
IA = M.impute(Zm_tr, tm_tr, em_tr, m=5, seed=1); IB = M.impute(Zm, tm, em, m=5, seed=11)


def nb(r, t, e):
    kn = e | (t >= H); hi = r >= TAU; Wn = kn.sum()
    return float((hi & e & kn).sum() / Wn - (hi & ~e & kn).sum() / Wn * TAU / (1 - TAU))


per = []
for k in range(5):
    LP, RK = {}, {}
    for name, f in P.items():
        cols = BASE + list(f); Xtr, Xte = IA[k][cols].to_numpy(), IB[k][cols].to_numpy()
        m = M.Cox(1.0).fit(Xtr, tm_tr, em_tr); LP[name] = m.lp(Xte); RK[name] = T.risk_at(m, T.h0_at(m, Xtr, tm_tr, em_tr, H), Xte)
    C = {n_: M.cindex(LP[n_], tm, em) for n_ in P}
    for n_ in P:
        ref = REF[k]['aware@routine' if n_ == 'aware' else n_]
        assert abs(C[n_] - ref) < 1e-9, (k, n_, C[n_], ref)
    est = {'dC matched': C['matched'] - C['routine'], 'dC aware': C['aware'] - C['routine'],
           'dNB matched': nb(RK['matched'], tm, em) - nb(RK['routine'], tm, em), 'dNB aware': nb(RK['aware'], tm, em) - nb(RK['routine'], tm, em)}
    rng = np.random.default_rng(20261113 + k); n = len(tm); bs = {q: [] for q in est}
    for _ in range(NB_BOOT):
        i = rng.integers(0, n, n); Cb = {n_: M.cindex(LP[n_][i], tm[i], em[i]) for n_ in P}; nr = nb(RK['routine'][i], tm[i], em[i])
        bs['dC matched'].append(Cb['matched'] - Cb['routine']); bs['dC aware'].append(Cb['aware'] - Cb['routine'])
        bs['dNB matched'].append(nb(RK['matched'][i], tm[i], em[i]) - nr); bs['dNB aware'].append(nb(RK['aware'][i], tm[i], em[i]) - nr)
    for q in est:
        per.append(dict(imp=k, quantity=q, est=est[q], var=float(np.var(bs[q], ddof=1))))
    print('imputation', k, 'C', {a: round(b, 4) for a, b in C.items()}, {a: round(b, 4) for a, b in est.items()}, flush=True)

d = pd.DataFrame(per); out = []
m_, n = 5, len(tm)
for q, g in d.groupby('quantity', sort=False):
    Q, U, B = g.est.mean(), g['var'].mean(), g.est.var(ddof=1); Tv = U + (1 + 1 / m_) * B
    lam = (1 + 1 / m_) * B / Tv if Tv > 0 else 0.0
    nu_old = (m_ - 1) / max(lam, 1e-12) ** 2; nu_obs = (n - 1 + 1) / (n - 1 + 3) * (n - 1) * (1 - lam); nu = 1 / (1 / nu_old + 1 / nu_obs)
    h = stats.t.ppf(0.975, nu) * np.sqrt(Tv)
    out.append(dict(quantity=q, estimate=Q, low=Q - h, high=Q + h, within_var=U, between_var=B, df=nu))
res = pd.DataFrame(out)
pd.set_option('display.width', 200)
print(res.round(5).to_string(index=False))
d.to_csv(Path(__file__).parent / 't30_mi_per_imputation.csv', index=False); res.to_csv(Path(__file__).parent / 't30_mi_pooled.csv', index=False)
json.dump(dict(counts=cnt, n=int(n), deaths=int(em.sum()), pooled=out), open(Path(__file__).parent / 't30_mi.json', 'w'), indent=1, default=float)
