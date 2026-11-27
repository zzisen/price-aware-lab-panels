#!/usr/bin/env python3
"""Task 18 code check on synthetic data with a known answer (no NHANES data):
1. the Cox fit recovers known coefficients and the Breslow 9-year risk matches the true risk;
2. Harrell's C matches a brute-force count;
3. the cost function reproduces hand-computed Medicare bills;
4. the beam search finds the informative item when one item carries all the signal;
5. the sequential policy sends nobody on when eps = 1 and everybody on when eps = 0."""
import numpy as np
import pandas as pd
import dmlib as L

rng = np.random.default_rng(20261106)
ok = True


def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, detail)
    ok &= bool(cond)


# 1. Cox recovery: exponential hazard 0.002/month * exp(x b), censoring at 108 months
n, beta = 20000, np.array([0.8, -0.5, 0.0])
X = rng.normal(size=(n, 3))
T = rng.exponential(1 / (0.002 * np.exp(X @ beta)))
t = np.minimum(T, L.HORIZON); e = T <= L.HORIZON
m = L.Cox(1e-6).fit(X, t, e)
b = m.b / m.s
check('Cox coefficients', np.abs(b - beta).max() < 0.05, np.round(b, 3))
true = 1 - np.exp(-0.002 * L.HORIZON * np.exp(X @ beta))
check('9-year risk', np.abs(m.risk(X) - true).mean() < 0.01, round(float(np.abs(m.risk(X) - true).mean()), 4))

# 2. Harrell's C against brute force
k = 400; lp = rng.normal(size=k); tt = rng.integers(1, 50, size=k).astype(float); ee = rng.random(k) < 0.5
num = den = 0.0
for i in range(k):
    for j in range(k):
        if ee[i] and tt[i] < tt[j]:
            den += 1; num += 1.0 if lp[i] > lp[j] else 0.5 if lp[i] == lp[j] else 0.0
check('Harrell C', abs(L.cindex(lp, tt, ee) - num / den) < 1e-12, round(num / den, 4))

# 3. cost
check('cost CMP', L.cost(['CMP']) == round(10.56 + 9.34, 2))
check('cost glucose alone', L.cost(['glucose']) == round(3.93 + 9.34, 2))
check('cost glucose+creatinine uses the cheaper BMP', L.cost(['glucose', 'creatinine']) == round(8.46 + 9.34, 2))
check('cost BMP members bought singly use the BMP price', L.cost(['glucose', 'bun', 'creatinine', 'sodium', 'potassium', 'chloride', 'co2', 'calcium']) == round(8.46 + 9.34, 2))
check('cost urine ACR needs no venipuncture', L.cost(['acr']) == round(5.78 + 5.18, 2))
check('cost CMP + albumin = CMP', L.cost(['CMP', 'albumin']) == L.cost(['CMP']))

# 4. beam search: only 'albumin' carries signal
cols = L.FEATURES
Z = pd.DataFrame(rng.normal(size=(3000, len(cols))), columns=cols)
Z['age'] = rng.normal(size=3000); Z['female'] = rng.integers(0, 2, 3000).astype(float)
T = rng.exponential(1 / (0.003 * np.exp(1.0 * Z.albumin.to_numpy())))
t = np.minimum(T, L.HORIZON); e = T <= L.HORIZON
folds = np.array_split(rng.permutation(3000), 5)
sel, c, _ = L.beam_budget(Z, t, e, 15.0, 1.0, folds, width=2, pool=['albumin', 'glucose', 'crp', 'CBC', 'acr'])
check('beam picks the informative item', 'albumin' in sel, sel)

# 5. sequential policy extremes
r = np.vstack([rng.random(500), rng.random(500), rng.random(500)])
fl = [L.FlipModel(0.2).fit(r[0], r[2]), L.FlipModel(0.2).fit(r[1], r[2])]
st, _, _ = L.sequential_policy(r, fl, 1.01, [0, 10, 20]); check('eps above 1: nobody moves on', (st == 0).all())
st, _, _ = L.sequential_policy(r, fl, 0.0, [0, 10, 20]); check('eps 0: everybody reaches the last stage', (st == 2).all())
print('SYNTHETIC CHECK PASSED' if ok else 'SYNTHETIC CHECK FAILED')
