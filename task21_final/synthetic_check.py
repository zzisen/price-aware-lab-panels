"""Task 21 code checks on synthetic data (no survey data)."""
import numpy as np
import dm2lib as M
import t21lib as X
ok = True


def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, detail); ok &= bool(cond)


f = M.features_of
check('identity price perturbation = US bill', all(X.make_cost_us({})(f(s)) == M.cost_us(f(s)) for s in (['CMP', 'CBC'], ['BMP'], ['CMP', 'CBC', 'LIPID', 'hba1c'], ['acr'], ['glucose', 'crp'])))
check('perturbed CBC price doubles CBC component', X.make_cost_us({'CBC': 2.0})(f(['CBC'])) == round(2 * 7.77 + 9.34, 2))
check('AU coning keeps top 3', X.cost_au_coned(f(['CMP', 'CBC', 'LIPID', 'hba1c'])) == round(17.80 + 17.70 + 16.80 + 5.95, 2), X.cost_au_coned(f(['CMP', 'CBC', 'LIPID', 'hba1c'])))
rng = np.random.default_rng(3); n, p = 4000, 8
Z = rng.normal(size=(n, p)); T = rng.exponential(1 / (0.003 * np.exp(0.9 * Z[:, 0] - 0.6 * Z[:, 1]))); t = np.minimum(T, 108); e = T <= 108
path = X.lasso_path(Z, t, e, np.ones(p), n_lambda=15)
first = [np.flatnonzero(np.abs(b) > 0) for _, b in path]
entry = next(s for s in first if len(s) > 0)
check('lasso enters a true signal first', set(entry) <= {0, 1}, entry)
check('lasso path ends with both signals active', {0, 1} <= set(first[-1]), first[-1])
w = np.ones(p); w[0] = 50
path2 = X.lasso_path(Z, t, e, w, n_lambda=15)
check('heavy price weight delays an expensive feature', next(i for i, (_, b) in enumerate(path2) if b[0] != 0) > next(i for i, (_, b) in enumerate(path) if b[0] != 0))
v = rng.normal(size=500); B = X.rcs(v, X.rcs_knots(v))
check('spline basis shape and linear tail', B.shape == (500, 2) and np.allclose(np.diff(X.rcs(np.array([10.0, 11.0, 12.0]), X.rcs_knots(v))[:, 1], 2), 0, atol=1e-8))
pr = X.pool([0.01, 0.01, 0.01], [0.0, 0.0, 0.0], [0.02, 0.02, 0.02])
check('pool of identical estimates', abs(pr['est'] - 0.01) < 1e-12 and pr['I2'] == 0 and abs((pr['hi'] - pr['lo']) - 2 * 1.96 * 0.0051 / np.sqrt(3)) < 1e-3, pr)
pr2 = X.pool([0.0, 0.05], [-0.01, 0.04], [0.01, 0.06])
check('pool of discordant estimates has I2 > 0.9', pr2['I2'] > 0.9, round(pr2['I2'], 3))
par = dict(X.HE_BASE)
h = X.he_eval([1, 0, 1, 0], [1, 1, 0, 0], [10, 10, 10, 10], par)
check('health-economic arithmetic', abs(h['cost'] - (10 + 300 * 0.5)) < 1e-9 and abs(h['qaly'] - 0.05 * 0.25 * X.discounted_ly(8, 0.03) * 0.8) < 1e-12, h)
print('SYNTHETIC CHECK PASSED' if ok else 'SYNTHETIC CHECK FAILED')
