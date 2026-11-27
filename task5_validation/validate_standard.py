#!/usr/bin/env python3
"""Task 6: standard multi-feature methods on the held-out generic validation tasks (plan_standard_methods.md).
Same subsamples, standardisation and scoring as validate_generic.py, so rows pair with validation/<task>.csv.
Writes standard/<task>.csv with outer squared-error sums (plain and guarded test features) and the number of
distinct measurements each fitted model uses. Run with .env312:  python validate_standard.py <task> [--time]"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, ElasticNetCV, MultiTaskLassoCV, RidgeCV
import rulelib as L
from target_resolution import group_folds
from validate_generic import LEARNING, subsamples

warnings.filterwarnings('ignore', category=ConvergenceWarning)
N_ALPHAS = 30
RIDGE_ALPHAS = np.logspace(-2, 5, 22)
METHODS = ['Lasso', 'ElasticNet', 'MultiTaskLasso', 'Ridge']


def fit_predict(xz, yz, xt_plain, xt_guard, cv):
    """Return {method: (pred_plain, pred_guard, n_measurements)}; predictions on the standardized target scale."""
    out = {}
    for name, make in [('Lasso', lambda: LassoCV(n_alphas=N_ALPHAS, cv=cv, max_iter=5000, tol=1e-3)),
                       ('ElasticNet', lambda: ElasticNetCV(l1_ratio=0.5, n_alphas=N_ALPHAS, cv=cv, max_iter=5000, tol=1e-3))]:
        coefs, pp, pg = [], [], []
        for t in range(yz.shape[1]):
            m = make().fit(xz, yz[:, t])
            coefs.append(m.coef_); pp.append(m.predict(xt_plain)); pg.append(m.predict(xt_guard))
        out[name] = (np.column_stack(pp), np.column_stack(pg), int((np.abs(np.array(coefs)) > 0).any(axis=0).sum()))
    m = MultiTaskLassoCV(n_alphas=N_ALPHAS, cv=cv, max_iter=5000, tol=1e-3).fit(xz, yz)
    out['MultiTaskLasso'] = (m.predict(xt_plain), m.predict(xt_guard), int((np.abs(m.coef_) > 0).any(axis=0).sum()))
    m = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv).fit(xz, yz)
    out['Ridge'] = (m.predict(xt_plain), m.predict(xt_guard), xz.shape[1])
    return out


def main(name, timing=False):
    assert name not in LEARNING
    task, x, y, fam = L.load_task(name)
    rows = []
    t0 = time.time()
    for ev, seed, fold, n, r, sub, held, groups in subsamples(task, x):
        inner = seed + fold + 1
        xs, ys, gs = x[sub], y[sub], groups[sub]
        xm = xs.mean(axis=0); xsd = xs.std(axis=0, ddof=1); xsd = np.where(xsd == 0, 1.0, xsd)
        ym = ys.mean(axis=0); ysd = ys.std(axis=0, ddof=1); ysd = np.where(ysd == 0, 1.0, ysd)
        xz = (xs - xm) / xsd; yz = (ys - ym) / ysd
        xt = (x[held] - xm) / xsd
        zmax = np.abs(xz).max(axis=0)
        xt_g = np.where(np.abs(xt) > zmax + L.GUARD_MARGIN, 0.0, xt)   # same guard as R4: training mean for that measurement
        yt = (y[held] - ym) / ysd
        folds = group_folds(gs, 5, inner)
        idx = np.arange(len(xs))
        cv = [(np.setdiff1d(idx, h), h) for h in folds]
        res = fit_predict(xz, yz, xt, xt_g, cv)
        row = {'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub), 'n_label': 'all' if n is None else str(n),
               'rep': -1 if r is None else r, 'cells': yt.size}
        for k, (pp, pg, nm) in res.items():
            row[k] = float(((pp - yt) ** 2).sum()); row[f'g_{k}'] = float(((pg - yt) ** 2).sum()); row[f'nmeas_{k}'] = nm
        rows.append(row)
        if timing and len(rows) in (1, 5):
            print('n_train', len(sub), 'seconds per subsample so far', round((time.time() - t0) / len(rows), 2), flush=True)
            if len(rows) == 5:
                return
    (L.HERE / 'standard').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(L.HERE / 'standard' / f'{name}.csv', index=False)
    print(name, 'subsamples', len(rows), 'minutes', round((time.time() - t0) / 60, 1))


if __name__ == '__main__':
    main(sys.argv[1], '--time' in sys.argv)
