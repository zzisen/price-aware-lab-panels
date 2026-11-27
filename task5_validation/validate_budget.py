#!/usr/bin/env python3
"""Task 7: budget-matched multi-task lasso on the held-out generic validation tasks (plan_budget_matched.md).
Same subsamples, standardisation and scoring as validate_generic.py, so rows pair with validation/<task>.csv.
Writes budget/<task>.csv. Run with .env312:  python validate_budget.py <task> [--time]"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path
import rulelib as L
from target_resolution import group_folds
from validate_generic import LEARNING, subsamples

warnings.filterwarnings('ignore', category=ConvergenceWarning)
N_GRID = 100
MAX_SUPPORT = 12
BUDGETS = {'k1': 1, 'k3': 3, 'k9': 9}


def path(x, y, alphas):
    """Multi-task lasso path on centred data: (intercepts per alpha, coefs [alpha, feature, target])."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    _, coefs, _ = lasso_path(x - xm, y - ym, alphas=alphas, max_iter=5000, tol=1e-3)
    coefs = np.transpose(coefs, (2, 1, 0))                    # (alpha, feature, target)
    return xm, ym, coefs


def relaxed(x, y, support):
    """Per-target least squares on the selected features (tiny ridge); returns (xm, ym, coef [feature, target])."""
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    coef = np.zeros((x.shape[1], y.shape[1]))
    if len(support):
        xs = x[:, support] - xm[support]
        coef[support] = np.linalg.solve(xs.T @ xs + 1e-8 * np.eye(len(support)), xs.T @ (y - ym))
    return xm, ym, coef


def predict(xt, xm, ym, coef):
    return ym + (xt - xm) @ coef


def grid(xz, yz):
    a_max = np.sqrt((((xz - xz.mean(0)).T @ (yz - yz.mean(0))) ** 2).sum(axis=1)).max() / len(xz)
    alphas = a_max * np.logspace(0, -1, N_GRID)
    _, _, coefs = path(xz, yz, alphas)
    size = (np.abs(coefs) > 0).any(axis=2).sum(axis=1)
    over = np.flatnonzero(size > MAX_SUPPORT)
    keep = N_GRID if not len(over) else over[0] + 1
    return alphas[:keep], coefs[:keep], size[:keep]


def cv_errors(xz, yz, alphas, folds):
    """Inner-CV squared error of each penalty, for the lasso fit and the relaxed refit."""
    e_l = np.zeros(len(alphas)); e_r = np.zeros(len(alphas))
    for held in folds:
        tr = np.ones(len(xz), bool); tr[held] = False
        xm, ym, coefs = path(xz[tr], yz[tr], alphas)
        for a in range(len(alphas)):
            e_l[a] += ((predict(xz[held], xm, ym, coefs[a]) - yz[held]) ** 2).sum()
            sup = np.flatnonzero((np.abs(coefs[a]) > 0).any(axis=1))
            e_r[a] += ((predict(xz[held], *relaxed(xz[tr], yz[tr], sup)) - yz[held]) ** 2).sum()
    return e_l, e_r


def main(name, timing=False):
    assert name not in LEARNING
    task, x, y, fam = L.load_task(name)
    v = pd.read_csv(L.HERE / 'validation' / f'{name}.csv')
    rows = []
    t0 = time.time()
    for i, (ev, seed, fold, n, r, sub, held, groups) in enumerate(subsamples(task, x)):
        inner = seed + fold + 1
        xs, ys, gs = x[sub], y[sub], groups[sub]
        assert v.outer_seed[i] == seed and v.fold[i] == fold and v.n_train[i] == len(sub)
        # R4's realised measurement count (guarded inner-CV actions of its chosen resolution)
        Lg, _ = L.unit_losses_g(xs, ys, gs, 5, inner)
        ch = L.choices(Lg.mean(axis=0), fam)
        r4 = v.choice_R4[i]
        k_r4 = 0 if r4 == 'C0' else len(set(ch[r4].tolist()))
        # standardisation and guard exactly as Task 6
        xm = xs.mean(axis=0); xsd = xs.std(axis=0, ddof=1); xsd = np.where(xsd == 0, 1.0, xsd)
        ym = ys.mean(axis=0); ysd = ys.std(axis=0, ddof=1); ysd = np.where(ysd == 0, 1.0, ysd)
        xz = (xs - xm) / xsd; yz = (ys - ym) / ysd
        xt = (x[held] - xm) / xsd
        xt_g = np.where(np.abs(xt) > np.abs(xz).max(axis=0) + L.GUARD_MARGIN, 0.0, xt)
        yt = (y[held] - ym) / ysd
        folds = group_folds(gs, 5, inner)
        alphas, coefs, size = grid(xz, yz)
        e_l, e_r = cv_errors(xz, yz, alphas, folds)
        row = {'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub), 'n_label': 'all' if n is None else str(n),
               'rep': -1 if r is None else r, 'cells': yt.size, 'k_R4': k_r4}
        for label, k in {'matched': k_r4, **BUDGETS}.items():
            ok = np.flatnonzero(size <= k) if k > 0 else np.array([], int)
            for var, err in (('MTL', e_l), ('MTLrelaxed', e_r)):
                if len(ok):
                    a = ok[np.argmin(err[ok])]
                    if var == 'MTL':
                        fit = (xz.mean(axis=0), yz.mean(axis=0), coefs[a])
                    else:
                        fit = relaxed(xz, yz, np.flatnonzero((np.abs(coefs[a]) > 0).any(axis=1)))
                    pp, pg, used = predict(xt, *fit), predict(xt_g, *fit), int(size[a])
                else:
                    pp = pg = np.zeros_like(yt); used = 0
                row[f'{var}_{label}'] = float(((pp - yt) ** 2).sum())
                row[f'g_{var}_{label}'] = float(((pg - yt) ** 2).sum())
                row[f'nmeas_{var}_{label}'] = used
        rows.append(row)
        if timing and len(rows) in (1, 5):
            print('n_train', len(sub), 'sec/subsample', round((time.time() - t0) / len(rows), 2), flush=True)
            if len(rows) == 5:
                return
        if len(rows) % 100 == 0:
            print(name, 'progress', len(rows), 'minutes', round((time.time() - t0) / 60, 1), flush=True)
    (L.HERE / 'budget').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(L.HERE / 'budget' / f'{name}.csv', index=False)
    print(name, 'subsamples', len(rows), 'minutes', round((time.time() - t0) / 60, 1))


if __name__ == '__main__':
    main(sys.argv[1], '--time' in sys.argv)
