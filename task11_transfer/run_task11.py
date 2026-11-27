#!/usr/bin/env python3
"""Task 11 (plan_transfer.md): learn R4's measurement choice on platform A, evaluate on B-only units of platform B
with n_cal = 25 / 50 calibration units (20 draws each). Every rule and comparator is a frozen Task 5/6/7 function,
called unchanged. Writes results/<task>.csv (one row per draw, loss sums on the calibration-standardised scale).
Run with .env312:  python run_task11.py <task>"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L  # noqa: E402
import dev_eval as D  # noqa: E402
import validate_budget as VB  # noqa: E402
from validate_standard import RIDGE_ALPHAS  # noqa: E402
from target_resolution import group_folds  # noqa: E402

warnings.filterwarnings('ignore', category=ConvergenceWarning)
N_CAL, REPS, SEED = [25, 50], 20, 20261102


def load(task, side):
    m = pd.read_csv(HERE / 'matrices' / f"{task['task']}_{side}.csv")
    return m[[f'cand_{c}' for c in task['candidates']]].to_numpy(float), m[task['targets']].to_numpy(float)


def standardise(xs, ys):
    xm = xs.mean(axis=0); xsd = xs.std(axis=0, ddof=1); xsd = np.where(xsd == 0, 1.0, xsd)
    ym = ys.mean(axis=0); ysd = ys.std(axis=0, ddof=1); ysd = np.where(ysd == 0, 1.0, ysd)
    return xm, xsd, ym, ysd


def r4_fit(x, y, fam, inner):
    """R4 exactly as in validate_generic / cache_subsample_g: guarded inner-CV actions, resolution by the
    averaged three-split middle layer, ties to the coarser candidate."""
    gs = np.arange(len(x))
    Lg, _ = L.unit_losses_g(x, y, gs, 5, inner)
    ch = L.choices(Lg.mean(axis=0), fam)
    mids = [L.middle_rows_g(x, y, gs, fam, inner, off) for off in L.MIDDLE_OFFSETS]
    r4 = D.argmin_rule(D.averaged_units({'middle': mids, 'groups': gs}), ['C0'] + L.RES)
    return r4, (None if r4 == 'C0' else np.asarray(ch[r4], int))


def mtl_support(xz, yz, k, folds):
    """Budget-matched relaxed MTL support, as validate_budget's 'matched' MTLrelaxed."""
    if k <= 0:
        return np.array([], int)
    alphas, coefs, size = VB.grid(xz, yz)
    _, e_r = VB.cv_errors(xz, yz, alphas, folds)
    ok = np.flatnonzero(size <= k)
    if not len(ok):
        return np.array([], int)
    a = ok[np.argmin(e_r[ok])]
    return np.flatnonzero((np.abs(coefs[a]) > 0).any(axis=1))


def main(name):
    task = next(t for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    fam = np.asarray(task['families'])
    xA, yA = load(task, 'A'); xB, yB = load(task, 'B')
    T = yA.shape[1]

    # ---- platform A: learn the rule once ----
    innerA = SEED + task['code']
    r4A, actA = r4_fit(xA, yA, fam, innerA)
    kT = 0 if actA is None else len(set(actA.tolist()))
    xmA, xsdA, ymA, ysdA = standardise(xA, yA)
    supA = mtl_support((xA - xmA) / xsdA, (yA - ymA) / ysdA, kT, group_folds(np.arange(len(xA)), 5, innerA))
    if actA is not None:      # full transfer: raw-scale single-feature OLS fitted on A
        xa = xA[:, actA]; ya = yA
        b = ((xa - xa.mean(0)) * (ya - ya.mean(0))).sum(0) / ((xa - xa.mean(0)) ** 2).sum(0)
        a0 = ya.mean(0) - b * xa.mean(0)
    print(name, 'A: R4 resolution', r4A, 'measurements', kT, '| transferred MTL support', len(supA), flush=True)

    # ---- platform B: calibrate on n_cal B-only units, test on the rest ----
    gB = np.arange(len(xB))
    rows = []
    for n_cal in N_CAL:
        for r in range(REPS):
            rng = np.random.default_rng([SEED, task['code'], n_cal, r])
            cal = np.sort(rng.choice(len(xB), n_cal, replace=False)); test = np.setdiff1d(gB, cal)
            inner = SEED + task['code'] + 1000 * n_cal + r
            xs, ys = xB[cal], yB[cal]
            cand, c0, _ = L.guarded_losses(xs, ys[:, :, None], xB[test], yB[test][:, :, None])
            nt = len(test); tt = np.arange(T)
            row = {'n_cal': n_cal, 'rep': r, 'n_test': nt, 'cells': nt * T, 'C0': float(c0.sum())}
            # transferred R4: A's choice, refitted on the calibration units with the guard
            row['R4_transfer'] = row['C0'] if actA is None else float(cand[np.arange(nt)[:, None], actA[None, :], tt[None, :]].sum())
            # R4 from scratch on the calibration units (as validate_generic)
            g = L.cache_subsample_g(xB, yB, gB, fam, cal, test, inner)
            r4s = D.argmin_rule(D.averaged_units({'middle': g['middle_g'], 'groups': gB[cal]}), ['C0'] + L.RES)
            row['R4_scratch'] = float(g['outer_g'][r4s])
            Lg, _ = L.unit_losses_g(xs, ys, gB[cal], 5, inner)
            ch = L.choices(Lg.mean(axis=0), fam)
            k_s = 0 if r4s == 'C0' else len(set(ch[r4s].tolist()))
            # standardisation and guard exactly as Task 6 / 7
            xm, xsd, ym, ysd = standardise(xs, ys)
            xz, yz = (xs - xm) / xsd, (ys - ym) / ysd
            xt = (xB[test] - xm) / xsd
            xt_g = np.where(np.abs(xt) > np.abs(xz).max(axis=0) + L.GUARD_MARGIN, 0.0, xt)
            yt = (yB[test] - ym) / ysd
            assert abs((yt ** 2).sum() - row['C0']) < 1e-6 * max(1.0, row['C0'])
            folds = group_folds(gB[cal], 5, inner)
            sup_s = mtl_support(xz, yz, k_s, folds)
            row['MTL_matched_scratch'] = float(((VB.predict(xt_g, *VB.relaxed(xz, yz, sup_s)) - yt) ** 2).sum())
            row['MTL_matched_transfer'] = float(((VB.predict(xt_g, *VB.relaxed(xz, yz, supA)) - yt) ** 2).sum())
            idx = np.arange(n_cal)
            m = RidgeCV(alphas=RIDGE_ALPHAS, cv=[(np.setdiff1d(idx, h), h) for h in folds]).fit(xz, yz)
            row['Ridge_scratch'] = float(((m.predict(xt_g) - yt) ** 2).sum())
            row['Ridge_scratch_plain'] = float(((m.predict(xt) - yt) ** 2).sum())
            # full transfer: A's measurements and A's raw-scale coefficients, scored on the same scale
            yhat = np.tile(yA.mean(0), (nt, 1)) if actA is None else a0 + b * xB[test][:, actA]
            row['Full_transfer'] = float((((yhat - ym) / ysd - yt) ** 2).sum())
            row.update({'k_R4_transfer': kT, 'k_R4_scratch': k_s, 'k_MTL_transfer': len(supA), 'k_MTL_scratch': len(sup_s),
                        'k_all': xB.shape[1], 'choice_R4_scratch': r4s, 'choice_R4_A': r4A})
            rows.append(row)
    (HERE / 'results').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / 'results' / f'{name}.csv', index=False)
    print(name, 'draws', len(rows), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
