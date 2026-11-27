#!/usr/bin/env python3
"""Task 3 (post hoc): GDSC learning curves, as specified in plan_learning_curve.md (frozen 2026-10-28 20:55 NZDT).

Outer folds and test rows are those of V3.2 (run_final_budget.py); only the outer-training rows are
subsampled. Everything inside a subsample reuses the V3.2 functions unchanged.
Run with P2_V3.3_work/.env312/python.exe. Writes gdsc_runs.csv and gdsc_check_full_n.csv here.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'base_v32'
sys.path.insert(0, str(BASE))
from target_resolution import group_folds, action_losses, select_actions, select_by_correlation  # noqa: E402
from evaluate_gdsc import TARGETS, FAMILIES, SEEDS  # noqa: E402
from exact_budget import globally_optimal_actions  # noqa: E402

SIZES = [25, 50, 100, 200, None]          # None = all outer-training rows
REPS = 20
POLICIES = ['C0', 'Shared', 'Family', 'Global3', 'Exact']
EVAL_CODE = {'repeated_random': 0, 'held_out_tissue': 1}


def load():
    frame = pd.read_csv(BASE / 'inputs' / 'GDSC' / 'exampleGDSC_2.3-1_readable.csv')
    assert frame.shape == (499, 857) and frame.columns[:7].tolist() == TARGETS
    x = frame.iloc[:, 20:].to_numpy(float); y = frame.iloc[:, :7].to_numpy(float)
    tissue = np.argmax(frame.iloc[:, 7:20].to_numpy(float), axis=1)
    return x, y, tissue


def draw(train_idx, n, groups, rng, min_groups):
    redraws = 0
    while True:
        sub = np.sort(rng.choice(train_idx, size=n, replace=False))
        if len(np.unique(groups[sub])) >= min_groups:
            return sub, redraws
        redraws += 1


def evaluate(x, y, groups, sub, held, inner_seed):
    """One subsample: select on sub (inner CV and correlation filter), score on held."""
    cv, rv = select_actions(x[sub], y[sub], FAMILIES, groups[sub], inner_splits=5, seed=inner_seed)
    cf, rf = select_by_correlation(x[sub], y[sub], FAMILIES, groups[sub])
    candidate, c0 = action_losses(x[sub], y[sub, :, None], x[held], y[held, :, None])
    out = {}
    for method, choice, risk in (('inner_cv', cv, rv), ('correlation_filter', cf, rf)):
        g3, _ = globally_optimal_actions(risk, 3)
        sel = {'Shared': choice['Shared'], 'Family': choice['Family'], 'Global3': g3, 'Exact': choice['Exact']}
        row = {'loss_sum_C0': float(c0.sum())}
        for p, a in sel.items():
            loss = candidate[np.arange(len(held))[:, None], a[None, :], np.arange(7)[None, :]]
            row[f'loss_sum_{p}'] = float(loss.sum())
        row['global3_equals_family'] = bool(np.array_equal(g3, choice['Family']))
        row['exact_distinct_features'] = int(len(np.unique(choice['Exact'])))
        out[method] = row
    return out


def main():
    x, y, tissue = load()
    rows = []
    designs = [('repeated_random', SEEDS, np.arange(499), 5, 3), ('held_out_tissue', [SEEDS[0]], tissue, 13, 5)]
    for ev, seeds, groups, nfold, min_groups in designs:
        for seed in seeds:
            for fold, held in enumerate(group_folds(groups, nfold, seed)):
                train_idx = np.setdiff1d(np.arange(len(x)), held)
                for n in SIZES:
                    reps = [None] if n is None else range(REPS)
                    for r in reps:
                        if n is None:
                            sub, redraws = train_idx, 0
                        else:
                            rng = np.random.default_rng([20261028, EVAL_CODE[ev], seed, fold, n, r])
                            sub, redraws = draw(train_idx, n, groups, rng, min_groups)
                        assert not set(groups[sub]) & set(groups[held])
                        res = evaluate(x, y, groups, sub, held, seed + fold + 1)
                        for method, vals in res.items():
                            rows.append({'evaluation': ev, 'outer_seed': seed, 'fold': fold,
                                         'n_train': len(sub), 'n_label': 'all' if n is None else n,
                                         'rep': -1 if r is None else r, 'selector': method,
                                         'n_test_rows': len(held), 'n_test_cells': len(held) * 7,
                                         'redraws': redraws, **vals})
                print(ev, seed, fold + 1, '/', nfold, flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(HERE / 'gdsc_runs.csv', index=False)

    # implementation check: n = all must reproduce V3.2 final_budget3 to 1e-10
    ref = pd.read_csv(BASE / 'results' / 'final_budget3' / 'summary.csv').set_index(['evaluation', 'selector'])
    full = runs[runs.n_label == 'all']
    check = []
    for (ev, method), g in full.groupby(['evaluation', 'selector']):
        for p, refcol in zip(POLICIES, ['C0', 'Shared', 'Family', 'Global3', 'Exact_unconstrained']):
            value = g[f'loss_sum_{p}'].sum() / g.n_test_cells.sum()
            check.append({'evaluation': ev, 'selector': method, 'policy': p, 'rerun': value,
                          'v32': ref.loc[(ev, method), refcol], 'abs_diff': abs(value - ref.loc[(ev, method), refcol])})
    check = pd.DataFrame(check)
    check.to_csv(HERE / 'gdsc_check_full_n.csv', index=False)
    print(check.to_string(index=False))
    assert check.abs_diff.max() < 1e-10, 'full-n rerun does not reproduce V3.2'
    print('full-n reproduction OK; total redraws', int(runs.redraws.sum()))


if __name__ == '__main__':
    main()
