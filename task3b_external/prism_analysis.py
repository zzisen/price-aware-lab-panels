#!/usr/bin/env python3
"""Task 3b (plan_external_prism.md, frozen 2026-10-28 21:17 NZDT): E1 replication and E2 transport in PRISM.

E1 reuses the Task 3 learning-curve evaluation (gdsc_learning_curve.evaluate) on the PRISM matrix.
E2 fixes each policy's measurement choices on all 499 GDSC rows and refits only intercept and slope in PRISM.
Run with .env312 after build_prism_matrix.py. Writes prism_runs.csv and transport_runs.csv here.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / 'base_v32'))
sys.path.insert(0, str(WORK / 'task3_learning_curve'))
from target_resolution import group_folds, action_losses, select_actions  # noqa: E402
from evaluate_gdsc import TARGETS, FAMILIES, SEEDS  # noqa: E402
from exact_budget import globally_optimal_actions  # noqa: E402
from gdsc_learning_curve import evaluate, draw, REPS  # noqa: E402

# dataset -> (matrix file, target columns in GDSC example order, grid, RNG evaluation codes, row subset)
GDSC2_TARGETS = ['Methotrexate', 'Refametinib', 'PD0325901', 'Trametinib', 'Selumetinib', 'Nilotinib', 'Axitinib']
DATASETS = {
    'prism': ('prism_matrix.csv', TARGETS, [25, 50, 100, 200, None], {'repeated_random': 10, 'held_out_tissue': 11}, None),
    'gdsc2': ('gdsc2_matrix.csv', GDSC2_TARGETS, [25, 50, 100, 200, 400, None], {'repeated_random': 20, 'held_out_tissue': 21}, None),
    'gdsc2_new': ('gdsc2_matrix.csv', GDSC2_TARGETS, [25, 50, 100, 200, 400, None], {'repeated_random': 22, 'held_out_tissue': 23}, 'new'),
}


def load(dataset):
    path, targets, _, _, subset = DATASETS[dataset]
    m = pd.read_csv(HERE / path)
    if subset == 'new':
        m = m[m.in_gdsc499 == 0].reset_index(drop=True)
    feats = [c for c in m.columns if c not in targets + ['depmap_id', 'tissue_group', 'in_gdsc499']]
    return m, m[feats].to_numpy(float), m[targets].to_numpy(float), feats


def gdsc_choices(prism_features):
    """Measurement choices for each policy on all 499 GDSC rows (inner CV, 5 folds, seed 20261027)."""
    frame = pd.read_csv(WORK / 'base_v32' / 'inputs' / 'GDSC' / 'exampleGDSC_2.3-1_readable.csv')
    x = frame.iloc[:, 20:].to_numpy(float); y = frame.iloc[:, :7].to_numpy(float)
    names = frame.columns[20:].tolist()
    choice, risk = select_actions(x, y, FAMILIES, np.arange(len(x)), inner_splits=5, seed=SEEDS[0])
    g3, _ = globally_optimal_actions(risk, 3)
    chosen = {'Shared': choice['Shared'], 'Family': choice['Family'], 'Global3': g3, 'Exact': choice['Exact']}
    out, missing = {}, {}
    for p, a in chosen.items():
        idx = [prism_features.index(names[i]) if names[i] in prism_features else -1 for i in a]
        out[p] = np.array(idx); missing[p] = [names[i] for i in a if names[i] not in prism_features]
    return out, missing, {p: [names[i] for i in a] for p, a in chosen.items()}


def transport_scores(x, y, sub, held, chosen):
    candidate, c0 = action_losses(x[sub], y[sub, :, None], x[held], y[held, :, None])
    row = {'loss_sum_C0': float(c0.sum())}
    for p, a in chosen.items():
        loss = np.where(a[None, :] >= 0,
                        candidate[np.arange(len(held))[:, None], np.maximum(a, 0)[None, :], np.arange(7)[None, :]], c0)
        row[f'loss_sum_{p}'] = float(loss.sum())
        row[f'loss_sum_{p}_new_units'] = None
    return row, candidate, c0


def main(dataset):
    m, x, y, feats = load(dataset)
    _, _, GRID, EVAL_CODE, subset = DATASETS[dataset]
    designs_wanted = ('repeated_random',) if subset == 'new' else ('repeated_random', 'held_out_tissue')
    tissue = pd.factorize(m.tissue_group)[0]
    new_unit = ~m.in_gdsc499.to_numpy(bool)
    chosen, missing, names = gdsc_choices(feats)
    pd.Series({p: '; '.join(v) for p, v in names.items()}).to_csv(HERE / f'{dataset}_transport_gdsc_choices.csv', header=['features'])
    print('GDSC-chosen features unavailable in PRISM:', missing, flush=True)
    rows, trows = [], []
    smallest = {}
    designs = [('repeated_random', SEEDS, np.arange(len(x)), 5, 3), ('held_out_tissue', [SEEDS[0]], tissue, len(np.unique(tissue)), 5)]
    designs = [d for d in designs if d[0] in designs_wanted]
    for ev, seeds, groups, nfold, min_groups in designs:
        folds = {s: group_folds(groups, nfold, s) for s in seeds}
        smallest[ev] = min(len(x) - len(h) for s in seeds for h in folds[s])
        grid = [n for n in GRID if n is None or n <= smallest[ev]]
        for seed in seeds:
            for fold, held in enumerate(folds[seed]):
                train_idx = np.setdiff1d(np.arange(len(x)), held)
                for n in grid:
                    for r in ([None] if n is None else range(REPS)):
                        if n is None:
                            sub, redraws = train_idx, 0
                        else:
                            rng = np.random.default_rng([20261028, EVAL_CODE[ev], seed, fold, n, r])
                            sub, redraws = draw(train_idx, n, groups, rng, min_groups)
                        assert not set(groups[sub]) & set(groups[held])
                        base = {'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub),
                                'n_label': 'all' if n is None else n, 'rep': -1 if r is None else r,
                                'n_test_rows': len(held), 'n_test_cells': len(held) * 7, 'redraws': redraws}
                        for method, vals in evaluate(x, y, groups, sub, held, seed + fold + 1).items():
                            rows.append({**base, 'selector': method, **vals})
                        if ev == 'repeated_random':
                            t, candidate, c0 = transport_scores(x, y, sub, held, chosen)
                            nu = new_unit[held]
                            for p, a in chosen.items():
                                loss = np.where(a[None, :] >= 0, candidate[np.arange(len(held))[:, None],
                                                np.maximum(a, 0)[None, :], np.arange(7)[None, :]], c0)
                                t[f'loss_sum_{p}_new_units'] = float(loss[nu].sum())
                            t['loss_sum_C0_new_units'] = float(c0[nu].sum())
                            trows.append({**base, 'n_test_cells_new_units': int(nu.sum()) * 7, **t})
                print(ev, seed, fold + 1, '/', nfold, flush=True)
    pd.DataFrame(rows).to_csv(HERE / f'{dataset}_runs.csv', index=False)
    pd.DataFrame(trows).to_csv(HERE / f'{dataset}_transport_runs.csv', index=False)
    print('smallest outer-training sizes', smallest, '| redraws', int(pd.DataFrame(rows).redraws.sum()))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'prism')
