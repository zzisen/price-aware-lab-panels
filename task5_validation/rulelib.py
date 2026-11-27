"""Task 5: shared machinery for resolution rules on any target-family task (development, learning and validation).

Generalises task4_adaptive/adaptive_rules.py to arbitrary target counts and families; with the GDSC example families it
reproduces the Task 3/4 losses exactly (checked in cache_generic.py).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
for p in ('base_v32', 'task3_learning_curve', 'task3b_external', 'task4_adaptive'):
    sys.path.insert(0, str(WORK / p))
from target_resolution import group_folds, action_losses, _choose_from_risks  # noqa: E402
from exact_budget import globally_optimal_actions  # noqa: E402

RES = ['Shared', 'Family', 'Global3', 'Exact']
SEEDS = [20261027, 20261028, 20261029, 20261030, 20261031]
GRID = [25, 50, 100, 200, None]
REPS = 20
MIDDLE_OFFSETS = [7919, 7920, 7921]


def choices(risk, fam):
    out = dict(_choose_from_risks(risk, fam))
    out['Global3'] = globally_optimal_actions(risk, 3)[0]
    return out


def unit_losses(x, y, groups, splits, seed):
    L = np.empty((len(x), x.shape[1], y.shape[1])); c0 = np.empty((len(x), y.shape[1]))
    for held in group_folds(groups, splits, seed):
        train = np.ones(len(x), bool); train[held] = False
        L[held], c0[held] = action_losses(x[train], y[train, :, None], x[held], y[held, :, None])
    return L, c0


def picked(candidate, a):
    """Per-row loss (mean over targets) of the actions a."""
    t = np.arange(candidate.shape[2])
    return candidate[np.arange(len(candidate))[:, None], a[None, :], t[None, :]].mean(axis=1)


def middle_rows(x, y, groups, fam, seed, offset):
    """Middle-held per-row losses (mean over targets) for C0 and each resolution, one middle split."""
    ng = len(np.unique(groups))
    out = {k: np.empty(len(x)) for k in ['C0'] + RES}
    for k, held in enumerate(group_folds(groups, min(5, ng), seed + offset)):
        train = np.ones(len(x), bool); train[held] = False
        L, _ = unit_losses(x[train], y[train], groups[train], min(5, len(np.unique(groups[train]))), seed + 100 + k)
        ch = choices(L.mean(axis=0), fam)
        candidate, c0 = action_losses(x[train], y[train, :, None], x[held], y[held, :, None])
        out['C0'][held] = c0.mean(axis=1)
        for res in RES:
            out[res][held] = picked(candidate, ch[res])
    return out


def cache_subsample(x, y, groups, fam, sub, held, inner_seed):
    """Everything a rule may use (training side) plus the outer losses of every candidate policy."""
    xs, ys, gs = x[sub], y[sub], groups[sub]
    L, c0_inner = unit_losses(xs, ys, gs, 5, inner_seed)
    ch = choices(L.mean(axis=0), fam)
    candidate, c0 = action_losses(xs, ys[:, :, None], x[held], y[held, :, None])
    t = np.arange(ys.shape[1])
    rec = {'groups': gs, 'n_test_cells': len(held) * ys.shape[1],
           'outer': {'C0': float(c0.sum()), **{r: float(picked(candidate, ch[r]).sum() * ys.shape[1]) for r in RES}},
           'inner_rows': {'C0': c0_inner.mean(axis=1), **{r: L[:, ch[r], t].mean(axis=1) for r in RES}},
           'middle': [middle_rows(xs, ys, gs, fam, inner_seed, off) for off in MIDDLE_OFFSETS],
           'n_candidates': xs.shape[1], 'n_targets': ys.shape[1]}
    return rec


def load_task(name):
    task = next(t for t in json.loads((HERE / 'tasks.json').read_text()) if t['task'] == name)
    m = pd.read_csv(HERE / 'matrices' / f'{name}.csv')
    feats = [c for c in m.columns if c not in task['targets'] + ['unit', 'tissue_group']]
    return task, m[feats].to_numpy(float), m[task['targets']].to_numpy(float), np.asarray(task['families'])


def task_subsamples(task, x):
    """Row splits, 5 outer folds x 5 seeds, grid capped at the smallest outer-training size, 20 draws per size."""
    import gdsc_learning_curve as G
    groups = np.arange(len(x))
    folds = {s: group_folds(groups, 5, s) for s in SEEDS}
    smallest = min(len(x) - len(h) for s in SEEDS for h in folds[s])
    grid = [n for n in GRID if n is None or n <= smallest]
    for seed in SEEDS:
        for fold, held in enumerate(folds[seed]):
            train_idx = np.setdiff1d(groups, held)
            for n in grid:
                for r in ([None] if n is None else range(REPS)):
                    if n is None:
                        sub = train_idx
                    else:
                        sub, _ = G.draw(train_idx, n, groups, np.random.default_rng([20261029, task['code'], seed, fold, n, r]), 3)
                    yield 'repeated_random', seed, fold, n, r, sub, held, groups


# ---- extrapolation guard (R4 development) -------------------------------------------------------------------------
GUARD_MARGIN = 1.0   # fall back to the training mean when |z_test| > max_train |z| + GUARD_MARGIN


def guarded_losses(x_train, y_train, x_test, y_test):
    """action_losses, except that a test unit whose standardized measurement lies more than GUARD_MARGIN training SDs
    beyond the training range is predicted by the training mean (C0) for that measurement."""
    candidate, c0 = action_losses(x_train, y_train, x_test, y_test)
    xm = x_train.mean(axis=0); xs = x_train.std(axis=0, ddof=1); xs = np.where(xs == 0, 1.0, xs)
    zmax = np.abs((x_train - xm) / xs).max(axis=0)
    out = np.abs((x_test - xm) / xs) > zmax + GUARD_MARGIN
    return np.where(out[:, :, None], c0[:, None, :], candidate), c0, float(out.mean())


def unit_losses_g(x, y, groups, splits, seed):
    L = np.empty((len(x), x.shape[1], y.shape[1])); c0 = np.empty((len(x), y.shape[1]))
    for held in group_folds(groups, splits, seed):
        train = np.ones(len(x), bool); train[held] = False
        L[held], c0[held], _ = guarded_losses(x[train], y[train, :, None], x[held], y[held, :, None])
    return L, c0


def middle_rows_g(x, y, groups, fam, seed, offset):
    ng = len(np.unique(groups))
    out = {k: np.empty(len(x)) for k in ['C0'] + RES}
    for k, held in enumerate(group_folds(groups, min(5, ng), seed + offset)):
        train = np.ones(len(x), bool); train[held] = False
        L, _ = unit_losses_g(x[train], y[train], groups[train], min(5, len(np.unique(groups[train]))), seed + 100 + k)
        ch = choices(L.mean(axis=0), fam)
        candidate, c0, _ = guarded_losses(x[train], y[train, :, None], x[held], y[held, :, None])
        out['C0'][held] = c0.mean(axis=1)
        for res in RES:
            out[res][held] = picked(candidate, ch[res])
    return out


def cache_subsample_g(x, y, groups, fam, sub, held, inner_seed):
    """Guarded counterpart of cache_subsample: guard in selection, middle layer and outer scoring."""
    xs, ys, gs = x[sub], y[sub], groups[sub]
    L, _ = unit_losses_g(xs, ys, gs, 5, inner_seed)
    ch = choices(L.mean(axis=0), fam)
    candidate, c0, frac = guarded_losses(xs, ys[:, :, None], x[held], y[held, :, None])
    return {'outer_g': {'C0': float(c0.sum()), **{r: float(picked(candidate, ch[r]).sum() * ys.shape[1]) for r in RES}},
            'middle_g': [middle_rows_g(xs, ys, gs, fam, inner_seed, off) for off in MIDDLE_OFFSETS], 'guard_fraction': frac}
