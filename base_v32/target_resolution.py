"""Reusable fixed-resolution measurement-policy evaluation.

An action is one observed scalar baseline feature; a target can have one or
more response coordinates. Action selection and scaling occur within the
training data. Cross-validation groups must be supplied by the caller. This
implementation compares fixed Shared, Family, and Exact policies; it does not
implement P2's adaptive-resolution procedure, costs, or budget constraints.
"""
from __future__ import annotations
import numpy as np


def _validate(x, y, families, groups):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if y.ndim == 2: y = y[:, :, None]
    groups = np.asarray(groups)
    if x.ndim != 2 or y.ndim != 3 or len(x) != len(y) or len(x) != len(groups):
        raise ValueError('Expected X[n,actions], Y[n,targets,outputs], groups[n]')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('X and Y must be complete and finite')
    if len(families) != y.shape[1]:
        raise ValueError('Provide one family label for each target')
    if len(np.unique(groups)) < 3:
        raise ValueError('At least three distinct groups are needed for nested evaluation')
    return x, y, np.asarray(families), groups


def group_folds(groups, n_splits, seed):
    """Deterministically balance whole groups across folds, never split a group."""
    groups = np.asarray(groups)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError('Fold count must be between 2 and the number of groups')
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(unique))
    order = sorted(permutation, key=lambda i: -counts[i])
    bins = [[] for _ in range(n_splits)]; sizes = np.zeros(n_splits, dtype=int)
    for idx in order:
        k = int(np.argmin(sizes)); bins[k].append(idx); sizes[k] += counts[idx]
    result = [np.flatnonzero(np.isin(inverse, b)) for b in bins]
    assert sum(len(a) for a in result) == len(groups)
    assert len(np.unique(np.concatenate(result))) == len(groups)
    assert all(len(a) for a in result)
    return result


def action_losses(x_train, y_train, x_test, y_test, alpha=0.0):
    """Mean squared error by test unit, candidate action, and target.

    Each response coordinate and candidate feature is standardized using this
    training set only. A zero training SD is replaced by 1. The intercept is
    unpenalized; alpha is the slope ridge penalty on the sum-of-squares scale.
    """
    xm = x_train.mean(axis=0); xs = x_train.std(axis=0, ddof=1)
    ym = y_train.mean(axis=0); ys = y_train.std(axis=0, ddof=1)
    xs = np.where(xs == 0, 1.0, xs); ys = np.where(ys == 0, 1.0, ys)
    xz = (x_train - xm) / xs; yz = (y_train - ym) / ys
    numerator = np.einsum('ia,ito->ato', xz, yz, optimize=True)
    denominator = (xz * xz).sum(axis=0)[:, None, None] + alpha
    slopes = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    pred = ((x_test - xm) / xs)[:, :, None, None] * slopes[None, :, :, :]
    yz_test = (y_test - ym) / ys
    per_action = ((pred - yz_test[:, None, :, :])**2).mean(axis=3)
    c0 = (yz_test**2).mean(axis=2)
    return per_action, c0


def select_actions(x, y, families, groups, *, inner_splits=5, seed=1, alpha=0.0):
    """Choose one action per target for each fixed policy using inner CV."""
    x, y, families, groups = _validate(x, y, families, groups)
    folds = group_folds(groups, inner_splits, seed)
    accumulated = np.zeros((x.shape[1], y.shape[1])); n_scored = 0
    for held in folds:
        train = np.ones(len(x), dtype=bool); train[held] = False
        if len(x[train]) < 2: raise ValueError('Inner training set must contain at least two rows')
        assert not set(groups[train]).intersection(groups[held])
        loss, _ = action_losses(x[train], y[train], x[held], y[held], alpha)
        accumulated += loss.sum(axis=0); n_scored += len(held)
    risks = accumulated / n_scored
    return _choose_from_risks(risks, families), risks


def _choose_from_risks(risks, families):
    shared = np.repeat(int(np.argmin(risks.mean(axis=1))), risks.shape[1])
    exact = np.argmin(risks, axis=0)
    family = np.empty(risks.shape[1], dtype=int)
    for label in np.unique(families):
        members = np.flatnonzero(families == label)
        family[members] = int(np.argmin(risks[:, members].mean(axis=1)))
    return {'Shared': shared, 'Family': family, 'Exact': exact}


def select_by_correlation(x, y, families, groups):
    """Conventional in-training univariate squared-correlation feature filter.

    This comparator uses the same one-feature-per-block action budget as each
    partition. It selects on the full outer-training data and has no inner CV.
    """
    x, y, families, groups = _validate(x, y, families, groups)
    xm = x.mean(axis=0); xs = x.std(axis=0, ddof=1)
    ym = y.mean(axis=0); ys = y.std(axis=0, ddof=1)
    xs = np.where(xs == 0, 1.0, xs); ys = np.where(ys == 0, 1.0, ys)
    xz = (x - xm) / xs; yz = (y - ym) / ys
    correlation = np.einsum('ia,ito->ato', xz, yz, optimize=True) / (len(x)-1)
    risks = 1.0 - np.mean(correlation**2, axis=2)
    return _choose_from_risks(risks, families), risks


def nested_evaluate(x, y, families, groups, *, outer_splits=5, inner_splits=5, seed=20261027, alpha=0.0, selection='inner_cv'):
    """Return all held-out target losses and selected actions by outer fold."""
    x, y, families, groups = _validate(x, y, families, groups)
    outer = group_folds(groups, outer_splits, seed)
    policies = ['C0', 'Shared', 'Family', 'Exact']
    losses = np.full((len(x), y.shape[1], len(policies)), np.nan)
    fold_id = np.full(len(x), -1, dtype=int); selected = []
    for k, held in enumerate(outer):
        mask = np.ones(len(x), dtype=bool); mask[held] = False
        if len(np.unique(groups[mask])) < inner_splits:
            raise ValueError('Too few training groups for inner splits')
        assert not set(groups[mask]).intersection(groups[held])
        if selection == 'inner_cv':
            choices, _ = select_actions(x[mask], y[mask], families, groups[mask],
                                        inner_splits=inner_splits, seed=seed + k + 1, alpha=alpha)
        elif selection == 'in_sample_r2':
            choices, _ = select_by_correlation(x[mask], y[mask], families, groups[mask])
        else:
            raise ValueError('Unknown selection method')
        candidate, c0 = action_losses(x[mask], y[mask], x[held], y[held], alpha)
        losses[held, :, 0] = c0
        for j, name in enumerate(policies[1:], start=1):
            losses[held, :, j] = candidate[np.arange(len(held))[:, None], choices[name][None, :], np.arange(y.shape[1])[None, :]]
        selected.append({'fold': k, 'choices': choices, 'held_out_groups': np.unique(groups[held]).tolist()})
        fold_id[held] = k
    if not np.isfinite(losses).all() or (fold_id < 0).any():
        raise RuntimeError('Incomplete held-out results')
    return {'losses': losses, 'fold_id': fold_id, 'selections': selected, 'policies': policies}
