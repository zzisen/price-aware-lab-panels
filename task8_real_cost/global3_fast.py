#!/usr/bin/env python3
"""Task 8: memory-safe replacement for exact_budget.globally_optimal_actions, for tasks with many targets (SAAD_LV, 36).

The frozen code enumerates every partition of the targets into <= K blocks (about 3**T / 6 partitions; T = 36 runs out
of memory). The same optimum is reached by enumerating every set of <= K actions and giving each target its best action
in the set: any assignment with <= K distinct actions uses one such set, and for a fixed set the per-target minimum is
optimal. Same objective, same constraint, so the same minimum score; check_equal.py confirms identical actions on the
tasks the frozen code can run."""
import itertools
import numpy as np


def globally_optimal_actions(risks, budget=3):
    risks = np.asarray(risks, dtype=float)
    if risks.ndim != 2 or not np.isfinite(risks).all():
        raise ValueError('Expected finite action-by-target scores')
    n_features, n_targets = risks.shape
    if budget < 1 or budget > n_features:
        raise ValueError('Invalid feature budget')
    best, best_cost = None, np.inf
    for k in range(1, budget + 1):
        for s in itertools.combinations(range(n_features), k):
            sub = risks[list(s)]
            cost = sub.min(axis=0).sum()
            if cost < best_cost - 1e-12:
                best_cost, best = cost, np.array(s)[sub.argmin(axis=0)]
    return best, {'score': float(risks[best, np.arange(n_targets)].mean()), 'distinct_features': len(np.unique(best))}
