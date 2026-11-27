#!/usr/bin/env python3
"""Task 37 (STAGE2_DEV_PLAN.md): clinical learning curves within outcome. Shared helpers.
Panels within the routine budget: A target-agnostic (cost-aware FSCA over orderable items, on all training laboratory data, no outcome),
B history-informed (greedy for the other outcomes jointly, full training), C redesigned for the target on the training subsample,
D routine bundle, base (age and sex only), ALL (every analyte; signal reference, ignores the budget).
Selection: greedy forward over orderable items by 5-fold CV log-likelihood (ridge Cox, lambda 1, for NHANES; ridge logistic C = 1 for CLHLS)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))


def greedy(score, items, cost, budget, start=()):
    """Forward selection over items; stops when nothing affordable improves the score. score: tuple(items) -> float (higher better)."""
    cur = tuple(start); best = score(cur)
    while True:
        cand = []
        for it in items:
            if it in cur:
                continue
            new = tuple(sorted(cur + (it,)))
            if cost(new) > budget + 1e-9:
                continue
            cand.append((score(new), new))
        if not cand:
            break
        s, new = max(cand)
        if s <= best + 1e-9:
            break
        cur, best = new, s
    return list(cur), best


def fsca_items(Xa, cols, items_map, items, cost, budget):
    """Cost-aware forward selection component analysis over orderable items: explained variance of all analytes (standardised) by OLS
    on the analytes measured. Xa: (n x p) standardised analyte matrix with column names cols."""
    Xa = np.asarray(Xa, float); idx = {c: i for i, c in enumerate(cols)}
    tot = float((Xa ** 2).sum())

    def ev(its):
        f = sorted({idx[a] for it in its for a in items_map[it] if a in idx})
        if not f:
            return 0.0
        S = Xa[:, f]; beta, *_ = np.linalg.lstsq(S, Xa, rcond=None)
        return 1 - float(((Xa - S @ beta) ** 2).sum()) / tot
    return greedy(ev, items, cost, budget)[0]
