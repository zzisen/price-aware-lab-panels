#!/usr/bin/env python3
"""Globally optimize a finite target-additive feature-cap selection score.

For each partition of the T targets into at most K blocks, select the best
feature independently for every block. Any target-to-feature assignment using
at most K distinct features induces one such partition, so the smallest score
among all partitions is the global optimum for the supplied score matrix.
This is a deterministic optimization of the finite, additive *training score*,
not a guarantee of minimum test risk. Supports modest numbers of targets;
7 targets and K=3 yield exactly 365 partitions.
"""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=None)
def target_partitions(n_targets: int, max_blocks: int):
    if n_targets<1 or max_blocks<1: raise ValueError('Positive targets and budget required')
    result=[]
    def visit(i,blocks):
        if i==n_targets:
            result.append(tuple(blocks));return
        bit=1<<i
        for j in range(len(blocks)):
            blocks[j] |= bit;visit(i+1,blocks);blocks[j] ^= bit
        if len(blocks)<max_blocks:
            blocks.append(bit);visit(i+1,blocks);blocks.pop()
    visit(1,[1])
    return tuple(result)

def globally_optimal_actions(risks, budget=3):
    risks=np.asarray(risks,dtype=float)
    if risks.ndim!=2 or not np.isfinite(risks).all(): raise ValueError('Expected finite action-by-target scores')
    n_features,n_targets=risks.shape
    if budget<1 or budget>n_features: raise ValueError('Invalid feature budget')
    partitions=target_partitions(n_targets,budget)
    subset_actions={};subset_cost={}
    for mask in range(1,1<<n_targets):
        members=[i for i in range(n_targets) if (mask>>i)&1]
        costs=risks[:,members].sum(axis=1)
        choice=int(np.argmin(costs))
        subset_actions[mask]=choice;subset_cost[mask]=float(costs[choice])
    # Canonical partition order and feature-index order resolve exact ties.
    costs=np.array([math.fsum(subset_cost[m] for m in partition) for partition in partitions])
    best_idx=int(np.argmin(costs));best=partitions[best_idx]
    actions=np.empty(n_targets,dtype=int)
    for mask in best:
        for target in range(n_targets):
            if (mask>>target)&1: actions[target]=subset_actions[mask]
    objective=float(risks[actions,np.arange(n_targets)].mean())
    lower=float(costs[best_idx]/n_targets)
    if len(np.unique(actions))>budget or abs(objective-lower)>1e-10:
        raise RuntimeError('Budget or exhaustive-objective check failed')
    return actions,{'n_partitions':len(partitions),'score':objective,'exhaustive_lower_bound':lower,
                    'partition_masks':list(best),'distinct_features':len(np.unique(actions))}
