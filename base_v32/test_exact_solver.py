#!/usr/bin/env python3
"""Compare exhaustive target-partition optimization with independent feature-set enumeration."""
from itertools import combinations
import numpy as np
from exact_budget import globally_optimal_actions,target_partitions

def main():
    assert len(target_partitions(7,3))==365
    rng=np.random.default_rng(20261028);trials=0
    for nfeatures in [3,4,5,6,7]:
        for _ in range(100):
            risks=rng.normal(size=(nfeatures,7))
            actions,certificate=globally_optimal_actions(risks,3)
            independently=min(np.min(risks[list(s)],axis=0).mean() for s in combinations(range(nfeatures),3))
            assert abs(certificate['score']-independently)<1e-12 and len(np.unique(actions))<=3
            trials+=1
    for ntargets in [3,7]:
        actions,certificate=globally_optimal_actions(np.zeros((5,ntargets)),3)
        assert len(np.unique(actions))<=3 and certificate['score']==0
    print(f'PASS: 365 target partitions, {trials} independent feature-set enumerations, ties.')
if __name__=='__main__':main()
