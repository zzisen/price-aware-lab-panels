#!/usr/bin/env python3
"""Independent ordinary least-squares check of the vectorized baseline scores."""
import numpy as np
from multifeature_baselines import forward_candidate_scores

for seed in range(100):
    rng=np.random.default_rng(seed)
    xt=rng.normal(size=(55,12)); xv=rng.normal(size=(21,12))
    yt=rng.normal(size=(55,7)); yv=rng.normal(size=(21,7))
    chosen=[1,5] if seed%2 else []
    risk=forward_candidate_scores(xt,yt,xv,yv,chosen)
    for j in range(12):
        if j in chosen:
            assert np.isinf(risk[j]); continue
        cols=chosen+[j]
        tr=np.column_stack([np.ones(len(xt)),xt[:,cols]])
        va=np.column_stack([np.ones(len(xv)),xv[:,cols]])
        beta=np.linalg.lstsq(tr,yt,rcond=None)[0]
        expected=np.sum((va@beta-yv)**2)
        assert abs(risk[j]-expected)<1e-9,(seed,j,risk[j],expected)
print('PASS: multifeature scores match direct OLS on 100 randomized problems.')
