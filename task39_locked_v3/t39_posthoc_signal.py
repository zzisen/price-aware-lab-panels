#!/usr/bin/env python3
"""Post hoc (after the Task 39 locked results were seen; not a test). Candidate refinement for a future frozen plan:
a signal check on the count shortcut. At n/k >= 10 redesign unless 5-fold CV favours the reusable panel by more than z fold-wise standard
errors (z = 1, 2); below 10, CV as in H. Evaluated descriptively in the 46 development sources and in the Task 39 sources.
Output t39_posthoc_signal.json."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
TY = {'G': 'GTEx', 'X': 'CRISPR', 'P': 'proteomics', 'R': 'RPPA'}
out = {}
for name, folder in [('development46', W / 'task36_rule_v3' / 'stage1_results'), ('locked2', HERE / 'results')]:
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(folder / '*_o0.csv'))], ignore_index=True)
    d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); d['best'] = d[['A', 'B', 'C']].max(axis=1)
    cvA = d.cv5_A < d.cv5_C; dd = d.cv5_C - d.cv5_A                                   # > 0: CV favours A
    r = {'H': d.best - np.where(d.nk >= 10, d.C, np.where(cvA, d.A, d.C)), 'CV': d.best - np.where(cvA, d.A, d.C),
         'always_C': d.best - d.C, 'always_A': d.best - d.A}
    for z in (1, 2):
        sig = dd > z * d.cv5_se_AmC
        r[f'H_sig{z}'] = d.best - np.where(d.nk >= 10, np.where(sig, d.A, d.C), np.where(cvA, d.A, d.C))
    res = {'all': {k: float(v.mean()) for k, v in r.items()}}
    for t, idx in d.groupby('type').groups.items():
        res[t] = {k: float(v[idx].mean()) for k, v in r.items()}
    hi = d.nk >= 10
    res['share_overruled_at_nk10'] = {f'z{z}': float((dd[hi] > z * d.cv5_se_AmC[hi]).mean()) for z in (1, 2)}
    out[name] = res
json.dump(out, open(HERE / 't39_posthoc_signal.json', 'w'), indent=1)
for n, res in out.items():
    print(n)
    for t, v in res.items():
        print(f'  {t:12s}', {k: round(x, 4) for k, x in v.items()})
