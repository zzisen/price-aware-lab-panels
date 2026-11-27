#!/usr/bin/env python3
"""Task 47 (additional, post hoc): is the 10% learnability threshold tied to the predictor? Alternative checks at n/k >= 10:
fixed tau in {0, 0.02, 0.05, 0.10}; and a predictor-scaled tau = 0.10 x (median cross-validated R2 of C at n/k >= 10 for that predictor
in its development data / the same median for ridge). Development data: task42 dev_results_T / dev_results_S (14 sources of test 3) and, for ridge,
task40 results (same 14 sources). Evaluated on locked test 4 (trees; QR + lasso) and locked test 3 (ridge). Regret as in the frozen summaries;
source bootstrap 2000 draws, seed 20264701."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; W = H.parent
TY = {'G': 'GTEx', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
def load(fo):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(W / fo / '*_o0.csv'))], ignore_index=True)
    d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); d['cvr2'] = 1 - d.cv5_C; return d
def med_learn(d):
    x = d[(d.nk >= 10) & (d.type == 'TCGA_RNAseq')]
    return float(x.cvr2.median())
ridge_dev = load('task40_locked_v4/results'); tree_dev = load('task42_locked_v5/dev_results_T'); s_dev = load('task42_locked_v5/dev_results_S')
mr, mt, ms = med_learn(ridge_dev), med_learn(tree_dev), med_learn(s_dev)
scales = {'ridge': 0.10, 'trees': 0.10 * mt / mr, 'qr_lasso': 0.10 * ms / mr}
def regrets(d, taus, balanced):
    best = d[['A', 'B', 'C']].max(axis=1); hi = d.nk >= 10; low = np.where(d.cv5_A < d.cv5_C, d.A, d.C)
    R = {f'tau_{t:g}': np.where(hi & (d.cvr2 >= t), d.C, low) for t in taus}
    R.update({'CV': low, 'H': np.where(hi, d.C, low), 'always_A': d.A, 'always_C': d.C})
    r = pd.DataFrame({k: best - v for k, v in R.items()}); r['source'] = d.source; r['type'] = d.type
    S = r.groupby(['type', 'source']).mean()
    tb = lambda X: X.groupby(level=0).mean().mean() if balanced else X.mean()
    pt = {c: float(tb(S[c])) for c in R}
    rng = np.random.default_rng(20264701); types = sorted(S.index.get_level_values(0).unique()); bs = []
    for _ in range(2000):
        parts = [S.loc[t].iloc[rng.integers(0, len(S.loc[t]), len(S.loc[t]))] for t in types]
        bs.append(pd.Series({c: (np.mean([p_[c].mean() for p_ in parts]) if balanced else pd.concat(parts)[c].mean()) for c in R}))
    B = pd.DataFrame(bs); ci = lambda x: [float(x.quantile(.025)), float(x.quantile(.975))]
    return {'regret': pt, 'minus_always_C': {c: [pt[c] - pt['always_C']] + ci(B[c] - B['always_C']) for c in R if c.startswith('tau')},
            'minus_CV': {c: [pt[c] - pt['CV']] + ci(B[c] - B['CV']) for c in R if c.startswith('tau')}}
base = [0, 0.02, 0.05, 0.10]
out = {'median_cv_R2_at_nk10_TCGA_dev': {'ridge': mr, 'trees': mt, 'qr_lasso': ms}, 'scaled_tau': scales}
out['locked4_trees'] = regrets(load('task42_locked_v5/results_T'), sorted(set(base + [round(scales['trees'], 4)])), False)
out['locked4_qr_lasso'] = regrets(load('task42_locked_v5/results_S'), sorted(set(base + [round(scales['qr_lasso'], 4)])), False)
out['locked3_ridge'] = regrets(ridge_dev, base, True)
json.dump(out, open(H / 'scale_check.json', 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or 'regret' not in v else {kk: {a: (round(b, 4) if isinstance(b, float) else [round(z, 4) for z in b]) for a, b in vv.items()} for kk, vv in v.items()}) for k, v in out.items()}, indent=1))
