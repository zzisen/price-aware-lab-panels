#!/usr/bin/env python3
"""Task 41 sensitivity (additional): the empirical crossover by task signal, with signal measured from TRAINING data only, to remove the coupling
between a test-set signal (C's test EV) and the test-set gain (C - A). Signal = 1 - cv5_C (five-fold CV explained variance of C within the training
subsample) at the task's largest n with k = 20, averaged over repeats. Same settings, bins and crossover interpolation as empirical.py."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; W = H.parent
TY = {'G': 'GTEx', 'R': 'RPPA', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
fs = [f for fo in ['task36_rule_v3/stage1_results', 'task39_locked_v3/results', 'task40_locked_v4/results'] for f in glob.glob(str(W / fo / '*_o0.csv'))]
d = pd.concat([pd.read_csv(f, usecols=['source', 'task', 'p', 'n', 'rep', 'k', 'A', 'C', 'cv5_C']) for f in fs], ignore_index=True)
d['type'] = d.source.str[0].map(TY); d['cvsig'] = 1 - d.cv5_C
st = d.groupby(['type', 'source', 'task', 'p', 'n', 'k'])[['A', 'C', 'cvsig']].mean().reset_index()
st['gap'] = st.C - st.A; st['nk'] = st.n / st.k
big = st[st.k == 20].sort_values('n').groupby(['source', 'task', 'p']).tail(1).set_index(['source', 'task', 'p']).cvsig.rename('signal')
st = st.join(big, on=['source', 'task', 'p'])
bins = [0, 2.5, 3.75, 5, 6.25, 7.5, 10, 12.5, 15, 20, 25, 40]; st['nkbin'] = pd.cut(st.nk, bins, right=False)
def crossover(g):
    m = g.groupby('nkbin', observed=True).gap.mean(); mids = np.array([iv.mid for iv in m.index]); v = m.values
    for i in range(1, len(v)):
        if v[i - 1] < 0 <= v[i]:
            return float(np.exp(np.interp(0, [v[i - 1], v[i]], np.log([mids[i - 1], mids[i]]))))
    return None if (v < 0).all() else (0.0 if (v >= 0).all() else None)
sb = pd.cut(st.signal, [-10, 0.05, 0.15, 0.3, 0.45, 0.6, 1.0])
out = {'by_cv_signal': {str(b): [crossover(g), int(g[['source', 'task']].drop_duplicates().shape[0])] for b, g in st.groupby(sb, observed=True)}}
nr = st[(st.type != 'RPPA') & (st.signal > 0.05)]
out['nonRPPA_by_p_cv_learnable'] = {int(p): crossover(g) for p, g in nr.groupby('p')}
from scipy.stats import spearmanr
tk = st[st.nk >= 10].groupby(['source', 'task', 'p']).agg(gap=('gap', 'mean'), signal=('signal', 'first'))
out['spearman_signal_gap_nk10'] = float(spearmanr(tk.signal, tk.gap).statistic)
json.dump(out, open(H / 'empirical_cvsignal.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
