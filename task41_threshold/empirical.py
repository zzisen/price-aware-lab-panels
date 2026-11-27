#!/usr/bin/env python3
"""Task 41 (post hoc, descriptive): where does redesign (C) overtake reuse (A)? Data: all denser-grid results already analysed
(Task 36 stage 1: 46 sources; Task 39: 17; Task 40: 14). Setting = source x task x p x n x k, averaged over repeats.
Signal of a task = test explained variance of C at the largest n and k = 20 (a property of the target, not used by the rule)."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
W = Path(__file__).resolve().parent.parent
TY = {'G': 'GTEx', 'R': 'RPPA', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
fs = [f for fo in ['task36_rule_v3/stage1_results', 'task39_locked_v3/results', 'task40_locked_v4/results'] for f in glob.glob(str(W / fo / '*_o0.csv'))]
d = pd.concat([pd.read_csv(f, usecols=['source', 'task', 'p', 'n', 'rep', 'k', 'A', 'B', 'C', 'cv5_A', 'cv5_C']) for f in fs], ignore_index=True)
d['type'] = d.source.str[0].map(TY); d['nk'] = d.n / d.k
st = d.groupby(['type', 'source', 'task', 'p', 'n', 'k'])[['A', 'C', 'cv5_C']].mean().reset_index()
st['gap'] = st.C - st.A; st['nk'] = st.n / st.k
# task signal: C at the largest n of the task, k = 20, p = 71
big = st[st.k == 20].sort_values('n').groupby(['source', 'task', 'p']).tail(1)
sig = big.set_index(['source', 'task', 'p']).C.rename('signal')
st = st.join(sig, on=['source', 'task', 'p'])
print('sources', d.source.nunique(), 'settings', len(st))
bins = [0, 2.5, 3.75, 5, 6.25, 7.5, 10, 12.5, 15, 20, 25, 40]
st['nkbin'] = pd.cut(st.nk, bins, right=False)
def crossover(g):
    m = g.groupby('nkbin', observed=True).gap.mean()
    mids = np.array([iv.mid for iv in m.index]); v = m.values
    for i in range(1, len(v)):
        if v[i - 1] < 0 <= v[i]:
            return float(np.exp(np.interp(0, [v[i - 1], v[i]], np.log([mids[i - 1], mids[i]]))))
    return float('nan') if (v < 0).all() else (0.0 if (v >= 0).all() else float('nan'))
out = {}
out['by_p'] = {int(p): crossover(g) for p, g in st.groupby('p')}
out['by_type'] = {t: crossover(g) for t, g in st.groupby('type')}
out['by_type_p'] = {f'{t}|{p}': crossover(g) for (t, p), g in st.groupby(['type', 'p'])}
sb = pd.cut(st.signal, [-1, 0.05, 0.15, 0.3, 0.45, 0.6, 1.0])
out['by_signal'] = {str(b): [crossover(g), int(g[['source', 'task']].drop_duplicates().shape[0])] for b, g in st.groupby(sb, observed=True)}
out['by_signal_p'] = {f'{b}|{p}': crossover(g) for (b, p), g in st.groupby([sb, 'p'], observed=True)}
out['curve_by_p'] = {int(p): {str(b): round(float(x), 4) for b, x in g.groupby('nkbin', observed=True).gap.mean().items()} for p, g in st.groupby('p')}
out['curve_by_signal'] = {str(s): {str(b): round(float(x), 4) for b, x in g.groupby('nkbin', observed=True).gap.mean().items()} for s, g in st.groupby(sb, observed=True)}
json.dump(out, open(Path(__file__).with_name('empirical.json'), 'w'), indent=1)
for k, v in out.items():
    print(k); print(' ', v)
# same sources at three pool sizes (all types except RPPA, whose larger pools vary in size)
nr = st[st.type != 'RPPA']
out['nonRPPA_by_p'] = {int(p): crossover(g) for p, g in nr.groupby('p')}
out['nonRPPA_by_p_learnable'] = {int(p): crossover(g) for p, g in nr[nr.signal > 0.05].groupby('p')}
out['curve_by_p_nonRPPA'] = {int(p): {str(b): round(float(x), 4) for b, x in g.groupby('nkbin', observed=True).gap.mean().items()} for p, g in nr[nr.signal > 0.05].groupby('p')}
json.dump(out, open(Path(__file__).with_name('empirical.json'), 'w'), indent=1)
print('nonRPPA', out['nonRPPA_by_p'], 'learnable', out['nonRPPA_by_p_learnable'])
