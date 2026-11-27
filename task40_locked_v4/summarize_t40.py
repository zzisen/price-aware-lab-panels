#!/usr/bin/env python3
"""Task 40 summary (written before the locked run; PLAN_FROZEN_v4.md). Usage: summarize_t40.py <tag> <folder> [<folder> ...]
Rule v4: n/k >= 10 and cross-validated explained variance of the redesigned panel (1 - cv5_C) >= 0.10 -> redesign (C);
otherwise 5-fold CV between A and C (ties to C). Regret = best of A, B, C on the test cell minus chosen. Type-balanced: mean over sources
within data type, then equal weight per type. Stratified source bootstrap (within type), 2000 draws, seed 20264099."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TY = {'G': 'GTEx', 'R': 'RPPA', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
ABUNDANCE = {'GTEx', 'RPPA', 'proteomics', 'TCGA_RNAseq'}
TAU, NB, SEED = 0.10, 2000, 20264099
PERTURB = {'CRISPR', 'RNAi'}
tag, folders = sys.argv[1], sys.argv[2:]
files = [f for fo in folders for f in glob.glob(str(Path(fo) / '*_o0.csv'))]
if tag == 't40':                                                    # locked run: completeness and no contamination
    assert folders == ['results'] and len(files) == 252, (folders, len(files))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); best = d[['A', 'B', 'C']].max(axis=1)
assert d.type.notna().all(), 'unknown source prefix'
if tag == 't40':
    assert set(d.type) == {'TCGA_RNAseq', 'RNAi'} and d.source.nunique() == 14, (set(d.type), d.source.nunique())
hi = d.nk >= 10; low = np.where(d.cv5_A < d.cv5_C, d.A, d.C); learn = (1 - d.cv5_C) >= TAU
R = {'v4': np.where(hi & learn, d.C, low), 'H': np.where(hi, d.C, low), 'NK10': np.where(hi, d.C, d.A), 'CV': low,
     'always_A': d.A, 'always_B': d.B, 'always_C': d.C}
reg = pd.DataFrame({k: best - v for k, v in R.items()}); reg['source'] = d.source; reg['type'] = d.type
S = reg.groupby(['type', 'source']).mean(); types = sorted(S.index.get_level_values(0).unique())
tb = lambda Sx: float(Sx.groupby(level=0).mean().mean())
rng = np.random.default_rng(SEED); boot = []
for b in range(NB):
    boot.append(pd.concat([S.loc[t].iloc[rng.integers(0, len(S.loc[t]), len(S.loc[t]))].mean() for t in types], axis=1))
bt = pd.DataFrame([b.mean(axis=1) for b in boot])                  # type-balanced per draw
bty = {t: pd.DataFrame([b[i] for b in boot]) for i, t in enumerate(types)}
ci = lambda x: [float(x.quantile(.025)), float(x.quantile(.975))]
o = {'tag': tag, 'cells': int(len(d)), 'sources': {t: int(len(S.loc[t])) for t in types}}
o['type_balanced'] = {c: tb(S[c]) for c in R}
o['by_type'] = {t: {c: float(S.loc[t][c].mean()) for c in R} for t in types}
fixed = min(['always_A', 'always_B', 'always_C'], key=lambda c: o['type_balanced'][c]); o['best_fixed'] = fixed
o['H1_v4_minus_best_fixed'] = [o['type_balanced']['v4'] - o['type_balanced'][fixed]] + ci(bt.v4 - bt[fixed])
o['H2_v4_minus_NK10'] = [o['type_balanced']['v4'] - o['type_balanced']['NK10']] + ci(bt.v4 - bt.NK10)
o['H3_v4_minus_H_abundance'] = {t: [o['by_type'][t]['v4'] - o['by_type'][t]['H']] + ci(bty[t].v4 - bty[t].H) for t in types if t in ABUNDANCE}
# H4: settings (source x task x p x n x k, averaged over repeats) with n/k >= 10 and learnable on average; share with C > A
st = d[hi].assign(learn=learn[hi].astype(float)).groupby(['source', 'task', 'p', 'n', 'k'])[['A', 'C', 'learn']].mean()
st = st[st.learn >= 0.5]
o['H5_v4_minus_CV'] = [o['type_balanced']['v4'] - o['type_balanced']['CV']] + ci(bt.v4 - bt.CV)
o['H6_v4_minus_H_perturbation'] = {t: [o['by_type'][t]['v4'] - o['by_type'][t]['H']] + ci(bty[t].v4 - bty[t].H) for t in types if t in PERTURB}
o['H4_settings'] = int(len(st)); o['H4_share_C_leads_A'] = float((st.C > st.A).mean()) if len(st) else None
o['H4_by_type'] = {t: [int(len(g)), float((g.C > g.A).mean())] for t, g in st.groupby(st.index.get_level_values(0).str[0].map(TY))}
o['secondary'] = {'v4_minus_H': [o['type_balanced']['v4'] - o['type_balanced']['H']] + ci(bt.v4 - bt.H),
                  'v4_minus_CV': [o['type_balanced']['v4'] - o['type_balanced']['CV']] + ci(bt.v4 - bt.CV),
                  'v4_minus_H_by_type': {t: [o['by_type'][t]['v4'] - o['by_type'][t]['H']] + ci(bty[t].v4 - bty[t].H) for t in types},
                  'v4_minus_fixed_by_type': {t: {f: [o['by_type'][t]['v4'] - o['by_type'][t][f]] + ci(bty[t].v4 - bty[t][f]) for f in ('always_A', 'always_C')} for t in types},
                  'best_fixed_within_type': {t: min(['always_A', 'always_B', 'always_C'], key=lambda c: o['by_type'][t][c]) for t in types},
                  'typewise_best_fixed_type_balanced': float(np.mean([min(o['by_type'][t][c] for c in ('always_A', 'always_B', 'always_C')) for t in types])),
                  'pooled_source_mean': {c: float(S[c].mean()) for c in R},
                  'v4_overrides_count_nk10': {t: float((~learn[hi & (d.type == t)]).mean()) for t in types},
                  'test_EV_by_type': {t: {m: float(d.loc[d.type == t, m].mean()) for m in 'ABC'} for t in types},
                  'per_source_v4_minus_best_fixed': {s: float(S.loc[(t, s), 'v4'] - S.loc[(t, s), fixed]) for t, s in S.index},
                  'per_source_v4_minus_H': {s: float(S.loc[(t, s), 'v4'] - S.loc[(t, s), 'H']) for t, s in S.index}}
o['pass'] = {'H1': o['H1_v4_minus_best_fixed'][2] < 0, 'H2': o['H2_v4_minus_NK10'][2] < 0,
             'H3': bool(o['H3_v4_minus_H_abundance']) and all(v[2] < 0.001 for v in o['H3_v4_minus_H_abundance'].values()),
             'H4': (o['H4_share_C_leads_A'] or 0) >= 0.80, 'H5': o['H5_v4_minus_CV'][2] < 0,
             'H6': bool(o['H6_v4_minus_H_perturbation']) and all(v[2] < 0 for v in o['H6_v4_minus_H_perturbation'].values())}
json.dump(o, open(HERE / f'{tag}_summary.json', 'w'), indent=1)
print(tag, 'cells', o['cells'], o['sources'], 'best fixed', fixed)
print('type-balanced', {c: round(v, 4) for c, v in o['type_balanced'].items()})
for t in types: print(' ', t, {c: round(v, 4) for c, v in o['by_type'][t].items()})
for k in ('H1_v4_minus_best_fixed', 'H2_v4_minus_NK10'): print(k, [round(x, 4) for x in o[k]])
print('H3', {t: [round(x, 4) for x in v] for t, v in o['H3_v4_minus_H_abundance'].items()})
print('H5 v4-CV', [round(x, 4) for x in o['H5_v4_minus_CV']], 'H6', {t: [round(x, 4) for x in v] for t, v in o['H6_v4_minus_H_perturbation'].items()})
print('within-type best fixed', o['secondary']['best_fixed_within_type'], 'typewise best fixed (type-balanced)', round(o['secondary']['typewise_best_fixed_type_balanced'], 4))
print('v4 - fixed by type', {t: {f: [round(x, 4) for x in v] for f, v in dd.items()} for t, dd in o['secondary']['v4_minus_fixed_by_type'].items()})
print('H4', o['H4_settings'], o['H4_share_C_leads_A'], o['H4_by_type'])
print('v4-H', [round(x, 4) for x in o['secondary']['v4_minus_H']], 'v4-CV', [round(x, 4) for x in o['secondary']['v4_minus_CV']])
print('override share at nk>=10', {t: round(v, 3) for t, v in o['secondary']['v4_overrides_count_nk10'].items()})
print('test EV', {t: {m: round(x, 3) for m, x in v.items()} for t, v in o['secondary']['test_EV_by_type'].items()})
print('PASS', o['pass'])
