#!/usr/bin/env python3
"""Task 42 summary (rule v4 unchanged; statistics as summarize_t40.py, seed 20264299). Usage: summarize_t42.py <tag> <folder> [...]
Locked tags: t42T (folder results_T) and t42S (folder results_S): 12 new TCGA RNA-seq cohorts.
Primary hypotheses (PLAN_FROZEN_v5.md section 6) are the T42_* keys: H1 non-inferiority (upper bound of v4 - best fixed < 0.005),
H2 (v4 - n/k alone, upper < 0), H3 (v4 - always A, upper < 0). Secondary T42_common_grid: the same on settings with n <= 100.
Keys H1..H6 and 'pass' are the Task 40 statistics, computed for reference only; they are not hypotheses of this plan."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
TY = {'G': 'GTEx', 'R': 'RPPA', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
ABUNDANCE = {'GTEx', 'RPPA', 'proteomics', 'TCGA_RNAseq'}
TAU, NB, SEED = 0.10, 2000, 20264299
PERTURB = {'CRISPR', 'RNAi'}
tag, folders = sys.argv[1], sys.argv[2:]
files = [f for fo in folders for f in glob.glob(str(Path(fo) / '*_o0.csv'))]
LOCKED = {'t42T': ('results_T', 'T'), 't42S': ('results_S', 'S')}
if tag in LOCKED:                                                   # locked run: completeness and no contamination
    assert folders == [LOCKED[tag][0]] and len(files) == 216, (folders, len(files))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); best = d[['A', 'B', 'C']].max(axis=1)
assert d.type.notna().all(), 'unknown source prefix'
if tag in LOCKED:
    assert set(d.type) == {'TCGA_RNAseq'} and d.source.nunique() == 12 and set(d['pipe']) == {LOCKED[tag][1]}, (set(d.type), d.source.nunique())
    assert not set(d.source) & {'T_' + c for c in ('BRCA', 'KIRC', 'HNSC', 'LGG', 'LUAD', 'THCA', 'LUSC', 'PRAD', 'STAD', 'BLCA')}
    NEW12 = {'T_' + c for c in ('CESC', 'COAD', 'ESCA', 'GBM', 'KIRP', 'LIHC', 'OV', 'PAAD', 'PCPG', 'SARC', 'TGCT', 'UCEC')}
    assert set(d.source) == NEW12, set(d.source) ^ NEW12
    assert len(d) == {'T': 3240, 'S': 20736}[LOCKED[tag][1]], len(d)
assert d['pipe'].nunique() == 1
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
# Task 42 primary hypotheses (set after development on analysed data, before any new value was read; PLAN_FROZEN_v5.md section 6)
MARGIN = 0.005
o['T42_H1_NI_v4_minus_best_fixed'] = o['H1_v4_minus_best_fixed']
o['T42_H2_v4_minus_NK10'] = o['H2_v4_minus_NK10']
o['T42_H3_v4_minus_always_A'] = [o['type_balanced']['v4'] - o['type_balanced']['always_A']] + ci(bt.v4 - bt.always_A)
o['T42_secondary'] = {'v4_minus_H': o['secondary']['v4_minus_H'], 'v4_minus_CV': o['secondary']['v4_minus_CV'],
                      'v4_minus_always_C': [o['type_balanced']['v4'] - o['type_balanced']['always_C']] + ci(bt.v4 - bt.always_C)}
def _tb_diff(dd, a_, b_, seed):
    bestx = dd[['A', 'B', 'C']].max(axis=1)
    hi_ = dd.nk >= 10; low_ = np.where(dd.cv5_A < dd.cv5_C, dd.A, dd.C); learn_ = (1 - dd.cv5_C) >= TAU
    RR = {'v4': np.where(hi_ & learn_, dd.C, low_), 'NK10': np.where(hi_, dd.C, dd.A), 'always_A': dd.A, 'always_B': dd.B, 'always_C': dd.C}
    rg = pd.DataFrame({k_: bestx - v_ for k_, v_ in RR.items()}); rg['source'] = dd.source.values
    Sx = rg.groupby('source').mean()
    fx = min(['always_A', 'always_B', 'always_C'], key=lambda c_: Sx[c_].mean())
    r_ = np.random.default_rng(seed); bs = [Sx.iloc[r_.integers(0, len(Sx), len(Sx))].mean() for _ in range(NB)]
    bd = pd.DataFrame(bs)
    out_ = {'best_fixed': fx}
    for nm, (u, v) in {'H1_NI_v4_minus_best_fixed': ('v4', fx), 'H2_v4_minus_NK10': ('v4', 'NK10'), 'H3_v4_minus_always_A': ('v4', 'always_A')}.items():
        out_[nm] = [float(Sx[u].mean() - Sx[v].mean())] + ci(bd[u] - bd[v])
    return out_
if d.type.nunique() == 1:
    o['T42_common_grid_n_le_100'] = _tb_diff(d[d.n <= 100].reset_index(drop=True), None, None, SEED + 1)
o['T42_pass'] = {'H1_NI': o['T42_H1_NI_v4_minus_best_fixed'][2] < MARGIN, 'H2': o['T42_H2_v4_minus_NK10'][2] < 0,
                 'H3': o['T42_H3_v4_minus_always_A'][2] < 0}
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
print('Task 40 statistics, reference only (not hypotheses of this plan):', o['pass'])
print('T42 H1 NI (margin 0.005)', [round(x, 4) for x in o['T42_H1_NI_v4_minus_best_fixed']], 'H2', [round(x, 4) for x in o['T42_H2_v4_minus_NK10']], 'H3 vs A', [round(x, 4) for x in o['T42_H3_v4_minus_always_A']])
print('T42 PASS', o['T42_pass'])
if 'T42_common_grid_n_le_100' in o:
    print('T42 secondary, common grid n <= 100', {k: (v if isinstance(v, str) else [round(x, 4) for x in v]) for k, v in o['T42_common_grid_n_le_100'].items()})
