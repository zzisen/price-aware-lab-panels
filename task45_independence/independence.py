#!/usr/bin/env python3
"""Task 45 (additional analysis): shared units across sources, and decision-rule comparisons under more conservative resampling.
A. Overlap audit: GTEx donors across tissues within each set; TCGA patients shared between RNA-seq cohorts (tests 3, 4) and the RPPA cohorts analysed
   earlier; cell lines shared between the RNAi sets (test 3) and the CRISPR sets (test 2) via DepMap Model.csv; CCLE proteomics lines vs CRISPR.
B. For locked tests 1-4, rule minus comparators with (i) the original source bootstrap, (ii) a two-stage bootstrap (sources within type, then tasks
   within source), (iii) platform clustering: all GTEx tissues of a test form one unit and the partitions of one screen (CRISPR, RNAi, proteomics) one
   unit; TCGA cohorts stay separate units; units are resampled within type (a type with one unit contributes without sampling variation, which is stated).
Rule per test: test 1 = H (count + CV, Task 34 columns), tests 2 = H, tests 3-4 = v4. Regret as in the frozen summaries."""
import glob, gzip, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent; EXT = W / 'external_data'
TY = {'G': 'GTEx', 'R': 'RPPA', 'X': 'CRISPR', 'P': 'proteomics', 'T': 'TCGA_RNAseq', 'N': 'RNAi'}
NB, SEED, TAU = 2000, 20264501, 0.10
out = {}


def gtex_donors(folder):
    res = {}
    for f in sorted((EXT / folder).glob('*.gct.gz')):
        with gzip.open(f, 'rt') as h:
            h.readline(); h.readline(); hdr = h.readline().rstrip('\n').split('\t')[3:]
        res[f.name.split('_v8_')[1][:-7]] = {'-'.join(s.split('-')[:2]) for s in hdr}
    return res


A = {}
for name, folder in [('development_12', 'gtex_v8_tissues'), ('locked1_17', 'gtex_v8_tissues_fresh'), ('locked2_10', 'gtex_v8_tissues_fresh2')]:
    d = gtex_donors(folder); allc = pd.Series([x for v in d.values() for x in v]).value_counts()
    pair = [len(d[a] & d[b]) / min(len(d[a]), len(d[b])) for i, a in enumerate(d) for b in list(d)[i + 1:]]
    A[f'GTEx_{name}'] = dict(tissues=len(d), donors=int(len(allc)), donors_in_2plus_tissues_share=float((allc >= 2).mean()),
                            mean_pairwise_overlap_of_smaller=float(np.mean(pair)))
rppa = pd.read_csv(EXT / 'tcga_rppa_xena' / 'TCGA-RPPA-pancan-clean.xena.gz', sep='\t', index_col=0, nrows=1).columns
rppa_pat = {s[:12] for s in rppa if s.endswith('-01')}
for tag, folder, pools in [('locked3', 'task40_locked_v4', 'T_*.pkl'), ('locked4', 'task42_locked_v5', 'T_*.pkl')]:
    import pickle
    pats = {}
    for pk in sorted((W / folder / 'pools').glob(pools)):
        x, _ = pickle.load(open(pk, 'rb')); pats[pk.stem] = {s[:12] for s in x.index}
    allp = set().union(*pats.values())
    A[f'TCGA_RNAseq_{tag}'] = dict(cohorts=len(pats), patients=len(allp), share_with_RPPA_any=float(len(allp & rppa_pat) / len(allp)),
                                  patients_in_2plus_cohorts=int(sum(pd.Series([p for v in pats.values() for p in v]).value_counts() >= 2)))
m = pd.read_csv(EXT / 'depmap24q4' / 'Model.csv', usecols=['ModelID', 'CCLEName'])
ach2ccle = dict(zip(m.ModelID, m.CCLEName))
import pickle
cr = set()
for pk in sorted((W / 'task39_locked_v3' / 'pools').glob('X_crispr*.pkl')):
    x, _ = pickle.load(open(pk, 'rb')); cr |= {ach2ccle.get(a) for a in x.index}
rn = set()
for pk in sorted((W / 'task40_locked_v4' / 'pools').glob('N_rnai*.pkl')):
    x, _ = pickle.load(open(pk, 'rb')); rn |= set(x.index)
pr = set()
for pk in sorted((W / 'task39_locked_v3' / 'pools').glob('P_ccle*.pkl')):
    x, _ = pickle.load(open(pk, 'rb')); pr |= {s.split('_TenPx')[0] for s in x.index}
A['RNAi_lines_also_in_CRISPR_sets'] = dict(rnai_lines=len(rn), shared=len(rn & cr), share=float(len(rn & cr) / len(rn)))
A['proteomics_lines_also_in_CRISPR_sets'] = dict(lines=len(pr), shared=len(pr & cr), share=float(len(pr & cr) / len(pr)))
out['A_overlap'] = A


def regrets(files, kind):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); best = d[['A', 'B', 'C']].max(axis=1); hi = d.nk >= 10
    if kind == 't1':
        low = np.where(d.inner_A < d.inner_C, d.A, d.C)
        R = {'rule': np.where(hi, d.C, low), 'CV': low}
    else:
        low = np.where(d.cv5_A < d.cv5_C, d.A, d.C)
        rule = np.where(hi & ((1 - d.cv5_C) >= TAU), d.C, low) if kind == 'v4' else np.where(hi, d.C, low)
        R = {'rule': rule, 'CV': low}
    R.update({'NK10': np.where(hi, d.C, d.A), 'always_A': d.A, 'always_B': d.B, 'always_C': d.C})
    r = pd.DataFrame({k: best - v for k, v in R.items()}); r['source'] = d.source; r['type'] = d.type; r['task'] = d.task
    return r


def unit_of(src, typ):
    return {'GTEx': 'GTEx', 'CRISPR': 'CRISPR', 'RNAi': 'RNAi', 'proteomics': 'proteomics'}.get(typ, src)


def summarise(r, balanced):
    res = {}
    r = r.assign(utype=r.type)
    if not balanced:
        r['type'] = 'all'
    S = r.groupby(['type', 'source']).mean(numeric_only=True); types = sorted(S.index.get_level_values(0).unique())
    tb = lambda X: X.groupby(level=0).mean().mean()
    fixed = min(['always_A', 'always_B', 'always_C'], key=lambda c: tb(S[c]))
    cols = ['rule', 'CV', 'NK10', fixed]
    pt = {c: float(tb(S[c])) for c in cols}
    rng = np.random.default_rng(SEED)
    T = r.groupby(['type', 'source', 'task']).mean(numeric_only=True)
    U = r.assign(unit=[unit_of(s, t) for s, t in zip(r.source, r.utype)]).groupby(['type', 'unit']).mean(numeric_only=True)
    b1, b2, b3 = [], [], []
    for _ in range(NB):
        a1, a2, a3 = [], [], []
        for t in types:
            s = S.loc[t]; idx = rng.integers(0, len(s), len(s)); a1.append(s.iloc[idx][cols].mean())
            srcs = s.index[idx]; parts = []
            for so in srcs:
                tt = T.loc[(t, so)]; parts.append(tt.iloc[rng.integers(0, len(tt), len(tt))][cols].mean())
            a2.append(pd.concat(parts, axis=1).mean(axis=1))
            u = U.loc[t]; a3.append(u.iloc[rng.integers(0, len(u), len(u))][cols].mean())
        b1.append(pd.concat(a1, axis=1).mean(axis=1)); b2.append(pd.concat(a2, axis=1).mean(axis=1)); b3.append(pd.concat(a3, axis=1).mean(axis=1))
    B = {'source': pd.DataFrame(b1), 'two_stage': pd.DataFrame(b2), 'platform_units': pd.DataFrame(b3)}
    upt = {c: float(U[c].groupby(level=0).mean().mean()) for c in cols}
    res['best_fixed'] = fixed; res['units_per_type'] = {t: int(len(U.loc[t])) for t in types}; res['sources_per_type'] = {t: int(len(S.loc[t])) for t in types}
    for c in ['CV', 'NK10', fixed]:
        res[f'rule_minus_{c}'] = {'estimate': pt['rule'] - pt[c], 'estimate_platform_units': upt['rule'] - upt[c],
                                  **{k: [float((v['rule'] - v[c]).quantile(.025)), float((v['rule'] - v[c]).quantile(.975))] for k, v in B.items()}}
    return res


tests = {
 'locked1': (glob.glob(str(W / 'task34_decision' / 'results' / '*.csv')), 't1'),
 'locked2': (glob.glob(str(W / 'task39_locked_v3' / 'results' / '*_o0.csv')), 'H'),
 'locked3': (glob.glob(str(W / 'task40_locked_v4' / 'results' / '*_o0.csv')), 'v4'),
 'locked4_trees': (glob.glob(str(W / 'task42_locked_v5' / 'results_T' / '*_o0.csv')), 'v4'),
 'locked4_QR_lasso': (glob.glob(str(W / 'task42_locked_v5' / 'results_S' / '*_o0.csv')), 'v4'),
}
B = {}
for k, (fs, kind) in tests.items():
    r = regrets(fs, kind)
    B[k] = summarise(r, k == 'locked3')                         # tests 1, 2 and 4 were summarised pooled over sources; test 3 type-balanced; print(k, json.dumps(B[k], default=float)[:900], flush=True)
out['B_resampling'] = B
json.dump(out, open(HERE / 'independence.json', 'w'), indent=1, default=float)
print(json.dumps(out['A_overlap'], indent=1, default=float))
