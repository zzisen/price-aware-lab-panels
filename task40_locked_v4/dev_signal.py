#!/usr/bin/env python3
"""Task 40 step 1 (DEV_PLAN.md): develop the signal check on 63 used sources (outer split 0). Type-balanced regret, stratified bootstrap."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
TY = {'G': 'GTEx', 'X': 'CRISPR', 'P': 'proteomics', 'R': 'RPPA'}
fs = glob.glob(str(W / 'task36_rule_v3' / 'stage1_results' / '*_o0.csv')) + glob.glob(str(W / 'task39_locked_v3' / 'results' / '*_o0.csv'))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); best = d[['A', 'B', 'C']].max(axis=1)
hi = d.nk >= 10; cvA = d.cv5_A < d.cv5_C; dd = d.cv5_C - d.cv5_A
low = np.where(cvA, d.A, d.C)
R = {'H': np.where(hi, d.C, low), 'CV': low, 'always_A': d.A, 'always_B': d.B, 'always_C': d.C}
for z in (0.5, 1, 1.5, 2, 3):
    R[f'S_se{z}'] = np.where(hi, np.where(dd > z * d.cv5_se_AmC, d.A, d.C), low)
for t in (0, 0.05, 0.10):
    R[f'S_r2_{t}'] = np.where(hi, np.where(1 - d.cv5_C < t, low, d.C), low)
reg = pd.DataFrame({k: best - v for k, v in R.items()}); reg['source'] = d.source; reg['type'] = d.type
S = reg.groupby(['type', 'source']).mean()                       # source means
types = sorted(S.index.get_level_values(0).unique())
def tb(Sx): return Sx.groupby(level=0).mean().mean()             # type-balanced
rng = np.random.default_rng(20264000); B = 2000
boot = []
for b in range(B):
    parts = []
    for t in types:
        st = S.loc[t]; idx = rng.integers(0, len(st), len(st))
        parts.append(st.iloc[idx].mean())
    boot.append(pd.concat(parts, axis=1).mean(axis=1))
boot = pd.DataFrame(boot)
cols = list(R)
out = {'cells': int(len(d)), 'sources': {t: int(len(S.loc[t])) for t in types},
       'type_balanced': {c: float(tb(S[c])) for c in cols},
       'by_type': {t: {c: float(S.loc[t][c].mean()) for c in cols} for t in types},
       'pooled_source_mean': {c: float(S[c].mean()) for c in cols},
       'overruled_nk10': {c: float((R[c][hi] != d.C[hi]).mean()) for c in cols if c.startswith('S_')}}
fixed = min(['always_A', 'always_B', 'always_C'], key=lambda c: out['type_balanced'][c]); out['best_fixed'] = fixed
out['diff_ci'] = {}
for c in cols:
    for ref in ('H', fixed):
        if c == ref: continue
        x = boot[c] - boot[ref]
        out['diff_ci'][f'{c}-{ref}'] = [float(tb(S[c]) - tb(S[ref])), float(x.quantile(.025)), float(x.quantile(.975))]
# selection per DEV_PLAN
cand = [c for c in cols if c.startswith('S_') or c == 'CV']
ok = [c for c in cand if all(out['by_type'][t][c] - out['by_type'][t]['H'] <= 0.001 for t in ('GTEx', 'RPPA')) and out['by_type']['CRISPR'][c] < out['by_type']['CRISPR']['H']]
out['eligible'] = ok
out['selected'] = min(ok, key=lambda c: out['type_balanced'][c]) if ok else 'H'
json.dump(out, open(HERE / 'dev_signal.json', 'w'), indent=1)
print('cells', out['cells'], out['sources'], 'best fixed', fixed)
print(f"{'rule':12s} {'typebal':>8s} " + ' '.join(f'{t:>10s}' for t in types))
for c in cols:
    print(f'{c:12s} {out["type_balanced"][c]:8.4f} ' + ' '.join(f'{out["by_type"][t][c]:10.4f}' for t in types))
print('eligible', ok, 'selected', out['selected'])
for k, v in out['diff_ci'].items(): print(k, [round(x, 4) for x in v])
print('overruled', out['overruled_nk10'])
