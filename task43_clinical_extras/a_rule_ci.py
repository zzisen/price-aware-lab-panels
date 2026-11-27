#!/usr/bin/env python3
"""Task 43 (a), post hoc: bootstrap intervals for the regret of the clinical rule chosen in Task 37 stage 2 and for its differences with
comparators. Reads task37_clinical_rule/results/*.csv (unchanged). Regret definitions reproduce summarize_stage2.py exactly (checked against
stage2_summary.json). Two bootstraps, 2000 draws each, seed 20264301:
  H (hierarchical, primary): within each cohort resample outcomes with replacement; within every selected outcome x events cell resample
     repetitions with replacement; overall = mean over cells (cells equally weighted, as in the development summary).
  R (repetitions only): cells fixed, repetitions resampled within cell.
Output a_rule_ci.json, a_rule_ci_print.txt."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent; T37 = W / 'task37_clinical_rule'
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(T37 / 'results' / '*.csv')))], ignore_index=True)
P = ['A', 'B', 'C', 'D', 'base']
d['best'] = d[[f'test_{p}' for p in P]].max(axis=1)


def nested(r, opts):
    return max(opts, key=lambda p: r[f'nest_{p}'])


def cnt(r, t):
    return (r.E / r.d_C if r.d_C > 0 else np.inf) >= t


rules = {'rule (E/d_C>=5, else nested CV incl. no tests)': lambda r: 'C' if cnt(r, 5) else nested(r, ['A', 'C', 'D', 'base']),
         'always redesign (C)': lambda r: 'C',
         'always reuse (A)': lambda r: 'A',
         'always history-trained (B)': lambda r: 'B',
         'always routine bundle (D)': lambda r: 'D',
         'always age and sex only': lambda r: 'base',
         'count only (E/d_C>=5: C, else routine D)': lambda r: 'C' if cnt(r, 5) else 'D',
         'count only (E/d_C>=5: C, else reuse A)': lambda r: 'C' if cnt(r, 5) else 'A',
         'nested CV only (A, C, D, no tests)': lambda r: nested(r, ['A', 'C', 'D', 'base']),
         'nested CV only (A, C, D)': lambda r: nested(r, ['A', 'C', 'D']),
         'omics analogue (E/d_C>=10, else nested A, C, D)': lambda r: 'C' if cnt(r, 10) else nested(r, ['A', 'C', 'D'])}
names = list(rules)
R = np.column_stack([d.best.to_numpy() - np.array([r[f'test_{rules[k](r)}'] for _, r in d.iterrows()]) for k in names])
d['cell'] = d.cohort + '_' + d.outcome + '_' + d.E.astype(str)
d['key'] = d.cohort + '_' + d.outcome
cells = sorted(d.cell.unique())
rows_of = {c: np.flatnonzero(d.cell.to_numpy() == c) for c in cells}
cells_of_key = {k: sorted(d.cell[d.key == k].unique()) for k in sorted(d.key.unique())}
keys_of_coh = {c: sorted(d.key[d.cohort == c].unique()) for c in sorted(d.cohort.unique())}


def point(sel_cells=None):
    sel = sel_cells or cells
    return np.mean([R[rows_of[c]].mean(0) for c in sel], axis=0)


est = point()
# check against the development summary
s2 = json.load(open(T37 / 'stage2_summary.json'))['regret']
chk = {'rule (E/d_C>=5, else nested CV incl. no tests)': 'E/d_C>=5:C|nested3', 'always redesign (C)': 'always_C', 'always reuse (A)': 'always_A',
       'always routine bundle (D)': 'always_D', 'always age and sex only': 'always_base', 'nested CV only (A, C, D, no tests)': 'nested_ACDbase',
       'nested CV only (A, C, D)': 'nested_ACD', 'count only (E/d_C>=5: C, else routine D)': 'E/d_C>=5:C|D', 'always history-trained (B)': 'always_B',
       'omics analogue (E/d_C>=10, else nested A, C, D)': 'E/d_C>=10:C|nested'}
maxdiff = max(abs(est[names.index(a)] - s2[b]) for a, b in chk.items())
assert maxdiff < 1e-12, maxdiff
by_coh = {c: dict(zip(names, np.mean([R[rows_of[x]].mean(0) for x in cells if x.startswith(c + '_')], axis=0).round(5).tolist())) for c in keys_of_coh}
rng = np.random.default_rng(20264301)
B = 2000
bh, br = np.empty((B, len(names))), np.empty((B, len(names)))
bh_coh = {c: np.empty((B, len(names))) for c in keys_of_coh}
for b in range(B):
    # hierarchical
    cm, cmc = [], {c: [] for c in keys_of_coh}
    for coh, ks in keys_of_coh.items():
        for k in rng.choice(ks, len(ks), replace=True):
            for c in cells_of_key[k]:
                ix = rows_of[c]
                v = R[rng.choice(ix, len(ix), replace=True)].mean(0)
                cm.append(v); cmc[coh].append(v)
    bh[b] = np.mean(cm, axis=0)
    for coh in keys_of_coh:
        bh_coh[coh][b] = np.mean(cmc[coh], axis=0)
    # repetitions only
    br[b] = np.mean([R[rng.choice(rows_of[c], len(rows_of[c]), replace=True)].mean(0) for c in cells], axis=0)


def ci(v):
    return [round(float(np.quantile(v, .025)), 5), round(float(np.quantile(v, .975)), 5)]


ri = names.index('rule (E/d_C>=5, else nested CV incl. no tests)')
out = dict(rows=len(d), cells=len(cells), outcomes=len(cells_of_key), seed=20264301, draws=B, check_max_abs_diff_vs_stage2=float(maxdiff),
           regret={k: dict(est=round(float(est[i]), 5), ci_hier=ci(bh[:, i]), ci_reps=ci(br[:, i])) for i, k in enumerate(names)},
           rule_minus={k: dict(est=round(float(est[ri] - est[i]), 5), ci_hier=ci(bh[:, ri] - bh[:, i]), ci_reps=ci(br[:, ri] - br[:, i]),
                               share_hier_below0=round(float(np.mean(bh[:, ri] - bh[:, i] < 0)), 3))
                       for i, k in enumerate(names) if i != ri},
           by_cohort=by_coh,
           by_cohort_rule_minus={coh: {k: dict(est=round(by_coh[coh][names[ri]] - by_coh[coh][k], 5),
                                               ci_hier=ci(bh_coh[coh][:, ri] - bh_coh[coh][:, i])) for i, k in enumerate(names) if i != ri}
                                 for coh in keys_of_coh})
json.dump(out, open(HERE / 'a_rule_ci.json', 'w'), indent=1)
with open(HERE / 'a_rule_ci_print.txt', 'w', encoding='utf-8') as f:
    def p(s=''):
        print(s); f.write(s + '\n')
    p(f'rows {len(d)} cells {len(cells)} outcomes {len(cells_of_key)}; reproduces stage2_summary (max diff {maxdiff:.1e})')
    p('\nregret (cells equally weighted): estimate, 95% CI hierarchical (outcomes within cohort, reps within cell), CI reps only')
    for k, v in out['regret'].items():
        p(f"{k:50s} {v['est']:.4f}  {v['ci_hier']}  {v['ci_reps']}")
    p('\nrule minus comparator')
    for k, v in out['rule_minus'].items():
        p(f"{k:50s} {v['est']:+.4f}  {v['ci_hier']}  {v['ci_reps']}  P(<0) hier {v['share_hier_below0']}")
    p('\nby cohort, rule minus comparator (hierarchical within cohort)')
    for coh, dd in out['by_cohort_rule_minus'].items():
        for k, v in dd.items():
            p(f"{coh} {k:50s} {v['est']:+.4f}  {v['ci_hier']}")
