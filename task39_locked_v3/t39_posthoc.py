#!/usr/bin/env python3
"""Task 39 post hoc descriptive analyses (NOT in PLAN_FROZEN_v3.md; written after the locked results were seen; the frozen results stand as
reported in t39_summary.json). Questions: why does redesign lose in CRISPR? Is the loss visible in training data?
1. by data type: tests H1-like comparisons within type (source bootstrap within type);
2. predictability: mean test explained variance of A, B, C by type and n/k band;
3. what cross-validation said at n/k >= 10 (share choosing A, share right) by type;
4. the same diagnostics in the 46 development sources for comparison.
Output t39_posthoc.json / print."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
TY = {'G': 'GTEx', 'X': 'CRISPR', 'P': 'proteomics', 'R': 'RPPA'}


def load(folder):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(folder / '*_o0.csv'))], ignore_index=True)
    d['nk'] = d.n / d.k; d['type'] = d.source.str[0].map(TY); d['best'] = d[['A', 'B', 'C']].max(axis=1)
    cvA = d.cv5_A < d.cv5_C
    d['r_A'], d['r_B'], d['r_C'] = d.best - d.A, d.best - d.B, d.best - d.C
    d['r_H'] = d.best - np.where(d.nk >= 10, d.C, np.where(cvA, d.A, d.C)); d['r_CV'] = d.best - np.where(cvA, d.A, d.C)
    d['cvA'] = cvA; d['est_gain'] = (d.cv5_A - d.cv5_C) / d.cv5_A
    return d


def boot_within(g, a, b, B=2000, seed=20263997):
    src = sorted(g.source.unique()); S = g.groupby('source')[[a, b]].sum().loc[src].to_numpy(); N = g.groupby('source').size().loc[src].to_numpy()
    rng = np.random.default_rng(seed); idx = np.stack([rng.choice(len(src), len(src)) for _ in range(B)])
    v = (S[idx, 0].sum(1) - S[idx, 1].sum(1)) / N[idx].sum(1)
    return [float(g[a].mean() - g[b].mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))]


out = {}
for name, folder in [('locked2', HERE / 'results'), ('development46', W / 'task36_rule_v3' / 'stage1_results')]:
    d = load(folder); res = {}
    for t, g in d.groupby('type'):
        fixed = min(['r_A', 'r_B', 'r_C'], key=lambda c: g[c].mean())
        hi = g[g.nk >= 10]
        res[t] = dict(sources=int(g.source.nunique()), regret={k[2:]: float(g[k].mean()) for k in ['r_A', 'r_B', 'r_C', 'r_H', 'r_CV']},
                      best_fixed=fixed[2:], H_minus_best_fixed=boot_within(g, 'r_H', fixed) if g.source.nunique() > 1 else None,
                      test_ev={p: float(g[p].mean()) for p in 'ABC'},
                      test_ev_nk10={p: float(hi[p].mean()) for p in 'ABC'},
                      nk10_C_leads_A=float((hi.C > hi.A).mean()),
                      nk10_cv_picks_A=float(hi.cvA.mean()), nk10_cv_right_when_A=float(((hi.cvA) & (hi.A > hi.C)).sum() / max(1, hi.cvA.sum())),
                      nk10_mean_cv_est_gain=float(hi.est_gain.median()),
                      gain_by_band={str(b): float((x.C - x.A).mean()) for b, x in g.groupby(pd.cut(g.nk, [0, 5, 10, 20, 50], right=False), observed=True)})
    out[name] = res
    if name == 'locked2':
        nc = d[d.type != 'CRISPR']
        out['locked2_without_CRISPR'] = dict(sources=int(nc.source.nunique()), regret={k[2:]: float(nc[k].mean()) for k in ['r_A', 'r_B', 'r_C', 'r_H', 'r_CV']},
                                             H_minus_C=boot_within(nc, 'r_H', 'r_C'), H_minus_A=boot_within(nc, 'r_H', 'r_A'))
json.dump(out, open(HERE / 't39_posthoc.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
