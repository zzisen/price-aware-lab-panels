#!/usr/bin/env python3
"""Task 39 locked-test summary, written before any result exists (PLAN_FROZEN_v3.md). Rule v3 = rule v2 = H: n/k >= 10 -> C; else the
lower 5-fold CV error of A and C (ties -> C). Regret = best of A, B, C on the test cell minus the chosen panel. Unit of resampling: source
(2000 draws, seed 20263999). Output t39_summary.json and print."""
import glob
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results'       # argument only for the dry run on Task 36 (old) data
TAG = sys.argv[2] if len(sys.argv) > 2 else 't39'
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(FOLDER / '*_o0.csv'))], ignore_index=True)
d['nk'] = d.n / d.k; d['type'] = d.source.str.split('_').str[0].map({'G': 'GTEx', 'X': 'CRISPR', 'P': 'proteomics', 'R': 'RPPA'})
d['best'] = d[['A', 'B', 'C']].max(axis=1)
d['r_A'], d['r_B'], d['r_C'] = d.best - d.A, d.best - d.B, d.best - d.C
d['r_H'] = d.best - np.where(d.nk >= 10, d.C, np.where(d.cv5_A < d.cv5_C, d.A, d.C))
d['r_NK10'] = d.best - np.where(d.nk >= 10, d.C, d.A)
d['r_CV5'] = d.best - np.where(d.cv5_A < d.cv5_C, d.A, d.C)
d['r_H7.5_BUR'] = d.best - np.where(d.nk >= 7.5, d.C, np.where(d.cv5_A + d.app_A - d.all5_A < d.cv5_C + d.app_C - d.all5_C, d.A, d.C))
fixed = min(['r_A', 'r_B', 'r_C'], key=lambda c: d[c].mean())
src = sorted(d.source.unique()); N = d.groupby('source').size().loc[src].to_numpy()
rng = np.random.default_rng(20263999); IDX = np.stack([rng.choice(len(src), len(src)) for _ in range(2000)])


def diff(a, b):
    S = d.groupby('source')[[a, b]].sum().loc[src].to_numpy()
    v = (S[IDX, 0].sum(1) - S[IDX, 1].sum(1)) / N[IDX].sum(1)
    return [float(d[a].mean() - d[b].mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))]


hi = d[d.nk >= 10]
cb = (hi.C > hi.B).mean()
rho = spearmanr(d.C - d.B, d.nk).statistic
G = {s: g for s, g in d.groupby('source')}
r2 = np.random.default_rng(20263998); rb = []
for _ in range(500):                                                         # source bootstrap for Spearman (500 draws: cost)
    x = pd.concat([G[src[i]] for i in r2.choice(len(src), len(src))])
    rb.append(spearmanr(x.C - x.B, x.nk).statistic)
sett = hi.assign(g=hi.C - hi.A).groupby(['source', 'task', 'p', 'n', 'k']).g.mean()
out = dict(cells=len(d), sources=len(src), sources_by_type={t: int(g.source.nunique()) for t, g in d.groupby('type')},
           regret={c[2:]: float(d[c].mean()) for c in ['r_A', 'r_B', 'r_C', 'r_H', 'r_NK10', 'r_CV5', 'r_H7.5_BUR']},
           regret_by_type={t: {c[2:]: float(g[c].mean()) for c in ['r_A', 'r_B', 'r_C', 'r_H', 'r_NK10', 'r_CV5', 'r_H7.5_BUR']} for t, g in d.groupby('type')},
           best_fixed=fixed[2:],
           H1_H_minus_best_fixed=diff('r_H', fixed), H2_H_minus_NK10=diff('r_H', 'r_NK10'),
           H3=dict(C_beats_B_nk10=float(cb), spearman_CminusB_nk=float(rho), spearman_lo=float(np.quantile(rb, .025)), spearman_hi=float(np.quantile(rb, .975))),
           H4=dict(setting_C_leads_A_nk10=float((sett > 0).mean()), cell_C_leads_A_nk10=float((hi.C > hi.A).mean()),
                   setting_A_leads_by_gt002=float((sett < -0.02).mean())),
           secondary=dict(H_minus_C=diff('r_H', 'r_C'), H_minus_CV5=diff('r_H', 'r_CV5'), H75BUR_minus_H=diff('r_H7.5_BUR', 'r_H'),
                          H_beats_C_sources=int(sum(G[s].r_H.mean() < G[s].r_C.mean() for s in src)),
                          source_mean_gain_CminusA_nk10={s: float((g.C - g.A).mean()) for s, g in hi.groupby('source')}))
out['pass'] = dict(H1=out['H1_H_minus_best_fixed'][2] < 0, H2=out['H2_H_minus_NK10'][2] < 0,
                   H3=cb >= 0.8 and out['H3']['spearman_lo'] > 0, H4=out['H4']['setting_C_leads_A_nk10'] >= 0.8)
json.dump(out, open(HERE / f'{TAG}_summary.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
