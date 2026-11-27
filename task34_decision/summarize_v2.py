#!/usr/bin/env python3
"""Task 34 decisions, plan v2 (PLAN_FROZEN_v2.md section 5). Usage: summarize_v2.py <dev_results|results> <out_prefix>
Rule H (hybrid): C if n/k >= 10, else the lower 5-fold inner-CV error of A and C. Regret = best of A, B, C on the test cell minus the chosen.
Bootstrap over sources (2000, seed 20263499)."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
folder, prefix = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / folder / '*.csv'))], ignore_index=True)
d['best'] = d[['A', 'B', 'C']].max(axis=1); d['nk'] = d.n / d.k
cv = np.where(d.inner_A < d.inner_C, 'A', 'C')
choice = {'H': np.where(d.nk >= 10, 'C', cv), 'CV': cv, 'NK': np.where(d.nk >= 10, 'C', 'A'),
          'D3': np.where(d.n / (d.k * np.log(np.e * d.p / d.k)) >= 3.38, 'C', 'A'), 'LNP': np.where(d.n / (d.k * np.log(d.p)) >= 10 / np.log(71), 'C', 'A'),
          'D1v1': d.D1.to_numpy()}
for r, ch in choice.items():
    d['reg_' + r] = d.best - np.select([ch == 'A', ch == 'B', ch == 'C'], [d.A, d.B, d.C])
for m in 'ABC':
    d['reg_' + m] = d.best - d[m]
d['CmB'] = d.C - d.B; d['est'] = (d.inner_A - d.inner_C) / d.inner_A; d['real'] = d.C - d.A
src = sorted(d.source.unique()); G = {s: d[d.source == s] for s in src}


def boot(f):
    rng = np.random.default_rng(20263499)
    v = [f(pd.concat([G[s] for s in rng.choice(src, len(src))], ignore_index=True)) for _ in range(2000)]
    return float(f(d)), [float(np.quantile(v, .025)), float(np.quantile(v, .975))]


fixed = {m: float(d['reg_' + m].mean()) for m in 'ABC'}; bestfix = min(fixed, key=fixed.get)
h1 = boot(lambda x: x.reg_H.mean() - x['reg_' + bestfix].mean())
h2 = boot(lambda x: x.reg_H.mean() - x.reg_NK.mean())
hi = d[d.nk >= 10]; h3a = float((hi.CmB > 0).mean()); h3b = boot(lambda x: spearmanr(x.CmB, x.nk)[0])
q = d.assign(qq=pd.qcut(d.est, 5, labels=False)).groupby('qq')[['est', 'real']].mean()
res = dict(folder=folder, cells=len(d), sources=len(src), regret_fixed=fixed, best_fixed=bestfix,
           regret_rules={r: float(d['reg_' + r].mean()) for r in choice},
           H1=dict(diff=h1[0], ci=h1[1], passed=h1[1][1] < 0),
           H2=dict(diff=h2[0], ci=h2[1], passed=h2[1][1] < 0),
           H3=dict(share_C_beats_B_at_nk10=h3a, spearman=h3b[0], ci=h3b[1], passed=bool(h3a >= 0.8 and h3b[1][0] > 0)),
           sources_H_beats_bestfixed=float(np.mean([G[s].reg_H.mean() < G[s]['reg_' + bestfix].mean() for s in src])),
           sources_H_beats_NK=float(np.mean([G[s].reg_H.mean() < G[s].reg_NK.mean() for s in src])),
           gain_calibration=dict(spearman=float(spearmanr(d.est, d.real)[0]), sign_agreement=float(((d.est > 0) == (d.real > 0)).mean()),
                                 quintiles=q.round(4).to_dict('list')),
           by_family={f: {c: float(d[d.source.str[0] == f][c].mean()) for c in ['reg_H', 'reg_C', 'reg_A', 'reg_NK', 'reg_CV']} for f in 'GR'},
           low_nk_C_wins_by_p=d[d.nk < 10].groupby('p').real.apply(lambda s: float((s > 0).mean())).to_dict())
tab = d.groupby(['n', 'k'])[['reg_H', 'reg_C', 'reg_A', 'reg_NK', 'reg_CV']].mean().round(4)
res['by_n_k'] = tab.reset_index().to_dict('records')
json.dump(res, open(HERE / f'{prefix}_summary_v2.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'by_n_k'}, indent=1, default=float)); print(tab.to_string())
