#!/usr/bin/env python3
"""Task 34 decisions (PLAN_FROZEN.md section 5). Written before any locked result existed.
Regret = best of A, B, C on the test cell minus the chosen panel. D1 from the rows; D2: C if n/k >= 10 else A; D3: C if n/(k ln(e p/k)) >= 3.38
else A. Bootstrap over sources (2000, seed 20263499). Output: summary.json, summary_print.txt (stdout)."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / 'results' / '*.csv'))], ignore_index=True)
d['best'] = d[['A', 'B', 'C']].max(axis=1)
d['nk'] = d.n / d.k; d['cx'] = d.n / (d.k * np.log(np.e * d.p / d.k))
d['D2'] = np.where(d.nk >= 10, 'C', 'A'); d['D3'] = np.where(d.cx >= 3.38, 'C', 'A')
for s in ['D1', 'D2', 'D3']:
    d['ev_' + s] = np.choose(d[s].map({'A': 0, 'B': 1, 'C': 2}).to_numpy(), [d.A.to_numpy(), d.B.to_numpy(), d.C.to_numpy()])
    d['reg_' + s] = d.best - d['ev_' + s]
for m in 'ABC':
    d['reg_' + m] = d.best - d[m]
d['CmA'] = d.C - d.A; d['CmB'] = d.C - d.B
src = sorted(d.source.unique()); G = {s: d[d.source == s] for s in src}


def stat(f):
    rng = np.random.default_rng(20263499); out = []
    for _ in range(2000):
        out.append(f(pd.concat([G[s] for s in rng.choice(src, len(src))], ignore_index=True)))
    return float(f(d)), [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


fixed = {m: float(d['reg_' + m].mean()) for m in 'ABC'}
bestfix = min(fixed, key=fixed.get)
h1 = stat(lambda x: x.reg_D1.mean() - x['reg_' + bestfix].mean())
h2 = stat(lambda x: spearmanr(x.CmA, x.cx)[0] - spearmanr(x.CmA, x.nk)[0])
hi = d[d.nk >= 10]
h3a = float((hi.CmB > 0).mean())
h3b = stat(lambda x: spearmanr(x.CmB, x.nk)[0])
res = dict(cells=len(d), sources=len(src), regret_fixed=fixed, best_fixed=bestfix,
           regret_rules={s: float(d['reg_' + s].mean()) for s in ['D1', 'D2', 'D3']},
           correct_rules={s: float((d['ev_' + s] == d.best).mean()) for s in ['D1', 'D2', 'D3']},
           H1=dict(diff=h1[0], ci=h1[1], passed=h1[1][1] < 0),
           H2=dict(diff=h2[0], ci=h2[1], spearman_cx=float(spearmanr(d.CmA, d.cx)[0]), spearman_nk=float(spearmanr(d.CmA, d.nk)[0]), passed=h2[1][0] > 0),
           H3=dict(share_C_beats_B_at_nk10=h3a, spearman=h3b[0], ci=h3b[1], passed=bool(h3a >= 0.8 and h3b[1][0] > 0)),
           D1_choices=d.D1.value_counts(normalize=True).to_dict())
tab = d.groupby(['p', 'n', 'k'])[['A', 'B', 'C', 'reg_D1', 'reg_D2', 'reg_D3', 'reg_A', 'reg_B', 'reg_C']].mean().round(4)
res['by_p_n_k'] = tab.reset_index().to_dict('records')
res['by_family'] = d.assign(fam=d.source.str[0]).groupby('fam')[['reg_D1', 'reg_A', 'reg_B', 'reg_C', 'CmA', 'CmB']].mean().round(4).to_dict()
json.dump(res, open(HERE / 'summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != 'by_p_n_k'}, indent=1, default=float))
print(tab.to_string())
