#!/usr/bin/env python3
"""Task 30 C (post hoc, PLAN.md): split-level spread of the appendix figure S4 comparison (Task 17 G1). No model is refitted."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / 'task17_gates'))
import summarize_gates as S  # noqa: E402
g = pd.read_csv(W / 'task17_gates/summary/g1_by_source.csv')
d = S.g1_rows(); d = d[d.n.astype(str) == '100']
# per task x outer split: pooled EV over repetitions; NHANES sources have one task, as in the summary
a = d.groupby(['source', 'task', 'outer', 'k', 'predictor', 'method'])[['sse', 'c0']].sum().reset_index(); a['ev'] = 1 - a.sse / a.c0
a = a.groupby(['source', 'outer', 'k', 'predictor', 'method']).ev.mean().reset_index()   # mean over tasks within a source and split
rows, summ = [], []
rng = np.random.default_rng(20261113)
for (k, p), gg in g.groupby(['k', 'predictor']):
    diffs = []
    for _, r in gg.iterrows():
        h = a[(a.source == r.source) & (a.k == k) & (a.predictor == p)].pivot_table(index='outer', columns='method', values='ev')
        dd = h[r.aware_side] - h[r.best_reuse]
        for o, v in dd.items():
            rows.append(dict(k=k, predictor=p, source=r.source, outer=o, aware=r.aware_side, reuse=r.best_reuse, diff=float(v)))
        diffs.append(dd.to_numpy())
    D = np.array(diffs)                                   # sources x splits
    src_mean = D.mean(1); boot = [src_mean[rng.integers(0, len(src_mean), len(src_mean))].mean() for _ in range(2000)]
    summ.append(dict(k=k, predictor=p, cells=int(D.size), lead_positive=int((D > 0).sum()), lead_ge_002=int((D >= 0.02).sum()), min_diff=float(D.min()),
                     mean_lead=float(src_mean.mean()), low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975)),
                     check_mean_vs_summary=float(abs(src_mean - gg.sort_values('source').set_index('source').loc[gg.source, 'aware_minus_reuse'].to_numpy()).max())))
s = pd.DataFrame(summ)
pd.set_option('display.width', 220)
print(s.round(4).to_string(index=False))
pd.DataFrame(rows).to_csv(Path(__file__).parent / 't30_bench_splits.csv', index=False); s.to_csv(Path(__file__).parent / 't30_bench_summary.csv', index=False)
n_tasks = d.groupby('source').task.nunique().to_dict(); size = d.groupby('source').n_actual.mean().to_dict()
print('tasks per source', n_tasks); print(json.dumps({'reps_per_split': int(d.rep.nunique())}))
