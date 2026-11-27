#!/usr/bin/env python3
"""Task 44 summary: by grouping (hgnc, corum, reactome, random), mean test explained variance of A and C, C - A at n/k >= 10, share of cells with
n/k >= 10 judged learnable (1 - cv5_C >= 0.10), and mean regret of the rule v4, the rule without the check (H), cross-validation alone and fixed practices.
Intervals: bootstrap over the 12 tasks of each grouping (2 partitions x 6 tasks), 2000 draws, seed 20264401."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / 'results' / '*_o0.csv'))], ignore_index=True)
d['grouping'] = d.source.str.split('_').str[0]; d['nk'] = d.n / d.k; hi = d.nk >= 10
best = d[['A', 'B', 'C']].max(axis=1); low = np.where(d.cv5_A < d.cv5_C, d.A, d.C); learn = (1 - d.cv5_C) >= 0.10
R = {'v4': np.where(hi & learn, d.C, low), 'H': np.where(hi, d.C, low), 'CV': low, 'always_A': d.A, 'always_C': d.C}
for k, v in R.items():
    d['r_' + k] = best - v
d['gap'] = d.C - d.A; d['learn'] = learn.astype(float)
out = {}; rng = np.random.default_rng(20264401)
for g, h in d.groupby('grouping'):
    T = h.groupby('task')
    agg = T[['A', 'C', 'r_v4', 'r_H', 'r_CV', 'r_always_A', 'r_always_C']].mean()
    hh = h[h.nk >= 10].groupby('task')[['gap', 'learn']].mean()
    o = {c: float(agg[c].mean()) for c in agg}
    o['C_minus_A_nk10'] = float(hh.gap.mean()); o['learnable_share_nk10'] = float(hh.learn.mean())
    bs = [agg.iloc[rng.integers(0, len(agg), len(agg))].mean() for _ in range(2000)]; bs = pd.DataFrame(bs)
    hb = pd.DataFrame([hh.iloc[rng.integers(0, len(hh), len(hh))].mean() for _ in range(2000)])
    o['ci_C'] = [float(bs.C.quantile(.025)), float(bs.C.quantile(.975))]
    o['ci_C_minus_A_nk10'] = [float(hb.gap.quantile(.025)), float(hb.gap.quantile(.975))]
    o['v4_minus_H'] = [float((bs.r_v4 - bs.r_H).mean())] + [float((bs.r_v4 - bs.r_H).quantile(q)) for q in (.025, .975)]
    o['v4_minus_always_A'] = [float((agg.r_v4 - agg.r_always_A).mean())] + [float((bs.r_v4 - bs.r_always_A).quantile(q)) for q in (.025, .975)]
    o['tasks'] = int(len(agg))
    out[g] = o
json.dump(out, open(HERE / 't44_summary.json', 'w'), indent=1)
for g, o in out.items():
    print(g, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) else v) for k, v in o.items()})
