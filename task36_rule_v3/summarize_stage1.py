#!/usr/bin/env python3
"""Task 36 stage 1 summary (STAGE1_DEV_PLAN.md). Usage: summarize_stage1.py <results folder> <prefix>
Regret = best of A, B, C on the test cell minus the chosen panel. Decision methods use training-only quantities.
Selection of v3 (written before the run): replaces H only if mean regret is at least 0.001 lower over all sources, not higher in either
data type, and the source-bootstrap upper bound of the difference is below 0; otherwise H is kept. Simplest qualifying method wins."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
folder, prefix = sys.argv[1], sys.argv[2]
OUTER = sys.argv[3] if len(sys.argv) > 3 else '0'   # main analysis: outer split 0 only (plan adjustment 04:30); 'all' = sensitivity
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / folder / '*.csv'))] if Path(folder).is_absolute() is False and (HERE / folder).exists()
              else [pd.read_csv(f) for f in glob.glob(str(Path(folder) / '*.csv'))], ignore_index=True)
d = d[d.outer == 0].reset_index(drop=True) if OUTER == '0' else d
d['nk'] = d.n / d.k; d['type'] = d.source.str.split('_').str[0]
d['best'] = d[['A', 'B', 'C']].max(axis=1)


def lc(m):
    m1, m2 = (d.n // 2).astype(float), 0.8 * d.n
    b = (d['half_' + m] - d['cv5_' + m]) / (1 / m1 - 1 / m2)
    return d['cv5_' + m] + b * (1 / d.n - 1 / m2)


est = {'CV5': (d.cv5_A, d.cv5_C), 'CV10': (d.cv10_A, d.cv10_C),
       'BUR': (d.cv5_A + d.app_A - d.all5_A, d.cv5_C + d.app_C - d.all5_C), 'LC': (lc('A'), lc('C'))}
pick = {}
pick['always_C'] = np.full(len(d), 'C'); pick['always_A'] = np.full(len(d), 'A'); pick['always_B'] = np.full(len(d), 'B')
for t in [5, 7.5, 10, 12.5, 15, 20]:
    pick[f'NK{t}'] = np.where(d.nk >= t, 'C', 'A')
for name, (a, c) in est.items():
    pick[name] = np.where(a < c, 'A', 'C')
    for t in [5, 7.5, 10, 12.5, 15]:
        pick[f'H{t}_{name}'] = np.where(d.nk >= t, 'C', pick[name])
pick['CV5_ABC'] = np.array(['ABC'[i] for i in np.argmin(d[['cv5_A', 'cv5_B', 'cv5_C']].to_numpy(), axis=1)])
dd = d.cv5_A - d.cv5_C
for z in [1.0, 2.0]:
    for dflt in 'AC':
        unsure = dd.abs() <= z * d.cv5_se_AmC
        pick[f'TRI{z}_{dflt}'] = np.where(d.nk >= 10, 'C', np.where(unsure, dflt, np.where(dd > 0, 'C', 'A')))
        if z == 1.0 and dflt == 'A':
            cover = float(unsure[d.nk < 10].mean())
reg = {}
for k, ch in pick.items():
    d['r_' + k] = d.best - np.select([ch == 'A', ch == 'B', ch == 'C'], [d.A, d.B, d.C])
    reg[k] = float(d['r_' + k].mean())
src = sorted(d.source.unique()); G = {s: d[d.source == s] for s in src}


def boot(col, ref='r_H10_CV5', B=2000):
    """Source-level bootstrap of the difference in mean regret (cells pooled within the resampled sources); vectorised via per-source sums."""
    rng = np.random.default_rng(20263699)
    S = d.groupby('source')[[col, ref]].sum().loc[src].to_numpy(); N = d.groupby('source').size().loc[src].to_numpy()
    idx = np.stack([rng.choice(len(src), len(src)) for _ in range(B)])
    v = (S[idx, 0].sum(1) - S[idx, 1].sum(1)) / N[idx].sum(1)
    return [float(d[col].mean() - d[ref].mean()), float(np.quantile(v, .025)), float(np.quantile(v, .975))]


bytype = {k: {t: float(g['r_' + k].mean()) for t, g in d.groupby('type')} for k in pick}
rank = sorted(reg, key=reg.get)
cmp = {k: boot('r_' + k) for k in rank[:12] if k != 'H10_CV5'}
qual = [k for k, v in cmp.items() if v[0] <= -0.001 and v[2] < 0 and all(bytype[k][t] <= bytype['H10_CV5'][t] for t in bytype[k])]
out = dict(cells=len(d), sources=len(src), by_type_sources={t: int(g.source.nunique()) for t, g in d.groupby('type')},
           regret=dict(sorted(reg.items(), key=lambda kv: kv[1])), regret_by_type=bytype, vs_H10_CV5=cmp, qualifying=qual,
           unsure_share_low_nk_TRI1=cover,
           v3=(qual[0] if qual else 'H10_CV5'))
json.dump(out, open(HERE / f'{prefix}_summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'regret_by_type'}, indent=1, default=float))
print('by type (top 15):')
for k in rank[:15]:
    print(f'{k:16s} {reg[k]:.5f} ' + ' '.join(f'{t} {bytype[k][t]:.5f}' for t in sorted(bytype[k])))
