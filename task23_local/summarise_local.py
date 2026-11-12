#!/usr/bin/env python3
"""Summarise Task 23 feasibility runs (training data only). Output: summary.txt"""
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in sorted(HERE.glob('local_*.csv')) if not f.stem.endswith('_ab')], ignore_index=True)
d['kind'] = np.where(d.site.str.len() == 1, 'cycle', 'race')
out = []
key = ['site', 'panel', 'n', 'rep']
w = d.pivot_table(index=key + ['kind'], columns='method', values=['C', 'abs_log_OE', 'NB']).reset_index()
M = ['A import', 'B recalibrate', 'C local only', 'D borrow strength']
for kind in ['cycle', 'race']:
    for metric in ['C', 'abs_log_OE', 'NB']:
        sub = w[w.kind == kind]
        rows = []
        for (p, n), g in sub.groupby(['panel', 'n']):
            r = dict(panel=p, n=n, reps=len(g))
            for m in M:
                r[m] = g[(metric, m)].mean()
            for m in M[1:]:
                diff = g[(metric, m)] - g[(metric, 'A import')]
                r[f'{m[0]}-A'] = diff.mean()
                r[f'{m[0]}>A'] = ((diff > 0) if metric != 'abs_log_OE' else (diff < 0)).mean()
            rows.append(r)
        t = pd.DataFrame(rows)
        out.append(f'\n=== {kind} sites, {metric} (mean over sites x reps; for abs_log_OE "better" = smaller) ===')
        out.append(t.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
# per site, D - A in C at n=1000 and 2000
out.append('\n=== D - A in C by site (matched / routine / target-aware), mean over reps ===')
s = w.copy(); s['DA'] = s[('C', 'D borrow strength')] - s[('C', 'A import')]
s.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in s.columns]
out.append(s.groupby(['site', 'panel', 'n']).DA.mean().unstack('n').to_string(float_format=lambda v: f'{v:+.4f}'))
s['CA'] = s['C_C local only'] - s['C_A import']
out.append('\n=== C(local only) - A in C by site ===')
out.append(s.groupby(['site', 'panel', 'n']).CA.mean().unstack('n').to_string(float_format=lambda v: f'{v:+.4f}'))
out.append('\n=== chosen penalty for D (share at each value) ===')
out.append(d[d.method == 'D borrow strength'].groupby('n').lam.value_counts(normalize=True).unstack().to_string(float_format=lambda v: f'{v:.2f}'))
txt = '\n'.join(out); (HERE / 'summary.txt').write_text(txt, encoding='utf-8'); print(txt)
