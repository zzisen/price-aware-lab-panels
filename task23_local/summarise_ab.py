#!/usr/bin/env python3
"""Summarise the baseline-only recalibration run (local_*_ab.csv; 30 reps). Output: summary_ab.txt"""
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in sorted(HERE.glob('local_*_ab.csv'))], ignore_index=True)
d['kind'] = np.where(d.site.str.len() == 1, 'cycle', 'race')
w = d.pivot_table(index=['site', 'kind', 'panel', 'n', 'rep'], columns='method', values=['abs_log_OE', 'NB', 'OE', 'recal']).reset_index()
out = []
for kind in ['cycle', 'race']:
    rows = []
    for (p, n), g in w[w.kind == kind].groupby(['panel', 'n']):
        r = dict(panel=p, n=n, reps=len(g))
        for m in ['A import', 'B0 baseline only', 'B recalibrate', 'T test then recalibrate']:
            r[f'|logOE| {m[:2]}'] = g[('abs_log_OE', m)].mean()
        r['T recal share'] = g[('recal', 'T test then recalibrate')].mean()
        for m in ['B0 baseline only', 'B recalibrate', 'T test then recalibrate']:
            r[f'{m[:2]} better OE'] = (g[('abs_log_OE', m)] < g[('abs_log_OE', 'A import')]).mean()
            r[f'{m[:2]} dNB'] = (g[('NB', m)] - g[('NB', 'A import')]).mean()
        r['B0 vs B |logOE|'] = (g[('abs_log_OE', 'B0 baseline only')] - g[('abs_log_OE', 'B recalibrate')]).mean()
        rows.append(r)
    out.append(f'\n=== {kind} sites ===\n' + pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
out.append('\n=== by site, matched panel: |logOE| A, B, T; share recalibrated by T; T - A net benefit ===')
g = w[w.panel == 'matched']
t = pd.DataFrame({'A': g[('abs_log_OE', 'A import')], 'B': g[('abs_log_OE', 'B recalibrate')], 'T': g[('abs_log_OE', 'T test then recalibrate')],
                  'T recal': g[('recal', 'T test then recalibrate')], 'T dNB': g[('NB', 'T test then recalibrate')] - g[('NB', 'A import')], 'site': g.site, 'n': g.n})
out.append(t.groupby(['site', 'n']).mean().to_string(float_format=lambda v: f'{v:.4f}'))
txt = '\n'.join(out); (HERE / 'summary_ab.txt').write_text(txt, encoding='utf-8'); print(txt)
