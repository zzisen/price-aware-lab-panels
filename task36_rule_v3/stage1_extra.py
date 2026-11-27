#!/usr/bin/env python3
"""Task 36 stage 1 descriptive extras on outer split 0: decomposition of H's residual regret, regret by n/k band, cell-level vs
averaged lead of C at n/k >= 10, and where H's choice in the CV zone is wrong. Output stage1_extra.json."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / 'stage1_results' / '*_o0.csv'))], ignore_index=True)
d['nk'] = d.n / d.k; d['type'] = d.source.str.split('_').str[0]; d['best'] = d[['A', 'B', 'C']].max(axis=1)
hp = np.where(d.nk >= 10, 'C', np.where(d.cv5_A < d.cv5_C, 'A', 'C'))
d['rH'] = d.best - np.where(hp == 'A', d.A, d.C)
d['rOracleAC'] = d.best - d[['A', 'C']].max(axis=1)       # regret left even with a perfect A-vs-C choice (B best)
d['rC'] = d.best - d.C
band = pd.cut(d.nk, [0, 5, 7.5, 10, 15, 50], right=False)
out = dict(mean_regret_H=float(d.rH.mean()), oracle_AC=float(d.rOracleAC.mean()), always_C=float(d.rC.mean()))
out['by_band'] = {str(b): dict(cells=int(len(g)), H=float(g.rH.mean()), always_C=float(g.rC.mean()), oracle_AC=float(g.rOracleAC.mean()),
                              cv_picks_A=float((g.cv5_A < g.cv5_C).mean()), A_truly_better=float((g.A > g.C).mean()),
                              cv_A_right=float(((g.cv5_A < g.cv5_C) & (g.A > g.C)).sum() / max(1, (g.cv5_A < g.cv5_C).sum())))
                  for b, g in d.groupby(band, observed=True)}
hi = d[d.nk >= 10]; gain = hi.C - hi.A
agg = hi.assign(g=gain).groupby(['source', 'task', 'p', 'n', 'k']).g.mean()
out['nk_ge10'] = dict(cell_C_leads=float((gain > 0).mean()), cell_A_leads_by_gt002=float((gain < -0.02).mean()),
                      setting_C_leads=float((agg > 0).mean()), setting_A_leads_by_gt002=float((agg < -0.02).mean()),
                      source_mean_gain_min=float(hi.assign(g=gain).groupby('source').g.mean().min()),
                      sources_negative=int((hi.assign(g=gain).groupby('source').g.mean() < 0).sum()))
out['H_minus_C_by_type'] = {t: float(g.rH.mean() - g.rC.mean()) for t, g in d.groupby('type')}
out['H_beats_C_sources'] = int((d.groupby('source').apply(lambda g: g.rH.mean() < g.rC.mean())).sum())
json.dump(out, open(HERE / 'stage1_extra.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
