"""Descriptive (not prespecified): the K1-K3 checks applied to R1 instead of R2, after R2 failed its criteria."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
s = pd.read_csv(HERE / 'rules_summary.csv'); c = pd.read_csv(HERE / 'rules_contrasts.csv'); f = pd.read_csv(HERE / 'rules_choice_frequencies.csv')
out = {}
for an, ss in s.groupby('analysis', sort=False):
    fixed = [p for p in ['Shared', 'Family', 'Exact≤3', 'Exact'] if ss[p].notna().all()]
    labels = ss.n_label.tolist()
    fr = f[(f.analysis == an) & (f.rule == 'R1')].set_index('n_label').reindex(labels)[fixed if 'Exact≤3' not in fixed else ['Shared', 'Family', 'Global3', 'Exact']].fillna(0)
    modal = fr.idxmax(axis=1); sh = fr['Shared']
    ce = c[(c.analysis == an) & (c.contrast == 'R1 − Exact')].set_index('n_label').reindex(labels)
    r1 = ss['R1'] if 'regret_R1_paired' not in ss or ss.regret_R1_paired.isna().all() else None
    reg_r1 = ss.regret_R1_paired if r1 is None else ss.regret_R1
    out[an] = {'K1 (R1)': bool(modal.iloc[0] == 'Shared' and modal.iloc[-1] != 'Shared' and all(np.diff(sh.values) <= 1e-12)),
               'K2 (R1)': bool(ce['rep_97.5'].iloc[0] < 0 and (ss.R1 <= ss.Exact + 1e-12).all()),
               'K3 (R1)': bool(all(reg_r1.max() < ss[f'regret_{p}'].max() for p in fixed)),
               'R1 modal by n': dict(zip(labels, modal)), 'R1 max regret': round(float(reg_r1.max()), 4),
               'best fixed max regret': round(min(float(ss[f'regret_{p}'].max()) for p in fixed), 4)}
(HERE / 'r1_descriptive.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False))
