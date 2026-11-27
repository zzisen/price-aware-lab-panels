#!/usr/bin/env python3
"""Task 11 summary (plan_transfer.md sections 5-6). Run with .env312 after run_task11.py.
Writes summary/task11_by_task.csv and summary/task11_endpoints.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OUT = HERE / 'summary'
METHODS = ['R4_transfer', 'R4_scratch', 'MTL_matched_scratch', 'MTL_matched_transfer', 'Ridge_scratch',
           'Ridge_scratch_plain', 'Full_transfer']
K = ['k_R4_transfer', 'k_R4_scratch', 'k_MTL_transfer', 'k_MTL_scratch', 'k_all']


def sign(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[np.abs(d) > 1e-12]
    w = int((d > 0).sum())
    return {'wins': w, 'losses': int((d < 0).sum()), 'p_two_sided': float(binomtest(w, len(d), 0.5).pvalue) if len(d) else None}


def main():
    OUT.mkdir(exist_ok=True)
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    rows = []
    for t in tasks:
        f = pd.read_csv(HERE / 'results' / f"{t['task']}.csv")
        for n_cal, h in f.groupby('n_cal'):
            assert len(h) == 20
            ev = {m: 1 - h[m].sum() / h.C0.sum() for m in METHODS}
            rows.append({'task': t['task'], 'train': t['train'], 'test': t['test'], 'n_cal': int(n_cal),
                         'n_test': int(h.n_test.iloc[0]), **{f'ev_{m}': v for m, v in ev.items()},
                         **{k: float(h[k].mean()) for k in K}, 'choice_R4_A': h.choice_R4_A.iloc[0],
                         'support': bool(ev['R4_transfer'] > 0 and ev['R4_transfer'] >= ev['R4_scratch'])})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'task11_by_task.csv', index=False)
    out = {}
    for n_cal, h in d.groupby('n_cal'):
        needed = int(np.ceil(2 / 3 * len(h)))
        out[f'n_cal_{n_cal}'] = {
            'tasks': len(h), 'tasks_supporting': int(h.support.sum()), 'needed': needed,
            'transfer_established': bool(h.support.sum() >= needed),
            'mean_ev': {m: float(h[f'ev_{m}'].mean()) for m in METHODS},
            'mean_k': {k: float(h[k].mean()) for k in K},
            'R4_transfer_vs_R4_scratch': sign(h.ev_R4_transfer, h.ev_R4_scratch),
            'R4_transfer_vs_MTL_matched_transfer': sign(h.ev_R4_transfer, h.ev_MTL_matched_transfer),
            'R4_transfer_vs_MTL_matched_scratch': sign(h.ev_R4_transfer, h.ev_MTL_matched_scratch),
            'MTL_matched_transfer_vs_MTL_matched_scratch': sign(h.ev_MTL_matched_transfer, h.ev_MTL_matched_scratch)}
    out['primary'] = 'n_cal_25 (plan section 5)'
    (OUT / 'task11_endpoints.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    cols = ['task', 'n_cal', 'ev_R4_transfer', 'ev_R4_scratch', 'ev_MTL_matched_transfer', 'ev_MTL_matched_scratch',
            'ev_Ridge_scratch', 'ev_Full_transfer', 'k_R4_transfer', 'k_R4_scratch', 'k_MTL_transfer', 'choice_R4_A', 'support']
    print(d[cols].round(3).to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
