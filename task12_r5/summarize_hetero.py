#!/usr/bin/env python3
"""Apply the pre-written criterion of plan_hetero.md to hetero_results/*.csv. Prints the per-cell table and verdict,
writes hetero_summary/criterion.csv and hetero_ev_by_task.csv. Run with .env312."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'hetero_summary'


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted((HERE / 'hetero_results').glob('*.csv'))])
    d['n'] = d['n'].astype(str)
    g = d.groupby(['task', 'n', 'k', 'method', 'predictor'])
    ev = g.apply(lambda h: 1 - h.sse.sum() / h.c0.sum(), include_groups=False).rename('ev').reset_index()
    ev.to_csv(OUT / 'hetero_ev_by_task.csv', index=False)
    nat = ev[ev.predictor == 'native'].pivot_table(index=['n', 'k', 'task'], columns='method', values='ev')
    nat['best_fixed_fine'] = nat[['R5fix_Family|group', 'R5fix_Family|union', 'R5fix_Exact|group', 'R5fix_Exact|union']].max(axis=1)
    rows = []
    for n in ['50', '100']:
        for k in [5, 10]:
            h = nat.loc[(n, k)]
            T = len(h); need = int(np.ceil(2 / 3 * T))
            diff = h['R5'] - h['R5fix_Shared|group']; up = h['best_fixed_fine'] - h['R5fix_Shared|group']
            rows.append({'n': n, 'k': k, 'tasks': T, 'needed': need, 'R5_wins_vs_Shared': int((diff > 0).sum()),
                         'R5_mean_gain': round(float(diff.mean()), 4),
                         'cell_pass': bool((diff > 0).sum() >= need and diff.mean() >= 0.01),
                         'ceiling_wins_vs_Shared': int((up > 0).sum()), 'ceiling_mean_gain': round(float(up.mean()), 4)})
    c = pd.DataFrame(rows); c.to_csv(OUT / 'criterion.csv', index=False)
    print(c.to_string(index=False))
    print('\nVERDICT: resolution works' if c.cell_pass.sum() >= 3 else '\nVERDICT: resolution does NOT work (criterion 1 fails)',
          f'({int(c.cell_pass.sum())}/4 cells pass, need 3)')
    print('\n== mean EV by method (native), all n/k ==')
    print(nat.groupby(level=['n', 'k']).mean().round(3).to_string())
    print('\n== panel x predictor (mean EV) ==')
    p = ev[ev.method.isin(['random', 'decorr', 'R5', 'R5_Shared']) & ev.predictor.isin(['ridge', 'gbt'])]
    print(p.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values='ev').round(3).to_string())
    print('\n== per task, R5 native minus Shared (n 50/100, k 5/10) ==')
    print((nat['R5'] - nat['R5fix_Shared|group']).loc[['50', '100']].unstack('task').loc[(slice(None), [5, 10]), :].round(3).to_string())


if __name__ == '__main__':
    main()
