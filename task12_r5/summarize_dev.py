#!/usr/bin/env python3
"""Summarise dev_results/*.csv (development data). Pooled EV = 1 - sum SSE / sum C0 per (task, n, k, method,
predictor); then mean over tasks and task win counts against decorr. Writes dev_summary/*.csv and prints tables."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'dev_summary'


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted((HERE / 'dev_results').glob('*.csv'))])
    d['n'] = d['n'].astype(str)
    g = d.groupby(['task', 'n', 'k', 'method', 'predictor'])
    ev = g.apply(lambda h: 1 - h.sse.sum() / h.c0.sum(), include_groups=False).rename('ev')
    ev = pd.concat([ev, g.k_used.mean(), g.pooled_r.mean(), g.target_r.mean()], axis=1).reset_index()
    ev.to_csv(OUT / 'dev_ev_by_task.csv', index=False)
    order = ['25', '50', '100', 'all']
    tasks = ev.task.nunique()

    print(f'== mean EV over {tasks} tasks: panel x predictor ==')
    p = ev[ev.method.isin(['random', 'decorr', 'R5', 'R5_Shared'])]
    tab = p.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values='ev').reindex(order, level=0)
    print(tab.round(3).to_string())

    print('\n== R5 native vs fixed resolutions (mean EV) ==')
    q = ev[ev.method.str.startswith('R5') & (ev.predictor == 'native')]
    print(q.pivot_table(index=['n', 'k'], columns='method', values='ev').reindex(order, level=0).round(3).to_string())

    print('\n== wins over decorr (tasks out of %d) ==' % tasks)
    rows = []
    for (n, k), h in ev[ev.k.isin([3, 5, 10, 20, 30])].groupby(['n', 'k']):
        w = h.pivot_table(index='task', columns=['method', 'predictor'], values='ev')
        r = {'n': n, 'k': k}
        for pr in ['ridge', 'gbt']:
            r[f'R5panel>decorr_{pr}'] = int((w[('R5', pr)] > w[('decorr', pr)]).sum())
            r[f'R5panel>random_{pr}'] = int((w[('R5', pr)] > w[('random', pr)]).sum())
        r['R5native>decorr_ridge'] = int((w[('R5', 'native')] > w[('decorr', 'ridge')]).sum())
        r['R5native>R5fixShared'] = int((w[('R5', 'native')] > w[('R5fix_Shared|group', 'native')]).sum())
        rows.append(r)
    wins = pd.DataFrame(rows).set_index(['n', 'k']).reindex(order, level=0)
    wins.to_csv(OUT / 'dev_wins.csv'); print(wins.to_string())

    print('\n== references (mean EV) ==')
    ref = ev[ev.method.isin(['all', 'R4'])].groupby(['n', 'method', 'predictor'])[['ev', 'k_used']].mean()
    print(ref.round(3).to_string())

    print('\n== 90% criterion: best R5 native/panel EV at k<=10 vs decorr at k=30 (mean over tasks) ==')
    for pr in ['ridge', 'gbt']:
        for n in order:
            h = ev[ev.n == n]
            dec30 = h[(h.method == 'decorr') & (h.predictor == pr) & (h.k == 30)].ev.mean()
            r5 = max(h[(h.method == 'R5') & (h.predictor == pr) & (h.k == k)].ev.mean() for k in (5, 10))
            print(pr, n, 'R5 k<=10', round(r5, 3), 'decorr k30', round(dec30, 3), 'ratio', round(r5 / dec30, 2))

    print('\n== R5 chosen option counts ==')
    print(d[(d.method == 'R5')].groupby(['n', 'k']).option.value_counts().unstack(fill_value=0).reindex(order, level=0).to_string())

    print('\n== pooled vs per-target Pearson (mean over tasks, k=10) ==')
    print(ev[ev.k == 10].pivot_table(index=['n'], columns=['predictor', 'method'], values=['pooled_r', 'target_r'])
          .reindex(order).round(2).to_string())


if __name__ == '__main__':
    main()
