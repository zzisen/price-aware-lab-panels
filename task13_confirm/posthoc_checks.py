#!/usr/bin/env python3
"""Task 13, run after summarize_confirm.py. POST HOC, descriptive only; nothing here enters P, M or the journal order.
Integrity checks on results/, per-task margins, measurements actually used, S2 per task, and the top-5 hit rate of a
reference the plan did not list: rank drugs by their training mean, no patient measurement at all (same splits as job).
Run with .env312:  python posthoc_checks.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261103


def ev(h):
    return 1 - h.sse.sum() / h.c0.sum()


tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
sets = {t['task']: t['set'] for t in tasks}
d = pd.concat([pd.read_csv(f) for f in sorted((HERE / 'results').glob('*.csv'))])
d['n'] = d['n'].astype(str); d['set'] = d.task.map(sets)

print('== integrity ==')
u = d.drop_duplicates(['task', 'outer', 'n', 'rep']).groupby(['set', 'task', 'n']).size().unstack()
print(u.to_string())
print('top5_hit non-null by set:', d.groupby('set').top5_hit.apply(lambda s: int(s.notna().sum())).to_dict())
print('Exact rows by set:', d[d.method.str.contains('Exact')].groupby('set').size().to_dict())
print('R4 rows, skipped_D3 / real, by set:', d[d.method == 'R4'].groupby('set').option.apply(
    lambda s: (int((s == 'skipped_D3').sum()), int((s != 'skipped_D3').sum()))).to_dict())
print('NaN sse outside skipped R4:', int(d[d.option.ne('skipped_D3')].sse.isna().sum()))

d = d[d.option.ne('skipped_D3')].copy()
g = d.groupby(['set', 'task', 'n', 'k', 'method', 'predictor'])
t = pd.concat([g.apply(ev, include_groups=False).rename('ev'), g.k_used.mean()], axis=1).reset_index()

print('\n== R5 minus decorr, per task, n = 100 (main) ==')
h = t[(t.set == 'main') & (t.n == '100') & t.method.isin(['R5', 'decorr', 'random']) & t.predictor.isin(['ridge', 'gbt'])]
w = h.pivot_table(index=['predictor', 'k', 'task'], columns='method', values='ev')
w['gain'] = w.R5 - w.decorr; w['gain_vs_random'] = w.R5 - w.random
print(w.groupby(level=['predictor', 'k'])[['gain', 'gain_vs_random']].agg(['mean', 'min', 'max']).round(3).to_string())
print('wins vs random:', (w.gain_vs_random > 0).groupby(level=['predictor', 'k']).sum().to_dict())

print('\n== measurements actually used by the R5 panel (budget k) ==')
h = t[(t.method == 'R5') & (t.predictor == 'ridge')]
print(h.pivot_table(index=['set', 'n'], columns='k', values='k_used').round(1).to_string())

print('\n== option chosen by the middle layer, n = 100, share of rows ==')
h = d[(d.method == 'R5') & (d.n == '100') & d.k.isin([5, 10])]
print(h.groupby(['set', 'k']).option.value_counts(normalize=True).round(2).unstack(fill_value=0).to_string())

print('\n== S2 per task, n = 100: R5 minus Shared, and within - between family correlation ==')
rows = []
for tk in tasks:
    if tk['set'] == 's1':
        continue
    y = pd.read_csv(HERE / 'matrices' / f"{tk['task']}.csv")[tk['targets']]
    c = y.corr().to_numpy(); f = np.asarray(tk['families'])
    same = f[:, None] == f[None, :]; off = ~np.eye(len(f), dtype=bool)
    x = t[(t.task == tk['task']) & (t.n == '100') & (t.predictor == 'native')]
    r = {'task': tk['task'], 'gap': round(float(c[same & off].mean() - c[~same].mean()), 3)}
    for k in (5, 10):
        v = x[x.k == k].set_index('method').ev
        r[f'R5_gain_k{k}'] = round(float(v['R5'] - v['R5fix_Shared|group']), 4)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print('\n== S1 top-5 hit of the training-mean ranking (no patient measurement), mean over 3 tasks ==')
out = {}
for tk in [t_ for t_ in tasks if t_['set'] == 's1']:
    m = pd.read_csv(HERE / 'matrices' / f"{tk['task']}.csv")
    y = m[tk['targets']].to_numpy(float)
    for o in range(5):
        perm = np.random.default_rng([SEED, tk['code'], o]).permutation(len(y))
        te, trn = perm[: len(y) // 5], perm[len(y) // 5:]
        lo_t = np.argsort(y[te], axis=1, kind='stable')[:, :5]
        for n in [100, None]:
            for r in (range(4) if n else [0]):
                sub = trn if n is None else np.random.default_rng([SEED, tk['code'], o, n, r]).choice(trn, n, replace=False)
                lo_p = set(np.argsort(y[sub].mean(0), kind='stable')[:5])
                out.setdefault(n or 'all', []).append(np.mean([len(lo_p & set(a)) / 5 for a in lo_t]))
print({k: round(float(np.mean(v)), 3) for k, v in out.items()})
