#!/usr/bin/env python3
"""Task 13 summary: applies plan_r5_confirm FROZEN 20261101T2045 sections 4 (P1, P2, M, journal order) and 5 (S1-S4).
EV of a block = 1 - sum SSE / sum C0 over the rows in the block. Writes summary/*.csv and summary/endpoints.json.
Run with .env312 after run_confirm.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'summary'
CELLS = [(5, 'ridge'), (5, 'gbt'), (10, 'ridge'), (10, 'gbt')]


def ev(h):
    return 1 - h.sse.sum() / h.c0.sum()


def cell_rule(per_unit, wins_needed):
    """per_unit: DataFrame indexed by unit (task or outer split), columns R5 / decorr. Returns the cell record."""
    m5, md = per_unit['R5'].mean(), per_unit['decorr'].mean()
    w = int((per_unit['R5'] > per_unit['decorr']).sum())
    return {'R5_mean': round(float(m5), 4), 'decorr_mean': round(float(md), 4), 'wins': w, 'of': len(per_unit),
            'good': bool(m5 > md and w >= wins_needed), 'not_worse': bool(m5 >= md)}


def rule(cells):
    good = sum(c['good'] for c in cells.values())
    rest_ok = all(c['not_worse'] for c in cells.values() if not c['good'])
    return {'cells_good': good, 'rest_not_worse': rest_ok, 'pass': bool(good >= 3 and rest_ok)}


def efficiency(tab):
    """tab: EV by (unit, k, predictor) for method R5 / decorr, averaged over units. P2 per predictor."""
    out = {}
    for p in ['ridge', 'gbt']:
        r5 = max(tab[('R5', p, k)] for k in (5, 10)); dec = tab[('decorr', p, 30)]
        ok = r5 >= 0.9 * dec if dec > 0 else r5 >= dec
        out[p] = {'R5_best_k5_k10': round(float(r5), 4), 'decorr_k30': round(float(dec), 4),
                  'ratio': round(float(r5 / dec), 3) if dec > 0 else None, 'pass': bool(ok)}
    out['pass'] = bool(out['ridge']['pass'] and out['gbt']['pass'])
    return out


def main():
    OUT.mkdir(exist_ok=True)
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    sets = {t['task']: t['set'] for t in tasks}
    d = pd.concat([pd.read_csv(f) for f in sorted((HERE / 'results').glob('*.csv'))])
    d = d[d.option.ne('skipped_D3')].copy()                    # DEVIATIONS D3: no R4 reference on S1
    d['n'] = d['n'].astype(str); d['set'] = d.task.map(sets)
    end = {}

    # ---- P: BeatAML main, n = 100, unit = task ----
    m = d[(d.set == 'main') & (d.n == '100')]
    by_task = m.groupby(['task', 'method', 'predictor', 'k']).apply(ev, include_groups=False)
    cells = {}
    for k, p in CELLS:
        u = pd.DataFrame({meth: by_task.xs((meth, p, k), level=['method', 'predictor', 'k']) for meth in ['R5', 'decorr']})
        cells[f'k{k}_{p}'] = cell_rule(u, 4)
    P1 = {'cells': cells, **rule(cells)}
    P2 = efficiency(by_task.groupby(level=['method', 'predictor', 'k']).mean())
    end['P1'], end['P2'] = P1, P2
    end['P_pass'] = bool(P1['pass'] and P2['pass'])

    # ---- M: NHANES, n = 100, unit = outer split ----
    nh = d[(d.set == 'nhanes') & (d.n == '100')]
    by_split = nh.groupby(['outer', 'method', 'predictor', 'k']).apply(ev, include_groups=False)
    mcells = {}
    for k, p in CELLS:
        u = pd.DataFrame({meth: by_split.xs((meth, p, k), level=['method', 'predictor', 'k']) for meth in ['R5', 'decorr']})
        mcells[f'k{k}_{p}'] = cell_rule(u, 4)
    M1 = {'cells': mcells, **rule(mcells)}
    M2 = efficiency(by_split.groupby(level=['method', 'predictor', 'k']).mean())
    end['M_cells'], end['M_efficiency'] = M1, M2
    end['M_pass'] = bool(M1['pass'] and M2['pass'])

    end['journal_first'] = ('Patterns' if end['P_pass'] and end['M_pass'] else
                            'npj Precision Oncology' if end['P_pass'] else
                            'no superiority claim; boundary-study positioning (PLOS Comp Biol / Bioinformatics)')
    (OUT / 'endpoints.json').write_text(json.dumps(end, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(end, indent=1, ensure_ascii=False))

    # ---- descriptive tables: every n, k ----
    g = d.groupby(['set', 'task', 'n', 'k', 'method', 'predictor'])
    t = pd.concat([g.apply(ev, include_groups=False).rename('ev'), g.k_used.mean(), g.pooled_r.mean(), g.target_r.mean()]
                  + ([g.top5_hit.mean()] if 'top5_hit' in d else []), axis=1).reset_index()
    t.to_csv(OUT / 'ev_by_task.csv', index=False)
    order = ['25', '50', '100', '500', 'all']
    for s in ['main', 'nhanes', 's1']:
        h = t[(t.set == s) & t.method.isin(['random', 'decorr', 'R5', 'R5_Shared'])]
        tab = h.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values='ev')
        print(f'\n== {s}: mean EV, panel x predictor ==')
        print(tab.reindex([o for o in order if o in tab.index.get_level_values(0)], level=0).round(3).to_string())

    # S1: advantage and top-5 hit rate
    s1 = t[(t.set == 's1') & t.n.isin(['100', 'all']) & t.k.isin([5, 10, 20]) & t.method.isin(['R5', 'decorr', 'random'])]
    print('\n== S1 (40 targets): mean EV and top-5 hit ==')
    print(s1.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values=['ev', 'top5_hit']).round(3).to_string())

    # S2: resolution, R5 native vs Shared-only, plus between-family correlation of targets
    rows = []
    for s in ['main', 'nhanes']:
        h = t[(t.set == s) & (t.n == '100') & t.k.isin([5, 10]) & (t.predictor == 'native') & t.method.str.startswith('R5')]
        w = h.pivot_table(index=['k', 'task'], columns='method', values='ev')
        for k, x in w.groupby(level='k'):
            diff = x['R5'] - x['R5fix_Shared|group']
            fine = x[[c for c in x if c.startswith('R5fix_') and 'Shared' not in c]].max(axis=1) - x['R5fix_Shared|group']
            rows.append({'set': s, 'k': k, 'R5_wins_vs_Shared': int((diff > 0).sum()), 'of': len(diff),
                         'R5_mean_gain': round(float(diff.mean()), 4), 'best_fine_mean_gain': round(float(fine.mean()), 4)})
    print('\n== S2 resolution (n = 100) =='); print(pd.DataFrame(rows).to_string(index=False))
    corr = []
    for tk in tasks:
        if tk['set'] == 's1':
            continue
        y = pd.read_csv(HERE / 'matrices' / f"{tk['task']}.csv")[tk['targets']]
        c = y.corr().to_numpy(); f = np.asarray(tk['families'])
        same = f[:, None] == f[None, :]; off = ~np.eye(len(f), dtype=bool)
        corr.append({'task': tk['task'], 'within_r': round(float(c[same & off].mean()), 3), 'between_r': round(float(c[~same].mean()), 3)})
    print(pd.DataFrame(corr).to_string(index=False))

    # S3 / S4
    print('\n== S3 pooled vs per-target Pearson, k = 10 (mean over tasks) ==')
    h = t[(t.k == 10) & t.method.isin(['R5', 'decorr']) & t.predictor.isin(['ridge', 'gbt'])]
    print(h.pivot_table(index=['set', 'n'], columns=['predictor', 'method'], values=['pooled_r', 'target_r']).round(2).to_string())
    print('\n== S4 references: R4 (native) and all candidates ==')
    r = d[d.method.isin(['R4', 'all'])].groupby(['set', 'task', 'n', 'method', 'predictor'])   # R4's k varies: pool over it
    ref = pd.concat([r.apply(ev, include_groups=False).rename('ev'), r.k_used.mean()], axis=1)
    ref = ref.groupby(level=['set', 'n', 'method', 'predictor']).mean()
    print(ref.round(3).to_string())


if __name__ == '__main__':
    main()
