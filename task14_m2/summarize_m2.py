#!/usr/bin/env python3
"""Task 14 summary: applies plan_r5_m2 FROZEN sections 3, 4 and 5 (A, B, journal order) and reports section 6.
EV of a block = 1 - sum SSE / sum C0. NHANES: EV per outer split within a cohort, then the mean over splits. Kinase: EV per
task pooled over splits and repetitions, then the mean over tasks. Writes summary/*.csv and summary/endpoints.json.
Run with .env312 after run_m2.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'summary'
CELLS = [(5, 'ridge'), (5, 'gbt'), (10, 'ridge'), (10, 'gbt')]
KS = [3, 5, 10, 20, 30]


def ev(h):
    return 1 - h.sse.sum() / h.c0.sum()


def load(folder):
    d = pd.concat([pd.read_csv(f) for f in sorted((HERE / folder).glob('*.csv'))])
    d['n'] = d['n'].astype(str)
    return d


def cell_rule(per_unit, wins_needed):
    m5, md = per_unit['R5'].mean(), per_unit['decorr'].mean()
    w = int((per_unit['R5'] > per_unit['decorr']).sum())
    return {'R5_mean': round(float(m5), 4), 'decorr_mean': round(float(md), 4), 'wins': w, 'of': len(per_unit),
            'good': bool(m5 > md and w >= wins_needed), 'not_worse': bool(m5 >= md), 'positive': bool(m5 > 0)}


def rule(cells):
    good = sum(c['good'] for c in cells.values())
    rest_ok = all(c['not_worse'] for c in cells.values() if not c['good'])
    positive = all(c['positive'] for c in cells.values())
    return {'cells_good': good, 'rest_not_worse': rest_ok, 'all_positive': positive, 'pass': bool(good >= 3 and rest_ok and positive)}


def cells_of(by_unit, wins_needed):
    out = {}
    for k, p in CELLS:
        u = pd.DataFrame({m: by_unit.xs((m, p, k), level=['method', 'predictor', 'k']) for m in ['R5', 'decorr']})
        out[f'k{k}_{p}'] = cell_rule(u, wins_needed)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    end = {}

    # ---- A: each NHANES cohort, n = 100, outer split as unit (not judged on efficiency) ----
    nh = load('results_nhanes')
    a = {}
    for t, h in nh[nh.n == '100'].groupby('task'):
        by = h.groupby(['outer', 'method', 'predictor', 'k']).apply(ev, include_groups=False)
        cells = cells_of(by, 4)
        tab = by.groupby(level=['method', 'predictor', 'k']).mean()
        eff = {p: {'R5_best_k5_k10': round(float(max(tab[('R5', p, 5)], tab[('R5', p, 10)])), 4),
                   'decorr_k30_is_22': round(float(tab[('decorr', p, 30)]), 4), 'random_k30': round(float(tab[('random', p, 30)]), 4)} for p in ['ridge', 'gbt']}
        a[t] = {'cells': cells, **rule(cells), 'efficiency_reported_only': eff}
    n_pass = sum(c['pass'] for c in a.values())
    end['A'] = {'cohorts': a, 'cohorts_passed': n_pass, 'of': len(a), 'pass': bool(n_pass >= 2)}

    # ---- B: kinase, chemotype split, n = 100, task as unit ----
    kc = load('results_kinase_chemotype'); kr = load('results_kinase_random')
    h = kc[kc.n == '100']
    by = h.groupby(['task', 'method', 'predictor', 'k']).apply(ev, include_groups=False)
    cells = cells_of(by, 4)
    tab = by.groupby(level=['method', 'predictor', 'k']).mean()
    eff = {}
    for p in ['ridge', 'gbt']:
        r5 = max(tab[('R5', p, k)] for k in (5, 10)); dec = tab[('decorr', p, 30)]
        eff[p] = {'R5_best_k5_k10': round(float(r5), 4), 'decorr_k30': round(float(dec), 4), 'ratio': round(float(r5 / dec), 3) if dec > 0 else None,
                  'pass': bool(r5 >= .9 * dec if dec > 0 else r5 >= dec)}
    eff['pass'] = bool(eff['ridge']['pass'] and eff['gbt']['pass'])
    b1 = {'cells': cells, **rule(cells)}
    end['B'] = {'accuracy': b1, 'efficiency': eff, 'pass': bool(b1['pass'] and eff['pass'])}
    end['journal_first'] = 'Patterns' if end['A']['pass'] and end['B']['pass'] else 'npj Precision Oncology'
    (OUT / 'endpoints.json').write_text(json.dumps(end, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(end, indent=1, ensure_ascii=False))

    # ---- descriptive grids ----
    rows = []
    for label, d, unit in [('nhanes', nh, 'outer'), ('kinase_chemotype', kc, 'task'), ('kinase_random', kr, 'task')]:
        keys = ['task'] if unit == 'outer' else []
        for key, h in (d.groupby(keys) if keys else [(('all',), d)]):
            g = h.groupby([unit, 'n', 'k', 'method', 'predictor'])
            e = pd.concat([g.apply(ev, include_groups=False).rename('ev'), g.k_used.mean(), g.pooled_r.mean(), g.target_r.mean()], axis=1)
            m = e.groupby(level=['n', 'k', 'method', 'predictor']).mean().reset_index()
            m.insert(0, 'block', key[0] if keys else label); m.insert(0, 'set', label)
            rows.append(m)
    t = pd.concat(rows, ignore_index=True)
    t.to_csv(OUT / 'ev_grid.csv', index=False)
    order = ['25', '50', '100', '500', 'all']
    for (s, b), h in t[t.method.isin(['random', 'decorr', 'R5', 'R5_Shared']) & t.predictor.isin(['ridge', 'gbt'])].groupby(['set', 'block']):
        tab = h.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values='ev')
        print(f'\n== {s} / {b}: mean EV ==')
        print(tab.reindex([o for o in order if o in tab.index.get_level_values(0)], level=0).round(3).to_string())

    print('\n== measurements actually used, ridge rows: R5 and decorr ==')
    h = t[t.method.isin(['R5', 'decorr']) & (t.predictor == 'ridge') & t.k.isin(KS)]
    print(h.pivot_table(index=['set', 'block', 'method', 'n'], columns='k', values='k_used').round(1).to_string())

    # per-task gains, kinase
    print('\n== kinase, chemotype split, n = 100: R5 minus decorr per task ==')
    w = kc[(kc.n == '100') & kc.method.isin(['R5', 'decorr', 'random']) & kc.predictor.isin(['ridge', 'gbt'])].groupby(['predictor', 'k', 'task', 'method']).apply(ev, include_groups=False).unstack('method')
    w['gain'] = w.R5 - w.decorr; w['gain_vs_random'] = w.R5 - w.random
    print(w.groupby(level=['predictor', 'k'])[['gain', 'gain_vs_random']].agg(['mean', 'min', 'max']).round(3).to_string())

    # target resolution
    res = []
    for label, d, unit in [('nhanes', nh, 'outer'), ('kinase_chemotype', kc, 'task')]:
        h = d[(d.n == '100') & d.k.isin([5, 10]) & (d.predictor == 'native') & d.method.str.startswith('R5')]
        for key, x in (h.groupby('task') if unit == 'outer' else [('all', h)]):
            e = x.groupby([unit, 'k', 'method']).apply(ev, include_groups=False).unstack('method')
            diff = e['R5'] - e['R5fix_Shared|group']
            for k, v in diff.groupby(level='k'):
                res.append({'set': label, 'block': key, 'k': k, 'R5_wins_vs_Shared': int((v > 0).sum()), 'of': len(v), 'mean_gain': round(float(v.mean()), 4)})
    print('\n== target resolution, n = 100 =='); print(pd.DataFrame(res).to_string(index=False))

    # correlations, references
    print('\n== pooled vs per-target Pearson, k = 10, n = 100 ==')
    h = t[(t.k == 10) & (t.n == '100') & t.method.isin(['R5', 'decorr']) & t.predictor.isin(['ridge', 'gbt'])]
    print(h.pivot_table(index=['set', 'block', 'predictor'], columns='method', values=['pooled_r', 'target_r']).round(2).to_string())
    print('\n== references: R4 and all candidates, n = 100 and largest ==')
    for label, d, unit in [('nhanes', nh, 'outer'), ('kinase_chemotype', kc, 'task')]:
        r = d[d.method.isin(['R4', 'all'])]
        keys = (['task'] if unit == 'outer' else []) + [unit, 'n', 'method', 'predictor']
        g = r.groupby(keys)
        e = pd.concat([g.apply(ev, include_groups=False).rename('ev'), g.k_used.mean()], axis=1)
        lv = (['task'] if unit == 'outer' else []) + ['n', 'method', 'predictor']
        print(label); print(e.groupby(level=lv).mean().round(3).to_string())

    # chemotype test-set sizes and option choices
    opt = pd.concat([nh.assign(set='nhanes'), kc.assign(set='kinase_chemotype')])
    opt = opt[(opt.method == 'R5') & (opt.n == '100') & opt.k.isin([5, 10])]
    print('\n== option chosen, n = 100 ==')
    print(opt.groupby(['set', 'k']).option.value_counts(normalize=True).round(2).unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
