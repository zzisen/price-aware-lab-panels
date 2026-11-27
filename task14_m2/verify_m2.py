"""Independent check of Task 14: recomputes the decisions from the raw part files, without using summarize_m2.py.
Also checks the integrity of the run: row counts, missing values, realised chemotype splits, cohort overlap."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
W = H.parent
sys.path[:0] = [str(H), str(W / 'task12_r5'), str(W / 'task5_validation')]
import run_m2
import dev_r5 as V

tasks = json.loads((H / 'tasks.json').read_text(encoding='utf-8'))
end = json.loads((H / 'summary' / 'endpoints.json').read_text(encoding='utf-8'))


def parts(folder):
    d = pd.concat([pd.read_csv(f).assign(part=f.stem) for f in sorted((H / folder / 'parts').glob('*.csv'))], ignore_index=True)
    d['n'] = d['n'].astype(str)
    return d


def ev_of(d):
    return 1 - d.sse.to_numpy().sum() / d.c0.to_numpy().sum()


nh, kc, kr = parts('results_nhanes'), parts('results_kinase_chemotype'), parts('results_kinase_random')

print('== integrity ==')
for label, d in [('nhanes', nh), ('kinase_chemotype', kc), ('kinase_random', kr)]:
    main = d[d.method.isin(['R5', 'decorr', 'random'])]
    print(label, 'rows', len(d), '| parts', d.part.nunique(), '| NaN in sse/c0 for main methods', int(main[['sse', 'c0']].isna().sum().sum()))
    reps = d[d.n != 'all'].groupby(['task', 'outer', 'n']).rep.nunique()
    print('  reps per task/outer/n: min', reps.min(), 'max', reps.max(), '| n values', sorted(d.n.unique()))
    cnt = main.groupby(['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k']).size()
    print('  duplicated main rows', int((cnt > 1).sum()))

print('\n== A recomputed ==')
for t in sorted(nh.task.unique()):
    for k, p in [(5, 'ridge'), (5, 'gbt'), (10, 'ridge'), (10, 'gbt')]:
        h = nh[(nh.task == t) & (nh.n == '100') & (nh.k == k) & (nh.predictor == p)]
        r = np.array([ev_of(h[(h.outer == o) & (h.method == 'R5')]) for o in range(5)])
        c = np.array([ev_of(h[(h.outer == o) & (h.method == 'decorr')]) for o in range(5)])
        s = end['A']['cohorts'][t]['cells'][f'k{k}_{p}']
        assert abs(r.mean() - s['R5_mean']) < 1e-4 and abs(c.mean() - s['decorr_mean']) < 1e-4 and int((r > c).sum()) == s['wins'], (t, k, p)
        print(f'{t} k{k} {p}: R5 {r.mean():.4f} decorr {c.mean():.4f} wins {(r > c).sum()}/5 | smallest split gain {(r - c).min():+.4f}')

print('\n== B recomputed ==')
hb = kc[kc.n == '100']
T = sorted(hb.task.unique())
tab = {}
for k in [5, 10, 30]:
    for p in ['ridge', 'gbt']:
        for m in ['R5', 'decorr', 'random']:
            h = hb[(hb.k == k) & (hb.predictor == p) & (hb.method == m)]
            tab[(m, p, k)] = np.array([ev_of(h[h.task == t]) for t in T])
for k, p in [(5, 'ridge'), (5, 'gbt'), (10, 'ridge'), (10, 'gbt')]:
    r, c = tab[('R5', p, k)], tab[('decorr', p, k)]
    s = end['B']['accuracy']['cells'][f'k{k}_{p}']
    assert abs(r.mean() - s['R5_mean']) < 1e-4 and abs(c.mean() - s['decorr_mean']) < 1e-4 and int((r > c).sum()) == s['wins'], (k, p)
    print(f'k{k} {p}: R5 {r.mean():.4f} decorr {c.mean():.4f} wins {(r > c).sum()}/6')
for p in ['ridge', 'gbt']:
    best = max(tab[('R5', p, 5)].mean(), tab[('R5', p, 10)].mean())
    dec = tab[('decorr', p, 30)].mean()
    rnd = tab[('random', p, 30)].mean()
    print(f'efficiency {p}: best R5 {best:.4f} / decorr k30 {dec:.4f} = {best / dec:.4f} | needed EV for 0.90 = {0.9 * dec:.4f} (short by {0.9 * dec - best:.4f}) | vs random k30 {best / rnd:.3f}')
    per = tab[('R5', p, 10)] / tab[('decorr', p, 30)]
    print('   R5 k10 / decorr k30 per task:', np.round(per, 3), '| tasks at or above 0.90:', int((per >= .9).sum()))

print('\n== kinase random split, report only ==')
hr = kr[kr.n == '100']
for p in ['ridge', 'gbt']:
    g = lambda m, k: np.mean([ev_of(hr[(hr.task == t) & (hr.predictor == p) & (hr.method == m) & (hr.k == k)]) for t in T])  # noqa: E731
    best = max(g('R5', 5), g('R5', 10))
    print(f'{p}: best R5 {best:.4f} / decorr k30 {g("decorr", 30):.4f} = {best / g("decorr", 30):.3f}')

print('\n== realised chemotype splits ==')
chem = pd.read_csv(H / 'kin_chemotypes.csv', dtype=str, index_col=0)
print('compounds', len(chem), '| chemotypes', chem.chemotype.nunique(), '| missing chemotype', int(chem.chemotype.isna().sum()))
shares, sizes, cross = [], [], 0
for t in [x for x in tasks if x['set'] == 'kinase']:
    m = pd.read_csv(H / 'matrices' / f"{t['task']}.csv", usecols=['unit'])
    u = m.unit.astype(str).to_numpy()
    for o in range(5):
        rng = np.random.default_rng([V.SEED, t['code'], o])
        te, trn = run_m2.chemotype_split(u, chem, rng)
        a, b = set(chem.loc[u[te]].chemotype), set(chem.loc[u[trn]].chemotype)
        cross += len(a & b)
        sizes.append(len(te))
        shares.append(len(te) / len(u))
        assert not set(te) & set(trn) and len(te) + len(trn) == len(u)
print('test compounds', min(sizes), '-', max(sizes), '| share', round(min(shares), 3), '-', round(max(shares), 3), '| chemotypes on both sides', cross)

print('\n== cohorts and tasks ==')
ids = {t['task']: set(pd.read_csv(H / 'matrices' / f"{t['task']}.csv", usecols=['unit']).unit.astype(str)) for t in tasks if t['set'] == 'nhanes'}
ids['M_665'] = set(pd.read_csv(W / 'task13_confirm' / 'matrices' / 'NHANES_I.csv', usecols=['unit']).unit.astype(float).astype(int).astype(str))
names = list(ids)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        print(f'overlap {a} {b}:', len(ids[a] & ids[b]))
print({k: len(v) for k, v in ids.items()})
for t in tasks:
    cols = pd.read_csv(H / 'matrices' / f"{t['task']}.csv", nrows=0).columns
    cand = {c[5:] for c in cols if c.startswith('cand_')}
    print(t['task'], 'candidates', len(cand), '| targets', len(t['targets']), '| targets among candidates', len(cand & set(t['targets'])), '| families', sorted(set(t['families'])))
kin = [set(t['targets']) for t in tasks if t['set'] == 'kinase']
print('distinct kinase targets over 6 tasks', len(set().union(*kin)))

print('\n== Task 13 grid for comparison (mean EV, Beat AML and first-stage NHANES) ==')
g = pd.read_csv(W / 'manuscript_r5' / 'assets' / 'ev_grid.csv')
print('sets', g.set.unique().tolist(), '| n', sorted(g.n.astype(str).unique()))
g = g[g.method.isin(['R5', 'decorr', 'random']) & g.predictor.isin(['ridge', 'gbt']) & g.k.isin([5, 10, 20, 30])]
for s, h in g.groupby('set'):
    print(f'-- {s}')
    print(h.pivot_table(index=['n', 'k'], columns=['predictor', 'method'], values='ev').round(3).to_string())
