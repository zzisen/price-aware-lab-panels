#!/usr/bin/env python3
"""Route B: where do target-aware panels pay off? Descriptive synthesis of finished results; nothing is refitted.
Reads manuscript_r5/assets (ev_units.csv, m2_units.csv), task12_r5/dev_summary. All of this is post hoc.
Kinds of data: Beat AML (9 targets), NHANES (mean of four cohorts), PKIS2 (chemotype splits). 40-target Beat AML and
PKIS2 random splits are kept apart. The 0.02 margin is the one prespecified for the 40-target tasks in the first stage.
Writes summary/*.csv and summary/route_b.json. Run with .env312:  python route_b.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
A = W / 'manuscript_r5' / 'assets'
OUT = HERE / 'summary'
OUT.mkdir(exist_ok=True)
KS = [3, 5, 10, 20, 30]
PS = ['ridge', 'gbt']
MARGIN = .02
COHORTS = ['M', 'NH_A1_I', 'NH_A2_H', 'NH_A3_G']
KINDS = ['Beat AML', 'NHANES', 'PKIS2']


def load():
    u1 = pd.read_csv(A / 'ev_units.csv').astype({'n': str, 'unit': str})
    u2 = pd.read_csv(A / 'm2_units.csv').astype({'n': str, 'unit': str})
    src = {'main': 'Beat AML', 's1': 'Beat AML 40 targets', 'nhanes': 'M'}
    u1['source'] = u1.set.map(src)
    u2['source'] = np.where(u2.set == 'nhanes', u2.block, np.where(u2.set == 'kinase_chemotype', 'PKIS2', 'PKIS2 random'))
    c = ['source', 'unit', 'n', 'k', 'method', 'predictor', 'ev', 'k_used']
    return pd.concat([u1[c], u2[c]], ignore_index=True)


def train_sizes():
    """Numeric n for the largest row of each source. Beat AML: 178. NHANES: 500. PKIS2 'all': everything outside the test set."""
    return {'Beat AML': 178, 'Beat AML 40 targets': 178, 'NHANES': 500, 'M': 500, 'NH_A1_I': 500, 'NH_A2_H': 500, 'NH_A3_G': 500,
            'PKIS2': 645 - 146, 'PKIS2 random': 645 - 129}


def mean_table(u):
    """Mean EV per source, n, k, method, predictor, with a NHANES row that averages the four cohorts."""
    t = u.groupby(['source', 'n', 'k', 'method', 'predictor'])[['ev', 'k_used']].mean().reset_index()
    h = t[t.source.isin(COHORTS)].groupby(['n', 'k', 'method', 'predictor'])[['ev', 'k_used']].mean().reset_index()
    h.insert(0, 'source', 'NHANES')
    return pd.concat([t, h], ignore_index=True)


def label_n(s, n):
    return 'max' if n in ('all', '500') else n


def cells(t):
    """One row per source, predictor, n, k: R5, decorrelation, random, the better reusable panel, and the class of the cell."""
    w = t[t.method.isin(['R5', 'decorr', 'random']) & t.predictor.isin(PS) & t.k.isin(KS)]
    p = w.pivot_table(index=['source', 'predictor', 'n', 'k'], columns='method', values='ev').reset_index()
    p['best_reusable'] = p[['decorr', 'random']].max(axis=1)
    p['gain_decorr'] = p.R5 - p.decorr
    p['gain_best'] = p.R5 - p.best_reusable
    sizes = train_sizes()
    p['n_num'] = [sizes[s] if n in ('all', '500') else int(n) for s, n in zip(p.source, p.n)]
    p['n_per_k'] = p.n_num / p.k
    p['n_label'] = [label_n(s, n) for s, n in zip(p.source, p.n)]

    def cls(r):
        if max(r.R5, r.best_reusable) <= 0:
            return 'no signal'
        if r.gain_best >= MARGIN:
            return 'target-aware'
        if r.gain_best <= -MARGIN:
            return 'reusable'
        return 'similar'
    p['class'] = p.apply(cls, axis=1)
    return p


def equivalent_size(t, source, p, n, k):
    """Size of the better reusable panel that matches R5 at budget k, by interpolation on log k. Descriptive."""
    g = t[(t.source == source) & (t.n == n) & (t.predictor == p)].set_index(['method', 'k']).ev
    target = g[('R5', k)]
    curve = np.array([max(g[('decorr', q)], g[('random', q)]) for q in KS])
    curve = np.maximum.accumulate(curve)
    if target >= curve[-1]:
        return '>30', target
    if target <= curve[0]:
        return '<3', target
    i = int(np.searchsorted(curve, target))
    lo, hi = curve[i - 1], curve[i]
    x = np.log(KS[i - 1]) + (target - lo) / (hi - lo) * (np.log(KS[i]) - np.log(KS[i - 1]))
    return f'{np.exp(x):.0f}', target


def main():
    u = load()
    t = mean_table(u)
    c = cells(t)
    c.to_csv(OUT / 'cells.csv', index=False)
    res = {}
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 40)

    print('== 1. Map: class of each cell against the better reusable panel, margin 0.02 ==')
    order = ['25', '50', '100', 'max']
    for s in KINDS + ['Beat AML 40 targets', 'PKIS2 random']:
        for p in PS:
            h = c[(c.source == s) & (c.predictor == p)]
            print(f'\n{s}, {p}: gain over the better reusable panel')
            print(h.pivot_table(index='n_label', columns='k', values='gain_best').reindex(order).round(3).to_string())
            print(h.pivot_table(index='n_label', columns='k', values='class', aggfunc='first').reindex(order).to_string())

    print('\n== 2. Agreement across the three kinds of data ==')
    k3 = c[c.source.isin(KINDS)]
    rows = []
    for (p, n, k), h in k3.groupby(['predictor', 'n_label', 'k']):
        rows.append({'predictor': p, 'n': n, 'k': k, 'target_aware': int((h['class'] == 'target-aware').sum()), 'similar': int((h['class'] == 'similar').sum()),
                     'reusable': int((h['class'] == 'reusable').sum()), 'no_signal': int((h['class'] == 'no signal').sum()),
                     'R5_above_decorr': int((h.gain_decorr > 0).sum()), 'min_gain_best': round(float(h.gain_best.min()), 3), 'max_gain_best': round(float(h.gain_best.max()), 3)})
    ag = pd.DataFrame(rows); ag.to_csv(OUT / 'agreement.csv', index=False)
    for p in PS:
        print(f'\n{p}: number of kinds (of 3) where target-aware is ahead by at least the margin')
        print(ag[ag.predictor == p].pivot_table(index='n', columns='k', values='target_aware').reindex(order).astype(int).to_string())
        print(f'{p}: number of kinds where the reusable panel is ahead by at least the margin')
        print(ag[ag.predictor == p].pivot_table(index='n', columns='k', values='reusable').reindex(order).astype(int).to_string())

    print('\n== 3. Training units per measurement ==')
    bins = [0, 5, 10, 20, 1e9]; names = ['<5', '5-<10', '10-<20', '>=20']
    k3 = k3.assign(ratio=pd.cut(k3.n_per_k, bins, right=False, labels=names))
    tab = k3.groupby(['predictor', 'ratio'], observed=True).agg(cells=('gain_best', 'size'), above=('gain_best', lambda v: int((v > 0).sum())), margin=('gain_best', lambda v: int((v >= MARGIN).sum())),
                                                                 below=('gain_best', lambda v: int((v <= -MARGIN).sum())), mean_gain=('gain_best', 'mean')).round(3)
    print(tab.to_string()); tab.reset_index().to_csv(OUT / 'ratio_bins.csv', index=False)
    print('\nsame, per kind, ridge and trees, share of cells with gain above 0')
    print(k3.groupby(['source', 'predictor', 'ratio'], observed=True).gain_best.agg(lambda v: f'{(v > 0).sum()}/{len(v)}').unstack('ratio').to_string())

    print('\n== 4. Is a threshold on units per measurement portable? Pick it on two kinds, test on the third ==')
    grid = [2.5, 5, 7.5, 10, 12.5, 15, 20]
    rows = []
    for p in PS:
        d = k3[k3.predictor == p]
        for held in KINDS:
            train = d[d.source != held]; test = d[d.source == held]
            acc = {th: float(((train.n_per_k >= th) == (train.gain_best > 0)).mean()) for th in grid}
            best = max(sorted(acc), key=lambda th: acc[th])
            hit = (test.n_per_k >= best)
            rows.append({'predictor': p, 'held_out': held, 'threshold': best, 'train_accuracy': round(acc[best], 3), 'test_accuracy': round(float((hit == (test.gain_best > 0)).mean()), 3),
                         'test_above_threshold_positive': f'{int((test[hit].gain_best > 0).sum())}/{int(hit.sum())}', 'test_below_threshold_not_positive': f'{int((test[~hit].gain_best <= 0).sum())}/{int((~hit).sum())}'})
    lo = pd.DataFrame(rows); print(lo.to_string(index=False)); lo.to_csv(OUT / 'threshold_portability.csv', index=False)

    print('\n== 5. Fixed threshold of 10 units per measurement in every source, including the ones kept apart and development ==')
    rows = []
    for s, h in c.groupby('source'):
        for p in PS:
            x = h[h.predictor == p]
            a, b = x[x.n_per_k >= 10], x[x.n_per_k < 10]
            rows.append({'source': s, 'predictor': p, 'at_least_10_positive': f'{int((a.gain_best > 0).sum())}/{len(a)}', 'at_least_10_margin': f'{int((a.gain_best >= MARGIN).sum())}/{len(a)}', 'mean_gain_at_least_10': round(float(a.gain_best.mean()), 3),
                         'below_10_positive': f'{int((b.gain_best > 0).sum())}/{len(b)}', 'below_10_reusable_ahead': f'{int((b.gain_best <= -MARGIN).sum())}/{len(b)}', 'mean_gain_below_10': round(float(b.gain_best.mean()), 3)})
    dev = pd.read_csv(W / 'task12_r5' / 'dev_summary' / 'dev_ev_by_task.csv').astype({'n': str})
    dm = dev[dev.method.isin(['R5', 'decorr', 'random']) & dev.predictor.isin(PS) & dev.k.isin(KS) & (dev.n != 'all')].groupby(['n', 'k', 'method', 'predictor']).ev.mean().unstack('method').reset_index()
    dm['gain_best'] = dm.R5 - dm[['decorr', 'random']].max(axis=1); dm['n_per_k'] = dm.n.astype(int) / dm.k
    for p in PS:
        x = dm[dm.predictor == p]; a, b = x[x.n_per_k >= 10], x[x.n_per_k < 10]
        rows.append({'source': 'Development cell lines (not independent)', 'predictor': p, 'at_least_10_positive': f'{int((a.gain_best > 0).sum())}/{len(a)}', 'at_least_10_margin': f'{int((a.gain_best >= MARGIN).sum())}/{len(a)}', 'mean_gain_at_least_10': round(float(a.gain_best.mean()), 3),
                     'below_10_positive': f'{int((b.gain_best > 0).sum())}/{len(b)}', 'below_10_reusable_ahead': f'{int((b.gain_best <= -MARGIN).sum())}/{len(b)}', 'mean_gain_below_10': round(float(b.gain_best.mean()), 3)})
    fx = pd.DataFrame(rows); print(fx.to_string(index=False)); fx.to_csv(OUT / 'threshold_10.csv', index=False)

    print('\n== 6. Gain by panel size at n = 100 (R5 minus decorrelation; R5 minus the better reusable panel) ==')
    h = c[c.n == '100']
    print(h.pivot_table(index=['source', 'predictor'], columns='k', values='gain_decorr').round(3).to_string())
    print(h.pivot_table(index=['source', 'predictor'], columns='k', values='gain_best').round(3).to_string())

    print('\n== 7. Gain by training size at k = 5 and 10, R5 minus decorrelation ==')
    for k in (5, 10):
        print(f'k = {k}')
        print(c[c.k == k].pivot_table(index=['source', 'predictor'], columns='n_label', values='gain_decorr')[order].round(3).to_string())

    print('\n== 8. Size of the better reusable panel that matches R5 (interpolated, descriptive) ==')
    rows = []
    for s in KINDS + COHORTS + ['Beat AML 40 targets', 'PKIS2 random']:
        for p in PS:
            for n in (['100', '500'] if s in COHORTS + ['NHANES'] else ['100', 'all']):
                for k in (5, 10):
                    eq, ev5 = equivalent_size(t, s, p, n, k)
                    rows.append({'source': s, 'predictor': p, 'n': n, 'k': k, 'R5_ev': round(float(ev5), 3), 'reusable_equivalent': eq})
    eq = pd.DataFrame(rows); eq.to_csv(OUT / 'equivalent_size.csv', index=False)
    print(eq.pivot_table(index=['source', 'predictor', 'n'], columns='k', values='reusable_equivalent', aggfunc='first').to_string())

    print('\n== 9. Share of the gap to all candidates closed by R5: (R5 - better reusable) / (all - better reusable) ==')
    rows = []
    allev = t[t.method == 'all'].set_index(['source', 'n', 'predictor']).ev
    for s in KINDS + ['Beat AML 40 targets']:
        for p in PS:
            for n in (['100', '500'] if s == 'NHANES' else ['100', 'all']):
                for k in (5, 10):
                    r = c[(c.source == s) & (c.predictor == p) & (c.n == n) & (c.k == k)].iloc[0]
                    a = float(allev[(s, n, p)])
                    rows.append({'source': s, 'predictor': p, 'n': n, 'k': k, 'R5': round(r.R5, 3), 'reusable': round(r.best_reusable, 3), 'all': round(a, 3), 'share': round((r.R5 - r.best_reusable) / (a - r.best_reusable), 2) if a > r.best_reusable else None})
    gp = pd.DataFrame(rows); print(gp.to_string(index=False)); gp.to_csv(OUT / 'gap_share.csv', index=False)

    print('\n== 10. Target resolution by training size: R5 minus Shared-only, native predictions ==')
    n5 = t[t.method.isin(['R5', 'R5fix_Shared|group']) & (t.predictor == 'native') & t.k.isin([5, 10])].pivot_table(index=['source', 'k'], columns=['n', 'method'], values='ev')
    rows = {}
    for n in order[:3] + ['all', '500']:
        if (n, 'R5') in n5.columns:
            rows[n] = n5[(n, 'R5')] - n5[(n, 'R5fix_Shared|group')]
    rs = pd.DataFrame(rows); print(rs.round(3).to_string()); rs.to_csv(OUT / 'resolution_by_n.csv')

    print('\n== 11. Measurements actually used by R5 at the budget, ridge rows ==')
    print(t[(t.method == 'R5') & (t.predictor == 'ridge') & t.k.isin([5, 10, 30])].pivot_table(index=['source', 'n'], columns='k', values='k_used').round(1).to_string())

    print('\n== 12. Which predictor at a small panel: R5 panel with ridge minus with trees ==')
    print(c.pivot_table(index=['source', 'n_label', 'k'], columns='predictor', values='R5').assign(d=lambda x: x.ridge - x.gbt).d.unstack('k')[[5, 10]].round(3).reindex(KINDS, level=0).to_string())

    print('\n== 13. Target set width: the one comparison available (Beat AML; 9 targets and 71 candidates vs 40 and 40) ==')
    print(c[c.source.str.startswith('Beat') & (c.predictor == 'ridge')].pivot_table(index=['source', 'n_label'], columns='k', values='gain_decorr').round(3).to_string())

    print('\n== 14. Cell counts ==')
    print(k3['class'].value_counts().to_string())
    res['class_counts'] = k3['class'].value_counts().to_dict()
    (OUT / 'route_b.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
