#!/usr/bin/env python3
"""Task 16 summary: applies plan_route_b_final FROZEN sections 4 and 5 (V, W) and reports C, R, D and U.
Written before any result was viewed. EV of a block = 1 - sum SSE / sum C0, per task pooled over splits and repetitions, then the mean over tasks.
NHANES cohorts have one task, so EV is taken per outer split and averaged, as in the earlier stages.
'Ahead' means R5 leads both reusable panels by at least 0.02. Writes summary/*.csv and summary/endpoints.json. Run with .env312 after run_v.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
OUT = HERE / 'summary'
RES = HERE / 'results'
MARGIN = .02
KS = [3, 5, 10, 20, 30]
PS = ['ridge', 'gbt']
SOURCES = ['E1', 'E2', 'E3', 'E4']
BOOT = 2000
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 600); pd.set_option('display.max_columns', 40)


def ev(h):
    return 1 - h.sse.sum() / h.c0.sum()


def parts(a):
    return pd.concat([pd.read_csv(f) for f in sorted((RES / a / 'parts').glob('*.csv'))], ignore_index=True)


def by_task(d, keys):
    g = d.groupby(keys)
    return g.apply(ev, include_groups=False).rename('ev').reset_index()


def classify(p):
    p = p.copy()
    p['best_reusable'] = p[['decorr', 'random']].max(axis=1)
    p['lead'] = p.R5 - p.best_reusable
    p['lead_decorr'] = p.R5 - p.decorr

    def cls(r):
        if max(r.R5, r.best_reusable) <= 0:
            return 'no signal'
        if r.lead >= MARGIN:
            return 'target-aware'
        if r.lead <= -MARGIN:
            return 'reusable'
        return 'similar'
    p['class'] = p.apply(cls, axis=1)
    return p


def v_cells(v, src):
    v = v.assign(source=v.task.map(src), n=v.n.astype(str))
    t = by_task(v[v.method.isin(['R5', 'decorr', 'random']) & v.predictor.isin(PS) & v.k.isin(KS)], ['source', 'task', 'n', 'k', 'method', 'predictor'])
    m = t.groupby(['source', 'predictor', 'n', 'k', 'method']).ev.mean().unstack('method').reset_index()
    sizes = v.groupby(['source', 'n']).n_actual.mean()
    m['n_actual'] = [sizes[(s, n)] for s, n in zip(m.source, m.n)]
    m['n_per_k'] = m.n_actual / m.k
    return classify(m)


def decide_v(c, v, src):
    v = v.assign(source=v.task.map(src), n=v.n.astype(str))
    a = by_task(v[(v.method == 'all') & (v.predictor == 'ridge') & (v.n == 'all')], ['source', 'task']).groupby('source').ev.mean()
    out = {}
    for s in SOURCES:
        h = c[c.source == s]
        r, t = h[h.predictor == 'ridge'], h[h.predictor == 'gbt']
        r10, r5 = r[r.n_per_k >= 10], r[r.n_per_k < 5]
        t10 = t[(t.n_per_k >= 10) & (t['class'] != 'no signal')]
        share = lambda x: round(float((x['class'] == 'target-aware').mean()), 3) if len(x) else None  # noqa: E731
        v1 = bool(len(r10) and share(r10) >= .8 and not (r10['class'] == 'reusable').any())
        v2 = bool(len(t10) and share(t10) >= .8)
        v3 = bool(len(r5) and share(r5) <= .25)
        out[s] = {'all_candidates_ridge_largest_n': round(float(a[s]), 4), 'informative': bool(a[s] >= .10),
                  'V1_ridge_ratio_10_plus': {'cells': len(r10), 'target_aware': int((r10['class'] == 'target-aware').sum()), 'reusable': int((r10['class'] == 'reusable').sum()), 'no_signal': int((r10['class'] == 'no signal').sum()), 'share': share(r10), 'pass': v1},
                  'V2_trees_ratio_10_plus': {'cells_with_signal': len(t10), 'target_aware': int((t10['class'] == 'target-aware').sum()), 'share': share(t10), 'pass': v2},
                  'V3_ridge_ratio_below_5': {'cells': len(r5), 'target_aware': int((r5['class'] == 'target-aware').sum()), 'share': share(r5), 'pass': v3},
                  'pass': bool(v1 and v2 and v3)}
    inf = [s for s in SOURCES if out[s]['informative']]
    passed = [s for s in inf if out[s]['pass']]
    return {'sources': out, 'informative': inf, 'passed': passed, 'pass': bool(len(inf) >= 3 and len(passed) >= 3)}


def rules(c):
    rows = []
    for p in PS:
        for s in SOURCES + ['all sources']:
            h = c[(c.predictor == p) & (c.source == s if s != 'all sources' else c.source.notna())]
            y = h['class'] == 'target-aware'
            for name, pred in [('n/k >= 10', h.n_per_k >= 10), ('n >= 100', h.n_actual >= 100), ('k <= 10', h.k <= 10)]:
                rows.append({'predictor': p, 'source': s, 'rule': name, 'accuracy': round(float((pred == y).mean()), 3), 'cells': len(h)})
    return pd.DataFrame(rows).pivot_table(index=['predictor', 'source'], columns='rule', values='accuracy')


def decide_w(w):
    w = w.copy()
    t = by_task(w[w.predictor == 'ridge'], ['source', 'task', 'replicate', 'size', 'k', 'method']).pivot_table(index=['source', 'replicate', 'size', 'k'], columns='method', values='ev').reset_index()
    t['lead'] = t.R5 - t[['decorr', 'random']].max(axis=1); t['lead_decorr'] = t.R5 - t.decorr
    m = t.groupby(['source', 'size', 'k'])[['R5', 'decorr', 'random', 'lead', 'lead_decorr']].mean().reset_index()
    out = {}
    for s, h in m.groupby('source'):
        top = h['size'].max()
        rows = {k: float(h[(h['size'] == 9) & (h.k == k)].lead.iloc[0] - h[(h['size'] == top) & (h.k == k)].lead.iloc[0]) for k in (5, 10)}
        out[s] = {'largest_target_set': int(top), 'lead_drop_k5': round(rows[5], 4), 'lead_drop_k10': round(rows[10], 4), 'supports': bool(rows[5] >= .01 and rows[10] >= .01)}
    n = sum(v['supports'] for v in out.values())
    return m, {'sources': out, 'supporting': n, 'of': len(out), 'pass': bool(n >= 4)}


def summarize_c(cc):
    t = by_task(cc[cc.predictor.isin(PS)], ['source', 'predictor', 'task', 'replicate', 'size', 'k', 'method']).pivot_table(index=['source', 'predictor', 'replicate', 'size', 'k'], columns='method', values='ev').reset_index()
    t['lead'] = t.R5 - t[['decorr', 'random']].max(axis=1); t['lead_decorr'] = t.R5 - t.decorr
    return t.groupby(['source', 'predictor', 'size', 'k'])[['R5', 'decorr', 'random', 'lead', 'lead_decorr']].mean().round(3).reset_index()


def summarize_r(r):
    g = r.groupby(['source', 'task', 'n', 'k', 'predictor', 'method', 'set'])[['sse', 'c0']].sum().unstack('set')
    rows = []
    for (s, task, n, k, p), h in g.groupby(level=[0, 1, 2, 3, 4]):
        h = h.droplevel([0, 1, 2, 3, 4])
        sa, sb, ca, cb = h[('sse', 'A')], h[('sse', 'B')], h[('c0', 'A')], h[('c0', 'B')]

        def mix(m, f):
            return 1 - ((1 - f) * sa[m] + f * sb[m]) / ((1 - f) * ca[m] + f * cb[m])
        fs = np.linspace(0, 1, 101)
        d = np.array([mix('R5_A', f) - mix('decorr', f) for f in fs])
        b = np.array([mix('R5_A', f) - max(mix('decorr', f), mix('random', f)) for f in fs])
        rows.append({'source': s, 'task': task, 'n': n, 'k': k, 'predictor': p, 'dedicated_on_A': mix('R5_A', 0), 'wrong_panel_on_B': mix('R5_A', 1), 'dedicated_on_B': mix('R5_B', 1), 'decorr_on_A': mix('decorr', 0), 'decorr_on_B': mix('decorr', 1),
                     'random_on_A': mix('random', 0), 'random_on_B': mix('random', 1), 'mix_third': mix('R5_A', 1 / 3), 'mix_two_thirds': mix('R5_A', 2 / 3), 'decorr_third': mix('decorr', 1 / 3), 'decorr_two_thirds': mix('decorr', 2 / 3),
                     'break_even_vs_decorr': float(fs[np.argmax(d <= 0)]) if (d <= 0).any() else 1.0, 'break_even_vs_better': float(fs[np.argmax(b <= 0)]) if (b <= 0).any() else 1.0})
    t = pd.DataFrame(rows)
    m = t.groupby(['source', 'n', 'k', 'predictor']).mean(numeric_only=True).round(3).reset_index()
    return t, m


LEGACY = lambda t: 'Beat AML' if t.startswith('BEAT') else 'PKIS2' if t.startswith('KIN') else {'NHANES_I': 'M', 'NH_A1_I': 'A1', 'NH_A2_H': 'A2', 'NH_A3_G': 'A3'}[t]  # noqa: E731


def summarize_d(v, u, src):
    v = v[v.n.astype(str) == '100'].assign(source=lambda x: x.task.map(src), outer_key='all')
    u = u.assign(source=u.task.map(LEGACY))
    u['outer_key'] = np.where(u.source.isin(['M', 'A1', 'A2', 'A3']), u.outer.astype(str), 'all')
    d = pd.concat([v, u], ignore_index=True)
    d = d[d.predictor.isin(PS) & d.method.isin(['R5', 'decorr', 'random'])]
    t = by_task(d, ['source', 'task', 'outer_key', 'k', 'method', 'predictor']).groupby(['source', 'k', 'method', 'predictor']).ev.mean().unstack('method').reset_index()
    rows = []
    for (s, p), h in t.groupby(['source', 'predictor']):
        h = h.sort_values('k'); h['better'] = h[['decorr', 'random']].max(axis=1)
        for k in (5, 10):
            target = float(h[h.k == k].R5.iloc[0])
            row = {'source': s, 'predictor': p, 'R5_k': k, 'R5_ev': round(target, 3)}
            for lab, col in [('better_reusable', 'better'), ('decorr', 'decorr'), ('random', 'random')]:
                hit = h[h[col] >= target]
                row[f'smallest_{lab}'] = int(hit.k.iloc[0]) if len(hit) else '>30'
            rows.append(row)
    return t, pd.DataFrame(rows)


def clusters(tasks_units, source):
    if source == 'E1':
        return pd.read_csv(HERE / 'pools' / 'E1_groups.csv', dtype=str, index_col=0).group.reindex(tasks_units).to_numpy()
    if source == 'PKIS2':
        return pd.read_csv(W / 'task14_m2' / 'kin_chemotypes.csv', dtype=str, index_col=0).chemotype.reindex(tasks_units).to_numpy()
    return np.asarray(tasks_units)


def bootstrap(a, source, rng):
    """Resample independent units across all tasks of a source, keeping tasks and panels paired. Test-sample uncertainty only."""
    a = a.astype({'unit': str})
    a['cluster'] = clusters(a.unit.to_numpy(), source)
    g = a.groupby(['task', 'method', 'predictor', 'k', 'cluster'])[['sse', 'c0']].sum()
    ids = sorted(a.cluster.unique()); pos = {c: i for i, c in enumerate(ids)}
    w = rng.multinomial(len(ids), [1 / len(ids)] * len(ids), size=BOOT).astype(float)
    arr = {}
    for key, h in g.groupby(level=[0, 1, 2, 3]):
        cl = h.index.get_level_values(4).map(pos).to_numpy()
        s, c = np.zeros(len(ids)), np.zeros(len(ids))
        s[cl], c[cl] = h.sse.to_numpy(), h.c0.to_numpy()
        arr[key] = 1 - (w @ s) / (w @ c)
    tasks = sorted({k[0] for k in arr})
    mean = lambda m, p, k: np.mean([arr[(t, m, p, k)] for t in tasks], axis=0)  # noqa: E731
    rows = []
    for p in PS:
        for k in (5, 10):
            d = mean('R5', p, k) - mean('decorr', p, k)
            rows.append({'source': source, 'predictor': p, 'measure': f'R5 minus decorrelation, k = {k}', 'low': round(float(np.percentile(d, 2.5)), 3), 'median': round(float(np.median(d)), 3), 'high': round(float(np.percentile(d, 97.5)), 3)})
        r = mean('R5', p, 10) / mean('decorr', p, 30)
        rows.append({'source': source, 'predictor': p, 'measure': 'R5 k = 10 / decorrelation k = 30', 'low': round(float(np.percentile(r, 2.5)), 3), 'median': round(float(np.median(r)), 3), 'high': round(float(np.percentile(r, 97.5)), 3)})
    return rows


def check_cache(u):
    """Plan section 9: the rerun must match the cached results."""
    rows = []
    t13, t14 = W / 'task13_confirm' / 'results' / 'parts', W / 'task14_m2'
    for (task, o), h in u.groupby(['task', 'outer']):
        s = LEGACY(task)
        f = t13 / f'{task}_{o}.csv' if s in ('Beat AML', 'M') else (t14 / 'results_kinase_chemotype' / 'parts' / f'{task}_{o}.csv' if s == 'PKIS2' else t14 / 'results_nhanes' / 'parts' / f'{task}_{o}.csv')
        old = pd.read_csv(f); old['n'] = old.n.astype(str)
        m = h.assign(n=h.n.astype(str)).merge(old, on=['n', 'rep', 'method', 'predictor', 'k'], suffixes=('', '_old'))
        m = m[m.method != 'all']
        rows.append({'task': task, 'outer': o, 'rows': len(m), 'largest_difference': float(np.abs(m.sse - m.sse_old).max())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    src = {t['task']: t['source'] for t in tasks if t['analysis'] == 'V'}
    end = {}

    # ---- V ----
    v = parts('V')
    c = v_cells(v, src); c.to_csv(OUT / 'v_cells.csv', index=False)
    end['V'] = decide_v(c, v, src)

    # ---- W ----
    wm, end['W'] = decide_w(parts('W')); wm.round(4).to_csv(OUT / 'w_curve.csv', index=False)
    (OUT / 'endpoints.json').write_text(json.dumps(end, indent=1, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(end, indent=1, ensure_ascii=False))

    # ---- report only ----
    order = ['25', '50', '100', '200', '400', 'all']
    for s in SOURCES:
        for p in PS:
            h = c[(c.source == s) & (c.predictor == p)]
            idx = [o for o in order if o in set(h.n)]
            print(f'\n== {s}, {p}: lead over the better reusable panel; class ==')
            print(h.pivot_table(index='n', columns='k', values='lead').reindex(idx).round(3).to_string())
            print(h.pivot_table(index='n', columns='k', values='class', aggfunc='first').reindex(idx).to_string())
            print('R5 EV'); print(h.pivot_table(index='n', columns='k', values='R5').reindex(idx).round(3).to_string())
    print('\n== mean training size =='); print(c.groupby(['source', 'n']).n_actual.mean().round(0).to_string())

    c = c.assign(ratio=pd.cut(c.n_per_k, [0, 5, 10, 20, 1e9], right=False, labels=['<5', '5-<10', '10-<20', '>=20']))
    tab = pd.crosstab([c.predictor, c.ratio], c['class'], dropna=False); print('\n== class by units per measurement, all new sources =='); print(tab.to_string()); tab.to_csv(OUT / 'v_ratio_table.csv')
    print('\n== same, per source =='); print(pd.crosstab([c.source, c.predictor, c.ratio], c['class'], dropna=False).to_string())
    print('\n== against decorrelation alone: cells with lead >= 0.02 ==')
    print(c.assign(ok=c.lead_decorr >= MARGIN).groupby(['source', 'predictor', 'ratio'], observed=True).ok.agg(lambda x: f'{int(x.sum())}/{len(x)}').unstack('ratio').to_string())
    ru = rules(c); ru.to_csv(OUT / 'v_rules.csv'); print('\n== which rule predicts target-aware ahead (accuracy) =='); print(ru.to_string())

    print('\n== W: lead by target-set size, Ridge =='); print(wm.pivot_table(index=['source', 'k'], columns='size', values='lead').round(3).to_string())
    print('\n== W: lead over decorrelation alone, Ridge =='); print(wm.pivot_table(index=['source', 'k'], columns='size', values='lead_decorr').round(3).to_string())

    cs = summarize_c(parts('C')); cs.to_csv(OUT / 'c_curve.csv', index=False)
    print('\n== C: lead by candidate count ==');  print(cs.pivot_table(index=['source', 'predictor', 'k'], columns='size', values='lead').round(3).to_string())

    rt, rm = summarize_r(parts('R')); rt.round(4).to_csv(OUT / 'r_by_task.csv', index=False); rm.to_csv(OUT / 'r_summary.csv', index=False)
    print('\n== R: changing targets =='); print(rm.to_string(index=False))

    u = parts('U')
    cc = check_cache(u); cc.to_csv(OUT / 'u_cache_check.csv', index=False)
    print('\n== U: rerun against the cache: rows', int(cc.rows.sum()), '| largest difference in panel SSE', float(cc.largest_difference.max()))
    dt, ds = summarize_d(v, u, src); dt.round(4).to_csv(OUT / 'd_curves.csv', index=False); ds.to_csv(OUT / 'd_equivalent.csv', index=False)
    print('\n== D: smallest tested reusable panel that reaches R5, n = 100 =='); print(ds.to_string(index=False))

    rng = np.random.default_rng(20261190)
    rows = []
    for s in SOURCES:
        a = pd.concat([pd.read_csv(f, dtype={'unit': str}) for f in sorted((RES / 'V' / 'units').glob(f'V_{s}_*.csv'))], ignore_index=True)
        rows += bootstrap(a, s, rng)
    a = pd.concat([pd.read_csv(f, dtype={'unit': str}) for f in sorted((RES / 'U' / 'units').glob('*.csv'))], ignore_index=True)
    a['source'] = a.task.map(LEGACY)
    for s, h in a.groupby('source'):
        rows += bootstrap(h, s, rng)
    bs = pd.DataFrame(rows); bs.to_csv(OUT / 'u_bootstrap.csv', index=False)
    print('\n== U: 95% resampling interval, test-sample uncertainty only =='); print(bs.to_string(index=False))


if __name__ == '__main__':
    main()
