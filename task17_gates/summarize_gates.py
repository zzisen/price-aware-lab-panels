#!/usr/bin/env python3
"""Task 17 summary: applies plan_gates sections 3, 4 and 5. Written before any new accuracy was viewed.
EV as in Task 16: 1 - sum SSE / sum C0 per task, pooled over splits and repetitions, then the mean over tasks; NHANES cohorts have one task, so EV is taken per outer split and averaged.
Writes summary/*.csv and summary/endpoints.json. Run with .env312 after run_gates.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
W = HERE.parent
FIXED = W / 'task16_route_b' / 'results_fixed'
RES = HERE / 'results'
OUT = HERE / 'summary'
AWARE = ['lasso', 'somp', 'mrmr', 'pls_vip', 'rf', 'gbt_imp', 'univ']
REUSE = ['qr', 'fsca', 'medoid', 'pca_loading', 'leverage', 'high_var', 'decorr', 'random']
SETTINGS = [(k, p) for k in (5, 10) for p in ('ridge', 'gbt')]
SOURCES = ['E1', 'E2', 'E3', 'E4', 'Beat AML', 'PKIS2', 'M', 'A1', 'A2', 'A3']
NH = ['M', 'A1', 'A2', 'A3']
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 800); pd.set_option('display.max_columns', 40)


def source_of(t):
    if t.startswith('V_'):
        return t.split('_')[1]
    if t.startswith('BEAT'):
        return 'Beat AML'
    if t.startswith('KIN'):
        return 'PKIS2'
    return {'NHANES_I': 'M', 'NH_A1_I': 'A1', 'NH_A2_H': 'A2', 'NH_A3_G': 'A3'}[t]


def ev_table(d, keys):
    """Mean over tasks of the pooled EV; per outer split inside NHANES."""
    d = d.assign(okey=np.where(d.source.isin(NH), d.outer.astype(str), 'all'))
    g = d.groupby(['task', 'okey'] + keys)[['sse', 'c0']].sum().reset_index()
    g['ev'] = 1 - g.sse / g.c0
    return g.groupby(keys).ev.mean().reset_index()


def g1_rows():
    base = pd.concat([pd.read_csv(f) for f in sorted((RES / 'G1' / 'parts').glob('*.csv'))], ignore_index=True)
    base = base[~base.method.isin(['decorr', 'random'])]                # taken from the rerun below, already checked equal
    old = []
    for folder in ['V', 'U', 'L']:
        for f in sorted((FIXED / folder / 'parts').glob('*.csv')):
            h = pd.read_csv(f, usecols=['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k', 'sse', 'c0'])
            old.append(h[h.method.isin(['R5', 'decorr', 'random']) & h.predictor.isin(['ridge', 'gbt']) & h.k.isin([5, 10, 20])])
    old = pd.concat(old, ignore_index=True)
    old['n'] = old.n.astype(str)
    old = old.drop_duplicates(['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k'])
    base['n'] = base.n.astype(str)
    have = base[['task', 'outer', 'n', 'rep']].drop_duplicates()
    old = old.merge(have, on=['task', 'outer', 'n', 'rep'])
    d = pd.concat([base, old], ignore_index=True)
    d['source'] = d.task.map(source_of)
    return d


def g1(d):
    size = d.groupby(['source', 'n']).n_actual.mean().dropna().reset_index()
    t = ev_table(d, ['source', 'n', 'k', 'predictor', 'method'])
    t.round(4).to_csv(OUT / 'g1_ev.csv', index=False)
    m = t[t.n == '100']
    out, rows = {'settings': {}}, []
    for k, p in SETTINGS:
        h = m[(m.k == k) & (m.predictor == p)].pivot_table(index='source', columns='method', values='ev').reindex(SOURCES)
        means = h.mean()
        rival = means[AWARE].idxmax(); reuse = means[REUSE].idxmax()
        team = 'R5' if means['R5'] >= means[rival] else rival
        d1 = h['R5'] - h[rival]; d2 = h[team] - h[reuse]
        s = {'rival': rival, 'r5_mean': round(float(means['R5']), 4), 'rival_mean': round(float(means[rival]), 4), 'r5_wins': int((d1 >= .01).sum()), 'r5_loses': int((d1 <= -.01).sum()),
             'best_reuse': reuse, 'reuse_mean': round(float(means[reuse]), 4), 'aware_side': team, 'aware_lead_sources': int((d2 >= .02).sum()), 'reuse_caught_sources': int((d2 <= 0).sum()),
             'oracle_rival_wins': int((h['R5'] - h[AWARE].max(axis=1) >= .01).sum()), 'oracle_rival_loses': int((h['R5'] - h[AWARE].max(axis=1) <= -.01).sum())}
        s['r5_setting'] = 'win' if s['r5_wins'] >= 5 else 'lose' if s['r5_loses'] >= 5 else 'neither'
        out['settings'][f'k{k}_{p}'] = s
        for src in SOURCES:
            rows.append({'k': k, 'predictor': p, 'source': src, 'R5': h.loc[src, 'R5'], 'rival': rival, 'rival_ev': h.loc[src, rival], 'r5_minus_rival': d1[src], 'aware_side': team, 'best_reuse': reuse, 'reuse_ev': h.loc[src, reuse], 'aware_minus_reuse': d2[src]})
    pd.DataFrame(rows).round(4).to_csv(OUT / 'g1_by_source.csv', index=False)
    v = [s['r5_setting'] for s in out['settings'].values()]
    wins, loses = v.count('win'), v.count('lose')
    out['G1A'] = 'R5 clearly ahead' if wins >= 2 and loses == 0 else 'R5 behind' if loses >= 2 else 'about the same'
    out['pass'] = out['G1A'] == 'R5 clearly ahead'
    lead = sum(s['aware_lead_sources'] >= 5 for s in out['settings'].values())
    caught = sum(s['reuse_caught_sources'] >= 5 for s in out['settings'].values())
    out['G1B'] = 'holds' if lead >= 3 else 'caught' if caught >= 2 else 'narrowed'

    # report only: the three V rules with all eight reusable panels on the other side
    n_act = d.groupby(['source', 'n']).n_actual.mean()
    e = t[t.source.isin(['E1', 'E2', 'E3', 'E4'])].pivot_table(index=['source', 'n', 'k', 'predictor'], columns='method', values='ev').reset_index()
    e['better_reuse'] = e[REUSE].max(axis=1); e['lead'] = e.R5 - e.better_reuse
    e['n_per_k'] = [n_act[(s, n)] / k for s, n, k in zip(e.source, e.n, e.k)]
    e[['source', 'n', 'k', 'predictor', 'R5', 'better_reuse', 'lead', 'n_per_k']].round(4).to_csv(OUT / 'g1_v_rules_strong.csv', index=False)
    rep = {}
    for s in ['E1', 'E2', 'E3', 'E4']:
        h = e[e.source == s]
        r10, t10, r5 = h[(h.predictor == 'ridge') & (h.n_per_k >= 10)], h[(h.predictor == 'gbt') & (h.n_per_k >= 10)], h[(h.predictor == 'ridge') & (h.n_per_k < 5)]
        rep[s] = {'ridge_ratio_10': f'{int((r10.lead >= .02).sum())}/{len(r10)}', 'trees_ratio_10': f'{int((t10.lead >= .02).sum())}/{len(t10)}', 'ridge_ratio_below_5': f'{int((r5.lead >= .02).sum())}/{len(r5)}'}
    out['V_rules_against_eight_reusable'] = rep
    print('\n== G1: EV at n = 100 =='); print(m.pivot_table(index=['predictor', 'k', 'source'], columns='method', values='ev').round(3).to_string())
    print('\n== G1: mean over the ten sources, n = 100 =='); print(m.groupby(['predictor', 'k', 'method']).ev.mean().unstack('method').round(3).to_string())
    print('\n== G1: mean over sources, other sizes =='); print(t[t.n != '100'].assign(size=lambda a: np.where(a.n == '50', '50', 'largest')).groupby(['size', 'predictor', 'k', 'method']).ev.mean().unstack('method').round(3).to_string())
    return out


def recovered(h):
    return float((h.v - h.v0).mean() / (h.vmax - h.v0).mean())


def g2():
    d = pd.concat([pd.read_csv(f, dtype={'unit': str}, compression=None) for f in sorted((RES / 'G2' / 'parts').glob('*.csv.gz'))], ignore_index=True)   # written uncompressed (DEVIATIONS D1)
    rows = []
    for (s, k, p, m), h in d.groupby(['source', 'k', 'predictor', 'method']):
        rows.append({'source': s, 'k': k, 'predictor': p, 'method': m, 'recovered': recovered(h), 'pct': h.pct.mean(), 'top5': h.top5.mean(), 'gap_mean': (h.vmax - h.v0).mean()})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / 'g2_by_method.csv', index=False)
    pop = d[d.method == 'R5'].groupby('source').apply(lambda h: (h.v0 == h.vmax).mean(), include_groups=False)
    print('\n== G2: share of patients for whom the population-best drug is already the best =='); print(pop.round(3).to_string())
    print('\n== G2: recovered share of the gap =='); print(t.pivot_table(index=['source', 'predictor', 'k'], columns='method', values='recovered').round(3).to_string())
    print('\n== G2: percentile of the chosen drug =='); print(t.pivot_table(index=['source', 'predictor', 'k'], columns='method', values='pct').round(3).to_string())
    print('\n== G2: overlap with the true top 5 =='); print(t.pivot_table(index=['source', 'predictor', 'k'], columns='method', values='top5').round(3).to_string())

    def bootstrap(h, rng):
        """Resamples patients; all tasks, splits and repetitions of a patient stay together."""
        g = h.assign(a=h.v - h.v0, b=h.vmax - h.v0).groupby('unit')[['a', 'b']].sum().to_numpy()
        w = rng.multinomial(len(g), [1 / len(g)] * len(g), size=2000)
        return np.percentile((w @ g[:, 0]) / (w @ g[:, 1]), 2.5)

    rng = np.random.default_rng(20261700)
    aware = [m for m in t.method.unique() if m in AWARE]
    reuse = [m for m in t.method.unique() if m in REUSE]
    out = {'settings': {}}
    for k, p in SETTINGS:
        h = t[(t.k == k) & (t.predictor == p)].pivot_table(index='source', columns='method', values='recovered')
        best_reuse = h[reuse].mean().idxmax()
        s = {'best_reuse': best_reuse, 'sources': {}}
        cand = {'R5': 0}
        for a in aware:
            cand[a] = 0
        for m in cand:
            for src in h.index:
                sub = d[(d.source == src) & (d.k == k) & (d.predictor == p) & (d.method == m)]
                low = bootstrap(sub, rng)
                ok_a = bool(h.loc[src, m] >= .10 and low > 0); ok_b = bool(h.loc[src, m] - h.loc[src, best_reuse] >= .02)
                cand[m] += ok_a and ok_b
                if m == 'R5':
                    s['sources'][src] = {'recovered': round(float(h.loc[src, m]), 4), 'low95': round(float(low), 4), 'best_reuse_value': round(float(h.loc[src, best_reuse]), 4), '2A': ok_a, '2B': ok_b}
        s['sources_meeting'] = {m: int(v) for m, v in cand.items()}
        out['settings'][f'k{k}_{p}'] = s
    r5 = any(s['sources_meeting']['R5'] >= 2 for s in out['settings'].values())
    other = sorted({m for s in out['settings'].values() for m, v in s['sources_meeting'].items() if m != 'R5' and v >= 2})
    out['pass'] = 'pass' if r5 else 'partial' if other else 'fail'
    out['baselines_meeting'] = other

    # report only: AUROC of calling responders (top quarter of each drug in the test patients) is not computable from the saved choices; the percentile and top-5 values above stand in for it
    return out


def spearman(a, b):
    return float(spearmanr(a, b).statistic)


def design(f, full):
    r, c = np.log(f.n.astype(float) / f.k), np.log(f.p / f.k)
    if full:
        G, H = f.B - f.A, f.All - f.A
        return np.column_stack([np.ones(len(f)), G, H, r, c, G * r])
    return np.column_stack([np.ones(len(f)), r, c])


def leads(d, rival):
    """R5 minus the better of the rival panels, per task x n x k x predictor."""
    a = d.groupby(['task', 'n', 'k', 'predictor', 'method'])[['sse', 'c0']].sum().reset_index()
    a['ev'] = 1 - a.sse / a.c0
    p = a.pivot_table(index=['task', 'n', 'k', 'predictor'], columns='method', values='ev').reset_index()
    p['lead'] = p.R5 - p[rival].max(axis=1)
    return p


def g3():
    keep = ['task', 'n', 'k', 'p', 'A', 'B', 'All']
    fdev = pd.concat([pd.read_csv(f) for f in sorted((RES / 'G3_dev_features').glob('*.csv'))], ignore_index=True)
    ftest = pd.concat([pd.read_csv(f) for f in sorted((RES / 'G3_test_features').glob('*.csv'))], ignore_index=True)
    fdev = fdev.groupby(['task', 'n', 'k', 'p'])[['A', 'B', 'All']].mean().reset_index()[keep]
    ftest = ftest.groupby(['task', 'n', 'k', 'p'])[['A', 'B', 'All']].mean().reset_index()[keep]

    d = []
    for folder in ['V', 'U', 'L']:
        for f in sorted((FIXED / folder / 'parts').glob('*.csv')):
            h = pd.read_csv(f, usecols=['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k', 'sse', 'c0'])
            h = h[h.method.isin(['R5', 'decorr', 'random']) & h.predictor.isin(['ridge', 'gbt']) & h.k.isin([5, 10, 20]) & h.n.astype(str).isin(['50', '100', '200'])]
            d.append(h)
    d = pd.concat(d, ignore_index=True).drop_duplicates(['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k'])
    d['n'] = d.n.astype(int)
    dev = leads(d, ['decorr', 'random']).merge(fdev, on=['task', 'n', 'k'])
    dev['source'] = dev.task.map(source_of)

    t = pd.concat([pd.read_csv(f) for f in sorted((RES / 'G3' / 'parts').glob('*.csv'))], ignore_index=True)
    test = leads(t, ['decorr', 'random']).merge(ftest, on=['task', 'n', 'k'])
    strong = leads(t, ['qr', 'fsca', 'medoid'])[['task', 'n', 'k', 'predictor', 'lead']].rename(columns={'lead': 'lead_strong'})
    test = test.merge(strong, on=['task', 'n', 'k', 'predictor'])
    test['source'] = test.task.str.replace(r'^G3_', '', regex=True).str.replace(r'_T\d$', '', regex=True)

    out = {'predictors': {}}
    tabs = []
    for pname in ['ridge', 'gbt']:
        a, b = dev[dev.predictor == pname], test[test.predictor == pname]
        beta = np.linalg.lstsq(design(a, True), a.lead, rcond=None)[0]
        base = np.linalg.lstsq(design(a, False), a.lead, rcond=None)[0]
        b = b.assign(pred=design(b, True) @ beta, pred_base=design(b, False) @ base)
        cell = b.groupby(['source', 'n', 'k']).agg(lead=('lead', 'mean'), lead_strong=('lead_strong', 'mean'), pred=('pred', 'mean'), pred_base=('pred_base', 'mean')).reset_index()
        cell['predictor'] = pname
        tabs.append(cell)
        rho = spearman(cell.pred, cell.lead); mae = float((cell.pred - cell.lead).abs().mean()); mae0 = float((cell.pred_base - cell.lead).abs().mean())
        y = cell.lead >= .02
        acc = float(((cell.pred >= .02) == y).mean()); acc_k = float(((cell.k <= 10) == y).mean()); acc_nk = float(((cell.n / cell.k >= 10) == y).mean())
        within = [spearman(h.pred, h.lead) for _, h in cell.groupby(['n', 'k'])]
        loso = []
        for s in a.source.unique():
            tr, va = a[a.source != s], a[a.source == s]
            bb = np.linalg.lstsq(design(tr, True), tr.lead, rcond=None)[0]
            v = va.assign(pred=design(va, True) @ bb).groupby(['n', 'k'])[['pred', 'lead']].mean()
            loso.append(float((v.pred - v.lead).abs().mean()))
        out['predictors'][pname] = {'coefficients': dict(zip(['intercept', 'G', 'H', 'log_n_over_k', 'log_p_over_k', 'G_x_log_n_over_k'], np.round(beta, 4).tolist())), 'dev_cells': len(a), 'test_cells': len(cell), 'test_sources': int(cell.source.nunique()),
                                    'spearman': round(rho, 3), 'mae': round(mae, 4), 'mae_n_and_candidates_only': round(mae0, 4), 'accuracy': round(acc, 3), 'accuracy_k_le_10': round(acc_k, 3), 'accuracy_n_per_k_ge_10': round(acc_nk, 3),
                                    'spearman_within_n_k_median': round(float(np.median(within)), 3), 'dev_leave_one_source_out_mae': round(float(np.mean(loso)), 4),
                                    'spearman_against_strong_reusable': round(spearman(cell.pred, cell.lead_strong), 3), 'mean_lead': round(float(cell.lead.mean()), 4), 'mean_lead_strong': round(float(cell.lead_strong.mean()), 4)}
    pd.concat(tabs, ignore_index=True).round(4).to_csv(OUT / 'g3_test_cells.csv', index=False)
    r = out['predictors']['ridge']
    out['a'] = bool(r['spearman'] >= .5)
    out['b'] = bool(r['mae'] < r['mae_n_and_candidates_only'] or r['accuracy'] >= r['accuracy_k_le_10'])
    out['pass'] = bool(out['a'] and out['b'])
    print('\n== G3: test cells, Ridge: actual lead by n and k, source means =='); print(tabs[0].pivot_table(index='source', columns=['n', 'k'], values='lead').round(3).to_string())
    print('\n== G3: mean by n and k =='); print(pd.concat(tabs).groupby(['predictor', 'n', 'k'])[['lead', 'lead_strong', 'pred', 'pred_base']].mean().round(3).to_string())
    print('\n== G3: by assay =='); print(pd.concat(tabs).assign(assay=lambda a: a.source.str[0]).groupby(['predictor', 'assay', 'k'])[['lead', 'lead_strong', 'pred']].mean().round(3).to_string())
    return out


def main():
    OUT.mkdir(exist_ok=True)
    end = {'G1': g1(g1_rows()), 'G2': g2(), 'G3': g3()}
    (OUT / 'endpoints.json').write_text(json.dumps(end, indent=1, ensure_ascii=False), encoding='utf-8')
    print('\n== decisions =='); print(json.dumps(end, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
