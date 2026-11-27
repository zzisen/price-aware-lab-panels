#!/usr/bin/env python3
"""Task 17: independent recalculation of the G1, G2 and G3 decisions from the raw result files. Does not import summarize_gates.
Differences from the summary path, on purpose: G1 uses the recomputed decorr and random from results/G1 (checked equal to the Task 16 rerun) and R5 from the rerun;
G2 uses its own bootstrap seed; G3 fits by the normal equations. Ends with post hoc diagnostics of G3 that the plan did not list (labelled as such).
Prints VERIFY PASSED if every decision matches summary/endpoints.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
FIXED = HERE.parent / 'task16_route_b' / 'results_fixed'
RES = HERE / 'results'
END = json.loads((HERE / 'summary' / 'endpoints.json').read_text(encoding='utf-8'))
AWARE = ['lasso', 'somp', 'mrmr', 'pls_vip', 'rf', 'gbt_imp', 'univ']
REUSE = ['qr', 'fsca', 'medoid', 'pca_loading', 'leverage', 'high_var', 'decorr', 'random']
NH = {'NHANES_I': 'M', 'NH_A1_I': 'A1', 'NH_A2_H': 'A2', 'NH_A3_G': 'A3'}
bad = []


def same(name, a, b, tol=0.0):
    ok = a == b if tol == 0 else abs(a - b) <= tol
    print(('  ok  ' if ok else '  DIFF'), name, a, b)
    if not ok:
        bad.append(name)


def src(t):
    if t in NH:
        return NH[t]
    return t.split('_')[1] if t.startswith('V_') else 'Beat AML' if t.startswith('BEAT') else 'PKIS2' if t.startswith('KIN') else '?'


def rho(a, b):
    a, b = rankdata(a), rankdata(b)
    return float(np.corrcoef(a, b)[0, 1])


def fixed_rows(n_values):
    out = []
    for folder in ['V', 'U', 'L']:
        for f in sorted((FIXED / folder / 'parts').glob('*.csv')):
            h = pd.read_csv(f, usecols=['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k', 'sse', 'c0'], dtype={'n': str})
            out.append(h[(h.method == 'R5') | h.method.isin(['decorr', 'random'])].query('predictor in ["ridge", "gbt"] and k in [5, 10, 20]').loc[lambda a: a.n.isin(n_values)])
    return pd.concat(out, ignore_index=True).drop_duplicates(['task', 'outer', 'n', 'rep', 'method', 'predictor', 'k'])


# ---------------- G1 ----------------
print('== G1 ==')
g = pd.concat([pd.read_csv(f, dtype={'n': str}) for f in sorted((RES / 'G1' / 'parts').glob('*.csv'))], ignore_index=True)
g = g[g.n == '100']
r5 = fixed_rows(['100']).query('method == "R5"')
r5 = r5.merge(g[['task', 'outer', 'rep']].drop_duplicates(), on=['task', 'outer', 'rep'])
d = pd.concat([g[['task', 'outer', 'rep', 'method', 'predictor', 'k', 'sse', 'c0']], r5[['task', 'outer', 'rep', 'method', 'predictor', 'k', 'sse', 'c0']]], ignore_index=True)
ev = {}
for (task, meth, p, k), h in d.groupby(['task', 'method', 'predictor', 'k']):
    if task in NH:
        e = np.mean([1 - x.sse.sum() / x.c0.sum() for _, x in h.groupby('outer')])
    else:
        e = 1 - h.sse.sum() / h.c0.sum()
    ev.setdefault((src(task), meth, p, k), []).append(e)
ev = pd.Series({key: np.mean(v) for key, v in ev.items()})
ev.index.names = ['source', 'method', 'predictor', 'k']
assert '?' not in ev.index.get_level_values('source')
verdicts, lead_n, caught_n = [], 0, 0
for k in (5, 10):
    for p in ('ridge', 'gbt'):
        t = ev.xs((p, k), level=['predictor', 'k']).unstack('method')
        assert len(t) == 10
        mean = t.mean()
        rival = max(AWARE, key=lambda a: mean[a]); reuse = max(REUSE, key=lambda a: mean[a])
        side = 'R5' if mean['R5'] >= mean[rival] else rival
        w = int(((t.R5 - t[rival]) >= .01).sum()); l = int(((t.R5 - t[rival]) <= -.01).sum())
        lead = int(((t[side] - t[reuse]) >= .02).sum()); caught = int(((t[side] - t[reuse]) <= 0).sum())
        e = END['G1']['settings'][f'k{k}_{p}']
        same(f'k{k}_{p} rival', rival, e['rival']); same(f'k{k}_{p} R5 mean', round(float(mean.R5), 4), e['r5_mean'], 1e-4)
        same(f'k{k}_{p} wins', w, e['r5_wins']); same(f'k{k}_{p} loses', l, e['r5_loses'])
        same(f'k{k}_{p} best reusable', reuse, e['best_reuse']); same(f'k{k}_{p} aware lead sources', lead, e['aware_lead_sources'])
        verdicts.append('win' if w >= 5 else 'lose' if l >= 5 else 'neither')
        lead_n += lead >= 5; caught_n += caught >= 5
g1a = 'R5 clearly ahead' if verdicts.count('win') >= 2 and verdicts.count('lose') == 0 else 'R5 behind' if verdicts.count('lose') >= 2 else 'about the same'
same('G1A', g1a, END['G1']['G1A'])
same('G1B', 'holds' if lead_n >= 3 else 'caught' if caught_n >= 2 else 'narrowed', END['G1']['G1B'])

# ---------------- G2 ----------------
print('== G2 ==')
d = pd.concat([pd.read_csv(f, dtype={'unit': str}, compression=None) for f in sorted((RES / 'G2' / 'parts').glob('*.csv.gz'))], ignore_index=True)
d['a'] = d.v - d.v0; d['b'] = d.vmax - d.v0
rec = d.groupby(['source', 'k', 'predictor', 'method'])[['a', 'b']].mean()
rec = rec.a / rec.b
rng = np.random.default_rng(777)
meet = {}
for k in (5, 10):
    for p in ('ridge', 'gbt'):
        t = rec.xs((k, p), level=['k', 'predictor']).unstack('method')
        reuse = max([m for m in REUSE if m in t.columns], key=lambda m: t[m].mean())
        same(f'k{k}_{p} best reusable', reuse, END['G2']['settings'][f'k{k}_{p}']['best_reuse'])
        for m in ['R5'] + [m for m in AWARE if m in t.columns]:
            cnt = 0
            for s in t.index:
                u = d[(d.source == s) & (d.k == k) & (d.predictor == p) & (d.method == m)].groupby('unit')[['a', 'b']].sum().to_numpy()
                idx = rng.integers(0, len(u), size=(2000, len(u)))
                low = np.percentile(u[idx, 0].sum(1) / u[idx, 1].sum(1), 2.5)
                cnt += bool(t.loc[s, m] >= .10 and low > 0 and t.loc[s, m] - t.loc[s, reuse] >= .02)
            meet.setdefault(m, []).append(cnt)
            same(f'k{k}_{p} {m} sources meeting', cnt, END['G2']['settings'][f'k{k}_{p}']['sources_meeting'][m])
g2 = 'pass' if max(meet['R5']) >= 2 else 'partial' if any(max(v) >= 2 for m, v in meet.items() if m != 'R5') else 'fail'
same('G2', g2, END['G2']['pass'])

# ---------------- G3 ----------------
print('== G3 ==')


def lead_table(rows, rivals):
    s = rows.groupby(['task', 'n', 'k', 'predictor', 'method'])[['sse', 'c0']].sum()
    e = (1 - s.sse / s.c0).unstack('method')
    return (e.R5 - e[rivals].max(axis=1)).rename('lead').reset_index()


def feats(folder):
    f = pd.concat([pd.read_csv(x) for x in sorted((RES / folder).glob('*.csv'))], ignore_index=True)
    return f.groupby(['task', 'n', 'k']).agg(p=('p', 'first'), A=('A', 'mean'), B=('B', 'mean'), All=('All', 'mean')).reset_index()


def X(f, full):
    r = np.log(f.n / f.k); c = np.log(f.p / f.k)
    cols = [np.ones(len(f)), r, c] if not full else [np.ones(len(f)), f.B - f.A, f.All - f.A, r, c, (f.B - f.A) * r]
    return np.column_stack(cols)


def ols(x, y):
    return np.linalg.solve(x.T @ x, x.T @ y)


dev_rows = fixed_rows(['50', '100', '200'])
dev_rows['n'] = dev_rows.n.astype(int)
dev = lead_table(dev_rows, ['decorr', 'random']).merge(feats('G3_dev_features'), on=['task', 'n', 'k'])
t3 = pd.concat([pd.read_csv(f) for f in sorted((RES / 'G3' / 'parts').glob('*.csv'))], ignore_index=True)
test = lead_table(t3, ['decorr', 'random']).merge(feats('G3_test_features'), on=['task', 'n', 'k'])
test['source'] = [s[3:s.rindex('_T')] for s in test.task]
post = {}
for p in ('ridge', 'gbt'):
    a, b = dev[dev.predictor == p], test[test.predictor == p].copy()
    b['pred'] = X(b, True) @ ols(X(a, True), a.lead.to_numpy())
    b['base'] = X(b, False) @ ols(X(a, False), a.lead.to_numpy())
    c = b.groupby(['source', 'n', 'k'])[['lead', 'pred', 'base']].mean().reset_index()
    e = END['G3']['predictors'][p]
    y = c.lead >= .02
    same(f'{p} dev cells', len(a), e['dev_cells']); same(f'{p} test cells', len(c), e['test_cells'])
    same(f'{p} spearman', round(rho(c.pred, c.lead), 3), e['spearman'], 1e-3)
    same(f'{p} mae', round(float((c.pred - c.lead).abs().mean()), 4), e['mae'], 1e-4)
    same(f'{p} mae control', round(float((c.base - c.lead).abs().mean()), 4), e['mae_n_and_candidates_only'], 1e-4)
    same(f'{p} accuracy', round(float(((c.pred >= .02) == y).mean()), 3), e['accuracy'], 1e-3)
    same(f'{p} accuracy k<=10', round(float(((c.k <= 10) == y).mean()), 3), e['accuracy_k_le_10'], 1e-3)
    if p == 'ridge':
        same('G3 a', bool(rho(c.pred, c.lead) >= .5), END['G3']['a'])
        same('G3 b', bool((c.pred - c.lead).abs().mean() < (c.base - c.lead).abs().mean() or ((c.pred >= .02) == y).mean() >= ((c.k <= 10) == y).mean()), END['G3']['b'])
    within = [(rho(h.pred, h.lead), rho(h.base, h.lead) if h.base.nunique() > 1 else float('nan'), rho(h.pred - h.base, h.lead - h.base)) for _, h in c.groupby(['n', 'k'])]
    post[p] = {'share_of_cells_with_lead_ge_0.02': round(float(y.mean()), 3), 'accuracy_of_always_yes': round(float(y.mean()), 3),
               'accuracy_of_control_model': round(float(((c.base >= .02) == y).mean()), 3), 'spearman_of_control_model': round(rho(c.base, c.lead), 3),
               'mean_abs_gap_formula_vs_control': round(float((c.pred - c.base).abs().mean()), 4),
               'spearman_within_n_k_median_formula': round(float(np.median([w[0] for w in within])), 3),
               'spearman_within_n_k_median_control': round(float(np.nanmedian([w[1] for w in within])), 3) if not all(np.isnan(w[1]) for w in within) else 'constant within n and k',
               'p_values_in_test': sorted(set(int(v) for v in b.p))}
print('\n== G3 post hoc diagnostics (not in the plan) ==')
print(json.dumps(post, indent=1))
(HERE / 'summary' / 'g3_post_hoc.json').write_text(json.dumps(post, indent=1), encoding='utf-8')
print('\nVERIFY PASSED' if not bad else f'\nVERIFY FAILED: {bad}')
