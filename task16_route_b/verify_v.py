#!/usr/bin/env python3
"""Task 16: recomputes the decisions and main report values from results_fixed/parts, by plain loops over the raw rows. Does not import summarize_v.
Compares them with summary_fixed/, then lists what changed against the first run (summary/). Run with .env312 after summarize_fixed.py."""
import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / 'results_fixed'
SUM = HERE / 'summary_fixed'
OLD = HERE / 'summary'
PANELS = ('R5', 'decorr', 'random')
BUDGETS = {3, 5, 10, 20, 30}
LEGACY = {'NHANES_I': 'M', 'NH_A1_I': 'A1', 'NH_A2_H': 'A2', 'NH_A3_G': 'A3'}
failures = []


def rows_of(stage):
    for f in sorted((RES / stage / 'parts').glob('*.csv')):
        with f.open(encoding='utf-8', newline='') as h:
            yield from csv.DictReader(h)


def same(label, a, b, tol):
    ok = abs(a - b) <= tol
    if not ok:
        failures.append(f'{label}: recomputed {a}, summary {b}')
    return ok


def label(r5, better):
    if max(r5, better) <= 0:
        return 'no signal'
    if r5 - better >= .02:
        return 'target-aware'
    if r5 - better <= -.02:
        return 'reusable'
    return 'similar'


# ---- V ----
sums = defaultdict(lambda: [0., 0.])
size_sum = defaultdict(lambda: [0., 0])
exact_seen, jobs = defaultdict(bool), set()
for r in rows_of('V'):
    src = r['task'].split('_')[1]
    jobs.add((r['task'], r['outer']))
    size_sum[(src, r['n'])][0] += float(r['n_actual']); size_sum[(src, r['n'])][1] += 1
    if r['method'] == 'R5' and r.get('option', '').startswith('Exact'):
        exact_seen[(r['task'], r['outer'])] = True
    if r['predictor'] in ('ridge', 'gbt'):
        s = sums[(src, r['task'], r['n'], int(r['k']), r['method'], r['predictor'])]
        s[0] += float(r['sse']); s[1] += float(r['c0'])
ev_by_cell = defaultdict(list)
for (src, task, n, k, m, p), (sse, c0) in sums.items():
    ev_by_cell[(src, p, n, k, m)].append(1 - sse / c0)
mean = lambda x: sum(x) / len(x)  # noqa: E731
cells = {}
for (src, p, n, k, m), vals in ev_by_cell.items():
    if m == 'R5' and k in BUDGETS:
        r5, d, rn = mean(vals), mean(ev_by_cell[(src, p, n, k, 'decorr')]), mean(ev_by_cell[(src, p, n, k, 'random')])
        total, count = size_sum[(src, n)]
        cells[(src, p, n, k)] = {'r5': r5, 'decorr': d, 'random': rn, 'lead': r5 - max(d, rn), 'ratio': total / count / k, 'size': total / count, 'class': label(r5, max(d, rn))}

end = json.loads((SUM / 'endpoints.json').read_text(encoding='utf-8'))
print('== V, recomputed ==')
print('jobs', len(jobs), '| jobs with no Exact row after the fix', sum(1 for j in jobs if not exact_seen[j]))
passed, informative = [], []
for s in ['E1', 'E2', 'E3', 'E4']:
    allc = mean(ev_by_cell[(s, 'ridge', 'all', max(k for (a, b, c, k, d) in ev_by_cell if a == s and d == 'all'), 'all')])
    r10 = [c for (a, p, n, k), c in cells.items() if a == s and p == 'ridge' and c['ratio'] >= 10]
    t10 = [c for (a, p, n, k), c in cells.items() if a == s and p == 'gbt' and c['ratio'] >= 10 and c['class'] != 'no signal']
    r5 = [c for (a, p, n, k), c in cells.items() if a == s and p == 'ridge' and c['ratio'] < 5]
    ahead = lambda x: sum(c['class'] == 'target-aware' for c in x)  # noqa: E731
    v1 = ahead(r10) / len(r10) >= .8 and not any(c['class'] == 'reusable' for c in r10)
    v2 = ahead(t10) / len(t10) >= .8
    v3 = ahead(r5) / len(r5) <= .25
    print(f'{s}: all candidates {allc:.4f} | V1 {ahead(r10)}/{len(r10)} {v1} | V2 {ahead(t10)}/{len(t10)} {v2} | V3 {ahead(r5)}/{len(r5)} {v3} | pass {v1 and v2 and v3}')
    if allc >= .10:
        informative.append(s)
        if v1 and v2 and v3:
            passed.append(s)
    e = end['V']['sources'][s]
    same(f'{s} all candidates', allc, e['all_candidates_ridge_largest_n'], 6e-5)
    same(f'{s} V1', ahead(r10), e['V1_ridge_ratio_10_plus']['target_aware'], 0); same(f'{s} V2', ahead(t10), e['V2_trees_ratio_10_plus']['target_aware'], 0); same(f'{s} V3', ahead(r5), e['V3_ridge_ratio_below_5']['target_aware'], 0)
    same(f'{s} pass', v1 and v2 and v3, e['pass'], 0)
v_pass = len(informative) >= 3 and len(passed) >= 3
print('informative', informative, '| passed', passed, '| V pass', v_pass, '| at n/k >= 10:', sum(c['class'] == 'target-aware' for c in cells.values() if c['ratio'] >= 10), 'of', sum(c['ratio'] >= 10 for c in cells.values()))
same('V pass', v_pass, end['V']['pass'], 0)

with (SUM / 'v_cells.csv').open(encoding='utf-8', newline='') as h:
    for r in csv.DictReader(h):
        c = cells[(r['source'], r['predictor'], r['n'], int(r['k']))]
        same(f"cell {r['source']} {r['predictor']} {r['n']} {r['k']}", c['lead'], float(r['lead']), 1e-9)
        if c['class'] != r['class']:
            failures.append(f"class {r['source']} {r['predictor']} {r['n']} {r['k']}")
for p in ('ridge', 'gbt'):
    h = [c for (a, q, n, k), c in cells.items() if q == p]
    h2 = [(k, c) for (a, q, n, k), c in cells.items() if q == p]
    acc = {'n/k >= 10': mean([(c['ratio'] >= 10) == (c['class'] == 'target-aware') for c in h]),
           'n >= 100': mean([(c['size'] >= 100) == (c['class'] == 'target-aware') for c in h]),
           'k <= 10': mean([(k <= 10) == (c['class'] == 'target-aware') for k, c in h2])}
    print(p, 'rule accuracy', {a: round(b, 3) for a, b in acc.items()})

# ---- W ----
print('\n== W, recomputed ==')
w = defaultdict(lambda: [0., 0.])
for r in rows_of('W'):
    if r['predictor'] == 'ridge':
        s = w[(r['source'], r['replicate'], int(r['size']), int(r['k']), r['method'])]
        s[0] += float(r['sse']); s[1] += float(r['c0'])
leads = defaultdict(list)
for (src, rep, size, k, m) in w:
    if m == 'R5':
        e = {a: 1 - w[(src, rep, size, k, a)][0] / w[(src, rep, size, k, a)][1] for a in PANELS}
        leads[(src, size, k)].append(e['R5'] - max(e['decorr'], e['random']))
support = 0
for src in sorted({a for a, _, _ in leads}):
    top = max(s for a, s, _ in leads if a == src)
    d5 = mean(leads[(src, 9, 5)]) - mean(leads[(src, top, 5)]); d10 = mean(leads[(src, 9, 10)]) - mean(leads[(src, top, 10)])
    yes = d5 >= .01 and d10 >= .01
    support += yes
    print(f'{src}: largest {top} | drop at k = 5 {d5:.4f} | at k = 10 {d10:.4f} | supports {yes}')
    e = end['W']['sources'][src]
    same(f'W {src} k5', d5, e['lead_drop_k5'], 6e-5); same(f'W {src} k10', d10, e['lead_drop_k10'], 6e-5)
print('supporting', support, 'of 6 | W pass', support >= 4)
same('W pass', support >= 4, end['W']['pass'], 0)

# ---- R ----
print('\n== R break-even, recomputed against summary ==')
g = defaultdict(lambda: [0., 0.])
for r in rows_of('R'):
    s = g[(r['source'], r['task'], r['n'], r['k'], r['predictor'], r['method'], r['set'])]
    s[0] += float(r['sse']); s[1] += float(r['c0'])
per = defaultdict(list)
for (src, task, n, k, p, m, st) in list(g):
    if m != 'R5_A' or st != 'A':
        continue
    def mix(me, f):  # noqa: E301
        a, b = g[(src, task, n, k, p, me, 'A')], g[(src, task, n, k, p, me, 'B')]
        return 1 - ((1 - f) * a[0] + f * b[0]) / ((1 - f) * a[1] + f * b[1])
    be = next((i / 100 for i in range(101) if mix('R5_A', i / 100) - mix('decorr', i / 100) <= 0), 1.0)
    per[(src, n, k, p)].append(be)
with (SUM / 'r_summary.csv').open(encoding='utf-8', newline='') as h:
    n_r = 0
    for r in csv.DictReader(h):
        same(f"R {r['source']} {r['n']} {r['k']} {r['predictor']}", mean(per[(r['source'], r['n'], r['k'], r['predictor'])]), float(r['break_even_vs_decorr']), 6e-4)
        n_r += 1
print('blocks compared', n_r)

# ---- D ----
print('\n== D, recomputed against summary ==')
d = defaultdict(lambda: [0., 0.])
for r in rows_of('V'):
    if r['n'] == '100' and r['method'] in PANELS and r['predictor'] in ('ridge', 'gbt'):
        s = d[(r['task'].split('_')[1], r['task'], 'all', int(r['k']), r['method'], r['predictor'])]
        s[0] += float(r['sse']); s[1] += float(r['c0'])
for r in rows_of('U'):
    if r['method'] in PANELS and r['predictor'] in ('ridge', 'gbt'):
        src = 'Beat AML' if r['task'].startswith('BEAT') else 'PKIS2' if r['task'].startswith('KIN') else LEGACY[r['task']]
        s = d[(src, r['task'], r['outer'] if src in ('M', 'A1', 'A2', 'A3') else 'all', int(r['k']), r['method'], r['predictor'])]
        s[0] += float(r['sse']); s[1] += float(r['c0'])
dm = defaultdict(list)
for (src, task, o, k, m, p), (sse, c0) in d.items():
    dm[(src, p, k, m)].append(1 - sse / c0)
with (SUM / 'd_equivalent.csv').open(encoding='utf-8', newline='') as h:
    n_d = 0
    for r in csv.DictReader(h):
        src, p, k0 = r['source'], r['predictor'], int(r['R5_k'])
        target = mean(dm[(src, p, k0, 'R5')])
        ks = sorted(k for a, q, k, m in dm if a == src and q == p and m == 'decorr')
        hit = next((k for k in ks if mean(dm[(src, p, k, 'decorr')]) >= target), '>30')
        if str(hit) != r['smallest_decorr']:
            failures.append(f'D {src} {p} {k0}: recomputed {hit}, summary {r["smallest_decorr"]}')
        n_d += 1
print('rows compared', n_d)


# ---- against the first run ----
def table(folder, name):
    with (folder / name).open(encoding='utf-8', newline='') as h:
        return list(csv.DictReader(h))


print('\n== changes against the first run ==')
keys = ('source', 'predictor', 'n', 'k')
old = {tuple(r[a] for a in keys): r for r in table(OLD, 'v_cells.csv')}
changed, largest = [], 0.
for r in table(SUM, 'v_cells.csv'):
    o = old[tuple(r[a] for a in keys)]
    largest = max(largest, abs(float(r['lead']) - float(o['lead'])))
    if r['class'] != o['class']:
        changed.append((*[r[a] for a in keys], o['class'], '->', r['class'], round(float(o['lead']), 3), '->', round(float(r['lead']), 3)))
print('V cells', len(old), '| largest change in lead', round(largest, 4), '| class changes', len(changed))
for c in changed:
    print('  ', *c)
for name, ks in [('w_curve.csv', ('source', 'size', 'k')), ('c_curve.csv', ('source', 'predictor', 'size', 'k'))]:
    a, b = table(SUM, name), {tuple(r[x] for x in ks): r for r in table(OLD, name)}
    print(name, 'rows', len(a), '| rows changed', sum(r != b[tuple(r[x] for x in ks)] for r in a))
ks = ('source', 'n', 'k', 'predictor')
a, b = table(SUM, 'r_summary.csv'), {tuple(r[x] for x in ks): r for r in table(OLD, 'r_summary.csv')}
diff = [(abs(float(r['break_even_vs_decorr']) - float(b[tuple(r[x] for x in ks)]['break_even_vs_decorr'])), *[r[x] for x in ks]) for r in a]
print('r_summary rows', len(a), '| break-even changed', sum(x[0] > 0 for x in diff), '| largest', max(diff))
ks = ('source', 'predictor', 'R5_k')
a, b = table(SUM, 'd_equivalent.csv'), {tuple(r[x] for x in ks): r for r in table(OLD, 'd_equivalent.csv')}
for r in a:
    o = b[tuple(r[x] for x in ks)]
    if r['smallest_decorr'] != o['smallest_decorr'] or r['smallest_better_reusable'] != o['smallest_better_reusable']:
        print('  D changed', *[r[x] for x in ks], 'decorr', o['smallest_decorr'], '->', r['smallest_decorr'], '| better', o['smallest_better_reusable'], '->', r['smallest_better_reusable'])

print('\nVERIFY', 'PASSED' if not failures else 'FAILED', '| differences from the summary:', len(failures))
for f in failures[:40]:
    print('  ', f)
