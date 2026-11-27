"""Task 5: rule variants with the extrapolation guard (development and learning data only).
Regret is always against the unguarded hindsight-best fixed resolution (the frozen reference)."""
import pickle
import numpy as np
import pandas as pd
import dev_eval as D


def load(an):
    recs = pickle.load(open(D.HERE / 'cache' / f'{an}.pkl', 'rb'))
    if an != 'radbill':
        g = pickle.load(open(D.HERE / 'cache' / f'{an}_guard.pkl', 'rb'))
        assert len(g) == len(recs)
        for r, q in zip(recs, g):
            r.update(q)
    return recs


def guarded_view(rec):
    if 'middle_g' not in rec:
        return rec
    return {**rec, 'middle': rec['middle_g']}


VARIANTS = {}
for c in (0.0, 0.5, 1.0):
    for avg in (False, True):
        for guard in (False, True):
            name = f"{'avg' if avg else ''}SE{c}+C0{' guard' if guard else ''}"
            def f(rec, c=c, avg=avg, guard=guard):
                r = guarded_view(rec) if guard else rec
                u = D.averaged_units(r) if avg else D.units(r)
                return D.se_rule(u, D.candidates(r, True), c), guard
            VARIANTS[name] = f
VARIANTS['R1'] = lambda rec: (D.RULES['R1'](rec), False)
VARIANTS['R3'] = lambda rec: (D.RULES['R3'](rec), False)

rows = []
for an in D.ANALYSES:
    recs = load(an)
    fixed = [k for k in ['Shared', 'Family', 'Global3', 'Exact'] if k in recs[0]['outer']]
    recs_rows = []
    for r in recs:
        row = {'n_label': r['n_label'], 'n_train': r['n_train'], 'cells': r['n_test_cells'], **r['outer'],
               **{f'g_{k}': v for k, v in r.get('outer_g', r['outer']).items()}}
        for name, f in VARIANTS.items():
            key, guard = f(r)
            row[f'rule_{name}'] = (r.get('outer_g', r['outer']) if guard else r['outer'])[key]
        recs_rows.append(row)
    fr = pd.DataFrame(recs_rows)
    for n_label, h in fr.groupby('n_label'):
        risk = {k: h[k].sum() / h.cells.sum() for k in h.columns if k not in ('n_label', 'n_train', 'cells')}
        best = min(risk[k] for k in fixed)
        rows.append({'analysis': an, 'n_label': n_label, **{k: v - best for k, v in risk.items()}})
reg = pd.DataFrame(rows)
mx = reg.groupby('analysis', sort=False).max(numeric_only=True).reindex(D.ANALYSES)
mx = mx.drop(columns=[c for c in mx if c.startswith('g_C0')])
mx.to_csv(D.HERE / 'dev_variants_maxregret.csv')
with pd.option_context('display.width', 300, 'display.max_columns', 40):
    print(mx[[c for c in mx if c.startswith('rule_') or c.startswith('g_') or c in ('Shared',)]].round(3).T)
    print(mx.mean().round(4).sort_values().head(20))
    for c in [c for c in mx if c.startswith('rule_') and c not in ('rule_R1', 'rule_R3')]:
        print(c, 'wins vs R1', int((mx[c] <= mx.rule_R1 + 1e-12).sum()), 'vs R3', int((mx[c] <= mx.rule_R3 + 1e-12).sum()))
