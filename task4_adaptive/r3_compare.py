#!/usr/bin/env python3
"""Task 4b: R3 against R1 with the replacement criterion of plan_r3.md. Run with .env312 after r3_rules.py and radbill_r3.py."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
TAGS = [('gdsc1_repeated_random', 'GDSC1, cell-line splits'), ('gdsc1_held_out_tissue', 'GDSC1, tissue held out'),
        ('gdsc2_repeated_random', 'GDSC2, cell-line splits'), ('gdsc2_held_out_tissue', 'GDSC2, tissue held out'),
        ('gdsc2_new', 'GDSC2 new lines'), ('prism_repeated_random', 'PRISM, cell-line splits'),
        ('prism_held_out_tissue', 'PRISM, tissue held out')]
FIXED = ['Shared', 'Family', 'Global3', 'Exact']


def table(frame, rules, fixed, cells='n_test_cells'):
    rows = []
    for n_label, h in frame.groupby('n_label', sort=False):
        r = {'n_label': n_label, 'n_mean': h.n_train.mean()}
        for k in fixed + rules:
            r[k] = h[f'loss_sum_{k}'].sum() / h[cells].sum()
        best = min(r[k] for k in fixed)
        for k in fixed + rules:
            r[f'regret_{k}'] = r[k] - best
        for k in rules:
            if f'choice_{k}' in h:
                vc = h[f'choice_{k}'].value_counts(normalize=True)
                r[f'{k}_Shared_share'] = float(vc.get('Shared', vc.get('SHARED', 0.0)))
        rows.append(r)
    return pd.DataFrame(rows).sort_values('n_mean')


def main():
    out, tables = {}, []
    for tag, label in TAGS:
        a = pd.read_csv(HERE / f'{tag}_rules.csv'); b = pd.read_csv(HERE / f'{tag}_r3.csv')
        key = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
        f = a.merge(b.drop(columns=['loss_sum_R1', 'choice_R1', 'n_train', 'n_test_cells']), on=key)
        assert len(f) == len(a)
        f['n_label'] = f.n_label.astype(str)
        t = table(f, ['R1', 'R3', 'R3r', 'R3r1se'], FIXED); t.insert(0, 'analysis', label); tables.append(t)
    rc = pd.read_csv(HERE / 'radbill_rules_cells.csv'); r3 = pd.read_csv(HERE / 'radbill_r3_cells.csv')
    f = r3.merge(rc, on=['outer_participant', 'm_train', 'rep', 'target'])
    assert len(f) == len(r3)
    g = f.groupby(['outer_participant', 'm_train', 'rep'])
    agg = g.size().rename('n_test_cells').reset_index()
    for k, c in {'Shared': 'SHARED', 'Family': 'FAMILY', 'Exact': 'EXACT', 'R1': 'R1', 'R3': 'R3'}.items():
        agg[f'loss_sum_{k}'] = g[c].sum().values
    agg['choice_R3'] = g.choice_R3.first().values
    agg['n_train'] = agg.m_train; agg['n_label'] = agg.m_train.astype(str); agg.loc[agg.rep < 0, 'n_label'] = 'all'
    t = table(agg, ['R1', 'R3'], ['Shared', 'Family', 'Exact']); t.insert(0, 'analysis', 'Radbill pacing (R1 reps)'); tables.append(t)
    allt = pd.concat(tables)
    allt.to_csv(HERE / 'r3_summary.csv', index=False)
    wins = 0
    for label, t in allt.groupby('analysis', sort=False):
        mr = {k: round(float(t[f'regret_{k}'].max()), 4) for k in ['R1', 'R3', 'R3r', 'R3r1se'] if f'regret_{k}' in t and t[f'regret_{k}'].notna().all()}
        mr['best fixed'] = round(min(float(t[f'regret_{k}'].max()) for k in FIXED if f'regret_{k}' in t and t[f'regret_{k}'].notna().all()), 4)
        out[label] = {'max regret': mr, 'R3 < R1': mr['R3'] < mr['R1']}
        wins += mr['R3'] < mr['R1']
    out['criterion'] = {'1 GDSC1 rows R3 < R1': out['GDSC1, cell-line splits']['R3 < R1'], '2 wins out of 8': wins,
                        'R3 replaces R1': bool(out['GDSC1, cell-line splits']['R3 < R1'] and wins >= 6)}
    (HERE / 'r3_criterion.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(allt[[c for c in allt if not c.startswith('regret_')]].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
