#!/usr/bin/env python3
"""Heterogeneous-family development tasks (plan_hetero.md). Families = drugs sharing a molecular target (compound_match
TARGET); 6 families x 3 targets per task, families chosen for high within / low between correlation.
Writes hetero/tasks.json and hetero/matrices/<task>.csv. Run with .env312."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
EXT = W / 'external_data' / 'pharmacodb'
T5 = W / 'task5_validation'
OUT = HERE / 'hetero'
COVERAGE, MIN_ROWS, FAMS, MIN_FAMS, BETWEEN = 0.95, 100, 4, 3, 0.3      # FAMS 6 -> 4: see plan_hetero.md deviation


def main():
    (OUT / 'matrices').mkdir(parents=True, exist_ok=True)
    long = pd.read_csv(EXT / 'experiments_long.csv').dropna(subset=['AAC'])
    mapping = pd.read_csv(T5 / 'cell_line_mapping.csv').dropna(subset=['unit'])
    long = long.merge(mapping, on=['dataset', 'cell_line'])
    cm = pd.read_csv(EXT / 'compound_match.csv').dropna(subset=['pharmacodb_name', 'TARGET'])
    tgt = cm.drop_duplicates('pharmacodb_name').set_index('pharmacodb_name').TARGET
    tasks, code = [], 1200
    for ds in ['GDSC1', 'GDSC2', 'PRISM']:
        w = long[long.dataset == ds].pivot_table(index='unit', columns='compound', values='AAC', aggfunc='mean')
        w = w[w.columns[w.notna().mean() >= 0.5]]                    # drugs tested on at least half the lines
        w = w[w.notna().mean(axis=1) >= COVERAGE]                     # lines that have nearly all of them
        w = w[w.columns[w.notna().mean() >= COVERAGE]].dropna()
        c = w.corr()
        groups = {}
        for d in w.columns:
            if d in tgt.index:
                groups.setdefault(tgt[d], []).append(d)
        trios = {}
        for g, ds_ in groups.items():
            if len(ds_) < 3:
                continue
            best = max(((a, b, e) for i, a in enumerate(ds_) for j, b in enumerate(ds_[i + 1:], i + 1) for e in ds_[j + 1:]),
                       key=lambda t: c.loc[list(t), list(t)].values[np.triu_indices(3, 1)].mean())
            trios[g] = (list(best), float(c.loc[list(best), list(best)].values[np.triu_indices(3, 1)].mean()))
        order = sorted(trios, key=lambda g: -trios[g][1])
        print(ds, len(w), 'units', w.shape[1], 'drugs;', len(trios), 'target groups with >=3 drugs')
        used = set()
        for t in range(2):
            pick = []
            for g in order:
                if g in used:
                    continue
                if all(c.loc[trios[g][0], trios[h][0]].values.mean() <= BETWEEN for h in pick):
                    pick.append(g)
                if len(pick) == FAMS:
                    break
            if len(pick) < MIN_FAMS:
                print('  only', len(pick), 'families:', pick)
                break
            used |= set(pick)
            targets = [d for g in pick for d in trios[g][0]]
            fams = [g for g in pick for _ in range(3)]
            assert len(set(fams)) >= MIN_FAMS
            cands = [d for d in w.columns if d not in targets]
            frame = pd.concat([w[cands].add_prefix('cand_'), w[targets]], axis=1)
            if len(frame) < MIN_ROWS:
                break
            name = f'HET_{ds}_T{t + 1}'
            frame.reset_index().to_csv(OUT / 'matrices' / f'{name}.csv', index=False)
            within = np.mean([trios[g][1] for g in pick])
            between = np.mean([c.loc[trios[a][0], trios[b][0]].values.mean() for i, a in enumerate(pick) for b in pick[i + 1:]])
            tasks.append({'task': name, 'targets': targets, 'families': fams, 'code': code, 'units': len(frame),
                          'candidates': len(cands), 'within_r': round(float(within), 3), 'between_r': round(float(between), 3)})
            print(' ', name, pick, 'within', round(within, 2), 'between', round(between, 2), 'cands', len(cands))
            code += 1
    (OUT / 'tasks.json').write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
