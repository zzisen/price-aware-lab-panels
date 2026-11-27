#!/usr/bin/env python3
"""Task 11 (plan_transfer.md): build the cross-platform transfer tasks.
Writes tasks.json, task_counts.csv and matrices/<task>_A.csv / <task>_B.csv into this folder.
Target selection follows task5_validation/build_tasks.build_tasks; the family map is the one Task 5 used for these
PharmacoDB platforms (compound_match.csv). The plan text named gdsc_pathways(); both maps are printed so the
difference, if any, is on record before anything is run (DEVIATIONS.md). Run with .env312."""
import json
import sys
from itertools import permutations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
sys.path.insert(0, str(WORK / 'task5_validation'))
import build_tasks as BT  # noqa: E402

PAIRS = [('GDSC1', 'PRISM'), ('PRISM', 'GDSC1'), ('GDSC2', 'PRISM'), ('PRISM', 'GDSC2'),
         ('GDSC1', 'CTRPv2'), ('CTRPv2', 'GDSC1')]
SAME_INSTITUTION = [{'GDSC1', 'GDSC2'}, {'CTRPv2', 'PRISM'}]
COVER, MIN_SHARED, MIN_A, MIN_B = 0.90, 20, 100, 80


def family_maps():
    match = pd.read_csv(WORK / 'external_data' / 'pharmacodb' / 'compound_match.csv').dropna(subset=['pharmacodb_name'])
    task5 = match.groupby('pharmacodb_name').TARGET_PATHWAY.agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).to_dict()
    path = BT.gdsc_pathways()
    return task5, path


def pick_targets(drugs, lines, fam):
    """build_tasks rule: per family with >=3 eligible drugs take the top 3 by (lines desc, name); families by
    (third drug's lines desc, family name); the first trio of families gives the 9 targets."""
    el = pd.DataFrame({'drug': drugs, 'lines': [lines[d] for d in drugs], 'family': [fam(d) for d in drugs]})
    el = el[el.family.notna() & ~el.family.isin(BT.NOT_FAMILIES) & ~el.drug.map(BT.base_name).isin(BT.DEV_DRUGS)]
    fams = []
    for f, g in el.groupby('family'):
        if len(g) >= 3:
            top = g.sort_values(['lines', 'drug'], ascending=[False, True]).head(3)
            fams.append((f, top.drug.tolist(), int(top.lines.iloc[2])))
    fams.sort(key=lambda t: (-t[2], t[0]))
    if len(fams) < 3:
        return None, None
    trio = fams[:3]
    return [d for _, ds, _ in trio for d in ds], [f for f, ds, _ in trio for _ in ds]


def main():
    (HERE / 'matrices').mkdir(exist_ok=True)
    long = pd.read_csv(WORK / 'external_data' / 'pharmacodb' / 'experiments_long.csv').dropna(subset=['AAC'])
    mp = pd.read_csv(WORK / 'task5_validation' / 'cell_line_mapping.csv').dropna(subset=['unit'])
    long = long.merge(mp, on=['dataset', 'cell_line'])
    wide = {d: g.pivot_table(index='unit', columns='compound', values='AAC', aggfunc='mean') for d, g in long.groupby('dataset')}
    fam5, fam_path = family_maps()
    tasks, counts = [], []
    for i, (a, b) in enumerate(PAIRS):
        assert {a, b} not in SAME_INSTITUTION
        wa, wb = wide[a], wide[b]
        shared = sorted(c for c in wa.columns.intersection(wb.columns)
                        if wa[c].notna().mean() >= COVER and wb[c].notna().mean() >= COVER)
        lines = wa[shared].notna().sum().to_dict()
        targets, families = pick_targets(shared, lines, lambda d: fam5.get(d))
        alt, _ = pick_targets(shared, lines, lambda d: fam_path.get(BT.base_name(d)))
        name = f'XFER_{a}_to_{b}'
        row = {'task': name, 'train': a, 'test': b, 'shared': len(shared), 'targets_task5_map': targets,
               'targets_plan_text_map': alt, 'same_targets_both_maps': targets == alt}
        if targets is None:
            row.update({'included': False, 'reason': 'no trio of families'}); counts.append(row); continue
        cand = [c for c in shared if c not in targets]
        A = wa[cand + targets].dropna()
        bonly = wb.index.difference(wa.index)
        B = wb.loc[bonly, cand + targets].dropna()
        row.update({'candidates': len(cand), 'A_units': len(A), 'B_only_units': len(B)})
        if len(shared) < MIN_SHARED or len(A) < MIN_A or len(B) < MIN_B:
            row.update({'included': False, 'reason': 'size rule'}); counts.append(row); continue
        assert not set(A.index) & set(B.index)
        for side, m in (('A', A), ('B', B)):
            m = m.copy(); m.columns = [f'cand_{c}' for c in cand] + targets; m.index.name = 'unit'
            m.reset_index().to_csv(HERE / 'matrices' / f'{name}_{side}.csv', index=False)
        row['included'] = True
        counts.append(row)
        tasks.append({'task': name, 'train': a, 'test': b, 'code': 1101 + i, 'targets': targets, 'families': families,
                      'candidates': cand})
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1), encoding='utf-8')
    pd.DataFrame(counts).to_csv(HERE / 'task_counts.csv', index=False)
    for r in counts:
        print(r['task'], 'included' if r.get('included') else f"EXCLUDED ({r.get('reason')})",
              {k: r.get(k) for k in ('shared', 'candidates', 'A_units', 'B_only_units', 'same_targets_both_maps')})
        print('   targets (Task 5 map):', r['targets_task5_map'])
        if not r['same_targets_both_maps']:
            print('   targets (plan-text map):', r['targets_plan_text_map'])
    for t in tasks:
        print(t['task'], 'families', list(dict.fromkeys(t['families'])))


if __name__ == '__main__':
    main()
