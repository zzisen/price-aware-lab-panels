#!/usr/bin/env python3
"""Task 14 (plan_r5_m2 FROZEN, sections 3 and 4): build the second-stage tasks.
A: NHANES cohorts not used in any R5 analysis. B: PKIS2 kinase tasks, same structure as the Beat AML main tasks.
Prints task definitions, sample sizes, missing counts and each kinase's own hit count only; no relation between kinases,
or between measurements and targets, is computed. Writes tasks.json (field "set": nhanes / kinase), matrices/ and
kin_chemotypes.csv. Run with .env312:  python -I -X utf8 build_m2.py"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
EXT = W / 'external_data'
NH_FILE = {'LBXGH': 'GHB', 'LBDHDD': 'HDL', 'LBXHSCRP': 'HSCRP', 'LBXVIDMS': 'VID', 'LBXBPB': 'PBCD', 'LBXBCD': 'PBCD', 'LBXTHG': 'PBCD'}
NH_FAM = {'LBXGH': 'metabolic', 'LBDHDD': 'metabolic', 'LBXHSCRP': 'inflammation', 'LBXVIDMS': 'nutrition', 'LBXBPB': 'exposure', 'LBXBCD': 'exposure', 'LBXTHG': 'exposure'}
POOL, MIN_MEMBERS, MIN_FAMS, HIT = 80, 3, 4, 50


def save(name, x, y):
    frame = pd.concat([x.add_prefix('cand_'), y], axis=1)
    frame.index.name = 'unit'
    frame.reset_index().to_csv(HERE / 'matrices' / f'{name}.csv', index=False)
    return len(frame)


def nhanes(name, folder, sfx, code, cands, exclude=()):
    read = lambda f: pd.read_sas(EXT / folder / f'{f}_{sfx}.xpt', format='xport').set_index('SEQN')  # noqa: E731
    targets = [k for k in NH_FILE if (EXT / folder / f'{NH_FILE[k]}_{sfx}.xpt').exists()]
    demo = read('DEMO')[['RIDAGEYR', 'RIAGENDR']]
    cbc, bio = read('CBC'), read('BIOPRO')
    tg = pd.concat([read(NH_FILE[k])[[k]] for k in targets], axis=1)
    adults = demo[demo.RIDAGEYR >= 20]
    f = adults.join(cbc[[c for c in cands if c in cbc]], how='inner').join(bio[[c for c in cands if c in bio]], how='inner').join(tg, how='inner').dropna()
    f = f[~f.index.isin(exclude)]
    pop = f[['RIDAGEYR', 'RIAGENDR']]
    f = f.drop(columns=['RIDAGEYR', 'RIAGENDR'])
    assert sum(c in f for c in cands) == 44
    for c in f:
        pos = f[c][f[c] > 0]
        f[c] = np.log(f[c] + pos.min() / 2)
    f.index = f.index.astype(int).astype(str)
    n = save(name, f[cands], f[targets])
    print(f'{name}: adults {len(adults)} | complete {n} | women {(pop.RIAGENDR == 2).mean():.3f} | age {pop.RIDAGEYR.min():.0f}-{pop.RIDAGEYR.max():.0f} | age>=50 {(pop.RIDAGEYR >= 50).mean():.3f} | candidates 44 | targets {len(targets)}')
    return {'task': name, 'targets': targets, 'families': [NH_FAM[k] for k in targets], 'code': code, 'set': 'nhanes'}


def kinase_families():
    """Top-level headings of UniProt pkinfam.txt, human genes only."""
    fam, cur, lines = {}, None, (EXT / 'uniprot_pkinfam' / 'pkinfam.txt').read_text(encoding='utf-8').splitlines()
    for i, ln in enumerate(lines):
        if i + 1 < len(lines) and set(lines[i + 1].strip()) == {'='} and ln.strip():
            cur = ln.strip()
            continue
        m = re.match(r'^(\S+)\s+\S+_HUMAN\s', ln)
        if cur and m:
            fam.setdefault(m.group(1), cur)
    return fam


def kinase():
    t = pd.read_excel(EXT / 'pkis2' / 'pkis2_S4_Table.xlsx', sheet_name=0)
    print('PKIS2: table rows', len(t), '| rows without a compound ID, dropped', int(t.Compound.isna().sum()))
    t = t[t.Compound.notna()].reset_index(drop=True)
    assert t.Compound.is_unique
    meta = ['Regno', 'Compound', 'Chemotype', 'Smiles', '>90', '>80', '>70']
    assert list(t.columns[:7]) == meta
    names = list(t.columns[7:])
    print('  compounds', len(t), '| assays', len(names), '| chemotypes', t.Chemotype.nunique(), '| missing chemotype', int(t.Chemotype.isna().sum()))
    x = t[names].apply(pd.to_numeric, errors='coerce'); x.index = t.Compound.astype(str)
    print('  missing values', int(x.isna().sum().sum()), '| assays with any missing', int(x.isna().any().sum()))
    fam = kinase_families()
    wild = [c for c in names if not re.search(r'[(,][A-Z]\d+[A-Z]\)', c)]
    print('  dropped as mutant', [c for c in names if c not in wild])
    base = {c: re.split(r'[(-]', c)[0] for c in wild}
    hits = (x[wild] >= HIT).sum()
    matched = [c for c in wild if base[c] in fam]
    print('  matched to pkinfam', len(matched), '| not matched', len(wild) - len(matched))
    best = {}
    for c in sorted(matched):
        b = base[c]
        if b not in best or hits[c] > hits[best[b]]:
            best[b] = c
    print('  one assay per gene', len(best), '| merged', sorted(c for c in matched if best[base[c]] != c))
    order = sorted(best.values(), key=lambda c: (-hits[c], c))
    size = POOL
    while True:
        pool = order[:size]
        members = {}
        for c in pool:
            members.setdefault(fam[base[c]], []).append(c)
        elig = sorted(g for g, m in members.items() if len(m) >= MIN_MEMBERS)
        if len(elig) >= MIN_FAMS or size >= len(order):
            break
        size += 20
    print('  kinase pool', len(pool), '| hits (>=50% inhibition) lowest', int(hits[pool[-1]]), 'next', int(hits[order[len(pool)]]), '| eligible families', {g: len(members[g]) for g in elig}, '| not eligible', {g: len(m) for g, m in members.items() if g not in elig})
    y = x[pool]
    keep = y.notna().all(axis=1)
    y = y[keep]
    print('  compounds complete on the pool', len(y))
    chem = pd.DataFrame({'unit': t.Compound.astype(str), 'chemotype': t.Chemotype}).set_index('unit').loc[y.index]
    sizes = chem.chemotype.value_counts(dropna=False)
    print('  chemotypes among them', len(sizes), '| largest', sizes.head(5).to_dict(), '| share of largest', round(sizes.iloc[0] / len(y), 3), '| singletons', int((sizes == 1).sum()))
    chem.reset_index().to_csv(HERE / 'kin_chemotypes.csv', index=False)
    tasks = []
    for t_ in range(6):
        rng = np.random.default_rng(20261110 + t_)
        fams = list(rng.choice(elig, 3, replace=False))
        targets = [str(i) for g in fams for i in rng.choice(sorted(members[g]), 3, replace=False)]
        cands = [c for c in pool if c not in targets]
        name = f'KIN_T{t_ + 1}'
        n = save(name, y[cands], y[targets])
        tasks.append({'task': name, 'targets': targets, 'families': [g for g in fams for _ in range(3)], 'code': 1410 + t_, 'set': 'kinase'})
        print(' ', name, 'units', n, 'candidates', len(cands), '| families', [g.split(' ')[0] for g in fams], '| targets', targets)
    print('  duplicate tasks:', len({tuple(sorted(a['targets'])) for a in tasks}) < len(tasks))
    return tasks


def main():
    (HERE / 'matrices').mkdir(exist_ok=True)
    m = pd.read_csv(W / 'task13_confirm' / 'matrices' / 'NHANES_I.csv')
    cands = [c[5:] for c in m.columns if c.startswith('cand_')]
    used = set(m.unit.astype(float))
    tasks = [nhanes('NH_A1_I', 'nhanes_2015_2016', 'I', 1400, cands, used),
             nhanes('NH_A2_H', 'nhanes_2013_2014', 'H', 1401, cands),
             nhanes('NH_A3_G', 'nhanes_2011_2012', 'G', 1402, cands)] + kinase()
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
