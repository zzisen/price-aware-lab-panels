#!/usr/bin/env python3
"""Task 17 G3 (plan_gates section 5): pools and tasks of the 24 unused sources, by the rules of task16_route_b/build_v.py for E3 and E4.
Each pool rule looks at a measurement's own variance or median level, never at relations between measurements. Prints sizes and missing counts only.
Writes pools/*.csv and tasks_g3.json. Run with .env312:  python -X utf8 build_g3.py"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
EXT = W / 'external_data'
POOLS = HERE / 'pools'
sys.path.insert(0, str(W / 'task16_route_b'))
from rppa_families import FAMILY as RPPA_FAMILY  # noqa: E402

TISSUES = ['muscle_skeletal', 'skin_sun_exposed_lower_leg', 'adipose_subcutaneous', 'artery_tibial', 'thyroid', 'nerve_tibial', 'lung', 'esophagus_mucosa', 'heart_left_ventricle', 'colon_transverse', 'breast_mammary_tissue', 'pancreas']
CANCERS = ['KIRC', 'LGG', 'OV', 'UCEC', 'THCA', 'LUAD', 'STAD', 'COAD', 'PRAD', 'HNSC', 'BLCA', 'LUSC']
NOT_FAMILY = {'Other', 'other', 'unannotated'}


def save(name, x, fam):
    x = x.copy(); x.index.name = 'unit'
    x.reset_index().to_csv(POOLS / f'{name}.csv', index=False)
    print(f'{name}: units {len(x)} | measurements {x.shape[1]} | missing {int(x.isna().sum().sum())} | families with at least 3 {sum(v >= 3 for v in pd.Series(fam).value_counts())}', flush=True)
    return x, fam


def rppa():
    t = pd.read_csv(EXT / 'tcga_rppa_xena' / 'TCGA-RPPA-pancan-clean.xena.gz', sep='\t', index_col=0, na_values=['NA', 'NaN', '']).T
    p = pd.read_csv(EXT / 'tcga_rppa_xena' / 'Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', usecols=['sample', 'cancer type abbreviation']).set_index('sample')
    out = {}
    for c in CANCERS:
        ids = [s for s in t.index if s.endswith('-01') and s in p.index and p.loc[s, 'cancer type abbreviation'] == c]
        x = t.loc[ids]
        done = x.columns[x.notna().all()]
        var = x[done].var()
        top = sorted(done, key=lambda a: (-var[a], a))[:80]
        out[f'R_{c}'] = save(f'R_{c}', x[top], {a: RPPA_FAMILY[a] for a in top})
    return out


def gtex():
    h = pd.read_csv(EXT / 'hgnc' / 'hgnc_complete_set.txt', sep='\t', low_memory=False, usecols=['symbol', 'locus_group', 'gene_group', 'ensembl_gene_id'])
    h = h[(h.locus_group == 'protein-coding gene') & h.gene_group.notna() & h.ensembl_gene_id.notna()].set_index('ensembl_gene_id')
    out = {}
    for i, tissue in enumerate(TISSUES):
        t = pd.read_csv(EXT / 'gtex_v8_tissues' / f'gene_tpm_2017-06-05_v8_{tissue}.gct.gz', sep='\t', skiprows=2)
        t['ens'] = t.Name.str.split('.').str[0]
        t = t[t.ens.isin(h.index)].drop_duplicates('ens').set_index('ens').drop(columns=['id', 'Name', 'Description'])
        expressed = t.index[t.median(axis=1) >= 1]
        members = {}
        for e in expressed:
            for g in str(h.gene_group[e]).split('|'):
                members.setdefault(g, []).append(e)
        elig = sorted(g for g, m in members.items() if len(m) >= 10)
        rng = np.random.default_rng(20261700 + i)
        used, fam, n = set(), {}, 0
        for g in rng.permutation(elig):
            free = sorted(set(members[g]) - used)
            if len(free) < 4:
                continue
            for e in rng.choice(free, 4, replace=False):
                used.add(e); fam[h.symbol[e]] = str(g)
            n += 1
            if n == 20:
                break
        x = np.log2(t.loc[sorted(used)] + 1).T
        x.columns = [h.symbol[e] for e in x.columns]
        out[f'G_{tissue}'] = save(f'G_{tissue}', x, fam)
    return out


def main():
    POOLS.mkdir(exist_ok=True)
    pools = {**rppa(), **gtex()}
    tasks = []
    for i, (s, (x, fam)) in enumerate(pools.items()):
        names = sorted(x.columns)
        mem = {}
        for c in names:
            if fam.get(c) is not None and fam[c] not in NOT_FAMILY:
                mem.setdefault(fam[c], []).append(c)
        mem = {g: m for g, m in mem.items() if len(m) >= 3}
        for t in range(6):
            rng = np.random.default_rng(20262000 + i * 10 + t)
            fams = list(rng.choice(sorted(mem), 3, replace=False))
            tg = [str(c) for g in fams for c in rng.choice(sorted(mem[g]), 3, replace=False)]
            tasks.append({'analysis': 'G3', 'source': s, 'task': f'G3_{s}_T{t + 1}', 'code': 3000 + i * 10 + t, 'split': 'random', 'targets': tg, 'families': [str(fam[c]) for c in tg], 'candidates': [c for c in names if c not in tg]})
    (HERE / 'tasks_g3.json').write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding='utf-8')
    print(pd.DataFrame(tasks).groupby('source').agg(tasks=('task', 'size'), candidates=('candidates', lambda v: sorted({len(i) for i in v}))).to_string())


if __name__ == '__main__':
    main()
