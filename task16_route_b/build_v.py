#!/usr/bin/env python3
"""Task 16 (plan_route_b_final FROZEN, sections 3, 5, 6, 7): build the pools and all task definitions.
Prints task definitions, sample sizes and missing counts only; each pool rule looks at a measurement's own coverage, variance or
median level, never at relations between measurements. Writes pools/*.csv, pools/E1_groups.csv and tasks.json.
Run with .env312:  python -I -X utf8 build_v.py"""
import gzip
import io
import json
import re
import sys
import tarfile
import warnings
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import rdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rppa_families import FAMILY as RPPA_FAMILY  # noqa: E402

W = HERE.parent
EXT = W / 'external_data'
POOLS = HERE / 'pools'
NOT_FAMILY = {'Other', 'other', 'unannotated'}


def save_pool(name, x, fam):
    x = x.copy(); x.index.name = 'unit'
    x.reset_index().to_csv(POOLS / f'{name}.csv', index=False)
    print(f'{name}: units {len(x)} | measurements {x.shape[1]} | missing {int(x.isna().sum().sum())} | families {pd.Series({c: fam.get(c, "unannotated") for c in x}).value_counts().to_dict()}')
    return x, fam


def e1():
    z = zipfile.ZipFile(EXT / 'fpmtb_malani2022' / 'Functional_Precision_Medicine_Tumor_Board_AML.zip')
    root = 'Functional_Precision_Medicine_Tumor_Board_AML/'
    d = pd.read_excel(io.BytesIO(z.read(root + 'File_3.1_Drug_response_sDSS_164S_17Healthy.xlsx'))).set_index('Drug_ID')
    names = d.pop('Drug_name')
    d = d[[c for c in d if c.startswith('AML_')]].apply(pd.to_numeric, errors='coerce').T
    cov, var = d.notna().mean(), d.var()
    order = sorted(d.columns, key=lambda c: (-cov[c], -var[c], c))     # DEVIATIONS: 151 drugs tie at full coverage, broken by own variance
    top = order[:80]
    print('E1: boundary coverage', round(cov[top[-1]], 3), '| drugs at full coverage', int((cov == 1).sum()))
    x = d[top].dropna()
    lib = pd.read_excel(io.BytesIO(z.read(root + 'File_2_Drug_library_515D.xlsx'))).set_index('FIMM batch ID')
    fam = {c: lib.Drug_Sub_Class[c] for c in top if c in lib.index and isinstance(lib.Drug_Sub_Class[c], str)}
    ann = pd.read_excel(io.BytesIO(z.read(root + 'File_0_Common_sample_annotation_252S.xlsx'))).set_index('Sample_ID')
    g = ann.Patient_ID.loc[x.index]
    print('E1: samples', len(x), '| patients', g.nunique())
    g.rename('group').rename_axis('unit').reset_index().to_csv(POOLS / 'E1_groups.csv', index=False)
    return save_pool('E1', x, fam)


def e2():
    out = EXT / 'pace_dietrich2018' / 'extracted'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lp = rdata.conversion.convert(rdata.parser.parse_file(out / 'lpdAll.RData'), default_encoding='utf-8')['lpdAll']
        dr = rdata.conversion.convert(rdata.parser.parse_file(out / 'drugs.RData'), default_encoding='utf-8')['drugs']
    ex = lp.assayData.maps[0]['exprs']
    vals = pd.DataFrame(ex.values, index=[str(i) for i in ex.coords[ex.dims[0]].values], columns=[str(i) for i in ex.coords[ex.dims[1]].values])
    fd = lp.featureData.data
    fd.index = [str(i) for i in fd.index]
    v = fd[(fd.type == 'viab') & fd.subtype.astype(str).isin(['1', '2', '3', '4', '5'])]
    x = vals.loc[v.index].astype(float).groupby(v.id.astype(str).values).mean().T
    diag = lp.phenoData.data.Diagnosis
    diag.index = [str(i) for i in diag.index]
    keep = [c for c in x.index if diag[c] != 'hMNC']                  # hMNC samples are healthy donors; the plan uses blood cancers
    print('E2: samples', len(x), '| excluded healthy hMNC', len(x) - len(keep), '| diagnoses', diag[keep].value_counts().head(5).to_dict())
    x = x.loc[keep].dropna()
    dr.index = [str(i) for i in dr.index]
    fam = {c: dr.target_category[c] for c in x if c in dr.index and isinstance(dr.target_category[c], str)}
    return save_pool('E2', x, fam)


def e3():
    x = pd.read_csv(EXT / 'tcga_rppa_xena' / 'TCGA-RPPA-pancan-clean.xena.gz', sep='\t', index_col=0, na_values=['NA', 'NaN', '']).T
    assert all(c in RPPA_FAMILY for c in x.columns), [c for c in x.columns if c not in RPPA_FAMILY]
    p = pd.read_csv(EXT / 'tcga_rppa_xena' / 'Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', usecols=['sample', 'cancer type abbreviation']).set_index('sample')
    ids = [s for s in x.index if s.endswith('-01') and s in p.index and p.loc[s, 'cancer type abbreviation'] == 'BRCA']
    x = x.loc[ids]
    done = x.columns[x.notna().all()]
    print('E3: BRCA primary tumours', len(x), '| proteins', x.shape[1], '| proteins with no missing', len(done))
    var = x[done].var()
    top = sorted(done, key=lambda c: (-var[c], c))[:80]
    return save_pool('E3', x[top], {c: RPPA_FAMILY[c] for c in top})


def e4():
    h = pd.read_csv(EXT / 'hgnc' / 'hgnc_complete_set.txt', sep='\t', low_memory=False, usecols=['symbol', 'locus_group', 'gene_group', 'ensembl_gene_id'])
    h = h[(h.locus_group == 'protein-coding gene') & h.gene_group.notna() & h.ensembl_gene_id.notna()].set_index('ensembl_gene_id')
    t = pd.read_csv(EXT / 'gtex_v8_blood' / 'gene_tpm_2017-06-05_v8_whole_blood.gct.gz', sep='\t', skiprows=2)
    t['ens'] = t.Name.str.split('.').str[0]
    t = t[t.ens.isin(h.index)].drop_duplicates('ens').set_index('ens').drop(columns=['id', 'Name', 'Description'])
    print('E4: samples', t.shape[1], '| protein-coding genes with a gene group', len(t))
    expressed = t.index[t.median(axis=1) >= 1]
    members = {}
    for e in expressed:
        for g in str(h.gene_group[e]).split('|'):
            members.setdefault(g, []).append(e)
    elig = sorted(g for g, m in members.items() if len(m) >= 10)
    print('E4: expressed genes', len(expressed), '| groups with at least 10', len(elig))
    rng = np.random.default_rng(20261169)
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
    return save_pool('E4', x, fam)


def beat():
    d = EXT / 'beataml2'
    m = pd.read_csv(W / 'task13_confirm' / 'matrices' / 'BEAT_T1.csv').set_index('unit')
    m.columns = [c[5:] if c.startswith('cand_') else c for c in m.columns]
    dg = pd.read_excel(d / 'beataml_drug_families.xlsx', sheet_name='drug_gene')
    dg = dg[dg.inhibitor.isin(m.columns)].dropna(subset=['family']).reset_index(drop=True)
    prim = dg.sort_values(['inhibitor', 'db_min_kd'], kind='stable').drop_duplicates('inhibitor').set_index('inhibitor').family
    return save_pool('BEAT', m[sorted(m.columns)], {c: prim[c] for c in m.columns if c in prim.index})


def kin():
    sys.path.insert(0, str(W / 'task14_m2'))
    import build_m2
    fams = build_m2.kinase_families()
    m = pd.read_csv(W / 'task14_m2' / 'matrices' / 'KIN_T1.csv').set_index('unit')
    m.columns = [c[5:] if c.startswith('cand_') else c for c in m.columns]
    fam = {c: fams[re.split(r'[(-]', c)[0]] for c in m.columns if re.split(r'[(-]', c)[0] in fams}
    return save_pool('KIN', m[sorted(m.columns)], fam)


def members_of(names, fam, skip=()):
    out = {}
    for c in names:
        if c not in skip and fam.get(c) is not None and fam[c] not in NOT_FAMILY:
            out.setdefault(fam[c], []).append(c)
    return {g: m for g, m in out.items() if len(m) >= 3}


def three_by_three(rng, names, fam, skip=()):
    mem = members_of(names, fam, skip)
    fams = list(rng.choice(sorted(mem), 3, replace=False))
    return [str(i) for g in fams for i in rng.choice(sorted(mem[g]), 3, replace=False)]


def main():
    POOLS.mkdir(exist_ok=True)
    pools = {'E1': e1(), 'E2': e2(), 'E3': e3(), 'E4': e4(), 'BEAT': beat(), 'KIN': kin()}
    idx = {'E1': 1, 'E2': 2, 'E3': 3, 'E4': 4, 'BEAT': 5, 'KIN': 6}
    split = {'E1': 'group', 'E2': 'random', 'E3': 'random', 'E4': 'random', 'BEAT': 'random', 'KIN': 'chemotype'}
    tasks = []

    def add(a, s, name, code, targets, cands, **kw):
        fam = pools[s][1]
        assert not set(targets) & set(cands) and len(set(cands)) == len(cands)
        tasks.append({'analysis': a, 'source': s, 'task': name, 'code': code, 'split': split[s], 'targets': targets, 'families': [str(fam.get(c, 'unannotated')) for c in targets],
                      'candidates': cands, **kw})

    for s, (x, fam) in pools.items():
        i = idx[s]; base = 20261120 + i * 10; names = sorted(x.columns)
        print(s, 'eligible families', {g: len(m) for g, m in members_of(names, fam).items()})
        if s not in ('BEAT', 'KIN'):
            for t in range(6):
                tg = three_by_three(np.random.default_rng(base + t), names, fam)
                add('V', s, f'V_{s}_T{t + 1}', 1600 + i * 10 + t, tg, [c for c in names if c not in tg])
        n_cand, sizes = (30, [3, 9, 20, 30]) if s == 'E2' else (40, [3, 9, 20, 40])
        for t in range(4):
            perm = [str(c) for c in np.random.default_rng(base + 100 + t).permutation(names)]
            cands, rest = perm[:n_cand], perm[n_cand:]
            for size in sizes:
                add('W', s, f'W_{s}_T{t + 1}_t{size}', 1700 + i * 10 + t, rest[:size], cands, no_exact=size >= 30, size=size, replicate=t)
        for t in range(4):
            rng = np.random.default_rng(base + 200 + t)
            tg = three_by_three(rng, names, fam)
            rest = [str(c) for c in rng.permutation([c for c in names if c not in tg])]
            for size in [20, 40, len(rest)]:
                add('C', s, f'C_{s}_T{t + 1}_c{size}', 1800 + i * 10 + t, tg, rest[:size], size=size, replicate=t)
        for t in range(6):
            rng = np.random.default_rng(base + 300 + t)
            a = three_by_three(rng, names, fam)
            b = three_by_three(rng, names, fam, skip=a)
            add('R', s, f'R_{s}_T{t + 1}', 1900 + i * 10 + t, a, [c for c in names if c not in a + b], targets_b=b, families_b=[str(fam[c]) for c in b])
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding='utf-8')
    c = pd.DataFrame(tasks).groupby(['analysis', 'source']).agg(tasks=('task', 'size'), candidates=('candidates', lambda v: sorted({len(i) for i in v})), targets=('targets', lambda v: sorted({len(i) for i in v})))
    print(c.to_string())
    for t in tasks:
        if t['analysis'] == 'V':
            print(' ', t['task'], '| families', sorted(set(t['families'])), '| targets', t['targets'])


if __name__ == '__main__':
    main()
