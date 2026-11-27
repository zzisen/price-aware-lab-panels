#!/usr/bin/env python3
"""Task 5: build the validation tasks exactly as fixed in plan_validation_suite.md (frozen 2026-10-29 01:35 NZDT).

Task construction uses only drug annotations and per-drug counts of measured lines; response values are only
written into the per-task matrices. Run with .env312 -I. Writes, in this folder:
  depmap_features_all.csv  GDSC-example features for every DepMap 24Q4 model with complete omics (local only)
  tasks.json, task_counts.csv, matrices/<task>.csv
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
EXT = WORK / 'external_data'
sys.path.insert(0, str(WORK / 'task3b_external'))
import build_prism_matrix as BP  # noqa: E402

DEV_DRUGS = {'methotrexate', 'refametinib', 'pd0325901', 'pd-0325901', 'pd-184352', 'ci-1040', 'selumetinib', 'azd6244',
             'trametinib', 'nilotinib', 'axitinib', 'rdea119'}
MIN_LINES, MAX_TASKS = 30, 4
NOT_FAMILIES = {'Other', 'Unclassified', 'other', 'unclassified'}
(HERE / 'matrices').mkdir(exist_ok=True)


def base_name(name):
    """Lower-case name without any parenthetical (addendum 1)."""
    return re.sub(r'\s*\(.*?\)', '', str(name)).strip().lower()


def gdsc_pathways():
    """Name or synonym (lower case, no parenthetical) -> GDSC TARGET_PATHWAY; None when names map to several pathways."""
    g = pd.read_csv(EXT / 'gdsc2' / 'screened_compounds_rel_8.5.csv')
    out = {}
    for _, r in g.iterrows():
        for n in [r.DRUG_NAME] + ([] if pd.isna(r.SYNONYMS) else str(r.SYNONYMS).split(',')):
            k = base_name(n)
            if k:
                out.setdefault(k, set()).add(r.TARGET_PATHWAY)
    return {k: (next(iter(v)) if len(v) == 1 else None) for k, v in out.items()}


def norm(name):
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


def example_features():
    return pd.read_csv(WORK / 'base_v32' / 'inputs' / 'GDSC' / 'exampleGDSC_2.3-1_readable.csv', nrows=1).columns[20:].tolist()


def depmap_features():
    """The Task 3b feature rules applied to every DepMap model with expression, copy number and mutation calls."""
    path = HERE / 'depmap_features_all.csv'
    if path.exists():
        return pd.read_csv(path, index_col=0)
    names = example_features()
    d = EXT / 'depmap24q4'
    expr = pd.read_csv(d / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv', index_col=0)
    cn = pd.read_csv(d / 'OmicsAbsoluteCNGene.csv', index_col=0)
    mut = pd.read_csv(d / 'OmicsSomaticMutations.csv', usecols=['ModelID', 'HugoSymbol', 'VariantInfo'])
    fus = pd.read_csv(d / 'OmicsFusionFiltered.csv', usecols=['ModelID', 'LeftGene', 'RightGene'])
    keep = mut.VariantInfo.fillna('').str.upper().str.contains('|'.join(BP.PROTEIN_ALTERING))
    mutated = mut[keep].groupby('HugoSymbol').ModelID.apply(set).to_dict()
    lines = sorted(set(expr.index) & set(cn.index) & set(mut.ModelID))
    e_sym, c_sym = BP.symbol_table(expr.columns), BP.symbol_table(cn.columns)
    aliases = json.loads((WORK / 'task3b_external' / 'hgnc_cache.json').read_text())
    gene_of = lambda f: f.rsplit('.', 1)[0] if f.endswith(('.CNV', '.MUT')) else f
    cols = {}
    for f in names:
        g = gene_of(f)
        if f == 'MS.HL.MUT':
            continue
        if f in BP.FUSIONS:
            a, b = BP.FUSIONS[f]
            l, r = fus.LeftGene.str.split(' ').str[0], fus.RightGene.str.split(' ').str[0]
            hit = set(fus.ModelID[((l == a) & (r == b)) | ((l == b) & (r == a))])
            cols[f] = [1.0 if m in hit else 0.0 for m in lines]; continue
        if f.endswith('.MUT'):
            use = BP.resolve(BP.SPECIAL.get(f, g), mutated, aliases)
            cols[f] = [1.0 if use and m in mutated[use] else 0.0 for m in lines]; continue
        table, frame = (c_sym, cn) if f.endswith('.CNV') else (e_sym, expr)
        use = BP.resolve(g, table, aliases)
        if use is not None:
            cols[f] = frame.loc[lines, table[use]].to_numpy(float)
    X = pd.DataFrame(cols, index=lines).dropna()
    X.index.name = 'ModelID'
    X.to_csv(path)
    return X


def build_tasks(dataset, y_long, families, X, tissue, code_start):
    """y_long: unit, drug, value. families: drug -> family. Returns the task list and the per-drug counts."""
    y = y_long.groupby(['unit', 'drug']).value.median().unstack()
    y = y.loc[y.index.intersection(X.index)]
    counts = y.notna().sum().rename('lines').reset_index()
    counts['family'] = counts.drug.map(families)
    counts['dataset'] = dataset
    el = counts[(counts.lines >= MIN_LINES) & counts.family.notna() & ~counts.family.isin(NOT_FAMILIES)
                & ~counts.drug.map(base_name).isin(DEV_DRUGS)]
    fams = []
    for fam, g in el.groupby('family'):
        if len(g) >= 3:
            top = g.sort_values(['lines', 'drug'], ascending=[False, True]).head(3)
            fams.append((fam, top.drug.tolist(), int(top.lines.iloc[2])))
    fams.sort(key=lambda t: (-t[2], t[0]))
    tasks = []
    for k in range(0, len(fams) - 2, 3):
        if len(tasks) >= MAX_TASKS:
            break
        trio = fams[k:k + 3]
        drugs = [d for _, ds, _ in trio for d in ds]
        rows = y[drugs].dropna().index
        if len(rows) < MIN_LINES:
            continue
        name = f'{dataset}_T{len(tasks) + 1}'
        m = pd.concat([y.loc[rows, drugs], tissue.reindex(rows).rename('tissue_group'), X.loc[rows]], axis=1)
        m.index.name = 'unit'
        m.reset_index().to_csv(HERE / 'matrices' / f'{name}.csv', index=False)
        tasks.append({'task': name, 'dataset': dataset, 'code': code_start + len(tasks), 'targets': drugs,
                      'families': [f for f, ds, _ in trio for _ in ds], 'n_units': int(len(rows)), 'n_features': int(X.shape[1])})
    return tasks, counts


def drug_screens(X, tissue):
    match = pd.read_csv(EXT / 'pharmacodb' / 'compound_match.csv').dropna(subset=['pharmacodb_name'])
    # a compound annotated with more than one GDSC pathway (different GDSC IDs) is left without a family
    fam = match.groupby('pharmacodb_name').TARGET_PATHWAY.agg(lambda s: s.iloc[0] if s.nunique() == 1 else None).to_dict()
    long = pd.read_csv(EXT / 'pharmacodb' / 'experiments_long.csv').dropna(subset=['AAC'])
    model = pd.read_csv(EXT / 'depmap24q4' / 'Model.csv', usecols=['ModelID', 'StrippedCellLineName'])
    model['key'] = model.StrippedCellLineName.map(norm)
    uniq = model.groupby('key').ModelID.agg(lambda s: s.iloc[0] if len(s) == 1 else None).dropna().to_dict()
    long['unit'] = long.cell_line.map(norm).map(uniq)
    mapping = long.drop_duplicates(['dataset', 'cell_line'])[['dataset', 'cell_line', 'unit']]
    mapping.to_csv(HERE / 'cell_line_mapping.csv', index=False)
    long = long.dropna(subset=['unit']).rename(columns={'compound': 'drug', 'AAC': 'value'})
    tasks, counts = [], []
    for i, ds in enumerate(sorted(long.dataset.unique())):
        t, c = build_tasks(ds, long[long.dataset == ds], fam, X, tissue, 100 + 10 * i)
        tasks += t; counts.append(c)
    return tasks, counts


def beat_aml():
    d = EXT / 'beataml2'
    fits = pd.read_csv(d / 'beataml_probit_curve_fits_v4_dbgap.txt', sep='\t')
    fits = fits[(fits.type == 'single-agent') & (fits.paper_inclusion == True) & np.isfinite(fits.auc)]  # noqa: E712
    expr = pd.read_csv(d / 'beataml_waves1to4_norm_exp_dbgap.txt', sep='\t')
    wes = pd.read_csv(d / 'beataml_wes_wv1to4_mutations_dbgap.txt', sep='\t', usecols=['dbgap_sample_id', 'symbol', 'variant_classification'])
    path = gdsc_pathways()   # addendum 1: GDSC TARGET_PATHWAY via name or synonym
    families = {i: path.get(base_name(i)) for i in fits.inhibitor.unique()}
    samples = [c for c in expr.columns if c not in ('stable_id', 'display_label', 'description', 'biotype')]
    rna = (fits.dropna(subset=['dbgap_rnaseq_sample'])[lambda f: f.dbgap_rnaseq_sample.isin(samples)]
           .sort_values('dbgap_rnaseq_sample').groupby('dbgap_subject_id').dbgap_rnaseq_sample.first())
    dna = (fits.dropna(subset=['dbgap_dnaseq_sample'])[lambda f: f.dbgap_dnaseq_sample.isin(set(wes.dbgap_sample_id))]
           .sort_values('dbgap_dnaseq_sample').groupby('dbgap_subject_id').dbgap_dnaseq_sample.first())
    patients = sorted(set(rna.index) & set(dna.index))
    pa = wes[~wes.variant_classification.fillna('').str.lower().isin(
        ['silent', 'synonymous_variant', 'intron', 'intron_variant', "3'utr", "5'utr", '3_prime_utr_variant', '5_prime_utr_variant'])]
    mutated = pa.groupby('symbol').dbgap_sample_id.apply(set).to_dict()
    e = expr.drop_duplicates('display_label').set_index('display_label')[samples]
    names = example_features()
    aliases = json.loads((WORK / 'task3b_external' / 'hgnc_cache.json').read_text())
    cols, mapping = {}, []
    for f in names:
        if f.endswith('.CNV') or f in BP.FUSIONS or f == 'MS.HL.MUT':
            continue
        if f.endswith('.MUT'):
            g = BP.SPECIAL.get(f, f.rsplit('.', 1)[0])
            use = BP.resolve(g, mutated, aliases)
            cols[f] = [1.0 if use and dna[p] in mutated[use] else 0.0 for p in patients]
        else:
            use = BP.resolve(f, e.index, aliases)
            if use is None:
                mapping.append({'feature': f, 'mapped_to': ''}); continue
            cols[f] = e.loc[use, [rna[p] for p in patients]].to_numpy(float)
        mapping.append({'feature': f, 'mapped_to': use or ''})
    X = pd.DataFrame(cols, index=[str(p) for p in patients]).dropna(axis=1)
    pd.DataFrame(mapping).to_csv(HERE / 'beataml_feature_mapping.csv', index=False)
    y_long = fits.assign(unit=fits.dbgap_subject_id.astype(str)).rename(columns={'inhibitor': 'drug', 'auc': 'value'})[['unit', 'drug', 'value']]
    tissue = pd.Series('AML', index=X.index)
    return build_tasks('BeatAML', y_long, families, X, tissue, 300)


def julia_task():
    """Human atrial trabeculae task as fixed in plan_validation_addendum2.md."""
    d = pd.read_csv(EXT / 'julia_figshare_28180337' / 'Julia_human_atrial_all_data.csv')
    freqs = ['0.1778', '0.3162', '0.5623', '1', '1.778', '3.162', '5.623', '10', '17.78', '31.62', '56.23', '100']
    mag = lambda cond, f: np.hypot(d[f'CM ({cond}) [freq={f}] real'], d[f'CM ({cond}) [freq={f}] imag'])
    X = pd.DataFrame({'SS F (baseline)': d['SS F (baseline)'], **{f'|CM (baseline)| {f} Hz': mag('baseline', f) for f in freqs}})
    conds = [('0.1 ATP', 'ATP'), ('1 ATP', 'ATP'), ('0 Pi', 'Pi'), ('10 Pi', 'Pi')]
    targets, families, Y = [], [], {}
    for fam in ('ATP', 'Pi'):
        for cond, f in conds:
            if f != fam:
                continue
            for name, col in ((f'SS F ({cond})', d[f'SS F ({cond})']), (f'|CM ({cond})| 1 Hz', mag(cond, '1'))):
                Y[name] = col; targets.append(name); families.append(fam)
    m = pd.concat([pd.DataFrame(Y), pd.Series('trabecula', index=d.index, name='tissue_group'), X], axis=1)
    m.insert(0, 'unit', d['Muscle'].astype(str))
    m = m.dropna()
    m.to_csv(HERE / 'matrices' / 'HumanAtrial_T1.csv', index=False)
    return {'task': 'HumanAtrial_T1', 'dataset': 'HumanAtrial', 'code': 400, 'targets': targets, 'families': families,
            'n_units': int(len(m)), 'n_features': int(X.shape[1]), 'design': 'leave_one_out'}


def main():
    X = depmap_features()
    model = pd.read_csv(EXT / 'depmap24q4' / 'Model.csv', usecols=['ModelID', 'OncotreeLineage']).set_index('ModelID').OncotreeLineage
    lin = model.reindex(X.index).fillna('Unknown')
    tissue = lin.where(lin.map(lin.value_counts()) >= 10, 'Other')
    tasks, counts = drug_screens(X, tissue)
    t2, c2 = beat_aml()
    tasks += t2; counts.append(c2)
    tasks.append(julia_task())
    pd.concat(counts).to_csv(HERE / 'task_counts.csv', index=False)
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1))
    print('features (DepMap)', X.shape)
    for t in tasks:
        print(t['task'], t['n_units'], t['n_features'], list(zip(t['families'][::3], [t['targets'][i:i + 3] for i in (0, 3, 6)])))


if __name__ == '__main__':
    main()
