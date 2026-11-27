#!/usr/bin/env python3
"""Task 3b: build the PRISM external matrix with the rules fixed in plan_external_prism.md (frozen 21:17 NZDT).

Inputs (read only, downloaded by external_data/download_external.sh): PRISM 19Q4 secondary screen, DepMap 24Q4
omics, GDSC release 5 (identity recovery only). Output: prism_matrix.csv (local; contains an in_gdsc499 flag that
derives from recovered GDSC identities, so it is not for redistribution) and feature_mapping.csv.
Run with .env312 (needs network once for HGNC symbol lookups, cached in hgnc_cache.json).
"""
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
EXT = WORK / 'external_data'
TARGETS = ['Methotrexate', 'RDEA119', 'PD-0325901', 'CI-1040', 'AZD6244', 'Nilotinib', 'Axitinib']
PRISM_NAME = {'methotrexate': 'Methotrexate', 'refametinib': 'RDEA119', 'PD-0325901': 'PD-0325901',
              'PD-184352': 'CI-1040', 'selumetinib': 'AZD6244', 'nilotinib': 'Nilotinib', 'axitinib': 'Axitinib'}
PROTEIN_ALTERING = ('MISSENSE', 'NONSENSE', 'FRAME_SHIFT', 'IN_FRAME', 'SPLICE_SITE', 'START_CODON', 'NONSTOP',
                    'STOP_CODON', 'DE_NOVO_START')
FUSIONS = {'BCR_ABL.MUT': ('BCR', 'ABL1'), 'EWS_FLI1.MUT': ('EWSR1', 'FLI1'), 'MLL_AFF1.MUT': ('KMT2A', 'AFF1')}
SPECIAL = {'CDKN2a.p14..MUT': 'CDKN2A'}


def responses():
    d = pd.read_csv(EXT / 'prism' / 'secondary-screen-dose-response-curve-parameters.csv',
                    usecols=['depmap_id', 'screen_id', 'name', 'auc', 'passed_str_profiling'])
    d = d[(d.screen_id == 'HTS002') & d.name.isin(PRISM_NAME) & (d.passed_str_profiling.astype(str).str.upper() == 'TRUE')]
    d = d[np.isfinite(d.auc)]
    y = d.assign(target=d.name.map(PRISM_NAME)).groupby(['depmap_id', 'target']).auc.median().unstack()[TARGETS]
    return y.dropna(), d


GDSC2_DRUGS = {1008: 'Methotrexate', 1014: 'Refametinib', 1060: 'PD0325901', 1372: 'Trametinib', 1062: 'Selumetinib',
               1013: 'Nilotinib', 1021: 'Axitinib'}   # Trametinib stands in for CI-1040 (plan_external_gdsc2_addendum.md)
GDSC2_TARGETS = list(GDSC2_DRUGS.values())


def responses_gdsc2():
    d = pd.read_csv(EXT / 'gdsc2' / 'GDSC2_fitted_dose_response_27Oct23.csv', usecols=['COSMIC_ID', 'DRUG_ID', 'LN_IC50'])
    d = d[d.DRUG_ID.isin(GDSC2_DRUGS) & np.isfinite(d.LN_IC50)]
    model = pd.read_csv(EXT / 'depmap24q4' / 'Model.csv', usecols=['ModelID', 'COSMICID']).dropna()
    to_model = dict(zip(model.COSMICID.astype(int), model.ModelID))
    d = d.assign(depmap_id=d.COSMIC_ID.astype(int).map(to_model), target=d.DRUG_ID.map(GDSC2_DRUGS)).dropna(subset=['depmap_id'])
    y = d.groupby(['depmap_id', 'target']).LN_IC50.median().unstack()[GDSC2_TARGETS]
    return y.dropna(), d


def hgnc_lookup(symbols, cache_path):
    """Previous symbols and aliases from HGNC, kept separately (previous symbols take priority)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for s in symbols:
        if s in cache:
            continue
        cache[s] = {}
        for field in ('symbol', 'prev_symbol', 'alias_symbol'):
            req = urllib.request.Request(f'https://rest.genenames.org/search/{field}/{s}', headers={'Accept': 'application/json'})
            cache[s][field] = sorted({doc['symbol'] for doc in json.load(urllib.request.urlopen(req, timeout=30))['response']['docs']})
            time.sleep(0.12)
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def resolve(symbol, available, aliases):
    """Exact symbol; else a unique available previous-symbol hit; else a unique available alias hit; else None.
    A symbol that is itself a current HGNC symbol (e.g. the TRB locus) is never redirected to another gene."""
    if symbol in available:
        return symbol
    if aliases.get(symbol, {}).get('symbol'):
        return None
    for field in ('prev_symbol', 'alias_symbol'):
        hits = [h for h in aliases.get(symbol, {}).get(field, []) if h in available]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return None
    return None


def symbol_table(columns):
    return {c.split(' (')[0]: c for c in columns}


def main(dataset='prism'):
    y, raw = responses() if dataset == 'prism' else responses_gdsc2()
    gdsc = pd.read_csv(WORK / 'base_v32' / 'inputs' / 'GDSC' / 'exampleGDSC_2.3-1_readable.csv', nrows=1)
    names = gdsc.columns[20:].tolist()
    expr = pd.read_csv(EXT / 'depmap24q4' / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv', index_col=0)
    cn = pd.read_csv(EXT / 'depmap24q4' / 'OmicsAbsoluteCNGene.csv', index_col=0)
    mut = pd.read_csv(EXT / 'depmap24q4' / 'OmicsSomaticMutations.csv', usecols=['ModelID', 'HugoSymbol', 'VariantInfo'])
    fus = pd.read_csv(EXT / 'depmap24q4' / 'OmicsFusionFiltered.csv', usecols=['ModelID', 'LeftGene', 'RightGene'])
    model = pd.read_csv(EXT / 'depmap24q4' / 'Model.csv', usecols=['ModelID', 'OncotreeLineage', 'COSMICID'])
    for name, frame in (('expression', expr), ('copy number', cn)):
        assert frame.index.str.startswith('ACH-').all(), f'{name} is not indexed by ModelID'
    print('variant classes:', mut.VariantInfo.value_counts().to_dict())
    keep = mut.VariantInfo.fillna('').str.upper().str.contains('|'.join(PROTEIN_ALTERING))
    mutated = mut[keep].groupby('HugoSymbol').ModelID.apply(set).to_dict()
    mut_models = set(mut.ModelID)

    lines = sorted(set(y.index) & set(expr.index) & set(cn.index) & mut_models)
    e_sym, c_sym = symbol_table(expr.columns), symbol_table(cn.columns)
    gene_of = lambda f: f.rsplit('.', 1)[0] if f.endswith(('.CNV', '.MUT')) else f
    unmatched = sorted({gene_of(f) for f in names if f not in FUSIONS and f not in SPECIAL and f != 'MS.HL.MUT'
                        and gene_of(f) not in (c_sym if f.endswith('.CNV') else e_sym if not f.endswith('.MUT') else mutated)})
    aliases = hgnc_lookup(unmatched, HERE / 'hgnc_cache.json')

    cols, mapping = {}, []
    for f in names:
        g = gene_of(f)
        if f == 'MS.HL.MUT':
            mapping.append({'feature': f, 'source': 'excluded', 'mapped_to': '', 'note': 'no documented MSI call/threshold in DepMap 24Q4 files'}); continue
        if f in FUSIONS:
            a, b = FUSIONS[f]
            l, r = fus.LeftGene.str.split(' ').str[0], fus.RightGene.str.split(' ').str[0]
            hit = set(fus.ModelID[((l == a) & (r == b)) | ((l == b) & (r == a))])
            cols[f] = [1.0 if m in hit else 0.0 for m in lines]
            mapping.append({'feature': f, 'source': 'OmicsFusionFiltered', 'mapped_to': f'{a}-{b}', 'note': f'{sum(cols[f]):.0f} positive'}); continue
        if f.endswith('.MUT'):
            gene = SPECIAL.get(f, g)
            use = resolve(gene, mutated, aliases)
            cols[f] = [1.0 if use and m in mutated[use] else 0.0 for m in lines]
            mapping.append({'feature': f, 'source': 'OmicsSomaticMutations', 'mapped_to': use or '(no mutations in DepMap)',
                            'note': 'alias' if use and use != gene else ''}); continue
        table, frame, src = (c_sym, cn, 'OmicsAbsoluteCNGene') if f.endswith('.CNV') else (e_sym, expr, 'OmicsExpressionProteinCodingGenesTPMLogp1')
        use = resolve(g, table, aliases)
        if use is None:
            mapping.append({'feature': f, 'source': 'excluded', 'mapped_to': '', 'note': 'symbol not found, or HGNC previous/alias symbol ambiguous'}); continue
        cols[f] = frame.loc[lines, table[use]].to_numpy(float)
        mapping.append({'feature': f, 'source': src, 'mapped_to': use, 'note': 'alias' if use != g else ''})
    X = pd.DataFrame(cols, index=lines)
    complete = X.notna().all(axis=1)
    X = X[complete]
    out = y.loc[X.index].copy()
    lineage = model.set_index('ModelID').OncotreeLineage.reindex(X.index).fillna('Unknown')
    counts = lineage.value_counts()
    out['tissue_group'] = lineage.where(lineage.map(counts) >= 10, 'Other')

    # identities of the 499 GDSC example rows (release-5 matching, as in base_v32/rebuild_release5_local.py)
    src = pd.read_csv(EXT / 'gdsc_release5' / 'gdsc_en_input_w5.csv', index_col=0)
    ex_full = pd.read_csv(WORK / 'base_v32' / 'inputs' / 'GDSC' / 'exampleGDSC_2.3-1_readable.csv')
    anchors = ex_full.columns[20:30].tolist()
    S = np.log(src.loc[anchors].to_numpy(float).T)
    idx = []
    for row in ex_full[anchors].to_numpy(float):
        c = np.flatnonzero(np.max(np.abs(S - row), axis=1) < 1e-10); assert len(c) == 1; idx.append(int(c[0]))
    meta = pd.read_csv(EXT / 'gdsc_release5' / 'gdsc_cell_lines_w5.csv', dtype=str)[['CELL_LINE_NAME', 'COSMIC_ID']].drop_duplicates('CELL_LINE_NAME').set_index('CELL_LINE_NAME')
    cosmic499 = set(meta.loc[src.columns[idx], 'COSMIC_ID'].astype(float).astype(int))
    cosmic_of = model.set_index('ModelID').COSMICID.reindex(out.index)
    out['in_gdsc499'] = cosmic_of.isin(cosmic499).astype(int)
    out = pd.concat([out, X], axis=1)
    out.index.name = 'depmap_id'
    out.reset_index().to_csv(HERE / f'{dataset}_matrix.csv', index=False)
    mp = pd.DataFrame(mapping); mp.to_csv(HERE / f'feature_mapping_{dataset}.csv', index=False)
    print(dataset, 'lines with all 7 responses:', len(y), '| with omics:', len(lines), '| complete features:', len(out))
    print('features kept', X.shape[1], 'of', len(names), '| excluded:', mp[mp.source == 'excluded'].feature.tolist())
    print('aliases used:', mp[mp.note == 'alias'][['feature', 'mapped_to']].values.tolist())
    print('tissue groups:', out.tissue_group.value_counts().to_dict())
    print('in GDSC 499:', int(out.in_gdsc499.sum()), '| new units:', int((1 - out.in_gdsc499).sum()))


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'prism')
