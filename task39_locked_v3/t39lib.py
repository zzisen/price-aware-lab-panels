#!/usr/bin/env python3
"""Task 39 (stage 3 locked test of rule v3): pools for sources never analysed before.
- GTEx v8: 10 tissues in external_data/gtex_v8_tissues_fresh2 (same construction as Task 34, t34lib.gtex_pool).
- DepMap 24Q4 CRISPR gene effect (cell-line gene dependency): 5 disjoint random partitions of the cell lines (seed 20263950).
- CCLE mass-spectrometry proteomics (Nusinow 2020): one row per cell line (first replicate), 2 disjoint random partitions (seed 20263960).
Families for CRISPR and proteomics: HGNC gene groups (>= 10 available members), 20 families x 4 genes as the 80-measurement pool,
plus nested random extensions to p = 200 and 1000, exactly as for GTEx. Tasks and jobs: t34lib.make_tasks / t36lib.run_job."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task36_rule_v3'))
import t36lib as L  # noqa: E402
T = L.T; EXT = T.EXT
GTEX2 = ['brain_cortex', 'brain_caudate_basal_ganglia', 'brain_hypothalamus', 'brain_spinal_cord_cervical_c-1', 'cells_ebv-transformed_lymphocytes',
         'minor_salivary_gland', 'ovary', 'small_intestine_terminal_ileum', 'uterus', 'vagina']


def hgnc():
    h = pd.read_csv(EXT / 'hgnc' / 'hgnc_complete_set.txt', sep='\t', low_memory=False, usecols=['symbol', 'locus_group', 'gene_group'])
    return h[h.locus_group == 'protein-coding gene'].drop_duplicates('symbol').set_index('symbol')


def generic_pool(x, seed, p_levels=(71, 200, 1000)):
    """x: units x gene symbols, complete. Same family logic as t34lib.gtex_pool."""
    h = hgnc(); genes = sorted(g for g in x.columns if g in h.index)
    members = {}
    for g in genes:
        if isinstance(h.gene_group.get(g), str):
            for f in h.gene_group[g].split('|'):
                members.setdefault(f, []).append(g)
    elig = sorted(f for f, m in members.items() if len(m) >= 10)
    rng = np.random.default_rng(seed)
    used, fam, n = [], {}, 0
    for f in rng.permutation(elig):
        free = sorted(set(members[f]) - set(used))
        if len(free) < 4:
            continue
        for g in rng.choice(free, 4, replace=False):
            used.append(str(g)); fam[str(g)] = str(f)
        n += 1
        if n == 20:
            break
    rest = [g for g in genes if g not in set(used)]
    extra = [str(g) for g in rng.permutation(rest)[:max(p_levels) - 71]]
    return x[used + extra].astype(float), fam, extra


def crispr_sources(k=5, seed=20263950):
    d = pd.read_csv(EXT / 'depmap24q4_crispr' / 'CRISPRGeneEffect.csv', index_col=0)
    d.columns = [c.split(' (')[0] for c in d.columns]
    d = d.loc[:, ~d.columns.duplicated()]
    d = d.loc[:, d.notna().all()]
    part = np.random.default_rng(seed).permutation(len(d)) % k
    return {f'X_crispr{i + 1}': d.iloc[np.flatnonzero(part == i)] for i in range(k)}


def proteomics_sources(k=2, seed=20263960):
    p = pd.read_csv(EXT / 'ccle_proteomics_nusinow2020' / 'protein_quant_current_normalized.csv.gz')
    sam = [c for c in p.columns if '_TenPx' in c and not c.endswith('_Peptides')]
    line = pd.Series([c.split('_TenPx')[0] for c in sam], index=sam)
    keep = line[~line.duplicated()].index.tolist()
    q = p[['Gene_Symbol'] + keep].dropna(subset=['Gene_Symbol'])
    q = q[~q.Gene_Symbol.duplicated()].set_index('Gene_Symbol')[keep].T
    q = q.loc[:, q.notna().all()]
    part = np.random.default_rng(seed).permutation(len(q)) % k
    return {f'P_ccle{i + 1}': q.iloc[np.flatnonzero(part == i)] for i in range(k)}
