#!/usr/bin/env python3
"""Task 40 (third locked test, rule v4): pools for sources never analysed before.
- TCGA RNA-seq (UCSC Xena HiSeqV2, log2(RSEM normalised count + 1)): the 10 cohorts with the most primary tumours (-01) among 13 downloaded
  (chosen by sample count only). Primary tumours only; HGNC protein-coding symbols; expressed = median log2 value >= 1 (RSEM >= 1, the
  analogue of median TPM >= 1 in GTEx); no further transform. Pool by t39lib.generic_pool (validated on old GTEx, Task 39 validate_old.json).
- DEMETER2 combined RNAi gene dependency (Achilles + DRIVE + Marcotte): genes measured in >= 99% of cell lines, then cell lines complete on
  those genes; cell lines randomly split into floor(lines / 150) (max 5) disjoint parts (seed 20264050). Pool by generic_pool.
Tasks and jobs: t34lib.make_tasks / t36lib.run_job, as in Tasks 36 and 39."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task39_locked_v3'))
import t39lib as L39  # noqa: E402
T = L39.T; EXT = L39.EXT; L = L39.L
generic_pool = L39.generic_pool
TCGA10 = ['BRCA', 'KIRC', 'HNSC', 'LGG', 'LUAD', 'THCA', 'LUSC', 'PRAD', 'STAD', 'BLCA']   # by primary-tumour count (inspect_shape_print.txt)


def tcga_matrix(cohort):
    x = pd.read_csv(EXT / 'tcga_rnaseq_xena' / f'TCGA.{cohort}.HiSeqV2.gz', sep='\t', index_col=0)
    x = x[[c for c in x.columns if c.endswith('-01')]]
    pc = set(L39.hgnc().index)
    x = x[x.index.isin(pc)]
    x = x[~x.index.duplicated()]
    x = x[x.median(axis=1) >= 1]
    x = x.T
    return x.loc[:, x.notna().all()]


def rnai_sources(seed=20264050):
    d = pd.read_csv(EXT / 'demeter2_rnai' / 'D2_combined_gene_dep_scores.csv', index_col=0)
    d.index = [str(i).split(' (')[0] for i in d.index]
    d = d[~d.index.duplicated()]
    d = d[d.isna().mean(axis=1) <= 0.01]
    d = d.loc[:, d.notna().all(axis=0)].T                                     # lines x genes, complete
    k = min(5, len(d) // 150)
    part = np.random.default_rng(seed).permutation(len(d)) % k
    return {f'N_rnai{i + 1}': d.iloc[np.flatnonzero(part == i)] for i in range(k)}
