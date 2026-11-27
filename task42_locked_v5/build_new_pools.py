#!/usr/bin/env python3
"""Task 42: pools for the locked run. Cohorts = every downloaded TCGA HiSeqV2 cohort not analysed before with >= 150 primary tumours (-01),
by sample count only (structure_counts.txt): CESC COAD ESCA GBM KIRP LIHC OV PAAD PCPG SARC TGCT UCEC (THYM, 120, excluded).
Matrix construction exactly as t40lib.tcga_matrix; pool by t39lib.generic_pool; tasks by t34lib.make_tasks. Seeds 20264200 + i, codes 4200000 + 10 i
(i = position in the alphabetical list). Prints structure only (units, genes, pool size, families)."""
import pickle
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task40_locked_v4'))
import t40lib as L  # noqa: E402
NEW = ['CESC', 'COAD', 'ESCA', 'GBM', 'KIRP', 'LIHC', 'OV', 'PAAD', 'PCPG', 'SARC', 'TGCT', 'UCEC']


def matrix(c):
    folder = 'tcga_rnaseq_xena' if c in ('LIHC', 'OV', 'UCEC') else 'tcga_rnaseq_xena_t42'
    x = pd.read_csv(L.EXT / folder / f'TCGA.{c}.HiSeqV2.gz', sep='\t', index_col=0)
    x = x[[s for s in x.columns if s.endswith('-01')]]
    pc = set(L.L39.hgnc().index)
    x = x[x.index.isin(pc)]; x = x[~x.index.duplicated()]; x = x[x.median(axis=1) >= 1]; x = x.T
    return x.loc[:, x.notna().all()]


if __name__ == '__main__':
    (HERE / 'pools').mkdir(exist_ok=True); lines = []
    for i, c in enumerate(NEW):
        x0 = matrix(c)
        x, fam, extra = L.generic_pool(x0, 20264200 + i)
        tasks = L.T.make_tasks(f'T_{c}', x, fam, extra, 4200000 + i * 10, (71, 200, 1000))
        pickle.dump((x, tasks), open(HERE / 'pools' / f'T_{c}.pkl', 'wb'))
        lines.append(f'T_{c} units {len(x0)} genes {x0.shape[1]} pool {x.shape[1]} families {len(set(fam.values()))} extra {len(extra)} tasks {len(tasks)}')
        print(lines[-1], flush=True)
    open(HERE / 'structure_check_print.txt', 'w').write('\n'.join(lines) + '\n')
