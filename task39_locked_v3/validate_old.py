#!/usr/bin/env python3
"""Before any Task 39 source is used: check t39lib.generic_pool (the pool builder for CRISPR and proteomics) on two GTEx tissues already
analysed (dev, Task 34/36). The tissue matrix is built as in t34lib.gtex_pool (log2 TPM+1, expressed protein-coding genes, symbols),
then pooled by generic_pool with the dev seeds; p = 71, outer split 0. Compared with the Task 36 stage 1 rows of the same tissues
(built by gtex_pool): mean C - A by n/k band, regret of rule H and of always-C. Output validate_old.json."""
import glob
import json
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent; sys.path.insert(0, str(HERE))
import t39lib as L  # noqa: E402
TIS = [('muscle_skeletal', 0), ('lung', 6)]
OUTD = HERE / 'validate_old_results'


def matrix(t):
    h = pd.read_csv(L.EXT / 'hgnc' / 'hgnc_complete_set.txt', sep='\t', low_memory=False, usecols=['symbol', 'locus_group', 'ensembl_gene_id'])
    h = h[(h.locus_group == 'protein-coding gene') & h.ensembl_gene_id.notna()].drop_duplicates('ensembl_gene_id').set_index('ensembl_gene_id')
    x = pd.read_csv(L.EXT / 'gtex_v8_tissues' / f'gene_tpm_2017-06-05_v8_{t}.gct.gz', sep='\t', skiprows=2)
    x['ens'] = x.Name.str.split('.').str[0]
    x = x[x.ens.isin(h.index)].drop_duplicates('ens').set_index('ens').drop(columns=['id', 'Name', 'Description'])
    x = x[x.median(axis=1) >= 1]
    x = np.log2(x + 1).T; x.columns = [h.symbol[e] for e in x.columns]
    return x.loc[:, ~x.columns.duplicated()]


def job(a):
    t, i, ti = a
    x, fam, extra = L.generic_pool(matrix(t), 20262400 + i)
    tasks = [k for k in L.T.make_tasks('V_' + t, x, fam, extra, 2400000 + i * 10, (71,))]
    if ti >= len(tasks):
        return None
    f = OUTD / f"{tasks[ti]['task']}_p71_o0.csv"
    if not f.exists():
        L.L.run_job(tasks[ti], x, 0, f)
    return f.name


def summ(d):
    d = d.copy(); d['nk'] = d.n / d.k; d['best'] = d[['A', 'B', 'C']].max(axis=1)
    h = np.where(d.nk >= 10, d.C, np.where(d.cv5_A < d.cv5_C, d.A, d.C))
    band = pd.cut(d.nk, [0, 5, 10, 50], right=False)
    return dict(cells=len(d), regret_H=float((d.best - h).mean()), regret_C=float((d.best - d.C).mean()), regret_A=float((d.best - d.A).mean()),
                gain_by_band={str(b): float((g.C - g.A).mean()) for b, g in d.groupby(band, observed=True)})


if __name__ == '__main__':
    OUTD.mkdir(exist_ok=True)
    with Pool(12) as pl:
        print(pl.map(job, [(t, i, ti) for t, i in TIS for ti in range(6)]))
    res = {}
    for t, _ in TIS:
        new = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUTD / f'V_{t}_T*_p71_o0.csv'))])
        old = pd.concat([pd.read_csv(f) for f in glob.glob(str(W / 'task36_rule_v3' / 'stage1_results' / f'G_{t}_T*_p71_o0.csv'))])
        res[t] = dict(generic_pool=summ(new), gtex_pool=summ(old))
    json.dump(res, open(HERE / 'validate_old.json', 'w'), indent=1); print(json.dumps(res, indent=1))
