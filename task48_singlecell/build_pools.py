#!/usr/bin/env python3
"""Task 48 (fifth locked test, single-cell expression): pools from 10x FACS-purified PBMC populations (Zheng et al. 2017), never analysed.
Units = metacells: within each purified population, cells are assigned at random (seed 20264800 + i) to groups of 20 (remainder dropped);
counts are summed per group, scaled to counts per million and log2(CPM + 1) transformed (the analogue of TCGA log2(RSEM + 1)).
Genes: HGNC protein-coding symbols, median log2(CPM + 1) >= 1 across metacells, complete. Pool by t39lib.generic_pool (seed 20264800 + i),
tasks by t34lib.make_tasks (codes 4800000 + 10 i), exactly as locked tests 3 and 4. Prints structure only."""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task40_locked_v4'))
import t40lib as L  # noqa: E402
EXT = W / 'external_data' / 'tenx_purified_t48'
TYPES = ['b_cells', 'cd14_monocytes', 'cd34', 'cd4_t_helper', 'cd56_nk', 'cytotoxic_t', 'memory_t', 'naive_cytotoxic', 'naive_t', 'regulatory_t']
SIZE = 20


def metacells(ct, seed):
    d = EXT / ct / 'filtered_matrices_mex' / 'hg19'
    m = mmread(open(d / 'matrix.mtx', 'rb')).tocsc()                       # genes x cells
    genes = pd.read_csv(d / 'genes.tsv', sep='\t', header=None)[1].astype(str).values
    nc = m.shape[1]; rng = np.random.default_rng(seed); perm = rng.permutation(nc); g = nc // SIZE
    sums = np.zeros((m.shape[0], g))
    for j in range(g):
        sums[:, j] = np.asarray(m[:, perm[j * SIZE:(j + 1) * SIZE]].sum(axis=1)).ravel()
    cpm = sums / sums.sum(axis=0, keepdims=True) * 1e6
    x = pd.DataFrame(np.log2(cpm + 1).T, columns=genes, index=[f'{ct}_mc{j}' for j in range(g)])
    x = x.loc[:, ~x.columns.duplicated()]
    pc = set(L.L39.hgnc().index)
    x = x[[c for c in x.columns if c in pc]]
    x = x.loc[:, x.median(axis=0) >= 1]
    return x.loc[:, x.notna().all()], nc


if __name__ == '__main__':
    (HERE / 'pools').mkdir(exist_ok=True); lines = []
    for i, ct in enumerate(TYPES):
        x0, nc = metacells(ct, 20264800 + i)
        x, fam, extra = L.generic_pool(x0, 20264800 + i)
        tasks = L.T.make_tasks(f'S_{ct}', x, fam, extra, 4800000 + i * 10, (71, 200, 1000))
        pickle.dump((x, tasks), open(HERE / 'pools' / f'S_{ct}.pkl', 'wb'))
        lines.append(f'S_{ct} cells {nc} metacells {len(x0)} genes {x0.shape[1]} pool {x.shape[1]} families {len(set(fam.values()))} extra {len(extra)} tasks {len(tasks)}')
        print(lines[-1], flush=True)
    open(HERE / 'structure_check_print.txt', 'w').write('\n'.join(lines) + '\n')
