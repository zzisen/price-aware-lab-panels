#!/usr/bin/env python3
"""Task 44 (exploratory, additional analysis): is the lack of learnable signal in CRISPR a property of the data type or of how targets were grouped?
Data: DepMap 24Q4 CRISPR gene effects, partitions X_crispr1 and X_crispr2 of Task 39 (already analysed). Only the family definition changes:
  hgnc     - HGNC gene groups (as in Task 39; reproduces its pools)
  corum    - CORUM human complexes (subunit gene names)
  reactome - Reactome pathways with 10-100 genes
  random   - random gene sets of 20 genes (low-signal reference)
Families need >= 10 available members (as generic_pool). Pools, tasks and jobs exactly as Tasks 39/40 (generic_pool logic, make_tasks, t42lib.run_job
pipeline R = ridge + FSCA + somp). Gene sets come from external knowledge or chance, never from the screen's correlations.
Usage: run_t44.py [workers]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(W / 'task42_locked_v5'))
GS = W / 'external_data' / 'gene_sets_t44'
GROUPINGS = ['hgnc', 'corum', 'reactome', 'random']
PARTS = ['X_crispr1', 'X_crispr2']


def members(kind, genes, seed):
    import t42lib as L
    g = set(genes)
    if kind == 'hgnc':
        h = L.L40.L39.hgnc(); m = {}
        for x in sorted(g & set(h.index)):
            if isinstance(h.gene_group.get(x), str):
                for f in h.gene_group[x].split('|'):
                    m.setdefault(f, []).append(x)
        return m
    if kind == 'corum':
        c = pd.read_csv(GS / 'corum_human.txt', sep='\t', usecols=['complex_name', 'subunits_gene_name'])
        return {r.complex_name: sorted({x.strip() for x in str(r.subunits_gene_name).split(';')} & g) for r in c.itertuples()}
    if kind == 'reactome':
        m = {}
        for line in open(GS / 'ReactomePathways.gmt', encoding='utf-8'):
            f = line.rstrip('\n').split('\t')
            if 10 <= len(f) - 2 <= 100:
                m[f[0]] = sorted(set(f[2:]) & g)
        return m
    rng = np.random.default_rng(seed + 991); gl = rng.permutation(sorted(g))
    return {f'random_{i}': sorted(gl[i * 20:(i + 1) * 20]) for i in range(200)}


def pool(x, kind, seed, p_levels=(71, 200, 1000)):
    """generic_pool with a given membership (same steps and seed use as t39lib.generic_pool)."""
    import t42lib as L
    pc = set(L.L40.L39.hgnc().index); genes = sorted(c for c in x.columns if c in pc)
    mem = {f: v for f, v in members(kind, genes, seed).items()}
    elig = sorted(f for f, v in mem.items() if len(v) >= 10)
    rng = np.random.default_rng(seed)
    used, fam, n = [], {}, 0
    for f in rng.permutation(elig):
        free = sorted(set(mem[f]) - set(used))
        if len(free) < 4:
            continue
        for gname in rng.choice(free, 4, replace=False):
            used.append(str(gname)); fam[str(gname)] = str(f)
        n += 1
        if n == 20:
            break
    rest = [gname for gname in genes if gname not in set(used)]
    extra = [str(gname) for gname in rng.permutation(rest)[:max(p_levels) - 71]]
    return x[used + extra].astype(float), fam, extra, len(elig)


def build():
    import t42lib as L
    src = L.L40.L39.crispr_sources()
    (HERE / 'pools').mkdir(exist_ok=True); info = []
    for i, part in enumerate(PARTS):
        seed, code0 = 20263950 + i, 3950000 + i * 10            # Task 39 seeds and codes for these partitions
        for kind in GROUPINGS:
            pk = HERE / 'pools' / f'{kind}_{part}.pkl'
            if pk.exists():
                continue
            x, fam, extra, ne = pool(src[part], kind, seed)
            tasks = L.T.make_tasks(f'{kind}_{part}', x, fam, extra, code0, (71, 200, 1000))
            pickle.dump((x, tasks), open(pk, 'wb')); info.append(f'{kind} {part} eligible families {ne} pool {x.shape[1]} tasks {len(tasks)}')
            print(info[-1], flush=True)
    open(HERE / 'build_print.txt', 'a').write('\n'.join(info) + '\n')


def work(job):
    import t42lib as L
    L.GL.low_priority()
    pk, ti = job
    x, tasks = pickle.load(open(pk, 'rb')); tk = tasks[ti]
    f = HERE / 'results' / f"{tk['task']}_p{tk['p']}_o0.csv"
    if f.exists():
        return f.name, 0.0
    return f.name, round(L.run_job(tk, x, 0, f, 'R'), 1)


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    build()
    (HERE / 'results').mkdir(exist_ok=True)
    jobs = []
    for pk in sorted((HERE / 'pools').glob('*.pkl')):
        _, tasks = pickle.load(open(pk, 'rb')); jobs += [(str(pk), ti) for ti in range(len(tasks))]
    log = open(HERE / 'run_log.txt', 'a'); log.write(f'start {time.ctime()} jobs {len(jobs)}\n'); log.flush(); t0 = time.time()
    with Pool(workers) as pl:
        for i, (n, sec) in enumerate(pl.imap_unordered(work, jobs)):
            log.write(f'{i + 1}/{len(jobs)} {n} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
