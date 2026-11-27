#!/usr/bin/env python3
"""Task 36 stage 1 driver: all 46 used sources of Task 34 (same pools, tasks and seeds), denser grid (t36lib). Parallel over jobs.
Usage: run_stage1.py [workers] [limit]   -> stage1_results/<task>_p<p>_o<o>.csv, stage1_log.txt"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / 'stage1_results'
DEV_T = ['muscle_skeletal', 'skin_sun_exposed_lower_leg', 'adipose_subcutaneous', 'artery_tibial', 'thyroid', 'nerve_tibial', 'lung',
         'esophagus_mucosa', 'heart_left_ventricle', 'colon_transverse', 'breast_mammary_tissue', 'pancreas']
FRESH_T = ['skin_not_sun_exposed_suprapubic', 'adipose_visceral_omentum', 'artery_aorta', 'artery_coronary', 'esophagus_muscularis',
           'esophagus_gastroesophageal_junction', 'colon_sigmoid', 'stomach', 'testis', 'heart_atrial_appendage', 'cells_cultured_fibroblasts',
           'adrenal_gland', 'pituitary', 'spleen', 'prostate', 'brain_cerebellum', 'liver']
DEV_C = ['KIRC', 'LGG', 'OV', 'UCEC', 'THCA', 'LUAD', 'STAD', 'COAD', 'PRAD', 'HNSC', 'BLCA', 'LUSC']
USED = set(DEV_C) | {'BRCA'}
_cache = {}


def specs():
    s = [('G', t, 'gtex_v8_tissues', 20262400 + i, 2400000 + i * 10) for i, t in enumerate(DEV_T)]
    s += [('G', t, 'gtex_v8_tissues_fresh', 20263400 + i, 3400000 + i * 10) for i, t in enumerate(FRESH_T)]
    return s


def pool_of(spec):
    import pickle
    import t36lib as L
    pk = HERE / 'stage1_pools' / ('_'.join(str(v) for v in spec[:2]) + '.pkl')
    if spec not in _cache and pk.exists():
        _cache.clear(); _cache[spec] = pickle.load(open(pk, 'rb'))
    if spec not in _cache:
        _cache.clear()
        kind = spec[0]
        if kind == 'G':
            _, t, folder, seed, code0 = spec
            x, fam, extra = L.T.gtex_pool(t, folder, seed)
            tasks = L.T.make_tasks('G_' + t, x, fam, extra, code0, (71, 200, 1000))
        else:
            _, c, code0 = spec
            x, fam, extra = L.T.rppa_pools([c], min_n=150)[c]
            tasks = L.T.make_tasks('R_' + c, x, fam, extra, code0, (71, 71 + len(extra)))
        _cache[spec] = (x, tasks)
        pk.parent.mkdir(exist_ok=True); pickle.dump(_cache[spec], open(pk, 'wb'))
    return _cache[spec]


def work(job):
    import t36lib as L
    L.T.GL.low_priority()
    spec, ti, o = job
    x, tasks = pool_of(spec)
    tk = tasks[ti]
    f = OUT / f"{tk['task']}_p{tk['p']}_o{o}.csv"
    if f.exists():
        return f.name, 0.0
    return f.name, round(L.run_job(tk, x, o, f), 1)


FULL = 'o0' not in sys.argv
if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != 'o0' else None
    import t36lib as L
    p = pd.read_csv(L.T.EXT / 'tcga_rppa_xena' / 'Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', usecols=['sample', 'cancer type abbreviation'])
    fresh_c = sorted(set(p['cancer type abbreviation'].dropna()) - USED - {'BRCA'})
    sp = specs()
    rp_dev = sorted(L.T.rppa_pools(DEV_C, 150)); rp_fresh = sorted(L.T.rppa_pools(fresh_c, 150))
    sp += [('R', c, 2500000 + i * 10) for i, c in enumerate(rp_dev)]
    sp += [('R', c, 3500000 + i * 10) for i, c in enumerate(rp_fresh)]
    jobs = []
    for s in sp:
        _, tasks = pool_of(s)
        jobs += [(s, ti, o) for ti in range(len(tasks)) for o in range(3 if FULL else 1)]
    _cache.clear()
    if limit:
        jobs = jobs[:limit]
    log = open(HERE / 'stage1_log.txt', 'a', encoding='utf-8')
    log.write(f'start {time.ctime()} jobs {len(jobs)} workers {workers}\n'); log.flush()
    t0 = time.time()
    with Pool(workers) as pl:
        for i, (name, sec) in enumerate(pl.imap_unordered(work, jobs, chunksize=1)):
            log.write(f'{i + 1}/{len(jobs)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
