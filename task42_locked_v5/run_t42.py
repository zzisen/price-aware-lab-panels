#!/usr/bin/env python3
"""Task 42 runner. Usage:
  run_t42.py validate            pipeline R through this code path on two Task 40 jobs; must reproduce Task 40 rows exactly
  run_t42.py time                time one job of each pipeline (T_BRCA, p = 1000)
  run_t42.py dev <pipe> [workers]     development / dry run on the 31 sources of locked tests 2 and 3 (pools cached there)
  run_t42.py locked <pipe> [workers]  the locked run on new sources (pools/ built by build_new_pools.py) -> results_<pipe>/"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; W = HERE.parent; sys.path.insert(0, str(HERE))
DEV_POOLS = sorted((W / 'task39_locked_v3' / 'pools').glob('*.pkl')) + sorted((W / 'task40_locked_v4' / 'pools').glob('*.pkl'))
# run order: tumour RNA-seq first (the analogue of the new data), then RNAi, CRISPR, proteomics, GTEx
DEV_POOLS = [q for q in DEV_POOLS if q.stem[:2] in ('T_', 'N_')]          # development: the 14 sources of locked test 3 (closest analogues)
DEV_POOLS = sorted(DEV_POOLS, key=lambda q: ('T_N_X_P_'.find(q.stem[:2]) if q.stem[:2] in ('T_', 'N_', 'X_', 'P_') else 99, q.stem))
_cache = {}


def pool(pk):
    pk = Path(pk)
    if pk not in _cache:
        _cache.clear(); _cache[pk] = pickle.load(open(pk, 'rb'))
    return _cache[pk]


def work(job):
    import t42lib as L
    L.GL.low_priority()
    pk, ti, out, pipe = job
    x, tasks = pool(pk); tk = tasks[ti]
    f = Path(out) / f"{tk['task']}_p{tk['p']}_o0.csv"
    if f.exists():
        return f.name, 0.0
    return f.name, round(L.run_job(tk, x, 0, f, pipe), 1)


def jobs_for(pools, out, pipe):
    js = []
    for pk in pools:
        _, tasks = pool(pk)
        js += [(str(pk), ti, str(out), pipe) for ti in range(len(tasks))]
    _cache.clear()
    return js


def run(js, workers, log_name):
    log = open(HERE / log_name, 'a', encoding='utf-8'); log.write(f'start {time.ctime()} jobs {len(js)}\n'); log.flush(); t0 = time.time()
    with Pool(workers) as pl:
        for i, (name, sec) in enumerate(pl.imap_unordered(work, js)):
            log.write(f'{i + 1}/{len(js)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'validate':
        import pandas as pd
        out = HERE / 'validate_results'; out.mkdir(exist_ok=True)
        pk = W / 'task40_locked_v4' / 'pools' / 'T_BLCA.pkl'
        _, tasks = pool(pk); pick = [i for i, t in enumerate(tasks) if t['p'] in (71, 1000)][:2]
        with Pool(2) as pl:
            names = [n for n, _ in pl.map(work, [(str(pk), i, str(out), 'R') for i in pick])]
        res = {}
        for n in names:
            a = pd.read_csv(out / n); b = pd.read_csv(W / 'task40_locked_v4' / 'results' / n)
            b = b.merge(a[['n', 'rep', 'k']], on=['n', 'rep', 'k'])
            a = a.sort_values(['n', 'rep', 'k']).reset_index(drop=True); b = b.sort_values(['n', 'rep', 'k']).reset_index(drop=True)
            cols = ['A', 'B', 'C', 'cv5_A', 'cv5_C']
            res[n] = dict(rows=[len(a), len(b)], max_abs_diff=float((a[cols] - b[cols]).abs().max().max()))
        json.dump(res, open(HERE / 'validate_codepath.json', 'w'), indent=1); print(res)
    elif mode == 'time':
        import t42lib as L
        pk = W / 'task40_locked_v4' / 'pools' / 'T_BRCA.pkl'; x, tasks = pool(pk)
        ti = [i for i, t in enumerate(tasks) if t['p'] == 1000][0]
        for pipe in sys.argv[2:] or ['S', 'T']:
            out = HERE / f'time_{pipe}'; out.mkdir(exist_ok=True)
            print(pipe, round(L.run_job(tasks[ti], x, 0, out / 'job.csv', pipe), 1), 's', flush=True)
    elif mode == 'dev':
        pipe = sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        out = HERE / f'dev_results_{pipe}'; out.mkdir(exist_ok=True)
        js = jobs_for(DEV_POOLS, out, pipe)
        if 'rev' in sys.argv:                                                 # a second runner from the other end
            js = js[::-1]
        json.dump([p.stem for p in DEV_POOLS], open(HERE / 'dev_sources.json', 'w'))
        run(js, workers, f'dev_log_{pipe}' + ('_rev' if 'rev' in sys.argv else '') + '.txt')
    elif mode == 'locked':
        pipe = sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        out = HERE / f'results_{pipe}'
        assert not out.exists() or not any(out.iterdir()), 'results folder must be empty before the locked run'
        out.mkdir(exist_ok=True)
        run(jobs_for(sorted((HERE / 'pools').glob('*.pkl')), out, pipe), workers, f'run_log_{pipe}.txt')
