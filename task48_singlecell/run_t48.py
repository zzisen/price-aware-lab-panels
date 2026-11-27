#!/usr/bin/env python3
"""Task 48 locked run: jobs identical in form to locked test 3 (pipeline R = ridge + FSCA + somp through t42lib.run_job, which reproduced
Task 40 rows exactly). Usage: run_t48.py [workers]. Refuses to run if results/ already contains files."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task42_locked_v5'))


def work(job):
    import t42lib as L
    L.GL.low_priority()
    pk, ti = job
    x, tasks = pickle.load(open(pk, 'rb')); tk = tasks[ti]
    f = HERE / 'results' / f"{tk['task']}_p{tk['p']}_o0.csv"
    return f.name, round(L.run_job(tk, x, 0, f, 'R'), 1)


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = HERE / 'results'
    assert not out.exists() or not any(out.iterdir()), 'results must be empty'
    out.mkdir(exist_ok=True)
    jobs = []
    for pk in sorted((HERE / 'pools').glob('*.pkl')):
        _, tasks = pickle.load(open(pk, 'rb')); jobs += [(str(pk), ti) for ti in range(len(tasks))]
    log = open(HERE / 'run_log.txt', 'a'); log.write(f'start {time.ctime()} jobs {len(jobs)}\n'); log.flush(); t0 = time.time()
    with Pool(workers) as pl:
        for i, (n, sec) in enumerate(pl.imap_unordered(work, jobs)):
            log.write(f'{i + 1}/{len(jobs)} {n} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
