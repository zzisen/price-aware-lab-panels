#!/usr/bin/env python3
"""Task 39 locked test (PLAN_FROZEN_v3.md): rule v2 (frozen Task 34) and rule v3 on sources never analysed before.
Jobs identical in form to Task 36 stage 1 (t36lib.run_job, denser grid). Parallel. Output: results/<task>_p<p>_o<o>.csv, run_log.txt
Usage: run_t39.py [workers]"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
OUT = HERE / 'results'
_cache = {}


def specs():
    import t39lib as L
    s = [('G', t, 20263900 + i, 3900000 + i * 10) for i, t in enumerate(L.GTEX2)]
    s += [('X', f'X_crispr{i + 1}', 20263950 + i, 3950000 + i * 10) for i in range(5)]
    s += [('P', f'P_ccle{i + 1}', 20263960 + i, 3960000 + i * 10) for i in range(2)]
    return s


def pool_of(spec):
    import pickle
    import t39lib as L
    pk = HERE / 'pools' / (spec[1] + '.pkl')
    if spec not in _cache and pk.exists():
        _cache.clear(); _cache[spec] = pickle.load(open(pk, 'rb'))
    if spec not in _cache:
        _cache.clear()
        kind, name, seed, code0 = spec
        if kind == 'G':
            x, fam, extra = L.T.gtex_pool(name, 'gtex_v8_tissues_fresh2', seed); src = 'G_' + name
        elif kind == 'X':
            x, fam, extra = L.generic_pool(_src('X')[name], seed); src = name
        else:
            x, fam, extra = L.generic_pool(_src('P')[name], seed); src = name
        _cache[spec] = (x, L.T.make_tasks(src, x, fam, extra, code0, (71, 200, 1000)))
        pk.parent.mkdir(exist_ok=True); pickle.dump(_cache[spec], open(pk, 'wb'))
    return _cache[spec]


_S = {}


def _src(kind):
    import t39lib as L
    if kind not in _S:
        _S[kind] = L.crispr_sources() if kind == 'X' else L.proteomics_sources()
    return _S[kind]


def work(job):
    import t39lib as L
    L.T.GL.low_priority()
    spec, ti, o = job
    x, tasks = pool_of(spec); tk = tasks[ti]
    f = OUT / f"{tk['task']}_p{tk['p']}_o{o}.csv"
    if f.exists():
        return f.name, 0.0
    return f.name, round(L.L.run_job(tk, x, o, f), 1)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    jobs, meta = [], []
    for s in specs():
        x, tasks = pool_of(s)
        meta.append(dict(source=s[1], units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks})))
        jobs += [(s, ti, 0) for ti in range(len(tasks))]                   # outer split 0 only (as Task 36 main analysis)
    _cache.clear(); _S.clear()
    json.dump(meta, open(HERE / 'sources.json', 'w'), indent=1)
    log = open(HERE / 'run_log.txt', 'a', encoding='utf-8'); log.write(f'start {time.ctime()} jobs {len(jobs)}\n'); log.flush()
    t0 = time.time()
    with Pool(workers) as pl:
        for i, (name, sec) in enumerate(pl.imap_unordered(work, jobs)):
            log.write(f'{i + 1}/{len(jobs)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
