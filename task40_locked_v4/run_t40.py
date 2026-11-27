#!/usr/bin/env python3
"""Task 40 third locked test (PLAN_FROZEN_v4.md): jobs identical in form to Tasks 36/39 (t36lib.run_job), outer split 0.
Usage: run_t40.py [workers] [validate]
  validate: before any new source is read, re-run two Task 39 CRISPR jobs through this code path (generic_pool + make_tasks + run_job)
            into validate_results/ and check they reproduce the Task 39 rows exactly."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, sys, time, pickle
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
_cache, _S = {}, {}


def specs():
    import t40lib as L
    s = [('T', f'T_{c}', 20264000 + i, 4000000 + i * 10) for i, c in enumerate(L.TCGA10)]
    s += [('N', f'N_rnai{i + 1}', 20264050 + i, 4050000 + i * 10) for i in range(5)]
    return s


def _src(kind):
    import t40lib as L
    if kind not in _S:
        _S[kind] = L.rnai_sources() if kind == 'N' else L.L39.crispr_sources()
    return _S[kind]


def pool_of(spec, pools):
    import t40lib as L
    pk = pools / (spec[1] + '.pkl')
    if spec not in _cache and pk.exists():
        _cache.clear(); _cache[spec] = pickle.load(open(pk, 'rb'))
    if spec not in _cache:
        _cache.clear()
        kind, name, seed, code0 = spec
        if kind == 'T':
            x = L.tcga_matrix(name[2:])
        elif kind == 'N':
            src = _src('N')
            if name not in src:
                return None
            x = src[name]
        else:
            x = _src('X')[name]
        x, fam, extra = L.generic_pool(x, seed)
        _cache[spec] = (x, L.T.make_tasks(name, x, fam, extra, code0, (71, 200, 1000)))
        pk.parent.mkdir(exist_ok=True); pickle.dump(_cache[spec], open(pk, 'wb'))
    return _cache[spec]


def work(job):
    import t40lib as L
    L.T.GL.low_priority()
    spec, ti, o, out, pools = job
    x, tasks = pool_of(spec, pools); tk = tasks[ti]
    f = out / f"{tk['task']}_p{tk['p']}_o{o}.csv"
    if f.exists():
        return f.name, 0.0
    return f.name, round(L.L.run_job(tk, x, o, f), 1)


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if len(sys.argv) > 2 and sys.argv[2] == 'validate':
        import pandas as pd
        out, pools = HERE / 'validate_results', HERE / 'validate_pools'; out.mkdir(exist_ok=True)
        spec = ('X', 'X_crispr1', 20263950, 3950000)
        x, tasks = pool_of(spec, pools)
        pick = [i for i, t in enumerate(tasks) if t['p'] == 71][:2]
        with Pool(min(workers, 2)) as pl:
            names = [n for n, _ in pl.map(work, [(spec, i, 0, out, pools) for i in pick])]
        res = {}
        for n in names:
            a = pd.read_csv(out / n); b = pd.read_csv(HERE.parent / 'task39_locked_v3' / 'results' / n)
            num = a.select_dtypes('number').columns
            res[n] = dict(rows=[len(a), len(b)], max_abs_diff=float((a[num] - b[num]).abs().max().max()))
        json.dump(res, open(HERE / 'validate_codepath.json', 'w'), indent=1); print(res)
        sys.exit()
    out, pools = HERE / 'results', HERE / 'pools'; out.mkdir(exist_ok=True)
    jobs, meta = [], []
    for s in specs():
        r = pool_of(s, pools)
        if r is None:
            continue
        x, tasks = r
        meta.append(dict(source=s[1], units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks})))
        jobs += [(s, ti, 0, out, pools) for ti in range(len(tasks))]
    _cache.clear(); _S.clear()
    json.dump(meta, open(HERE / 'sources.json', 'w'), indent=1)
    log = open(HERE / 'run_log.txt', 'a', encoding='utf-8'); log.write(f'start {time.ctime()} jobs {len(jobs)}\n'); log.flush()
    t0 = time.time()
    with Pool(workers) as pl:
        for i, (name, sec) in enumerate(pl.imap_unordered(work, jobs)):
            log.write(f'{i + 1}/{len(jobs)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
