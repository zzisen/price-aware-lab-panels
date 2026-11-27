#!/usr/bin/env python3
"""Task 17 (plan_gates): runs G1, G2 and G3 with two workers, one numeric thread each, at below-normal priority. Finished jobs are skipped on a restart.
Same subsamples and seeds as task16_route_b/run_v. decorr, random and R5 are checked against the rerun of Task 16 wherever they overlap; a difference stops the run.
Run with .env312:  python -X utf8 run_gates.py [check]"""
import json
import os
import sys
import time
from pathlib import Path

for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[v] = '1'

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed, parallel_config  # noqa: E402

HERE = Path(__file__).resolve().parent
W = HERE.parent
T16 = W / 'task16_route_b'
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(T16))
import gatelib as L  # noqa: E402
import run_v  # noqa: E402

OUT = HERE / 'results'
FIXED = T16 / 'results_fixed'
N_JOBS = 2
K13 = [5, 10, 20]
REPS = 4
SENS = {'BEAT': -1.0, 'E1': 1.0, 'E2': -1.0}        # plan section 4: sign that makes larger mean more sensitive
G2_AWARE = ['lasso', 'somp', 'mrmr', 'pls_vip', 'rf']
G2_REUSE = ['qr', 'fsca', 'medoid']
G3_REUSE = ['qr', 'fsca', 'medoid']


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load(spec):
    if spec.get('pool'):
        m = pd.read_csv(HERE / 'pools' / f"{spec['source']}.csv", dtype={'unit': str}).set_index('unit')
        return m.index.to_numpy(), m[spec['candidates']].to_numpy(float), m[spec['targets']].to_numpy(float), np.asarray(spec['families']), spec['code'], spec['split']
    return run_v.load_spec(spec)


def setup(spec, o, n, r, units, x, y, trn, te, code):
    sd = [run_v.libs()[1].SEED, code, o, n or 0, r]
    sub = trn if n is None else np.random.default_rng(sd).choice(trn, n, replace=False)
    xs = x[sub]; ym = y[sub].mean(0); ysd = y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
    ys = (y[sub] - ym) / ysd; yt = (y[te] - ym) / ysd
    seed = int(np.random.default_rng(sd).integers(1 << 30))
    return sd, sub, xs, ym, ysd, ys, yt, seed


def job_g1(spec, o, ns):
    """Baseline panels on the G1 sources. R5 comes from the Task 16 rerun; decorr and random are recomputed as a check."""
    part = OUT / 'G1' / 'parts' / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    L.low_priority()
    R, V = L.r5_libs()
    t0 = time.time()
    units, x, y, fam, code, split = load(spec)
    te, trn = run_v.outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows = []
    for n in ns:
        if n is not None and n >= len(trn):
            continue
        for r in (range(REPS) if n else [0]):
            sd, sub, xs, ym, ysd, ys, yt, seed = setup(spec, o, n, r, units, x, y, trn, te, code)
            xt = x[te]
            for k in K13:
                if k >= x.shape[1]:
                    continue
                panels = {'decorr': V.decorr(xs, k), 'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False)}
                panels.update({a: f(xs, ys, k, seed) for a, f in L.AWARE.items()})
                panels.update({a: f(xs, k, seed) for a, f in L.REUSE.items()})
                for meth, cols in panels.items():
                    for pname, pred in [('ridge', R.ridge_predict(xs, ys, xt, cols)), ('gbt', V.gbt_predict(xs, ys, xt, cols, seed))]:
                        rows.append({'task': spec['task'], 'outer': o, 'n': 'all' if n is None else n, 'n_actual': len(sub), 'rep': r, 'method': meth, 'predictor': pname, 'k': k, 'k_used': len(cols), **V.metrics(pred, yt, ym, ysd)})
    write(pd.DataFrame(rows), part)
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def job_g2(spec, o, n):
    """Drug choice per test patient. Saves one row per patient for each panel, predictor and budget."""
    part = OUT / 'G2' / 'parts' / f"{spec['task']}_{o}.csv.gz"
    if part.exists():
        return
    L.low_priority()
    R, V = L.r5_libs()
    R.RES5 = ['Shared', 'Family']                                       # as in the W tasks: 30 or 40 targets do not use Exact
    t0 = time.time()
    units, x, y, fam, code, split = load(spec)
    sign = SENS[spec['source']]
    te, trn = run_v.outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows, check = [], []
    for r in range(REPS):
        sd, sub, xs, ym, ysd, ys, yt, seed = setup(spec, o, n, r, units, x, y, trn, te, code)
        xt = x[te]; truth = sign * y[te]
        paths = {res: R.select_path(xs, ys, fam, res, seed) for res in R.RES5}
        chosen, _ = R.middle_choice(xs, ys, fam, V.KS, seed)
        pop = int(np.argmax(sign * ym))
        order = np.argsort(np.argsort(truth, axis=1), axis=1) / (truth.shape[1] - 1)
        best5 = np.argsort(-truth, axis=1)[:, :5]
        for k in K13:
            panels = {'R5': run_v.r5_at(R, paths, chosen, k, xs, ys, xt)[0], 'decorr': V.decorr(xs, k), 'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False)}
            panels.update({a: L.AWARE[a](xs, ys, k, seed) for a in G2_AWARE})
            panels.update({a: L.REUSE[a](xs, k, seed) for a in G2_REUSE})
            for meth, cols in panels.items():
                for pname, pred in [('ridge', R.ridge_predict(xs, ys, xt, cols)), ('gbt', V.gbt_predict(xs, ys, xt, cols, seed))]:
                    raw = sign * (pred * ysd + ym)
                    pick = np.argmax(raw, axis=1)
                    top5 = np.argsort(-raw, axis=1)[:, :5]
                    i = np.arange(len(te))
                    rows.append(pd.DataFrame({'task': spec['task'], 'source': spec['source'], 'outer': o, 'rep': r, 'method': meth, 'predictor': pname, 'k': k, 'unit': units[te],
                                              'v': truth[i, pick], 'v0': truth[:, pop], 'vmax': truth.max(1), 'pct': order[i, pick], 'top5': [len(set(a) & set(b)) / 5 for a, b in zip(top5, best5)]}))
                    if meth in ('R5', 'decorr', 'random'):
                        check.append({'rep': r, 'method': meth, 'predictor': pname, 'k': k, 'sse': float(((pred - yt) ** 2).sum())})
    write(pd.concat(rows, ignore_index=True), part)
    write(pd.DataFrame(check), OUT / 'G2' / 'check' / f"{spec['task']}_{o}.csv")
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def features(spec, o, ns, out):
    """A, B and All for one job, from the training subsamples only."""
    part = out / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    L.low_priority()
    R, V = L.r5_libs()
    units, x, y, fam, code, split = load(spec)
    te, trn = run_v.outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows = []
    for n in ns:
        if n >= len(trn):
            continue
        for r in range(REPS):
            sd, sub, xs, ym, ysd, ys, yt, seed = setup(spec, o, n, r, units, x, y, trn, te, code)
            for k, f in L.structure(xs, ys, [a for a in K13 if a < x.shape[1]], seed + 1).items():
                rows.append({'task': spec['task'], 'outer': o, 'n': n, 'rep': r, 'k': k, 'p': x.shape[1], 'targets': y.shape[1], **f})
    write(pd.DataFrame(rows), part)


def job_g3(spec, o, ns):
    """R5 and the reusable panels on an unused source."""
    part = OUT / 'G3' / 'parts' / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    L.low_priority()
    R, V = L.r5_libs()
    R.RES5 = ['Shared', 'Family', 'Exact']
    t0 = time.time()
    units, x, y, fam, code, split = load(spec)
    te, trn = run_v.outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows = []
    for n in ns:
        if n >= len(trn):
            continue
        for r in range(REPS):
            sd, sub, xs, ym, ysd, ys, yt, seed = setup(spec, o, n, r, units, x, y, trn, te, code)
            xt = x[te]
            paths = {res: R.select_path(xs, ys, fam, res, seed) for res in R.RES5}
            chosen, _ = R.middle_choice(xs, ys, fam, V.KS, seed)
            for k in K13:
                panel, _, opt = run_v.r5_at(R, paths, chosen, k, xs, ys, xt)
                panels = {'R5': panel, 'decorr': V.decorr(xs, k), 'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False)}
                panels.update({a: L.REUSE[a](xs, k, seed) for a in G3_REUSE})
                for meth, cols in panels.items():
                    for pname, pred in [('ridge', R.ridge_predict(xs, ys, xt, cols)), ('gbt', V.gbt_predict(xs, ys, xt, cols, seed))]:
                        rows.append({'task': spec['task'], 'source': spec['source'], 'outer': o, 'n': n, 'rep': r, 'method': meth, 'predictor': pname, 'k': k, 'k_used': len(cols), 'option': opt if meth == 'R5' else '', **V.metrics(pred, yt, ym, ysd)})
    write(pd.DataFrame(rows), part)
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def g1_specs():
    tasks = json.loads((T16 / 'tasks.json').read_text(encoding='utf-8'))
    return [t for t in tasks if t['analysis'] == 'V'] + run_v.legacy_specs()


def largest(spec):
    return 500 if spec['task'].startswith('NH') else None


def fixed_rows(task, o):
    """The Task 16 rerun rows of one job, from every folder that holds it."""
    fr = []
    for folder in ['V', 'U', 'L']:
        f = FIXED / folder / 'parts' / f'{task}_{o}.csv'
        if f.exists():
            fr.append(pd.read_csv(f))
    d = pd.concat(fr, ignore_index=True)
    d['n'] = d.n.astype(str)
    return d.drop_duplicates(['n', 'rep', 'method', 'predictor', 'k'])


def check_g1(specs):
    worst, compared = 0.0, 0
    for s in specs:
        for o in range(5):
            new = pd.read_csv(OUT / 'G1' / 'parts' / f"{s['task']}_{o}.csv")
            new['n'] = new.n.astype(str)
            new = new[new.method.isin(['decorr', 'random'])]
            m = new.merge(fixed_rows(s['task'], o), on=['n', 'rep', 'method', 'predictor', 'k'], suffixes=('', '_old'))
            compared += len(m)
            assert len(m) == len(new), f"{s['task']} {o}: rows missing in the Task 16 rerun"
            worst = max(worst, float(np.abs(m.sse - m.sse_old).max()))
    print('G1 check against the Task 16 rerun: rows', compared, '| largest SSE difference', worst, flush=True)
    assert worst == 0, 'decorr or random differ from the Task 16 rerun'


def check_g2(specs):
    worst, compared = 0.0, 0
    for s in specs:
        for o in range(3):
            new = pd.read_csv(OUT / 'G2' / 'check' / f"{s['task']}_{o}.csv")
            old = pd.read_csv(FIXED / 'W' / 'parts' / f"{s['task']}_{o}.csv")
            m = new.merge(old[old.n == 100], on=['rep', 'method', 'predictor', 'k'], suffixes=('', '_old'))
            compared += len(m)
            assert len(m) == len(new), f"{s['task']} {o}: rows missing in W"
            worst = max(worst, float(np.abs(m.sse - m.sse_old).max()))
    print('G2 check against W of the Task 16 rerun: rows', compared, '| largest SSE difference', worst, flush=True)
    assert worst == 0, 'R5, decorr or random differ from W'


def smoke():
    """A short G1 job, a short G2 job and one structure call, written to results_check/. Checks that the rules return k distinct measurements. Prints no accuracy."""
    global OUT
    OUT = HERE / 'results_check'
    specs = {s['task']: s for s in g1_specs()}
    t0 = time.time(); job_g1(specs['NH_A3_G'], 4, [100]); print('G1 job with n = 100 only', round(time.time() - t0), 's', flush=True)
    d = pd.read_csv(OUT / 'G1' / 'parts' / 'NH_A3_G_4.csv')
    assert (d.k_used == d.k).all(), d[d.k_used != d.k][['method', 'k', 'k_used']].drop_duplicates()
    new = d[d.method.isin(['decorr', 'random'])].assign(n=lambda a: a.n.astype(str))
    m = new.merge(fixed_rows('NH_A3_G', 4), on=['n', 'rep', 'method', 'predictor', 'k'], suffixes=('', '_old'))
    print('rows', len(m), 'of', len(new), '| largest SSE difference', float(np.abs(m.sse - m.sse_old).max()), flush=True)
    print('methods', sorted(d.method.unique()), flush=True)
    tasks = json.loads((T16 / 'tasks.json').read_text(encoding='utf-8'))
    w = {t['task']: t for t in tasks if t['analysis'] == 'W'}
    t0 = time.time(); job_g2(w['W_E1_T1_t40'], 0, 100); print('G2 job', round(time.time() - t0), 's', flush=True)
    n = pd.read_csv(OUT / 'G2' / 'check' / 'W_E1_T1_t40_0.csv')
    old = pd.read_csv(FIXED / 'W' / 'parts' / 'W_E1_T1_t40_0.csv')
    m = n.merge(old[old.n == 100], on=['rep', 'method', 'predictor', 'k'], suffixes=('', '_old'))
    print('G2 check rows', len(m), 'of', len(n), '| largest SSE difference', float(np.abs(m.sse - m.sse_old).max()), flush=True)
    t0 = time.time(); features(specs['BEAT_T1'], 0, [100, 200], OUT / 'features'); print('features job', round(time.time() - t0), 's', flush=True)
    print('smoke test finished', flush=True)


def main():
    L.low_priority(); L.stay_awake()
    tasks = json.loads((T16 / 'tasks.json').read_text(encoding='utf-8'))
    g1 = g1_specs()
    g2 = [t for t in tasks if t['analysis'] == 'W' and t['size'] >= 30 and t['source'] in SENS]
    g3 = json.loads((HERE / 'tasks_g3.json').read_text(encoding='utf-8'))
    for t in g3:
        t['pool'] = True
    assert (len(g1), len(g2), len(g3)) == (40, 12, 144), (len(g1), len(g2), len(g3))
    stages = [
        ('G1', [delayed(job_g1)(s, o, [50, 100, largest(s)]) for s in g1 for o in range(5)]),
        ('G2', [delayed(job_g2)(s, o, 100) for s in g2 for o in range(5)]),
        ('G3_dev_features', [delayed(features)(s, o, [50, 100, 200], OUT / 'G3_dev_features') for s in g1 for o in range(5)]),
        ('G3', [delayed(job_g3)(s, o, [50, 100, 200]) for s in g3 for o in range(3)]),
        ('G3_test_features', [delayed(features)(s, o, [50, 100, 200], OUT / 'G3_test_features') for s in g3 for o in range(3)]),
    ]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        for name, jobs in stages:
            t0 = time.time()
            print(f'=== stage {name}: {len(jobs)} jobs', flush=True)
            Parallel(n_jobs=N_JOBS, verbose=1)(jobs)
            print(f'=== stage {name} finished in {round((time.time() - t0) / 60)} min', flush=True)
            if name == 'G1':
                check_g1(g1)
            elif name == 'G2':
                check_g2(g2)
    print('GATES FINISHED', flush=True)


if __name__ == '__main__':
    smoke() if sys.argv[1:] == ['check'] else main()
