#!/usr/bin/env python3
"""Task 41 (post hoc): simulation of where redesign (somp) overtakes reuse (FSCA) as a function of n/k, pool size p and target signal.
Candidates follow a factor model (10 latent factors, communality 0.2-0.8). Nine targets in three families; each family's targets are
weighted sums of 8 family-specific candidate measurements plus a broad factor component, scaled so that the population R^2 of each
target on all candidates equals `signal`. Panels from the same code as the main analysis (gatelib.fsca, gatelib.somp, guarded RidgeCV);
score = explained variance on 2000 test units (training-standardised, as in the main analysis).
Usage: sim.py [workers] -> sim_results.csv, sim_summary.json"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task34_decision'))
PS, SIGS, NS, KS, REPS, NTEST, SEED = (71, 200, 1000), (0.02, 0.1, 0.25, 0.5, 0.75), (25, 50, 75, 100, 150, 200, 300, 400), (5, 10, 20), 16, 2000, 20264100


def draw(p, signal, n, rng):
    r = 10; N = n + NTEST
    load = rng.normal(size=(r, p)); comm = rng.uniform(0.2, 0.8, p)
    load *= np.sqrt(comm / (load ** 2).sum(0))
    F = rng.normal(size=(N, r)); X = F @ load + rng.normal(size=(N, p)) * np.sqrt(1 - comm)
    Y = np.empty((N, 9))
    for fam in range(3):
        S = rng.choice(p, 8, replace=False)
        for t in range(3):
            w = rng.normal(size=8); z = X[:, S] @ w
            z = 0.7 * z / z.std() + 0.3 * (F @ rng.normal(size=r)) / np.sqrt(r)
            z = z / z.std()
            Y[:, fam * 3 + t] = np.sqrt(signal) * z + np.sqrt(1 - signal) * rng.normal(size=N)
    return X[:n], Y[:n], X[n:], Y[n:]


def one(args):
    import t34lib as T
    T.GL.low_priority()
    p, sig, n, rep = args
    rng = np.random.default_rng([SEED, p, int(sig * 100), n, rep])
    xs, y, xt, yt0 = draw(p, sig, n, rng)
    m, s = y.mean(0), y.std(0, ddof=1); ys, yt = (y - m) / s, (yt0 - m) / s
    a = T.GL.fsca(xs, max(KS), 0); c = T.GL.somp(xs, ys, max(KS), 0)
    out = []
    for k in KS:
        out.append(dict(p=p, signal=sig, n=n, rep=rep, k=k, A=T.ev(xs, ys, xt, yt, a[:k]), C=T.ev(xs, ys, xt, yt, c[:k])))
    return out


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [(p, s, n, r) for p in PS for s in SIGS for n in NS for r in range(REPS)]
    with Pool(workers) as pl:
        rows = [x for res in pl.imap_unordered(one, jobs, chunksize=4) for x in res]
    d = pd.DataFrame(rows); d.to_csv(HERE / 'sim_results.csv', index=False)
    d['nk'] = d.n / d.k; d['gap'] = d.C - d.A
    g = d.groupby(['p', 'signal', 'nk']).gap.mean().reset_index()

    def cross(h):
        h = h.sort_values('nk'); v, x = h.gap.values, np.log(h.nk.values)
        for i in range(1, len(v)):
            if v[i - 1] < 0 <= v[i]:
                return float(np.exp(np.interp(0, [v[i - 1], v[i]], [x[i - 1], x[i]])))
        return None if (v < 0).all() else (float(h.nk.min()) if (v >= 0).all() else None)
    summ = {f'p{p}_s{s}': cross(h) for (p, s), h in g.groupby(['p', 'signal'])}
    json.dump(summ, open(HERE / 'sim_summary.json', 'w'), indent=1)
    print(summ)
