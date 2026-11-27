#!/usr/bin/env python3
"""Task 36 (STAGE1_DEV_PLAN.md): one job = (source, p level, task, outer split) on the denser grid, recording training-only quantities for
several decision methods. Panels as in Task 34 (t34lib.paths). Errors are mean squared errors per sample and target on the
training-standardised scale. Test score = explained variance of held-out targets (as Task 34)."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task34_decision'))
import t34lib as T  # noqa: E402
R = T.R
NS, KS, KMAX = (50, 75, 100, 150, 200), (5, 8, 10, 12, 15, 20), 20


def zs(y):
    m, s = y.mean(0), y.std(0, ddof=1)
    return (y - m) / np.where(s == 0, 1, s)


def mse(xa, ya, xb, yb, cols):
    pred = R.ridge_predict(xa, ya, xb, np.asarray(cols, int))
    return ((pred - yb) ** 2).mean(1)                                       # per-sample MSE (mean over targets)


def run_job(task, frame, o, out_csv, ns=NS, ks=KS):
    t0 = time.time()
    cand, tg = task['candidates'], task['targets']
    X = frame[cand].to_numpy(float); Y = frame[tg].to_numpy(float); YH = frame[task['hist_targets']].to_numpy(float)
    hist_idx = np.array([cand.index(c) for c in task['hist_targets'] if c in cand], int)
    rng = np.random.default_rng([task['code'], o])
    perm = rng.permutation(len(X)); nte = int(round(0.2 * len(X))); te, trn = perm[:nte], perm[nte:]
    rows = []
    for n in ns:
        if n > len(trn):
            continue
        for r in range(T.REPS):
            sub = np.random.default_rng([task['code'], o, n, r]).choice(trn, n, replace=False)
            ym, ysd = Y[sub].mean(0), Y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys, yt = (Y[sub] - ym) / ysd, (Y[te] - ym) / ysd
            yh = zs(YH[sub]); xs = X[sub]
            P = T.paths(xs, ys, yh, hist_idx)
            rec = {k: {} for k in ks}
            for k in ks:
                for m in 'ABC':
                    rec[k][m] = T.ev(xs, ys, X[te], yt, P[m][:k])
                for m in 'AC':
                    rec[k]['app_' + m] = float(mse(xs, ys, xs, ys, P[m][:k]).mean())
            # 5-fold (A, B, C), with fold-wise A - C differences and fold models scored on all training units (Burman)
            for nf, tag, ms in [(5, 'cv5', 'ABC'), (10, 'cv10', 'AC')]:
                folds = np.random.default_rng([task['code'], o, n, r, 7 if nf == 5 else 11]).permutation(n) % nf
                err = {(m, k): np.zeros(n) for m in ms for k in ks}
                allf = {(m, k): 0.0 for m in 'AC' for k in ks}
                for f in range(nf):
                    a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                    ya = zs(ys[a]); mu, sd = ys[a].mean(0), ys[a].std(0, ddof=1); sd = np.where(sd == 0, 1, sd)
                    yb = (ys[b] - mu) / sd; yall = (ys - mu) / sd
                    Pi = T.paths(xs[a], ya, zs(yh[a]), hist_idx)
                    for m in ms:
                        for k in ks:
                            err[(m, k)][b] = mse(xs[a], ya, xs[b], yb, Pi[m][:k])
                            if nf == 5 and m in 'AC':
                                allf[(m, k)] += float(mse(xs[a], ya, xs, yall, Pi[m][:k]).mean()) / nf
                for k in ks:
                    for m in ms:
                        rec[k][f'{tag}_{m}'] = float(err[(m, k)].mean())
                    if nf == 5:
                        dif = [float((err[('A', k)][folds == f] - err[('C', k)][folds == f]).mean()) for f in range(nf)]
                        rec[k]['cv5_se_AmC'] = float(np.std(dif, ddof=1) / np.sqrt(nf))
                        for m in 'AC':
                            rec[k]['all5_' + m] = allf[(m, k)]
            # half splits (train on 0.5n, test on the rest), three repeats
            hs = {(m, k): [] for m in 'AC' for k in ks}
            for h in range(3):
                pp = np.random.default_rng([task['code'], o, n, r, 13, h]).permutation(n); a, b = pp[:n // 2], pp[n // 2:]
                ya = zs(ys[a]); mu, sd = ys[a].mean(0), ys[a].std(0, ddof=1); sd = np.where(sd == 0, 1, sd)
                Pi = T.paths(xs[a], ya, zs(yh[a]), hist_idx)
                for m in 'AC':
                    for k in ks:
                        hs[(m, k)].append(float(mse(xs[a], ya, xs[b], (ys[b] - mu) / sd, Pi[m][:k]).mean()))
            for k in ks:
                for m in 'AC':
                    rec[k]['half_' + m] = float(np.mean(hs[(m, k)]))
                rows.append(dict(source=task['source'], task=task['task'], p=task['p'], outer=o, n=n, rep=r, k=k, **rec[k]))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return time.time() - t0
