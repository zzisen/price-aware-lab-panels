#!/usr/bin/env python3
"""Task 42 (fourth locked test): rule v4, unchanged, applied with a different predictor or different selectors.
Pipelines (panels A = reusable, B = target-aware on historical targets, C = target-aware on current targets):
  'R' = ridge + FSCA + somp (the pipeline of Tasks 36-40; used only to check that this code path reproduces Task 40 rows exactly)
  'T' = gradient-boosted trees (50 trees, depth 2, one model per target, as in the benchmark) + FSCA + somp
  'S' = ridge + pivoted QR (reusable) + lasso (target-aware; gatelib.lasso computed separately for every k)
One job = (source, task, p, outer split 0). Grid n in (50, 75, 100, 150, 200), k in (5, 8, 10, 12, 15, 20), 4 repeats, as Task 36;
pipeline T uses n in (50, 100, 200), k in (5, 10, 20) and 2 repeats (GRID, REPS).
Recorded per cell: test explained variance of A, B, C; 5-fold CV mean squared error of A and C (panels re-selected in each fold,
targets re-standardised on the fold's training part), exactly as t36lib.run_job."""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import enet_path
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task40_locked_v4'))
import t40lib as L40  # noqa: E402
T = L40.T; GL = T.GL; R = T.R
NS, KS, KMAX = (50, 75, 100, 150, 200), (5, 8, 10, 12, 15, 20), 20


def zs(y):
    m, s = y.mean(0), y.std(0, ddof=1)
    return (y - m) / np.where(s == 0, 1, s)


def gbt_predict(xtr, ytr, xte, cols):
    cols = np.asarray(cols, int)
    if not len(cols):
        return np.repeat(ytr.mean(0, keepdims=True), len(xte), 0)
    return np.column_stack([GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0)
                            .fit(xtr[:, cols], ytr[:, t]).predict(xte[:, cols]) for t in range(ytr.shape[1])])


def ridge_predict(xtr, ytr, xte, cols):
    return R.ridge_predict(xtr, ytr, xte, np.asarray(cols, int))


def lasso_panels(xs, ys, ks):
    """gatelib.lasso(xs, ys, k) for every k in ks from one enet path (identical result, path computed once)."""
    xz = GL.standard(xs); yc = ys - ys.mean(0)
    _, coefs, _ = enet_path(xz, yc, l1_ratio=1.0, n_alphas=60, eps=1e-3, max_iter=2000, tol=1e-3)
    norm = np.sqrt((coefs ** 2).sum(0)); nnz = (norm > 0).sum(0); sc = GL.abs_corr(xz, yc).mean(1)
    out = {}
    for k in ks:
        hit = np.flatnonzero(nnz >= k); step = norm[:, hit[0] if len(hit) else -1]
        out[k] = GL.fill([j for j in np.argsort(-step, kind='stable') if step[j] > 0], sc, k)
    return out


PIPES = {'R': ('ridge', 'fsca', 'somp'), 'T': ('gbt', 'fsca', 'somp'), 'S': ('ridge', 'qr', 'lasso')}
PRED = {'ridge': ridge_predict, 'gbt': gbt_predict}
# grid per pipeline: trees are about 10x slower, so pipeline T uses the Task 34 grid (n 50/100/200, k 5/10/20)
GRID = {'R': (NS, KS), 'S': (NS, KS), 'T': ((50, 100, 200), (5, 10, 20))}
REPS = {'R': 4, 'S': 4, 'T': 2}                                       # repeats per (n): trees use 2


def panels(xs, ys, yh, hist_idx, pipe, need_b=True, KS=KS):
    """dict m -> {k: column indices}."""
    _, reuse, aware = PIPES[pipe]
    a = GL.fsca(xs, KMAX, 0) if reuse == 'fsca' else GL.qr(xs, KMAX, 0)
    P = {'A': {k: a[:k] for k in KS}}
    keep = np.setdiff1d(np.arange(xs.shape[1]), hist_idx)
    if aware == 'somp':
        c = GL.somp(xs, ys, KMAX, 0); P['C'] = {k: c[:k] for k in KS}
        if need_b:
            b = keep[GL.somp(xs[:, keep], yh, KMAX, 0)]; P['B'] = {k: b[:k] for k in KS}
    else:
        P['C'] = lasso_panels(xs, ys, KS)
        if need_b:
            P['B'] = {k: keep[v] for k, v in lasso_panels(xs[:, keep], yh, KS).items()}
    return P


def run_job(task, frame, o, out_csv, pipe):
    t0 = time.time(); pred = PRED[PIPES[pipe][0]]; NS_, KS_ = GRID[pipe]
    cand, tg = task['candidates'], task['targets']
    X = frame[cand].to_numpy(float); Y = frame[tg].to_numpy(float); YH = frame[task['hist_targets']].to_numpy(float)
    hist_idx = np.array([cand.index(c) for c in task['hist_targets'] if c in cand], int)
    rng = np.random.default_rng([task['code'], o])
    perm = rng.permutation(len(X)); nte = int(round(0.2 * len(X))); te, trn = perm[:nte], perm[nte:]
    rows = []
    for n in NS_:
        if n > len(trn):
            continue
        for r in range(REPS[pipe]):
            sub = np.random.default_rng([task['code'], o, n, r]).choice(trn, n, replace=False)
            ym, ysd = Y[sub].mean(0), Y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys, yt = (Y[sub] - ym) / ysd, (Y[te] - ym) / ysd
            yh = zs(YH[sub]); xs = X[sub]
            P = panels(xs, ys, yh, hist_idx, pipe, KS=KS_)
            rec = {k: {} for k in KS_}
            for k in KS_:
                for m in 'ABC':
                    p_ = pred(xs, ys, X[te], P[m][k])
                    rec[k][m] = 1 - float(((p_ - yt) ** 2).sum()) / float((yt ** 2).sum())
            folds = np.random.default_rng([task['code'], o, n, r, 7]).permutation(n) % 5
            err = {(m, k): np.zeros(n) for m in 'AC' for k in KS_}
            for f in range(5):
                a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                ya = zs(ys[a]); mu, sd = ys[a].mean(0), ys[a].std(0, ddof=1); sd = np.where(sd == 0, 1, sd)
                yb = (ys[b] - mu) / sd
                Pi = panels(xs[a], ya, zs(yh[a]), hist_idx, pipe, need_b=False, KS=KS_)
                for m in 'AC':
                    for k in KS_:
                        err[(m, k)][b] = ((pred(xs[a], ya, xs[b], Pi[m][k]) - yb) ** 2).mean(1)
            for k in KS_:
                rows.append(dict(source=task['source'], task=task['task'], p=task['p'], outer=o, n=n, rep=r, k=k, pipe=pipe,
                                 A=rec[k]['A'], B=rec[k]['B'], C=rec[k]['C'],
                                 cv5_A=float(err[('A', k)].mean()), cv5_C=float(err[('C', k)].mean())))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return time.time() - t0
