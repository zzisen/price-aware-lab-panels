#!/usr/bin/env python3
"""Task 17 (plan_gates): baseline panel rules, the structure features of G3 and the shared run set-up. Every rule sees the training subsample only.
Target-aware rules take (xs, ys, k, seed); reusable rules take (xs, k, seed). Each returns an int array of k candidate indices."""
import ctypes
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import qr as scipy_qr
from scipy.spatial.distance import squareform
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, enet_path

W = Path(__file__).resolve().parent.parent
warnings.filterwarnings('ignore')


class SafeNP:
    """numpy, except that einsum over an empty inner dimension returns zeros (task16_route_b DEVIATIONS D2)."""
    def __getattr__(self, a):
        return getattr(np, a)

    @staticmethod
    def einsum(subs, a, b):
        assert subs == 'ip,jpt->jit'
        if a.shape[1] == 0:
            return np.zeros((b.shape[0], a.shape[0], b.shape[2]))
        return np.einsum(subs, a, b)


def low_priority():
    k = ctypes.WinDLL('kernel32')
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)


def stay_awake():
    """Holds off idle sleep until this process exits. No system setting is changed."""
    k = ctypes.WinDLL('kernel32')
    k.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    k.SetThreadExecutionState(0x80000001)


def r5_libs():
    """The frozen R5 modules with the fix applied."""
    sys.path.insert(0, str(W / 'task12_r5')); sys.path.insert(0, str(W / 'task5_validation'))
    import r5lib as R
    import dev_r5 as V
    R.np = SafeNP()
    return R, V


def standard(x):
    m = x.mean(0); s = x.std(0, ddof=1)
    return (x - m) / np.where(s == 0, 1.0, s)


def fill(chosen, scores, k):
    """Chosen indices in order, topped up to k by descending score."""
    out = [int(j) for j in dict.fromkeys(chosen)]
    for j in np.argsort(-np.asarray(scores), kind='stable'):
        if len(out) >= k:
            break
        if int(j) not in out:
            out.append(int(j))
    return np.array(out[:k], int)


def abs_corr(xz, yz):
    return np.abs(xz.T @ yz) / (len(xz) - 1)


# ---- target-aware ----
def lasso(xs, ys, k, seed):
    xz = standard(xs); yc = ys - ys.mean(0)
    _, coefs, _ = enet_path(xz, yc, l1_ratio=1.0, n_alphas=60, eps=1e-3, max_iter=2000, tol=1e-3)
    norm = np.sqrt((coefs ** 2).sum(0))                                 # (p, alphas)
    hit = np.flatnonzero((norm > 0).sum(0) >= k)
    step = norm[:, hit[0] if len(hit) else -1]
    return fill([j for j in np.argsort(-step, kind='stable') if step[j] > 0], abs_corr(xz, yc).mean(1), k)


def somp(xs, ys, k, seed):
    xz = standard(xs); yc = ys - ys.mean(0)
    res, S = yc.copy(), []
    for _ in range(k):
        score = ((xz.T @ res) ** 2).sum(1)
        score[S] = -1
        S.append(int(np.argmax(score)))
        A = xz[:, S]
        res = yc - A @ np.linalg.solve(A.T @ A + 1e-6 * len(xz) * np.eye(len(S)), A.T @ yc)
    return np.array(S, int)


def mrmr(xs, ys, k, seed):
    xz = standard(xs); yz = standard(ys)
    n = len(xz)
    r2 = np.minimum(abs_corr(xz, yz) ** 2, 1 - 1e-9)
    rel = (r2 / (1 - r2) * (n - 2)).mean(1)                             # mean F statistic over targets
    c = np.abs(np.nan_to_num(np.corrcoef(xz, rowvar=False)))
    S = [int(np.argmax(rel))]
    while len(S) < k:
        score = rel / np.maximum(c[:, S].mean(1), 1e-3)
        score[S] = -np.inf
        S.append(int(np.argmax(score)))
    return np.array(S, int)


def pls_vip(xs, ys, k, seed):
    xz = standard(xs); yc = ys - ys.mean(0)
    a = min(5, len(xz) - 1, xz.shape[1])
    m = PLSRegression(n_components=a, scale=False).fit(xz, yc)
    t, wts, q = m.x_scores_, m.x_weights_, m.y_loadings_
    ss = (t ** 2).sum(0) * (q ** 2).sum(0)
    w2 = (wts / np.linalg.norm(wts, axis=0)) ** 2
    return fill([], np.sqrt(xz.shape[1] * (w2 @ ss) / ss.sum()), k)


_last = {}


def once(name, xs, seed, f):
    """Importances do not depend on k, so each subsample fits them once."""
    key = (id(xs), seed)
    if name not in _last or _last[name][0] != key:
        _last[name] = (key, f())
    return _last[name][1]


def rf(xs, ys, k, seed):
    imp = once('rf', xs, seed, lambda: RandomForestRegressor(n_estimators=200, max_features=1 / 3, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(xs, ys).feature_importances_)
    return fill([], imp, k)


def gbt_imp(xs, ys, k, seed):
    imp = once('gbt_imp', xs, seed, lambda: np.mean([GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=seed).fit(xs, ys[:, t]).feature_importances_ for t in range(ys.shape[1])], axis=0))
    return fill([], imp, k)


def univ(xs, ys, k, seed):
    return fill([], abs_corr(standard(xs), standard(ys)).mean(1), k)


# ---- reusable ----
def qr(xs, k, seed):
    _, _, piv = scipy_qr(standard(xs), mode='economic', pivoting=True)
    return fill(piv.tolist(), xs.var(0), k)


def fsca(xs, k, seed):
    """Forward selection: each step takes the measurement whose component explains most of what remains of all candidates."""
    e = standard(xs); S = []
    for _ in range(k):
        norm = (e ** 2).sum(0)
        score = np.where(norm > 1e-9, ((e.T @ e) ** 2).sum(1) / np.maximum(norm, 1e-9), -1)
        score[S] = -1
        j = int(np.argmax(score))
        S.append(j)
        q = e[:, j] / max(np.linalg.norm(e[:, j]), 1e-9)
        e = e - np.outer(q, q @ e)
    return np.array(S, int)


def medoid(xs, k, seed):
    c = np.abs(np.nan_to_num(np.corrcoef(standard(xs), rowvar=False)))
    d = 1 - c; np.fill_diagonal(d, 0); d = (d + d.T) / 2
    lab = fcluster(linkage(squareform(d, checks=False), 'average'), t=k, criterion='maxclust')
    S = []
    for g in np.unique(lab):
        m = np.flatnonzero(lab == g)
        S.append(int(m[np.argmax(c[np.ix_(m, m)].mean(1))]))
    return fill(S, xs.var(0), k)


def pca_loading(xs, k, seed):
    _, _, vt = np.linalg.svd(standard(xs), full_matrices=False)
    S = []
    for pc in vt[:k]:
        a = np.abs(pc); a[S] = -1
        S.append(int(np.argmax(a)))
    return fill(S, xs.var(0), k)


def leverage(xs, k, seed):
    _, _, vt = np.linalg.svd(standard(xs), full_matrices=False)
    return fill([], (vt[:k] ** 2).sum(0), k)


def high_var(xs, k, seed):
    return fill([], xs.var(0, ddof=1), k)


AWARE = {'lasso': lasso, 'somp': somp, 'mrmr': mrmr, 'pls_vip': pls_vip, 'rf': rf, 'gbt_imp': gbt_imp, 'univ': univ}
REUSE = {'qr': qr, 'fsca': fsca, 'medoid': medoid, 'pca_loading': pca_loading, 'leverage': leverage, 'high_var': high_var}


# ---- G3 ----
def ev_of(sse, c0):
    return 1 - sse / c0


def structure(xs, ys, ks, seed):
    """5-fold CV explained variance of principal components (A), PLS components (B) and all candidates (All), from the training subsample only."""
    n, p = xs.shape
    folds = np.array_split(np.random.default_rng(seed).permutation(n), 5)
    sa = dict.fromkeys(ks, 0.0); sb = dict.fromkeys(ks, 0.0); sall = 0.0; c0 = 0.0
    alphas = np.logspace(-2, 4, 13)
    for te in folds:
        tr = np.setdiff1d(np.arange(n), te)
        m = xs[tr].mean(0); s = xs[tr].std(0, ddof=1); s = np.where(s == 0, 1.0, s)
        xa, xe = (xs[tr] - m) / s, (xs[te] - m) / s
        ym = ys[tr].mean(0); yt, ye = ys[tr] - ym, ys[te] - ym
        c0 += float((ye ** 2).sum())
        sall += float(((RidgeCV(alphas=alphas).fit(xa, yt).predict(xe).reshape(len(te), -1) - ye) ** 2).sum())
        _, _, vt = np.linalg.svd(xa, full_matrices=False)
        for k in ks:
            a = min(k, len(tr) - 2, p)
            pa, pe = xa @ vt[:a].T, xe @ vt[:a].T
            w = np.linalg.solve(pa.T @ pa + 1e-3 * len(tr) * np.eye(a), pa.T @ yt)
            sa[k] += float(((pe @ w - ye) ** 2).sum())
            pls = PLSRegression(n_components=a, scale=False).fit(xa, yt)
            sb[k] += float(((pls.predict(xe).reshape(len(te), -1) - ye) ** 2).sum())
    return {k: {'A': ev_of(sa[k], c0), 'B': ev_of(sb[k], c0), 'All': ev_of(sall, c0)} for k in ks}
