"""R5 (development version, NOT frozen; see R5_DESIGN.md).

Step 1  target-aware selection at a target resolution: targets are split into groups (Shared: one group; Family: one
        group per family; Exact: one group per target). A joint greedy forward selection grows each group's measurement
        set; adding a measurement already in the union is free, a new one costs one unit of budget. Each step takes the
        addition with the largest drop in guarded inner-CV error (free additions first); it stops when no addition
        lowers the error or the union reaches KMAX. The path gives the panel for every budget k <= KMAX.
Step 2  refit: ridge per target on its group's set ('group') or on the whole union ('union').
Step 3  resolution and refit mode chosen per budget by a nested middle-layer CV, ties to the coarser option, C0 included.
Guard  as R4: a test unit with any used measurement beyond the training range + GUARD_MARGIN SDs is predicted by the
        training mean."""
import numpy as np
from sklearn.linear_model import RidgeCV

GUARD_MARGIN = 1.0
KMAX = 30
SEL_ALPHAS = np.array([0.03, 0.3, 3.0, 30.0])        # x n_train: ridge penalties tried inside selection
FIT_ALPHAS = np.logspace(-2, 4, 13)
RES5 = ['Shared', 'Family', 'Exact']
MODES = ['group', 'union']


def folds_of(n, k, seed):
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def zstats(x):
    m = x.mean(0); s = x.std(0, ddof=1)
    return m, np.where(s == 0, 1.0, s)


def out_of_range(xtr, xte):
    m, s = zstats(xtr)
    return np.abs((xte - m) / s) > np.abs((xtr - m) / s).max(0) + GUARD_MARGIN


def groups_of(res, fam):
    t = np.arange(len(fam))
    if res == 'Shared':
        return [t]
    if res == 'Family':
        return [t[fam == f] for f in dict.fromkeys(fam)]
    return [np.array([i]) for i in t]


def add_errors(x, y, S, folds):
    """Guarded CV SSE (best alpha) of ridge on S + [j], for every candidate j (J,), and of S alone (scalar)."""
    J = x.shape[1]; p = len(S) + 1
    tot = np.zeros((len(SEL_ALPHAS), J)); base = np.zeros(len(SEL_ALPHAS))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(x)), te)
        m, s = zstats(x[tr]); xz = (x - m) / s
        ym = y[tr].mean(0); yt = y[tr] - ym; ye = y[te] - ym
        A, B = xz[tr][:, S], xz[tr]                       # (ntr,p-1), (ntr,J)
        oor = out_of_range(x[tr], x[te])                  # (nte,J)
        okA = ~oor[:, S].any(1) if len(S) else np.ones(len(te), bool)
        G = np.empty((J, p, p)); R = np.empty((J, p, y.shape[1]))
        G[:, :p - 1, :p - 1] = A.T @ A
        ab = A.T @ B
        G[:, :p - 1, p - 1] = ab.T; G[:, p - 1, :p - 1] = ab.T; G[:, p - 1, p - 1] = (B * B).sum(0)
        R[:, :p - 1, :] = (A.T @ yt)[None]; R[:, p - 1, :] = B.T @ yt
        Ae, Be = xz[te][:, S], xz[te]
        c0 = (ye ** 2).sum(1)                             # per test unit, fallback error
        for ia, a in enumerate(SEL_ALPHAS * len(tr)):
            W = np.linalg.solve(G + a * np.eye(p)[None], R)                     # (J,p,T)
            # an empty inner dimension (no measurement chosen yet) contributes zero; np.einsum can return inf/NaN there in some processes
            ein = np.einsum('ip,jpt->jit', Ae, W[:, :p - 1]) if Ae.shape[1] else np.zeros((W.shape[0], Ae.shape[0], W.shape[2]))
            pred = ein + Be.T[:, :, None] * W[:, p - 1][:, None, :]
            err = ((pred - ye[None]) ** 2).sum(2)                               # (J,nte)
            ok = okA[None] & ~oor.T
            tot[ia] += np.where(ok, err, c0[None]).sum(1)
            if len(S):
                w = np.linalg.solve(A.T @ A + a * np.eye(p - 1), A.T @ yt)
                e = ((Ae @ w - ye) ** 2).sum(1)
                base[ia] += np.where(okA, e, c0).sum()
            else:
                base[ia] += c0.sum()
    return tot.min(0), base.min()


def select_path(x, y, fam, res, seed, kmax=KMAX):
    """Joint greedy path. Returns {k: [set per group]} for k = 1..stop (the state reached with union size k)."""
    G = groups_of(res, fam); fo = folds_of(len(x), 5, seed)
    S = [[] for _ in G]; U = []
    cache = [add_errors(x, y[:, g], [], fo) for g in G]
    path = {}
    while True:
        best = None
        for free in (True, False):
            for gi, (err, cur) in enumerate(cache):
                gain = cur - err
                gain[S[gi]] = -np.inf
                mask = np.isin(np.arange(x.shape[1]), U)
                gain[~mask if free else mask] = -np.inf
                j = int(np.argmax(gain))
                if gain[j] > 1e-9 * max(cur, 1e-12) and (best is None or gain[j] > best[0]):
                    best = (gain[j], gi, j)
            if best is not None:
                break
        if best is None or (best[2] not in U and len(U) >= kmax):
            break
        _, gi, j = best
        if j not in U:
            if U:
                path[len(U)] = [list(s) for s in S]
            U.append(j)
        S[gi].append(j)
        cache[gi] = add_errors(x, y[:, G[gi]], S[gi], fo)
    if U:
        path[len(U)] = [list(s) for s in S]
    return G, path


def panel_at(path, k):
    """Sets per group for budget k: the largest recorded state with union size <= k (None -> C0)."""
    ks = [b for b in path if b <= k]
    return path[max(ks)] if ks else None


def ridge_predict(xtr, ytr, xte, cols):
    """RidgeCV on standardised cols, guarded; returns predictions on the ytr scale."""
    if not len(cols):
        return np.repeat(ytr.mean(0, keepdims=True), len(xte), 0)
    m, s = zstats(xtr[:, cols])
    mdl = RidgeCV(alphas=FIT_ALPHAS).fit((xtr[:, cols] - m) / s, ytr)
    pred = mdl.predict((xte[:, cols] - m) / s).reshape(len(xte), -1)
    bad = out_of_range(xtr[:, cols], xte[:, cols]).any(1)
    pred[bad] = ytr.mean(0)
    return pred


def r5_predict(xtr, ytr, xte, G, sets, mode):
    pred = np.empty((len(xte), ytr.shape[1]))
    union = sorted({j for s in sets for j in s}) if sets else []
    for g, s in zip(G, sets if sets else [[] for _ in G]):
        pred[:, g] = ridge_predict(xtr, ytr[:, g], xte, s if mode == 'group' else union)
    return pred


def options():
    return ['C0'] + [f'{r}|{m}' for r in RES5 for m in MODES if not (r == 'Shared' and m == 'union')]


def middle_choice(x, y, fam, ks, seed):
    """Nested middle layer (one 5-fold split): per budget k, the option with the lowest held-out SSE."""
    opts = options(); sse = {k: dict.fromkeys(opts, 0.0) for k in ks}
    for i, te in enumerate(folds_of(len(x), 5, seed + 7919)):
        tr = np.setdiff1d(np.arange(len(x)), te)
        ym = y[tr].mean(0)
        c0 = float(((y[te] - ym) ** 2).sum())
        paths = {r: select_path(x[tr], y[tr], fam, r, seed + 100 + i) for r in RES5}
        for k in ks:
            sse[k]['C0'] += c0
            for o in opts[1:]:
                r, mode = o.split('|')
                G, path = paths[r]; sets = panel_at(path, k)
                if sets is None:
                    sse[k][o] += c0
                    continue
                sse[k][o] += float(((r5_predict(x[tr], y[tr], x[te], G, sets, mode) - y[te]) ** 2).sum())
    out = {}
    for k in ks:
        v = np.array([sse[k][o] for o in opts])
        out[k] = opts[int(np.flatnonzero(v <= v.min() * (1 + 1e-12))[0])]      # ties -> first (coarser)
    return out, sse
