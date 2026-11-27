#!/usr/bin/env python3
"""R5 risk PILOT on development data (GDSC2 drug-as-measurement tasks from Task 8; already used in Tasks 8-11).
Not a confirmatory analysis. Question: with the same predictor, does a target-aware panel beat the prior art's
unsupervised decorrelation panel when few drugs are measured, and how does that depend on training size?

Panels of k drugs, chosen on the training units only:
  random   : k random candidates
  decorr   : Abdel-Rehim et al. rule - drop the lowest-variance 50%, rank by median Pearson correlation, greedy
             removal of candidates correlated above a threshold; threshold set so the panel has k drugs
  target   : greedy forward selection minimising 5-fold CV error of a multi-target ridge (target-aware)
Predictor for all: RidgeCV on the k drugs. Score: explained variance on held-out units (train-standardised).
Run with .env312:  python pilot_panels.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

W = Path(__file__).resolve().parent.parent
T8 = W / 'task8_real_cost'
TASKS = ['DRUG_GDSC2_T1', 'DRUG_GDSC2_T2', 'DRUG_GDSC2_T3', 'DRUG_GDSC2_T4']
KS, NS, OUTER, REPS = [3, 5, 10, 20, 30], [25, 50, 100, None], 5, 6
ALPHAS = np.logspace(-2, 4, 13)


def decorr(x, k):
    var = x.var(axis=0)
    keep = np.flatnonzero(var >= np.median(var))
    c = np.corrcoef(x[:, keep], rowvar=False)
    c = np.nan_to_num(c)
    order = keep[np.argsort(np.median(c, axis=1), kind='stable')]
    pos = {j: i for i, j in enumerate(keep)}

    def run(thr):
        avail, out = list(order), []
        while avail:
            s = avail.pop(0); out.append(s)
            avail = [j for j in avail if c[pos[s], pos[j]] <= thr]
        return out
    lo, hi = -1.0, 1.0
    for _ in range(40):          # smallest threshold whose panel has at least k drugs
        mid = (lo + hi) / 2
        if len(run(mid)) >= k:
            hi = mid
        else:
            lo = mid
    return np.array(run(hi)[:k])


def target_aware(xz, yz, k, rng):
    folds = np.array_split(rng.permutation(len(xz)), 5)
    chosen, left = [], list(range(xz.shape[1]))
    for _ in range(k):
        best, err_best = None, np.inf
        for j in left:
            cols = chosen + [j]; err = 0.0
            for h in folds:
                tr = np.setdiff1d(np.arange(len(xz)), h)
                m = Ridge(alpha=1.0).fit(xz[tr][:, cols], yz[tr])
                err += ((m.predict(xz[h][:, cols]) - yz[h]) ** 2).sum()
            if err < err_best:
                best, err_best = j, err
        chosen.append(best); left.remove(best)
    return np.array(chosen)


def score(xz, yz, xt, yt, cols):
    m = RidgeCV(alphas=ALPHAS).fit(xz[:, cols], yz)
    return float(((m.predict(xt[:, cols]) - yt) ** 2).sum()), float((yt ** 2).sum())


def main():
    rows = []
    for name in TASKS:
        task = next(t for t in json.loads((T8 / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
        m = pd.read_csv(T8 / 'matrices' / f'{name}.csv')
        x = m[[c for c in m if c.startswith('cand_')]].to_numpy(float)
        y = m[task['targets']].to_numpy(float)
        for o in range(OUTER):
            rng = np.random.default_rng([20261101, o])
            perm = rng.permutation(len(x)); test = perm[: len(x) // 5]; train = perm[len(x) // 5:]
            for n in NS:
                for r in (range(REPS) if n else [0]):
                    sub = train if n is None else np.random.default_rng([20261101, o, n, r]).choice(train, n, replace=False)
                    xs, ys = x[sub], y[sub]
                    xm, xsd = xs.mean(0), np.where(xs.std(0, ddof=1) == 0, 1, xs.std(0, ddof=1))
                    ym, ysd = ys.mean(0), np.where(ys.std(0, ddof=1) == 0, 1, ys.std(0, ddof=1))
                    xz, yz = (xs - xm) / xsd, (ys - ym) / ysd
                    xt, yt = (x[test] - xm) / xsd, (y[test] - ym) / ysd
                    ta = target_aware(xz, yz, max(KS), np.random.default_rng([7, o, r]))
                    for k in KS:
                        panels = {'random': np.random.default_rng([11, o, r, k]).choice(x.shape[1], k, replace=False),
                                  'decorr': decorr(xs, k), 'target': ta[:k]}
                        for meth, cols in panels.items():
                            sse, c0 = score(xz, yz, xt, yt, cols)
                            rows.append({'task': name, 'outer': o, 'n': 'all' if n is None else n, 'rep': r, 'k': k,
                                         'method': meth, 'sse': sse, 'c0': c0})
                    sse, c0 = score(xz, yz, xt, yt, np.arange(x.shape[1]))
                    rows.append({'task': name, 'outer': o, 'n': 'all' if n is None else n, 'rep': r, 'k': x.shape[1],
                                 'method': 'all', 'sse': sse, 'c0': c0})
        print(name, 'done', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(__file__).resolve().parent / 'pilot_panels.csv', index=False)
    ev = d.groupby(['n', 'method', 'k']).apply(lambda h: 1 - h.sse.sum() / h.c0.sum(), include_groups=False).rename('ev').reset_index()
    tab = ev[ev.method != 'all'].pivot_table(index=['n', 'k'], columns='method', values='ev')
    tab['target_minus_decorr'] = tab['target'] - tab['decorr']
    full = ev[ev.method == 'all'].set_index('n').ev
    print(tab.round(3).to_string())
    print('\nall 84 candidates (Ridge):', full.round(3).to_dict())


if __name__ == '__main__':
    main()
