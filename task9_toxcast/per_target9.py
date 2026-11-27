#!/usr/bin/env python3
"""Task 9: per-target outer losses at the full training size, for R4, C0 and the full-panel Ridge.

Copy of task8_real_cost/per_target.py with two changes, both forced by the approved deviation (only n = all is run):
  - HERE points at this folder, so tasks.json / matrices / validation are read from here;
  - L.GRID is fixed to [None] so that subsamples() yields exactly the rows in validation/<task>.csv.
Everything else, including the R4 choice and the frozen code paths, is unchanged.
Run with .env312:  python per_target9.py <task>   -> per_target/<task>.csv"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L  # noqa: E402
import validate_generic  # noqa: E402
from validate_standard import RIDGE_ALPHAS  # noqa: E402
from target_resolution import group_folds  # noqa: E402


def main(name):
    L.HERE = HERE
    task = next(t for t in json.loads((L.HERE / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    L.GRID = [None]
    _, x, y, fam = L.load_task(name)
    v = pd.read_csv(L.HERE / 'validation' / f'{name}.csv')
    T = y.shape[1]
    sse = {k: np.zeros(T) for k in ['C0', 'R4', 'Ridge']}
    cells = np.zeros(T)
    used = []
    for i, (ev, seed, fold, n, r, sub, held, groups) in enumerate(validate_generic.subsamples(task, x)):
        if n is not None:
            continue
        assert v.outer_seed[i] == seed and v.fold[i] == fold and v.n_train[i] == len(sub)
        inner = seed + fold + 1
        xs, ys, gs = x[sub], y[sub], groups[sub]
        Lg, _ = L.unit_losses_g(xs, ys, gs, 5, inner)
        ch = L.choices(Lg.mean(axis=0), fam)
        cand, c0, _ = L.guarded_losses(xs, ys[:, :, None], x[held], y[held, :, None])
        r4 = v.choice_R4[i]
        used.append([] if r4 == 'C0' else sorted({int(a) for a in ch[r4]}))
        sse['C0'] += c0.sum(axis=0)
        sse['R4'] += c0.sum(axis=0) if r4 == 'C0' else cand[:, ch[r4], np.arange(T)].sum(axis=0)
        xm = xs.mean(axis=0); xsd = xs.std(axis=0, ddof=1); xsd = np.where(xsd == 0, 1.0, xsd)
        ym = ys.mean(axis=0); ysd = ys.std(axis=0, ddof=1); ysd = np.where(ysd == 0, 1.0, ysd)
        idx = np.arange(len(xs))
        cv = [(np.setdiff1d(idx, h), h) for h in group_folds(gs, 5, inner)]
        m = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv).fit((xs - xm) / xsd, (ys - ym) / ysd)
        sse['Ridge'] += ((m.predict((x[held] - xm) / xsd) - (y[held] - ym) / ysd) ** 2).sum(axis=0)
        cells += len(held)
    out = pd.DataFrame({'target': task['targets'], 'family': task['families'], 'cells': cells,
                        **{f'sse_{k}': s for k, s in sse.items()}})
    out['ev_R4'] = 1 - out.sse_R4 / out.sse_C0
    out['ev_Ridge'] = 1 - out.sse_Ridge / out.sse_C0
    (HERE / 'per_target').mkdir(exist_ok=True)
    out.to_csv(HERE / 'per_target' / f'{name}.csv', index=False)
    feats = [c for c in pd.read_csv(L.HERE / 'matrices' / f'{name}.csv', nrows=0).columns
             if c not in task['targets'] + ['unit', 'tissue_group']]
    (HERE / 'per_target' / f'{name}_r4_features.json').write_text(
        json.dumps([[feats[a] for a in u] for u in used]), encoding='utf-8')
    print(name, 'done')


if __name__ == '__main__':
    main(sys.argv[1])
