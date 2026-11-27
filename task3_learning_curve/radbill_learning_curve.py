#!/usr/bin/env python3
"""Task 3 (post hoc): Radbill learning curve, as specified in plan_learning_curve.md (frozen 2026-10-28 20:55 NZDT).

Leave-one-contributor-out outer folds as in V3.2; the training participants are subsampled. Selection and
prediction reuse the frozen V3 functions unchanged. Run with the base miniconda Python (V3 code environment)
from any directory. Writes radbill_runs.csv and radbill_cells.csv here.
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v3_release_copy' / 'analysis' / 'v3' / 'code'))
import adaptive_resolution as m  # noqa: E402

SIZES = [6, 8, 10, 12, 14, 16]
REPS = 50
POLICIES = ['C0', 'SHARED', 'FAMILY', 'EXACT', 'CONVENTIONAL']
CONVENTIONAL = {'QT': 'DTend_QT', 'PP': 'DTend_PP'}
V32 = {'C0': 0.8495551603003364, 'SHARED': 0.854274384776178, 'FAMILY': 0.5854607893338551,
       'EXACT': 0.5117711175951137, 'CONVENTIONAL': 0.5046457996976736}


def predict_outer(data, train, outer):
    """Standardized absolute errors for every scored target of `outer`; None if any is not estimable."""
    selected = {res: m.radbill_select(data, train, res, {'selection_layer': 'LEARNING_CURVE'})[0]
                for res in ('SHARED', 'FAMILY', 'EXACT')}
    cells = []
    for target in m.RAD_TARGETS:
        if (outer, target) not in data['support']:
            continue
        observed = float(data['ps'][outer]['target'][target])
        ys = np.asarray([data['ps'][u]['target'][target] for u in train if data['ps'][u]['target'][target] is not None], float)
        if len(ys) < 2:
            return None
        sd = m.sample_sd(ys)
        if not math.isfinite(sd) or sd <= 1e-12:
            return None
        row = {'target': target, 'C0': abs(float(ys.mean()) - observed) / sd}
        for res in ('SHARED', 'FAMILY', 'EXACT'):
            cand = str(m.action_for(selected[res], res, target, 'RADBILL_HUMAN_PACING'))
            out = m.radbill_predict(data, train, outer, cand, target)
            if out is None:
                return None
            row[res] = abs(out[0] - observed) / out[1]
            row[res + '_measurement'] = cand
        out = m.radbill_predict(data, train, outer, CONVENTIONAL[m.RAD_FAMILY[target]], target)
        if out is None:
            return None
        row['CONVENTIONAL'] = abs(out[0] - observed) / out[1]
        cells.append(row)
    return cells


def main():
    data = m.load_all_data()['RADBILL_HUMAN_PACING']
    units = list(data['units'])
    assert len(units) == 17 and len(data['support']) == 126
    cells = []
    for outer in units:
        pool = np.array([u for u in units if u != outer])
        for size in SIZES:
            reps = [None] if size == len(pool) else range(REPS)
            for r in reps:
                redraws, attempt = 0, 0
                while True:
                    if r is None:
                        train = [int(u) for u in pool]
                    else:
                        rng = np.random.default_rng([20261028, 2, int(outer), size, r, attempt])
                        train = sorted(int(u) for u in rng.choice(pool, size=size, replace=False))
                    result = predict_outer(data, train, outer)
                    if result is not None:
                        break
                    if r is None:
                        raise RuntimeError('full training set not estimable')
                    redraws += 1; attempt += 1
                for c in result:
                    cells.append({'outer_participant': outer, 'm_train': size, 'rep': -1 if r is None else r,
                                  'redraws': redraws, **c})
            for cache in (m.PREDICTION_CACHE, m.MODEL_CACHE, m.SELECTION_CACHE):
                cache.clear()
        print('outer', outer, 'done', flush=True)
    cells = pd.DataFrame(cells)
    cells.to_csv(HERE / 'radbill_cells.csv', index=False)
    full = cells[cells.m_train == 16]
    assert len(full) == 126
    check = pd.DataFrame([{'policy': p, 'rerun': full[p].mean(), 'v32': V32[p], 'abs_diff': abs(full[p].mean() - V32[p])}
                          for p in POLICIES])
    check.to_csv(HERE / 'radbill_check_full_m.csv', index=False)
    print(check.to_string(index=False))
    assert check.abs_diff.max() < 1e-10, 'full-m rerun does not reproduce V3.2'
    # per draw: rep index r is shared across outer participants, so "draw r" pools all 17 outer folds
    runs = cells.groupby(['m_train', 'rep'])[POLICIES].mean().reset_index()
    runs['n_cells'] = cells.groupby(['m_train', 'rep']).size().values
    runs.to_csv(HERE / 'radbill_runs.csv', index=False)
    print('full-m reproduction OK; redraw events', int(cells.drop_duplicates(['outer_participant', 'm_train', 'rep']).redraws.sum()))


if __name__ == '__main__':
    main()
