#!/usr/bin/env python3
"""Task 5: Radbill cache for R4 development (development data). The 20 draws per m used for R1, plus the full set.
Middle layer: leave-one-participant-out, as in V3; per-participant middle losses for C0 and each resolution.
Outer losses are taken from task4_adaptive/radbill_rules_cells.csv. Run with the base miniconda Python."""
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task4_adaptive'))
import radbill_rules as RR  # noqa: E402
RL, m = RR.RL, RR.m
RES = {'Shared': 'SHARED', 'Family': 'FAMILY', 'Exact': 'EXACT'}


def middle(data, train):
    out = {k: {} for k in ['C0', *RES]}
    for middle_unit in train:
        dmid = [u for u in train if u != middle_unit]
        acts = {k: m.radbill_select(data, dmid, v, {'selection_layer': 'TASK5_MIDDLE'})[0] for k, v in RES.items()}
        cells = {k: [] for k in out}
        for t in m.RAD_TARGETS:
            if (middle_unit, t) not in data['support']:
                continue
            obs = data['ps'][middle_unit]['target'][t]
            preds = {k: m.radbill_predict(data, dmid, middle_unit, str(m.action_for(acts[k], v, t, RR.SYS)), t) for k, v in RES.items()}
            ys = np.asarray([data['ps'][u]['target'][t] for u in dmid if data['ps'][u]['target'][t] is not None], float)
            sd = m.sample_sd(ys) if len(ys) >= 2 else math.nan
            if any(p is None for p in preds.values()) or not (math.isfinite(sd) and sd > 1e-12):
                continue   # common cells only, so that resolutions are compared on the same cells
            cells['C0'].append(abs(ys.mean() - obs) / sd)
            for k in RES:
                cells[k].append(abs(preds[k][0] - obs) / preds[k][1])
        for k in out:
            out[k][middle_unit] = cells[k]
    return out


def main():
    data = m.load_all_data()[RR.SYS]
    units = list(data['units'])
    ref = pd.read_csv(HERE.parent / 'task4_adaptive' / 'radbill_rules_cells.csv')
    recs = []
    for outer in units:
        pool = np.array([u for u in units if u != outer])
        for size in RL.SIZES:
            for r in ([None] if size == len(pool) else range(RR.R1_REPS)):
                attempt = 0
                while True:
                    if r is None:
                        train = [int(u) for u in pool]
                    else:
                        rng = np.random.default_rng([20261028, 2, int(outer), size, r, attempt])
                        train = sorted(int(u) for u in rng.choice(pool, size=size, replace=False))
                    if RL.predict_outer(data, train, outer) is not None:
                        break
                    attempt += 1
                rep = -1 if r is None else r
                cells = ref[(ref.outer_participant == outer) & (ref.m_train == size) & (ref.rep == rep)]
                recs.append({'outer': {'C0': cells.C0.sum(), 'Shared': cells.SHARED.sum(), 'Family': cells.FAMILY.sum(), 'Exact': cells.EXACT.sum()},
                             'n_test_cells': len(cells), 'outer_unit': outer, 'n_label': 'all' if r is None else str(size), 'rep': rep,
                             'n_train': size, 'middle': middle(data, train), 'n_candidates': 4, 'n_targets': len(m.RAD_TARGETS)})
            for cache in (m.PREDICTION_CACHE, m.MODEL_CACHE, m.SELECTION_CACHE):
                cache.clear()
        print('outer', outer, 'done', flush=True)
    (HERE / 'cache').mkdir(exist_ok=True)
    with open(HERE / 'cache' / 'radbill.pkl', 'wb') as f:
        pickle.dump(recs, f, protocol=4)
    print('records', len(recs))


if __name__ == '__main__':
    main()
