#!/usr/bin/env python3
"""Task 4b (post hoc): R3 (one-SE rule on nested leave-one-participant-out risks) on the Radbill learning curve, as
specified in plan_r3.md (frozen 2026-10-29 00:38 NZDT). The 20 reps per m on which R1 was run, plus the full set.
Run with the base miniconda Python. Writes radbill_r3_cells.csv here."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import radbill_rules as RR  # noqa: E402
RL, m = RR.RL, RR.m


def middle_unit_losses(data, train):
    """Mean middle-held loss of each participant for each resolution (cells not estimable are skipped)."""
    out = {res: {} for res in RR.RES}
    for res in RR.RES:
        for middle in train:
            dmid = [u for u in train if u != middle]
            actions = m.radbill_select(data, dmid, res, {'selection_layer': 'TASK4B_MIDDLE'})[0]
            losses = []
            for t in m.RAD_TARGETS:
                if (middle, t) not in data['support']:
                    continue
                o = m.radbill_predict(data, dmid, middle, str(m.action_for(actions, res, t, RR.SYS)), t)
                if o is not None:
                    losses.append(abs(o[0] - data['ps'][middle]['target'][t]) / o[1])
            out[res][middle] = losses
    return out


def choose(units):
    # pooled risk over cells, exactly as R1; SE over participants of paired per-participant mean differences
    risk = {res: float(np.mean([v for u in units[res].values() for v in u])) for res in RR.RES}
    best = RR.pick(risk)
    common = [u for u in units[best] if all(units[res][u] for res in RR.RES)]
    for res in RR.RES:
        d = np.array([np.mean(units[res][u]) - np.mean(units[best][u]) for u in common])
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        if risk[res] <= risk[best] + se + 1e-12:
            return best, res


def main():
    data = m.load_all_data()[RR.SYS]
    units = list(data['units'])
    cells = []
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
                    result = RL.predict_outer(data, train, outer)
                    if result is not None:
                        break
                    attempt += 1
                r1, r3 = choose(middle_unit_losses(data, train))
                for c in result:
                    cells.append({'outer_participant': outer, 'm_train': size, 'rep': -1 if r is None else r, 'target': c['target'],
                                  'R1_check': c[r1], 'R3': c[r3], 'choice_R3': r3, 'choice_R1_check': r1})
            for cache in (m.PREDICTION_CACHE, m.MODEL_CACHE, m.SELECTION_CACHE):
                cache.clear()
        print('outer', outer, 'done', flush=True)
    cells = pd.DataFrame(cells)
    ref = pd.read_csv(HERE / 'radbill_rules_cells.csv')
    mg = cells.merge(ref[['outer_participant', 'm_train', 'rep', 'target', 'R1']], on=['outer_participant', 'm_train', 'rep', 'target'])
    assert len(mg) == len(cells)
    diff = float((mg.R1_check - mg.R1).abs().max())
    print('R1 reproduction max abs diff', diff)
    assert diff < 1e-10
    cells.to_csv(HERE / 'radbill_r3_cells.csv', index=False)


if __name__ == '__main__':
    main()
