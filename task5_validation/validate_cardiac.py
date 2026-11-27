#!/usr/bin/env python3
"""Task 5 validation on the V3 cardiac systems not used in development: Awinda, RLC-1, Tanner (plan_r4_frozen.md).
Outer and middle layers are leave-one-unit-out, exactly as in V3's adaptive procedure. Rules:
  R1  V3 rule: lowest middle risk among the resolutions (must reproduce the V3 selections);
  R3  coarsest resolution within one paired SE (over middle units) of the minimum;
  R4  lowest middle risk among C0 and the resolutions (no guard in the V3 predictors).
Run with the base miniconda Python. Writes validation/cardiac_<system>.csv (unit-target rows) and cardiac_choices.csv."""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v3_release_copy' / 'analysis' / 'v3' / 'code'))
import adaptive_resolution as m  # noqa: E402

SYSTEMS = ['RLC1_RAT', 'TANNER_RAT_TRABECULA', 'AWINDA_MOUSE']


def c0_rows(system, data, train, test):
    rows = []
    if system == 'RLC1_RAT':
        idx = {u: i for i, u in enumerate(data['units'])}
        for t in m.TARGETS_RLC:
            y = data['y'][t]; tr = [idx[u] for u in train]
            ym, ys = float(np.mean(y[tr])), m.sample_sd(y[tr])
            rows.append({'outer_unit': test, 'target': t, 'loss': abs(float(y[idx[test]]) - ym) / ys})
    elif system == 'TANNER_RAT_TRABECULA':
        for t in m.TANNER_TARGETS:
            ym = float(np.mean([data['y'][t][u] for u in train]))
            rows.append({'outer_unit': test, 'target': t, 'loss': abs(float(data['y'][t][test]) - ym)})
    else:
        test_ix = data['mouse_cell_ix'][str(test)]
        for t in m.AW_TARGETS:
            pred = m.awinda_predict(data, train, str(test), t, None)
            obs = data['Y'][t][test_ix]
            nrmse = 100.0 * np.sqrt(np.mean((pred - obs) ** 2, axis=1)) / np.sqrt(np.mean(obs ** 2, axis=1))
            rows += [{'outer_unit': str(test), 'target': t, 'loss': float(v)} for v in nrmse]
    return rows


def policy_rows(system, data, train, test, policy):
    if policy == 'C0':
        return c0_rows(system, data, train, test)
    actions, _ = m.SELECTORS[system](data, train, policy, {'selection_layer': 'TASK5_VALIDATION'})
    if system == 'AWINDA_MOUSE':
        return m.predict_awinda(data, train, str(test), policy, actions)[1]
    return m.predict_non_awinda(system, data, train, test, policy, actions)


def pick(risk, order):
    best = min(risk[k] for k in order)
    return next(k for k in order if risk[k] <= best + 1e-12)


def main():
    all_choices = []
    for system in SYSTEMS:
        data = m.load_all_data()[system]
        units = data['units']
        policies = ['C0'] + m.RESOLUTION_ORDER[system]
        out_rows = []
        for outer in units:
            douter = [u for u in units if u != outer]
            mid_rows = {p: [] for p in policies}
            for middle in douter:
                dmid = [u for u in douter if u != middle]
                for p in policies:
                    mid_rows[p] += policy_rows(system, data, dmid, middle, p)
            risk = {p: m.middle_system_risk(system, mid_rows[p]) for p in policies}
            unit_loss = {p: np.array([m.middle_unit_loss(system, u, mid_rows[p]) for u in douter]) for p in policies}
            res = m.RESOLUTION_ORDER[system]
            r1 = pick(risk, res)
            r3 = next(p for p in res if risk[p] <= risk[r1] + float(np.std(unit_loss[p] - unit_loss[r1], ddof=1) / math.sqrt(len(douter))) + 1e-12)
            r4 = pick(risk, policies)
            all_choices.append({'system': system, 'outer_unit': str(outer), 'R1': r1, 'R3': r3, 'R4': r4,
                                **{f'middle_{p}': risk[p] for p in policies}})
            for p in policies:
                for row in policy_rows(system, data, douter, outer, p):
                    out_rows.append({'system': system, 'unit': str(outer), 'target': row['target'], 'policy': p, 'loss': row['loss'],
                                     'R1': p == r1, 'R3': p == r3, 'R4': p == r4})
            print(system, outer, r1, r3, r4, flush=True)
        pd.DataFrame(out_rows).to_csv(HERE / 'validation' / f'cardiac_{system}.csv', index=False)
        for cache in (m.PREDICTION_CACHE, m.MODEL_CACHE, m.SELECTION_CACHE):
            cache.clear()
    pd.DataFrame(all_choices).to_csv(HERE / 'validation' / 'cardiac_choices.csv', index=False)


if __name__ == '__main__':
    main()
