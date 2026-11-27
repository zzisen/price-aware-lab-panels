#!/usr/bin/env python3
"""Task 13 (plan_r5_confirm FROZEN 20261101T2045, section 6 step 4): run the frozen dev_r5.job on the confirmation
tasks. The frozen modules are imported unchanged; only their run settings are set inside each worker:
  T8 -> this folder, OUT -> results/, REPS = 4, NS per task set (plan 3.1 / 3.2);
  S1 tasks: r5lib.RES5 = ['Shared', 'Family'] (plan section 5, S1);
  S1 tasks: the R4 reference is skipped (DEVIATIONS D3);
  metrics() is wrapped to add top5_hit (S1, plan section 5): for each unit, overlap of the 5 lowest predicted and the
  5 lowest observed AUCs among the targets, divided by 5. sse / c0 / Pearson values are untouched.
Run once with .env312:  python run_confirm.py"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / 'task12_r5'
NS = {'main': [25, 50, 100, None], 's1': [25, 50, 100, None], 'nhanes': [25, 50, 100, 500]}


def run(name, o, s):
    sys.path.insert(0, str(FROZEN)); sys.path.insert(0, str(HERE.parent / 'task5_validation'))
    import r5lib as R
    import dev_r5 as V
    V.T8, V.OUT, V.REPS, V.NS = HERE, HERE / 'results', 4, NS[s]
    # workers are reused across jobs: reset every setting on each call
    if not hasattr(V, 'metrics0'):
        V.metrics0, V.r4_sse0 = V.metrics, V.r4_sse
    R.RES5, V.metrics, V.r4_sse = ['Shared', 'Family', 'Exact'], V.metrics0, V.r4_sse0
    if s == 's1':
        R.RES5 = ['Shared', 'Family']
        V.r4_sse = lambda *a: ('skipped_D3', float('nan'), 0)   # DEVIATIONS D3: R4 enumerates partitions of 40 targets
        base = V.metrics0

        def metrics(pred, yt, ym, ysd):
            out = base(pred, yt, ym, ysd)
            p, t = pred * ysd + ym, yt * ysd + ym
            lo_p, lo_t = np.argsort(p, axis=1, kind='stable')[:, :5], np.argsort(t, axis=1, kind='stable')[:, :5]
            out['top5_hit'] = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(lo_p, lo_t)]))
            return out
        V.metrics = metrics
    return V.job(name, o)


def main():
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    jobs = [(t['task'], o, t['set']) for t in tasks for o in range(5)]
    res = Parallel(n_jobs=14, verbose=5)(delayed(run)(*j) for j in jobs)
    d = pd.DataFrame([r for rs in res for r in rs])
    for t, h in d.groupby('task'):
        h.to_csv(HERE / 'results' / f'{t}.csv', index=False)


if __name__ == '__main__':
    main()
