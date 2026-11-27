#!/usr/bin/env python3
"""Task 5: cache training-side quantities and outer losses for R4 development (development analyses and the four
learning tasks only; plan_r4_split_and_gate.md). Run with .env312:
  python cache_generic.py gdsc1 repeated_random     (development analysis)
  python cache_generic.py task GDSC1_T4             (learning task)
"""
import pickle
import sys
import numpy as np
import pandas as pd
import rulelib as L

LEARNING = {'CTRPv2_T3', 'GDSC1_T4', 'PRISM_T4', 'gCSI_T1'}


def main(args):
    out = []
    if args[0] == 'task':
        assert args[1] in LEARNING, 'only learning tasks may be cached before the gate'
        task, x, y, fam = L.load_task(args[1])
        tag = args[1]
        it = ((ev, seed, fold, n, r, sub, held, groups, x, y) for ev, seed, fold, n, r, sub, held, groups in L.task_subsamples(task, x))
    else:
        import adaptive_rules as A
        fam = A.FAM
        tag = f'{args[0]}_{args[1]}' if len(args) > 1 else args[0]
        it = ((ev, seed, fold, n, r, sub, held, groups, x, y) for ev, seed, fold, n, r, sub, held, groups, x, y, _ in A.subsamples(args[0], args[1] if len(args) > 1 else None))
    last = None
    for ev, seed, fold, n, r, sub, held, groups, x, y in it:
        rec = L.cache_subsample(x, y, groups, fam, sub, held, seed + fold + 1)
        rec.update({'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_label': 'all' if n is None else str(n), 'rep': -1 if r is None else r,
                    'n_train': len(sub)})
        out.append(rec)
        if (seed, fold) != last:
            print(tag, seed, fold, flush=True); last = (seed, fold)
    (L.HERE / 'cache').mkdir(exist_ok=True)
    with open(L.HERE / 'cache' / f'{tag}.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    if args[0] != 'task':   # implementation check against Task 4
        ref = pd.read_csv(L.WORK / 'task4_adaptive' / f'{tag}_rules.csv')
        ref['n_label'] = ref.n_label.astype(str)
        ref = ref.set_index(['evaluation', 'outer_seed', 'fold', 'n_label', 'rep'])
        diff = max(abs(rec['outer'][k] - ref.loc[(rec['evaluation'], rec['outer_seed'], rec['fold'], rec['n_label'], rec['rep']), f'loss_sum_{k}'])
                   for rec in out for k in ['C0'] + L.RES)
        print(tag, 'records', len(out), '| max abs diff vs Task 4 outer losses', diff)
        assert diff < 1e-8
    else:
        print(tag, 'records', len(out))


if __name__ == '__main__':
    main(sys.argv[1:])
