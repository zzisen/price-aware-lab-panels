#!/usr/bin/env python3
"""Task 5 validation (plan_validation_suite.md + addenda, plan_r4_frozen.md): run the frozen rules once on one held-out
generic task. Writes validation/<task>.csv with outer loss sums for C0, the fixed resolutions (plain and guarded) and the
rules R1, R3, R4, plus the rules' choices. Run with .env312:  python validate_generic.py <task>"""
import sys
import numpy as np
import pandas as pd
import rulelib as L
import dev_eval as D
from target_resolution import action_losses

LEARNING = {'CTRPv2_T3', 'GDSC1_T4', 'PRISM_T4', 'gCSI_T1'}


def subsamples(task, x):
    if task.get('design') == 'leave_one_out':
        groups = np.arange(len(x))
        for fold in range(len(x)):
            held = np.array([fold])
            yield 'leave_one_out', L.SEEDS[0], fold, None, None, np.setdiff1d(groups, held), held, groups
    else:
        yield from L.task_subsamples(task, x)


def main(name):
    assert name not in LEARNING
    task, x, y, fam = L.load_task(name)
    rows = []
    for ev, seed, fold, n, r, sub, held, groups in subsamples(task, x):
        inner = seed + fold + 1
        xs, ys, gs = x[sub], y[sub], groups[sub]
        # plain pipeline: fixed resolutions, R1 and R3 (single middle split, as frozen in Task 4)
        Lp, _ = L.unit_losses(xs, ys, gs, 5, inner)
        ch = L.choices(Lp.mean(axis=0), fam)
        cand, c0 = action_losses(xs, ys[:, :, None], x[held], y[held, :, None])
        outer = {'C0': float(c0.sum()), **{k: float(L.picked(cand, ch[k]).sum() * ys.shape[1]) for k in L.RES}}
        mid = {'middle': [L.middle_rows(xs, ys, gs, fam, inner, L.MIDDLE_OFFSETS[0])], 'groups': gs, 'outer': outer}
        r1 = D.argmin_rule(D.units(mid), L.RES)
        r3 = D.se_rule(D.units(mid), L.RES, 1.0)
        # guarded pipeline: R4 and the guard-only ablation
        g = L.cache_subsample_g(x, y, groups, fam, sub, held, inner)
        r4 = D.argmin_rule(D.averaged_units({'middle': g['middle_g'], 'groups': gs}), ['C0'] + L.RES)
        rows.append({'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub), 'n_label': 'all' if n is None else str(n),
                     'rep': -1 if r is None else r, 'cells': len(held) * ys.shape[1],
                     **outer, **{f'g_{k}': v for k, v in g['outer_g'].items()},
                     'R1': outer[r1], 'R3': outer[r3], 'R4': g['outer_g'][r4],
                     'choice_R1': r1, 'choice_R3': r3, 'choice_R4': r4, 'guard_fraction': g['guard_fraction']})
    (L.HERE / 'validation').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(L.HERE / 'validation' / f'{name}.csv', index=False)
    print(name, 'subsamples', len(rows))


if __name__ == '__main__':
    main(sys.argv[1])
