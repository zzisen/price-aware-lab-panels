#!/usr/bin/env python3
"""Task 4b (post hoc): R3, the one-SE rule on nested resolution risks, and R3r, repeated nested CV, as specified in
plan_r3.md (frozen 2026-10-29 00:38 NZDT). Same subsamples and outer test rows as adaptive_rules.py.
Run with .env312:  python r3_rules.py <gdsc1|gdsc2|gdsc2_new|prism> [repeated_random|held_out_tissue]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaptive_rules as A  # noqa: E402
from target_resolution import group_folds, action_losses  # noqa: E402

MIDDLE_OFFSETS = [7919, 7920, 7921]


def nested_rows(x, y, groups, seed, offset):
    """Middle-held loss of every row (mean over targets) for each resolution; one middle split."""
    ng = len(np.unique(groups))
    out = {k: np.empty(len(x)) for k in A.RES}
    for k, held in enumerate(group_folds(groups, min(5, ng), seed + offset)):
        train = np.ones(len(x), bool); train[held] = False
        splits = min(5, len(np.unique(groups[train])))
        L, _ = A.unit_losses(x[train], y[train], groups[train], splits, seed + 100 + k)
        ch = A.choices(L.mean(axis=0), with_weights=False)
        candidate, _ = action_losses(x[train], y[train, :, None], x[held], y[held, :, None])
        t = np.arange(candidate.shape[2])
        for res in A.RES:
            out[res][held] = candidate[np.arange(len(held))[:, None], ch[res][None, :], t[None, :]].mean(axis=1)
    return out


def one_se(rows, groups):
    """Coarsest resolution within one paired SE (over design units) of the minimum nested risk."""
    u, inv = np.unique(groups, return_inverse=True)
    unit = {k: np.bincount(inv, v) / np.bincount(inv) for k, v in rows.items()}
    risk = {k: float(v.mean()) for k, v in rows.items()}
    best = A.pick(risk, A.RES)
    for res in A.RES:
        d = unit[res] - unit[best]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        if risk[res] <= risk[best] + se + 1e-12:
            return res


def main(dataset, only=None):
    rows = []
    for ev, seed, fold, n, r, sub, held, groups, x, y, code in A.subsamples(dataset, only):
        xs, ys, gs = x[sub], y[sub], groups[sub]
        inner = seed + fold + 1
        L, _ = A.unit_losses(xs, ys, gs, 5, inner)
        ch = A.choices(L.mean(axis=0), with_weights=False)
        candidate, _ = action_losses(xs, ys[:, :, None], x[held], y[held, :, None])
        reps = [nested_rows(xs, ys, gs, inner, off) for off in MIDDLE_OFFSETS]
        first = reps[0]
        avg = {k: np.mean([rp[k] for rp in reps], axis=0) for k in A.RES}
        choice = {'R1': A.pick({k: float(v.mean()) for k, v in first.items()}, A.RES),
                  'R3': one_se(first, gs),
                  'R3r': A.pick({k: float(v.mean()) for k, v in avg.items()}, A.RES),
                  'R3r1se': one_se(avg, gs)}
        row = {'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub), 'n_label': 'all' if n is None else n,
               'rep': -1 if r is None else r, 'n_test_cells': len(held) * y.shape[1]}
        for rule, key in choice.items():
            row[f'loss_sum_{rule}'] = A.scored(candidate, ch[key]); row[f'choice_{rule}'] = key
        rows.append(row)
    out = pd.DataFrame(rows)
    tag = dataset + (f'_{only}' if only else '')
    out.to_csv(HERE / f'{tag}_r3.csv', index=False)
    ref = pd.read_csv(HERE / f'{tag}_rules.csv')
    key = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
    for f in (ref, out):
        f['n_label'] = f['n_label'].astype(str)
    mg = out.merge(ref[key + ['loss_sum_R1', 'choice_R1']], on=key, suffixes=('', '_t4'))
    assert len(mg) == len(out)
    diff = float((mg.loss_sum_R1 - mg.loss_sum_R1_t4).abs().max())
    print(tag, 'subsamples', len(out), '| R1 reproduction max abs diff', diff, '| choice agreement', (mg.choice_R1 == mg.choice_R1_t4).mean())
    assert diff < 1e-10


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
