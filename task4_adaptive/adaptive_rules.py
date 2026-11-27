#!/usr/bin/env python3
"""Task 4 (post hoc): choosing the resolution from the training data, as specified in plan_adaptive_rules.md
(frozen 2026-10-29 00:11 NZDT).

Rules, all fitted on the training subsample only:
  R1  nested CV over resolutions (V3's adaptive rule);
  R2  inner-CV risk plus a bootstrap estimate of selection excess (primary); R2c0 adds C0 as a candidate;
  S   partial pooling of Shared, Family and target-specific selection scores, weight by nested CV (Snest)
      or by the bootstrap correction (Sboot).
The subsamples are exactly those of Task 3 (GDSC1) and Task 3b (GDSC2, GDSC2 new lines, PRISM).
Run with .env312:  python adaptive_rules.py <gdsc1|gdsc2|gdsc2_new|prism> [repeated_random|held_out_tissue]
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
for p in ('base_v32', 'task3_learning_curve', 'task3b_external'):
    sys.path.insert(0, str(WORK / p))
from target_resolution import group_folds, action_losses, _choose_from_risks  # noqa: E402
from evaluate_gdsc import FAMILIES, SEEDS  # noqa: E402
from exact_budget import globally_optimal_actions  # noqa: E402
import gdsc_learning_curve as G  # noqa: E402
import prism_analysis as P  # noqa: E402

RES = ['Shared', 'Family', 'Global3', 'Exact']            # coarse to fine; ties go to the coarser
WEIGHTS = sorted([(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)], key=lambda w: (-w[0], -w[1]))
B = 100
FAM = np.asarray(FAMILIES)
FAM_MEMBERS = [np.flatnonzero(FAM == FAM[t]) for t in range(len(FAM))]


def unit_losses(x, y, groups, splits, seed):
    """Per-unit inner-CV losses L[i, a, t] and C0 losses; L.mean(0) is the select_actions risk matrix."""
    L = np.empty((len(x), x.shape[1], y.shape[1])); c0 = np.empty((len(x), y.shape[1]))
    for held in group_folds(groups, splits, seed):
        train = np.ones(len(x), bool); train[held] = False
        L[held], c0[held] = action_losses(x[train], y[train, :, None], x[held], y[held, :, None])
    return L, c0


def choices(risk, with_weights=True):
    out = dict(_choose_from_risks(risk, FAM))
    out['Global3'] = globally_optimal_actions(risk, 3)[0]
    if with_weights:
        shared = risk.mean(axis=1)[:, None]
        fam = np.stack([risk[:, m].mean(axis=1) for m in FAM_MEMBERS], axis=1)
        for w in WEIGHTS:
            out[w] = np.argmin(w[0] * shared + w[1] * fam + w[2] * risk, axis=0)
    return out


def scored(candidate, a):
    return float(candidate[np.arange(len(candidate))[:, None], a[None, :], np.arange(candidate.shape[2])[None, :]].sum())


def nested(x, y, groups, seed):
    """Pooled middle-held risk of each resolution and each pooling weight (R1 and S-nested)."""
    ng = len(np.unique(groups))
    sums, cells = {}, 0
    for k, held in enumerate(group_folds(groups, min(5, ng), seed + 7919)):
        train = np.ones(len(x), bool); train[held] = False
        splits = min(5, len(np.unique(groups[train])))
        L, _ = unit_losses(x[train], y[train], groups[train], splits, seed + 100 + k)
        ch = choices(L.mean(axis=0))
        candidate, _ = action_losses(x[train], y[train, :, None], x[held], y[held, :, None])
        for key, a in ch.items():
            sums[key] = sums.get(key, 0.0) + scored(candidate, a)
        cells += candidate.shape[0] * candidate.shape[2]
    return {k: v / cells for k, v in sums.items()}


def bootstrap_excess(L, groups, rng):
    """Efron optimism of the selection step for every resolution and pooling weight (R2 and S-boot)."""
    risk = L.mean(axis=0)
    u, inv = np.unique(groups, return_inverse=True)
    W = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv] for _ in range(B)]).astype(float)
    keep = W.sum(axis=1) > 0
    W = W[keep]
    star = np.einsum('bi,iat->bat', W, L, optimize=True) / W.sum(axis=1)[:, None, None]
    opt = {}
    t_idx = np.arange(L.shape[2])
    for b in range(len(W)):
        for key, a in choices(star[b]).items():
            opt.setdefault(key, []).append(float((risk[a, t_idx] - star[b][a, t_idx]).mean()))
    return {k: float(np.mean(v)) for k, v in opt.items()}


def pick(scores, keys):
    best = min(scores[k] for k in keys)
    return next(k for k in keys if scores[k] <= best + 1e-12)


def evaluate_rules(x, y, groups, sub, held, inner_seed, rng):
    xs, ys, gs = x[sub], y[sub], groups[sub]
    L, c0_inner = unit_losses(xs, ys, gs, 5, inner_seed)
    risk = L.mean(axis=0)
    ch = choices(risk)
    candidate, c0 = action_losses(xs, ys[:, :, None], x[held], y[held, :, None])
    plug = {k: float(risk[a, np.arange(risk.shape[1])].mean()) for k, a in ch.items()}
    excess = bootstrap_excess(L, gs, rng)
    corrected = {k: plug[k] + excess[k] for k in ch}
    mid = nested(xs, ys, gs, inner_seed)
    choice = {'R1': pick(mid, RES), 'R2': pick(corrected, RES), 'Snest': pick(mid, WEIGHTS), 'Sboot': pick(corrected, WEIGHTS)}
    c0_risk = float(c0_inner.mean())
    choice['R2c0'] = 'C0' if c0_risk <= min(corrected[k] for k in RES) + 1e-12 else choice['R2']
    row = {'loss_sum_C0': float(c0.sum())}
    for k in RES:
        row[f'loss_sum_{k}'] = scored(candidate, ch[k])
    for rule, key in choice.items():
        row[f'loss_sum_{rule}'] = row['loss_sum_C0'] if key == 'C0' else scored(candidate, ch[key])
        row[f'choice_{rule}'] = key if isinstance(key, str) else 'w=' + ','.join(f'{v:g}' for v in key)
    for k in RES:
        row[f'plug_{k}'] = plug[k]; row[f'excess_{k}'] = excess[k]; row[f'nested_{k}'] = mid[k]
    row['inner_C0'] = c0_risk
    return row


def subsamples(dataset, only=None):
    """Yield (design, seed, fold, n, r, sub, held, groups, x, y, eval_code) exactly as in Task 3 / 3b."""
    if dataset == 'gdsc1':
        x, y, tissue = G.load()
        designs = [('repeated_random', SEEDS, np.arange(499), 5, 3), ('held_out_tissue', [SEEDS[0]], tissue, 13, 5)]
        grid_all, codes = G.SIZES, G.EVAL_CODE
    else:
        m, x, y, _ = P.load(dataset)
        _, _, grid_all, codes, subset = P.DATASETS[dataset]
        tissue = pd.factorize(m.tissue_group)[0]
        designs = [('repeated_random', SEEDS, np.arange(len(x)), 5, 3), ('held_out_tissue', [SEEDS[0]], tissue, len(np.unique(tissue)), 5)]
        if subset == 'new':
            designs = designs[:1]
    for ev, seeds, groups, nfold, min_groups in designs:
        if only and ev != only:
            continue
        folds = {s: group_folds(groups, nfold, s) for s in seeds}
        smallest = min(len(x) - len(h) for s in seeds for h in folds[s])
        grid = grid_all if dataset == 'gdsc1' else [n for n in grid_all if n is None or n <= smallest]
        for seed in seeds:
            for fold, held in enumerate(folds[seed]):
                train_idx = np.setdiff1d(np.arange(len(x)), held)
                for n in grid:
                    for r in ([None] if n is None else range(G.REPS)):
                        if n is None:
                            sub = train_idx
                        else:
                            rng = np.random.default_rng([20261028, codes[ev], seed, fold, n, r])
                            sub, _ = G.draw(train_idx, n, groups, rng, min_groups)
                        yield ev, seed, fold, n, r, sub, held, groups, x, y, codes[ev]


def main(dataset, only=None):
    rows = []
    last = None
    for ev, seed, fold, n, r, sub, held, groups, x, y, code in subsamples(dataset, only):
        rng = np.random.default_rng([20261029, code, seed, fold, 999 if n is None else n, 999 if r is None else r])  # 999 codes n = all
        res = evaluate_rules(x, y, groups, sub, held, seed + fold + 1, rng)
        rows.append({'evaluation': ev, 'outer_seed': seed, 'fold': fold, 'n_train': len(sub), 'n_label': 'all' if n is None else n,
                     'rep': -1 if r is None else r, 'n_test_cells': len(held) * y.shape[1], **res})
        if (ev, seed, fold) != last:
            print(dataset, ev, seed, fold, flush=True); last = (ev, seed, fold)
    out = pd.DataFrame(rows)
    tag = dataset + (f'_{only}' if only else '')
    out.to_csv(HERE / f'{tag}_rules.csv', index=False)

    # implementation check: fixed-resolution losses equal the Task 3 / 3b inner-CV runs
    ref = pd.read_csv(WORK / 'task3_learning_curve' / 'gdsc_runs.csv') if dataset == 'gdsc1' else pd.read_csv(WORK / 'task3b_external' / f'{dataset}_runs.csv')
    ref = ref[ref.selector == 'inner_cv'].copy()
    key = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
    for frame in (ref, out):
        frame['n_label'] = frame['n_label'].astype(str)
    merged = out.merge(ref, on=key, suffixes=('', '_ref'))
    assert len(merged) == len(out), (len(merged), len(out))
    diff = max(float((merged[f'loss_sum_{k}'] - merged[f'loss_sum_{k}_ref']).abs().max()) for k in ['C0'] + RES)
    print(tag, 'subsamples', len(out), '| max abs diff vs Task 3/3b fixed losses', diff)
    assert diff < 1e-10


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
