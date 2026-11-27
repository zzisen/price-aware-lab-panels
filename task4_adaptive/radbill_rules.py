#!/usr/bin/env python3
"""Task 4 (post hoc): resolution rules on the Radbill learning curve (plan_adaptive_rules.md, frozen 2026-10-29 00:11 NZDT).

Training sets are regenerated exactly as in task3_learning_curve/radbill_learning_curve.py. Candidates: SHARED, FAMILY,
EXACT. R2 (bootstrap selection-excess correction) on all 50 reps; R1 (nested leave-one-participant-out, V3's rule) on the
first 20 reps and the full set. Run with the base miniconda Python (V3 code environment). Writes radbill_rules_cells.csv
and radbill_rules_runs.csv here.
"""
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task3_learning_curve'))
import radbill_learning_curve as RL  # noqa: E402
m = RL.m
RES = ['SHARED', 'FAMILY', 'EXACT']
SYS = 'RADBILL_HUMAN_PACING'
B = 100
R1_REPS = 20


def unit_scores(data, train):
    """Per-participant inner leave-one-out scores (the units behind radbill_inner_score_cube) and C0 scores."""
    s = {t: {} for t in m.RAD_TARGETS}; c0 = {t: {} for t in m.RAD_TARGETS}
    for hold in sorted(train):
        inner = [i for i in train if i != hold]
        p = data['ps'][hold]
        for t in m.RAD_TARGETS:
            if p['target'][t] is None or not all(p['cand'][c] is not None for c in m.RAD_CANDIDATES):
                continue
            pred = {c: m.radbill_predict(data, inner, hold, c, t) for c in m.RAD_CANDIDATES}
            if any(v is None for v in pred.values()):
                continue
            s[t][hold] = np.array([abs(pred[c][0] - p['target'][t]) / pred[c][1] for c in m.RAD_CANDIDATES])
            ys = np.asarray([data['ps'][u]['target'][t] for u in inner if data['ps'][u]['target'][t] is not None], float)
            sd = m.sample_sd(ys) if len(ys) >= 2 else math.nan
            c0[t][hold] = abs(ys.mean() - p['target'][t]) / sd if math.isfinite(sd) and sd > 1e-12 else math.nan
    return s, c0


def means(s, w):
    """Weighted mean score per target and candidate (inf where no weighted unit), as in the inner score cube."""
    out = {}
    for t in m.RAD_TARGETS:
        units = [u for u in s[t] if w.get(u, 0) > 0]
        if not units:
            out[t] = np.full(len(m.RAD_CANDIDATES), math.inf); continue
        ww = np.array([w[u] for u in units], float)
        out[t] = (ww[:, None] * np.stack([s[t][u] for u in units])).sum(0) / ww.sum()
    return out


def select(mean, res):
    """Candidate index per target, with radbill_select's blocks, candidate sets and tie order."""
    ci = {c: i for i, c in enumerate(m.RAD_CANDIDATES)}
    if res == 'SHARED':
        blocks = [(m.RAD_TARGETS, m.RAD_CANDIDATES)]
    elif res == 'FAMILY':
        blocks = [([t for t in m.RAD_TARGETS if m.RAD_FAMILY[t] == f], cands) for f, cands in m.RAD_FAMILY_CANDS.items()]
    else:
        blocks = [([t], m.RAD_CANDIDATES) for t in m.RAD_TARGETS]
    out = {}
    for targets, cands in blocks:
        score = {c: float(np.mean([mean[t][ci[c]] for t in targets])) if all(math.isfinite(mean[t][ci[c]]) for t in targets) else math.inf
                 for c in cands}
        chosen = min(cands, key=lambda c: (score[c], ci[c]))
        out.update({t: ci[chosen] for t in targets})
    return out


def r2(data, train, rng):
    s, c0 = unit_scores(data, train)
    base = means(s, {u: 1 for u in train})
    plug, excess = {}, {}
    for res in RES:
        a = select(base, res)
        vals = [base[t][a[t]] for t in m.RAD_TARGETS if math.isfinite(base[t][a[t]])]
        plug[res] = float(np.mean(vals))
        excess[res] = []
    for _ in range(B):
        draw = rng.choice(train, size=len(train), replace=True)
        w = {u: int((draw == u).sum()) for u in train}
        star = means(s, w)
        for res in RES:
            a = select(star, res)
            o = [base[t][a[t]] - star[t][a[t]] for t in m.RAD_TARGETS if math.isfinite(base[t][a[t]]) and math.isfinite(star[t][a[t]])]
            if o:
                excess[res].append(float(np.mean(o)))
    excess = {k: float(np.mean(v)) for k, v in excess.items()}
    corrected = {k: plug[k] + excess[k] for k in RES}
    c0_inner = float(np.nanmean([np.nanmean(list(c0[t].values())) for t in m.RAD_TARGETS if c0[t]]))
    return plug, excess, corrected, c0_inner


def r1(data, train):
    risks, skipped = {}, 0
    for res in RES:
        losses = []
        for middle in train:
            dmid = [u for u in train if u != middle]
            actions = m.radbill_select(data, dmid, res, {'selection_layer': 'TASK4_MIDDLE'})[0]
            for t in m.RAD_TARGETS:
                if (middle, t) not in data['support']:
                    continue
                out = m.radbill_predict(data, dmid, middle, str(m.action_for(actions, res, t, SYS)), t)
                if out is None:
                    skipped += 1; continue
                losses.append(abs(out[0] - data['ps'][middle]['target'][t]) / out[1])
        risks[res] = float(np.mean(losses))
    return risks, skipped


def pick(scores):
    best = min(scores.values())
    return next(k for k in RES if scores[k] <= best + 1e-12)


def main():
    data = m.load_all_data()[SYS]
    units = list(data['units'])
    ref = pd.read_csv(HERE.parent / 'task3_learning_curve' / 'radbill_cells.csv')
    cells, runs_meta = [], []
    for outer in units:
        pool = np.array([u for u in units if u != outer])
        for size in RL.SIZES:
            for r in ([None] if size == len(pool) else range(RL.REPS)):
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
                rep = -1 if r is None else r
                plug, excess, corrected, c0_inner = r2(data, train, np.random.default_rng([20261029, 2, int(outer), size, 999 if rep < 0 else rep]))
                choice = {'R2': pick(corrected)}
                choice['R2c0'] = 'C0' if c0_inner <= min(corrected.values()) + 1e-12 else choice['R2']
                meta = {'outer_participant': outer, 'm_train': size, 'rep': rep, **{f'plug_{k}': plug[k] for k in RES},
                        **{f'excess_{k}': excess[k] for k in RES}, 'inner_C0': c0_inner}
                if r is None or r < R1_REPS:
                    mid, skipped = r1(data, train)
                    choice['R1'] = pick(mid)
                    meta.update({f'nested_{k}': mid[k] for k in RES}, middle_cells_skipped=skipped)
                meta.update({f'choice_{k}': v for k, v in choice.items()})
                runs_meta.append(meta)
                for c in result:
                    row = {'outer_participant': outer, 'm_train': size, 'rep': rep, 'target': c['target'],
                           **{p: c[p] for p in RL.POLICIES}}
                    for rule, key in choice.items():
                        row[rule] = c[key]
                    cells.append(row)
            for cache in (m.PREDICTION_CACHE, m.MODEL_CACHE, m.SELECTION_CACHE):
                cache.clear()
        print('outer', outer, 'done', flush=True)
    cells = pd.DataFrame(cells)
    cells.to_csv(HERE / 'radbill_rules_cells.csv', index=False)
    pd.DataFrame(runs_meta).to_csv(HERE / 'radbill_rules_choices.csv', index=False)
    key = ['outer_participant', 'm_train', 'rep', 'target']
    merged = cells.merge(ref, on=key, suffixes=('', '_ref'))
    assert len(merged) == len(cells) == len(ref)
    diff = max(float((merged[p] - merged[p + '_ref']).abs().max()) for p in RL.POLICIES)
    print('max abs diff vs Task 3 fixed losses', diff)
    assert diff < 1e-10
    runs = cells.groupby(['m_train', 'rep'])[RL.POLICIES + ['R2', 'R2c0', 'R1']].mean().reset_index()
    runs.to_csv(HERE / 'radbill_rules_runs.csv', index=False)


if __name__ == '__main__':
    main()
