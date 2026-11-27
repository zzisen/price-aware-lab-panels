#!/usr/bin/env python3
"""Task 5: offline evaluation of candidate rules on the cached development and learning data
(plan_r4_split_and_gate.md). A rule maps one cached record to a policy key (C0 or a resolution).
Run with .env312:  python dev_eval.py"""
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSES = ['gdsc1_repeated_random', 'gdsc1_held_out_tissue', 'gdsc2_repeated_random', 'gdsc2_held_out_tissue', 'gdsc2_new',
            'prism_repeated_random', 'prism_held_out_tissue', 'radbill', 'CTRPv2_T3', 'GDSC1_T4', 'PRISM_T4', 'gCSI_T1']
ORDER = ['C0', 'Shared', 'Family', 'Global3', 'Exact']     # coarse to fine


def units(rec, seed=0):
    """Per design unit: {policy: (mean loss per unit, cells per unit)} for one middle split."""
    mid = rec['middle']
    if isinstance(mid, dict):          # Radbill: {policy: {participant: [cell losses]}}
        keep = [u for u in mid['C0'] if len(mid['C0'][u])]
        return {k: (np.array([np.mean(v[u]) for u in keep]), np.array([len(v[u]) for u in keep], float)) for k, v in mid.items()}
    rows = mid[seed]
    g = rec['groups']
    _, inv = np.unique(g, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    return {k: (np.bincount(inv, v) / cnt, cnt) for k, v in rows.items()}


def averaged_units(rec):
    if isinstance(rec['middle'], dict):
        return units(rec)
    us = [units(rec, s) for s in range(len(rec['middle']))]
    return {k: (np.mean([u[k][0] for u in us], axis=0), us[0][k][1]) for k in us[0]}


def risk(u):
    return {k: float((m * c).sum() / c.sum()) for k, (m, c) in u.items()}


def candidates(rec, with_c0):
    present = [k for k in ORDER if k in rec['outer']]
    return present if with_c0 else [k for k in present if k != 'C0']


def argmin_rule(u, cands):
    r = risk(u)
    best = min(r[k] for k in cands)
    return next(k for k in cands if r[k] <= best + 1e-12)


def se_rule(u, cands, c=1.0):
    """Coarsest candidate within c paired SEs (over units) of the minimum."""
    r = risk(u)
    best = argmin_rule(u, cands)
    for k in cands:
        d = u[k][0] - u[best][0]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        if r[k] <= r[best] + c * se + 1e-12:
            return k


RULES = {
    'R1': lambda rec: argmin_rule(units(rec), candidates(rec, False)),
    'R3': lambda rec: se_rule(units(rec), candidates(rec, False)),
}


def evaluate(rules, analyses=ANALYSES):
    rows = []
    for an in analyses:
        recs = pickle.load(open(HERE / 'cache' / f'{an}.pkl', 'rb'))
        fixed = [k for k in ['Shared', 'Family', 'Global3', 'Exact'] if k in recs[0]['outer']]
        frame = pd.DataFrame([{'n_label': r['n_label'], 'n_train': r['n_train'], 'cells': r['n_test_cells'],
                               **{k: v for k, v in r['outer'].items()},
                               **{f'rule_{name}': r['outer'][f(r)] for name, f in rules.items()}} for r in recs])
        for n_label, h in frame.groupby('n_label'):
            risk_ = {k: h[k].sum() / h.cells.sum() for k in fixed + ['C0'] + [f'rule_{x}' for x in rules]}
            best = min(risk_[k] for k in fixed)
            rows.append({'analysis': an, 'n_label': n_label, 'n_mean': h.n_train.mean(), **{k: v - best for k, v in risk_.items()}})
    reg = pd.DataFrame(rows)
    mx = reg.groupby('analysis', sort=False).max(numeric_only=True).drop(columns='n_mean').reindex(analyses)
    return reg, mx


def gate(mx, cand):
    others = [c for c in mx.columns if c != cand]
    mean = mx.mean()
    cond1 = all(mean[cand] < mean[o] for o in others if o != 'C0')
    wins_r1 = int((mx[cand] <= mx['rule_R1'] + 1e-12).sum()); wins_r3 = int((mx[cand] <= mx['rule_R3'] + 1e-12).sum())
    return {'mean max regret': mean.round(4).to_dict(), 'cond1 lowest mean': cond1, 'wins vs R1': wins_r1, 'wins vs R3': wins_r3,
            'gate passed': bool(cond1 and wins_r1 >= 9 and wins_r3 >= 9)}


if __name__ == '__main__':
    reg, mx = evaluate(RULES)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(mx.round(4))
