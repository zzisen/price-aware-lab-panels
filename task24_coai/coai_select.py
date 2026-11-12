#!/usr/bin/env python3
"""Task 24: CoAI (Erion et al., Nat Biomed Eng 2022) applied to our training data and fee schedules. Training data only (NHANES 1999-2010).
Faithful re-implementation of the published algorithm (github.com/suinleelab/coai, coai/knapsack.py GroupOptimizer and coai/base.py),
without its shap / ortools / dill dependencies:
  1. base model = logistic regression for the binary label (9-year death), hyperparameters chosen as in the paper's Method 7
     (L1 or L2; strength 1e-5..1e5; SAGA) on an 80/20 train/validation split with all features, then refit on all training data;
  2. importance of feature i = sum over people of |SHAP| (linear model, independent features: beta_i * (x_ij - mean_i)), as in
     DynamicOptimizer.calculate_global_importances (reduction = sum of absolute values);
  3. group importance = sum over the group's features; group cost = mean cost of its features (GroupOptimizer.feats_at_thresh);
  4. 0/1 knapsack: maximise total importance subject to total group cost <= budget (exact dynamic programme over cents; the
     package uses OR-tools' dynamic-programming solver on integer-scaled values);
  5. the selected features define the panel; CoAI then refits the base model on them.
Price inputs (both reported):
  'published' : single-test Medicare prices, added up (as in the CoAI outpatient task); CBC is one group (billable only as a panel);
  'bundled'   : generous variant, CoAI given the real bundles as groups (CMP 14 analytes $10.56, lipids $13.39, CBC $7.77).
Budgets: the US prices of our matched panel ($32.41) and of the routine bundle ($50.77); the venipuncture fee ($9.34) is taken off
the knapsack budget because CoAI has no draw cost. China (CHNS menu of 15 shared analytes, additive Chinese prices): yuan 43.8 and 74.8,
venipuncture 6.3 taken off. Age and sex are free and always included (as our panels).
Output: coai_panels.json"""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent; T21 = HERE.parent / 'task21_final'
sys.path.insert(0, str(T21))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
L = M.L
L.low_priority()
SEED = 20261110
STRENGTHS = [10.0 ** k for k in range(-5, 6)]


def base_model(X, y):
    """Paper Method 7: choose L1/L2 and strength on a single 80/20 split by validation AUC, then refit on all data."""
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y); best = (-1, None)
    for pen in ['l1', 'l2']:
        for s in STRENGTHS:
            m = LogisticRegression(penalty=pen, C=1.0 / s, solver='saga', max_iter=20000, tol=1e-6).fit(Xa, ya)
            a = roc_auc_score(yv, m.decision_function(Xv))
            if a > best[0]:
                best = (a, dict(penalty=pen, C=1.0 / s))
    m = LogisticRegression(solver='saga', max_iter=20000, tol=1e-6, **best[1]).fit(X, y)
    return m, best


def knapsack(values, costs_cents, budget_cents):
    """Exact 0/1 knapsack (maximise sum of values, total cost <= budget). Returns chosen indices."""
    n = len(values); B = int(budget_cents); dp = np.zeros(B + 1); take = np.zeros((n, B + 1), bool)
    for i in range(n):
        c, v = int(costs_cents[i]), values[i]
        if c > B:
            continue
        cand = np.full(B + 1, -np.inf); cand[c:] = dp[:B + 1 - c] + v
        better = cand > dp + 1e-15; take[i] = better; dp = np.where(better, cand, dp)
    sel, b = [], B
    for i in range(n - 1, -1, -1):
        if take[i, b]:
            sel.append(i); b -= int(costs_cents[i])
    return sorted(sel)


def groups_us(variant):
    g = {}
    if variant == 'published':
        for a in L.CHEM:
            g[a] = ([a], L.CHEM[a][2])
    else:
        g['CMP'] = (L.PANELS['CMP'][0], L.PANELS['CMP'][1]); g['LIPID'] = (L.PANELS['LIPID'][0], L.PANELS['LIPID'][1])
        for a in L.CHEM:
            if a not in L.PANELS['CMP'][0] and a not in L.PANELS['LIPID'][0]:
                g[a] = ([a], L.CHEM[a][2])
    g['CBC'] = (list(M.CBC2), 7.77); g['hba1c'] = (['hba1c'], 9.71); g['crp'] = (['crp'], 12.95); g['acr'] = (['acr'], 5.78 + 5.18)
    return g


def groups_cn():
    g = {a: ([a], T.CN_PRICE[a]) for a in T.SHARED if a not in T.CBC3}
    g['CBC3'] = (list(T.CBC3), T.CN_CBC)
    return g


def run(Z, y, groups, budgets, draw, cost_fn):
    feats = [f for _, (fs, _) in groups.items() for f in fs]
    cols = M.BASE + feats; X = Z[cols].to_numpy(float); X = (X - X.mean(0)) / X.std(0)
    m, best = base_model(X, y)
    phi = np.abs(m.coef_[0] * (X - X.mean(0))).sum(0)                     # sum of |linear SHAP| per feature
    imp = dict(zip(cols, phi.tolist()))
    names = list(groups); gv = np.array([sum(imp[f] for f in groups[k][0]) for k in names])
    gc = np.array([round(groups[k][1] * 100) for k in names])           # group cost in cents (CoAI: mean of member costs = the group cost)
    out = []
    for B in budgets:
        sel = knapsack(gv, gc, round((B - draw) * 100))
        items = [names[i] for i in sel]; f = [x for k in items for x in groups[k][0]]
        out.append(dict(budget=B, items=items, features=f, additive_cost=round(float(gc[sel].sum()) / 100 + draw, 2), billed_cost=cost_fn(f),
                        importance=float(gv[sel].sum())))
    return dict(hyper=dict(val_auc=best[0], **best[1]), importance=imp, group_importance=dict(zip(names, gv.tolist())), panels=out)


if __name__ == '__main__':
    x, ctr = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
    known = e | (t >= M.HORIZON); y = (e & (t <= M.HORIZON))
    Zk, yk = Z[known], y[known].astype(int)
    res = dict(n_train=int(known.sum()), deaths_9y=int(yk.sum()), excluded_unknown_9y_status=int((~known).sum()))
    for v in ['published', 'bundled']:
        res['US_' + v] = run(Zk, yk, groups_us(v), [32.41, 50.77], 9.34, M.cost_us)
    Zc = T.transform_sub(x, x, T.SHARED)
    res['CN'] = run(Zc[known], yk, groups_cn(), [43.8, 74.8], T.CN_VENI, T.cost_cn)
    json.dump(res, open(HERE / 'coai_panels.json', 'w'), indent=1)
    for k in ['US_published', 'US_bundled', 'CN']:
        print(k, res[k]['hyper'])
        for p in res[k]['panels']:
            print('  budget', p['budget'], '| items', p['items'], '| additive', p['additive_cost'], '| billed', p['billed_cost'])
