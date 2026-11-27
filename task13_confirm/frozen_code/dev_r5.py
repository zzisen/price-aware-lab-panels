#!/usr/bin/env python3
"""R5 development run (DEVELOPMENT data only: GDSC1 / GDSC2 / PRISM drug-as-measurement tasks of Task 8, all used
before). Panel rule {random, decorr (prior art), R5 and its fixed-resolution variants} x predictor {ridge, GBT 50 trees
depth 2 (prior art)}, k = 3/5/10/20/30, n = 25/50/100/all. R4 (frozen) and all-candidate fits as references.
Scores on held-out units, y standardised on the training subsample. Writes dev_results/<task>.csv.
Run with .env312:  python dev_r5.py [task ...]   (no argument: all development tasks, in parallel)"""
import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor

HERE = Path(__file__).resolve().parent
W = HERE.parent
T8 = Path(os.environ.get('R5_TASKDIR', W / 'task8_real_cost'))   # hetero run: R5_TASKDIR=hetero
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(W / 'task5_validation'))
import r5lib as R  # noqa: E402
import rulelib as L  # noqa: E402
import dev_eval as D  # noqa: E402
from pilot_panels import decorr  # noqa: E402

warnings.filterwarnings('ignore')
TASKS = ['DRUG_GDSC1_T1', 'DRUG_GDSC1_T2', 'DRUG_GDSC1_T3', 'DRUG_GDSC2_T1', 'DRUG_GDSC2_T2', 'DRUG_GDSC2_T3',
         'DRUG_GDSC2_T4', 'DRUG_PRISM_T1', 'DRUG_PRISM_T2', 'DRUG_PRISM_T3']
KS, NS, OUTER, REPS, SEED = [3, 5, 10, 20, 30], [25, 50, 100, None], 5, 3, 20261103
OUT = HERE / os.environ.get('R5_OUT', 'dev_results')


def gbt_predict(xtr, ytr, xte, cols, seed):
    if not len(cols):
        return np.repeat(ytr.mean(0, keepdims=True), len(xte), 0)
    return np.column_stack([GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=seed)
                            .fit(xtr[:, cols], ytr[:, t]).predict(xte[:, cols]) for t in range(ytr.shape[1])])


def metrics(pred, yt, ym, ysd):
    raw_p, raw_t = pred * ysd + ym, yt * ysd + ym
    pooled = np.corrcoef(raw_p.ravel(), raw_t.ravel())[0, 1] if raw_p.std() > 0 else np.nan
    per = [np.corrcoef(raw_p[:, t], raw_t[:, t])[0, 1] if raw_p[:, t].std() > 0 else np.nan for t in range(yt.shape[1])]
    return {'sse': float(((pred - yt) ** 2).sum()), 'c0': float((yt ** 2).sum()),
            'pooled_r': float(pooled), 'target_r': float(np.nanmean(per)) if np.isfinite(per).any() else np.nan}


def r4_sse(xs, ys, x_te, y_te, fam, inner):
    gs = np.arange(len(xs))
    Lg, _ = L.unit_losses_g(xs, ys, gs, 5, inner)
    ch = L.choices(Lg.mean(axis=0), fam)
    mids = [L.middle_rows_g(xs, ys, gs, fam, inner, off) for off in L.MIDDLE_OFFSETS]
    r4 = D.argmin_rule(D.averaged_units({'middle': mids, 'groups': gs}), ['C0'] + L.RES)
    cand, c0, _ = L.guarded_losses(xs, ys[:, :, None], x_te, y_te[:, :, None])
    sse = float(c0.sum()) if r4 == 'C0' else float(L.picked(cand, ch[r4]).sum() * ys.shape[1])
    return r4, sse, (0 if r4 == 'C0' else len(set(np.asarray(ch[r4]).tolist())))


def job(name, o):
    task = next(t for t in json.loads((T8 / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    fam = np.asarray(task['families'])
    m = pd.read_csv(T8 / 'matrices' / f'{name}.csv')
    x = m[[c for c in m if c.startswith('cand_')]].to_numpy(float); y = m[task['targets']].to_numpy(float)
    rng = np.random.default_rng([SEED, task['code'], o])
    perm = rng.permutation(len(x)); te = perm[: len(x) // 5]; trn = perm[len(x) // 5:]
    rows = []
    part = OUT / 'parts' / f'{name}_{o}.csv'
    if part.exists():
        return pd.read_csv(part).to_dict('records')
    for n in NS:
        if n is not None and n >= len(trn):      # GDSC1_T3 has 87 training units: no n = 100 subsample
            continue
        for r in (range(REPS) if n else [0]):
            sd = [SEED, task['code'], o, n or 0, r]
            sub = trn if n is None else np.random.default_rng(sd).choice(trn, n, replace=False)
            xs = x[sub]; ym = y[sub].mean(0); ysd = y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys = (y[sub] - ym) / ysd; yt = (y[te] - ym) / ysd; xt = x[te]
            seed = int(np.random.default_rng(sd).integers(1 << 30))
            base = {'task': name, 'outer': o, 'n': 'all' if n is None else n, 'rep': r}
            paths = {res: R.select_path(xs, ys, fam, res, seed) for res in R.RES5}
            chosen, _ = R.middle_choice(xs, ys, fam, KS, seed)

            def add(method, pred_name, k, k_used, pred, **extra):
                rows.append({**base, 'method': method, 'predictor': pred_name, 'k': k, 'k_used': k_used,
                             **metrics(pred, yt, ym, ysd), **extra})
            for k in KS:
                panels = {'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False),
                          'decorr': decorr(xs, k)}
                opt = chosen[k]
                if opt == 'C0':
                    r5_panel = np.array([], int); r5_pred = np.zeros_like(yt)
                else:
                    res, mode = opt.split('|'); G, path = paths[res]; sets = R.panel_at(path, k)
                    r5_panel = np.array(sorted({j for s in sets for j in s}), int) if sets else np.array([], int)
                    r5_pred = R.r5_predict(xs, ys, xt, G, sets, mode) if sets else np.zeros_like(yt)
                add('R5', 'native', k, len(r5_panel), r5_pred, option=opt)
                for res in R.RES5:
                    G, path = paths[res]; sets = R.panel_at(path, k)
                    for mode in R.MODES:
                        if res == 'Shared' and mode == 'union':
                            continue
                        pred = R.r5_predict(xs, ys, xt, G, sets, mode) if sets else np.zeros_like(yt)
                        ku = len({j for s in sets for j in s}) if sets else 0
                        add(f'R5fix_{res}|{mode}', 'native', k, ku, pred)
                G, path = paths['Shared']; sets = R.panel_at(path, k)
                panels['R5'] = r5_panel
                panels['R5_Shared'] = np.array(sets[0] if sets else [], int)
                for meth, cols in panels.items():
                    add(meth, 'ridge', k, len(cols), R.ridge_predict(xs, ys, xt, cols))
                    add(meth, 'gbt', k, len(cols), gbt_predict(xs, ys, xt, cols, seed))
            allc = np.arange(x.shape[1])
            add('all', 'ridge', x.shape[1], x.shape[1], R.ridge_predict(xs, ys, xt, allc))
            add('all', 'gbt', x.shape[1], x.shape[1], gbt_predict(xs, ys, xt, allc, seed))
            r4, sse, k4 = r4_sse(xs, ys, xt, yt, fam, seed)
            rows.append({**base, 'method': 'R4', 'predictor': 'native', 'k': k4, 'k_used': k4, 'sse': sse,
                         'c0': float((yt ** 2).sum()), 'option': r4})
            print(name, o, base['n'], r, 'done', flush=True)
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def main(tasks):
    OUT.mkdir(exist_ok=True)
    jobs = [(t, o) for t in tasks for o in range(OUTER)]
    res = Parallel(n_jobs=min(14, len(jobs)), verbose=5)(delayed(job)(t, o) for t, o in jobs)
    d = pd.DataFrame([r for rs in res for r in rs])
    for t, h in d.groupby('task'):
        h.to_csv(OUT / f'{t}.csv', index=False)


if __name__ == '__main__':
    main(sys.argv[1:] or TASKS)
