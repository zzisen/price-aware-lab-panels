#!/usr/bin/env python3
"""Task 14 (plan_r5_m2 FROZEN, section 7 step 4): run the frozen method on the second-stage tasks.
job() is a copy of the frozen task12_r5/dev_r5.py job(); r5lib, dev_r5 and pilot_panels are imported unchanged. The copy
differs only in the outer split (marked below, chemotype split for the kinase decision) and in taking its settings as
arguments. 'check' reruns BEAT_T1 outer 0 through the copy with the random split: it must reproduce the Task 13 cache.
Run with .env312:  python run_m2.py check   then   python run_m2.py"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
W = HERE.parent
NS = {'nhanes': [25, 50, 100, 500], 'kinase': [25, 50, 100, None], 'beat': [25, 50, 100, None]}
REPS = 4


def chemotype_split(units, chem, rng):
    """Plan section 4: whole chemotypes go to the test set until it holds at least 20% of the compounds, at most 30%."""
    labels = chem.loc[units].chemotype.to_numpy()
    n, te = len(labels), []
    for g in rng.permutation(sorted(set(labels))):
        idx = np.flatnonzero(labels == g)
        if len(te) + len(idx) > .30 * n:
            continue
        te.extend(idx.tolist())
        if len(te) >= .20 * n:
            break
    te = np.array(te)
    return rng.permutation(te), rng.permutation(np.setdiff1d(np.arange(n), te))


def job(name, o, root, out, ns, split):
    sys.path.insert(0, str(W / 'task12_r5')); sys.path.insert(0, str(W / 'task5_validation'))
    import r5lib as R
    import dev_r5 as V
    KS, SEED, decorr, metrics, gbt_predict, r4_sse = V.KS, V.SEED, V.decorr, V.metrics, V.gbt_predict, V.r4_sse
    root, out = Path(root), Path(out)
    task = next(t for t in json.loads((root / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    fam = np.asarray(task['families'])
    m = pd.read_csv(root / 'matrices' / f'{name}.csv')
    x = m[[c for c in m if c.startswith('cand_')]].to_numpy(float); y = m[task['targets']].to_numpy(float)
    rng = np.random.default_rng([SEED, task['code'], o])
    # ---- the only change to the frozen job: the outer split ----
    if split == 'random':
        perm = rng.permutation(len(x)); te = perm[: len(x) // 5]; trn = perm[len(x) // 5:]
    else:
        chem = pd.read_csv(root / 'kin_chemotypes.csv', dtype=str, index_col=0)
        te, trn = chemotype_split(m['unit'].astype(str).to_numpy(), chem, rng)
    # ---- end of change ----
    rows = []
    part = out / 'parts' / f'{name}_{o}.csv'
    if part.exists():
        return pd.read_csv(part).to_dict('records')
    for n in ns:
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
            print(name, split, o, base['n'], r, 'done', flush=True)
    part.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(part, index=False)
    return rows


def check():
    """Regression check (plan section 4): the copy with the random split must reproduce the Task 13 cache of BEAT_T1 outer 0."""
    root, out = W / 'task13_confirm', HERE / 'regression_check'
    Parallel(n_jobs=14)(delayed(job)(*a) for a in [('BEAT_T1', 0, str(root), str(out), NS['beat'], 'random')])
    new = pd.read_csv(out / 'parts' / 'BEAT_T1_0.csv'); old = pd.read_csv(root / 'results' / 'parts' / 'BEAT_T1_0.csv')
    assert new.equals(old), 'the copy does not reproduce the Task 13 cache'
    print('REGRESSION CHECK PASSED: rows', len(new), 'identical to the Task 13 cache', flush=True)


def main():
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    jobs = []
    for t in tasks:
        for o in range(5):
            if t['set'] == 'nhanes':
                jobs.append((t['task'], o, str(HERE), str(HERE / 'results_nhanes'), NS['nhanes'], 'random'))
            else:
                jobs.append((t['task'], o, str(HERE), str(HERE / 'results_kinase_chemotype'), NS['kinase'], 'chemotype'))
                jobs.append((t['task'], o, str(HERE), str(HERE / 'results_kinase_random'), NS['kinase'], 'random'))
    res = Parallel(n_jobs=14, verbose=5)(delayed(job)(*j) for j in jobs)
    frames = {}
    for j, rs in zip(jobs, res):
        frames.setdefault(j[3], []).extend(rs)
    for folder, rows in frames.items():
        d = pd.DataFrame(rows)
        for t, h in d.groupby('task'):
            h.to_csv(Path(folder) / f'{t}.csv', index=False)


if __name__ == '__main__':
    check() if sys.argv[1:] == ['check'] else main()
