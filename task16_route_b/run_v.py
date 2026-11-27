#!/usr/bin/env python3
"""Task 16 (plan_route_b_final FROZEN): run V, D, U, R, W and C with the frozen R5 code.
job_main is a copy of the frozen dev_r5 job via task14_m2/run_m2.job. r5lib, dev_r5 and pilot_panels are imported unchanged. The copy differs in:
the outer split can hold out whole patients; it saves per-unit errors at n = 100; n and k are arguments; R4 and most R5fix rows are not run.
'check' must reproduce the Task 13 and Task 14 caches before anything else runs.
Run with .env312:  python run_v.py check   then   python run_v.py"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
W = HERE.parent
T13, T14 = W / 'task13_confirm', W / 'task14_m2'
LADDER = [25, 50, 100, 200, 400]
FINE = [8, 13, 15, 17, 25]
SAVE_UNITS = (5, 10, 30)
REPS = 4
N_JOBS = 14


def libs():
    sys.path.insert(0, str(W / 'task12_r5')); sys.path.insert(0, str(W / 'task5_validation'))
    import r5lib as R
    import dev_r5 as V
    return R, V


def chemotype_split(units, chem, rng):
    """Copy of task14_m2/run_m2.chemotype_split."""
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


def group_split(labels, rng):
    """Whole patients go to the test set until it holds at least 20% of the samples."""
    n, te = len(labels), []
    for g in rng.permutation(sorted(set(labels))):
        te.extend(np.flatnonzero(labels == g).tolist())
        if len(te) >= .20 * n:
            break
    te = np.array(te)
    return rng.permutation(te), rng.permutation(np.setdiff1d(np.arange(n), te))


def outer_split(split, units, rng):
    if split == 'random':
        perm = rng.permutation(len(units))
        return perm[: len(units) // 5], perm[len(units) // 5:]
    if split == 'chemotype':
        return chemotype_split(units, pd.read_csv(T14 / 'kin_chemotypes.csv', dtype=str, index_col=0), rng)
    g = pd.read_csv(HERE / 'pools' / 'E1_groups.csv', dtype=str, index_col=0).group
    return group_split(g.loc[units].to_numpy(), rng)


def load_spec(spec):
    """Returns units, candidate matrix, target matrix, families, code, split."""
    if spec.get('legacy'):
        root = Path(spec['legacy'])
        t = next(x for x in json.loads((root / 'tasks.json').read_text(encoding='utf-8')) if x['task'] == spec['task'])
        m = pd.read_csv(root / 'matrices' / f"{spec['task']}.csv")
        return m['unit'].astype(str).to_numpy(), m[[c for c in m if c.startswith('cand_')]].to_numpy(float), m[t['targets']].to_numpy(float), np.asarray(t['families']), t['code'], spec['split']
    m = pd.read_csv(HERE / 'pools' / f"{spec['source']}.csv", dtype={'unit': str}).set_index('unit')
    tg = list(spec['targets']) + list(spec.get('targets_b', []))
    fam = list(spec['families']) + list(spec.get('families_b', []))
    return m.index.to_numpy(), m[spec['candidates']].to_numpy(float), m[tg].to_numpy(float), np.asarray(fam), spec['code'], spec['split']


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def r5_at(R, paths, chosen, k, xs, ys, xt):
    """As in the frozen job: the R5 panel and native prediction at budget k."""
    opt = chosen[k]
    if opt == 'C0':
        return np.array([], int), np.zeros((len(xt), ys.shape[1])), opt
    res, mode = opt.split('|'); G, path = paths[res]; sets = R.panel_at(path, k)
    panel = np.array(sorted({j for s in sets for j in s}), int) if sets else np.array([], int)
    pred = R.r5_predict(xs, ys, xt, G, sets, mode) if sets else np.zeros((len(xt), ys.shape[1]))
    return panel, pred, opt


def job_main(spec, o, out, ns, ks, fine=False, units_at=None):
    """V, D and U. ns may hold None for all training units."""
    part = out / 'parts' / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    R, V = libs()
    R.RES5 = ['Shared', 'Family'] if spec.get('no_exact') else ['Shared', 'Family', 'Exact']
    t0 = time.time()
    units, x, y, fam, code, split = load_spec(spec)
    te, trn = outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows, per_unit = [], []
    for n in ns:
        if n is not None and n >= len(trn):
            continue
        for r in (range(REPS) if n else [0]):
            sd = [V.SEED, code, o, n or 0, r]
            sub = trn if n is None else np.random.default_rng(sd).choice(trn, n, replace=False)
            xs = x[sub]; ym = y[sub].mean(0); ysd = y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys = (y[sub] - ym) / ysd; yt = (y[te] - ym) / ysd; xt = x[te]
            seed = int(np.random.default_rng(sd).integers(1 << 30))
            base = {'task': spec['task'], 'outer': o, 'n': 'all' if n is None else n, 'n_actual': len(sub), 'rep': r}
            paths = {res: R.select_path(xs, ys, fam, res, seed) for res in R.RES5}
            chosen, _ = R.middle_choice(xs, ys, fam, V.KS, seed)

            def add(method, pred_name, k, k_used, pred, **extra):
                rows.append({**base, 'method': method, 'predictor': pred_name, 'k': k, 'k_used': k_used, **V.metrics(pred, yt, ym, ysd), **extra})
                if n == units_at and k in SAVE_UNITS and method in ('R5', 'decorr', 'random') and pred_name in ('ridge', 'gbt'):
                    per_unit.append(pd.DataFrame({'task': spec['task'], 'outer': o, 'rep': r, 'method': method, 'predictor': pred_name, 'k': k, 'unit': units[te],
                                                  'sse': ((pred - yt) ** 2).sum(1), 'c0': (yt ** 2).sum(1)}))
            for k in ks:
                panels = {'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False), 'decorr': V.decorr(xs, k)}
                r5_panel, r5_pred, opt = r5_at(R, paths, chosen, k, xs, ys, xt)
                add('R5', 'native', k, len(r5_panel), r5_pred, option=opt)
                G, path = paths['Shared']; sets = R.panel_at(path, k)
                add('R5fix_Shared|group', 'native', k, len({j for s in sets for j in s}) if sets else 0, R.r5_predict(xs, ys, xt, G, sets, 'group') if sets else np.zeros_like(yt))
                panels['R5'] = r5_panel
                panels['R5_Shared'] = np.array(sets[0] if sets else [], int)
                for meth, cols in panels.items():
                    add(meth, 'ridge', k, len(cols), R.ridge_predict(xs, ys, xt, cols))
                    add(meth, 'gbt', k, len(cols), V.gbt_predict(xs, ys, xt, cols, seed))
            if fine and n == 100:
                for k in FINE:
                    if k >= x.shape[1]:
                        continue
                    panels = {'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False), 'decorr': V.decorr(xs, k)}
                    for meth, cols in panels.items():
                        add(meth, 'ridge', k, len(cols), R.ridge_predict(xs, ys, xt, cols))
                        add(meth, 'gbt', k, len(cols), V.gbt_predict(xs, ys, xt, cols, seed))
            allc = np.arange(x.shape[1])
            add('all', 'ridge', x.shape[1], x.shape[1], R.ridge_predict(xs, ys, xt, allc))
            add('all', 'gbt', x.shape[1], x.shape[1], V.gbt_predict(xs, ys, xt, allc, seed))
    if per_unit:
        write(pd.concat(per_unit, ignore_index=True), out / 'units' / f"{spec['task']}_{o}.csv")
    write(pd.DataFrame(rows), part)
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def job_wc(spec, o, out, n, ks):
    """W and C: one training size, four repetitions, panels only."""
    part = out / 'parts' / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    R, V = libs()
    R.RES5 = ['Shared', 'Family'] if spec.get('no_exact') else ['Shared', 'Family', 'Exact']
    t0 = time.time()
    units, x, y, fam, code, split = load_spec(spec)
    te, trn = outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows = []
    for r in range(REPS):
        sd = [V.SEED, code, o, n, r]
        sub = np.random.default_rng(sd).choice(trn, n, replace=False)
        xs = x[sub]; ym = y[sub].mean(0); ysd = y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
        ys = (y[sub] - ym) / ysd; yt = (y[te] - ym) / ysd; xt = x[te]
        seed = int(np.random.default_rng(sd).integers(1 << 30))
        paths = {res: R.select_path(xs, ys, fam, res, seed) for res in R.RES5}
        chosen, _ = R.middle_choice(xs, ys, fam, V.KS, seed)
        for k in ks:
            if k >= x.shape[1]:
                continue
            panel, _, opt = r5_at(R, paths, chosen, k, xs, ys, xt)
            panels = {'R5': panel, 'decorr': V.decorr(xs, k), 'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False)}
            for meth, cols in panels.items():
                for name, pred in [('ridge', R.ridge_predict(xs, ys, xt, cols)), ('gbt', V.gbt_predict(xs, ys, xt, cols, seed))]:
                    rows.append({'task': spec['task'], 'source': spec['source'], 'replicate': spec['replicate'], 'size': spec['size'], 'outer': o, 'n': n, 'rep': r, 'method': meth, 'predictor': name, 'k': k, 'k_used': len(cols),
                                 'option': opt if meth == 'R5' else '', **V.metrics(pred, yt, ym, ysd)})
    write(pd.DataFrame(rows), part)
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def job_r(spec, o, out, ns, ks):
    """R: panels chosen for target set A or B, each refitted on the training data for all 18 targets. Per-target errors are saved."""
    part = out / 'parts' / f"{spec['task']}_{o}.csv"
    if part.exists():
        return
    R, V = libs()
    R.RES5 = ['Shared', 'Family', 'Exact']
    t0 = time.time()
    units, x, y, fam, code, split = load_spec(spec)
    na = len(spec['targets']); ia, ib = np.arange(na), np.arange(na, y.shape[1])
    names = list(spec['targets']) + list(spec['targets_b'])
    te, trn = outer_split(split, units, np.random.default_rng([V.SEED, code, o]))
    rows = []
    for n in ns:
        for r in range(REPS):
            sd = [V.SEED, code, o, n, r]
            sub = np.random.default_rng(sd).choice(trn, n, replace=False)
            xs = x[sub]; ym = y[sub].mean(0); ysd = y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys = (y[sub] - ym) / ysd; yt = (y[te] - ym) / ysd; xt = x[te]
            seed = int(np.random.default_rng(sd).integers(1 << 30))
            pick = {}
            for lab, idx in [('R5_A', ia), ('R5_B', ib)]:
                paths = {res: R.select_path(xs, ys[:, idx], fam[idx], res, seed) for res in R.RES5}
                chosen, _ = R.middle_choice(xs, ys[:, idx], fam[idx], V.KS, seed)
                pick[lab] = (paths, chosen, idx)
            for k in ks:
                panels = {'decorr': V.decorr(xs, k), 'random': np.random.default_rng(sd + [k]).choice(x.shape[1], k, replace=False)}
                for lab, (paths, chosen, idx) in pick.items():
                    panels[lab] = r5_at(R, paths, chosen, k, xs, ys[:, idx], xt)[0]
                for meth, cols in panels.items():
                    for pname, pred in [('ridge', R.ridge_predict(xs, ys, xt, cols)), ('gbt', V.gbt_predict(xs, ys, xt, cols, seed))]:
                        sse, c0 = ((pred - yt) ** 2).sum(0), (yt ** 2).sum(0)
                        for j, tg in enumerate(names):
                            rows.append({'task': spec['task'], 'source': spec['source'], 'outer': o, 'n': n, 'rep': r, 'method': meth, 'predictor': pname, 'k': k, 'k_used': len(cols), 'target': tg, 'set': 'A' if j < na else 'B', 'sse': sse[j], 'c0': c0[j]})
    write(pd.DataFrame(rows), part)
    print(spec['task'], o, 'done', round(time.time() - t0), 's', flush=True)


def legacy_specs():
    t13 = json.loads((T13 / 'tasks.json').read_text(encoding='utf-8'))
    t14 = json.loads((T14 / 'tasks.json').read_text(encoding='utf-8'))
    out = [{'legacy': str(T13), 'task': t['task'], 'split': 'random'} for t in t13 if t['set'] in ('main', 'nhanes')]
    for t in t14:
        out.append({'legacy': str(T14), 'task': t['task'], 'split': 'chemotype' if t['set'] == 'kinase' else 'random'})
    return out


def compare(new_file, old_file):
    """Every row of the new file that has a counterpart in the cache must match it exactly."""
    new, old = pd.read_csv(new_file), pd.read_csv(old_file)
    new['n'], old['n'] = new.n.astype(str), old.n.astype(str)
    keys = ['n', 'rep', 'method', 'predictor', 'k']
    m = new.merge(old, on=keys, suffixes=('', '_old'))
    cols = ['sse', 'c0', 'k_used', 'pooled_r', 'target_r']
    main_rows = m[m.method != 'all']        # DEVIATIONS D1: 'all' rows differ by floating-point rounding when the thread count differs
    exact = max(float(np.nanmax(np.abs(main_rows[c] - main_rows[c + '_old']))) for c in cols)
    worst = max(float(np.nanmax(np.abs(m[c] - m[c + '_old']))) for c in cols)
    return len(new), len(m), exact, worst


def check():
    specs = {s['task']: s for s in legacy_specs()}
    out = HERE / 'regression_check'
    Parallel(n_jobs=2)(delayed(job_main)(specs[t], 0, out, [100], [3, 5, 10, 20, 30], False, 100) for t in ['BEAT_T1', 'KIN_T1'])
    for t, cache in [('BEAT_T1', T13 / 'results' / 'parts' / 'BEAT_T1_0.csv'), ('KIN_T1', T14 / 'results_kinase_chemotype' / 'parts' / 'KIN_T1_0.csv')]:
        rows, matched, exact, worst = compare(out / 'parts' / f'{t}_0.csv', cache)
        print(t, 'rows', rows, '| matched in cache', matched, '| largest difference, panel rows', exact, '| all rows', worst, flush=True)
        assert matched == rows and exact == 0 and worst < 1e-10, f'{t}: the copy does not reproduce the cache'
    labels = pd.read_csv(HERE / 'pools' / 'E1_groups.csv', dtype=str, index_col=0).group
    te, trn = group_split(labels.to_numpy(), np.random.default_rng(1))
    assert not set(labels.iloc[te]) & set(labels.iloc[trn])
    print('group split dry run: test', len(te), 'train', len(trn), '| patients on both sides 0', flush=True)
    print('REGRESSION CHECK PASSED', flush=True)


def main():
    tasks = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))
    out = HERE / 'results'
    stages = [
        ('V', [delayed(job_main)(t, o, out / 'V', LADDER + [None], [3, 5, 10, 20, 30], True, 100) for t in tasks if t['analysis'] == 'V' for o in range(5)]),
        ('U', [delayed(job_main)(s, o, out / 'U', [100], [3, 5, 10, 20, 30], True, 100) for s in legacy_specs() for o in range(5)]),
        ('R', [delayed(job_r)(t, o, out / 'R', [50, 100], [5, 10, 20]) for t in tasks if t['analysis'] == 'R' for o in range(5)]),
        ('W', [delayed(job_wc)(t, o, out / 'W', 100, [5, 10, 20]) for t in tasks if t['analysis'] == 'W' for o in range(3)]),
        ('C', [delayed(job_wc)(t, o, out / 'C', 100, [5, 10]) for t in tasks if t['analysis'] == 'C' for o in range(3)]),
    ]
    for name, jobs in stages:
        t0 = time.time()
        print(f'=== stage {name}: {len(jobs)} jobs', flush=True)
        Parallel(n_jobs=N_JOBS, verbose=1)(jobs)
        print(f'=== stage {name} finished in {round((time.time() - t0) / 60)} min', flush=True)


if __name__ == '__main__':
    check() if sys.argv[1:] == ['check'] else main()
