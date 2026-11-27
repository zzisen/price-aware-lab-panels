#!/usr/bin/env python3
"""Task 37 stage 2 (STAGE2_DEV_PLAN.md): clinical learning curves within outcome (development data only: NHANES 1999-2010 training,
NHANES III fixed test; CLHLS Task 33 split). Usage: run_stage2.py prep | run [workers]
prep -> panels_fixed.json (A and B per cohort/outcome; full-training C for reference)
run  -> results/<cohort>_<outcome>_E<E>_r<rep>.csv (one row per subsample)"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(W / 'task21_final'))
import c37lib as G  # noqa: E402
OUT = HERE / 'results'
BUDGET_US = 50.77
NH_OUT = ['all', 'cvd', 'cancer', 'other']
CL_OUT = ['T1', 'T2', 'T3']
E_LEVELS = {'NH': [25, 50, 100, 200, 400], 'CL_T1': [25, 50, 100, 200, 400, 776], 'CL_T2': [25, 50, 85], 'CL_T3': [25, 36]}
_D = {}


# ---------------- NHANES ----------------
def nh():
    if 'nh' not in _D:
        import dm2lib as M
        import t20lib as T
        x, _ = M.load(list('ABCDEF')); Z = M.transform(x)
        xt = M.load_nh3()[0]; Zt = M.transform(xt, ref=x)
        tv, ev = T.outcome_at(xt, 108); uc = xt.ucod.astype(str).to_numpy()
        test = {'all': ev, 'cvd': ev & np.isin(uc, ['001', '005']), 'cancer': ev & (uc == '002'),
                'other': ev & ~np.isin(uc, ['001', '005', '002'])}
        train = {o: M.outcome(x, None if o == 'all' else o) for o in NH_OUT}
        _D['nh'] = dict(M=M, x=x, Z=Z, Zt=Zt, tv=np.asarray(tv, float), test=test, train=train)
    return _D['nh']


def nh_items():
    M = nh()['M']
    return [i for i in M.ITEMS]


def nh_cost(items):
    return nh()['M'].cost(list(items)) if items else 0.0


def nh_score(Z, t, e, folds, base=('age', 'female')):
    M = nh()['M']; cache = {}

    def s(items):
        key = tuple(sorted(M.features_of(list(items))))
        if key not in cache:
            cache[key] = M.gain_ll(Z, t, e, list(key), 1.0, folds, list(base))
        return cache[key]
    return s


def nh_fit_c(Z, t, e, feats, Zt, tt, et):
    M = nh()['M']; cols = ['age', 'female'] + list(feats)
    lp = M.Cox(1.0).fit(Z[cols].to_numpy(), t, e).lp(Zt[cols].to_numpy())
    return float(M.cindex(lp, tt, et))


# ---------------- CLHLS (Task 33 data, split and model) ----------------
def cl():
    if 'cl' not in _D:
        src = open(W / 'task33_clhls_ageing' / 'run_t33.py', encoding='utf-8').read()
        head = src.split('def beam(tr, y, folds, budget, width=4):')[0]
        ns = {'__file__': str(W / 'task33_clhls_ageing' / 'run_t33.py'), '__name__': 'clhls_data'}
        exec(compile(head, 'run_t33_head', 'exec'), ns)
        _D['cl'] = ns
    return _D['cl']


def cl_score(tr, y, folds):
    ns = cl(); cache = {}

    def s(items):
        k = tuple(sorted(items))
        if k not in cache:
            cache[k] = ns['cv_ll'](tr, y, ns['feats_of'](k), folds)
        return cache[k]
    return s


def cl_auc(tr, y, te, yt, feats):
    ns = cl(); import t20lib as T
    m = ns['fit'](ns['design'](tr, tr, feats), y)
    return float(T.auc(m.predict_proba(ns['design'](te, tr, feats))[:, 1], yt))


# ---------------- prep: fixed panels ----------------
def prep_job(arg):
    cohort, o = arg
    if cohort == 'NH':
        d = nh(); M = d['M']; folds = np.array_split(np.random.default_rng(1).permutation(len(d['x'])), 5)
        others = [q for q in NH_OUT if q != o]
        scs = {q: nh_score(d['Z'], *d['train'][q], folds) for q in others}
        b0 = {q: scs[q](()) for q in others}
        joint = lambda its: sum((scs[q](its) - b0[q]) / abs(b0[q]) for q in others)  # noqa: E731
        B, _ = G.greedy(joint, nh_items(), nh_cost, BUDGET_US)
        Cf, _ = G.greedy(nh_score(d['Z'], *d['train'][o], folds), nh_items(), nh_cost, BUDGET_US)
        return f'NH_{o}', dict(B=B, C_full=Cf)
    ns = cl(); x = ns['x']
    others = [q for q in CL_OUT if q != o]
    scs, b0 = {}, {}
    for q in others:
        dq = x[x[q].notna() & x.train]; f = np.random.default_rng(1).permutation(np.arange(len(dq)) % 5)
        scs[q] = cl_score(dq, dq[q].to_numpy().astype(int), f); b0[q] = scs[q](())
    joint = lambda its: sum((scs[q](its) - b0[q]) / abs(b0[q]) for q in others)  # noqa: E731
    B, _ = G.greedy(joint, list(ns['ITEMS']), lambda its: ns['cost'](ns['feats_of'](its)), ns['BUDGET'])
    d = x[x[o].notna() & x.train]; f = np.random.default_rng(1).permutation(np.arange(len(d)) % 5)
    Cf, _ = G.greedy(cl_score(d, d[o].to_numpy().astype(int), f), list(ns['ITEMS']), lambda its: ns['cost'](ns['feats_of'](its)), ns['BUDGET'])
    return f'CL_{o}', dict(B=B, C_full=Cf)


def prep(workers):
    d = nh(); M = d['M']
    feats = M.FEATURES; Xa = d['Z'][feats].to_numpy(); Xa = (Xa - Xa.mean(0)) / Xa.std(0)
    A_nh = G.fsca_items(Xa, feats, M.ITEMS, nh_items(), nh_cost, BUDGET_US)
    ns = cl(); x = ns['x']; tr = x[x.train]; Xc = np.log(np.maximum(tr[ns['MENU']].astype(float), 0) + 1).to_numpy()
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    A_cl = G.fsca_items(Xc, list(ns['MENU']), ns['ITEMS'], list(ns['ITEMS']), lambda its: ns['cost'](ns['feats_of'](its)), ns['BUDGET'])
    res = dict(A_NH=A_nh, A_CL=A_cl, D_NH=list(M.STANDARD['CMP+CBC+LIPID+A1C']), D_CL=list(ns['ROUTINE']), budget_cl=ns['BUDGET'])
    _D.clear()
    with Pool(workers) as pl:
        for k, v in pl.imap_unordered(prep_job, [('NH', o) for o in NH_OUT] + [('CL', o) for o in CL_OUT]):
            res[k] = v; print(k, v, flush=True)
    json.dump(res, open(HERE / 'panels_fixed.json', 'w'), indent=1)


# ---------------- one subsample ----------------
def run_job(job):
    cohort, o, E, r = job
    f = OUT / f'{cohort}_{o}_E{E}_r{r}.csv'
    if f.exists():
        return f.name, 0.0
    t0 = time.time()
    P = json.load(open(HERE / 'panels_fixed.json'))
    rng = np.random.default_rng([37, 0 if cohort == 'NH' else 1, NH_OUT.index(o) if cohort == 'NH' else CL_OUT.index(o), E, r])
    if cohort == 'NH':
        d = nh(); M = d['M']; t, e = d['train'][o]
        ev, ne = np.flatnonzero(e), np.flatnonzero(~e)
        m = int(round(E * len(ne) / len(ev))) if E < len(ev) else len(ne)
        sub = np.sort(np.concatenate([rng.choice(ev, min(E, len(ev)), replace=False), rng.choice(ne, m, replace=False)]))
        Zs, ts, es = d['Z'].iloc[sub], t[sub], e[sub]
        folds = np.array_split(np.random.default_rng([37, E, r]).permutation(len(sub)), 5)
        items, cost, budget = nh_items(), nh_cost, BUDGET_US
        feats_of = lambda its: M.features_of(list(its))  # noqa: E731
        sc = nh_score(Zs, ts, es, folds)
        pan = dict(A=P['A_NH'], B=P[f'NH_{o}']['B'], D=P['D_NH'], base=[])
        C, _ = G.greedy(sc, items, cost, budget); pan['C'] = C
        cvll = {k: sc(tuple(v)) for k, v in pan.items()}
        cvll['ALL'] = M.gain_ll(Zs, ts, es, list(M.FEATURES), 1.0, folds)
        # nested CV for C (re-selected inside each outer fold) and the fixed panels on the same folds
        nest = {k: 0.0 for k in ['A', 'B', 'C', 'D', 'base']}
        X_all = Zs.reset_index(drop=True)
        for te in folds:
            trn = np.setdiff1d(np.arange(len(ts)), te)
            fi = np.array_split(np.random.default_rng([37, E, r, 3]).permutation(len(trn)), 5)
            Ci, _ = G.greedy(nh_score(X_all.iloc[trn], ts[trn], es[trn], fi), items, cost, budget)
            for k, v in list(pan.items()) + [('C', Ci)]:
                cols = ['age', 'female'] + feats_of(v)
                X = X_all[cols].to_numpy()
                lp = M.Cox(1.0).fit(X[trn], ts[trn], es[trn]).lp(X[te])
                if k == 'C' and v is Ci or k != 'C':
                    nest[k] += M.partial_loglik(lp, ts[te], es[te])
        test = {k: nh_fit_c(Zs, ts, es, feats_of(v), d['Zt'], d['tv'], d['test'][o]) for k, v in pan.items()}
        test['ALL'] = nh_fit_c(Zs, ts, es, list(M.FEATURES), d['Zt'], d['tv'], d['test'][o])
        d_items = {k: feats_of(v) for k, v in pan.items()}
    else:
        ns = cl(); x = ns['x']; dd = x[x[o].notna()]; trd, ted = dd[dd.train], dd[~dd.train]
        y = trd[o].to_numpy().astype(int); ev, ne = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        m = int(round(E * len(ne) / len(ev))) if E < len(ev) else len(ne)
        sub = np.sort(np.concatenate([rng.choice(ev, min(E, len(ev)), replace=False), rng.choice(ne, m, replace=False)]))
        tr, ys = trd.iloc[sub], y[sub]
        folds = np.random.default_rng([37, E, r]).permutation(np.arange(len(sub)) % 5)
        items = list(ns['ITEMS']); cost = lambda its: ns['cost'](ns['feats_of'](its)); budget = ns['BUDGET']  # noqa: E731
        feats_of = ns['feats_of']
        sc = cl_score(tr, ys, folds)
        pan = dict(A=P['A_CL'], B=P[f'CL_{o}']['B'], D=None, base=[])
        C, _ = G.greedy(sc, items, cost, budget); pan['C'] = C
        cvll = {k: (sc(tuple(v)) if k != 'D' else ns['cv_ll'](tr, ys, list(P['D_CL']), folds)) for k, v in pan.items()}
        cvll['ALL'] = ns['cv_ll'](tr, ys, list(ns['MENU']), folds)
        nest = {k: 0.0 for k in ['A', 'B', 'C', 'D', 'base']}
        for kf in range(5):
            a, b = np.flatnonzero(folds != kf), np.flatnonzero(folds == kf)
            fi = np.random.default_rng([37, E, r, 3]).permutation(np.arange(len(a)) % 5)
            Ci, _ = G.greedy(cl_score(tr.iloc[a], ys[a], fi), items, cost, budget)
            for k, v in [('A', pan['A']), ('B', pan['B']), ('base', []), ('C', Ci)]:
                fe = feats_of(tuple(v)) if v else []
                mdl = ns['fit'](ns['design'](tr.iloc[a], tr.iloc[a], fe), ys[a])
                p = np.clip(mdl.predict_proba(ns['design'](tr.iloc[b], tr.iloc[a], fe))[:, 1], 1e-12, 1 - 1e-12)
                nest[k] += float(np.sum(ys[b] * np.log(p) + (1 - ys[b]) * np.log(1 - p)))
            fe = list(P['D_CL']); mdl = ns['fit'](ns['design'](tr.iloc[a], tr.iloc[a], fe), ys[a])
            p = np.clip(mdl.predict_proba(ns['design'](tr.iloc[b], tr.iloc[a], fe))[:, 1], 1e-12, 1 - 1e-12)
            nest['D'] += float(np.sum(ys[b] * np.log(p) + (1 - ys[b]) * np.log(1 - p)))
        yt = ted[o].to_numpy().astype(int)
        d_items = {k: (feats_of(tuple(v)) if v else []) for k, v in pan.items() if k != 'D'}; d_items['D'] = list(P['D_CL'])
        test = {k: cl_auc(tr, ys, ted, yt, d_items[k]) for k in ['A', 'B', 'C', 'D', 'base']}
        test['ALL'] = cl_auc(tr, ys, ted, yt, list(ns['MENU']))
    row = dict(cohort=cohort, outcome=o, E=int(es.sum()) if cohort == 'NH' else int(ys.sum()), n=len(sub), rep=r,
               C_items='+'.join(pan['C']), d_C=len(d_items['C']), d_D=len(d_items['D']), k_items_C=len(pan['C']), p=(35 if cohort == 'NH' else len(cl()['MENU'])))
    for k in ['A', 'B', 'C', 'D', 'base', 'ALL']:
        row['test_' + k] = test[k]; row['cv_' + k] = cvll.get(k, np.nan)
    for k in ['A', 'B', 'C', 'D', 'base']:
        row['nest_' + k] = nest[k]
    pd.DataFrame([row]).to_csv(f, index=False)
    return f.name, round(time.time() - t0, 1)


def jobs():
    J = []
    for o in NH_OUT:
        for E in E_LEVELS['NH']:
            if E > 200 and o != 'all':
                continue                                                   # n would exceed 10 000 (compute limit; STAGE2_DEV_PLAN note)
            J += [('NH', o, E, r) for r in range(5 if E <= 100 else 3)]
    for o in CL_OUT:
        for E in E_LEVELS['CL_' + o]:
            J += [('CL', o, E, r) for r in range(10 if E < max(E_LEVELS['CL_' + o]) else 3)]
    return sorted(J, key=lambda j: -j[2])                                   # largest first (longest)


if __name__ == '__main__':
    mode = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    if mode == 'prep':
        prep(workers)
    else:
        OUT.mkdir(exist_ok=True); J = jobs()
        if len(sys.argv) > 3:
            J = [j for j in J if j[2] == int(sys.argv[3])][:2]
        log = open(HERE / 'stage2_log.txt', 'a', encoding='utf-8'); log.write(f'start {time.ctime()} jobs {len(J)}\n'); log.flush()
        t0 = time.time()
        with Pool(workers) as pl:
            for i, (name, sec) in enumerate(pl.imap_unordered(run_job, J)):
                log.write(f'{i + 1}/{len(J)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
        log.write(f'done {time.ctime()}\n')
