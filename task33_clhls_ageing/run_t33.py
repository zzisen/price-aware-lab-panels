#!/usr/bin/env python3
"""Task 33 (PLAN_FROZEN.md): CLHLS ageing outcomes - does the best laboratory panel change with the target?
Targets: T1 death by the 2018 wave, T2 incident ADL disability at 2018, T3 incident cognitive impairment (education-specific MMSE cut-offs) at 2018.
Within CLHLS: stratified 60/40 split (seed 20261125); ridge logistic (C = 1) with age and sex; analytes log(x + 1), standardised on training;
beam search width 4 under the routine budget, 5-fold CV log-likelihood (fold seed 1); Chinese prices. Test-set AUC, 1000 bootstrap resamples
(seed 20261126). Output: t33_results.json, t33_print.txt (stdout)."""
import itertools
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task25_newdata'))
import clhls_lib as C  # noqa: E402
M, T = C.M, C.T
try:
    import psutil
    psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
EXT = C.EXT

# ---------------- data ----------------
x, counts = C.load_clhls()
alb = []
for wave in (2012, 2014):
    b = pd.read_csv(next(EXT.rglob(f'biomarker_dataset_CLHLS_{wave}.tab')), sep='\t'); b.columns = [c.strip('"').lower() for c in b.columns]
    b = b.drop_duplicates('id').set_index('id'); alb.append(pd.DataFrame({'albumin': C.clean(b['alb']), 'wave': wave}))
alb = pd.concat(alb).reset_index().rename(columns={'index': 'id'})
x = x.reset_index().rename(columns={'index': 'id'}) if 'id' not in x.columns else x.reset_index(drop=True)
x = x.merge(alb, left_on=['pid', 'wave'], right_on=['id', 'wave'], how='left', suffixes=('', '_b'))
counts['albumin_missing'] = int(x.albumin.isna().sum()); x = x[x.albumin.notna()].copy(); counts['analysed'] = len(x)

ADL = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
MM1 = ['c11', 'c12', 'c13', 'c14', 'c15', 'c21a', 'c21b', 'c21c', 'c31a', 'c31b', 'c31c', 'c31d', 'c31e', 'c32', 'c41a', 'c41b', 'c41c',
       'c51a', 'c51b', 'c52', 'c53a', 'c53b', 'c53c']
SAV = {2012: ('clhls_2011_2018_longitudinal_dataset_released_version1.sav', ['dth11_14', 'dth14_18']),
       2014: ('clhls_2014_2018_longitudinal_dataset_released_version1.sav', ['dth14_18'])}


def adl_status(L, suf):
    v = pd.concat([pd.to_numeric(L[a + suf], errors='coerce') for a in ADL], axis=1)
    miss = v.isin([8, 9]).any(axis=1) | v.isna().any(axis=1)
    dis = v.isin([2, 3]).any(axis=1).astype(float)
    return dis.where(~miss)


def mmse(L, suf):
    v = pd.concat([pd.to_numeric(L[a + suf], errors='coerce') for a in MM1], axis=1)
    miss = (v == 9).any(axis=1) | v.isna().any(axis=1)
    s = v.where(v == 1, 0).sum(axis=1)
    f = pd.to_numeric(L['c16' + suf], errors='coerce')
    miss |= (f == 99) | f.isna()
    s = s + np.where(f == 88, 0, np.minimum(f.fillna(0), 7))
    return s.where(~miss)


def impaired(score, edu):
    e = pd.to_numeric(edu, errors='coerce'); e = e.where(~e.isin([88, 99]), 0).fillna(0)
    cut = np.where(e == 0, 18, np.where(e <= 6, 21, 25))
    return (score < cut).astype(float).where(score.notna())


rows = []
for wave, (sav, dvars) in SAV.items():
    p = next(EXT.rglob(sav)); _, meta = pyreadstat.read_sav(str(p), metadataonly=True, encoding='gbk')
    need = ['id', 'f1'] + dvars + [a + s for a in ADL + MM1 + ['c16'] for s in ('', '_18')]
    L, _ = pyreadstat.read_sav(str(p), usecols=[c for c in need if c in meta.column_names], encoding='gbk')
    L = L.drop_duplicates('id').set_index('id')
    d = pd.DataFrame(index=L.index); d['wave'] = wave
    st = [pd.to_numeric(L[v], errors='coerce') for v in dvars]
    died = np.zeros(len(L), bool)
    for s in st:
        died |= s.isin([1, 2]).to_numpy()
    alive = (st[-1] == 0).to_numpy() & ~died
    d['T1'] = np.where(died, 1.0, np.where(alive, 0.0, np.nan))
    a0, a18 = adl_status(L, ''), adl_status(L, '_18')
    m0, m18 = impaired(mmse(L, ''), L['f1']), impaired(mmse(L, '_18'), L['f1'])
    d['adl0'], d['adl18'], d['cog0'], d['cog18'], d['alive18'] = a0, a18, m0, m18, alive
    d['T2'] = np.where((a0 == 0) & alive & a18.notna(), a18, np.nan)
    d['T3'] = np.where((m0 == 0) & alive & m18.notna(), m18, np.nan)
    known = died | alive
    d['C2'] = np.where((a0 == 0) & known & (died | a18.notna()), np.where(died, 1.0, a18), np.nan)
    d['C3'] = np.where((m0 == 0) & known & (died | m18.notna()), np.where(died, 1.0, m18), np.nan)
    rows.append(d.reset_index())
O = pd.concat(rows)
x = x.merge(O, left_on=['pid', 'wave'], right_on=['id', 'wave'], how='left', suffixes=('', '_o'))
x['age_group'] = np.where(x.age < 80, '65-79', '80+')

# ---------------- split ----------------
rng = np.random.default_rng(20261125)
x['train'] = False
for _, g in x.groupby(['wave', 'age_group']):
    idx = rng.permutation(g.index.to_numpy()); x.loc[idx[:int(round(0.6 * len(idx)))], 'train'] = True

MENU = C.MENU + ['albumin']
ITEMS = {a: [a] for a in C.CHEM_CL + ['albumin']}; ITEMS['CBC'] = C.CBC_CL
ROUTINE = list(C.ROUTINE_CL)
cost = lambda f: T.cost_cn(list(f)) if len(f) else 0.0  # noqa: E731
BUDGET = cost(ROUTINE)


def design(df, ref, feats):
    Z = pd.DataFrame({'age': df.age.astype(float), 'female': (df.sex == 2).astype(float)}, index=df.index)
    for a in feats:
        Z[a] = np.log(np.maximum(df[a].astype(float), 0) + 1)
    R = pd.DataFrame({'age': ref.age.astype(float), 'female': (ref.sex == 2).astype(float)}, index=ref.index)
    for a in feats:
        R[a] = np.log(np.maximum(ref[a].astype(float), 0) + 1)
    mu, sd = R.mean(), R.std().replace(0, 1)
    return ((Z - mu) / sd).to_numpy()


def fit(X, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def cv_ll(tr, y, feats, folds):
    ll = 0.0
    for k in range(5):
        te_ = folds == k
        a, b = tr[~te_], tr[te_]
        m = fit(design(a, a, feats), y[~te_]); p = np.clip(m.predict_proba(design(b, a, feats))[:, 1], 1e-12, 1 - 1e-12)
        ll += float(np.sum(y[te_] * np.log(p) + (1 - y[te_]) * np.log(1 - p)))
    return ll


def feats_of(items):
    return sorted(set(a for it in items for a in ITEMS[it]))


def beam(tr, y, folds, budget, width=4):
    cache = {}

    def sc(items):
        k = tuple(sorted(items))
        if k not in cache:
            cache[k] = cv_ll(tr, y, feats_of(k), folds)
        return cache[k]
    sc(())
    frontier = [()]
    while frontier:
        cand = set()
        for s in frontier:
            for it in ITEMS:
                if it not in s:
                    n = tuple(sorted(s + (it,)))
                    if cost(feats_of(n)) <= budget + 1e-9:
                        cand.add(n)
        cand = [c for c in cand if c not in cache or True]
        scored = sorted(((sc(c), c) for c in cand), reverse=True)
        frontier = [c for _, c in scored[:width]] if scored else []
    return cache


res = dict(plan_hash=open(HERE / 'PLAN_HASH.txt').read().split()[0], counts=counts, budget=BUDGET, routine=ROUTINE, targets={})
PAN = {}
for Tn in ['T1', 'T2', 'T3']:
    d = x[x[Tn].notna()]; tr = d[d.train]; y = tr[Tn].to_numpy().astype(int)
    folds = np.random.default_rng(1).permutation(np.arange(len(tr)) % 5)
    cache = beam(tr, y, folds, BUDGET)
    rcv = cv_ll(tr, y, ROUTINE, folds)
    aware = max(((v, -cost(feats_of(k))), k) for k, v in cache.items())[1]
    ok = [k for k, v in cache.items() if v >= rcv - 1e-9]
    matched = sorted(((cost(feats_of(k)), -cache[k]), k) for k in ok)[0][1] if ok else None
    PAN[Tn] = dict(aware=feats_of(aware), aware_items=list(aware), matched=(feats_of(matched) if matched is not None else None),
                   matched_items=(list(matched) if matched is not None else None))
    res['targets'][Tn] = dict(n=len(d), events=int(d[Tn].sum()), n_train=len(tr), events_train=int(y.sum()), n_test=int((~d.train).sum()),
                              events_test=int(d.loc[~d.train, Tn].sum()), aware=PAN[Tn]['aware_items'], aware_price=cost(PAN[Tn]['aware']),
                              matched=PAN[Tn]['matched_items'], matched_price=(cost(PAN[Tn]['matched']) if PAN[Tn]['matched'] else None),
                              routine_cv=rcv, aware_cv=cache[aware], evaluated=len(cache))
    print(Tn, res['targets'][Tn], flush=True)


def evaluate(Tn, sets, pairs, seed=20261126, nb=1000, split_col='train'):
    d = x[x[Tn].notna()]; tr, te = d[d[split_col]], d[~d[split_col]]
    ytr, yte = tr[Tn].to_numpy().astype(int), te[Tn].to_numpy().astype(bool)
    S = {k: fit(design(tr, tr, f), ytr).predict_proba(design(te, tr, f))[:, 1] for k, f in sets.items()}
    pt = {k: T.auc(v, yte) for k, v in S.items()}
    r = np.random.default_rng(seed); bs = {f'{a} - {b}': [] for a, b in pairs}
    n = len(yte)
    for _ in range(nb):
        i = r.integers(0, n, n)
        if yte[i].all() or (~yte[i]).all():
            continue
        for a, b in pairs:
            bs[f'{a} - {b}'].append(T.auc(S[a][i], yte[i]) - T.auc(S[b][i], yte[i]))
    ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else [np.nan, np.nan]  # noqa: E731
    return pt, {f'{a} - {b}': dict(point=pt[a] - pt[b], ci=ci(bs[f'{a} - {b}'])) for a, b in pairs}, (te, S)


EV = {}
for Tn in ['T1', 'T2', 'T3']:
    sets = {'age+sex only': [], 'routine': ROUTINE, 'aware (own target)': PAN[Tn]['aware'], 'death panel': PAN['T1']['aware']}
    if PAN[Tn]['matched']:
        sets['matched (own target)'] = PAN[Tn]['matched']
    pairs = [('aware (own target)', 'death panel'), ('aware (own target)', 'routine'), ('routine', 'age+sex only'), ('aware (own target)', 'age+sex only')]
    if 'matched (own target)' in sets:
        pairs.append(('matched (own target)', 'routine'))
    pt, con, (te, S) = evaluate(Tn, sets, pairs)
    sub = {}
    for g in ['65-79', '80+']:
        m = (te.age_group == g).to_numpy(); yt = te[Tn].to_numpy().astype(bool)[m]
        if yt.any() and (~yt).any():
            sub[g] = {k: T.auc(v[m], yt) for k, v in S.items()} | dict(n=int(m.sum()), events=int(yt.sum()))
    EV[Tn] = dict(auc=pt, contrasts=con, by_age=sub, prices={k: cost(f) for k, f in sets.items()})
res['evaluation'] = EV

b1 = [EV[t]['contrasts']['aware (own target) - death panel'] for t in ('T2', 'T3')]
b2 = [EV[t]['contrasts']['aware (own target) - routine'] for t in ('T2', 'T3')]
res['B1_pass'] = bool(all(c['point'] > 0 for c in b1) and any(c['ci'][0] > 0 for c in b1))
res['B2_pass'] = bool(any(c['ci'][0] > 0 for c in b2))

J = lambda a, b: len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)  # noqa: E731
res['jaccard'] = {f'{a}-{b}': J(PAN[a]['aware'], PAN[b]['aware']) for a, b in itertools.combinations(['T1', 'T2', 'T3'], 2)}

# composites (descriptive): panels for T2/T3 and death panel, evaluated on C2/C3
for Cn, Tn in [('C2', 'T2'), ('C3', 'T3')]:
    sets = {'age+sex only': [], 'routine': ROUTINE, 'aware (T panel)': PAN[Tn]['aware'], 'death panel': PAN['T1']['aware']}
    pt, con, _ = evaluate(Cn, sets, [('aware (T panel)', 'death panel'), ('aware (T panel)', 'routine')])
    res['composite_' + Cn] = dict(n=int(x[Cn].notna().sum()), events=int(x[Cn].sum()), auc=pt, contrasts=con)

# robustness: 10 further split seeds, re-selecting panels; point estimates only
rob = []
for s in range(1, 11):
    r_ = np.random.default_rng(s); col = f'train_{s}'; x[col] = False
    for _, g in x.groupby(['wave', 'age_group']):
        idx = r_.permutation(g.index.to_numpy()); x.loc[idx[:int(round(0.6 * len(idx)))], col] = True
    P2 = {}
    for Tn in ['T1', 'T2', 'T3']:
        d = x[x[Tn].notna()]; tr = d[d[col]]; y = tr[Tn].to_numpy().astype(int)
        folds = np.random.default_rng(1).permutation(np.arange(len(tr)) % 5)
        cache = beam(tr, y, folds, BUDGET); P2[Tn] = feats_of(max(((v, -cost(feats_of(k))), k) for k, v in cache.items())[1])
    row = dict(seed=s)
    for Tn in ['T2', 'T3']:
        pt, _, _ = evaluate(Tn, {'own': P2[Tn], 'death': P2['T1'], 'routine': ROUTINE}, [('own', 'death')], nb=0, split_col=col)
        row[Tn] = dict(own_minus_death=pt['own'] - pt['death'], own_minus_routine=pt['own'] - pt['routine'], own=P2[Tn])
    row['T1'] = P2['T1']; rob.append(row); print('seed', s, row, flush=True)
res['robustness'] = rob
res['robustness_share_positive'] = {Tn: dict(vs_death=float(np.mean([r[Tn]['own_minus_death'] > 0 for r in rob])),
                                             vs_routine=float(np.mean([r[Tn]['own_minus_routine'] > 0 for r in rob]))) for Tn in ['T2', 'T3']}
json.dump(res, open(HERE / 't33_results.json', 'w'), indent=1, default=float)
print(json.dumps({k: res[k] for k in ['counts', 'budget', 'B1_pass', 'B2_pass', 'jaccard', 'robustness_share_positive']}, indent=1, default=float))
for Tn in ['T1', 'T2', 'T3']:
    print(Tn, res['targets'][Tn]['n'], res['targets'][Tn]['events'], 'aware', res['targets'][Tn]['aware'], res['targets'][Tn]['aware_price'])
    for k, v in EV[Tn]['auc'].items():
        print(f'   {k:24s} {EV[Tn]["prices"][k]:6.1f}  AUC {v:.4f}')
    for k, v in EV[Tn]['contrasts'].items():
        print(f"   {k:48s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
for Cn in ['C2', 'C3']:
    print(Cn, res['composite_' + Cn])
