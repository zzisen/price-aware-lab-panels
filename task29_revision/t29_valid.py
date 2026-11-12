#!/usr/bin/env python3
"""Task 29 C5 and D3 (PLAN.md): external validation of the complete-menu panels (NHANES III, 2011-14, 2015-16, Understanding Society) and of
the programme references (NBPHS in CHNS and CLHLS; NHS Health Check in Understanding Society). Usage: t29_valid.py <NH3|GH|I|UKHLS|CHNS|CLHLS>
Models: ridge Cox (lambda 1) fitted on NHANES 1999-2010 with age and sex, exactly as in the published analyses. Discrimination: Harrell's C
(survival cohorts) or AUC (binary cohorts); net benefit at the cohort threshold (CLHLS: inverse-probability-of-censoring weighted; CHNS:
72-month risk, threshold 6%). Paired bootstrap, 1000 resamples. Output: valid_<cohort>.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t29lib as Q  # noqa: E402
M, T, L = Q.M, Q.T, Q.L
import t21lib as X  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]; NB_ = 1000; rng = np.random.default_rng(20261113)
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr)
out = dict(cohort=COH)

if COH in ('NH3', 'GH', 'I'):
    SU = json.load(open(Q.HERE / 'sel_us.json')); SEL = json.load(open(Q.W / 'task19_dm2' / 'selected_all.json'))
    H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]; TAU = {108: .10, 60: .05, 36: .03}[H]
    xte = M.load_nh3()[0] if COH == 'NH3' else (pd.concat([T.load_gh(c)[0] for c in 'GH']) if COH == 'GH' else X.load_i()[0])
    Ztr, Z = M.transform(xtr), M.transform(xte, ref=xtr); t, e = T.outcome_at(xte, H); kind = 'C'
    cp = {p['budget']: p['features'] for p in SU['coai_full']['panels']}; bm, br = sorted(cp)
    P = {'age+sex only': [], 'routine': SU['routine']['features'], 'CMP+CBC': M.features_of(['CMP', 'CBC']),
         'old matched': M.features_of(SEL['US']['matched']['items']), 'old aware': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']),
         'new matched': SU['matched']['features'], 'new aware': SU['aware']['features'], 'CoAI full @matched': cp[bm], 'CoAI full @routine': cp[br],
         'lasso-pw within routine (full)': SU['lasso_full']['price-weighted']['within']['features']}
    if SU['lasso_full']['price-weighted']['matched']:
        P['lasso-pw matched (full)'] = SU['lasso_full']['price-weighted']['matched']['features']
    if not xte.crp.notna().all():
        P = {k: v for k, v in P.items() if 'crp' not in v}
    price = lambda f: Q.cost_full(f) if f else 0.0  # noqa: E731
    pairs = [('new matched', 'routine'), ('new aware', 'routine'), ('new matched', 'CMP+CBC'), ('new matched', 'CoAI full @matched'),
             ('new aware', 'CoAI full @routine'), ('new aware', 'lasso-pw within routine (full)'), ('new matched', 'old matched'), ('new aware', 'old aware')]
    nbpairs = [('new matched', 'routine'), ('new aware', 'routine')]
    w_ = None
elif COH == 'UKHLS':
    import ukhls_menu as U
    import ukhls_lib as K
    U.install(); SU = json.load(open(Q.HERE / 'sel_uk.json')); S0 = json.load(open(Q.W / 'task27_ukhls' / 'selected_ukhls.json'))
    H, TAU, kind = 108, .10, 'AUC'
    xte, _ = K.load_ukhls(); Ztr, Z = T.transform_sub(xtr, xtr, U.MENU), T.transform_sub(xte, xtr, U.MENU); y = xte.died.to_numpy().astype(bool)
    cp = {p['budget']: p['features'] for p in SU['coai_full']['panels']}
    P = {'age+sex only': [], 'routine': SU['routine']['features'], 'old matched': S0['matched']['features'], 'old aware': S0['aware_at_routine']['features'],
         'new matched': SU['matched']['features'], 'new aware': SU['aware_measured']['features'], 'new aware (as-ordered budget)': SU['aware_ordered']['features'],
         'NHS Health Check': SU['nhshc']['features'], 'matched to Health Check': SU['matched_hc']['features'], 'Health Check budget': SU['aware_hc']['features'],
         'CoAI full @matched': cp[sorted(cp)[0]], 'CoAI full @measured routine': cp[sorted(cp)[1]], 'CoAI full @ordered routine': cp[sorted(cp)[2]]}
    price = lambda f: Q.cost_full(f) if f else 0.0  # noqa: E731
    pairs = [('new matched', 'routine'), ('new aware', 'routine'), ('new aware (as-ordered budget)', 'routine'), ('new matched', 'CoAI full @matched'),
             ('new aware', 'CoAI full @measured routine'), ('Health Check budget', 'NHS Health Check'), ('matched to Health Check', 'NHS Health Check'),
             ('Health Check budget', 'routine'), ('NHS Health Check', 'age+sex only')]
    nbpairs = [('new matched', 'routine'), ('new aware', 'routine'), ('Health Check budget', 'NHS Health Check'), ('NHS Health Check', 'age+sex only')]
else:
    import clhls_lib as C
    T.install_cn(); C.install(); SC = json.load(open(Q.HERE / 'sel_cn.json'))[COH]
    if COH == 'CHNS':
        S0 = json.load(open(Q.W / 'task20_chns' / 'selected_cn.json')); menu = T.SHARED; H, TAU, kind = 72, .06, 'AUC'
        xte, _ = T.load_chns(); y = xte.died.to_numpy().astype(bool); rout = T.CN_ROUTINE_SHARED
    else:
        S0 = json.load(open(Q.W / 'task25_newdata' / 'selected_clhls.json')); menu = C.MENU; H, TAU, kind = 60, .05, 'C'
        xa, _ = C.load_clhls(); msk = (xa.age_group == '65-79').to_numpy(); xte = xa[msk]; t, e = C.outcome(xa); t, e = t[msk], e[msk]; rout = C.ROUTINE_CL
    Ztr, Z = T.transform_sub(xtr, xtr, menu), T.transform_sub(xte, xtr, menu)
    P = {'age+sex only': [], 'routine': rout, 'matched': S0['matched']['features'], 'aware@routine': S0['aware_at_routine']['features'],
         'NBPHS check': SC['nbphs']['features'], 'matched to NBPHS': SC['matched']['features'], 'NBPHS budget': SC['aware']['features']}
    price = lambda f: T.cost_cn(f) if f else 0.0  # noqa: E731
    pairs = [('NBPHS budget', 'NBPHS check'), ('matched to NBPHS', 'NBPHS check'), ('matched', 'NBPHS check'), ('aware@routine', 'NBPHS check'),
             ('NBPHS check', 'age+sex only'), ('routine', 'NBPHS check')]
    nbpairs = [('NBPHS budget', 'NBPHS check'), ('matched to NBPHS', 'NBPHS check'), ('NBPHS check', 'age+sex only')]

LP, R = {}, {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    LP[k] = m.lp(Z[cols].to_numpy()); R[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H), Z[cols].to_numpy())
if kind == 'C':
    w, died = Q.ipcw(t, e, H)
    disc = lambda s, i: M.cindex(s[i], t[i], e[i])  # noqa: E731

    def nbf(r, i):
        wi, di = Q.ipcw(t[i], e[i], H); return Q.nb_ipcw(r[i], wi, di, TAU)
    n = len(t); out['deaths'] = int(e.sum())
else:
    disc = lambda s, i: T.auc(s[i], y[i])  # noqa: E731
    nbf = lambda r, i: Q.nb_ipcw(r[i], np.ones(len(i)), y[i], TAU)  # noqa: E731
    n = len(y); out['deaths'] = int(y.sum())
allidx = np.arange(n)
out.update(n=n, horizon=H, tau=TAU, metric=kind,
           panels={k: dict(features=list(f), price=price(f), order=(Q.order_full(f) if COH in ('NH3', 'GH', 'I', 'UKHLS') else T.describe_cn(f)),
                           disc=disc(LP[k], allidx), NB=nbf(R[k], allidx), mean_risk=float(R[k].mean())) for k, f in P.items()})
bc = {f'{a} - {b}': [] for a, b in pairs}; bn = {f'{a} - {b}': [] for a, b in nbpairs}
for _ in range(NB_):
    i = rng.integers(0, n, n)
    if kind == 'AUC' and (y[i].all() or (~y[i]).all()):
        continue
    D = {k: disc(LP[k], i) for k in set(x_ for p in pairs for x_ in p)}
    for a, b in pairs:
        bc[f'{a} - {b}'].append(D[a] - D[b])
    for a, b in nbpairs:
        bn[f'{a} - {b}'].append(nbf(R[a], i) - nbf(R[b], i))
pp = out['panels']
out['contrasts'] = {f'{a} - {b}': dict(point=pp[a]['disc'] - pp[b]['disc'], ci=Q.ci(bc[f'{a} - {b}'])) for a, b in pairs}
out['nb_contrasts'] = {f'{a} - {b}': dict(point=pp[a]['NB'] - pp[b]['NB'], ci=Q.ci(bn[f'{a} - {b}'])) for a, b in nbpairs}
json.dump(out, open(Q.HERE / f'valid_{COH}.json', 'w'), indent=1, default=float)
for k, v in pp.items():
    print(f"{k:34s} {v['price']:7.2f} {kind} {v['disc']:.4f} NB {v['NB']:.4f} {v['order']}")
for k, v in {**out['contrasts'], **{'NB ' + a: b for a, b in out['nb_contrasts'].items()}}.items():
    print(f"{k:70s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
