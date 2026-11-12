#!/usr/bin/env python3
"""Task 21 phase-1 training report (NHANES 1999-2010 only; no validation data). Out-of-fold (5-fold) performance of the candidate panels,
spline-Cox sensitivity, cardiovascular panels against guideline laboratory sets, income/education subgroups, price-perturbation stability
and the sequential policy on cross-fitted risks. Output: dev_train_report.txt / .json"""
import json
import numpy as np
import pandas as pd
import dm2lib as M
import t21lib as X
M.L.low_priority()
SEL = json.load(open('../task19_dm2/selected_all.json')); CVD = json.load(open('selected_cvd.json'))
lc = pd.read_csv('lasso_candidates.csv')
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
cl = M.clinical(x); sesg = X.ses(x)
folds = np.array_split(np.random.default_rng(11).permutation(len(t)), 5)
ROUT = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C'])
rt_cv = M.gain_ll(Z, t, e, ROUT, 1.0, np.array_split(np.random.default_rng(1).permutation(len(t)), 5))


def oof(Zd, cols, tt, ee, spline=False):
    lp = np.zeros(len(tt))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(tt)), te)
        if spline:
            A, B = X.spline_design(Zd.iloc[tr], Zd.iloc[te], cols)
        else:
            A, B = Zd[cols].to_numpy()[tr], Zd[cols].to_numpy()[te]
        lp[te] = M.Cox(1.0).fit(A, tt[tr], ee[tr]).lp(B)
    return lp


def lasso_pick(kind):
    s = lc[lc.kind == kind]
    m = s[s.cv >= rt_cv].sort_values(['US', 'cv'], ascending=[True, False]).head(1)
    a = s[s.US <= M.cost(M.STANDARD['CMP+CBC+LIPID+A1C']) + 1e-9].sort_values('cv', ascending=False).head(1)
    f = lambda d: d.features.iloc[0].split() if len(d) and isinstance(d.features.iloc[0], str) else []  # noqa: E731
    return f(m), f(a)


lw_m, lw_a = lasso_pick('price-weighted'); lu_m, lu_a = lasso_pick('unweighted')
P = {'age+sex only': [], 'routine': ROUT, 'CMP+CBC': M.features_of(['CMP', 'CBC']), 'CBC+BMP (IMRS-type)': M.features_of(['CBC', 'BMP']),
     'PhenoAge analytes': M.PHENO_FEATS, 'matched (beam)': M.features_of(SEL['US']['matched']['items']),
     'aware@routine (beam)': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']),
     'matched (price-weighted lasso)': lw_m, 'within routine price (price-weighted lasso)': lw_a,
     'matched (unweighted lasso)': lu_m, 'within routine price (unweighted lasso)': lu_a}
rep = {'all_cause': [], 'spline': [], 'cvd': [], 'groups': [], 'price': {}, 'sequential': {}}
LP = {}
for k, f in P.items():
    LP[k] = oof(Z, M.BASE + list(f), t, e)
    rep['all_cause'].append(dict(panel=k, n_features=len(f), US=M.cost_us(f) if f else 0.0, C_oof=M.cindex(LP[k], t, e), features=' '.join(f)))
for k in ['routine', 'matched (beam)', 'aware@routine (beam)', 'PhenoAge analytes']:
    lp = oof(Z, M.BASE + list(P[k]), t, e, spline=True)
    rep['spline'].append(dict(panel=k, C_oof_linear=M.cindex(LP[k], t, e), C_oof_spline=M.cindex(lp, t, e)))
# cardiovascular death, clinical base
ok = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)].reset_index(drop=True)
tc, ec = M.outcome(x, 'cvd'); tc, ec = tc[ok], ec[ok]
saved = folds; folds = np.array_split(np.random.default_rng(11).permutation(len(tc)), 5)
CB = M.BASE + M.CLIN
C = {k: v['features'] for k, v in CVD['guideline'].items()}
C.update({'CVD matched to routine': CVD['matched_routine']['features'], 'CVD matched to PREVENT full': CVD['matched_prevent_full']['features'],
          'CVD aware@routine price': CVD['aware_at_routine']['features'], 'CVD aware@PREVENT-full price': CVD['aware_at_prevent_full']['features']})
for k, f in C.items():
    lp = oof(Zc, CB + list(f), tc, ec)
    rep['cvd'].append(dict(panel=k, US=M.cost_us(f) if f else 0.0, C_oof=M.cindex(lp, tc, ec), features=' '.join(f)))
folds = saved
# subgroups (all-cause, routine vs matched)
G = M.groups(x)
for code, lab in X.INC.items():
    G[lab] = (sesg.income_grp == code).to_numpy()
for code, lab in X.EDU.items():
    G[lab] = (sesg.edu_grp == code).to_numpy()
for g, m in G.items():
    if e[m].sum() < 30:
        continue
    r = dict(group=g, n=int(m.sum()), deaths=int(e[m].sum()))
    for k in ['routine', 'matched (beam)', 'aware@routine (beam)']:
        r[k] = M.cindex(LP[k][m], t[m], e[m])
    rep['groups'].append(r)
# price perturbation: re-optimise over the Task 19 candidate cache
cache = pd.read_csv('../task19_dm2/selected_all_cache.csv')
cache = {tuple(str(f).split()) if isinstance(f, str) else (): v for f, v in zip(cache.features, cache.cv)}
rng = np.random.default_rng(20261107); same_m = same_a = cheap = 0; picks = []
base_m = tuple(sorted(P['matched (beam)'])); base_a = tuple(sorted(P['aware@routine (beam)']))
thr = cache[tuple(sorted(ROUT))]
for _ in range(500):
    pf = X.make_cost_us({c: rng.uniform(0.7, 1.3) for c in X.COMPONENTS})
    m = X.cheapest_matching(cache, thr, pf); a = X.best_within(cache, pf(ROUT), pf)
    same_m += tuple(sorted(m)) == base_m; same_a += tuple(sorted(a)) == base_a; cheap += pf(m) <= 0.8 * pf(ROUT)
    picks.append(' '.join(M.describe(m)))
rep['price'] = dict(draws=500, matched_unchanged=same_m / 500, aware_unchanged=same_a / 500, matched_cost_le_80pct=cheap / 500,
                    matched_choices=pd.Series(picks).value_counts(normalize=True).head(5).round(3).to_dict(),
                    AU_coned=dict(routine=X.cost_au_coned(ROUT), matched_AU=X.cost_au_coned(M.features_of(SEL['AU']['matched']['items']))))
# sequential policy on cross-fitted training risks (tau 0.10, eps 0.02)
okc = ok
Zs = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(okc)].reset_index(drop=True); ts, es = t[okc], e[okc]
fo = np.array_split(np.random.default_rng(1).permutation(len(ts)), 5)
steps = [[], M.features_of(['CMP']), M.features_of(['CMP', 'CBC']), M.features_of(['CMP', 'CBC', 'acr'])]

rcv = np.zeros((4, len(ts)))
for te in fo:
    tr = np.setdiff1d(np.arange(len(ts)), te)
    for s, f in enumerate(steps):
        Xa = Zs[CB + f].to_numpy(); rcv[s, te] = M.Cox(1.0).fit(Xa[tr], ts[tr], es[tr]).risk(Xa[te])
fl = [M.L.FlipModel(0.10).fit(rcv[s], rcv[-1]) for s in range(3)]
st, risk, _ = M.L.sequential_policy(rcv, fl, 0.02, [0, 19.9, 27.67, 38.63])
cst = np.array([0, 19.9, 27.67, 38.63])[st]
rep['sequential'] = dict(n=int(len(ts)), share_cmpcbc=float(cst.mean() / 27.67), agree_final=float(((risk >= .1) == (rcv[-1] >= .1)).mean()),
                         no_lab=float((st == 0).mean()), note='cross-fitted training risks (optimistic: flip models fitted on the same risks)')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
for k in ['all_cause', 'spline', 'cvd', 'groups']:
    print(f'\n== {k}'); print(pd.DataFrame(rep[k]).round(4).to_string())
print('\n== price', json.dumps(rep['price'], indent=1)); print('== sequential', json.dumps(rep['sequential']))
json.dump(rep, open('dev_train_report.json', 'w'), indent=1, default=str)
