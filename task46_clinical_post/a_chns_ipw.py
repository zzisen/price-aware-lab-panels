#!/usr/bin/env python3
"""Task 46 (a) -- ADDITIONAL (POST HOC) ANALYSIS, added 2026-11-27 after all planned results were known.
CHNS: inverse-probability weighting for unknown vital status by the 2015 survey, replicating the Understanding Society approach in
task29_revision/t29_censor.py (section A3): logistic model (sklearn, C = 1, standardised covariates) for P(known status) fitted in all
age-eligible participants with complete laboratory data; known-status participants weighted by 1 / P(known), weights clipped at the 1st and
99th percentiles among known-status participants; weighted AUC (t29lib.wauc), weighted O/E and weighted net benefit (72-month risk, 6%);
paired bootstrap over the known-status participants with the weights held fixed, 1000 resamples (as in the UK analysis).
Covariates for everyone: age, sex, province, urban/rural (2009 survey), the 15 panel analytes (log scale, as in the models). Survey wave is
constant in CHNS (all blood in 2009), so it cannot enter. Nobody is dropped: the unweighted analysis uses the same 4542 people as Task 21.
Step 0 reproduces the published unweighted CHNS numbers exactly (task21_final/t21_CHNS_endpoints.json, task29_revision/valid_CHNS.json).
Existing files are only read. Output: a_chns_ipw.json (this folder)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task29_revision'))
import t29lib as Q  # noqa: E402
M, T, L = Q.M, Q.T, Q.L
from sklearn.linear_model import LogisticRegression  # noqa: E402
L.low_priority()
out = dict(label='ADDITIONAL (POST HOC) ANALYSIS - not in any frozen plan', date='2026-11-27')

# ---------------- data ----------------
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr)
xh, cnt = T.load_chns()                      # published known-status complete-case sample (4542)


def load_chns_all():
    """t20lib.load_chns without the known-status filter (same code otherwise); adds 'known', 'province', 'urban'."""
    b = pd.read_sas(T.CHNS / 'biomarker_09.sas7bdat', format='sas7bdat'); b.columns = [c.upper() for c in b.columns]; b = b.set_index('IDIND')
    s = pd.read_sas(T.CHNS / 'surveys_pub_12.sas7bdat', format='sas7bdat'); s.columns = [c.lower() for c in s.columns]
    m = pd.read_sas(T.CHNS / 'mast_pub_12.sas7bdat', format='sas7bdat'); m.columns = [c.upper() for c in m.columns]; m = m.set_index('IDIND')
    s09 = s[s.wave == 2009].groupby('idind'); age09 = s09.age.first(); seen15 = set(s[s.wave == 2015].idind)
    x = pd.DataFrame(index=b.index)
    x['age'] = age09.reindex(b.index); x['sex'] = m.GENDER.reindex(b.index)
    x['glucose'] = b.GLUCOSE_MG; x['bun'] = 2.8 * b.UREA; x['creatinine'] = b.CRE_MG; x['uric_acid'] = b.UA_MG
    x['albumin'] = b.ALB / 10; x['protein'] = b.TP / 10; x['alt'] = b.ALT
    x['tc'] = b.TC_MG; x['tg'] = b.TG_MG; x['hdl'] = b.HDL_C_MG; x['hba1c'] = b.HBA1C
    x['crp'] = b.HS_CRP / 10; x['wbc'] = b.WBC; x['hgb'] = b.HGB / 10; x['plt'] = b.PLT
    x['province'] = s09.t1.first().reindex(b.index); x['urban'] = s09.t2.first().reindex(b.index)
    x = x[(x.age >= L.AGE[0]) & (x.age <= L.AGE[1])]; n_age = len(x)
    dod_y, dod_rpt = m.DOD_Y.reindex(x.index), m.DOD_RPT.reindex(x.index)
    died = ((dod_rpt >= 2011) & (dod_rpt <= 2015)) | ((dod_y >= 2009) & (dod_y <= 2015))
    alive = (~died) & x.index.isin(list(seen15))
    x['died'] = died.astype(int); x['known'] = (died | alive).to_numpy()
    n_age_known = int(x.known.sum())
    x = x.dropna(subset=T.SHARED + ['age', 'sex'])
    x['crp'] = np.where(x.crp < M.CRP_CUT, M.CRP_FLOOR, x.crp); x['cycle'] = 'CHNS'
    return x, dict(age_40_79=n_age, age_40_79_known=n_age_known)


xa, ca = load_chns_all(); kn = xa.known.to_numpy(bool)
assert set(xa.index[kn]) == set(xh.index) and kn.sum() == len(xh) == 4542, 'known-status complete sample differs from Task 21'
xa = pd.concat([xa.loc[xh.index], xa[~kn]])           # known people first, in the published order
kn = xa.known.to_numpy(bool); y = xa.died.to_numpy().astype(bool)
out['sample'] = dict(**ca, complete_labs=int(len(xa)), complete_labs_known=int(kn.sum()), complete_labs_unknown=int((~kn).sum()),
                     unknown_share_complete=float(1 - kn.mean()), deaths_known=int(y[kn].sum()),
                     province_missing=int(xa.province.isna().sum()), urban_missing=int(xa.urban.isna().sum()),
                     known_share_by_urban={str(int(k)): float(v) for k, v in xa.groupby('urban').known.mean().items()},
                     known_share_by_province={str(int(k)): float(v) for k, v in xa.groupby('province').known.mean().items()},
                     known_share_by_age={'40-59': float(kn[xa.age.to_numpy() < 60].mean()), '60-79': float(kn[xa.age.to_numpy() >= 60].mean())},
                     known_share_by_sex={'male': float(kn[xa.sex.to_numpy() == 1].mean()), 'female': float(kn[xa.sex.to_numpy() == 2].mean())})
print('sample', json.dumps(out['sample']), flush=True)

# ---------------- models (identical to run_t21_chns.py / t29_valid.py) ----------------
SEL = json.load(open(W / 'task20_chns' / 'selected_cn.json'))
RT = 'routine (shared items)'
P = {'age+sex only': [], RT: T.CN_ROUTINE_SHARED, 'matched': SEL['matched']['features'], 'aware@routine': SEL['aware_at_routine']['features']}
Ztr = T.transform_sub(xtr, xtr, T.SHARED); Za = T.transform_sub(xa, xtr, T.SHARED)
LP, R72 = {}, {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    LP[k] = m.lp(Za[cols].to_numpy()); R72[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, 72), Za[cols].to_numpy())

# ---------------- step 0: reproduce the unweighted numbers ----------------
S = {k: v[kn] for k, v in LP.items()}; yk = y[kn]; n = len(yk)
pairs = [('matched', RT), ('aware@routine', RT), (RT, 'age+sex only')]
rng = np.random.default_rng(20261107)                 # run_t21_chns.py seed and loop
B = {f'dAUC {a} - {b}': [] for a, b in pairs}
for _ in range(1000):
    i = rng.integers(0, n, n)
    if yk[i].all() or (~yk[i]).all():
        continue
    A = {k: T.auc(S[k][i], yk[i]) for k in set(x_ for p in pairs for x_ in p)}
    for a, b in pairs:
        B[f'dAUC {a} - {b}'].append(A[a] - A[b])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
rep = dict(AUC={k: T.auc(S[k], yk) for k in P}, NB_72_tau6={k: Q.nb_ipcw(R72[k][kn], np.ones(n), yk, .06) for k in P},
           OE_72={k: float(yk.sum() / R72[k][kn].sum()) for k in P},
           point={f'dAUC {a} - {b}': T.auc(S[a], yk) - T.auc(S[b], yk) for a, b in pairs}, boot={k: ci(v) for k, v in B.items()})
t21 = json.load(open(W / 'task21_final' / 't21_CHNS_endpoints.json')); v29 = json.load(open(W / 'task29_revision' / 'valid_CHNS.json'))
chk = {}
for k in rep['point']:
    chk[k + ' point'] = abs(rep['point'][k] - t21['point'][k]) < 1e-12
    chk[k + ' CI'] = bool(np.allclose(rep['boot'][k], t21['boot'][k], atol=1e-12))
for k29, k in [('age+sex only', 'age+sex only'), ('routine', RT), ('matched', 'matched'), ('aware@routine', 'aware@routine')]:
    chk[f'AUC {k29}'] = abs(rep['AUC'][k] - v29['panels'][k29]['disc']) < 1e-12
    chk[f'NB {k29}'] = abs(rep['NB_72_tau6'][k] - v29['panels'][k29]['NB']) < 1e-12
rep['matches_published'] = chk; rep['all_match'] = bool(all(chk.values()))
out['reproduce_unweighted'] = rep; print('reproduce', json.dumps(rep), flush=True)
assert rep['all_match'], chk

# ---------------- known-status model and weights ----------------
prov = pd.get_dummies(xa.province.astype(int), prefix='prov', drop_first=True).astype(float)


def ps_design(with_labs=True, with_region=True):
    parts = [Za[M.BASE].to_numpy(float)]
    if with_labs:
        parts.append(Za[T.SHARED].to_numpy(float))
    if with_region:
        parts += [prov.to_numpy(), (xa.urban.to_numpy() == 1).astype(float)[:, None]]
    F = np.column_stack(parts); return (F - F.mean(0)) / F.std(0)


def weights(F, clip=True):
    pk = LogisticRegression(C=1.0, max_iter=5000).fit(F, kn).predict_proba(F)[:, 1]
    w = np.where(kn, 1 / pk, 0.0)
    if clip:
        lo_, hi_ = np.quantile(w[kn], [0.01, 0.99]); w[kn] = np.clip(w[kn], lo_, hi_)
    return pk, w


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


F = ps_design(); pk, ipw = weights(F)
c_known = T.auc(pk, kn)
bal = {}
for nm, col in [('age', Za.age), ('female', Za.female), ('urban', pd.Series((xa.urban == 1).astype(float), index=xa.index))] + [(a, Za[a]) for a in T.SHARED]:
    v = col.to_numpy(float); sd = v.std(); wk_ = ipw[kn]
    bal[nm] = dict(smd_known_vs_all_unweighted=float((v[kn].mean() - v.mean()) / sd),
                   smd_known_vs_all_weighted=float((np.average(v[kn], weights=wk_) - v.mean()) / sd))
out['known_model'] = dict(covariates='age, female, 15 log analytes, province dummies, urban (standardised); logistic C=1',
                          AUC_known_vs_unknown=c_known, known_prob=dict(min=float(pk.min()), median=float(np.median(pk)), max=float(pk.max())),
                          weights_known=dict(min=float(ipw[kn].min()), median=float(np.median(ipw[kn])), max=float(ipw[kn].max()),
                                             sum=float(ipw[kn].sum()), ESS=ess(ipw[kn])), balance=bal)
print('known model', json.dumps(out['known_model']), flush=True)


def weighted_block(w_all, tag, nboot=1000, seed=20261127):
    wk = w_all[kn]; Rk = {k: v[kn] for k, v in R72.items()}
    res = dict(panels={k: dict(AUC_ipw=Q.wauc(S[k], yk, wk), OE_72_ipw=float((wk * yk).sum() / (wk * Rk[k]).sum()),
                               NB_72_tau6_ipw=Q.nb_ipcw(Rk[k], wk / wk.mean(), yk, .06)) for k in P},
               weighted_death_risk=float((wk * yk).sum() / wk.sum()))
    keys = [f'dAUC {a} - {b}' for a, b in pairs] + [f'dNB {a} - {b}' for a, b in pairs[:2]]
    bs = {k: [] for k in keys}; r_ = np.random.default_rng(seed)
    for _ in range(nboot):
        j = r_.integers(0, n, n); yj, wj = yk[j], wk[j]
        if yj.all() or (~yj).all():
            continue
        A = {k: Q.wauc(S[k][j], yj, wj) for k in P}; N = {k: Q.nb_ipcw(Rk[k][j], wj / wj.mean(), yj, .06) for k in [RT, 'matched', 'aware@routine']}
        for a, b in pairs:
            bs[f'dAUC {a} - {b}'].append(A[a] - A[b])
        for a, b in pairs[:2]:
            bs[f'dNB {a} - {b}'].append(N[a] - N[b])
    pp = res['panels']
    res['contrasts'] = {}
    for a, b in pairs:
        res['contrasts'][f'dAUC {a} - {b}'] = dict(point=pp[a]['AUC_ipw'] - pp[b]['AUC_ipw'], ci=Q.ci(bs[f'dAUC {a} - {b}']))
    for a, b in pairs[:2]:
        res['contrasts'][f'dNB {a} - {b}'] = dict(point=pp[a]['NB_72_tau6_ipw'] - pp[b]['NB_72_tau6_ipw'], ci=Q.ci(bs[f'dNB {a} - {b}']))
    print(tag, json.dumps(res), flush=True)
    return res


out['ipw_main'] = weighted_block(ipw, 'IPW main')
# unweighted contrasts with the same bootstrap draws (seed 20261127) for a like-for-like comparison
out['unweighted_same_draws'] = weighted_block(np.where(kn, 1.0, 0.0), 'unweighted same draws')
# sensitivity variants of the weight model
_, w_nolab = weights(ps_design(with_labs=False)); out['ipw_no_labs'] = weighted_block(w_nolab, 'IPW without analytes', nboot=1000, seed=20261128)
_, w_noclip = weights(F, clip=False); out['ipw_unclipped'] = weighted_block(w_noclip, 'IPW unclipped', nboot=1000, seed=20261129)
out['ipw_no_labs']['ESS'] = ess(w_nolab[kn]); out['ipw_unclipped']['ESS'] = ess(w_noclip[kn]); out['ipw_unclipped']['max_w'] = float(w_noclip[kn].max())
json.dump(out, open(HERE / 'a_chns_ipw.json', 'w'), indent=1, default=float)
print('done')
