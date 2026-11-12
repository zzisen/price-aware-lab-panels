#!/usr/bin/env python3
"""Task 26 feasibility (post hoc; on validation cohorts that were already analysed once: NHANES III, 2011-14, 2015-16, CHNS, CLHLS; Understanding
Society is NOT touched). Question: can calibration be transported WITHOUT local outcome data, by re-centring the laboratory part of the model on
the local laboratory distribution?
  Model: lp = age/sex terms + sum_j b_j (z_j - m_j) / s_j  (ridge Cox fitted on NHANES 1999-2010; z = log analyte).
  C1 simple re-centring:        shift = sum_j b_j / s_j * (mean_local z_j - m_j);  lp' = lp - shift.
  C2 age-sex-adjusted (primary, declared before results): expected z_j given age and sex from a training regression
        (z_j ~ 1 + age + female + age x female); shift = sum_j b_j / s_j * mean_local(z_j - zhat_j(age, sex)).
  Both use only the local laboratory values, age and sex of the people being scored; a shift is one constant per cohort, so ranking (C) is unchanged.
Calibration-in-the-large with censoring handled: O/E_KM (Kaplan-Meier risk at the horizon / mean predicted risk) and O/E_Poisson (deaths / sum of
predicted cumulative hazard to min(t, horizon)); CHNS (binary, about 6 years): deaths / sum of predicted 6-year risk among people with known status.
Output: dev_recenter.csv, dev_recenter.txt"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
for d in ['task21_final', 'task25_newdata']:
    sys.path.insert(0, str(W / d))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
import clhls_lib as C  # noqa: E402
M.L.low_priority()
SEL = json.load(open(W / 'task19_dm2' / 'selected_all.json')); SCN = json.load(open(W / 'task20_chns' / 'selected_cn.json'))
SCL = json.load(open(W / 'task25_newdata' / 'selected_clhls.json'))
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr)


def km(tt, ee, h):
    o = np.argsort(tt); tt, ee = tt[o], ee[o]; s = 1.0
    for u in np.unique(tt[ee & (tt <= h)]):
        s *= 1 - (ee & (tt == u)).sum() / (tt >= u).sum()
    return 1 - s


def zhat_model(Ztr, labs):
    A = np.column_stack([np.ones(len(Ztr)), Ztr.age, Ztr.female, Ztr.age * Ztr.female])
    coef = np.linalg.lstsq(A, Ztr[labs].to_numpy(), rcond=None)[0]
    return lambda Z: np.column_stack([np.ones(len(Z)), Z.age, Z.female, Z.age * Z.female]) @ coef


def shifts(m, cols, Ztr, Z):
    labs = [c for c in cols if c not in M.BASE]
    if not labs:
        return 0.0, 0.0
    j = [cols.index(c) for c in labs]; w = m.b[j] / m.s[j]
    c1 = float(w @ (Z[labs].to_numpy().mean(0) - m.m[j]))
    zh = zhat_model(Ztr, labs); c2 = float(w @ (Z[labs].to_numpy() - zh(Z)).mean(0) - w @ (Ztr[labs].to_numpy() - zh(Ztr)).mean(0))
    return c1, c2


rows = []


def survival_cohort(name, Ztr, Z, t, e, H, panels):
    for pname, f in panels.items():
        cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr); lp = m.lp(Z[cols].to_numpy())
        grid = np.arange(1, H + 1); H0 = np.array([T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, g) for g in grid])
        c1, c2 = shifts(m, cols, Ztr, Z); obs = km(t, e, H)
        for meth, sh in [('original', 0.0), ('C1 simple', c1), ('C2 age-sex adjusted', c2)]:
            l2 = lp - sh; r = 1 - np.exp(-H0[-1] * np.exp(l2)); Hi = np.interp(np.minimum(t, H), np.r_[0, grid], np.r_[0, H0]) * np.exp(l2)
            rows.append(dict(cohort=name, panel=pname, method=meth, n=len(t), deaths=int(e.sum()), shift=sh, OE_KM=obs / r.mean(), OE_Poisson=e.sum() / Hi.sum()))


# US cohorts
US = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'matched': M.features_of(SEL['US']['matched']['items']),
      'aware@routine': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']), 'age+sex only': []}
Ztr_us = M.transform(xtr)
for name, H, loader in [('NHANES III', 108, lambda: M.load_nh3()[0]), ('NHANES 2011-14', 60, lambda: pd.concat([T.load_gh(c)[0] for c in 'GH'])),
                        ('NHANES 2015-16', 36, lambda: X.load_i()[0])]:
    x = loader(); Z = M.transform(x, ref=xtr); t, e = T.outcome_at(x, H)
    survival_cohort(name, Ztr_us, Z, t, e, H, US); print(name, 'done', flush=True)
# CHNS (binary)
xc, _ = T.load_chns(); Ztr_cn = T.transform_sub(xtr, xtr, T.SHARED); Zc = T.transform_sub(xc, xtr, T.SHARED); y = xc.died.to_numpy().astype(bool)
for pname, f in {'routine': T.CN_ROUTINE_SHARED, 'matched': SCN['matched']['features'], 'aware@routine': SCN['aware_at_routine']['features'], 'age+sex only': []}.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr_cn[cols].to_numpy(), ttr, etr); lp = m.lp(Zc[cols].to_numpy()); H6 = T.h0_at(m, Ztr_cn[cols].to_numpy(), ttr, etr, 72)
    c1, c2 = shifts(m, cols, Ztr_cn, Zc)
    for meth, sh in [('original', 0.0), ('C1 simple', c1), ('C2 age-sex adjusted', c2)]:
        r = 1 - np.exp(-H6 * np.exp(lp - sh)); rows.append(dict(cohort='CHNS', panel=pname, method=meth, n=len(y), deaths=int(y.sum()), shift=sh, OE_KM=y.sum() / r.sum(), OE_Poisson=np.nan))
print('CHNS done', flush=True)
# CLHLS
xl, _ = C.load_clhls(); Ztr_cl = T.transform_sub(xtr, xtr, C.MENU); Zl = T.transform_sub(xl, xtr, C.MENU); tl, el = C.outcome(xl)
CL = {'routine': C.ROUTINE_CL, 'matched': SCL['matched']['features'], 'aware@routine': SCL['aware_at_routine']['features'], 'age+sex only': []}
for g in ['65-79', '80+']:
    k = (xl.age_group == g).to_numpy(); survival_cohort(f'CLHLS {g}', Ztr_cl, Zl[k], tl[k], el[k], 60, CL)
D = pd.DataFrame(rows); D.to_csv(HERE / 'dev_recenter.csv', index=False)
P = D.pivot_table(index=['cohort', 'panel'], columns='method', values='OE_KM', sort=False).round(3)
Q = D.pivot_table(index=['cohort', 'panel'], columns='method', values='OE_Poisson', sort=False).round(3)
Dm = D[D.panel != 'age+sex only'].copy(); Dm['abs_log'] = np.abs(np.log(Dm.OE_KM))
summ = Dm.groupby('method').abs_log.agg(['mean', 'max']).round(3)
txt = 'O/E (KM; CHNS: known-status binary)\n' + P.to_string() + '\n\nO/E (Poisson)\n' + Q.to_string() + '\n\nmean / max |log O/E| over lab panels and cohorts\n' + summ.to_string()
(HERE / 'dev_recenter.txt').write_text(txt, encoding='utf-8'); print(txt)
