#!/usr/bin/env python3
"""Task 21 unified final analysis, one validation cohort per call (plan_t21_FROZEN). Usage: run_t21.py <NH3|GH|I> [--dry]
Training: NHANES 1999-2010 (A-F). All panels were selected on training data before freezing:
  all-cause, US prices: ../task19_dm2/selected_all.json; lasso comparators: lasso_candidates.csv; cardiovascular: selected_cvd.json.
--dry: train on C, test on D (development check only)."""
import json
import sys
import numpy as np
import pandas as pd
import dm2lib as M
import t20lib as T
import t21lib as X
M.L.low_priority()
COH = sys.argv[1]; DRY = '--dry' in sys.argv; PRE = f"{'dry_' if DRY else ''}t21_{COH}_"
NBOOT, NSUB = (200, 100) if DRY else (1000, 200)
H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]
TAU_H = {108: 0.10, 60: 0.05, 36: 0.03}[H]           # decision threshold on the cohort-horizon risk scale
SEL = json.load(open('../task19_dm2/selected_all.json')); CVD = json.load(open('selected_cvd.json')); LC = pd.read_csv('lasso_candidates.csv')
ROUT = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']); BASE, CB = M.BASE, M.BASE + M.CLIN

# ---------------- data ----------------
if DRY:
    xtr, ctr = M.load(['C']); xte, cte = M.load(['D'])
else:
    xtr, ctr = M.load(list('ABCDEF'))
    if COH == 'NH3':
        xte, cte = M.load_nh3()
    elif COH == 'GH':
        parts = [T.load_gh(c) for c in 'GH']; xte = pd.concat([p for p, _ in parts]); cte = [n for _, n in parts]
    else:
        xte, cte = X.load_i()
ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr)
t, e = T.outcome_at(xte, H); Z = M.transform(xte, ref=xtr); n = len(t); w = xte.weight.to_numpy(float)
has_crp = bool(xte.crp.notna().all())
known = e | (t >= H)


# ---------------- panels ----------------
def lasso_pick(kind, rule):
    rt_cv = float(json.load(open('../task19_dm2/selected_all.json'))['US']['standard']['CMP+CBC+LIPID+A1C']['cv'])
    s = LC[LC.kind == kind]
    d = s[s.cv >= rt_cv].sort_values(['US', 'cv'], ascending=[True, False]) if rule == 'matched' else s[s.US <= M.cost(M.STANDARD['CMP+CBC+LIPID+A1C']) + 1e-9].sort_values('cv', ascending=False)
    return d.features.iloc[0].split() if len(d) and isinstance(d.features.iloc[0], str) else []


P = {'age+sex only': [], 'BMP': M.features_of(['BMP']), 'CBC': M.features_of(['CBC']), 'CMP': M.features_of(['CMP']), 'LIPID': M.features_of(['LIPID']),
     'CMP+CBC': M.features_of(['CMP', 'CBC']), 'CBC+BMP (IMRS-type)': M.features_of(['CBC', 'BMP']), 'CMP+CBC+LIPID': M.features_of(['CMP', 'CBC', 'LIPID']),
     'routine': ROUT, 'matched': M.features_of(SEL['US']['matched']['items']), 'aware@routine': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']),
     'lasso-pw matched': lasso_pick('price-weighted', 'matched'), 'lasso-pw within routine': lasso_pick('price-weighted', 'within'),
     'lasso-uw matched': lasso_pick('unweighted', 'matched'), 'lasso-uw within routine': lasso_pick('unweighted', 'within'),
     'PhenoAge analytes': M.PHENO_FEATS, 'all analytes': M.FEATURES}
P = {k: v for k, v in P.items() if has_crp or 'crp' not in v}
LP, RH, MOD = {}, {}, {}
for k, f in P.items():
    cols = BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr); MOD[k] = (m, cols)
    LP[k] = m.lp(Z[cols].to_numpy()); RH[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H), Z[cols].to_numpy())
if has_crp:
    xb_tr, xb = M.phenoage_xb(xtr), M.phenoage_xb(xte); mp = M.Cox(1e-6).fit(xb_tr[:, None], ttr, etr)
    LP['PhenoAge published'] = xb; RH['PhenoAge published'] = T.risk_at(mp, T.h0_at(mp, xb_tr[:, None], ttr, etr, H), xb[:, None])


def calib(r, tt, ee):
    kn = ee | (tt >= H); rr, yy = np.clip(r[kn], 1e-4, 1 - 1e-4), ee[kn].astype(float)
    u = np.log(rr / (1 - rr)); A = np.column_stack([np.ones(len(u)), u]); b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ b)); W = p * (1 - p) + 1e-9; st = np.linalg.solve(A.T @ (W[:, None] * A), A.T @ (yy - p)); b += st
        if np.abs(st).max() < 1e-8:
            break
    return float(yy.sum() / rr.sum()), float(b[1])


def nb(r, tt, ee, tau, ww=None):
    kn = ee | (tt >= H); ww = np.ones(len(tt)) if ww is None else ww; W = ww[kn].sum(); hi = r >= tau
    return float((ww * (hi & ee & kn)).sum() / W - (ww * (hi & ~ee & kn)).sum() / W * tau / (1 - tau))


rows = []
for k in LP:
    oe, sl = calib(RH[k], t, e)
    rows.append(dict(panel=k, features=' '.join(P.get(k, M.PHENO_FEATS)), US=(M.cost_us(P[k]) if P.get(k) else (M.cost_us(M.PHENO_FEATS) if k == 'PhenoAge published' else 0.0)),
                     C=M.cindex(LP[k], t, e), C_w=M.cindex(LP[k], t, e, w), OE=oe, slope=sl, NB=nb(RH[k], t, e, TAU_H), NB_w=nb(RH[k], t, e, TAU_H, w),
                     NB_half=nb(RH[k], t, e, TAU_H / 2), NB_double=nb(RH[k], t, e, TAU_H * 2)))
panels = pd.DataFrame(rows)

# ---------------- bootstrap of primary and key secondary contrasts ----------------
pairs = [('matched', 'routine'), ('aware@routine', 'routine'), ('matched', 'CBC+BMP (IMRS-type)'), ('matched', 'CMP+CBC'),
         ('lasso-pw matched', 'routine'), ('lasso-pw within routine', 'routine'), ('aware@routine', 'lasso-pw within routine'),
         ('lasso-uw within routine', 'routine'), ('CBC', 'BMP')]
if has_crp:
    pairs += [('matched', 'PhenoAge analytes'), ('matched', 'PhenoAge published')]
pairs = [p for p in pairs if p[0] in LP and p[1] in LP]
rng = np.random.default_rng(20261107); B = {f'dC {a} - {b}': [] for a, b in pairs}
B.update({'dNB matched - routine': [], 'dNB aware@routine - routine': []})
need = sorted(set(k for p in pairs for k in p))
for _ in range(NBOOT):
    i = rng.integers(0, n, n); Cb = {k: M.cindex(LP[k][i], t[i], e[i]) for k in need}
    for a, b in pairs:
        B[f'dC {a} - {b}'].append(Cb[a] - Cb[b])
    nr = nb(RH['routine'][i], t[i], e[i], TAU_H)
    B['dNB matched - routine'].append(nb(RH['matched'][i], t[i], e[i], TAU_H) - nr); B['dNB aware@routine - routine'].append(nb(RH['aware@routine'][i], t[i], e[i], TAU_H) - nr)
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
boot = {k: ci(v) for k, v in B.items()}
pc = panels.set_index('panel')
point = {f'dC {a} - {b}': float(pc.C[a] - pc.C[b]) for a, b in pairs}
point['dNB matched - routine'] = float(pc.NB['matched'] - pc.NB['routine']); point['dNB aware@routine - routine'] = float(pc.NB['aware@routine'] - pc.NB['routine'])

# ---------------- model-class sensitivity: restricted cubic spline Cox ----------------
spl = {}
for k in ['routine', 'matched', 'aware@routine']:
    cols = BASE + list(P[k]); A_, B_ = X.spline_design(Ztr, Z, cols)
    spl[k] = M.Cox(1.0).fit(A_, ttr, etr).lp(B_)
sp = {'C routine': M.cindex(spl['routine'], t, e), 'C matched': M.cindex(spl['matched'], t, e), 'C aware@routine': M.cindex(spl['aware@routine'], t, e)}
d1, d2 = [], []
for _ in range(NSUB):
    i = rng.integers(0, n, n); cr = M.cindex(spl['routine'][i], t[i], e[i])
    d1.append(M.cindex(spl['matched'][i], t[i], e[i]) - cr); d2.append(M.cindex(spl['aware@routine'][i], t[i], e[i]) - cr)
sp.update({'dC matched - routine': sp['C matched'] - sp['C routine'], 'dC matched CI': ci(d1), 'dC aware - routine': sp['C aware@routine'] - sp['C routine'], 'dC aware CI': ci(d2)})

# ---------------- people with complete clinical variables: sequential policy, cardiovascular death ----------------
cltr = M.clinical(xtr); oktr = cltr.notna().all(1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
cl = M.clinical(xte) if (DRY or COH == 'NH3') else (T.clinical_gh(xte) if COH == 'GH' else X.clinical_i(xte))
okc = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(okc)]; tc, ec = t[okc], e[okc]
fc = np.array_split(np.random.default_rng(1).permutation(int(oktr.sum())), 5)
Pq = M.FixedPath(SEL['sequential']['blocks'], 0.10, base=CB).fit(Ztrc, ttr[oktr], etr[oktr], fc); RR = Pq.risks(Zc)
st, risk9, cst = Pq.run(RR, 0.02)
fin9 = RR[Pq.final]
mrc = M.Cox(1.0).fit(Ztrc[CB + ROUT].to_numpy(), ttr[oktr], etr[oktr]); r9_rout = mrc.risk(Zc[CB + ROUT].to_numpy())
seq = dict(n=int(okc.sum()), mean_cost=float(cst.mean()), share_cmpcbc=float(cst.mean() / M.cost(['CMP', 'CBC'])), share_routine=float(cst.mean() / M.cost(M.STANDARD['CMP+CBC+LIPID+A1C'])),
           no_lab=float((cst == 0).mean()), agree_final=float(((risk9 >= .1) == (fin9 >= .1)).mean()), agree_routine=float(((risk9 >= .1) == (r9_rout >= .1)).mean()),
           C_seq=M.cindex(np.log(risk9), tc, ec), C_routine_clin=M.cindex(np.log(r9_rout), tc, ec))
if H == 108:
    seq['NB'] = nb(risk9, tc, ec, .1); seq['NB_routine'] = nb(r9_rout, tc, ec, .1)
sb = {'share': [], 'dNB': [], 'agree': []}
for _ in range(NBOOT):
    j = rng.integers(0, len(tc), len(tc))
    sb['share'].append(cst[j].mean() / M.cost(['CMP', 'CBC'])); sb['agree'].append(((risk9[j] >= .1) == (fin9[j] >= .1)).mean())
    if H == 108:
        sb['dNB'].append(nb(risk9[j], tc[j], ec[j], .1) - nb(r9_rout[j], tc[j], ec[j], .1))
seq['share_CI'] = ci(sb['share']); seq['agree_CI'] = ci(sb['agree'])
if H == 108:
    seq['dNB'] = seq['NB'] - seq['NB_routine']; seq['dNB_CI'] = ci(sb['dNB'])
# cardiovascular death (clinical base)
tv, ev = T.outcome_at(xte.iloc[np.flatnonzero(okc)], H)
ev &= xte.iloc[np.flatnonzero(okc)].ucod.isin(['001', '005']).to_numpy()
cv_tr_t, cv_tr_e = M.outcome(xtr, 'cvd'); cv_tr_t, cv_tr_e = cv_tr_t[oktr], cv_tr_e[oktr]
CV = {k: v['features'] for k, v in CVD['guideline'].items()}
CV.update({'CVD matched to routine': CVD['matched_routine']['features'], 'CVD matched to PREVENT full': CVD['matched_prevent_full']['features'],
           'CVD aware@routine': CVD['aware_at_routine']['features'], 'CVD aware@PREVENT-full price': CVD['aware_at_prevent_full']['features']})
CV = {k: v for k, v in CV.items() if has_crp or 'crp' not in v}
LPv = {k: M.Cox(1.0).fit(Ztrc[CB + list(f)].to_numpy(), cv_tr_t, cv_tr_e).lp(Zc[CB + list(f)].to_numpy()) for k, f in CV.items()}
cvd = {'events': int(ev.sum()), 'C': {k: M.cindex(v, tv, ev) for k, v in LPv.items()}, 'US': {k: (M.cost_us(f) if f else 0.0) for k, f in CV.items()}}
cpairs = [('CVD aware@PREVENT-full price', 'PREVENT full'), ('CVD matched to routine', 'routine'), ('CVD aware@routine', 'routine'), ('PCE/SCORE2 lipids', 'none'),
          ('CVD matched to PREVENT full', 'PREVENT full')]
cb = {f'dC {a} - {b}': [] for a, b in cpairs}
for _ in range(NBOOT):
    j = rng.integers(0, len(tv), len(tv))
    for a, b in cpairs:
        cb[f'dC {a} - {b}'].append(M.cindex(LPv[a][j], tv[j], ev[j]) - M.cindex(LPv[b][j], tv[j], ev[j]))
cvd['contrasts'] = {k: dict(point=cvd['C'][k.split(' - ')[0][3:]] - cvd['C'][k.split(' - ')[1]], ci=ci(v)) for k, v in cb.items()}

# ---------------- subgroups (sex, age, race, income, education) ----------------
G = M.groups(xte)
sg = X.ses(xte) if not DRY else X.ses(xte)
for code, lab in X.INC.items():
    G[lab] = (sg.income_grp == code).to_numpy()
for code, lab in X.EDU.items():
    G[lab] = (sg.edu_grp == code).to_numpy()
sub = []
for g, msk in G.items():
    idx = np.flatnonzero(msk); row = dict(group=g, n=len(idx), deaths=int(e[idx].sum()))
    if e[idx].sum() < 20:
        sub.append(dict(row, note='fewer than 20 deaths')); continue
    for k in ['routine', 'matched', 'aware@routine']:
        row['C ' + k] = M.cindex(LP[k][idx], t[idx], e[idx]); row['OE ' + k], row['slope ' + k] = calib(RH[k][idx], t[idx], e[idx])
        kn = known[idx]; hi = RH[k][idx] >= TAU_H
        row['sens ' + k] = float(hi[kn & e[idx]].mean()) if (kn & e[idx]).any() else np.nan; row['spec ' + k] = float((~hi[kn & ~e[idx]]).mean())
    dd = [M.cindex(LP['matched'][j], t[j], e[j]) - M.cindex(LP['routine'][j], t[j], e[j]) for j in (idx[rng.integers(0, len(idx), len(idx))] for _ in range(NSUB))]
    row['dC matched - routine'] = row['C matched'] - row['C routine']; row['dC CI lo'], row['dC CI hi'] = ci(dd)
    sub.append(row)
subd = pd.DataFrame(sub)

# ---------------- health economics (9-year cohort only) ----------------
he = {}
if H == 108:
    strat = {'no labs (clinical only)': [], 'routine': ROUT, 'CMP+CBC': M.features_of(['CMP', 'CBC']), 'CBC+BMP (IMRS-type)': M.features_of(['CBC', 'BMP']),
             'matched': P['matched'], 'aware@routine': P['aware@routine']}
    if has_crp:
        strat['PhenoAge analytes'] = M.PHENO_FEATS
    flags, costs = {}, {}
    for k, f in strat.items():
        r = M.Cox(1.0).fit(Ztrc[CB + list(f)].to_numpy(), ttr[oktr], etr[oktr]).risk(Zc[CB + list(f)].to_numpy())
        flags[k] = r >= .1; costs[k] = np.full(len(tc), M.cost_us(f) if f else 0.0)
    flags['sequential'] = risk9 >= .1; costs['sequential'] = cst
    died = ec & (tc <= H)
    base = {k: X.he_eval(flags[k], died, costs[k], X.HE_BASE) for k in flags}
    psa, best = X.he_psa(flags, died, costs, n=1000 if not DRY else 200)
    he = dict(base_case=base, psa_prob_best=best.to_dict(), psa_mean_nmb=psa.mean().to_dict(),
              per1000={k: dict(cost=1000 * float(costs[k].mean()), flagged=1000 * float(flags[k].mean()), deaths_flagged=1000 * float((flags[k] & died).mean()),
                               deaths_missed=1000 * float((~flags[k] & died).mean())) for k in flags},
              threshold_rrr={k: [round(r_, 3) for r_ in np.linspace(0, .15, 16) if X.he_eval(flags[k], died, costs[k], dict(X.HE_BASE, rrr=r_))['nmb']
                                 >= X.he_eval(flags['routine'], died, costs['routine'], dict(X.HE_BASE, rrr=r_))['nmb']] for k in flags})

# ---------------- multiple imputation (9-year cohort only) ----------------
mi = []
if H == 108 and not DRY:
    xm_tr, _ = M.load(list('ABCDEF'), keep_missing=True); xm, _ = M.load_nh3(keep_missing=True)
    Zm_tr, Zm = M.transform(xm_tr, ref=xm_tr), M.transform(xm, ref=xm_tr); tm_tr, em_tr = M.outcome(xm_tr); tm, em = M.outcome(xm)
    IA = M.impute(Zm_tr, tm_tr, em_tr, m=5, seed=1); IB = M.impute(Zm, tm, em, m=5, seed=11)
    for k_ in range(5):
        r_ = dict(imp=k_, n=len(tm), deaths=int(em.sum()))
        for name in ['routine', 'matched', 'aware@routine']:
            cols = BASE + list(P[name]); r_[name] = M.cindex(M.Cox(1.0).fit(IA[k_][cols].to_numpy(), tm_tr, em_tr).lp(IB[k_][cols].to_numpy()), tm, em)
        mi.append(r_)

# ---------------- output ----------------
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 60)
print('COHORT', COH, 'horizon', H, 'months; tau', TAU_H, '| TRAIN n', len(ttr), '| TEST', cte, 'n', n, 'deaths', int(e.sum()), '| CRP available', has_crp)
print(panels.round(4).to_string())
for k in boot:
    print(f'{k:60s} point {point[k]:+.4f}  95% [{boot[k][0]:+.4f}, {boot[k][1]:+.4f}]')
print('SPLINE', json.dumps(sp)); print('SEQUENTIAL', json.dumps(seq)); print('CVD', json.dumps(cvd)); print(subd.round(4).to_string())
if he:
    print('HEALTH ECONOMICS', json.dumps(he, default=float))
if mi:
    print('MI', pd.DataFrame(mi).mean(numeric_only=True).round(4).to_dict())
panels.to_csv(PRE + 'panels.csv', index=False); subd.to_csv(PRE + 'subgroups.csv', index=False)
json.dump(dict(cohort=COH, horizon=H, tau=TAU_H, train=ctr, test=cte, n=n, deaths=int(e.sum()), crp=has_crp, boot=boot, point=point, spline=sp, seq=seq, cvd=cvd,
               he=he, mi=mi), open(PRE + 'endpoints.json', 'w'), indent=1, default=float)
