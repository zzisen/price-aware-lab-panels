#!/usr/bin/env python3
"""Task 29 A and B (PLAN.md): censoring-aware net benefit and calibration slope (CLHLS), censoring before the horizon in the NHANES cohorts,
inverse-probability weighting for unknown 9-year status (Understanding Society), and aligned horizons (CHNS 72 months; Understanding Society
102 months). Published panels and models unchanged. Output: censor.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t29lib as Q  # noqa: E402
M, T, L = Q.M, Q.T, Q.L
import t21lib as X  # noqa: E402
import clhls_lib as C  # noqa: E402
import ukhls_menu as U  # noqa: E402
import ukhls_lib as K  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
M.L.low_priority()
NB_ = 1000
rng = np.random.default_rng(20261112)
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr)
out = {}


def fit(Ztr, Z, feats, H):
    cols = M.BASE + list(feats); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    return m.lp(Z[cols].to_numpy()), T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H), Z[cols].to_numpy())


def naive_nb(r, t, e, H, tau):
    kn = e | (t >= H); W_ = kn.sum(); hi = r >= tau
    return float((hi & e & kn).sum() / W_ - (hi & ~e & kn).sum() / W_ * tau / (1 - tau))


def naive_slope(r, t, e, H):
    kn = e | (t >= H); return Q.slope_w(r, e & (t <= H), kn.astype(float))


# ---------------- A1: CLHLS ----------------
S = json.load(open(Q.W / 'task25_newdata' / 'selected_clhls.json'))
xc, _ = C.load_clhls(); Ztr = T.transform_sub(xtr, xtr, C.MENU); Zc = T.transform_sub(xc, xtr, C.MENU); tc, ec = C.outcome(xc); H = C.H; TAU = 0.05
P = {'age+sex only': [], 'routine': C.ROUTINE_CL, 'matched': S['matched']['features'], 'aware@routine': S['aware_at_routine']['features']}
for tag, msk in [('CLHLS', (xc.age_group == '65-79').to_numpy()), ('CLHLS80', (xc.age_group == '80+').to_numpy())]:
    Z, t, e = Zc[msk], tc[msk], ec[msk]
    R = {k: fit(Ztr, Z, f, H)[1] for k, f in P.items()}
    w, died = Q.ipcw(t, e, H)
    res = dict(n=int(len(t)), deaths_by_H=int(died.sum()), censored_alive_before_H=float(((t < H) & ~e).mean()), km_risk=float((w * died).sum() / len(t)), panels={})
    for k, r in R.items():
        res['panels'][k] = dict(mean_risk=float(r.mean()), OE_ipcw=float((w * died).sum() / len(t) / r.mean()), NB_known_status=naive_nb(r, t, e, H, TAU),
                                NB_ipcw=Q.nb_ipcw(r, w, died, TAU), NB_km=Q.nb_km(r, t, e, H, TAU), slope_known_status=naive_slope(r, t, e, H),
                                slope_ipcw=Q.slope_w(r, died, w), NB_treat_all_ipcw=Q.nb_ipcw(np.ones(len(t)), w, died, TAU))
    bs = {'dNB matched - routine': [], 'dNB aware@routine - routine': []}
    n = len(t)
    for _ in range(NB_):
        i = rng.integers(0, n, n); wb, db = Q.ipcw(t[i], e[i], H); nr = Q.nb_ipcw(R['routine'][i], wb, db, TAU)
        bs['dNB matched - routine'].append(Q.nb_ipcw(R['matched'][i], wb, db, TAU) - nr); bs['dNB aware@routine - routine'].append(Q.nb_ipcw(R['aware@routine'][i], wb, db, TAU) - nr)
    pp = res['panels']
    res['contrasts_ipcw'] = {k: dict(point=pp[k.split(' - ')[0][4:]]['NB_ipcw'] - pp['routine']['NB_ipcw'], ci=Q.ci(v)) for k, v in bs.items()}
    out[tag] = res; print(tag, json.dumps(res, indent=1), flush=True)

# ---------------- A2: NHANES cohorts, censoring alive before the horizon ----------------
cen = {}
x3, _ = M.load_nh3(); t3 = x3.months.to_numpy(float); e3 = x3.died.to_numpy() == 1
cen['NHANES III (108 months)'] = dict(n=len(t3), share=float(((t3 < 108) & ~e3).mean()))
xg = pd.concat([T.load_gh(c)[0] for c in 'GH']); tg_ = xg.months.to_numpy(float); eg = xg.died.to_numpy() == 1
cen['NHANES 2011-14 (60 months)'] = dict(n=len(tg_), share=float(((tg_ < 60) & ~eg).mean()), min_followup_alive=float(tg_[~eg].min()))
xi, _ = X.load_i(); ti = xi.months.to_numpy(float); ei = xi.died.to_numpy() == 1
cen['NHANES 2015-16 (36 months)'] = dict(n=len(ti), share=float(((ti < 36) & ~ei).mean()))
out['nhanes_censoring'] = cen; print(json.dumps(cen), flush=True)

# ---------------- A3 and B2: Understanding Society ----------------
SU = json.load(open(Q.W / 'task27_ukhls' / 'selected_ukhls.json')); U.install()


def load_ukhls_all():
    """ukhls_lib.load_ukhls without the known-status filter (same code otherwise); adds 'known'."""
    import pyreadstat
    rd = lambda p, cols: pyreadstat.read_dta(str(p), usecols=cols)[0]  # noqa: E731
    cols = ['pidp', 'wave'] + [c for c, _ in U.UNITS.values()] + ['hba1c', 'indbdub_xw', 'dory']
    b = rd(K.NS / 'xlabblood_ns.dta', cols)
    b = b.sort_values('wave', kind='stable').drop_duplicates('pidp', keep='first').set_index('pidp')
    xw = rd(K.MS / 'xwaveid.dta', ['pidp', 'sex', 'birthy', 'dcsedfl_dv', 'dcsedw_dv', 'lwenum_dv']).set_index('pidp')
    age = pd.Series(np.nan, index=b.index); sex = pd.Series(np.nan, index=b.index)
    for w_, l in K.WL.items():
        ids = b.index[b.wave == w_]
        r = rd(K.NS / f'{l}_indresp_ns.dta', ['pidp', f'{l}_confage', f'{l}_nsex']).drop_duplicates('pidp').set_index('pidp')
        s = rd(K.NS / f'{l}_indsamp_ns.dta', ['pidp', f'{l}_age', f'{l}_nsex']).drop_duplicates('pidp').set_index('pidp')
        a1 = r[f'{l}_confage'].reindex(ids).where(lambda v: v >= 0); a2 = s[f'{l}_age'].reindex(ids).where(lambda v: v >= 0)
        a3 = (b.dory.reindex(ids).where(lambda v: v > 0) - xw.birthy.reindex(ids).where(lambda v: v > 0))
        age[ids] = a1.fillna(a2).fillna(a3)
        s1 = r[f'{l}_nsex'].reindex(ids).where(lambda v: v.isin([1, 2])); s2 = s[f'{l}_nsex'].reindex(ids).where(lambda v: v.isin([1, 2]))
        sex[ids] = s1.fillna(s2).fillna(xw.sex.reindex(ids).where(lambda v: v.isin([1, 2])))
    x = K.harmonise(b); x['age'] = age; x['sex'] = sex; x['wave'] = b.wave
    x = x[(x.age >= M.L.AGE[0]) & (x.age <= M.L.AGE[1])]
    o = xw.reindex(x.index)
    died, known = K.outcome_from(x.wave, o.dcsedfl_dv, o.dcsedw_dv, o.lwenum_dv)
    x['died'] = died.astype(int); x['known'] = known; x['cycle'] = 'UKHLS'
    return x.dropna(subset=U.MENU + ['age', 'sex'])


xu = load_ukhls_all(); kn = xu.known.to_numpy(bool)
Ztru = T.transform_sub(xtr, xtr, U.MENU); Zu = T.transform_sub(xu, xtr, U.MENU)
F = np.column_stack([Zu[M.BASE + U.MENU].to_numpy(float), (xu.wave == 3).to_numpy(float)])
F = (F - F.mean(0)) / F.std(0)
pk = LogisticRegression(C=1.0, max_iter=5000).fit(F, kn).predict_proba(F)[:, 1]
ipw = np.where(kn, 1 / pk, 0.0); lo_, hi_ = np.quantile(ipw[kn], [0.01, 0.99]); ipw[kn] = np.clip(ipw[kn], lo_, hi_)
RT = 'routine (shared items)'
PU = {'age+sex only': [], RT: SU['routine']['features'], 'matched': SU['matched']['features'], 'aware@routine': SU['aware_at_routine']['features']}
y = xu.died.to_numpy().astype(bool)
LPu, R108, R102 = {}, {}, {}
for k, f in PU.items():
    LPu[k], R108[k] = fit(Ztru, Zu, f, 108); R102[k] = fit(Ztru, Zu, f, 102)[1]
uk = dict(n_complete=int(len(xu)), n_known=int(kn.sum()), unknown_share=float(1 - kn.mean()), deaths_known=int(y[kn].sum()),
          known_prob=dict(min=float(pk.min()), median=float(np.median(pk)), max=float(pk.max())), panels={})
yk, wk = y[kn], ipw[kn]
for k in PU:
    uk['panels'][k] = dict(AUC_known=T.auc(LPu[k][kn], yk), AUC_ipw=Q.wauc(LPu[k][kn], yk, wk),
                           OE_108_known=float(yk.sum() / R108[k][kn].sum()), OE_108_ipw=float((wk * yk).sum() / (wk * R108[k][kn]).sum()),
                           OE_102_known=float(yk.sum() / R102[k][kn].sum()), slope_108=Q.slope_w(R108[k][kn], yk, np.ones(kn.sum())),
                           slope_102=Q.slope_w(R102[k][kn], yk, np.ones(kn.sum())),
                           NB_108_known=Q.nb_ipcw(R108[k][kn], np.ones(kn.sum()), yk, .10), NB_108_ipw=Q.nb_ipcw(R108[k][kn], wk / wk.mean(), yk, .10),
                           NB_102_known=Q.nb_ipcw(R102[k][kn], np.ones(kn.sum()), yk, .10))
bs = {k: [] for k in ['dAUC matched - routine', 'dAUC aware@routine - routine', 'dNB matched - routine', 'dNB aware@routine - routine']}
idx = np.flatnonzero(kn); nk = len(idx)
for _ in range(NB_):
    j = idx[rng.integers(0, nk, nk)]; yj, wj = y[j], ipw[j]
    if yj.all() or (~yj).all():
        continue
    ar = Q.wauc(LPu[RT][j], yj, wj); nr = Q.nb_ipcw(R108[RT][j], wj / wj.mean(), yj, .10)
    for p_ in ['matched', 'aware@routine']:
        bs[f'dAUC {p_} - routine'].append(Q.wauc(LPu[p_][j], yj, wj) - ar); bs[f'dNB {p_} - routine'].append(Q.nb_ipcw(R108[p_][j], wj / wj.mean(), yj, .10) - nr)
pp = uk['panels']
uk['contrasts_ipw'] = {k: dict(point=(pp[k.split(' - ')[0][5:]]['AUC_ipw'] - pp[RT]['AUC_ipw']) if k.startswith('dAUC') else
                               (pp[k.split(' - ')[0][4:]]['NB_108_ipw'] - pp[RT]['NB_108_ipw']), ci=Q.ci(v)) for k, v in bs.items()}
out['UKHLS'] = uk; print('UKHLS', json.dumps(uk, indent=1), flush=True)

# ---------------- B1: CHNS, 72-month risk ----------------
SC = json.load(open(Q.W / 'task20_chns' / 'selected_cn.json')); T.install_cn()
xh, _ = T.load_chns(); Ztrh = T.transform_sub(xtr, xtr, T.SHARED); Zh = T.transform_sub(xh, xtr, T.SHARED); yh = xh.died.to_numpy().astype(bool)
RC = 'routine (shared items)'
PC = {'age+sex only': [], RC: T.CN_ROUTINE_SHARED, 'matched': SC['matched']['features'], 'aware@routine': SC['aware_at_routine']['features']}
R72, R9 = {}, {}
for k, f in PC.items():
    R72[k] = fit(Ztrh, Zh, f, 72)[1]; R9[k] = fit(Ztrh, Zh, f, 108)[1]
one = np.ones(len(yh)); ch = dict(n=int(len(yh)), deaths=int(yh.sum()), panels={})
for k in PC:
    ch['panels'][k] = dict(OE_72=float(yh.sum() / R72[k].sum()), slope_72=Q.slope_w(R72[k], yh, one), NB_72_tau6=Q.nb_ipcw(R72[k], one, yh, .06),
                           NB_9y_tau10_old=Q.nb_ipcw(R9[k], one, yh, .10), flagged_72_tau6=float((R72[k] >= .06).mean()), flagged_9y_tau10=float((R9[k] >= .10).mean()),
                           per1000_deaths_missed_72=1000 * float(((R72[k] < .06) & yh).mean()), per1000_deaths_missed_9y=1000 * float(((R9[k] < .10) & yh).mean()),
                           threshold_6y_equivalent_to_9y_10pct=float(np.nanmin(R72[k][R9[k] >= .10])) if (R9[k] >= .10).any() else None)
ch['NB_treat_all_72_tau6'] = Q.nb_ipcw(np.ones(len(yh)), one, yh, .06)
bs = {'dNB matched - routine': [], 'dNB aware@routine - routine': []}; n = len(yh)
for _ in range(NB_):
    i = rng.integers(0, n, n); nr = Q.nb_ipcw(R72[RC][i], one[i], yh[i], .06)
    bs['dNB matched - routine'].append(Q.nb_ipcw(R72['matched'][i], one[i], yh[i], .06) - nr); bs['dNB aware@routine - routine'].append(Q.nb_ipcw(R72['aware@routine'][i], one[i], yh[i], .06) - nr)
ch['contrasts_72'] = {k: dict(point=ch['panels'][k.split(' - ')[0][4:]]['NB_72_tau6'] - ch['panels'][RC]['NB_72_tau6'], ci=Q.ci(v)) for k, v in bs.items()}
out['CHNS'] = ch; print('CHNS', json.dumps(ch, indent=1), flush=True)
json.dump(out, open(Q.HERE / 'censor.json', 'w'), indent=1, default=float)
print('done')
