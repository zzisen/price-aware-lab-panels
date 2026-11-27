#!/usr/bin/env python3
"""Task 32c, NHANES 2011-14: as t32c_equations.py (same equations, self-tests, contrasts and bootstrap), horizon 60 months.
Treatment from BPQ_G/BPQ_H (CDC, downloaded 2026-11-25): blood-pressure medicine = BPQ050A 1 (now taking prescribed medicine); 0 if BPQ050A,
BPQ040A or BPQ020 is 2. Lipid medicine (PREVENT statin term) = BPQ100D 1; 0 if BPQ100D, BPQ090D, BPQ080 or BPQ060 is 2. Output: t32c_GH.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
from t32c_equations import pce, prevent, egfr_ckdepi2021, wolbers, selftest  # noqa: E402
M.L.low_priority()
selftest()
H = 60
CVD = json.load(open(W / 'task21_final' / 'selected_cvd.json')); CB = M.BASE + M.CLIN
xtr, _ = M.load(list('ABCDEF')); Ztr = M.transform(xtr)
xte = pd.concat([T.load_gh(c)[0] for c in 'GH']); Z = M.transform(xte, ref=xtr)
cltr = M.clinical(xtr); oktr = cltr.notna().all(1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
cl = T.clinical_gh(xte)
bp = []
for c in 'GH':
    b = pd.read_sas(T.GH / c / f'BPQ_{c}.xpt', format='xport').set_index('SEQN'); b.index = [f'{c}{int(i)}' for i in b.index]; bp.append(b)
bp = pd.concat(bp)
print('index check', xte.index[:3].tolist(), bp.index[:3].tolist())
bp = bp.reindex(xte.index)
g = lambda v: bp[v].to_numpy()  # noqa: E731
bptx = np.where(g('BPQ050A') == 1, 1.0, np.where((g('BPQ050A') == 2) | (g('BPQ040A') == 2) | (g('BPQ020') == 2), 0.0, np.nan))
statin = np.where(g('BPQ100D') == 1, 1.0, np.where((g('BPQ100D') == 2) | (g('BPQ090D') == 2) | (g('BPQ080') == 2) | (g('BPQ060') == 2), 0.0, np.nan))
print('race codes', xte.race.value_counts().to_dict(), 'tc median', float(xte.tc.median()), 'hdl median', float(xte.hdl.median()),
      'creatinine median', float(xte.creatinine.median()), 'bptx missing', int(np.isnan(bptx).sum()), 'statin missing', int(np.isnan(statin).sum()))
ok = cl.notna().all(1).to_numpy() & ~np.isnan(bptx) & ~np.isnan(statin)
xo, clo = xte.iloc[np.flatnonzero(ok)], cl.iloc[np.flatnonzero(ok)]
Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
tv, eall = T.outcome_at(xo, H); cvd = xo.ucod.isin(['001', '005']).to_numpy(); ev = eall & cvd; comp = eall & ~cvd
age = xo.age.to_numpy(float); fem = (xo.sex == 2).to_numpy(); black = (xo.race == 4).to_numpy()
tc, hdl, scr = xo.tc.to_numpy(float), xo.hdl.to_numpy(float), xo.creatinine.to_numpy(float)
sbp, smk, dm = clo.sbp.to_numpy(float), clo.smoker.to_numpy(float), clo.diabetes.to_numpy(float)
bt, st = bptx[ok], statin[ok]
S = {'PCE (published)': pce(age, fem, black, tc, hdl, sbp, bt, smk, dm),
     'PREVENT base (published)': prevent(age, fem, tc, hdl, sbp, bt, st, smk, dm, egfr_ckdepi2021(scr, age, fem))}
cv_tr_t, cv_tr_e = M.outcome(xtr, 'cvd'); cv_tr_t, cv_tr_e = cv_tr_t[oktr], cv_tr_e[oktr]
CV = {k: v['features'] for k, v in CVD['guideline'].items() if k in ('PREVENT full', 'routine', 'none')}
CV.update({'CVD aware@PREVENT-full price': CVD['aware_at_prevent_full']['features'], 'CVD aware@routine': CVD['aware_at_routine']['features']})
CV = {k: v for k, v in CV.items() if 'crp' not in v}
for k, f in CV.items():
    S[k] = M.Cox(1.0).fit(Ztrc[CB + list(f)].to_numpy(), cv_tr_t, cv_tr_e).lp(Zc[CB + list(f)].to_numpy())
pairs = [('CVD aware@PREVENT-full price', 'PREVENT base (published)'), ('CVD aware@PREVENT-full price', 'PCE (published)'),
         ('PREVENT full', 'PREVENT base (published)'), ('none', 'PREVENT base (published)'), ('PREVENT base (published)', 'PCE (published)')]
out = dict(cohort='GH', n=len(tv), cvd_deaths=int(ev.sum()), other_deaths=int(comp.sum()), bptx_share=float(bt.mean()), lipid_med_share=float(st.mean()),
           C_censored={k: M.cindex(v, tv, ev) for k, v in S.items()}, C_competing={k: wolbers(v, tv, ev, comp) for k, v in S.items()})
rng = np.random.default_rng(20261128); n = len(tv); bc = {f'{a} - {b}': [] for a, b in pairs}; bw = {f'{a} - {b}': [] for a, b in pairs}
for _ in range(1000):
    j = rng.integers(0, n, n)
    Cc = {k: M.cindex(S[k][j], tv[j], ev[j]) for k in S}; Cw = {k: wolbers(S[k][j], tv[j], ev[j], comp[j]) for k in S}
    for a, b in pairs:
        bc[f'{a} - {b}'].append(Cc[a] - Cc[b]); bw[f'{a} - {b}'].append(Cw[a] - Cw[b])
ci = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]  # noqa: E731
out['contrasts_censored'] = {f'{a} - {b}': dict(point=out['C_censored'][a] - out['C_censored'][b], ci=ci(bc[f'{a} - {b}'])) for a, b in pairs}
out['contrasts_competing'] = {f'{a} - {b}': dict(point=out['C_competing'][a] - out['C_competing'][b], ci=ci(bw[f'{a} - {b}'])) for a, b in pairs}
json.dump(out, open(HERE / 't32c_GH.json', 'w'), indent=1, default=float)
print('n', n, 'CVD', int(ev.sum()), 'other', int(comp.sum()), 'BP meds', round(float(bt.mean()), 3), 'lipid meds', round(float(st.mean()), 3))
for k in S:
    print(f"{k:34s} C(censored) {out['C_censored'][k]:.4f}  C(competing) {out['C_competing'][k]:.4f}")
for k in out['contrasts_censored']:
    a, b = out['contrasts_censored'][k], out['contrasts_competing'][k]
    print(f"{k:62s} cens {a['point']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]  comp {b['point']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
