#!/usr/bin/env python3
"""Task 32b (PLAN.md): cardiovascular death with competing risks. Usage: t32b_cvd_compete.py <NH3|GH>
Panels, models and people exactly as run_t21.py (clinical base model, cause-specific ridge Cox fitted on NHANES 1999-2010 people with complete
clinical variables). Discrimination: (i) the original Harrell C with other deaths censored (reproduction check against t21_<COH>_*.json);
(ii) Wolbers' competing-risk concordance truncated at the cohort horizon: a cardiovascular death at t_i is compared with everyone still
event-free at t_i and with everyone who died of another cause before t_i. Paired bootstrap, 1000 resamples, seed 20261127. Output: t32b_<COH>.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]; H = {'NH3': 108, 'GH': 60}[COH]
CVD = json.load(open(W / 'task21_final' / 'selected_cvd.json'))
ROUT = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']); CB = M.BASE + M.CLIN
xtr, _ = M.load(list('ABCDEF')); Ztr = M.transform(xtr)
if COH == 'NH3':
    xte, _ = M.load_nh3()
else:
    xte = pd.concat([T.load_gh(c)[0] for c in 'GH'])
Z = M.transform(xte, ref=xtr)
cltr = M.clinical(xtr); oktr = cltr.notna().all(1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
cl = M.clinical(xte) if COH == 'NH3' else T.clinical_gh(xte)
okc = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(okc)]
xo = xte.iloc[np.flatnonzero(okc)]
tv, eall = T.outcome_at(xo, H)
cvd_cause = xo.ucod.isin(['001', '005']).to_numpy()
ev = eall & cvd_cause; comp = eall & ~cvd_cause
cv_tr_t, cv_tr_e = M.outcome(xtr, 'cvd'); cv_tr_t, cv_tr_e = cv_tr_t[oktr], cv_tr_e[oktr]
has_crp = bool(xte.crp.notna().all())
CV = {k: v['features'] for k, v in CVD['guideline'].items()}
CV.update({'CVD matched to routine': CVD['matched_routine']['features'], 'CVD matched to PREVENT full': CVD['matched_prevent_full']['features'],
           'CVD aware@routine': CVD['aware_at_routine']['features'], 'CVD aware@PREVENT-full price': CVD['aware_at_prevent_full']['features']})
CV = {k: v for k, v in CV.items() if has_crp or 'crp' not in v}
LPv = {k: M.Cox(1.0).fit(Ztrc[CB + list(f)].to_numpy(), cv_tr_t, cv_tr_e).lp(Zc[CB + list(f)].to_numpy()) for k, f in CV.items()}


def wolbers(s, t, e, c):
    """Competing-risk concordance: cases = cause events; controls = event-free beyond t_i, or competing event at or before t_i."""
    num = den = 0.0
    for i in np.flatnonzero(e):
        ok = (t > t[i]) | (c & (t <= t[i])); ok[i] = False
        d = s[i] - s[ok]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += ok.sum()
    return num / den if den else np.nan


cpairs = [('CVD aware@PREVENT-full price', 'PREVENT full'), ('CVD matched to routine', 'routine'), ('CVD aware@routine', 'routine'),
          ('PCE/SCORE2 lipids', 'none'), ('CVD matched to PREVENT full', 'PREVENT full')]
cpairs = [p for p in cpairs if p[0] in LPv and p[1] in LPv]
out = dict(cohort=COH, horizon=H, n=len(tv), cvd_deaths=int(ev.sum()), other_deaths=int(comp.sum()),
           C_censored={k: M.cindex(v, tv, ev) for k, v in LPv.items()}, C_competing={k: wolbers(v, tv, ev, comp) for k, v in LPv.items()},
           price={k: (M.cost_us(f) if f else 0.0) for k, f in CV.items()})
rng = np.random.default_rng(20261127); bs = {f'{a} - {b}': [] for a, b in cpairs}; n = len(tv)
for _ in range(1000):
    j = rng.integers(0, n, n)
    W_ = {k: wolbers(LPv[k][j], tv[j], ev[j], comp[j]) for k in set(x for p in cpairs for x in p)}
    for a, b in cpairs:
        bs[f'{a} - {b}'].append(W_[a] - W_[b])
ci = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]  # noqa: E731
out['contrasts_competing'] = {f'{a} - {b}': dict(point=out['C_competing'][a] - out['C_competing'][b], ci=ci(bs[f'{a} - {b}'])) for a, b in cpairs}
ref = json.load(open(next((W / 'task21_final').glob(f't21_{COH}_*.json')))) if list((W / 'task21_final').glob(f't21_{COH}_*.json')) else None
if ref and 'cvd' in ref:
    out['reproduction_max_abs_diff_C'] = max(abs(out['C_censored'][k] - ref['cvd']['C'][k]) for k in out['C_censored'] if k in ref['cvd']['C'])
json.dump(out, open(HERE / f't32b_{COH}.json', 'w'), indent=1, default=float)
print(COH, 'n', n, 'CVD', int(ev.sum()), 'other', int(comp.sum()), 'repro', out.get('reproduction_max_abs_diff_C'))
for k in LPv:
    print(f"{k:34s} ${out['price'][k]:6.2f}  C(censored) {out['C_censored'][k]:.4f}  C(competing) {out['C_competing'][k]:.4f}")
for k, v in out['contrasts_competing'].items():
    print(f"{k:50s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
