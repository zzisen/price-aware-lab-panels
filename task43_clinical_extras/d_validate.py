#!/usr/bin/env python3
"""Task 43 (b) and (c), post hoc validation in the NHANES validation cohorts, mirroring task21_final/run_t21.py (training NHANES 1999-2010,
cohort horizons 108/60/36 months, Harrell's C of the Cox linear predictor, 1000 bootstrap resamples of participants, seed 20261107,
one draw per resample, so the lambda = 1 intervals reproduce the Task 21 ones exactly when the panels are the same).
(b) for each lambda: routine, matched and aware@routine panels selected with that lambda (b_lambda/select_<lam>.json) and fitted with Cox(lambda).
(c) Bortz et al. 2023 inputs available in NHANES (with CRP where the cohort has CRP), refitted with Cox(1), against our panel selected at
    the same price (c_bortz_select.json), plus the Task 21 matched and aware@routine panels and the routine bundle.
Usage: d_validate.py <NH3|GH|I>. Output d_<cohort>.json"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
os.chdir(W / 'task21_final'); sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]
H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]
LAMS = [0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]
xtr, _ = M.load(list('ABCDEF'))
if COH == 'NH3':
    xte, _ = M.load_nh3()
elif COH == 'GH':
    xte = pd.concat([T.load_gh(c)[0] for c in 'GH'])
else:
    xte, _ = X.load_i()
ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr)
t, e = T.outcome_at(xte, H); Z = M.transform(xte, ref=xtr); n = len(t)
has_crp = bool(xte.crp.notna().all())
BASE = M.BASE
LP, INFO = {}, {}


def add(name, feats, lam):
    cols = BASE + list(feats)
    m = M.Cox(lam).fit(Ztr[cols].to_numpy(), ttr, etr)
    LP[name] = m.lp(Z[cols].to_numpy()); INFO[name] = dict(feats=list(feats), US=M.cost_us(feats) if feats else 0.0, lam=lam)


ROUT = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C'])
sel = {}
for lam in LAMS:
    f = HERE / 'b_lambda' / f'select_{lam:g}.json'
    if not f.exists():
        continue
    s = json.load(open(f)); sel[lam] = s
    add(f'routine|lam={lam:g}', ROUT, lam)
    add(f'matched|lam={lam:g}', M.features_of(s['matched']['items']), lam)
    add(f'aware@routine|lam={lam:g}', M.features_of(s['aware']['CMP+CBC+LIPID+A1C']['items']), lam)
cb = json.load(open(HERE / 'c_bortz_select.json'))
SEL = json.load(open(W / 'task19_dm2' / 'selected_all.json'))
bk = 'with_crp' if has_crp else 'no_crp'
add('Bortz inputs (NHANES-available)', [f for f in cb['bortz_feats'] if has_crp or f != 'crp'], 1.0)
add('aware@Bortz price', cb[bk]['aware']['feats'], 1.0)
add('routine', ROUT, 1.0)
add('matched (Task 21)', M.features_of(SEL['US']['matched']['items']), 1.0)
add('aware@routine (Task 21)', M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']), 1.0)
pairs = [(f'{a}|lam={lam:g}', f'routine|lam={lam:g}') for lam in sel for a in ('matched', 'aware@routine')]
pairs += [('aware@Bortz price', 'Bortz inputs (NHANES-available)'), ('matched (Task 21)', 'Bortz inputs (NHANES-available)'),
          ('aware@routine (Task 21)', 'Bortz inputs (NHANES-available)'), ('routine', 'Bortz inputs (NHANES-available)'),
          ('aware@Bortz price', 'routine')]
C = {k: M.cindex(v, t, e) for k, v in LP.items()}
rng = np.random.default_rng(20261107); NB = 1000
need = sorted(set(k for p in pairs for k in p))
D = {f'{a} - {b}': [] for a, b in pairs}
for _ in range(NB):
    i = rng.integers(0, n, n); Cb = {k: M.cindex(LP[k][i], t[i], e[i]) for k in need}
    for a, b in pairs:
        D[f'{a} - {b}'].append(Cb[a] - Cb[b])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
out = dict(cohort=COH, horizon=H, n=int(n), deaths=int(e.sum()), crp=has_crp, C=C, panels=INFO,
           dC={k: dict(point=float(C[k.split(' - ')[0]] - C[k.split(' - ')[1]]), ci=ci(v)) for k, v in D.items()})
# reproduction check against Task 21 (lambda = 1)
ref = json.load(open(W / 'task21_final' / f't21_{COH}_endpoints.json'))
out['check_vs_task21'] = {a: dict(point_new=out['dC'][f'{a}|lam=1 - routine|lam=1']['point'], point_t21=ref['point'][f'dC {a} - routine'],
                                  ci_new=out['dC'][f'{a}|lam=1 - routine|lam=1']['ci'], ci_t21=ref['boot'][f'dC {a} - routine'])
                          for a in ('matched', 'aware@routine') if 1.0 in sel}
json.dump(out, open(HERE / f'd_{COH}.json', 'w'), indent=1)
print(json.dumps(out['check_vs_task21'], indent=1))
for k, v in out['dC'].items():
    print(f"{k:75s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
