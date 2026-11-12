#!/usr/bin/env python3
"""Export the models used in the manuscript (v0.4) for the web calculator. Training data only (NHANES 1999-2010); no validation cohort is read.
  - Sequential ordering: the frozen value-of-information rule (Task 22): step models (clinical only; + CMP; + CBC; + urine ACR) and the
    printed chart for V = $10 000, $30 000 (main) and $100 000, copied from ../../task22_adaptive/rule_t22.json.
  - Fixed panels (age + sex base, as in the primary analysis): matched panel, target-aware panel at the routine price, routine bundle.
  - Self-test vectors: risks computed here in Python for 5 training records per model, so the page can check its own arithmetic.
Output: model_v2.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
M.L.low_priority()
SEL = json.load(open(W / 'task19_dm2' / 'selected_all.json')); RULE = json.load(open(W / 'task22_adaptive' / 'rule_t22.json'))
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
cl = M.clinical(x); ok = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
CB = M.BASE + M.CLIN
offset = {a: float(0.01 * x[a][x[a] > 0].median()) for a in M.FEATURES}
units = {'glucose': 'mg/dL', 'bun': 'mg/dL', 'creatinine': 'mg/dL', 'sodium': 'mmol/L', 'potassium': 'mmol/L', 'chloride': 'mmol/L', 'co2': 'mmol/L',
         'calcium': 'mg/dL', 'albumin': 'g/dL', 'protein': 'g/dL', 'alp': 'U/L', 'alt': 'U/L', 'ast': 'U/L', 'bilirubin': 'mg/dL', 'phosphorus': 'mg/dL',
         'ggt': 'U/L', 'wbc': '10³/µL', 'lymph_pct': '%', 'mono_pct': '%', 'gran_pct': '%', 'hgb': 'g/dL', 'mcv': 'fL', 'rdw': '%', 'plt': '10³/µL',
         'mpv': 'fL', 'acr': 'mg/g', 'tc': 'mg/dL', 'hdl': 'mg/dL', 'tg': 'mg/dL', 'hba1c': '%', 'bmi': 'kg/m²', 'sbp': 'mmHg'}
out = dict(source='NHANES 1999-2010, adults aged 40-79, complete cases; ridge Cox (lambda 1); 9-year all-cause death', n=int(len(t)), deaths=int(e.sum()),
           n_clinical=int(ok.sum()), log_offset=offset, units=units, models={}, tests={})
rng = np.random.default_rng(0)


def add(name, Zf, cols, tt, ee, extra=None):
    m = M.Cox(1.0).fit(Zf[cols].to_numpy(), tt, ee)
    out['models'][name] = dict(columns=cols, mean=m.m.tolist(), sd=m.s.tolist(), coef=m.b.tolist(), H0=float(m.H0), **(extra or {}))
    idx = rng.choice(len(Zf), 5, replace=False); X = Zf[cols].to_numpy()[idx]; raw = x.iloc[np.flatnonzero(ok)[idx] if Zf is Zc else idx]
    rec = []
    for i in range(5):
        r = {c: (float(raw[c].iloc[i]) if c in M.FEATURES else None) for c in cols}
        r['age'] = float(raw.age.iloc[i]); r['female'] = float(raw.sex.iloc[i] == 2)
        if Zf is Zc:
            for c in M.CLIN:
                r[c] = float(Zc[c].iloc[idx[i]])
        if 'acr' in r and r['acr'] is not None:
            r['acr'] *= 100                                           # the page takes mg/g; training stores albumin (mg/L) / creatinine (mg/dL)
        rec.append(dict(inputs=r, risk=float(m.risk(X[i:i + 1])[0])))
    out['tests'][name] = rec


steps = [[], M.features_of(['CMP']), M.features_of(['CMP', 'CBC']), M.features_of(['CMP', 'CBC', 'acr'])]
for k, f in enumerate(steps):
    add(f'step{k}', Zc, CB + f, t[ok], e[ok])
for name, items in [('matched', SEL['US']['matched']['items']), ('target-aware', SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']),
                    ('routine', M.STANDARD['CMP+CBC+LIPID+A1C'])]:
    f = M.features_of(items); add(name, Z, M.BASE + f, t, e, dict(items=items, price_US=M.cost(items)))
# Missing inputs: mean and covariance of the transformed inputs in training data (people with complete clinical items), used to replace a
# missing input by its conditional mean given the inputs present (see missing.py), and the 0.5th-99.5th percentiles of each raw input,
# used to flag values outside the training range.
sys.path.insert(0, str(HERE)); import missing as MS  # noqa: E402
JN = M.BASE + list(M.FEATURES) + list(M.CLIN)
out['joint'] = dict(names=JN, mean=Zc[JN].mean().tolist(), cov=np.cov(Zc[JN].to_numpy(), rowvar=False).tolist())
rawc = x.iloc[np.flatnonzero(ok)]
out['range'] = {a: [float(np.percentile(rawc[a] * (100 if a == 'acr' else 1), 0.5)), float(np.percentile(rawc[a] * (100 if a == 'acr' else 1), 99.5))]
                for a in M.FEATURES}
out['range'].update({c: [float(np.percentile(Zc[c], 0.5)), float(np.percentile(Zc[c], 99.5))] for c in ['bmi', 'sbp']})
out['tests_missing'] = []
for name, drop in [('matched', ['phosphorus']), ('matched', ['wbc', 'hgb', 'plt']), ('target-aware', ['acr', 'ggt']), ('step2', ['sbp', 'mcv'])]:
    mdl = out['models'][name]; Zf = Zc
    for i in rng.choice(len(Zf), 2, replace=False):
        zrow = {c: (None if c in drop else float(Zf[c].iloc[i])) for c in mdl['columns']}
        raw = {c: None for c in mdl['columns']}
        for c in mdl['columns']:
            if c in drop:
                continue
            if c in M.FEATURES:
                raw[c] = float(rawc[c].iloc[i]) * (100 if c == 'acr' else 1)
            elif c == 'age':
                raw[c] = float(rawc.age.iloc[i])
            elif c == 'female':
                raw[c] = float(rawc.sex.iloc[i] == 2)
            else:
                raw[c] = float(Zf[c].iloc[i])
        r, lo, hi = MS.predict(mdl, out['joint'], zrow)
        out['tests_missing'].append(dict(model=name, inputs=raw, risk=r, low=lo, high=hi))
out['step_costs_US'] = [0.0, M.cost(['CMP']), M.cost(['CMP', 'CBC']), M.cost(['CMP', 'CBC', 'acr'])]
out['chart'] = RULE['chart']; out['V_main'] = RULE['V_main']; out['tau'] = RULE['tau']
assert [round(v, 2) for v in out['step_costs_US']] == [round(v, 2) for v in RULE['costs'].values()], (out['step_costs_US'], RULE['costs'])
json.dump(out, open(HERE / 'model_v2.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('models', list(out['models']), '| step costs', out['step_costs_US'], '| chart V', list(out['chart']), '| tau', out['tau'])
