#!/usr/bin/env python3
"""Export the final 9-year all-cause mortality models fitted on the training layer (NHANES 1999-2010) for the web calculator.
Training data only; NHANES III is not read. Output: ../tool/calculator/model.json"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import dm2lib as M
M.L.low_priority()
SEL = json.load(open('selected_all.json'))
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
cl = M.clinical(x); ok = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
CB = M.BASE + M.CLIN
offset = {a: float(0.01 * x[a][x[a] > 0].median()) for a in M.FEATURES}
out = dict(source='NHANES 1999-2010, adults 40-79, complete cases; ridge Cox (lambda 1); 9-year all-cause death', n=int(len(t)), deaths=int(e.sum()),
           n_clinical=int(ok.sum()), log_offset=offset, units={'glucose': 'mg/dL', 'bun': 'mg/dL', 'creatinine': 'mg/dL (IDMS-standardised)', 'sodium': 'mmol/L',
           'potassium': 'mmol/L', 'chloride': 'mmol/L', 'co2': 'mmol/L', 'calcium': 'mg/dL', 'albumin': 'g/dL', 'protein': 'g/dL', 'alp': 'U/L', 'alt': 'U/L',
           'ast': 'U/L', 'bilirubin': 'mg/dL', 'phosphorus': 'mg/dL', 'ggt': 'U/L', 'wbc': '10^3/uL', 'lymph_pct': '%', 'mono_pct': '%', 'gran_pct': '% (neutrophil + eosinophil + basophil)',
           'hgb': 'g/dL', 'mcv': 'fL', 'rdw': '%', 'plt': '10^3/uL', 'mpv': 'fL', 'acr': 'mg/g', 'bmi': 'kg/m2', 'sbp': 'mmHg'}, models={})
steps = [[], M.features_of(['CMP']), M.features_of(['CMP', 'CBC']), M.features_of(['CMP', 'CBC', 'acr'])]
for k, f in enumerate(steps):
    m = M.Cox(1.0).fit(Zc[CB + f].to_numpy(), t[ok], e[ok])
    out['models'][f'step{k}'] = dict(columns=CB + f, mean=m.m.tolist(), sd=m.s.tolist(), coef=m.b.tolist(), H0=m.H0)
for name, items in [('matched', SEL['US']['matched']['items']), ('routine', M.STANDARD['CMP+CBC+LIPID+A1C'])]:
    f = M.features_of(items); m = M.Cox(1.0).fit(Z[M.BASE + f].to_numpy(), t, e)
    out['models'][name] = dict(items=items, columns=M.BASE + f, mean=m.m.tolist(), sd=m.s.tolist(), coef=m.b.tolist(), H0=m.H0)
out['step_costs_US'] = [0.0, M.cost(['CMP']), M.cost(['CMP', 'CBC']), M.cost(['CMP', 'CBC', 'acr'])]
out['intervals'] = {k: {s: v for s, v in SEL['sequential'][k].items() if s in ('start', 'CMP', 'CBC+CMP')} for k in ('tau=0.1', 'tau=0.2')}
p = Path('../tool/calculator'); p.mkdir(parents=True, exist_ok=True)
json.dump(out, open(p / 'model.json', 'w'), indent=1)
print('exported', list(out['models']), out['step_costs_US'], json.dumps(out['intervals']['tau=0.1']))
