#!/usr/bin/env python3
"""Task 43 (c), post hoc: our search at the price of the Bortz et al. 2023 (Commun Biol 6:1089) inputs available in NHANES.
Bortz full model: 25 stably selected biomarkers (Fig. 3a of the paper) + age + sex. Available in NHANES 1999-2010 (our feature set):
  CMP members (glucose, albumin, urea, creatinine, ALP, ALT), GGT, total cholesterol, CBC (RDW, MCV, WBC with neutrophil/granulocyte,
  monocyte and lymphocyte percentages, platelets, MPV), HbA1c, CRP.
Not available: cystatin C, SHBG, 25-OH vitamin D, apolipoprotein A, platelet distribution width, mean sphered cell volume,
  high-light-scatter reticulocyte percentage, plateletcrit (only platelets and MPV), MCH and RBC count (not in our harmonised CBC features).
Procedure as task19_dm2/final_select.py: same training layer, folds, lambda 1, beam width 4, US prices; the cache starts from the
official task19 cache (task19_dm2/selected_all_cache.csv) and the panel is best_affordable at the budget.
Budgets: price of the NHANES-available Bortz inputs with CRP, and without CRP (for NHANES 2011-2014, which lacks CRP).
Output c_bortz_select.json"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
os.chdir(W / 'task21_final'); sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
M.L.low_priority()
BORTZ = ['glucose', 'albumin', 'bun', 'creatinine', 'alp', 'alt', 'ggt', 'tc', 'rdw', 'mcv', 'wbc', 'gran_pct', 'mono_pct', 'lymph_pct',
         'plt', 'mpv', 'hba1c', 'crp']
BORTZ_NOCRP = [f for f in BORTZ if f != 'crp']
assert set(BORTZ) <= set(M.FEATURES), set(BORTZ) - set(M.FEATURES)
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
cc = pd.read_csv(W / 'task19_dm2' / 'selected_all_cache.csv')
cache = {tuple(str(f).split()) if isinstance(f, str) else (): float(v) for f, v in zip(cc.features, cc.cv)}
# check: cached scores are reproduced by the code path (routine bundle and the matched panel)
chk = {}
for items in (M.STANDARD['CMP+CBC+LIPID+A1C'], ['CBC', 'CMP', 'phosphorus']):
    k = tuple(sorted(M.features_of(items)))
    chk['+'.join(items)] = dict(cache=cache[k], recomputed=M.gain_ll(Z, t, e, list(k), 1.0, folds))
print(chk, flush=True)
out = dict(bortz_feats=BORTZ, check=chk, cache_start=len(cache))
for name, feats in [('with_crp', BORTZ), ('no_crp', BORTZ_NOCRP)]:
    B = M.cost_us(feats)
    k = tuple(sorted(feats))
    if k not in cache:
        cache[k] = M.gain_ll(Z, t, e, list(k), 1.0, folds)
    t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, 'US', cache=cache)
    ai, ac, av = M.best_affordable(cache, B, 'US')
    out[name] = dict(budget=B, bortz_cv=cache[k], beam=dict(items=it, cost=c, cv=v), aware=dict(items=ai, cost=ac, cv=av, feats=M.features_of(ai)),
                     secs=round(time.time() - t0))
    print(name, out[name], flush=True)
out['cache_end'] = len(cache)
json.dump(out, open(HERE / 'c_bortz_select.json', 'w'), indent=1, default=str)
