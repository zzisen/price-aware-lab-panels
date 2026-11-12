#!/usr/bin/env python3
"""Table 1 summaries for the two cohorts added after table1.csv was built (Understanding Society, CLHLS 65-79), from the final analysis samples.
Also the Understanding Society median interval from nurse visit to end-point wave (appendix table S19). Output: table1_new_cohorts.json"""
import json
import sys
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parent.parent
for d in ['task21_final', 'task27_ukhls', 'task25_newdata']:
    sys.path.insert(0, str(W / d))
import dm2lib as M  # noqa: E402
import ukhls_lib as K  # noqa: E402
import clhls_lib as C  # noqa: E402
M.L.low_priority()
out = {}
xu, cu = K.load_ukhls()
out['Understanding Society'] = dict(n=int(len(xu)), deaths=int(xu.died.sum()), age_mean=float(xu.age.mean()), age_sd=float(xu.age.std()),
                                     age60_79_pct=float(100 * (xu.age >= 60).mean()), female_pct=float(100 * (xu.sex == 2).mean()), counts={k: v for k, v in cu.items() if not isinstance(v, dict)})
xl, _ = C.load_clhls(); k = (xl.age_group == '65-79').to_numpy(); x6 = xl[k]; t6, e6 = C.outcome(x6)
out['CLHLS 65-79'] = dict(n=int(len(x6)), deaths=int(np.asarray(e6).sum()), age_mean=float(x6.age.mean()), age_sd=float(x6.age.std()),
                          female_pct=float(100 * (x6.sex == 2).mean()))
json.dump(out, open(Path(__file__).parent / 'table1_new_cohorts.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
