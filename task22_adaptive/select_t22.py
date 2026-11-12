#!/usr/bin/env python3
"""Task 22: fit the VOI stopping rule on all training data (NHANES 1999-2010, complete clinical variables) before freezing.
Writes rule_t22.json (gain-model coefficients and the printable rule chart for V = 10k, 30k, 100k). Training data only."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t22lib as V  # noqa: E402
M = V.M
M.L.low_priority()

x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x)
cl = M.clinical(x); ok = cl.notna().all(axis=1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
folds = np.array_split(np.random.default_rng(1).permutation(int(ok.sum())), 5)
rule = V.VOIRule().fit(Zc, t[ok], e[ok], folds)
out = dict(train='NHANES 1999-2010 (A-F), complete clinical variables', n=int(ok.sum()), deaths=int(e[ok].sum()), order=V.ORDER, tau=V.TAU,
           V_main=V.V_MAIN, V_sensitivity=list(V.V_SENS), costs={'+'.join(s) or 'clinical only': rule.cost(s) for s in rule.states},
           gain_models=rule.coef(), chart={str(int(v)): rule.chart(v) for v in (V.V_MAIN,) + V.V_SENS})
json.dump(out, open(HERE / 'rule_t22.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
