#!/usr/bin/env python3
"""Task 28 (post hoc, see PLAN.md): performance of the calculator's missing-input handling in NHANES III and NHANES 2011-14."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(W / 'tool' / 'calculator'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import missing as MS  # noqa: E402
M.L.low_priority()
MOD = json.load(open(W / 'tool' / 'calculator' / 'model_v2.json', encoding='utf-8')); J = MOD['joint']
CMP = M.features_of(['CMP']); CBC = M.features_of(['CBC'])
xtr, _ = M.load(list('ABCDEF'))
rows, single = [], []
rng = np.random.default_rng(20261112)
for coh, H in [('NH3', 108), ('GH', 60)]:
    if coh == 'NH3':
        xte, _ = M.load_nh3()
    else:
        xte = pd.concat([T.load_gh(c)[0] for c in 'GH'])
    t, e = T.outcome_at(xte, H); Z = M.transform(xte, ref=xtr)
    for name in ['matched', 'target-aware', 'routine']:
        m = MOD['models'][name]; cols = m['columns']; X = Z[cols].to_numpy(float)
        r0, _, _ = MS.predict_pattern(m, J, X, []); lp0 = np.log(-np.log(1 - r0)); C0 = M.cindex(lp0, t, e); d0 = r0 >= 0.10

        def score(scn, r, lo, hi):
            lp = np.log(-np.log(1 - r)); d = r >= 0.10; flag = (lo >= 0.10) != (hi >= 0.10); same = d == d0
            return dict(cohort=coh, model=name, scenario=scn, C_complete=C0, C=M.cindex(lp, t, e), dC=M.cindex(lp, t, e) - C0,
                        same_decision=same.mean(), flagged=flag.mean(), same_if_not_flagged=same[~flag].mean() if (~flag).any() else np.nan,
                        changed_and_flagged=(flag[~same].mean() if (~same).any() else np.nan))
        labs = [c for c in cols if c not in ('age', 'female')]
        for a in labs:
            rec = score(f'missing: {a}', *MS.predict_pattern(m, J, X, [a])); single.append(rec)
        for blk, nm in [(CBC, 'CBC'), (CMP, 'CMP')]:
            miss = [c for c in blk if c in cols]
            if miss:
                rows.append(score(f'whole {nm} missing', *MS.predict_pattern(m, J, X, miss)))
        for p in (0.10, 0.30):
            R = rng.random((len(X), len(labs))) < p
            r, lo, hi = np.empty(len(X)), np.empty(len(X)), np.empty(len(X))
            pats = {}
            for i, row in enumerate(R):
                pats.setdefault(tuple(row), []).append(i)
            for pat, idx in pats.items():
                miss = [a for a, b in zip(labs, pat) if b]
                rr, ll, hh = MS.predict_pattern(m, J, X[idx], miss); r[idx], lo[idx], hi[idx] = rr, ll, hh
            rows.append(score(f'{int(p * 100)}% of values missing at random', r, lo, hi))
        s = pd.DataFrame([x for x in single if x['cohort'] == coh and x['model'] == name])
        rows.append(dict(cohort=coh, model=name, scenario='one analyte missing: worst of each', C_complete=C0, C=s.C.min(), dC=s.dC.min(),
                         same_decision=s.same_decision.min(), flagged=s.flagged.max(), same_if_not_flagged=s.same_if_not_flagged.min(),
                         changed_and_flagged=s.changed_and_flagged.min()))
        print(coh, name, 'C', round(C0, 4), flush=True)
out = pd.DataFrame(rows); sing = pd.DataFrame(single)
out.to_csv(Path(__file__).parent / 't28_summary.csv', index=False); sing.to_csv(Path(__file__).parent / 't28_single.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(out.round(4).to_string(index=False))
print(sing.sort_values('dC').head(8).round(4).to_string(index=False))
