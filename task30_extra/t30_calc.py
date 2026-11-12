#!/usr/bin/env python3
"""Task 30 B (post hoc, PLAN.md): coverage and width of the calculator's 80% risk range, and net benefit with imputed inputs.
Same cohorts, models and scenarios as task28_missing/run_t28.py."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
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
REF = pd.read_csv(W / 'task28_missing' / 't28_summary.csv')
xtr, _ = M.load(list('ABCDEF'))
rows = []
rng = np.random.default_rng(20261112)


for coh, H, tau in [('NH3', 108, 0.10), ('GH', 60, 0.05)]:
    xte = M.load_nh3()[0] if coh == 'NH3' else pd.concat([T.load_gh(c)[0] for c in 'GH'])
    t, e = T.outcome_at(xte, H); Z = M.transform(xte, ref=xtr); kn = e | (t >= H)

    def nb(r):
        hi = r >= tau; Wn = kn.sum()
        return float((hi & e & kn).sum() / Wn - (hi & ~e & kn).sum() / Wn * tau / (1 - tau))
    for name in ['matched', 'target-aware', 'routine']:
        m = MOD['models'][name]; cols = m['columns']; X = Z[cols].to_numpy(float)
        r0, _, _ = MS.predict_pattern(m, J, X, []); C0 = M.cindex(np.log(-np.log(1 - r0)), t, e)
        # cohort-horizon risk on the calculator's own scale: rescale the cumulative hazard by the training Breslow ratio
        Xtr = M.transform(xtr)[cols].to_numpy(float); ttr, etr = M.outcome(xtr)
        fit = M.Cox(1.0).fit(Xtr, ttr, etr); ratio = T.h0_at(fit, Xtr, ttr, etr, H) / T.h0_at(fit, Xtr, ttr, etr, 108)
        hz = lambda r: 1 - (1 - r) ** ratio  # noqa: E731
        nb0 = nb(hz(r0))

        def score(scn, r, lo, hi):
            inside = (r0 >= lo - 1e-12) & (r0 <= hi + 1e-12); wd = 100 * (hi - lo)
            return dict(cohort=coh, model=name, scenario=scn, C_complete=C0, C=M.cindex(np.log(-np.log(1 - r)), t, e),
                        coverage=float(inside.mean()), width_median_pp=float(np.median(wd)), width_p90_pp=float(np.quantile(wd, .9)),
                        NB_complete=nb0, dNB=nb(hz(r)) - nb0)
        labs = [c for c in cols if c not in ('age', 'female')]
        single = [score(f'missing: {a}', *MS.predict_pattern(m, J, X, [a])) for a in labs]
        s = pd.DataFrame(single)
        rows.append(dict(cohort=coh, model=name, scenario='one analyte missing (median over analytes)', C_complete=C0, C=s.C.median(),
                         coverage=s.coverage.median(), width_median_pp=s.width_median_pp.median(), width_p90_pp=s.width_p90_pp.median(), NB_complete=nb0, dNB=s.dNB.median()))
        rows.append(dict(cohort=coh, model=name, scenario='one analyte missing (worst analyte)', C_complete=C0, C=s.C.min(),
                         coverage=s.coverage.min(), width_median_pp=s.width_median_pp.max(), width_p90_pp=s.width_p90_pp.max(), NB_complete=nb0, dNB=s.dNB.min()))
        for blk, nm in [(CBC, 'CBC'), (CMP, 'CMP')]:
            miss = [c for c in blk if c in cols]
            if miss:
                rows.append(score(f'whole {nm} missing', *MS.predict_pattern(m, J, X, miss)))
        for p in (0.10, 0.30):
            R = rng.random((len(X), len(labs))) < p
            r, lo, hi = np.empty(len(X)), np.empty(len(X)), np.empty(len(X)); pats = {}
            for i, row in enumerate(R):
                pats.setdefault(tuple(row), []).append(i)
            for pat, idx in pats.items():
                rr, ll, hh = MS.predict_pattern(m, J, X[idx], [a for a, b in zip(labs, pat) if b]); r[idx], lo[idx], hi[idx] = rr, ll, hh
            rows.append(score(f'{int(p * 100)}% of values missing at random', r, lo, hi))
        ref = REF[(REF.cohort == coh) & (REF.model == name)].C_complete.iloc[0]
        assert abs(ref - C0) < 1e-9, (coh, name, ref, C0)
        print(coh, name, 'C', round(C0, 4), 'matches task 28', flush=True)
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).parent / 't30_calc.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20)
print(out.round(4).to_string(index=False))
