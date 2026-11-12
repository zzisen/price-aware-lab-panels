#!/usr/bin/env python3
"""Table 1 (cohort characteristics) for the Lancet Digital Health draft. Descriptive only; same loaders and complete-case rules as Task 21.
Run from anywhere: uses ../task21_final as working directory for the shared libraries. Output: manuscript_ldh/table1.csv"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; T21 = HERE.parent / 'task21_final'
os.chdir(T21); sys.path.insert(0, str(T21))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()


def describe(name, role, x, died, horizon, country, years, ses=True):
    r = dict(cohort=name, role=role, country=country, survey_years=years, horizon=horizon, n=len(x), deaths=int(died.sum()),
             death_pct=100 * died.mean(), age_mean=x.age.mean(), age_sd=x.age.std(), age60plus_pct=100 * (x.age >= 60).mean(),
             female_pct=100 * (x.sex == 2).mean())
    if 'race' in x:
        for k, v in M.RACE.items():
            r[f'{v}_pct'] = 100 * (x.race == k).mean()
    if ses:
        s = X.ses(x)
        r['income_lt1.3_pct'] = 100 * (s.income_grp == 0).mean(); r['income_missing_pct'] = 100 * s.income_grp.isna().mean()
        r['edu_below_hs_pct'] = 100 * (s.edu_grp == 0).mean(); r['edu_missing_pct'] = 100 * s.edu_grp.isna().mean()
    return r


rows = []
xtr, _ = M.load(list('ABCDEF')); t, e = M.outcome(xtr)
rows.append(describe('NHANES 1999-2010', 'Training (panel selection and model fitting)', xtr, e, '9 years', 'USA', '1999-2010'))
x3, _ = M.load_nh3(); t, e = T.outcome_at(x3, 108)
rows.append(describe('NHANES III', 'External validation (primary)', x3, e, '9 years', 'USA', '1988-1994'))
xg = pd.concat([T.load_gh(c)[0] for c in 'GH']); t, e = T.outcome_at(xg, 60)
rows.append(describe('NHANES 2011-2014', 'External validation (primary)', xg, e, '5 years', 'USA', '2011-2014'))
xc, _ = T.load_chns()
rows.append(describe('CHNS', 'External validation (primary)', xc, xc.died.to_numpy().astype(bool), 'about 6 years (to the 2015 survey)', 'China', '2009',
                     ses=False))
xi, _ = X.load_i(); t, e = T.outcome_at(xi, 36)
rows.append(describe('NHANES 2015-2016', 'Supplementary validation (not pooled)', xi, e, '3 years', 'USA', '2015-2016'))
d = pd.DataFrame(rows); d.to_csv(HERE / 'table1.csv', index=False)
pd.set_option('display.width', 250); print(d.round(1).T.to_string())
