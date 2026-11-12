#!/usr/bin/env python3
"""Task 21 pooling and hierarchical decision (plan_t21_FROZEN). Usage: pool_t21.py [--dry] [--final]
Primary hypotheses, tested in order (a later one is claimed only if the earlier ones pass):
  H1  matched panel not worse than routine: pooled random-effects difference (C; AUC for binary cohorts), 95% CI lower bound > -0.01.
  H2  target-aware panel within the routine price better than routine: pooled 95% CI lower bound > 0.
  H3  sequential ordering (NHANES III, 9-year decisions): mean cost / CMP+CBC upper 95% <= 0.80, decision agreement >= 0.97, dNB lower 95% > -0.005;
      and in every other cohort with clinical variables, cost share upper 95% <= 0.80 and agreement >= 0.97.
Cohorts pooled: NHANES III, NHANES 2011-14, CHNS, and Understanding Society and CLHLS (65-79) when available (--final). NHANES 2015-16 is
supplementary (reported, not pooled). Without --final the pooled result is labelled interim."""
import json
import sys
from pathlib import Path
import t21lib as X
DRY = '--dry' in sys.argv; FINAL = '--final' in sys.argv; P = 'dry_' if DRY else ''
coh = {'NH3': f'{P}t21_NH3_endpoints.json', 'GH': f'{P}t21_GH_endpoints.json', 'CHNS': f'{P}t21_CHNS_endpoints.json',
       'UKHLS': f'{P}t21_UKHLS_endpoints.json', 'CLHLS': f'{P}t21_CLHLS_endpoints.json'}
E = {k: json.load(open(v)) for k, v in coh.items() if Path(v).exists()}
key = {'H1': ('dC matched - routine', 'dAUC matched - routine (shared items)'), 'H2': ('dC aware@routine - routine', 'dAUC aware@routine - routine (shared items)')}
out = {'cohorts': list(E), 'final': FINAL}
for h, (kc, ka) in key.items():
    est, lo, hi, names = [], [], [], []
    for c, d in E.items():
        k = kc if kc in d['point'] else ka
        est.append(d['point'][k]); lo.append(d['boot'][k][0]); hi.append(d['boot'][k][1]); names.append(c)
    out[h] = dict(per_cohort={c: dict(est=a, lo=b, hi=g) for c, a, b, g in zip(names, est, lo, hi)}, pooled=X.pool(est, lo, hi))
out['H1']['pass'] = out['H1']['pooled']['lo'] > -0.01
out['H2']['pass'] = bool(out['H1']['pass'] and out['H2']['pooled']['lo'] > 0)
s = E['NH3']['seq']
h3_nh3 = s['share_CI'][1] <= 0.80 and s['agree_final'] >= 0.97 and s['dNB_CI'][0] > -0.005
others = {c: (d['seq']['share_CI'][1] <= 0.80 and d['seq']['agree_final'] >= 0.97) for c, d in E.items() if c != 'NH3' and 'seq' in d}
out['H3'] = dict(NH3=dict(share_CI=s['share_CI'], agree=s['agree_final'], dNB=s['dNB'], dNB_CI=s['dNB_CI'], ok=h3_nh3), others=others,
                 pass_=bool(out['H2']['pass'] and h3_nh3 and all(others.values())))
if Path(f'{P}t21_I_endpoints.json').exists():
    d = json.load(open(f'{P}t21_I_endpoints.json'))
    out['supplementary_NHANES_2015_16'] = {k: dict(point=d['point'][k], ci=d['boot'][k]) for k in ['dC matched - routine', 'dC aware@routine - routine'] if k in d['point']}
out['status'] = 'final' if FINAL else 'interim (Understanding Society and CLHLS pending)'
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(f'{P}t21_pooled{"_final" if FINAL else "_interim"}.json', 'w'), indent=1, default=float)
