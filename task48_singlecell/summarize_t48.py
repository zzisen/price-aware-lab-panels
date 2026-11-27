#!/usr/bin/env python3
"""Task 48 summary (fifth locked test; PLAN_FROZEN.md). Usage: summarize_t48.py <tag> <folder>
Rule v4 unchanged (n/k >= 10 and 1 - cv5_C >= 0.10 -> C; otherwise 5-fold CV between A and C, ties to C). Regret = best of A, B, C minus chosen.
Sources weighted equally (single data type). Bootstrap over sources, 2000 draws, seed 20264899.
Primary: H1 rule - best fixed (always A/B/C chosen on these data) upper < 0; H2 rule - n/k alone upper < 0; H3 rule - cross-validation alone upper < 0.
Secondary: share of learnable settings at n/k >= 10 with C > A; rule - rule without the check; per-source differences; test EV."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
tag, folder = sys.argv[1], sys.argv[2]
files = glob.glob(str(HERE / folder / '*_o0.csv'))
if tag == 't48':
    assert folder == 'results' and len(files) == 180, len(files)
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if tag == 't48':
    assert d.source.nunique() == 10 and d.source.str.startswith('S_').all()
    assert len(d) == 20736 and d['pipe'].eq('R').all() and not d.duplicated(['source', 'task', 'p', 'n', 'rep', 'k']).any(), len(d)
d['nk'] = d.n / d.k; best = d[['A', 'B', 'C']].max(axis=1); hi = d.nk >= 10
low = np.where(d.cv5_A < d.cv5_C, d.A, d.C); learn = (1 - d.cv5_C) >= 0.10
R = {'rule': np.where(hi & learn, d.C, low), 'H': np.where(hi, d.C, low), 'NK10': np.where(hi, d.C, d.A), 'CV': low,
     'always_A': d.A, 'always_B': d.B, 'always_C': d.C}
r = pd.DataFrame({k: best - v for k, v in R.items()}); r['source'] = d.source.values
S = r.groupby('source').mean(); m = S.mean()
fixed = min(['always_A', 'always_B', 'always_C'], key=lambda c: m[c])
rng = np.random.default_rng(20264899)
B = pd.DataFrame([S.iloc[rng.integers(0, len(S), len(S))].mean() for _ in range(2000)])
ci = lambda x: [float(x.quantile(.025)), float(x.quantile(.975))]
o = {'tag': tag, 'cells': int(len(d)), 'sources': int(len(S)), 'regret': {c: float(m[c]) for c in R}, 'best_fixed': fixed}
o['H1_rule_minus_best_fixed'] = [float(m.rule - m[fixed])] + ci(B.rule - B[fixed])
o['H2_rule_minus_NK10'] = [float(m.rule - m.NK10)] + ci(B.rule - B.NK10)
o['H3_rule_minus_CV'] = [float(m.rule - m.CV)] + ci(B.rule - B.CV)
o['pass'] = {k: o[k][2] < 0 for k in ('H1_rule_minus_best_fixed', 'H2_rule_minus_NK10', 'H3_rule_minus_CV')}
st = d[hi].assign(learn=learn[hi].astype(float)).groupby(['source', 'task', 'p', 'n', 'k'])[['A', 'C', 'learn']].mean()
st = st[st.learn >= 0.5]
o['secondary'] = {'learnable_settings_nk10': int(len(st)), 'share_C_leads_A': float((st.C > st.A).mean()) if len(st) else None,
                  'rule_minus_H': [float(m.rule - m.H)] + ci(B.rule - B.H),
                  'rule_minus_always_A': [float(m.rule - m.always_A)] + ci(B.rule - B.always_A),
                  'rule_minus_always_C': [float(m.rule - m.always_C)] + ci(B.rule - B.always_C),
                  'check_withheld_share_nk10': float((~learn[hi]).mean()),
                  'test_EV': {c: float(d[c].mean()) for c in 'ABC'},
                  'per_source_rule_minus_best_fixed': {s: float(S.loc[s, 'rule'] - S.loc[s, fixed]) for s in S.index}}
# prespecified case-dependent tests (review M1), as in locked test 3
withheld = float((~learn[hi]).mean())
o['case'] = 'learnable' if withheld < 0.5 else 'not_learnable'
o['H4_no_cost_if_learnable'] = {'applies': withheld < 0.5, 'rule_minus_H': o['secondary']['rule_minus_H'], 'pass': o['secondary']['rule_minus_H'][2] < 0.001}
o['H5_protection_if_not_learnable'] = {'applies': withheld >= 0.5, 'rule_minus_H': o['secondary']['rule_minus_H'], 'pass': o['secondary']['rule_minus_H'][2] < 0}
o['H6_C_leads_A_in_learnable_settings'] = {'share': o['secondary']['share_C_leads_A'], 'pass': (o['secondary']['share_C_leads_A'] or 0) >= 0.80}
# prespecified sensitivity (review S3, S4): without CD14 monocytes; all sources at n <= 100; lineage-weighted (descriptive)
def sub(dd):
    bb = dd[['A', 'B', 'C']].max(axis=1); hh = dd.nk >= 10; ll = np.where(dd.cv5_A < dd.cv5_C, dd.A, dd.C); le = (1 - dd.cv5_C) >= 0.10
    rr = pd.DataFrame({'rule': bb - np.where(hh & le, dd.C, ll), 'NK10': bb - np.where(hh, dd.C, dd.A), 'CV': bb - ll,
                       'always_A': bb - dd.A, 'always_B': bb - dd.B, 'always_C': bb - dd.C}); rr['source'] = dd.source.values
    SS = rr.groupby('source').mean(); mm = SS.mean(); fx = min(['always_A', 'always_B', 'always_C'], key=lambda c: mm[c])
    g = np.random.default_rng(20264899); BB = pd.DataFrame([SS.iloc[g.integers(0, len(SS), len(SS))].mean() for _ in range(2000)])
    return {'best_fixed': fx, 'rule_minus_best_fixed': [float(mm.rule - mm[fx])] + ci(BB.rule - BB[fx]),
            'rule_minus_NK10': [float(mm.rule - mm.NK10)] + ci(BB.rule - BB.NK10), 'rule_minus_CV': [float(mm.rule - mm.CV)] + ci(BB.rule - BB.CV)}
o['sensitivity'] = {'without_cd14_monocytes': sub(d[d.source != 'S_cd14_monocytes']), 'n_le_100': sub(d[d.n <= 100])}
LIN = {'S_b_cells': 'B', 'S_cd14_monocytes': 'myeloid', 'S_cd34': 'progenitor', 'S_cd56_nk': 'NK'}
lin = S.index.map(lambda x: LIN.get(x, 'T'))
Sl = S.groupby(lin).mean(); ml = Sl.mean(); fl = min(['always_A', 'always_B', 'always_C'], key=lambda c: ml[c])
o['sensitivity']['lineage_weighted_descriptive'] = {'best_fixed': fl, 'rule_minus_best_fixed': float(ml.rule - ml[fl]), 'rule_minus_NK10': float(ml.rule - ml.NK10), 'rule_minus_CV': float(ml.rule - ml.CV)}
json.dump(o, open(HERE / f'{tag}_summary.json', 'w'), indent=1)
print(json.dumps({k: o[k] for k in ('case', 'H4_no_cost_if_learnable', 'H5_protection_if_not_learnable', 'H6_C_leads_A_in_learnable_settings', 'sensitivity')}, indent=1))
print(json.dumps({k: v for k, v in o.items() if k != 'secondary'}, indent=1)); print(json.dumps(o['secondary'], indent=1))
