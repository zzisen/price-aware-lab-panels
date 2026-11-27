"""POST HOC diagnosis: why does R4 beat the budget-matched relaxed MTL on the drug tasks but lose on Tox21?
Hypothesis 1 (sample size): R4's one-measurement-per-target restriction acts as regularisation, useful when
training samples are few and a handicap when they are many. Test: the drug tasks' own learning curve."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
W = Path(__file__).resolve().parent.parent
d = pd.read_csv(W / 'task8_real_cost' / 'summary' / 'part1_drug_per_n.csv')
d['n_label'] = d.n_label.astype(str)
order = ['25', '50', '100', '200', 'all']
print('DRUG tasks: R4 vs budget-matched relaxed MTL, separately at each training size')
print('%6s %8s %7s %8s %8s %9s %9s %7s' % ('n', 'n_mean', 'tasks', 'R4 wins', 'p', 'ev_R4', 'ev_MTL', 'k_R4'))
rows = []
for n in order:
    h = d[d.n_label == n]
    if not len(h):
        continue
    diff = h.regret_R4.to_numpy() - h.regret_g_MTLrelaxed_matched.to_numpy()
    diff = diff[np.abs(diff) > 1e-12]
    w = int((diff < 0).sum())
    p = binomtest(w, len(diff), 0.5).pvalue if len(diff) else np.nan
    print('%6s %8.0f %7d %8s %8.4f %9.3f %9.3f %7.1f'
          % (n, h.n_mean.mean(), len(diff), f'{w}/{len(diff)}', p, h.ev_R4.mean(), h.ev_g_MTLrelaxed_matched.mean(), h.k_R4.mean()))
    rows.append({'dataset': 'drug', 'n_label': n, 'n_mean': h.n_mean.mean(), 'tasks': len(diff), 'R4_wins': w,
                 'p': p, 'ev_R4': h.ev_R4.mean(), 'ev_matched_MTL': h.ev_g_MTLrelaxed_matched.mean(),
                 'k_R4': h.k_R4.mean(), 'n_candidates': h.nmeas_Ridge.mean()})

c = pd.read_csv(W / 'task9_toxcast' / 'summary' / 'task9_criterion.csv')
diff = c.regret_R4.to_numpy() - c.regret_matched_relaxed_MTL.to_numpy() if 'regret_matched_relaxed_MTL' in c else \
       c.regret_R4.to_numpy() - c.regret_matched_MTL.to_numpy()
diff = diff[np.abs(diff) > 1e-12]
w = int((diff < 0).sum())
print('%6s %8.0f %7d %8s %8.4f %9.3f %9.3f %7.1f'
      % ('all', 5314, len(diff), f'{w}/{len(diff)}', binomtest(w, len(diff), 0.5).pvalue,
         c.ev_R4.mean(), c.ev_matched_relaxed_MTL.mean(), c.k_R4_endpoints.mean()))
rows.append({'dataset': 'tox21', 'n_label': 'all', 'n_mean': 5314.0, 'tasks': len(diff), 'R4_wins': w,
             'p': float(binomtest(w, len(diff), 0.5).pvalue), 'ev_R4': c.ev_R4.mean(),
             'ev_matched_MTL': c.ev_matched_relaxed_MTL.mean(), 'k_R4': c.k_R4_endpoints.mean(),
             'n_candidates': c.cand_endpoints.mean()})
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent / 'reversal_by_n.csv', index=False)
