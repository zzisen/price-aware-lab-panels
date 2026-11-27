"""POST HOC: is the reversal a sample-size (regularisation) effect rather than a dataset difference?
Compares every method against C0 across the drug learning curve, and checks whether the single-measurement
ceiling (R4 / full-panel Ridge) differs between the two datasets."""
from pathlib import Path
import pandas as pd
W = Path(__file__).resolve().parent.parent
d = pd.read_csv(W / 'task8_real_cost' / 'summary' / 'part1_drug_per_n.csv')
d['n_label'] = d.n_label.astype(str)
cols = ['ev_R4', 'ev_g_MTLrelaxed_matched', 'ev_MultiTaskLasso', 'ev_Ridge', 'ev_Lasso', 'ev_Exact', 'ev_Shared']
g = d.groupby('n_label')[cols + ['k_R4', 'nmeas_MultiTaskLasso', 'nmeas_Ridge', 'n_mean']].mean()
g = g.reindex([x for x in ['25', '50', '100', '200', 'all'] if x in g.index])
print('DRUG tasks, mean explained variance by training size:')
print(g[['n_mean'] + cols].round(3).to_string())
print()
print('measurements used:')
print(g[['k_R4', 'nmeas_MultiTaskLasso', 'nmeas_Ridge']].round(1).to_string())
print()
print('R4 minus budget-matched relaxed MTL (positive = R4 better):')
print((g.ev_R4 - g.ev_g_MTLrelaxed_matched).round(4).to_string())
print()
print('R4 as a fraction of full-panel Ridge:')
print((g.ev_R4 / g.ev_Ridge).round(3).to_string())
print()
c = pd.read_csv(W / 'task9_toxcast' / 'summary' / 'task9_criterion.csv')
print('TOX21 (n=5314): R4 %.3f | matched relaxed MTL %.3f | guarded Ridge %.3f | R4/Ridge %.3f | k_R4 %.1f of %.0f'
      % (c.ev_R4.mean(), c.ev_matched_relaxed_MTL.mean(), c.ev_Ridge_guarded.mean(),
         c.ev_R4.mean() / c.ev_Ridge_guarded.mean(), c.k_R4_endpoints.mean(), c.cand_endpoints.mean()))
