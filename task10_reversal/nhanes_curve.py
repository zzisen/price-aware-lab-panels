"""POST HOC: independent replication of the crossover on NHANES, whose learning curve runs to n=1000."""
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parent.parent
d = pd.read_csv(W / 'task8_real_cost' / 'summary' / 'part2_nhanes_per_n.csv')
d['n_label'] = d.n_label.astype(str)
order = [x for x in ['25', '50', '100', '200', '500', '1000', 'all'] if x in set(d.n_label)]
d = d.set_index('n_label').reindex(order)
cols = ['ev_R4', 'ev_g_MTLrelaxed_matched', 'ev_MultiTaskLasso', 'ev_Ridge']
print('NHANES (one task, 9 targets, 44 candidates), explained variance by training size:')
print(d[['n_mean'] + cols + ['k_R4']].round(3).to_string())
print()
print('R4 minus budget-matched relaxed MTL (positive = R4 better):')
print((d.ev_R4 - d.ev_g_MTLrelaxed_matched).round(4).to_string())
print()
print('R4 as a fraction of full-panel Ridge:')
print((d.ev_R4 / d.ev_Ridge).round(3).to_string())
