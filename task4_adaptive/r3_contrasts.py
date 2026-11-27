"""Task 4b: per-rep intervals for R3 - R1 and R3 - fixed resolutions (2.5-97.5 percentiles over reps, as in Task 3)."""
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
rows = []
for tag in ['gdsc1_repeated_random', 'gdsc1_held_out_tissue', 'gdsc2_repeated_random', 'gdsc2_held_out_tissue', 'gdsc2_new',
            'prism_repeated_random', 'prism_held_out_tissue']:
    a = pd.read_csv(HERE / f'{tag}_rules.csv'); b = pd.read_csv(HERE / f'{tag}_r3.csv')
    f = a.merge(b[['evaluation', 'outer_seed', 'fold', 'n_label', 'rep', 'loss_sum_R3']], on=['evaluation', 'outer_seed', 'fold', 'n_label', 'rep'])
    for n_label, h in f.groupby('n_label', sort=False):
        for other in ['R1', 'Shared', 'Family', 'Exact']:
            d = h.loss_sum_R3 - h[f'loss_sum_{other}']
            per = d.groupby(h.rep).sum() / h.n_test_cells.groupby(h.rep).sum()
            rows.append({'analysis': tag, 'n_label': n_label, 'contrast': f'R3 − {other}', 'mean': d.sum() / h.n_test_cells.sum(),
                         'rep_2.5': per.quantile(.025) if len(per) > 1 else None, 'rep_97.5': per.quantile(.975) if len(per) > 1 else None})
out = pd.DataFrame(rows); out.to_csv(HERE / 'r3_contrasts.csv', index=False)
print(out[out.contrast.isin(['R3 − R1', 'R3 − Exact'])].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
