"""Coverage only: BeatAML ex vivo patient samples x inhibitors (no drug-drug relation computed)."""
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parent.parent
d = W / 'external_data' / 'beataml2'
f = pd.read_csv(d / 'beataml_probit_curve_fits_v4_dbgap.txt', sep='\t')
f = f[(f.type == 'single-agent') & (f.paper_inclusion == True) & np.isfinite(f.auc)]  # noqa: E712
w = f.pivot_table(index='dbgap_subject_id', columns='inhibitor', values='auc', aggfunc='mean')
print('samples', w.shape[0], 'inhibitors', w.shape[1])
cov = w.notna().mean().sort_values(ascending=False)
for k in [30, 50, 80, 100, 120]:
    top = cov.index[:k]
    print(f'top {k} inhibitors by coverage (min coverage {cov.iloc[k-1]:.2f}): complete samples {int(w[top].notna().all(axis=1).sum())}')
