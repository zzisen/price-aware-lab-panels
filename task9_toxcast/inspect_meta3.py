"""Pre-plan check 3: coverage of the non-viability Tox21 endpoints (counts only; no candidate-target relations)."""
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parent.parent / 'external_data' / 'toxcast_v4_3'
a = pd.read_excel(D / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx')
t = a[a.assay_source_name == 'TOX21']
drop = t.intended_target_family.isin(['channel 1', 'channel 2', 'background measurement']) | (t.cell_viability_assay == 1)
keep = t[~drop]
m = pd.read_csv(D / 'mc5_6_compact.csv.gz', usecols=['dsstox_substance_id', 'aeid', 'hitc'])
m = m[m.aeid.isin(keep.aeid)].dropna(subset=['dsstox_substance_id'])
w = m.groupby(['dsstox_substance_id', 'aeid']).hitc.max().unstack()
print('matrix', w.shape)
per_ep = w.notna().sum()
print('endpoints tested on >= N chemicals:', {n: int((per_ep >= n).sum()) for n in [5000, 6000, 6500, 7000, 7200]})
for thr in [6000, 7000]:
    eps = per_ep[per_ep >= thr].index
    sub = w[eps]
    print(f'thr {thr}: endpoints {len(eps)}, chemicals complete {int(sub.notna().all(axis=1).sum())}')
eps = per_ep[per_ep >= 6000].index
full = w[eps].dropna()
print('final candidate pool shape', full.shape)
fam = keep.set_index('aeid').loc[eps, 'intended_target_family']
print('families', fam.value_counts().to_dict())
sub_fam = keep.set_index('aeid').loc[eps, 'intended_target_family_sub']
print('sub-families', sub_fam.value_counts().to_dict())
full.to_pickle(Path(__file__).resolve().parent / 'tox21_coverage_matrix.pkl')
