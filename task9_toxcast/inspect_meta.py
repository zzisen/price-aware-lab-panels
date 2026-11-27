"""Pre-plan coverage check for ToxCast (metadata and marginal distributions only; no relation between endpoints)."""
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parent.parent / 'external_data' / 'toxcast_v4_3'
a = pd.read_excel(D / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx')
print(a.shape); print(list(a.columns))
for c in ['assay_source_name', 'intended_target_family', 'intended_target_family_sub', 'assay_format_type', 'signal_direction', 'assay_function_type']:
    if c in a: print(c, a[c].value_counts().head(25).to_dict())
t = pd.read_excel(D / 'assay_target_mappings_invitrodb_v4_3_AUG2024.xlsx'); print('targets', t.shape, list(t.columns))
m = pd.read_csv(D / 'mc5_6_compact.csv.gz', usecols=['dsstox_substance_id', 'aeid', 'aenm', 'hitc', 'spid'])
print('hitc quantiles', m.hitc.quantile([0, .01, .05, .25, .5, .75, .9, .95, 1]).to_dict())
print('hitc == -1', int((m.hitc == -1).sum()), 'hitc<0', int((m.hitc < 0).sum()), 'hitc>=0.9', float((m.hitc >= 0.9).mean()))
t21 = m[m.aenm.str.startswith('TOX21_')]
ae = a.set_index('aeid')
eps = t21.aeid.unique()
print('Tox21 endpoints', len(eps), 'distinct aid', ae.loc[ae.index.intersection(eps), 'aid'].nunique() if 'aid' in ae else 'n/a')
if 'intended_target_family' in ae:
    print('Tox21 endpoints by family', ae.loc[ae.index.intersection(eps), 'intended_target_family'].value_counts().to_dict())
