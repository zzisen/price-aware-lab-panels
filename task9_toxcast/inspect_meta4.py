"""Pre-plan check 4: are ac50 / conc_max usable for a potency scale? Marginal counts only, no candidate-target relation."""
from pathlib import Path
import numpy as np
import pandas as pd
D = Path(__file__).resolve().parent.parent / 'external_data' / 'toxcast_v4_3'
a = pd.read_excel(D / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx')
t = a[a.assay_source_name == 'TOX21']
drop = t.intended_target_family.isin(['channel 1', 'channel 2', 'background measurement']) | (t.cell_viability_assay == 1)
eps = set(t[~drop].aeid)
m = pd.read_csv(D / 'mc5_6_compact.csv.gz',
                usecols=['dsstox_substance_id', 'aeid', 'hitc', 'ac50', 'conc_max', 'nconc', 'spid'])
m = m[m.aeid.isin(eps)].dropna(subset=['dsstox_substance_id'])
print('rows', len(m))
act = m.hitc >= 0.9
print('active fraction (hitc>=0.9)', round(float(act.mean()), 4))
print('active rows with ac50 missing', int(m.loc[act, 'ac50'].isna().sum()))
print('conc_max missing', int(m.conc_max.isna().sum()), 'conc_max quantiles', m.conc_max.quantile([0, .25, .5, .75, 1]).round(3).to_dict())
print('nconc quantiles', m.nconc.quantile([0, .5, 1]).to_dict())
print('ac50 of actives, log10 quantiles', np.log10(m.loc[act, 'ac50'].clip(lower=1e-6)).quantile([0, .05, .5, .95, 1]).round(3).to_dict())
print('ac50 above conc_max among actives', int((m.loc[act, 'ac50'] > m.loc[act, 'conc_max']).sum()))
rep = m.groupby(['dsstox_substance_id', 'aeid']).size()
print('cells', len(rep), 'cells with >1 sample', int((rep > 1).sum()))
