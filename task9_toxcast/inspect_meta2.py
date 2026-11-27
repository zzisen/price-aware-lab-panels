"""Pre-plan check 2: one primary endpoint per Tox21 assay (metadata only) and README text on negative hit calls."""
from pathlib import Path
import re
import pandas as pd
D = Path(__file__).resolve().parent.parent / 'external_data' / 'toxcast_v4_3'
try:
    from pypdf import PdfReader
    txt = ' '.join(p.extract_text() or '' for p in PdfReader(D / 'DB_release_README_SUMMARY.pdf').pages)
    for mt in re.finditer(r'[^.]{0,250}(negative|hitc)[^.]{0,250}\.', txt, flags=re.I):
        print('README:', ' '.join(mt.group(0).split())[:500])
except Exception as e:
    print('pdf read failed', e)
a = pd.read_excel(D / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx')
t = a[a.assay_source_name == 'TOX21'].copy()
drop = t.intended_target_family.isin(['channel 1', 'channel 2', 'background measurement']) | (t.cell_viability_assay == 1)
k = t[~drop]
print('Tox21 endpoints', len(t), 'after dropping channel/background/viability', len(k), 'aids', k.aid.nunique())
c = k.groupby('aid').size()
print('endpoints per aid', c.value_counts().to_dict())
for aid in c[c > 1].index[:12]:
    print(aid, k[k.aid == aid][['aeid', 'assay_component_endpoint_name', 'signal_direction', 'assay_function_type']].values.tolist())
print('families of kept', k.intended_target_family.value_counts().to_dict())
print('export_ready', k.export_ready.value_counts().to_dict())
