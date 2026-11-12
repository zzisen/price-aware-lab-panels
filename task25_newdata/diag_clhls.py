#!/usr/bin/env python3
"""Task 25 diagnosis after the first run (does NOT use the outcome): (a) linkage check - age and sex in the blood file vs the survey file;
(b) unit check - per-wave median of each analyte after conversion vs the NHANES training median; (c) duplicate ids. Output: diag_clhls.txt"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'task21_final'))
import clhls_lib as C  # noqa: E402
import dm2lib as M  # noqa: E402
xtr, _ = M.load(list('ABCDEF')); out = []
for wave, (tab, sav, waves) in C.WAVES.items():
    b = pd.read_csv(C.EXT / tab, sep='\t'); b.columns = [c.strip('"').lower() for c in b.columns]
    out.append(f'== wave {wave}: blood rows {len(b)}, unique ids {b.id.nunique()}, id dtype {b.id.dtype}, example ids {b.id.head(3).tolist()}')
    sv = next(C.EXT.rglob(sav)); L, _ = pyreadstat.read_sav(str(sv), usecols=['id', 'trueage', 'a1'], encoding='gbk')
    out.append(f'   survey rows {len(L)}, unique ids {L.id.nunique()}, example ids {L.id.head(3).tolist()}')
    m = b.drop_duplicates('id').merge(L.drop_duplicates('id'), on='id', suffixes=('_blood', '_survey'))
    out.append(f'   linked {len(m)}; age equal {np.mean(np.isclose(m.trueage_blood, m.trueage_survey)):.3f}; |age diff|<=1 {np.mean(np.abs(m.trueage_blood - m.trueage_survey) <= 1):.3f}; sex equal {np.mean(m.a1_blood == m.a1_survey):.3f}')
    dup = b[b.id.duplicated(keep=False)].sort_values('id').head(4)
    if len(dup):
        out.append('   duplicated id rows (first 4, age/sex only): ' + dup[['id', 'trueage', 'a1']].to_string(index=False).replace('\n', ' | '))
    for a, (src, f) in C.UNITS.items():
        col = next(c for c in ((src,) if isinstance(src, str) else src) if c in b.columns)
        v = C.clean(b[col]) * f
        out.append(f'   {a:10s} from {col:7s}: median {v.median():9.3f} (NHANES {xtr[a].median():9.3f}), p1 {v.quantile(.01):9.3f}, p99 {v.quantile(.99):9.3f}, missing {v.isna().mean():.2f}, raw max {pd.to_numeric(b[col], errors="coerce").max()}')
txt = '\n'.join(out); (HERE / 'diag_clhls.txt').write_text(txt, encoding='utf-8'); print(txt)
