#!/usr/bin/env python3
"""Understanding Society (UKHLS) loader for Task 27 (plan_t27_FROZEN.md). Rules fixed from variable names, labels, code counts and the
documentation only; no analyte value or outcome was summarised before freezing.
Blood: SN 7251 xlabblood_ns (nurse health assessment; wave 2 = 2010-12 general population sample, wave 3 = 2011-13 former BHPS sample).
Age and sex: SN 7251 <w>_indresp_ns (<w>_confage, age confirmed at the nurse visit; <w>_nsex); if absent, <w>_indsamp_ns (<w>_age at selection,
<w>_nsex); if still absent, SN 6614 xwaveid (year the sample reached the laboratory minus birth year; sex).
Outcome (SN 6614 xwaveid; the End User Licence has no date of death, only the wave by which a death was reported):
  died  = dcsedfl_dv == 1 and wave of the nurse visit <= dcsedw_dv <= nurse wave + 9 (death reported by the wave fielded about 9 years later);
  alive = not died and lwenum_dv (last wave enumerated) >= nurse wave + 9;
  otherwise the 9-year status is unknown and the person is not analysed (same rule as CHNS: known status only).
Analytes: negative values are Understanding Society codes. -31 (below the detection limit) -> the smallest valid value of that analyte in the file
(hs-CRP below detection is set to the Task 19 CRP floor directly); -32 (above the linear range) -> the largest valid value; any other code -> missing.
Units are converted to NHANES units (ukhls_menu.UNITS; HbA1c IFCC -> NGSP). A constant unit factor changes calibration only, not discrimination."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import ukhls_menu as U  # noqa: E402
EXT = W / 'external_data' / 'ukhls'
NS = EXT / '7251stata' / 'UKDA-7251-stata' / 'stata' / 'stata13'
MS = EXT / '6614stata' / 'UKDA-6614-stata' / 'stata' / 'stata14_se' / 'ukhls'
FOLLOW = 9                                  # waves after the nurse wave (each wave is fielded over 24 months, one wave started each year)
WL = {2: 'b', 3: 'c'}


def outcome_from(nurse_wave, dcsedfl, dcsedw, lwenum):
    """Vectorised 9-year status. Returns died (bool) and known (bool)."""
    w = np.asarray(nurse_wave, float); dfl = np.asarray(dcsedfl, float); dw = np.asarray(dcsedw, float); lw = np.asarray(lwenum, float)
    died = (dfl == 1) & (dw >= w) & (dw <= w + FOLLOW)
    alive = (~died) & (lw >= w + FOLLOW)
    return died, died | alive


def clean_analyte(v):
    """Apply the code rules to one analyte column (raw units)."""
    v = pd.Series(v, dtype=float); ok = v >= 0
    out = v.where(ok)
    if ok.any():
        out[v == -31] = v[ok].min(); out[v == -32] = v[ok].max()
    return out


def harmonise(raw):
    """raw: DataFrame with the UKHLS analyte columns (raw codes and units). Returns NHANES-unit columns named as in the NHANES menu."""
    x = pd.DataFrame(index=raw.index)
    for a, (col, f) in U.UNITS.items():
        x[a] = clean_analyte(raw[col]).to_numpy() * f
    x['hba1c'] = U.hba1c_ngsp(clean_analyte(raw['hba1c']).to_numpy())
    x['crp'] = np.where(x.crp < M.CRP_CUT, M.CRP_FLOOR, x.crp)          # NaN stays NaN (comparison is False -> keeps NaN)
    x.loc[(pd.Series(raw['hscrp'], dtype=float) == -31).to_numpy(), 'crp'] = M.CRP_FLOOR   # below detection is below the floor by definition
    return x


def load_ukhls():
    import pyreadstat
    rd = lambda p, cols: pyreadstat.read_dta(str(p), usecols=cols)[0]  # noqa: E731
    cols = ['pidp', 'wave'] + [c for c, _ in U.UNITS.values()] + ['hba1c', 'indbdub_xw', 'dory']
    b = rd(NS / 'xlabblood_ns.dta', cols); counts = {'blood_records': len(b)}
    b = b.sort_values('wave', kind='stable').drop_duplicates('pidp', keep='first').set_index('pidp'); counts['blood_people'] = len(b)
    xw = rd(MS / 'xwaveid.dta', ['pidp', 'sex', 'birthy', 'dcsedfl_dv', 'dcsedw_dv', 'lwenum_dv']).set_index('pidp')
    age = pd.Series(np.nan, index=b.index); sex = pd.Series(np.nan, index=b.index); src = pd.Series('', index=b.index)
    for w, l in WL.items():
        ids = b.index[b.wave == w]
        r = rd(NS / f'{l}_indresp_ns.dta', ['pidp', f'{l}_confage', f'{l}_nsex']).drop_duplicates('pidp').set_index('pidp')
        s = rd(NS / f'{l}_indsamp_ns.dta', ['pidp', f'{l}_age', f'{l}_nsex']).drop_duplicates('pidp').set_index('pidp')
        a1 = r[f'{l}_confage'].reindex(ids).where(lambda v: v >= 0); a2 = s[f'{l}_age'].reindex(ids).where(lambda v: v >= 0)
        a3 = (b.dory.reindex(ids).where(lambda v: v > 0) - xw.birthy.reindex(ids).where(lambda v: v > 0))
        age[ids] = a1.fillna(a2).fillna(a3)
        src[ids] = np.where(a1.notna(), 'confage', np.where(a2.notna(), 'selection', np.where(a3.notna(), 'birthyear', 'none')))
        s1 = r[f'{l}_nsex'].reindex(ids).where(lambda v: v.isin([1, 2])); s2 = s[f'{l}_nsex'].reindex(ids).where(lambda v: v.isin([1, 2]))
        s3 = xw.sex.reindex(ids).where(lambda v: v.isin([1, 2]))
        sex[ids] = s1.fillna(s2).fillna(s3)
    counts['age_source'] = src.value_counts().to_dict()
    x = harmonise(b); x['age'] = age; x['sex'] = sex; x['wave'] = b.wave; x['weight'] = b.indbdub_xw.where(b.indbdub_xw > 0)
    x = x[(x.age >= M.L.AGE[0]) & (x.age <= M.L.AGE[1])]; counts['age_40_79'] = len(x)
    o = xw.reindex(x.index)
    died, known = outcome_from(x.wave, o.dcsedfl_dv, o.dcsedw_dv, o.lwenum_dv)
    counts['not_in_xwaveid'] = int(o.dcsedfl_dv.isna().sum())
    x['died'] = died.astype(int); x = x[known]; counts['known_9y_status'] = len(x)
    x = x.dropna(subset=U.MENU + ['age', 'sex']); counts['complete'] = len(x)
    counts['by_wave'] = {int(k): int(v) for k, v in x.wave.value_counts().items()}
    x['cycle'] = 'UKHLS'
    return x, counts
