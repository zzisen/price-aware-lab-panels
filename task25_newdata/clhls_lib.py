#!/usr/bin/env python3
"""CLHLS loader and menu (Task 25; written from the documentation only, before any value was summarised).

Sources (external_data/clhls): biomarker_dataset_CLHLS_{2008,2012,2014}.tab and the 2008-2018, 2011-2018 and 2014-2018 longitudinal .sav files.
Rules fixed from the documentation (the three 'sample biomedical indicators' PDFs and the 2008-2018 codebook):
  - Blood waves 2008/09, 2012 and 2014 are linked by id to the longitudinal file of the same wave (2008-2018, 2011-2018, 2014-2018). A person in
    more than one blood wave is counted once, at the earliest.
  - Baseline date = interview date of that wave (yearin, monthin, dayin; month 99 -> 7, day 99/88 -> 15).
  - Follow-up: status variables dth08_11, dth11_14, dth14_18 (0 surviving, 1 died, 2 surviving 2011 but died before 2012 [treated as died],
    -9 lost, -8 not applicable). Deaths are dated by d11v*/d14v*/d18v* (validated year, month, day); surviving at a wave moves the last-alive
    date to that wave's interview date; lost to follow-up censors at the last-alive date. Unknown death month -> 7, day -> 15; an undatable death
    is placed midway between the last-alive date and the next wave's median interview date.
  - Outcome: death within 60 months (time in months, censored at 60).
  - Units: glucose mmol/L x 18.016; urea ('BUN', mmol/L) x 2.8 to BUN mg/dL; creatinine (documented as mmol/L, range 50-120, i.e. umol/L) / 88.4;
    uric acid umol/L / 59.48; total cholesterol and HDL mmol/L x 38.67; triglycerides mmol/L x 88.57; hs-CRP mg/L / 10 to mg/dL, then the
    Task 19 CRP floor; haemoglobin g/L / 10; white cells and platelets 10^9/L (= 10^3/uL); lymphocyte %, MCV and MPV as reported. A wrong
    unit factor changes calibration only, not discrimination (log scale, constant factor).
  - Missing: blank, negative, or >= 8888 (CLHLS sentinel codes).
  - Ages: 65 and over (65-79 main; 80 and over reported separately); sex from a1 (1 male, 2 female).
Menu: the 14 analytes measured in all three blood waves that NHANES also has. Chinese prices (t20lib): each chemistry priced singly, the blood count as
one item (yuan 19), venipuncture yuan 6.3."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
EXT = W / 'external_data' / 'clhls'
H = 60
CBC_CL = ['wbc', 'lymph_pct', 'hgb', 'mcv', 'plt', 'mpv']
CHEM_CL = ['glucose', 'bun', 'creatinine', 'uric_acid', 'tc', 'tg', 'hdl', 'crp']
MENU = CHEM_CL + CBC_CL
ITEMS_CL = {a: [a] for a in CHEM_CL}; ITEMS_CL['CBC_CL'] = CBC_CL
ROUTINE_CL = [a for a in T.CN_ROUTINE_FULL if a in CHEM_CL] + CBC_CL          # Chinese routine check-up items that CLHLS measured
WAVES = {2008: ('biomarker_dataset_CLHLS_2008.tab', 'clhls_2008_2018_longitudinal_dataset_released_version1.sav',
                [('dth08_11', '_11', 'd11'), ('dth11_14', '_14', 'd14'), ('dth14_18', '_18', 'd18')]),
         2012: ('biomarker_dataset_CLHLS_2012.tab', 'clhls_2011_2018_longitudinal_dataset_released_version1.sav',
                [('dth11_14', '_14', 'd14'), ('dth14_18', '_18', 'd18')]),
         2014: ('biomarker_dataset_CLHLS_2014.tab', 'clhls_2014_2018_longitudinal_dataset_released_version1.sav',
                [('dth14_18', '_18', 'd18')])}
UNITS = {'glucose': ('glu', 18.016), 'bun': ('bun', 2.8), 'creatinine': (('cre', 'crea'), 1 / 88.4), 'uric_acid': ('ua', 1 / 59.48),
         'tc': ('cho', 38.67), 'tg': ('tg', 88.57), 'hdl': (('hdl', 'hdlc'), 38.67), 'crp': ('crphs', 0.1),
         'wbc': ('wbc', 1.0), 'lymph_pct': ('lymph_a', 1.0), 'hgb': ('hgb', 0.1), 'mcv': ('mcv', 1.0), 'plt': ('plt', 1.0), 'mpv': ('mpv', 1.0)}


def install():
    M.ITEMS.update(ITEMS_CL); M.SCHEDULES['CN'] = T.cost_cn


def clean(v):
    v = pd.to_numeric(v, errors='coerce')
    return v.where((v >= 0) & (v < 8888))


def ymd(y, m, d):
    y = pd.to_numeric(y, errors='coerce'); m = pd.to_numeric(m, errors='coerce'); d = pd.to_numeric(d, errors='coerce')
    ok = (y >= 1990) & (y <= 2030)
    m = m.where((m >= 1) & (m <= 12), 7); d = d.where((d >= 1) & (d <= 31), 15)
    # DEVIATION D3: keep valid days 29-31; only a day beyond the end of its month (e.g. 30 February) is moved to the month's last day
    first = pd.to_datetime(dict(year=y.where(ok, 2000), month=m, day=1), errors='coerce')
    dim = first.dt.days_in_month
    s = first + pd.to_timedelta(np.minimum(d, dim) - 1, unit='D')
    return s.where(ok)


def survival(L, waves):
    """L: longitudinal frame (one row per person, columns as in the codebook). Returns (months, died within H)."""
    t0 = ymd(L['yearin'], L.get('monthin'), L.get('dayin'))
    last = t0.copy(); end = pd.Series(pd.NaT, index=L.index); died = pd.Series(False, index=L.index); open_ = pd.Series(True, index=L.index)
    for sv, suf, dp in waves:
        s = pd.to_numeric(L[sv], errors='coerce')
        nxt = ymd(L.get('yearin' + suf), L.get('monthin' + suf), L.get('dayin' + suf))
        med = nxt[s == 0].median() if (s == 0).any() else last.max()
        dd = ymd(L.get(dp + 'vyear'), L.get(dp + 'vmonth'), L.get(dp + 'vday'))
        dd = dd.fillna(last + (med - last) / 2)
        dead = open_ & s.isin([1, 2]); alive = open_ & (s == 0); gone = open_ & ~(s.isin([0, 1, 2]))
        end[dead] = dd[dead]; died[dead] = True; open_[dead] = False
        last[alive] = nxt[alive].fillna(last[alive])
        end[gone] = last[gone]; open_[gone] = False
    end[open_] = last[open_]
    months = (end - t0).dt.days / 30.4375
    e = died & (months <= H)
    # DEVIATION D4: return the raw follow-up (capped at H); the loader removes people without positive follow-up BEFORE the 0.5-month floor
    return np.minimum(months.to_numpy(float), H), e.to_numpy(), t0


def load_clhls():
    rows, counts = [], {}
    import pyreadstat
    for wave, (tab, sav, waves) in WAVES.items():
        b = pd.read_csv(EXT / tab, sep='\t'); b.columns = [c.strip('"').lower() for c in b.columns]
        need = ['id', 'yearin', 'monthin', 'dayin'] + [w[0] for w in waves] + [p + s for _, s, d in waves for p in ('yearin', 'monthin', 'dayin')] \
            + [d + 'v' + k for _, _, d in waves for k in ('year', 'month', 'day')]
        sv = next(EXT.rglob(sav))
        _, meta = pyreadstat.read_sav(str(sv), metadataonly=True, encoding='gbk')
        L, _ = pyreadstat.read_sav(str(sv), usecols=[c for c in need if c in meta.column_names], encoding='gbk')
        L = L.drop_duplicates('id').set_index('id')
        b = b.drop_duplicates('id').set_index('id'); counts[f'blood_{wave}'] = len(b)
        b = b[b.index.isin(L.index)]; counts[f'linked_{wave}'] = len(b)
        t, e, t0 = survival(L.reindex(b.index), waves)
        x = pd.DataFrame(index=b.index)
        x['age'] = clean(b.trueage); x['sex'] = clean(b.a1)
        for a, (src, f) in UNITS.items():
            col = next(c for c in ((src,) if isinstance(src, str) else src) if c in b.columns)
            if a == 'lymph_pct' and wave == 2012:
                col = 'lymph'          # DEVIATION D1: in the 2012 file 'lymph' holds the percentage and 'lymph_a' the count (100*lymph_a/wbc = 30.8%)
            x[a] = clean(b[col]) * f
        if wave == 2008:
            x['crp'] = np.nan          # DEVIATION D2: 2008 hs-CRP has no documented unit (PDF and metadata) and a median 14x the 2012/2014 values
        x['months'], x['died'], x['wave'], x['baseline'] = t, e.astype(int), wave, t0.to_numpy()
        rows.append(x)
    x = pd.concat(rows); counts['rows_2008_excluded'] = int((x.wave == 2008).sum())
    x = x[x.wave != 2008]          # DEVIATION D5: the 2008 blood wave cannot be used (D2), so it no longer occupies a person's earliest draw
    x['pid'] = x.index
    x = x.sort_values('wave').drop_duplicates('pid', keep='first'); counts['persons'] = len(x)
    x = x[x.age >= 65]; counts['age_65plus'] = len(x)
    x = x[x.baseline.notna() & (x.months > 0)]; counts['dated_follow_up'] = len(x)
    x['months'] = np.maximum(x.months, 0.5)
    x = x.dropna(subset=MENU + ['age', 'sex']); counts['complete'] = len(x)
    x['crp'] = np.where(x.crp < M.CRP_CUT, M.CRP_FLOOR, x.crp)
    x['age_group'] = np.where(x.age < 80, '65-79', '80+'); x['cycle'] = 'CLHLS'
    return x, counts


def outcome(x):
    return x.months.to_numpy(float), x.died.to_numpy().astype(bool)
