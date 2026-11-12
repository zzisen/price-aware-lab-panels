"""Price schedules. Each schedule maps a set of analytes (feature names) to the bill for ordering them together.
Included: US Medicare Clinical Laboratory Fee Schedule 2026 Q4 (USD), Australian MBS pathology fees (AUD), Japanese medical fee schedule 2026
(JPY, 1 point = 10 yen). Users can add their own schedule as a function of a set of analyte names."""
import itertools

CHEM_US = {'glucose': 3.93, 'bun': 3.95, 'creatinine': 5.12, 'sodium': 4.81, 'potassium': 4.76, 'chloride': 4.60, 'co2': 4.88, 'calcium': 5.16,
           'albumin': 4.95, 'protein': 3.67, 'alp': 5.18, 'alt': 5.30, 'ast': 5.18, 'bilirubin': 5.02, 'ggt': 7.20, 'uric_acid': 4.52, 'ldh': 6.04,
           'phosphorus': 4.74, 'iron': 6.47, 'tg': 5.74, 'tc': 4.35, 'hdl': 8.19}
BMP = ['glucose', 'bun', 'creatinine', 'sodium', 'potassium', 'chloride', 'co2', 'calcium']
CMP = BMP + ['albumin', 'protein', 'alp', 'alt', 'ast', 'bilirubin']
LIPID = ['tc', 'hdl', 'tg']
PANELS_US = {'BMP': (BMP, 8.46), 'CMP': (CMP, 10.56), 'LIPID': (LIPID, 13.39)}
CBC = ['wbc', 'lymph_pct', 'mono_pct', 'gran_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']
CHEM = list(CHEM_US)
BLOOD = set(CHEM) | set(CBC) | {'hba1c', 'crp'}


def us(f):
    f = set(f); chem = f & set(CHEM); best = float('inf')
    for use in itertools.product([0, 1], repeat=len(PANELS_US)):
        cov, pr = set(), 0.0
        for on, (m, p) in zip(use, PANELS_US.values()):
            if on:
                cov |= set(m); pr += p
        best = min(best, pr + sum(CHEM_US[a] for a in chem - cov))
    tot = best + (7.77 if f & set(CBC) else 0) + (9.71 if 'hba1c' in f else 0) + (12.95 if 'crp' in f else 0) + (10.96 if 'acr' in f else 0)
    return round(tot + (9.34 if f & BLOOD else 0), 2)


AU_66500 = (set(CHEM) - {'hdl', 'iron'}) | {'crp'}


def au(f):
    f = set(f); n = len(f & AU_66500)
    tot = [0, 9.70, 11.65, 13.65, 15.65][n] if n < 5 else 17.70
    tot += (11.05 if 'hdl' in f else 0) + (32.55 if 'iron' in f else 0) + (17.80 if f & set(CBC) else 0) + (16.80 if 'hba1c' in f else 0) + (20.10 if 'acr' in f else 0)
    return round(tot + (5.95 if f else 0), 2)


JP_11 = {'bilirubin', 'protein', 'albumin', 'bun', 'creatinine', 'uric_acid', 'alp', 'ggt', 'tg', 'potassium', 'calcium', 'glucose', 'ldh', 'iron'}
JP_17 = {'hdl', 'phosphorus', 'tc', 'ast', 'alt'}


def jp(f):
    f = set(f)
    pts = [11] * len(f & JP_11) + [11] * bool(f & {'sodium', 'chloride'}) + [17] * len(f & JP_17)
    n = len(pts); chem = (93 if n <= 7 else 99 if n <= 9 else 103) if n >= 5 else sum(pts)
    chem += 48 if 'co2' in f else 0
    tot = chem + (144 if chem else 0)
    hem = (36 if f & set(CBC) else 0) + (49 if 'hba1c' in f else 0); tot += hem + (125 if hem else 0)
    tot += (160 if 'crp' in f else 0) + (144 if 'acr' in f else 0)
    return float(10 * (tot + (40 if f & BLOOD else 0)))


SCHEDULES = {'US': us, 'AU': au, 'JP': jp}
ORDERABLE = {**{a: [a] for a in CHEM}, 'CBC': CBC, 'hba1c': ['hba1c'], 'crp': ['crp'], 'acr': ['acr'], 'BMP': BMP, 'CMP': CMP, 'LIPID': LIPID}


def features_of(items, orderable=ORDERABLE):
    return list(dict.fromkeys(f for i in items for f in orderable[i]))
