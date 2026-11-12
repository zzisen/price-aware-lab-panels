#!/usr/bin/env python3
"""Task 20 library (on top of dm2lib, copied unchanged from Task 19).
- Chinese price schedule: public 2026 fee lists of two Shandong primary-care centres (Jining Jiezhuang, Zhaoyuan Jinling; files in
  external_data/cn_prices). Each analyte billed separately (no panel discount), CBC 19 yuan, venipuncture 6.3 yuan.
- CHNS 2009 biomarkers + deaths reported up to the 2015 survey (binary 6-year outcome).
- NHANES 2011-2014 (G, H) loader (no CRP in these cycles) and 5-year outcome.
- Breslow baseline hazard at any horizon, AUC for a binary outcome."""
from pathlib import Path
import numpy as np
import pandas as pd
import dm2lib as M
import dmlib as L

EXT = L.EXT
GH = EXT / 'nhanes_gh'
CHNS = EXT / 'chns'

# ---------------- Chinese prices (yuan) ----------------
CN_PRICE = {'glucose': 4.5, 'bun': 5.0, 'creatinine': 5.0, 'uric_acid': 5.0, 'alt': 3.5, 'ast': 3.5, 'ggt': 3.5, 'alp': 4.5, 'bilirubin': 4.5,
            'protein': 4.5, 'albumin': 4.5, 'ldh': 4.5, 'tc': 4.0, 'tg': 4.5, 'hdl': 9.0, 'ldl': 11.0, 'hba1c': 27.0, 'crp': 26.0,
            'potassium': 3.0, 'sodium': 3.0, 'chloride': 3.0, 'calcium': 4.0, 'co2': 5.0}
CN_CBC, CN_VENI = 19.0, 6.3
CBC3 = ['wbc', 'hgb', 'plt']                    # the CBC indices CHNS reports
SHARED = ['glucose', 'bun', 'creatinine', 'uric_acid', 'albumin', 'protein', 'alt', 'tc', 'tg', 'hdl', 'hba1c', 'crp'] + CBC3


def cost_cn(f):
    f = set(f)
    tot = sum(CN_PRICE[a] for a in f if a in CN_PRICE) + (CN_CBC if f & set(M.CBC2) else 0)
    return round(tot + (CN_VENI if f else 0), 2)


# Chinese routine check-up: CBC + liver function (7) + kidney function (3) + lipids (4) + glucose
CN_ROUTINE_FULL = ['alt', 'ast', 'ggt', 'alp', 'bilirubin', 'protein', 'albumin', 'bun', 'creatinine', 'uric_acid', 'tc', 'tg', 'hdl', 'ldl', 'glucose', 'wbc']
CN_ROUTINE_SHARED = ['alt', 'protein', 'albumin', 'bun', 'creatinine', 'uric_acid', 'tc', 'tg', 'hdl', 'glucose'] + CBC3

# orderable items for the Chinese menu (single analytes and the 3-index CBC)
CN_ITEMS = {a: [a] for a in SHARED if a not in CBC3}
CN_ITEMS['CBC3'] = CBC3


def install_cn():
    """Make dm2lib's beam / best_affordable use the Chinese menu and prices."""
    M.ITEMS.update(CN_ITEMS)
    M.SCHEDULES['CN'] = cost_cn


def describe_cn(feats):
    f = set(feats); out = []
    if set(CBC3) <= f:
        out.append('CBC'); f -= set(CBC3)
    return out + sorted(f)


# ---------------- models ----------------
def h0_at(model, X, t, e, T):
    """Breslow cumulative baseline hazard of a fitted dmlib.Cox model at time T (same time unit as t)."""
    Z = (np.asarray(X, float) - model.m) / model.s
    o = np.argsort(-t, kind='stable'); Z, t, e = Z[o], t[o], e[o]
    r = np.exp(Z @ model.b)
    S0 = np.cumsum(r)[np.searchsorted(-t, -t, side='right') - 1]
    return float((e / S0)[t <= T].sum())


def risk_at(model, H0, X):
    return 1 - np.exp(-H0 * np.exp(model.lp(X)))


def auc(score, y):
    s = np.asarray(score, float); y = np.asarray(y, bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def transform_sub(x, ref, feats):
    z = pd.DataFrame(index=x.index)
    z['age'] = x.age.astype(float); z['female'] = (x.sex == 2).astype(float)
    for a in feats:
        v, r = x[a].astype(float), ref[a].astype(float)
        z[a] = np.log(np.maximum(v, 0) + 0.01 * max(r[r > 0].median(), 1e-6))
    return z


# ---------------- NHANES 2011-2014 ----------------
GH_FEATURES = [f for f in M.FEATURES if f != 'crp']


def load_gh(c):
    d = GH / c
    read = lambda s: pd.read_sas(d / f'{s}_{c}.xpt', format='xport').set_index('SEQN')  # noqa: E731
    demo = read('DEMO')[['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'WTMEC2YR']]
    demo.columns = ['age', 'sex', 'race', 'weight']
    counts = {'cycle': c, 'in_file': len(demo)}
    demo = demo[(demo.age >= L.AGE[0]) & (demo.age <= L.AGE[1])]; counts['age_40_79'] = len(demo)
    bio = read('BIOPRO'); cols = {}
    for a, (src, v, _) in L.CHEM.items():
        if src == 'bio':
            cols[a] = bio[v]
    cols['tc'] = read('TCHOL')['LBXTC']; cols['hdl'] = read('HDL')['LBDHDD']
    cbc = read('CBC')
    for a in ['wbc', 'lymph_pct', 'mono_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']:
        cols[a] = cbc[L.CBC[a]]
    cols['gran_pct'] = cbc['LBXNEPCT'] + cbc['LBXEOPCT'] + cbc['LBXBAPCT']
    cols['hba1c'] = read('GHB')['LBXGH']
    u = read('ALB_CR'); cols['acr'] = u['URXUMA'] / u['URXUCR']
    lab = pd.DataFrame(cols)
    mort = L.read_mort(d / f"NHANES_{'2011_2012' if c == 'G' else '2013_2014'}_MORT_2019_PUBLIC.dat")
    x = demo.join(mort, how='inner').dropna(subset=['months']); counts['mortality_eligible'] = len(x)
    x = x.join(lab, how='inner').dropna(subset=GH_FEATURES); counts['complete'] = len(x)
    x['crp'] = np.nan
    x['cycle'] = c
    x.index = [f'{c}{int(i)}' for i in x.index]
    return x, counts


def outcome_at(x, months):
    t = np.minimum(x.months.to_numpy(float), months)
    e = (x.died.to_numpy() == 1) & (x.months.to_numpy() <= months)
    return np.maximum(t, 0.5), e


def clinical_gh(x):
    out = []
    for c, g in x.groupby('cycle'):
        d = GH / c
        read = lambda f: pd.read_sas(d / f'{f}_{c}.xpt', format='xport').set_index('SEQN')  # noqa: E731
        seqn = pd.Index([int(i[1:]) for i in g.index])
        bmx, bpx, smq, diq = read('BMX'), read('BPX'), read('SMQ'), read('DIQ')
        f = pd.DataFrame(index=seqn)
        f['bmi'] = bmx.BMXBMI.reindex(seqn)
        f['sbp'] = bpx[[v for v in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if v in bpx]].replace(0, np.nan).mean(1).reindex(seqn)
        ever, now = smq.SMQ020.reindex(seqn), smq.SMQ040.reindex(seqn)
        f['smoker'] = np.where(ever == 2, 0.0, np.where(now.isin([1, 2]), 1.0, np.where(now == 3, 0.0, np.nan)))
        dq = diq.DIQ010.reindex(seqn)
        f['diabetes'] = np.where(dq == 1, 1.0, np.where(dq.isin([2, 3]), 0.0, np.nan))
        f.index = g.index; out.append(f)
    return pd.concat(out).reindex(x.index)


# ---------------- CHNS ----------------
def load_chns():
    """CHNS adults aged 40-79 with 2009 blood. Outcome: death reported by the 2015 survey (about 6 years). Known status = reported dead,
    or seen in the 2015 survey. Unit rules fixed from variable names before any value was read: *_MG columns are mg/dL; urea (mmol/L)
    x 2.8 = BUN mg/dL; albumin, total protein and haemoglobin assumed g/L (/10); hs-CRP assumed mg/L (/10 to mg/dL), then the Task 19
    CRP floor. A wrong unit assumption changes calibration only, not discrimination (log scale, constant factor)."""
    b = pd.read_sas(CHNS / 'biomarker_09.sas7bdat', format='sas7bdat')
    b.columns = [c.upper() for c in b.columns]
    b = b.set_index('IDIND')
    s = pd.read_sas(CHNS / 'surveys_pub_12.sas7bdat', format='sas7bdat'); s.columns = [c.lower() for c in s.columns]
    m = pd.read_sas(CHNS / 'mast_pub_12.sas7bdat', format='sas7bdat'); m.columns = [c.upper() for c in m.columns]; m = m.set_index('IDIND')
    counts = {'cycle': 'CHNS', 'blood_2009': len(b)}
    age09 = s[s.wave == 2009].groupby('idind').age.first()
    seen15 = set(s[s.wave == 2015].idind)
    x = pd.DataFrame(index=b.index)
    x['age'] = age09.reindex(b.index); x['sex'] = m.GENDER.reindex(b.index)
    x['glucose'] = b.GLUCOSE_MG; x['bun'] = 2.8 * b.UREA; x['creatinine'] = b.CRE_MG; x['uric_acid'] = b.UA_MG
    x['albumin'] = b.ALB / 10; x['protein'] = b.TP / 10; x['alt'] = b.ALT
    x['tc'] = b.TC_MG; x['tg'] = b.TG_MG; x['hdl'] = b.HDL_C_MG; x['hba1c'] = b.HBA1C
    x['crp'] = b.HS_CRP / 10; x['wbc'] = b.WBC; x['hgb'] = b.HGB / 10; x['plt'] = b.PLT
    x = x[(x.age >= L.AGE[0]) & (x.age <= L.AGE[1])]; counts['age_40_79'] = len(x)
    dod_y, dod_rpt = m.DOD_Y.reindex(x.index), m.DOD_RPT.reindex(x.index)
    died = ((dod_rpt >= 2011) & (dod_rpt <= 2015)) | ((dod_y >= 2009) & (dod_y <= 2015))
    alive = (~died) & x.index.isin(list(seen15))
    x['died'] = died.astype(int); x = x[died | alive]; counts['known_status'] = len(x)
    x = x.dropna(subset=SHARED + ['age', 'sex']); counts['complete'] = len(x)
    x['crp'] = np.where(x.crp < M.CRP_CUT, M.CRP_FLOOR, x.crp)
    x['cycle'] = 'CHNS'
    return x, counts
