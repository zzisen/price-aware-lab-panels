#!/usr/bin/env python3
"""Task 18 (npj Digital Medicine route): NHANES loading and harmonisation, orderable lab items with Medicare prices,
a ridge Cox model, Harrell's C, and cost-aware target-aware panel selection.
Harmonisation rules are the same for every cycle and were fixed from variable names and NHANES documentation only."""
import ctypes
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / 'external_data'
DEV = EXT / 'nhanes_dev'
HOLD = EXT / 'nhanes_holdout'
HORIZON = 108                     # months; the latest held-out cycle (2009-2010) has at least 9 years of follow-up to 2019-12-31
AGE = (40, 79)                    # 2007+ top-codes age at 80
VENI = 9.34                       # 36415 venipuncture, once per blood draw

# file stems per cycle: demographics, biochemistry, CBC, HbA1c, total cholesterol + HDL (two files from 2005), CRP, urine albumin/creatinine
FILES = {
    'A': dict(demo='DEMO', bio='LAB18', cbc='LAB25', ghb='LAB10', tc='LAB13', hdl='LAB13', crp='LAB11', acr='LAB16', mort='1999_2000'),
    'B': dict(demo='DEMO_B', bio='L40_B', cbc='L25_B', ghb='L10_B', tc='L13_B', hdl='L13_B', crp='L11_B', acr='L16_B', mort='2001_2002'),
    'C': dict(demo='DEMO_C', bio='L40_C', cbc='L25_C', ghb='L10_C', tc='L13_C', hdl='L13_C', crp='L11_C', acr='L16_C', mort='2003_2004'),
    'D': dict(demo='DEMO_D', bio='BIOPRO_D', cbc='CBC_D', ghb='GHB_D', tc='TCHOL_D', hdl='HDL_D', crp='CRP_D', acr='ALB_CR_D', mort='2005_2006'),
    'E': dict(demo='DEMO_E', bio='BIOPRO_E', cbc='CBC_E', ghb='GHB_E', tc='TCHOL_E', hdl='HDL_E', crp='CRP_E', acr='ALB_CR_E', mort='2007_2008'),
    'F': dict(demo='DEMO_F', bio='BIOPRO_F', cbc='CBC_F', ghb='GHB_F', tc='TCHOL_F', hdl='HDL_F', crp='CRP_F', acr='ALB_CR_F', mort='2009_2010'),
}
LAYER = {'C': DEV, 'D': DEV, 'A': HOLD, 'B': HOLD, 'E': HOLD, 'F': HOLD}

# analyte -> (source file key, NHANES variable, Medicare individual price)
CHEM = {
    'glucose': ('bio', 'LBXSGL', 3.93), 'bun': ('bio', 'LBXSBU', 3.95), 'creatinine': ('bio', 'LBXSCR', 5.12),
    'sodium': ('bio', 'LBXSNASI', 4.81), 'potassium': ('bio', 'LBXSKSI', 4.76), 'chloride': ('bio', 'LBXSCLSI', 4.60),
    'co2': ('bio', 'LBXSC3SI', 4.88), 'calcium': ('bio', 'LBXSCA', 5.16), 'albumin': ('bio', 'LBXSAL', 4.95),
    'protein': ('bio', 'LBXSTP', 3.67), 'alp': ('bio', 'LBXSAPSI', 5.18), 'alt': ('bio', 'LBXSATSI', 5.30),
    'ast': ('bio', 'LBXSASSI', 5.18), 'bilirubin': ('bio', 'LBXSTB', 5.02), 'ggt': ('bio', 'LBXSGTSI', 7.20),
    'uric_acid': ('bio', 'LBXSUA', 4.52), 'ldh': ('bio', 'LBXSLDSI', 6.04), 'phosphorus': ('bio', 'LBXSPH', 4.74),
    'iron': ('bio', 'LBXSIR', 6.47), 'tg': ('bio', 'LBXSTR', 5.74), 'tc': ('tc', 'LBXTC', 4.35), 'hdl': ('hdl', 'HDL', 8.19),
}
CBC = {'wbc': 'LBXWBCSI', 'lymph_pct': 'LBXLYPCT', 'mono_pct': 'LBXMOPCT', 'neut_pct': 'LBXNEPCT', 'eos_pct': 'LBXEOPCT',
       'baso_pct': 'LBXBAPCT', 'hgb': 'LBXHGB', 'mcv': 'LBXMCVSI', 'rdw': 'LBXRDW', 'plt': 'LBXPLTSI', 'mpv': 'LBXMPSI'}
SINGLE = {'hba1c': ('ghb', 'LBXGH', 9.71, True), 'crp': ('crp', 'LBXCRP', 12.95, True), 'acr': ('acr', None, 5.78 + 5.18, False)}
PANELS = {'BMP': (['glucose', 'bun', 'creatinine', 'sodium', 'potassium', 'chloride', 'co2', 'calcium'], 8.46),
          'CMP': (['glucose', 'bun', 'creatinine', 'sodium', 'potassium', 'chloride', 'co2', 'calcium', 'albumin', 'protein', 'alp', 'alt', 'ast', 'bilirubin'], 10.56),
          'LIPID': (['tc', 'hdl', 'tg'], 13.39)}
CBC_PRICE = 7.77

# orderable items: name -> list of feature columns
ITEMS = {a: [a] for a in CHEM}
ITEMS.update({'CBC': list(CBC), 'hba1c': ['hba1c'], 'crp': ['crp'], 'acr': ['acr']})
ITEMS.update({p: list(m) for p, (m, _) in PANELS.items()})
FEATURES = list(CHEM) + list(CBC) + ['hba1c', 'crp', 'acr']

# standard bundles clinicians order; each is a set of items
STANDARD = {'BMP': ['BMP'], 'CBC': ['CBC'], 'CMP': ['CMP'], 'LIPID': ['LIPID'], 'CMP+CBC': ['CMP', 'CBC'],
            'CMP+CBC+LIPID': ['CMP', 'CBC', 'LIPID'], 'CMP+CBC+LIPID+A1C': ['CMP', 'CBC', 'LIPID', 'hba1c']}


def low_priority():
    k = ctypes.WinDLL('kernel32')
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)


# ---------------- data ----------------
def read_mort(path):
    rows = []
    for line in path.read_text().splitlines():
        g = lambda a, b: line[a - 1:b].strip()  # noqa: E731
        rows.append((int(g(1, 6)), g(15, 15), g(16, 16), g(17, 19), g(46, 48)))
    m = pd.DataFrame(rows, columns=['SEQN', 'elig', 'mortstat', 'ucod', 'permth_exm'])
    m = m[m.elig == '1']
    m['died'] = m.mortstat.astype(int)
    m['months'] = pd.to_numeric(m.permth_exm, errors='coerce')
    return m.set_index('SEQN')[['died', 'months', 'ucod']]


def load_cycle(c):
    """One cycle: adults aged 40-79 with an eligible mortality record and every candidate analyte. Returns (frame, counts)."""
    f, d = FILES[c], LAYER[c] / c
    read = lambda s: pd.read_sas(d / f'{s}.xpt', format='xport').set_index('SEQN')  # noqa: E731
    demo = read(f['demo'])
    demo = demo[['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'WTMEC2YR']].rename(columns={'RIDAGEYR': 'age', 'RIAGENDR': 'sex', 'RIDRETH1': 'race', 'WTMEC2YR': 'weight'})
    counts = {'cycle': c, 'in_file': len(demo)}
    demo = demo[(demo.age >= AGE[0]) & (demo.age <= AGE[1])]
    counts['age_40_79'] = len(demo)
    cols = {}
    bio = read(f['bio'])
    for a, (src, v, _) in CHEM.items():
        if src == 'bio':
            cols[a] = bio[v] if v in bio else bio['LBD' + v[3:]]   # 2001-2002 names five of these LBD... (same units)
    tc = read(f['tc']); cols['tc'] = tc['LBXTC']
    h = read(f['hdl']); cols['hdl'] = h[next(v for v in ['LBDHDD', 'LBXHDD', 'LBDHDL'] if v in h)]   # name differs by cycle
    cbc = read(f['cbc'])
    for a, v in CBC.items():
        cols[a] = cbc[v]
    cols['hba1c'] = read(f['ghb'])['LBXGH']
    cols['crp'] = read(f['crp'])['LBXCRP']
    u = read(f['acr']); cols['acr'] = u['URXUMA'] / u['URXUCR']
    lab = pd.DataFrame(cols)
    # NHANES-recommended creatinine calibration (BIOPRO_D documentation; Selvin et al. 2007 for 1999-2000)
    if c == 'A':
        lab['creatinine'] = 1.013 * lab.creatinine + 0.147
    if c == 'D':
        lab['creatinine'] = -0.016 + 0.978 * lab.creatinine
    mort = read_mort(d / f"NHANES_{f['mort']}_MORT_2019_PUBLIC.dat")
    x = demo.join(mort, how='inner')
    counts['mortality_eligible'] = len(x)
    x = x.join(lab, how='inner')
    x = x.dropna(subset=FEATURES + ['months'])
    counts['complete'] = len(x)
    x['cycle'] = c
    x.index = [f'{c}{int(i)}' for i in x.index]
    return x, counts


def load(cycles):
    parts = [load_cycle(c) for c in cycles]
    return pd.concat([p for p, _ in parts]), [n for _, n in parts]


def outcome(x, cause=None):
    """Follow-up truncated at HORIZON months. cause: None = all-cause, 'cvd' = heart disease or stroke, 'cancer'."""
    t = np.minimum(x.months.to_numpy(float), HORIZON)
    e = (x.died.to_numpy() == 1) & (x.months.to_numpy() <= HORIZON)
    if cause == 'cvd':
        e &= x.ucod.isin(['001', '005']).to_numpy()
    if cause == 'cancer':
        e &= (x.ucod == '002').to_numpy()
    return np.maximum(t, 0.5), e


def transform(x, ref=None):
    """log for every analyte (all are positive), sex as 0/1, age as is. The small offset inside the log is 1% of the positive median in ref
    (default x itself); held-out data use the training frame as ref."""
    ref = x if ref is None else ref
    z = pd.DataFrame(index=x.index)
    z['age'] = x.age.astype(float); z['female'] = (x.sex == 2).astype(float)
    for a in FEATURES:
        v, r = x[a].astype(float), ref[a].astype(float)
        z[a] = np.log(np.maximum(v, 0) + 0.01 * max(r[r > 0].median(), 1e-6))
    return z


# ---------------- cost ----------------
def cost(items):
    """Cheapest Medicare billing of the analytes the items need: individual chemistries or BMP/CMP/lipid panels, plus CBC, HbA1c, CRP, urine ACR and one venipuncture if any blood is drawn."""
    feats = set(f for i in items for f in ITEMS[i])
    chem = feats & set(CHEM)
    best = np.inf
    for use in itertools.product([0, 1], repeat=len(PANELS)):
        covered, price = set(), 0.0
        for on, (p, (m, pr)) in zip(use, PANELS.items()):
            if on:
                covered |= set(m); price += pr
        price += sum(CHEM[a][2] for a in chem - covered)
        best = min(best, price)
    total = best + (CBC_PRICE if feats & set(CBC) else 0) + sum(SINGLE[s][2] for s in ['hba1c', 'crp', 'acr'] if s in feats)
    blood = bool(chem or feats & set(CBC) or 'hba1c' in feats or 'crp' in feats)
    return round(total + (VENI if blood else 0.0), 2)


def features_of(items):
    return list(dict.fromkeys(f for i in items for f in ITEMS[i]))


# ---------------- Cox ----------------
class Cox:
    """Ridge-penalised Cox model (Breslow ties), standardised inputs, Newton steps."""
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, t, e, w=None):
        X = np.asarray(X, float)
        self.m = X.mean(0); self.s = X.std(0); self.s[self.s == 0] = 1
        Z = (X - self.m) / self.s
        w = np.ones(len(t)) if w is None else np.asarray(w, float) / np.mean(w)
        o = np.argsort(-t, kind='stable'); Z, t, e, w = Z[o], t[o], e[o], w[o]
        n, p = Z.shape
        b = np.zeros(p)
        for _ in range(30):
            eta = Z @ b; r = w * np.exp(eta - eta.max())
            S0 = np.cumsum(r); S1 = np.cumsum(r[:, None] * Z, 0)
            # Breslow: for tied times use the risk set at the last index with that time
            last = np.searchsorted(-t, -t, side='right') - 1
            S0, S1 = S0[last], S1[last]
            we = w * e
            g = (we[:, None] * (Z - S1 / S0[:, None])).sum(0) - self.lam * b
            S2 = np.cumsum(r[:, None, None] * Z[:, :, None] * Z[:, None, :], 0)[last] if p <= 60 else None
            mu = S1 / S0[:, None]
            H = (we[:, None, None] * (S2 / S0[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + self.lam * np.eye(p)
            step = np.linalg.solve(H, g)
            step = step * min(1.0, 1.0 / max(np.abs(step).max(), 1e-12))   # damped Newton: no coordinate moves more than 1 SD per step
            b = b + step
            if np.abs(step).max() < 1e-6:
                break
        self.b = b
        # Breslow baseline cumulative hazard at HORIZON
        eta = Z @ b; r = w * np.exp(eta)
        S0 = np.cumsum(r)[np.searchsorted(-t, -t, side='right') - 1]
        self.H0 = float(((w * e) / S0)[t <= HORIZON].sum())
        self.loglik = float((w * e * (eta - np.log(S0))).sum())
        return self

    def lp(self, X):
        return ((np.asarray(X, float) - self.m) / self.s) @ self.b

    def risk(self, X):
        return 1 - np.exp(-self.H0 * np.exp(self.lp(X)))


def partial_loglik(lp, t, e):
    o = np.argsort(-t, kind='stable'); lp, t, e = lp[o], t[o], e[o]
    S0 = np.cumsum(np.exp(lp - lp.max()))[np.searchsorted(-t, -t, side='right') - 1]
    return float((e * (lp - lp.max() - np.log(S0))).sum())


def cindex(lp, t, e, w=None):
    """Harrell's C: pairs (i, j) with t_i < t_j and an event at t_i; ties in lp count one half. Optional pair weights w_i * w_j."""
    lp, t, e = np.asarray(lp, float), np.asarray(t, float), np.asarray(e, bool)
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    num = den = 0.0
    idx = np.flatnonzero(e)
    for ch in np.array_split(idx, max(1, len(idx) // 200)):
        later = t[None, :] > t[ch, None]
        ww = w[ch, None] * w[None, :] * later
        d = lp[ch, None] - lp[None, :]
        num += float((ww * ((d > 0) + 0.5 * (d == 0))).sum()); den += float(ww.sum())
    return num / den


# ---------------- selection ----------------
def gain_ll(Z, t, e, feats, lam, folds):
    """Cross-validated Cox partial log-likelihood of base + feats (sum over folds of the held-out fold's log-likelihood)."""
    cols = ['age', 'female'] + feats
    X = Z[cols].to_numpy(); s = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(t)), te)
        m = Cox(lam).fit(X[tr], t[tr], e[tr])
        s += partial_loglik(m.lp(X[te]), t[te], e[te])
    return s


def beam_budget(Z, t, e, budget, lam=1.0, folds=None, width=4, pool=None, cache=None, start=()):
    """Target-aware cost-aware beam search: states are item sets within budget, scored by cross-validated Cox log-likelihood;
    each round extends every kept state by one affordable item and keeps the best `width`. Returns (items, cost, score)."""
    pool = list(ITEMS) if pool is None else pool
    cache = {} if cache is None else cache

    def score(items):
        key = tuple(sorted(features_of(items)))
        if key not in cache:
            cache[key] = gain_ll(Z, t, e, list(key), lam, folds)
        return cache[key]
    beam = [(tuple(sorted(start)), score(list(start)))]
    best = beam[0]
    while True:
        cand = {}
        for items, s in beam:
            have = set(features_of(items))
            for it in pool:
                if it in items or set(ITEMS[it]) <= have:
                    continue
                new = tuple(sorted(items + (it,)))
                if new in cand or cost(new) > budget + 1e-9:
                    continue
                cand[new] = score(new)
        if not cand:
            break
        beam = sorted(cand.items(), key=lambda kv: -kv[1])[:width]
        if beam[0][1] > best[1] + 1e-9:
            best = beam[0]
        elif all(s <= best[1] + 1e-9 for _, s in beam):
            break
    items = list(best[0])
    return items, (cost(items) if items else 0.0), best[1]


def select_budget(Z, t, e, budget, lam=1.0, folds=None, pool=None):
    """Target-aware cost-aware greedy: repeatedly adds the orderable item with the largest cross-validated log-likelihood gain per extra dollar
    that keeps the total within budget; stops when no affordable item gains. Returns the item list and its cost."""
    pool = list(ITEMS) if pool is None else pool
    chosen, cur = [], gain_ll(Z, t, e, [], lam, folds)
    while True:
        best, best_val = None, 0.0
        c0 = cost(chosen) if chosen else 0.0
        for it in pool:
            if it in chosen or set(ITEMS[it]) <= set(features_of(chosen)):
                continue
            c1 = cost(chosen + [it])
            if c1 > budget + 1e-9:
                continue
            g = gain_ll(Z, t, e, features_of(chosen + [it]), lam, folds) - cur
            val = g / max(c1 - c0, 0.5)
            if g > 0 and val > best_val:
                best, best_val, best_g = it, val, g
        if best is None:
            return chosen, (cost(chosen) if chosen else 0.0)
        chosen.append(best); cur += best_g


# ---------------- sequential ordering ----------------
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class FlipModel:
    """P(the decision made with the next stage differs from the current decision | current risk), logistic in a quadratic of logit risk
    and of its distance to the threshold; fitted on cross-fitted training predictions."""
    def __init__(self, tau):
        self.tau = tau

    def design(self, r):
        u = logit(r); d = u - logit(self.tau)
        return np.column_stack([np.ones(len(r)), d, d ** 2, np.abs(d)])

    def fit(self, r_now, r_next):
        X = self.design(r_now); y = ((r_now >= self.tau) != (r_next >= self.tau)).astype(float)
        b = np.zeros(X.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
            step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
            b += step
            if np.abs(step).max() < 1e-8:
                break
        self.b = b
        return self

    def prob(self, r):
        return 1 / (1 + np.exp(-self.design(r) @ self.b))


def crossfit_risk(Z, t, e, stages, folds, lam=1.0):
    """Out-of-fold 9-year risk for each stage's feature list (a list of feature lists)."""
    out = np.zeros((len(stages), len(t)))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(t)), te)
        for s, f in enumerate(stages):
            cols = ['age', 'female'] + f
            m = Cox(lam).fit(Z[cols].to_numpy()[tr], t[tr], e[tr])
            out[s, te] = m.risk(Z[cols].to_numpy()[te])
    return out


def sequential_policy(r_test, flips, eps, stage_costs):
    """r_test: (stages, n) risks; flips[s] predicts a flip between stage s and the last stage. Each person moves to the next stage while the
    predicted flip probability is at least eps. Returns the stage reached, the risk used and the cost paid per person."""
    S, n = r_test.shape
    stage = np.zeros(n, int); active = np.ones(n, bool)
    for s in range(S - 1):
        go = active & (flips[s].prob(r_test[s]) >= eps)
        stage[go] = s + 1; active = go
    risk = r_test[stage, np.arange(n)]
    return stage, risk, np.asarray(stage_costs)[stage]


# ---------------- clinical base (sensitivity analysis) ----------------
CLIN_SFX = {'A': '', 'B': '_B', 'C': '_C', 'D': '_D', 'E': '_E', 'F': '_F'}
CLIN = ['bmi', 'sbp', 'smoker', 'diabetes']


def clinical(x):
    """BMI, mean of available systolic readings, current smoker (SMQ040 1 or 2; never smokers coded 0), self-reported diabetes (DIQ010 = 1).
    Returns a frame aligned to x (NaN where missing)."""
    out = []
    for c, g in x.groupby('cycle'):
        d = LAYER[c] / c; s = CLIN_SFX[c]
        read = lambda f: pd.read_sas(d / f'{f}{s}.xpt', format='xport').set_index('SEQN')  # noqa: E731
        seqn = pd.Index([int(i[1:]) for i in g.index])
        bmx, bpx, smq, diq = read('BMX'), read('BPX'), read('SMQ'), read('DIQ')
        f = pd.DataFrame(index=seqn)
        f['bmi'] = bmx.BMXBMI.reindex(seqn)
        f['sbp'] = bpx[[v for v in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if v in bpx]].replace(0, np.nan).mean(1).reindex(seqn)
        ever, now = smq.SMQ020.reindex(seqn), smq.SMQ040.reindex(seqn)
        f['smoker'] = np.where(ever == 2, 0.0, np.where(now.isin([1, 2]), 1.0, np.where(now == 3, 0.0, np.nan)))
        dq = diq.DIQ010.reindex(seqn)
        f['diabetes'] = np.where(dq == 1, 1.0, np.where(dq.isin([2, 3]), 0.0, np.nan))
        f.index = g.index
        out.append(f)
    return pd.concat(out).reindex(x.index)


# ---------------- evaluation helpers ----------------
def net_benefit(risk, t, e, tau, w=None):
    """Net benefit at threshold tau among people whose 9-year status is known (died by 9 years or followed 9 years)."""
    known = e | (t >= HORIZON)
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    hi = risk >= tau; W = w[known].sum()
    return float((w * (hi & e & known)).sum() / W - (w * (hi & ~e & known)).sum() / W * tau / (1 - tau))


def calibration(risk, t, e):
    """Observed/expected 9-year deaths among people with known status, and the calibration slope of logit risk."""
    known = e | (t >= HORIZON)
    r, y = risk[known], e[known].astype(float)
    u = logit(r); X = np.column_stack([np.ones(len(u)), u]); b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
        step = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8:
            break
    return float(y.sum() / r.sum()), float(b[1])
