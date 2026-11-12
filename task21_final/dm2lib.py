#!/usr/bin/env python3
"""Task 19 library, built on dmlib (Task 18, copied unchanged).
- Harmonised features shared by NHANES 1999-2010 and NHANES III: granulocyte % = neutrophil + eosinophil + basophil %, CRP below 0.25 mg/dL set
  to 0.21 (NHANES III detection limit), NHANES III creatinine standardised (0.960 x - 0.184, NHANES III laboratory documentation).
- NHANES III fixed-width reader driven by the SAS INPUT statements; fields that are all 8s across their full width are "blank but applicable".
- Three price schedules: US Medicare CLFS 2026 Q4 (USD), Australian MBS pathology (AUD), Japanese medical fee schedule 2026 (JPY, 1 point = 10 yen).
- PhenoAge (Levine 2018), a multi-path sequential ordering policy, multiple-imputation helpers, rule-chart extraction and group metrics."""
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import dmlib as L
from dmlib import Cox, cindex, partial_loglik, net_benefit, calibration, logit, FlipModel, HORIZON, AGE  # noqa: F401

NH3 = L.EXT / 'nhanes3_holdout'
CHEM = list(L.CHEM)
CBC2 = ['wbc', 'lymph_pct', 'mono_pct', 'gran_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']
FEATURES = CHEM + CBC2 + ['hba1c', 'crp', 'acr']
ITEMS = {a: [a] for a in CHEM}
ITEMS.update({'CBC': CBC2, 'hba1c': ['hba1c'], 'crp': ['crp'], 'acr': ['acr']})
ITEMS.update({p: list(m) for p, (m, _) in L.PANELS.items()})
STANDARD = L.STANDARD
CRP_CUT, CRP_FLOOR = 0.25, 0.21
BASE = ['age', 'female']
CLIN = L.CLIN


def features_of(items):
    return list(dict.fromkeys(f for i in items for f in ITEMS[i]))


# ---------------- continuous NHANES (1999-2010) ----------------
def load_cont(c, keep_missing=False):
    """One continuous cycle, harmonised. keep_missing=True keeps people with missing analytes (for multiple imputation)."""
    f, d = L.FILES[c], L.LAYER[c] / c
    read = lambda s: pd.read_sas(d / f'{s}.xpt', format='xport').set_index('SEQN')  # noqa: E731
    demo = read(f['demo'])[['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'WTMEC2YR']]
    demo.columns = ['age', 'sex', 'race', 'weight']
    counts = {'cycle': c, 'in_file': len(demo)}
    demo = demo[(demo.age >= AGE[0]) & (demo.age <= AGE[1])]; counts['age_40_79'] = len(demo)
    cols = {}
    bio = read(f['bio'])
    for a, (src, v, _) in L.CHEM.items():
        if src == 'bio':
            cols[a] = bio[v] if v in bio else bio['LBD' + v[3:]]
    cols['tc'] = read(f['tc'])['LBXTC']
    h = read(f['hdl']); cols['hdl'] = h[next(v for v in ['LBDHDD', 'LBXHDD', 'LBDHDL'] if v in h)]
    cbc = read(f['cbc'])
    for a in ['wbc', 'lymph_pct', 'mono_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']:
        cols[a] = cbc[L.CBC[a]]
    cols['gran_pct'] = cbc['LBXNEPCT'] + cbc['LBXEOPCT'] + cbc['LBXBAPCT']
    cols['hba1c'] = read(f['ghb'])['LBXGH']
    cols['crp'] = read(f['crp'])['LBXCRP']
    u = read(f['acr']); cols['acr'] = u['URXUMA'] / u['URXUCR']
    lab = pd.DataFrame(cols)
    if c == 'A':
        lab['creatinine'] = 1.013 * lab.creatinine + 0.147
    if c == 'D':
        lab['creatinine'] = -0.016 + 0.978 * lab.creatinine
    mort = L.read_mort(d / f"NHANES_{f['mort']}_MORT_2019_PUBLIC.dat")
    x = demo.join(mort, how='inner'); counts['mortality_eligible'] = len(x)
    x = x.join(lab, how='left')
    x = x.dropna(subset=['months'])
    counts['complete'] = int(x[FEATURES].notna().all(1).sum())
    if not keep_missing:
        x = x.dropna(subset=FEATURES)
    else:   # people with a blood or urine sample: at least half of the analytes measured
        x = x[x[FEATURES].notna().mean(axis=1) >= 0.5]; counts['half_measured'] = len(x)
    x['cycle'] = c
    x.index = [f'{c}{int(i)}' for i in x.index]
    return harmonise(x), counts


def harmonise(x):
    x = x.copy()
    x['crp'] = np.where(x.crp < CRP_CUT, CRP_FLOOR, x.crp)
    return x


def load(cycles, keep_missing=False):
    parts = [load_cont(c, keep_missing) for c in cycles]
    return pd.concat([p for p, _ in parts]), [n for _, n in parts]


# ---------------- NHANES III ----------------
NH3_LAB = {'glucose': 'SGP', 'bun': 'BUP', 'creatinine': 'CEP', 'sodium': 'NAPSI', 'potassium': 'SKPSI', 'chloride': 'CLPSI', 'co2': 'C3PSI',
           'calcium': 'SCP', 'albumin': 'AMP', 'protein': 'TPP', 'alp': 'APPSI', 'alt': 'ATPSI', 'ast': 'ASPSI', 'bilirubin': 'TBP', 'ggt': 'GGPSI',
           'uric_acid': 'UAP', 'ldh': 'LDPSI', 'phosphorus': 'PSP', 'iron': 'FEP', 'tg': 'TGP', 'tc': 'TCP', 'hdl': 'HDP',
           'wbc': 'WCP', 'lymph_pct': 'LMPPCNT', 'mono_pct': 'MOPPCNT', 'gran_pct': 'GRPPCNT', 'hgb': 'HGP', 'mcv': 'MVPSI', 'rdw': 'RWP',
           'plt': 'PLP', 'mpv': 'PVPSI', 'hba1c': 'GHP', 'crp': 'CRP', 'ualb': 'UBP', 'ucr': 'URP'}
NH3_RACE = {1: 3, 2: 4, 3: 1, 4: 5}     # DMARETHN (1 NH white, 2 NH black, 3 Mexican American, 4 other) -> RIDRETH1 codes


def sas_layout(path):
    txt = path.read_text(errors='replace')
    body = txt[txt.index('INPUT'):]; body = body[:body.index(';')]
    return {m.group(1): (int(m.group(2)) - 1, int(m.group(3) or m.group(2)))
            for m in re.finditer(r'^\s+([A-Z0-9_]+)\s+\$?\s*(\d+)(?:-(\d+))?\s*$', body, re.M)}


def read_fixed(name, variables):
    lay = sas_layout(NH3 / f'{name}.sas')
    specs = [lay[v] for v in variables]
    d = pd.read_fwf(NH3 / f'{name}.dat', colspecs=specs, names=variables, dtype=str, header=None)
    for v, (a, b) in zip(variables, specs):
        s = d[v].fillna('').str.strip()
        miss = (s == '') | ((s.str.len() == b - a) & s.str.fullmatch('8+'))
        d[v] = pd.to_numeric(s.where(~miss), errors='coerce')
    return d.set_index('SEQN')


def load_nh3(keep_missing=False):
    """NHANES III adults 40-79 with eligible mortality, harmonised to the continuous-NHANES features. FINAL VALIDATION ONLY."""
    lab = read_fixed('lab', ['SEQN', 'HSAGEIR', 'HSSEX', 'DMARETHN', 'WTPFEX6'] + list(NH3_LAB.values()))
    counts = {'cycle': 'N3', 'in_file': len(lab)}
    x = pd.DataFrame(index=lab.index)
    x['age'], x['sex'], x['race'], x['weight'] = lab.HSAGEIR, lab.HSSEX, lab.DMARETHN.map(NH3_RACE), lab.WTPFEX6
    for a, v in NH3_LAB.items():
        x[a] = lab[v]
    x['creatinine'] = 0.960 * x.creatinine - 0.184
    x['acr'] = x.ualb / x.ucr
    x = x.drop(columns=['ualb', 'ucr'])
    x = x[(x.age >= AGE[0]) & (x.age <= AGE[1])]; counts['age_40_79'] = len(x)
    mort = L.read_mort(NH3 / 'NHANES_III_MORT_2019_PUBLIC.dat')
    x = x.join(mort, how='inner').dropna(subset=['months']); counts['mortality_eligible'] = len(x)
    counts['complete'] = int(x[FEATURES].notna().all(1).sum())
    if not keep_missing:
        x = x.dropna(subset=FEATURES)
    else:
        x = x[x[FEATURES].notna().mean(axis=1) >= 0.5]; counts['half_measured'] = len(x)
    x['cycle'] = 'N3'
    x.index = [f'N3{int(i)}' for i in x.index]
    return harmonise(x), counts


def clinical_nh3(x):
    ex = read_fixed('exam', ['SEQN', 'BMPBMI', 'PEPMNK1R']); ad = read_fixed('adult', ['SEQN', 'HAR1', 'HAR3', 'HAD1'])
    seqn = pd.Index([int(i[2:]) for i in x.index])
    f = pd.DataFrame(index=seqn)
    f['bmi'] = ex.BMPBMI.reindex(seqn); f['sbp'] = ex.PEPMNK1R.reindex(seqn)
    h1, h3 = ad.HAR1.reindex(seqn), ad.HAR3.reindex(seqn)
    f['smoker'] = np.where(h1 == 2, 0.0, np.where(h3 == 1, 1.0, np.where(h3 == 2, 0.0, np.nan)))
    dq = ad.HAD1.reindex(seqn)
    f['diabetes'] = np.where(dq == 1, 1.0, np.where(dq == 2, 0.0, np.nan))
    f.index = x.index
    return f


def clinical(x):
    parts = []
    if (x.cycle != 'N3').any():
        parts.append(L.clinical(x[x.cycle != 'N3']))
    if (x.cycle == 'N3').any():
        parts.append(clinical_nh3(x[x.cycle == 'N3']))
    return pd.concat(parts).reindex(x.index)


def transform(x, ref=None):
    ref = x if ref is None else ref
    z = pd.DataFrame(index=x.index)
    z['age'] = x.age.astype(float); z['female'] = (x.sex == 2).astype(float)
    for a in FEATURES:
        v, r = x[a].astype(float), ref[a].astype(float)
        z[a] = np.log(np.maximum(v, 0) + 0.01 * max(r[r > 0].median(), 1e-6))
    return z


def outcome(x, cause=None):
    if cause == 'other':
        t, e = L.outcome(x)
        return t, e & ~x.ucod.isin(['001', '005', '002']).to_numpy()
    return L.outcome(x, cause)


# ---------------- prices ----------------
BLOOD = set(CHEM) | set(CBC2) | {'hba1c', 'crp'}


def cost_us(f):
    f = set(f); chem = f & set(CHEM); best = np.inf
    for use in itertools.product([0, 1], repeat=len(L.PANELS)):
        cov, pr = set(), 0.0
        for on, (m, p) in zip(use, L.PANELS.values()):
            if on:
                cov |= set(m); pr += p
        best = min(best, pr + sum(L.CHEM[a][2] for a in chem - cov))
    tot = best + (7.77 if f & set(CBC2) else 0) + (9.71 if 'hba1c' in f else 0) + (12.95 if 'crp' in f else 0) + (5.78 + 5.18 if 'acr' in f else 0)
    return round(tot + (9.34 if f & BLOOD else 0), 2)


AU_66500 = set(CHEM) - {'hdl', 'iron'} | {'crp'}


def cost_au(f):
    f = set(f); n = len(f & AU_66500)
    tot = [0, 9.70, 11.65, 13.65, 15.65][n] if n < 5 else 17.70
    tot += (11.05 if 'hdl' in f else 0) + (32.55 if 'iron' in f else 0) + (17.80 if f & set(CBC2) else 0)
    tot += (16.80 if 'hba1c' in f else 0) + (20.10 if 'acr' in f else 0)
    return round(tot + (5.95 if f else 0), 2)


JP_11 = {'bilirubin', 'protein', 'albumin', 'bun', 'creatinine', 'uric_acid', 'alp', 'ggt', 'tg', 'potassium', 'calcium', 'glucose', 'ldh', 'iron'}
JP_17 = {'hdl', 'phosphorus', 'tc', 'ast', 'alt'}


def cost_jp(f):
    f = set(f)
    pts = [11] * len(f & JP_11) + [11] * bool(f & {'sodium', 'chloride'}) + [17] * len(f & JP_17)
    n = len(pts); chem = (93 if n <= 7 else 99 if n <= 9 else 103) if n >= 5 else sum(pts)
    chem += 48 if 'co2' in f else 0
    tot = chem + (144 if chem else 0)
    hem = (36 if f & set(CBC2) else 0) + (49 if 'hba1c' in f else 0); tot += hem + (125 if hem else 0)
    tot += (16 + 144 if 'crp' in f else 0) + (99 + 11 + 34 if 'acr' in f else 0)
    return float(10 * (tot + (40 if f & BLOOD else 0)))


SCHEDULES = {'US': cost_us, 'AU': cost_au, 'JP': cost_jp}


def cost(items, sched='US'):
    return SCHEDULES[sched](features_of(items)) if items else 0.0


# ---------------- selection (dmlib's beam with harmonised items and any schedule) ----------------
def gain_ll(Z, t, e, feats, lam, folds, base=BASE):
    X = Z[base + feats].to_numpy(); s = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(len(t)), te)
        s += partial_loglik(Cox(lam).fit(X[tr], t[tr], e[tr]).lp(X[te]), t[te], e[te])
    return s


def beam(Z, t, e, budget, folds, sched='US', lam=1.0, width=4, pool=None, cache=None, start=(), base=BASE):
    pool = list(ITEMS) if pool is None else pool
    cache = {} if cache is None else cache

    def score(items):
        key = tuple(sorted(features_of(items)))
        if key not in cache:
            cache[key] = gain_ll(Z, t, e, list(key), lam, folds, base)
        return cache[key]
    beam_ = [(tuple(sorted(start)), score(list(start)))]; best = beam_[0]
    while True:
        cand = {}
        for items, s in beam_:
            have = set(features_of(items))
            for it in pool:
                if it in items or set(ITEMS[it]) <= have:
                    continue
                new = tuple(sorted(items + (it,)))
                if new in cand or cost(new, sched) > budget + 1e-9:
                    continue
                cand[new] = score(new)
        if not cand:
            break
        beam_ = sorted(cand.items(), key=lambda kv: -kv[1])[:width]
        if beam_[0][1] > best[1] + 1e-9:
            best = beam_[0]
        elif all(s <= best[1] + 1e-9 for _, s in beam_):
            break
    items = list(best[0])
    return items, cost(items, sched), best[1]


# ---------------- PhenoAge (Levine et al., Aging 2018) ----------------
PHENO_FEATS = ['albumin', 'creatinine', 'glucose', 'crp', 'lymph_pct', 'mcv', 'rdw', 'alp', 'wbc']


def phenoage_xb(x):
    """Linear predictor of the PhenoAge mortality score; albumin g/L, creatinine umol/L, glucose mmol/L, ln CRP (mg/dL), WBC 1000/uL."""
    return (-19.907 - 0.0336 * 10 * x.albumin + 0.0095 * 88.42 * x.creatinine + 0.1953 * 0.0555 * x.glucose + 0.0954 * np.log(x.crp)
            - 0.0120 * x.lymph_pct + 0.0268 * x.mcv + 0.3306 * x.rdw + 0.00188 * x.alp + 0.0554 * x.wbc + 0.0804 * x.age).to_numpy(float)


def phenoage_years(xb):
    g = 0.0076927
    M = 1 - np.exp(-np.exp(xb) * (np.exp(120 * g) - 1) / g)
    return 141.50225 + np.log(-0.00553 * np.log(1 - M)) / 0.090165


# ---------------- multi-path sequential ordering ----------------
class PathPolicy:
    """Blocks are orderable item groups (e.g. CMP, CBC, acr); a state is the set of blocks done. In every state a person stops if the predicted
    probability that the final decision (all blocks) differs from the current one is below eps; otherwise the next block is the one with the
    largest predicted probability of changing the current decision per extra unit of cost. Flip models are fitted on cross-fitted training risks."""
    def __init__(self, blocks, tau, sched='US', base=BASE, lam=1.0):
        self.blocks, self.tau, self.sched, self.base, self.lam = list(blocks), tau, sched, base, lam
        self.states = [frozenset(s) for k in range(len(blocks) + 1) for s in itertools.combinations(self.blocks, k)]
        self.final = frozenset(self.blocks)

    @staticmethod
    def items(s):
        return sorted(i for blk in s for i in blk.split('+'))      # a block may join items with '+', e.g. 'CBC+acr'

    def cols(self, s):
        return self.base + features_of(self.items(s))

    def scost(self, s):
        return cost(self.items(s), self.sched)

    def fit(self, Z, t, e, folds):
        self.models = {s: Cox(self.lam).fit(Z[self.cols(s)].to_numpy(), t, e) for s in self.states}
        rcv = {s: np.zeros(len(t)) for s in self.states}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(t)), te)
            for s in self.states:
                X = Z[self.cols(s)].to_numpy()
                rcv[s][te] = Cox(self.lam).fit(X[tr], t[tr], e[tr]).risk(X[te])
        self.stop = {s: FlipModel(self.tau).fit(rcv[s], rcv[self.final]) for s in self.states if s != self.final}
        self.step = {(s, b): FlipModel(self.tau).fit(rcv[s], rcv[s | {b}]) for s in self.states if s != self.final for b in self.blocks if b not in s}
        return self

    def risks(self, Z):
        return {s: self.models[s].risk(Z[self.cols(s)].to_numpy()) for s in self.states}

    def run(self, R, eps):
        n = len(next(iter(R.values())))
        state = [frozenset()] * n
        active = np.ones(n, bool)
        for _ in range(len(self.blocks)):
            for s in set(state[i] for i in np.flatnonzero(active)):
                idx = np.flatnonzero(active & np.array([st == s for st in state]))
                if s == self.final:
                    active[idx] = False; continue
                go = self.stop[s].prob(R[s][idx]) >= eps
                active[idx[~go]] = False
                idx = idx[go]
                if not len(idx):
                    continue
                cands = [b for b in self.blocks if b not in s]
                val = np.column_stack([self.step[(s, b)].prob(R[s][idx]) / max(self.scost(s | {b}) - self.scost(s), 1e-9) for b in cands])
                pick = val.argmax(1)
                for i, j in zip(idx, pick):
                    state[i] = s | {cands[j]}
        risk = np.array([R[state[i]][i] for i in range(n)])
        cst = np.array([self.scost(state[i]) for i in range(n)])
        return state, risk, cst


class FixedPath(PathPolicy):
    """Task 18 policy: blocks in a fixed order (stop rule only)."""
    def run(self, R, eps):
        n = len(next(iter(R.values()))); state = [frozenset()] * n; active = np.ones(n, bool)
        for k, b in enumerate(self.blocks):
            s = frozenset(self.blocks[:k])
            idx = np.flatnonzero(active)
            go = self.stop[s].prob(R[s][idx]) >= eps
            active[idx[~go]] = False
            for i in idx[go]:
                state[i] = s | {b}
        risk = np.array([R[state[i]][i] for i in range(n)])
        return state, risk, np.array([self.scost(state[i]) for i in range(n)])


def interval(flip, eps, grid=np.linspace(0.0005, 0.9995, 19991)):
    """Risk values at which a flip model says 'go on' (probability >= eps), as a list of [low, high] intervals."""
    go = flip.prob(grid) >= eps
    out, start = [], None
    for r, g in zip(grid, go):
        if g and start is None:
            start = r
        if not g and start is not None:
            out.append([float(start), float(prev)]); start = None
        prev = r
    if start is not None:
        out.append([float(start), float(prev)])
    return out


# ---------------- multiple imputation ----------------
def impute(Z, t, e, m=5, seed=0, extra=()):
    """m imputations of the log analytes (and extra columns) with chained equations; event indicator and Nelson-Aalen cumulative hazard
    are included as predictors (White & Royston 2009). Returns a list of completed frames."""
    o = np.argsort(t); H = np.zeros(len(t)); n = len(t)
    at_risk = n - np.arange(n); H[o] = np.cumsum(e[o] / at_risk)
    cols = BASE + list(extra) + FEATURES
    A = Z[cols].copy(); A['_e'] = e.astype(float); A['_H'] = H
    out = []
    for k in range(m):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + k, skip_complete=True)
        B = pd.DataFrame(imp.fit_transform(A), index=A.index, columns=A.columns)
        out.append(B[cols])
    return out


# ---------------- group metrics ----------------
RACE = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other race'}


def groups(x):
    g = {'female': x.sex.to_numpy() == 2, 'male': x.sex.to_numpy() == 1, 'age40-59': x.age.to_numpy() < 60, 'age60-79': x.age.to_numpy() >= 60}
    for k, v in RACE.items():
        g[v] = x.race.to_numpy() == k
    return g


def decision_metrics(risk, t, e, tau):
    known = e | (t >= HORIZON); hi = risk >= tau
    return dict(sens=float(hi[known & e].mean()) if (known & e).any() else np.nan, spec=float((~hi[known & ~e]).mean()), flagged=float(hi.mean()))


def describe(feats):
    """Feature set -> orderable items (CBC, CMP, BMP, LIPID collapsed where complete)."""
    f, out = set(feats), []
    for name, members in [('CBC', CBC2), ('CMP', L.PANELS['CMP'][0]), ('BMP', L.PANELS['BMP'][0]), ('LIPID', L.PANELS['LIPID'][0])]:
        if set(members) <= f:
            out.append(name); f -= set(members)
    return out + sorted(f)


def best_affordable(cache, budget, sched):
    """Best cross-validated score among every feature set evaluated during the searches (cache) whose price is within budget.
    Makes the chosen panel monotone in budget: a panel affordable at a lower budget stays a candidate at higher budgets."""
    ok = [(s, k) for k, s in cache.items() if SCHEDULES[sched](k) <= budget + 1e-9]   # the empty set costs 0
    s, k = max(ok)
    items = describe(k) if k else []
    return items, cost(items, sched) if items else 0.0, s
