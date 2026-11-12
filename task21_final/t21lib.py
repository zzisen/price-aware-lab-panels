#!/usr/bin/env python3
"""Task 21 (unified final analysis) library, on top of dmlib / dm2lib / t20lib (copied unchanged).
New pieces: NHANES 2015-2016 loader (hs-CRP), income/education groups, restricted-cubic-spline Cox (model-class sensitivity),
price-weighted lasso Cox path (cost-sensitive comparator), price perturbation and Australian coning, decision-analytic health-economic
model, random-effects pooling."""
import itertools
import numpy as np
import pandas as pd
import dmlib as L
import dm2lib as M
import t20lib as T

E = L.EXT


# ---------------- data: NHANES 2015-2016 ----------------
def load_i():
    d = E / 'nhanes_i' / 'I'
    read = lambda s: pd.read_sas(d / f'{s}_I.xpt', format='xport').set_index('SEQN')  # noqa: E731
    demo = read('DEMO')[['RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'WTMEC2YR']]; demo.columns = ['age', 'sex', 'race', 'weight']
    counts = {'cycle': 'I', 'in_file': len(demo)}
    demo = demo[(demo.age >= L.AGE[0]) & (demo.age <= L.AGE[1])]; counts['age_40_79'] = len(demo)
    bio = read('BIOPRO'); cols = {a: bio[v] for a, (s, v, _) in L.CHEM.items() if s == 'bio'}
    cols['tc'] = read('TCHOL')['LBXTC']; cols['hdl'] = read('HDL')['LBDHDD']
    cbc = read('CBC')
    for a in ['wbc', 'lymph_pct', 'mono_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']:
        cols[a] = cbc[L.CBC[a]]
    cols['gran_pct'] = cbc['LBXNEPCT'] + cbc['LBXEOPCT'] + cbc['LBXBAPCT']
    cols['hba1c'] = read('GHB')['LBXGH']; cols['crp'] = read('HSCRP')['LBXHSCRP'] / 10.0   # mg/L -> mg/dL
    u = read('ALB_CR'); cols['acr'] = u['URXUMA'] / u['URXUCR']
    lab = pd.DataFrame(cols)
    mort = L.read_mort(d / 'NHANES_2015_2016_MORT_2019_PUBLIC.dat')
    x = demo.join(mort, how='inner').dropna(subset=['months']); counts['mortality_eligible'] = len(x)
    x = x.join(lab, how='inner').dropna(subset=M.FEATURES); counts['complete'] = len(x)
    x['cycle'] = 'I'; x.index = [f'I{int(i)}' for i in x.index]
    return M.harmonise(x), counts


def clinical_i(x):
    d = E / 'nhanes_i' / 'I'
    read = lambda f: pd.read_sas(d / f'{f}_I.xpt', format='xport').set_index('SEQN')  # noqa: E731
    seqn = pd.Index([int(i[1:]) for i in x.index])
    bmx, bpx, smq, diq = read('BMX'), read('BPX'), read('SMQ'), read('DIQ')
    f = pd.DataFrame(index=seqn)
    f['bmi'] = bmx.BMXBMI.reindex(seqn)
    f['sbp'] = bpx[[v for v in ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4'] if v in bpx]].replace(0, np.nan).mean(1).reindex(seqn)
    ever, now = smq.SMQ020.reindex(seqn), smq.SMQ040.reindex(seqn)
    f['smoker'] = np.where(ever == 2, 0.0, np.where(now.isin([1, 2]), 1.0, np.where(now == 3, 0.0, np.nan)))
    dq = diq.DIQ010.reindex(seqn); f['diabetes'] = np.where(dq == 1, 1.0, np.where(dq.isin([2, 3]), 0.0, np.nan))
    f.index = x.index
    return f


# ---------------- income and education ----------------
def ses(x):
    """Poverty-income ratio group (<1.3, 1.3-3.5, >=3.5) and education group (<high school, high school, >high school)."""
    out = []
    for c, g in x.groupby('cycle'):
        if c == 'N3':
            lab = M.read_fixed('lab', ['SEQN', 'DMPPIR']); ad = M.read_fixed('adult', ['SEQN', 'HFA8R'])
            seqn = pd.Index([int(i[2:]) for i in g.index])
            pir = lab.DMPPIR.reindex(seqn).to_numpy(); yrs = ad.HFA8R.reindex(seqn).to_numpy()
            edu = np.where(yrs < 12, 0, np.where(yrs == 12, 1, np.where((yrs > 12) & (yrs <= 17), 2, np.nan)))
        else:
            folder = {'G': E / 'nhanes_gh' / 'G', 'H': E / 'nhanes_gh' / 'H', 'I': E / 'nhanes_i' / 'I'}.get(c, L.LAYER.get(c, L.DEV) / c)
            name = {'G': 'DEMO_G', 'H': 'DEMO_H', 'I': 'DEMO_I'}.get(c, L.FILES[c]['demo'] if c in L.FILES else None)
            dm = pd.read_sas(folder / f'{name}.xpt', format='xport').set_index('SEQN')
            seqn = pd.Index([int(i[1:]) for i in g.index])
            pir = dm.INDFMPIR.reindex(seqn).to_numpy(); ed = dm.DMDEDUC2.reindex(seqn).to_numpy()
            edu = np.where(np.isin(ed, [1, 2]), 0, np.where(ed == 3, 1, np.where(np.isin(ed, [4, 5]), 2, np.nan)))
        inc = np.where(pir < 1.3, 0, np.where(pir < 3.5, 1, np.where(pir >= 3.5, 2, np.nan)))
        out.append(pd.DataFrame({'income_grp': inc, 'edu_grp': edu}, index=g.index))
    return pd.concat(out).reindex(x.index)


INC = {0: 'income <1.3x poverty', 1: 'income 1.3-3.5x', 2: 'income >=3.5x'}
EDU = {0: 'below high school', 1: 'high school', 2: 'above high school'}


# ---------------- restricted cubic spline Cox ----------------
def rcs_knots(v):
    return np.quantile(v, [0.10, 0.50, 0.90])


def rcs(v, k):
    """Linear term plus one restricted cubic spline term (3 knots, Harrell's parametrisation)."""
    t1, t2, t3 = k
    p = lambda u: np.maximum(u, 0) ** 3  # noqa: E731
    s = (p(v - t1) - p(v - t2) * (t3 - t1) / (t3 - t2) + p(v - t3) * (t2 - t1) / (t3 - t2)) / (t3 - t1) ** 2
    return np.column_stack([v, s])


def spline_design(Ztr, Zte, cols):
    A, B = [], []
    for c in cols:
        if c in ('female', 'smoker', 'diabetes'):
            A.append(Ztr[c].to_numpy()[:, None]); B.append(Zte[c].to_numpy()[:, None]); continue
        k = rcs_knots(Ztr[c].to_numpy())
        A.append(rcs(Ztr[c].to_numpy(), k)); B.append(rcs(Zte[c].to_numpy(), k))
    return np.hstack(A), np.hstack(B)


# ---------------- price-weighted lasso Cox (cost-sensitive comparator) ----------------
def feature_price_weights(feats):
    """Marginal US price per analyte: individual chemistry price; CBC price shared over its indices; single tests at their price."""
    w = []
    for f in feats:
        if f in L.CHEM:
            w.append(L.CHEM[f][2])
        elif f in M.CBC2:
            w.append(7.77 / len(M.CBC2))
        else:
            w.append({'hba1c': 9.71, 'crp': 12.95, 'acr': 10.96}[f])
    w = np.array(w)
    return w / w.mean()


def _cox_grad(Z, t, e, b):
    o = np.argsort(-t, kind='stable'); Zs, ts, es = Z[o], t[o], e[o]
    eta = Zs @ b; r = np.exp(eta - eta.max())
    last = np.searchsorted(-ts, -ts, side='right') - 1
    S0 = np.cumsum(r)[last]; S1 = np.cumsum(r[:, None] * Zs, 0)[last]
    ll = float((es * (eta - eta.max() - np.log(S0))).sum())
    g = (es[:, None] * (Zs - S1 / S0[:, None])).sum(0)
    return -ll / len(t), -g / len(t)


def lasso_path(Z, t, e, w, n_lambda=40, ratio=0.005, max_iter=400):
    """Weighted-L1 Cox by proximal gradient with backtracking, warm starts down a lambda path. Unpenalised columns: w = 0.
    Returns a list of (lambda, coefficient vector)."""
    p = Z.shape[1]; b = np.zeros(p)
    _, g0 = _cox_grad(Z, t, e, b)
    pen = w > 0
    lmax = float(np.max(np.abs(g0[pen]) / w[pen]))
    out, step = [], 1.0
    for lam in np.geomspace(lmax, lmax * ratio, n_lambda):
        for _ in range(max_iter):
            f, g = _cox_grad(Z, t, e, b)
            while True:
                z = b - step * g
                bn = np.sign(z) * np.maximum(np.abs(z) - step * lam * w, 0)
                fn, _ = _cox_grad(Z, t, e, bn)
                d = bn - b
                if fn <= f + g @ d + (d @ d) / (2 * step) + 1e-12:
                    break
                step *= 0.5
            if np.max(np.abs(d)) < 1e-6:
                b = bn; break
            b = bn; step *= 1.2
        out.append((lam, b.copy()))
    return out


# ---------------- price perturbation and coning ----------------
def make_cost_us(scale):
    """US Medicare bill with every component price multiplied by scale[component] (dict; missing keys = 1)."""
    s = lambda k: scale.get(k, 1.0)  # noqa: E731
    chem = {a: L.CHEM[a][2] * s(a) for a in L.CHEM}
    panels = {p: (m, pr * s(p)) for p, (m, pr) in L.PANELS.items()}

    def f(feats):
        f_ = set(feats); ch = f_ & set(chem); best = np.inf
        for use in itertools.product([0, 1], repeat=len(panels)):
            cov, pr = set(), 0.0
            for on, (m, p_) in zip(use, panels.values()):
                if on:
                    cov |= set(m); pr += p_
            best = min(best, pr + sum(chem[a] for a in ch - cov))
        tot = best + (7.77 * s('CBC') if f_ & set(M.CBC2) else 0) + (9.71 * s('hba1c') if 'hba1c' in f_ else 0) + (12.95 * s('crp') if 'crp' in f_ else 0) \
            + (10.96 * s('acr') if 'acr' in f_ else 0)
        return round(tot + (9.34 * s('veni') if f_ & M.BLOOD else 0), 2)
    return f


COMPONENTS = list(L.CHEM) + list(L.PANELS) + ['CBC', 'hba1c', 'crp', 'acr', 'veni']


def cost_au_coned(feats):
    """Australian MBS with the general-practitioner 'coning' rule: only the three most expensive items are paid (plus the episode fee)."""
    f = set(feats); n = len(f & M.AU_66500)
    items = [([0, 9.70, 11.65, 13.65, 15.65][n] if n < 5 else 17.70)] if n else []
    items += [11.05] * ('hdl' in f) + [32.55] * ('iron' in f) + [17.80] * bool(f & set(M.CBC2)) + [16.80] * ('hba1c' in f) + [20.10] * ('acr' in f)
    return round(sum(sorted(items, reverse=True)[:3]) + (5.95 if f else 0), 2)


def cheapest_matching(cache, thr, price):
    ok = sorted(((price(k), -v), k) for k, v in cache.items() if v >= thr)
    return ok[0][1] if ok else None


def best_within(cache, budget, price):
    return max(((v, k) for k, v in cache.items() if price(k) <= budget + 1e-9), default=(None, None))[1]


# ---------------- health economics ----------------
HE_BASE = dict(intervention_cost=300.0, rrr=0.05, ly_per_death_averted=8.0, utility=0.80, discount=0.03, wtp=100000.0)
HE_RANGE = dict(intervention_cost=(100.0, 600.0), rrr=(0.0, 0.15), ly_per_death_averted=(5.0, 12.0), utility=(0.70, 0.90))


def discounted_ly(ly, r):
    return float(sum((1 + r) ** -k for k in range(int(round(ly)))))


def he_eval(flag, died, test_cost, par):
    """Decision-analytic model per person screened: everyone flagged receives the preventive intervention; among flagged people who
    would have died within the horizon, a fraction rrr of deaths is averted, each averted death gaining discounted quality-adjusted
    life-years. Returns cost, QALY and net monetary benefit per person."""
    flag = np.asarray(flag, bool); died = np.asarray(died, bool)
    cost = float(np.mean(test_cost) + par['intervention_cost'] * flag.mean())
    qaly = float(par['rrr'] * (flag & died).mean() * discounted_ly(par['ly_per_death_averted'], par['discount']) * par['utility'])
    return dict(cost=cost, qaly=qaly, nmb=qaly * par['wtp'] - cost)


def he_psa(flags, died, costs, n=1000, seed=20261107, wtp=100000.0):
    """Probabilistic sensitivity analysis: parameters drawn uniformly within HE_RANGE. flags/costs: dict strategy -> arrays.
    Returns per-draw NMB per strategy and the probability each strategy has the highest NMB."""
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        par = dict(HE_BASE, wtp=wtp, **{k: rng.uniform(*v) for k, v in HE_RANGE.items()})
        rows.append({k: he_eval(flags[k], died, costs[k], par)['nmb'] for k in flags})
    d = pd.DataFrame(rows)
    return d, d.idxmax(axis=1).value_counts(normalize=True).reindex(d.columns).fillna(0.0)


# ---------------- random-effects pooling ----------------
def pool(est, lo, hi):
    """DerSimonian-Laird random-effects pooling of differences with 95% intervals (SE from interval width).
    Returns pooled estimate, 95% CI, tau^2, I^2 and the 95% prediction interval."""
    est = np.asarray(est, float); se = (np.asarray(hi) - np.asarray(lo)) / (2 * 1.96)
    w = 1 / se ** 2; fe = (w * est).sum() / w.sum(); Q = (w * (est - fe) ** 2).sum(); k = len(est)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    ws = 1 / (se ** 2 + tau2); m = (ws * est).sum() / ws.sum(); sm = np.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    from scipy.stats import t as tdist
    pi = tdist.ppf(0.975, max(k - 2, 1)) * np.sqrt(tau2 + sm ** 2) if k > 2 else np.nan
    return dict(k=k, est=float(m), lo=float(m - 1.96 * sm), hi=float(m + 1.96 * sm), tau2=float(tau2), I2=float(I2),
                pi_lo=float(m - pi) if k > 2 else np.nan, pi_hi=float(m + pi) if k > 2 else np.nan)
