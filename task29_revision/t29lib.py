#!/usr/bin/env python3
"""Task 29 (post hoc revision analyses, PLAN.md) library: complete US fee menu, CoAI over every legal disjoint grouping, programme bundles,
censoring-aware net benefit and calibration slope. Built on the frozen libraries (dmlib / dm2lib / t20lib / t21lib), which are not modified."""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
for d in ['task21_final', 'task24_coai', 'task25_newdata', 'task27_ukhls']:
    sys.path.insert(0, str(W / d))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
L = M.L
CBC2 = list(M.CBC2)

# ---------------- complete US menu (CLFS 2026 Q4) ----------------
CHEM_PANELS = dict(L.PANELS)
CHEM_PANELS['ELEC'] = (['sodium', 'potassium', 'chloride', 'co2'], 7.01)                                     # 80051
CHEM_PANELS['RENAL'] = (['albumin', 'calcium', 'co2', 'chloride', 'creatinine', 'glucose', 'phosphorus', 'potassium', 'sodium', 'bun'], 8.68)  # 80069
CHEM_PANELS['HEPATIC'] = (['albumin', 'bilirubin', 'alp', 'protein', 'alt', 'ast'], 8.17)                     # 80076 (direct bilirubin not used)
CBC_ND = ['wbc', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']                                                          # 85027, no differential
P_CBC, P_CBCND, P_HGB = 7.77, 6.47, 2.37                                                                     # 85025, 85027, 85018
VENI = 9.34
_COMBOS = [c for c in itertools.product([0, 1], repeat=len(CHEM_PANELS))]


def chem_cost(chem):
    best, how = np.inf, None
    for use in _COMBOS:
        cov, pr, names = set(), 0.0, []
        for on, (nm, (m, p)) in zip(use, CHEM_PANELS.items()):
            if on:
                cov |= set(m); pr += p; names.append(nm)
        tot = pr + sum(L.CHEM[a][2] for a in chem - cov)
        if tot < best - 1e-9:
            best, how = tot, (names, sorted(chem - cov))
    return best, how


def cbc_cost(f):
    c = set(f) & set(CBC2)
    if not c:
        return 0.0, None
    if c <= {'hgb'}:
        return P_HGB, 'Hb'
    if c <= set(CBC_ND):
        return P_CBCND, 'CBC without differential'
    return P_CBC, 'CBC'


def cost_full(f):
    f = set(f); chem = f & set(L.CHEM)
    tot = (chem_cost(chem)[0] if chem else 0.0) + cbc_cost(f)[0]
    tot += (9.71 if 'hba1c' in f else 0) + (12.95 if 'crp' in f else 0) + (5.78 + 5.18 if 'acr' in f else 0)
    return round(tot + (VENI if f & M.BLOOD else 0), 2)


def order_full(f):
    """Readable order for a feature set under the complete menu."""
    f = set(f); chem = f & set(L.CHEM); out = []
    if chem:
        names, singles = chem_cost(chem)[1]; out += names + singles
    c = cbc_cost(f)[1]
    if c:
        out.append(c)
    out += [s for s in ['hba1c', 'crp', 'acr'] if s in f]
    return out


ITEMS_FULL = {'ELEC': CHEM_PANELS['ELEC'][0], 'RENAL': CHEM_PANELS['RENAL'][0], 'HEPATIC': CHEM_PANELS['HEPATIC'][0], 'CBCND': CBC_ND, 'HGB': ['hgb']}


def install_full(menu=None):
    """Register the new orderable items (restricted to a menu) and the price function as schedule 'USFULL'."""
    for k, v in ITEMS_FULL.items():
        vv = [a for a in v if menu is None or a in menu]
        if vv:
            M.ITEMS[k] = vv
    M.SCHEDULES['USFULL'] = cost_full


# ---------------- shared cross-validated score cache ----------------
def load_cache():
    """All earlier selections used the same score (5-fold CV Cox log-likelihood, lambda 1, same folds, same training data), so their caches
    are one function of the feature set. Returns dict sorted-feature-tuple -> score."""
    cache = {}
    for p in [W / 'task19_dm2' / 'selected_all_cache.csv', W / 'task20_chns' / 'selected_cn_cache.csv', HERE / 'cache_t29.csv']:
        if p.exists():
            d = pd.read_csv(p, keep_default_na=False)
            for fs, v in zip(d.features, d.cv):
                cache[tuple(sorted(str(fs).split()))] = float(v)
    return cache


def save_cache(cache):
    pd.DataFrame([dict(features=' '.join(k), cv=v) for k, v in cache.items()]).to_csv(HERE / 'cache_t29.csv', index=False)


# ---------------- CoAI over every legal disjoint grouping ----------------
def coai_partitions(menu):
    """Disjoint groupings a lab could bill: chemistry singly or through one panel family; lipids as panel or singly; blood count as full,
    without differential, or haemoglobin only. Restricted to the menu. Returns list of dict name -> (features, price)."""
    chem = [a for a in L.CHEM if a in menu]
    chem_choices = [[], ['BMP'], ['CMP'], ['ELEC'], ['HEPATIC'], ['ELEC', 'HEPATIC'], ['RENAL']]
    out = []
    for cc in chem_choices:
        for lip in [True, False]:
            for cb in ['CBC', 'CBCND', 'HGB']:
                g, used = {}, set()
                for p in cc:
                    m = [a for a in CHEM_PANELS[p][0] if a in menu]
                    if m:
                        g[p] = (m, CHEM_PANELS[p][1]); used |= set(m)
                lm = [a for a in ['tc', 'hdl', 'tg'] if a in menu]
                if lip and lm:
                    g['LIPID'] = (lm, CHEM_PANELS['LIPID'][1]); used |= set(lm)
                for a in chem:
                    if a not in used:
                        g[a] = ([a], L.CHEM[a][2])
                cm = [a for a in {'CBC': CBC2, 'CBCND': CBC_ND, 'HGB': ['hgb']}[cb] if a in menu]
                if cm:
                    g[cb] = (cm, {'CBC': P_CBC, 'CBCND': P_CBCND, 'HGB': P_HGB}[cb])
                for s, pr in [('hba1c', 9.71), ('crp', 12.95), ('acr', 10.96)]:
                    if s in menu:
                        g[s] = ([s], pr)
                out.append(g)
    return out


def coai_full(Z, y, menu, budgets, seed_mod):
    """CoAI importances from one base model on the whole menu; knapsack in every grouping; keep the grouping with the highest total
    importance whose billed price (complete menu) is within budget."""
    CO = seed_mod
    feats = [a for a in menu]
    cols = M.BASE + feats; X = Z[cols].to_numpy(float); X = (X - X.mean(0)) / X.std(0)
    m, best = CO.base_model(X, y)
    imp = dict(zip(cols, np.abs(m.coef_[0] * (X - X.mean(0))).sum(0).tolist()))
    res = []
    for B in budgets:
        cands = []
        for g in coai_partitions(menu):
            names = list(g); gv = np.array([sum(imp[f] for f in g[k][0]) for k in names]); gc = np.array([round(g[k][1] * 100) for k in names])
            sel = CO.knapsack(gv, gc, round((B - VENI) * 100)); f = sorted(set(x for i in sel for x in g[names[i]][0]))
            if cost_full(f) <= B + 1e-9:
                cands.append((float(gv[sel].sum()), -cost_full(f), f, [names[i] for i in sel]))
        v, c, f, items = max(cands, key=lambda z: (z[0], z[1]))
        res.append(dict(budget=B, features=f, items=items, billed_cost=-c, importance=v))
    return dict(hyper=dict(val_auc=best[0], **best[1]), panels=res)


# ---------------- programme bundles ----------------
# China, National Basic Public Health Service, annual health check for people aged 65+: blood count, liver function (ALT, AST, total bilirubin),
# kidney function (creatinine, urea), fasting glucose, lipids (TC, TG, LDL-C, HDL-C); urinalysis not measured in any cohort.
NBPHS = ['alt', 'ast', 'bilirubin', 'creatinine', 'bun', 'glucose', 'tc', 'tg', 'ldl', 'hdl', 'wbc']   # 'wbc' stands for the blood count
# England, NHS Health Check: total and HDL cholesterol (ordered as a lipid profile) and HbA1c.
NHSHC = ['tc', 'hdl', 'tg', 'hba1c']


def nbphs_price_full():
    """Chinese price of the whole NBPHS blood list as ordered (each analyte at its Chinese price, blood count, venipuncture)."""
    return round(sum(T.CN_PRICE[a] for a in NBPHS if a in T.CN_PRICE) + T.CN_CBC + T.CN_VENI, 2)


# ---------------- censoring-aware evaluation ----------------
def km_censor(t, e):
    """Reverse Kaplan-Meier G(s) = P(censoring time > s); returns a function giving G(s-) for an array s."""
    o = np.argsort(t, kind='stable'); ts, cs = t[o], (~e[o]).astype(float)
    uniq = np.unique(ts); G, g = [], 1.0
    for u in uniq:
        at = (ts >= u).sum(); d = cs[ts == u].sum()
        G.append(g); g *= 1 - d / at
    G = np.array(G)
    def before(s):
        i = np.searchsorted(uniq, s, side='left')   # G just before s: product over censoring times < s
        Gs = np.concatenate([G, [g]])
        return Gs[np.minimum(i, len(G))]
    return before


def ipcw(t, e, H):
    """Weights: deaths before H and people followed to H get 1/G(min(t,H)-); people censored alive before H get 0."""
    Gb = km_censor(t, e)
    died = e & (t <= H); fol = t >= H
    w = np.zeros(len(t)); s = np.minimum(t, H)
    k = died | fol; w[k] = 1.0 / np.maximum(Gb(s[k]), 1e-6)
    return w, died


def nb_ipcw(r, w, died, tau):
    hi = r >= tau; n = len(r)
    return float(((w * (hi & died)).sum() - (w * (hi & ~died)).sum() * tau / (1 - tau)) / n)


def nb_km(r, t, e, H, tau):
    """Vickers et al. 2008: NB = P(high) * [1 - S_high(H)] - P(high) * S_high(H) * tau / (1 - tau)."""
    hi = r >= tau
    if not hi.any():
        return 0.0
    tt, ee = t[hi], e[hi]; s = 1.0
    for u in np.unique(tt[ee & (tt <= H)]):
        s *= 1 - (ee & (tt == u)).sum() / (tt >= u).sum()
    p = hi.mean()
    return float(p * (1 - s) - p * s * tau / (1 - tau))


def slope_w(r, yy, w):
    k = w > 0; rr = np.clip(r[k], 1e-4, 1 - 1e-4); y_ = yy[k].astype(float); ww = w[k]
    u = np.log(rr / (1 - rr)); A = np.column_stack([np.ones(len(u)), u]); b = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-A @ b)); Wt = ww * p * (1 - p) + 1e-9; st = np.linalg.solve(A.T @ (Wt[:, None] * A), A.T @ (ww * (y_ - p))); b += st
        if np.abs(st).max() < 1e-9:
            break
    return float(b[1])


def wauc(s, yy, w):
    yy = yy.astype(bool); sn, wn = s[~yy], w[~yy]; o = np.argsort(sn); sn, wn = sn[o], wn[o]; cw = np.concatenate([[0.0], np.cumsum(wn)])
    lo, hi = np.searchsorted(sn, s[yy], 'left'), np.searchsorted(sn, s[yy], 'right')
    return float((w[yy] * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (w[yy].sum() * wn.sum()))


ci = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]  # noqa: E731
