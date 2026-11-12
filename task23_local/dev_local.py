#!/usr/bin/env python3
"""Task 23 feasibility (training data only, NHANES 1999-2010; no validation cohort is touched).
Question: a 'local site' has n local people. Should it (A) import the large model as is, (B) import and recalibrate locally (slope + local
baseline hazard), (C) refit the model on local data alone (ridge, penalty chosen by local cross-validation), or (D) borrow strength:
local model = slope x imported linear predictor + ridge-penalised local corrections, penalty chosen by local cross-validation?
Sites: each survey cycle (A-F) and, as a stress test with real population differences, three race/ethnicity groups.
For each site the large model is fitted on everyone outside the site; local samples of size n are drawn from the site and the rest of
the site is the test set. Outcome: 9-year all-cause death. Panels: routine, matched, target-aware (US prices, fixed).
Usage: dev_local.py <site> [<site> ...] [--reps R] [--ns 100,200,...] [--tag name]     Output: local_<site>.csv"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task21_final'))
import dm2lib as M  # noqa: E402
M.L.low_priority()

opt = {sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a.startswith('--')}
args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in opt]
REPS = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 10
NS = [int(v) for v in sys.argv[sys.argv.index('--ns') + 1].split(',')] if '--ns' in sys.argv else [100, 200, 500, 1000, 2000]; LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]; TAU = 0.10; H = M.HORIZON
PANELS = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'matched': M.features_of(['CBC', 'CMP', 'phosphorus']),
          'target-aware': M.features_of(['CBC', 'CMP', 'acr', 'ggt', 'phosphorus'])}
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); T, E = M.outcome(x)
SITES = {c: (x.cycle == c).to_numpy() for c in 'ABCDEF'}
SITES.update({'Mexican American': (x.race == 1).to_numpy(), 'Non-Hispanic Black': (x.race == 4).to_numpy(), 'Non-Hispanic White': (x.race == 3).to_numpy()})


# ---------------- Cox with offset and ridge on all but the first `free` columns ----------------
def _prep(t):
    o = np.argsort(-t, kind='stable'); last = np.searchsorted(-t[o], -t[o], side='right') - 1
    return o, last


def loglik(b, X, t, e, off):
    o, last = _prep(t); eta = (X @ b + off)[o]; c = eta.max()
    S0 = np.cumsum(np.exp(eta - c))[last]
    return float((e[o] * (eta - c - np.log(S0))).sum())


def fit(X, t, e, lam, off=None, free=0, b0=None):
    n, p = X.shape; off = np.zeros(n) if off is None else off
    o, last = _prep(t); Xo, eo, oo = X[o], e[o].astype(float), off[o]
    pen = np.full(p, lam); pen[:free] = 1e-6
    b = np.zeros(p) if b0 is None else b0.copy()
    for _ in range(40):
        eta = Xo @ b + oo; r = np.exp(eta - eta.max())
        S0 = np.cumsum(r)[last]; S1 = np.cumsum(r[:, None] * Xo, 0)[last]; mu = S1 / S0[:, None]
        g = (eo[:, None] * (Xo - mu)).sum(0) - pen * b
        S2 = np.cumsum(r[:, None, None] * Xo[:, :, None] * Xo[:, None, :], 0)[last]
        Hm = (eo[:, None, None] * (S2 / S0[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + np.diag(pen)
        st = np.linalg.solve(Hm, g); st *= min(1.0, 1.0 / max(np.abs(st).max(), 1e-12)); b = b + st
        if np.abs(st).max() < 1e-6:
            break
    return b


def breslow_H(lp, t, e, h):
    o, last = _prep(t); S0 = np.cumsum(np.exp(lp[o]))[last]
    return float((e[o] / S0)[t[o] <= h].sum())


def cv_lambda(X, t, e, off=None, free=0, k=3, seed=0):
    """Verweij-van Houwelingen cross-validated partial log-likelihood; returns the best penalty."""
    n = len(t); off = np.zeros(n) if off is None else off
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k); best, bl = -np.inf, LAMS[-1]
    for lam in LAMS:
        s = 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            b = fit(X[tr], t[tr], e[tr], lam, off[tr], free)
            s += loglik(b, X, t, e, off) - loglik(b, X[tr], t[tr], e[tr], off[tr])
        if s > best:
            best, bl = s, lam
    return bl


def evaluate(lp, risk, t, e):
    known = e | (t >= H); hi = risk >= TAU; W = known.sum()
    nb = float((hi & e & known).sum() / W - (hi & ~e & known).sum() / W * TAU / (1 - TAU))
    oe = float(e[known].sum() / max(risk[known].sum(), 1e-9))
    return dict(C=M.cindex(lp, t, e), OE=oe, abs_log_OE=abs(np.log(max(oe, 1e-9))), NB=nb)


rows = []
for site in args:
    ins = SITES[site]; out = ~ins
    for pname, feats in PANELS.items():
        cols = M.BASE + feats
        big = M.Cox(1.0).fit(Z.loc[out, cols].to_numpy(), T[out], E[out])
        idx_site = np.flatnonzero(ins)
        for n in NS:
            if n > len(idx_site) - 300:
                continue
            for rep in range(REPS):
                rng = np.random.default_rng(1000 * rep + n)
                loc = rng.choice(idx_site, n, replace=False); tst = np.setdiff1d(idx_site, loc)
                Xl, Xt = Z.iloc[loc][cols].to_numpy(), Z.iloc[tst][cols].to_numpy(); tl, el, tt, et = T[loc], E[loc], T[tst], E[tst]
                if el.sum() < 5:
                    continue
                m, s = Xl.mean(0), Xl.std(0); s[s == 0] = 1; Sl, St = (Xl - m) / s, (Xt - m) / s
                oL, oT = big.lp(Xl), big.lp(Xt)
                res = {}
                # A: import as is
                res['A import'] = evaluate(oT, big.risk(Xt), tt, et)
                # B: import, recalibrate slope and baseline hazard locally
                a = fit(oL[:, None], tl, el, 0.0, free=1)[0]; H0 = breslow_H(a * oL, tl, el, H)
                res['B recalibrate'] = evaluate(a * oT, 1 - np.exp(-H0 * np.exp(a * oT)), tt, et)
                # B0: import, re-estimate the baseline hazard only (slope fixed at 1)
                H0 = breslow_H(oL, tl, el, H)
                res['B0 baseline only'] = evaluate(oT, 1 - np.exp(-H0 * np.exp(oT)), tt, et)
                # T: test, then recalibrate only if the local O/E of the imported model differs from 1 (two-sided 5%)
                kl = el | (tl >= H); Ol = el[kl].sum(); El = big.risk(Xl)[kl].sum()
                use_b = abs(np.log(Ol / El)) > 1.96 / np.sqrt(Ol)
                res['T test then recalibrate'] = dict(res['B recalibrate'] if use_b else res['A import'], recal=int(use_b))
                if '--ab' in sys.argv:
                    for k, v in res.items():
                        rows.append(dict(site=site, panel=pname, n=n, rep=rep, local_deaths=int(el.sum()), method=k, **v))
                    continue
                # C: local model only
                lam = cv_lambda(Sl, tl, el, seed=rep); b = fit(Sl, tl, el, lam); H0 = breslow_H(Sl @ b, tl, el, H)
                res['C local only'] = dict(evaluate(St @ b, 1 - np.exp(-H0 * np.exp(St @ b)), tt, et), lam=lam)
                # D: borrow strength: slope on imported predictor (free) + penalised local corrections
                XD, XDt = np.column_stack([oL, Sl]), np.column_stack([oT, St])
                lam = cv_lambda(XD, tl, el, free=1, seed=rep); b = fit(XD, tl, el, lam, free=1, b0=np.r_[1.0, np.zeros(Sl.shape[1])])
                H0 = breslow_H(XD @ b, tl, el, H)
                res['D borrow strength'] = dict(evaluate(XDt @ b, 1 - np.exp(-H0 * np.exp(XDt @ b)), tt, et), lam=lam)
                for k, v in res.items():
                    rows.append(dict(site=site, panel=pname, n=n, rep=rep, local_deaths=int(el.sum()), method=k, **v))
            print(site, pname, n, 'done', flush=True)
    pd.DataFrame([r for r in rows if r['site'] == site]).to_csv(HERE / f"local_{site.replace(' ', '_')}{sys.argv[sys.argv.index('--tag') + 1] if '--tag' in sys.argv else ''}.csv", index=False)
