#!/usr/bin/env python3
"""Task 23: local recalibration in each validation cohort (plan_t23_FROZEN.md). Usage: run_t23.py <NH3|GH|I|CHNS> [--dry]
Each cohort plays a 'new hospital'. The imported model is the Task 21 model (NHANES 1999-2010, ridge Cox lambda 1). In each repetition a
local sample of n people is drawn from the cohort; the rest of the cohort is the test set. Methods:
  A  import as is;
  B  recalibrate on the local sample (slope on the imported linear predictor + local baseline risk);
  T  test, then recalibrate: use B only if the imported model's local O/E differs from 1 (|log O/E| > 1.96 / sqrt(local deaths)), else A.
Outputs: t23_<c>_reps.csv (one row per panel x n x rep x method) and t23_<c>_summary.json.
--dry: train on cycle C, 'cohort' = cycle D (development check only)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; T21 = HERE.parent / 'task21_final'
sys.path.insert(0, str(T21))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]; DRY = '--dry' in sys.argv; PRE = f"{'dry_' if DRY else ''}t23_{COH}_"
NS = (200, 500, 1000); REPS = 50 if DRY else 200; SEED = 20261110; MIN_TEST = 1000; MIN_LOCAL_DEATHS = 10
BIN = COH == 'CHNS'

# ---------------- data and imported models (same as Task 21) ----------------
if BIN:
    SEL = json.load(open(T21.parent / 'task20_chns' / 'selected_cn.json'))
    if DRY:
        xtr, _ = M.load(['C']); d0, _ = M.load(['D']); td, ed = T.outcome_at(d0, 72); keep = ed | (td >= 72)
        xte = d0[keep].copy(); xte['died'] = ed[keep].astype(int)
    else:
        xtr, _ = M.load(list('ABCDEF')); xte, _ = T.load_chns()
    ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, T.SHARED); Z = T.transform_sub(xte, xtr, T.SHARED)
    y = xte.died.to_numpy().astype(bool)
    PANELS = {'matched': SEL['matched']['features'], 'routine': T.CN_ROUTINE_SHARED, 'aware@routine': SEL['aware_at_routine']['features']}
    H, TAU = 72, 0.10                     # outcome: death by the 2015 survey (about 6 years); decision: 9-year risk >= 10% (as Task 21)
else:
    SEL = json.load(open(T21.parent / 'task19_dm2' / 'selected_all.json'))
    H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]; TAU = {108: 0.10, 60: 0.05, 36: 0.03}[H]
    if DRY:
        xtr, _ = M.load(['C']); xte, _ = M.load(['D'])
    else:
        xtr, _ = M.load(list('ABCDEF'))
        if COH == 'NH3':
            xte, _ = M.load_nh3()
        elif COH == 'GH':
            xte = pd.concat([T.load_gh(c)[0] for c in 'GH'])
        else:
            xte, _ = X.load_i()
    ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr); Z = M.transform(xte, ref=xtr)
    t, e = T.outcome_at(xte, H); known = e | (t >= H)
    PANELS = {'matched': M.features_of(SEL['US']['matched']['items']), 'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']),
              'aware@routine': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
N = len(Z)


# ---------------- recalibration ----------------
def cox_slope(lp, t, e):
    """Cox partial likelihood for one coefficient on lp (Newton)."""
    o = np.argsort(-t, kind='stable'); last = np.searchsorted(-t[o], -t[o], side='right') - 1; x, d = lp[o], e[o].astype(float); a = 1.0
    for _ in range(50):
        r = np.exp(a * x - (a * x).max()); S0 = np.cumsum(r)[last]; S1 = np.cumsum(r * x)[last]; S2 = np.cumsum(r * x * x)[last]
        g = (d * (x - S1 / S0)).sum(); h = (d * (S2 / S0 - (S1 / S0) ** 2)).sum(); st = g / h; a += st
        if abs(st) < 1e-8:
            break
    return a


def breslow(lp, t, e, h):
    o = np.argsort(-t, kind='stable'); last = np.searchsorted(-t[o], -t[o], side='right') - 1
    return float((e[o] / np.cumsum(np.exp(lp[o]))[last])[t[o] <= h].sum())


def cloglog(lp, yy, b0):
    """P(death) = 1 - exp(-exp(alpha + beta * lp)); Newton from b0."""
    Xd = np.column_stack([np.ones_like(lp), lp]); b = np.array(b0, float)
    for _ in range(50):
        eta = Xd @ b; mu = 1 - np.exp(-np.exp(eta)); mu = np.clip(mu, 1e-9, 1 - 1e-9); dmu = np.exp(eta - np.exp(eta))
        wgt = dmu ** 2 / (mu * (1 - mu)); z = (yy - mu) / dmu
        st = np.linalg.solve(Xd.T @ (wgt[:, None] * Xd), Xd.T @ (wgt * z)); b = b + st
        if np.abs(st).max() < 1e-8:
            break
    return b


def nb(hi, dead, use):
    W = use.sum()
    return float((hi & dead & use).sum() / W - (hi & ~dead & use).sum() / W * TAU / (1 - TAU))


# ---------------- run ----------------
rows, whole = [], {}
for pname, f in PANELS.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr); lp = m.lp(Z[cols].to_numpy())
    H0h = T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H); rh = 1 - np.exp(-H0h * np.exp(lp))           # imported risk at the outcome horizon
    if BIN:
        H09 = T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, 108); dec = 1 - np.exp(-H09 * np.exp(lp))       # decision scale (9-year), as Task 21
        dead, use = y, np.ones(N, bool)
    else:
        dec, dead, use = rh, e, known
    O, E_ = dead[use].sum(), rh[use].sum(); whole[pname] = dict(OE=float(O / E_), lo=float(O / E_ * np.exp(-1.96 / np.sqrt(O))), hi=float(O / E_ * np.exp(1.96 / np.sqrt(O))),
                                                                 deaths=int(O), n=int(use.sum()))
    for n in NS:
        if N - n < MIN_TEST:
            continue
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 1000 * rep + n); loc = np.zeros(N, bool); loc[rng.choice(N, n, replace=False)] = True; tst = ~loc
            Ol = int(dead[loc & use].sum())
            if Ol < MIN_LOCAL_DEATHS:
                rows.append(dict(panel=pname, n=n, rep=rep, local_deaths=Ol, method='skipped')); continue
            El = rh[loc & use].sum(); recal = bool(abs(np.log(Ol / El)) > 1.96 / np.sqrt(Ol))
            if BIN:
                al, be = cloglog(lp[loc], dead[loc].astype(float), [np.log(H0h), 1.0])
                rB = 1 - np.exp(-np.exp(al + be * lp)); dB = 1 - np.exp(-np.exp(al + np.log(H09 / H0h) + be * lp))
            else:
                a = cox_slope(lp[loc], t[loc], e[loc]); H0 = breslow(a * lp[loc], t[loc], e[loc], H)
                rB = 1 - np.exp(-H0 * np.exp(a * lp)); dB = rB
            R = {'A import': (rh, dec), 'B recalibrate': (rB, dB)}; R['T test then recalibrate'] = R['B recalibrate'] if recal else R['A import']
            for k, (r, d) in R.items():
                oe = float(dead[tst & use].sum() / r[tst & use].sum())
                rows.append(dict(panel=pname, n=n, rep=rep, local_deaths=Ol, method=k, OE=oe, abs_log_OE=abs(np.log(oe)),
                                 NB=nb(d >= TAU, dead, tst & use), flagged=float((d >= TAU)[tst & use].mean()), recal=int(recal) if k[0] == 'T' else np.nan))
        print(COH, pname, n, 'done', flush=True)
D = pd.DataFrame(rows); D.to_csv(HERE / f'{PRE}reps.csv', index=False)

# ---------------- summary ----------------
S = dict(cohort=COH, n=N, horizon_months=H, tau=TAU, reps=REPS, whole_cohort_import=whole, cells=[])
ok = D[D.method != 'skipped']
for (p, n), g in ok.groupby(['panel', 'n']):
    w = g.pivot_table(index='rep', columns='method', values=['abs_log_OE', 'NB', 'recal'])
    A_ = w[('abs_log_OE', 'A import')]
    cell = dict(panel=p, n=int(n), reps_used=int(len(w)), skipped=int((D[(D.panel == p) & (D.n == n)].method == 'skipped').sum()),
                abs_log_OE_import=float(A_.mean()), recal_share_T=float(w[('recal', 'T test then recalibrate')].mean()))
    for k in ['B recalibrate', 'T test then recalibrate']:
        dOE = w[('abs_log_OE', k)] - A_; dNB = w[('NB', k)] - w[('NB', 'A import')]
        cell[k] = dict(abs_log_OE=float(w[('abs_log_OE', k)].mean()), better_calibrated_share=float((dOE < 0).mean()),
                       dNB_mean=float(dNB.mean()), dNB_p2_5=float(dNB.quantile(.025)), dNB_p97_5=float(dNB.quantile(.975)))
    b = cell['B recalibrate']   # prespecified reading (plan section 4)
    cell['reading_B'] = ('helps' if b['better_calibrated_share'] >= 0.70 and b['dNB_mean'] >= -0.001 else
                         'not needed' if b['better_calibrated_share'] < 0.50 else 'uncertain')
    S['cells'].append(cell)
json.dump(S, open(HERE / f'{PRE}summary.json', 'w'), indent=1)
print(json.dumps(S, indent=1))
