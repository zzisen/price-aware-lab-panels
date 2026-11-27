#!/usr/bin/env python3
"""Task 38 (additional analysis, NC upgrade item 5): censoring-robust and design-based estimates for the primary US-priced comparisons.
Models: ridge Cox (lambda 1) with age, sex and log analytes, fitted in NHANES 1999-2010 (all-cause death), panels as frozen in Task 19
(matched CBC+CMP+phosphorus $32.41; target-aware CBC+CMP+ACR+GGT+phosphorus $50.57; routine CMP+CBC+lipids+HbA1c $50.77).
Cohorts: NHANES III and NHANES 2011-14 (the two cohorts with survey design variables).
Metrics: Harrell's C at the article's horizon (108 / 60 months; unweighted as in the article); Uno's C truncated at 60 months and cumulative/dynamic IPCW AUC at 60 months, both on all follow-up (to 108 / 96 months);
each also survey-weighted. Intervals: participant bootstrap (as in the article) and design-based bootstrap (PSUs resampled with replacement
within strata, survey weights kept). 500 resamples, seed 20261138. Output: t38_results.json, t38_print.txt."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
M.L.low_priority()
TAU, B, SEED = 60.0, 500, 20261138


def km_censor(t, e, w):
    """Weighted Kaplan-Meier of the censoring distribution, G(s-) evaluated at each t."""
    o = np.argsort(t, kind='mergesort'); ts, cs, ws = t[o], (~e[o]).astype(float), w[o]
    ut = np.unique(ts); G = np.ones(len(ut)); s = 1.0
    atrisk = np.array([ws[ts >= u].sum() for u in ut]); cens = np.array([ws[(ts == u)][cs[ts == u] == 1].sum() for u in ut])
    Gl = np.empty(len(ut))
    for i in range(len(ut)):
        Gl[i] = s
        s *= 1 - (cens[i] / atrisk[i] if atrisk[i] > 0 else 0)
    return lambda q: Gl[np.clip(np.searchsorted(ut, q, side='left'), 0, len(ut) - 1)]


def conc(lp, t, e, w, wi):
    """Sum over comparable pairs (i event, t_j > t_i) of w_i*w_j*wi_i*[lp_i > lp_j + 0.5 tie], and the pair-weight total.
    Vectorised over the distinct event times (months are integers, so there are few)."""
    num = den = 0.0
    ev = np.flatnonzero(e & (wi > 0))
    for u in np.unique(t[ev]):
        later = t > u
        if not later.any():
            continue
        lk = lp[later]; o = np.argsort(lk, kind='mergesort'); lk = lk[o]; ck = np.concatenate([[0.0], np.cumsum(w[later][o])])
        ii = ev[t[ev] == u]
        lo = np.searchsorted(lk, lp[ii], side='left'); hi = np.searchsorted(lk, lp[ii], side='right')
        ww = w[ii] * wi[ii]
        num += float((ww * (ck[lo] + 0.5 * (ck[hi] - ck[lo]))).sum()); den += float(ww.sum() * ck[-1])
    return num, den


def harrell(lp, t, e, w):
    n, d = conc(lp, t, e, w, np.ones(len(t)))
    return n / d


def uno(lp, t, e, w, tau=TAU):
    G = km_censor(t, e, w); g = G(t)
    wi = np.where(e & (t < tau), 1 / np.maximum(g, 1e-8) ** 2, 0.0)
    n, d = conc(lp, t, e, w, wi)
    return n / d


def td_auc(lp, t, e, w, tau=TAU):
    G = km_censor(t, e, w)
    case = e & (t <= tau); ctrl = t > tau
    wc = w[case] / np.maximum(G(t[case]), 1e-8); wk = w[ctrl] / max(G(np.array([tau]))[0], 1e-8)
    s = np.argsort(lp[ctrl]); lk = lp[ctrl][s]; ck = np.concatenate([[0], np.cumsum(wk[s])])
    lo = np.searchsorted(lk, lp[case], side='left'); hi = np.searchsorted(lk, lp[case], side='right')
    num = (wc * (ck[lo] + 0.5 * (ck[hi] - ck[lo]))).sum()
    return float(num / (wc.sum() * wk.sum()))


# ---------------- data ----------------
x, _ = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
sel = json.load(open(W / 'task19_dm2' / 'selected_all.json'))['US']
PAN = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'matched': M.features_of(sel['matched']['items']),
       'aware': M.features_of(sel['aware']['CMP+CBC+LIPID+A1C']['items'])}
cols = {k: M.BASE + v for k, v in PAN.items()}
fits = {k: M.Cox(1.0).fit(Z[c].to_numpy(), t, e) for k, c in cols.items()}

nh3 = M.load_nh3()[0]
des = M.read_fixed('lab', ['SEQN', 'SDPPSU6', 'SDPSTRA6']); des.index = [f'N3{int(i)}' for i in des.index]
nh3 = nh3.join(des)
gh = []
for c in 'GH':
    xc = T.load_gh(c)[0]
    d = pd.read_sas(W / 'external_data' / 'nhanes_gh' / c / f'DEMO_{c}.xpt', format='xport').set_index('SEQN')[['SDMVPSU', 'SDMVSTRA']]
    d.index = [f'{c}{int(i)}' for i in d.index]
    gh.append(xc.join(d))
gh = pd.concat(gh)
COH = {'NHANES III': (nh3, 'SDPSTRA6', 'SDPPSU6', 108, 108), 'NHANES 2011-14': (gh, 'SDMVSTRA', 'SDMVPSU', 60, 96)}

res = {'panels': {k: v for k, v in PAN.items()}, 'tau_months': TAU, 'cohorts': {}}
lines = []
for name, (xv, sv, pv, H, HL) in COH.items():
    Zv = M.transform(xv, ref=x); tv, ev = T.outcome_at(xv, H); tv = tv.astype(float); tl, el = T.outcome_at(xv, HL); tl = tl.astype(float)
    w1 = np.ones(len(tv)); ws = xv.weight.to_numpy(float)
    LP = {k: fits[k].lp(Zv[cols[k]].to_numpy()) for k in PAN}
    strata = xv[sv].to_numpy(); psu = (xv[sv].astype(int).astype(str) + '_' + xv[pv].astype(int).astype(str)).to_numpy()

    def metrics(idx, design_w):
        out = {}
        for k in PAN:
            lp, tt, ee, t2, e2 = LP[k][idx], tv[idx], ev[idx], tl[idx], el[idx]
            out[k] = dict(harrell=harrell(lp, tt, ee, w1[idx]), harrell_w=harrell(lp, tt, ee, design_w[idx]),
                          uno=uno(lp, t2, e2, w1[idx]), uno_w=uno(lp, t2, e2, design_w[idx]),
                          auc60=td_auc(lp, t2, e2, w1[idx]), auc60_w=td_auc(lp, t2, e2, design_w[idx]))
        return out
    base = metrics(np.arange(len(tv)), ws)
    rng = np.random.default_rng(SEED)
    units = {h: np.unique(psu[strata == h]) for h in np.unique(strata)}
    members = {u: np.flatnonzero(psu == u) for u in np.unique(psu)}
    bp, bd = [], []
    for b in range(B):
        bp.append(metrics(rng.integers(0, len(tv), len(tv)), ws))
        pick = np.concatenate([members[u] for h, us in units.items() for u in rng.choice(us, len(us))])
        bd.append(metrics(pick, ws))
    R = {'n': int(len(tv)), 'deaths': int(ev.sum()), 'deaths_by_60': int((ev & (tv <= 60)).sum()), 'strata': len(units), 'psus': len(members),
         'estimates': base, 'differences': {}}
    for comp in ['matched', 'aware']:
        for met in ['harrell', 'harrell_w', 'uno', 'uno_w', 'auc60', 'auc60_w']:
            dv = base[comp][met] - base['routine'][met]
            p = [s[comp][met] - s['routine'][met] for s in bp]; q = [s[comp][met] - s['routine'][met] for s in bd]
            R['differences'][f'{comp}-routine {met}'] = dict(diff=dv, ci_participant=[float(np.quantile(p, .025)), float(np.quantile(p, .975))],
                                                           ci_design=[float(np.quantile(q, .025)), float(np.quantile(q, .975))])
    res['cohorts'][name] = R
    lines.append(f'== {name}: n {R["n"]}, deaths {R["deaths"]}, by 60 months {R["deaths_by_60"]}, strata {R["strata"]}, PSUs {R["psus"]}')
    for kk, vv in R['differences'].items():
        lines.append(f'  {kk}: {vv["diff"]:+.4f} participant [{vv["ci_participant"][0]:+.4f}, {vv["ci_participant"][1]:+.4f}] design [{vv["ci_design"][0]:+.4f}, {vv["ci_design"][1]:+.4f}]')
    print('\n'.join(lines[-13:]), flush=True)
json.dump(res, open(HERE / 't38_results.json', 'w'), indent=1, default=float)
open(HERE / 't38_print.txt', 'w', encoding='utf-8').write('\n'.join(lines))
