#!/usr/bin/env python3
"""Task 22 (plan_t22_FROZEN.md): VOI stopping rule for sequential laboratory ordering, one validation cohort per call.
Usage: run_t22.py <NH3|GH|I> [--dry]     --dry: train on cycle C, test on cycle D (development check only).
Training: NHANES 1999-2010 (A-F), complete clinical variables; the fitted rule must equal rule_t22.json (checked).
Outputs: t22_<cohort>_endpoints.json, t22_<cohort>_person.csv, t22_<cohort>_subgroups.csv (prefix dry_ for --dry)."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t22lib as V  # noqa: E402
M = V.M
os.chdir(HERE.parent / 'task21_final')
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()

COH = sys.argv[1]; DRY = '--dry' in sys.argv; PRE = f"{'dry_' if DRY else ''}t22_{COH}_"
NBOOT, NSUB = (200, 100) if DRY else (1000, 200)
H = 108 if DRY else {'NH3': 108, 'GH': 60, 'I': 36}[COH]
TAU = V.TAU; ROUT_ITEMS = M.STANDARD['CMP+CBC+LIPID+A1C']; C_ROUT = M.cost(ROUT_ITEMS)

# ---------------- data ----------------
if DRY:
    xtr, ctr = M.load(['C']); xte, cte = M.load(['D'])
else:
    xtr, ctr = M.load(list('ABCDEF'))
    if COH == 'NH3':
        xte, cte = M.load_nh3()
    elif COH == 'GH':
        parts = [T.load_gh(c) for c in 'GH']; xte = pd.concat([p for p, _ in parts]); cte = [n for _, n in parts]
    else:
        xte, cte = X.load_i()
ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr); Z = M.transform(xte, ref=xtr)
cltr = M.clinical(xtr); oktr = cltr.notna().all(axis=1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
cl = M.clinical(xte) if (DRY or COH == 'NH3') else (T.clinical_gh(xte) if COH == 'GH' else X.clinical_i(xte))
okc = cl.notna().all(axis=1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(okc)]; xc = xte.iloc[np.flatnonzero(okc)]
t, e = T.outcome_at(xc, H); n = len(t); known = e | (t >= H)
ta, ea = ttr[oktr], etr[oktr]
folds = np.array_split(np.random.default_rng(1).permutation(int(oktr.sum())), 5)

# ---------------- rules ----------------
rule = V.VOIRule().fit(Ztrc, ta, ea, folds)
if not DRY:
    frozen = json.load(open(HERE / 'rule_t22.json'))
    for k, v in rule.coef().items():
        assert np.allclose(v['flip'], frozen['gain_models'][k]['flip'], atol=1e-9) and np.allclose(v['magnitude'], frozen['gain_models'][k]['magnitude'], atol=1e-9), k
R = rule.risks(Zc)
POL = {}
for v in (V.V_MAIN,) + V.V_SENS:
    k_, r_, c_ = rule.run(R, v); POL[f'VOI V={int(v)}'] = (r_, c_, k_)
cur = M.FixedPath(V.ORDER, TAU, base=V.CB).fit(Ztrc, ta, ea, folds); Rc = cur.risks(Zc)
st_c, r_c, c_c = cur.run(Rc, 0.02); POL['current rule (Task 19/21, eps 0.02)'] = (r_c, c_c, None)
REF = {'clinical only': [], 'CMP+CBC': ['CMP', 'CBC'], 'matched (CMP+CBC+phosphorus)': ['CBC', 'CMP', 'phosphorus'], 'CMP+CBC+acr': ['CMP', 'CBC', 'acr'],
       'routine': ROUT_ITEMS}
for k, it in REF.items():
    cols = V.CB + M.features_of(it)
    POL[k] = (M.Cox(1.0).fit(Ztrc[cols].to_numpy(), ta, ea).risk(Zc[cols].to_numpy()), np.full(n, M.cost(it) if it else 0.0), None)
MAIN = f'VOI V={int(V.V_MAIN)}'


def met(r, c, i):
    hi = r[i] >= TAU; ee, kk = e[i], known[i]
    nb = float((hi & ee & kk).sum() / kk.sum() - (hi & ~ee & kk).sum() / kk.sum() * TAU / (1 - TAU)) if H == 108 else np.nan
    return dict(cost=float(c[i].mean()), share_routine=float(c[i].mean() / C_ROUT), flagged_per1000=1000 * float(hi.mean()),
                deaths_flagged_per1000=1000 * float((hi & ee).mean()), missed_per1000=1000 * float((~hi & ee).mean()), NB=nb)


ALL = np.arange(n)
rows = []
for k, (r, c, _) in POL.items():
    m = met(r, c, ALL)
    m.update(policy=k, no_lab=float((c == 0).mean()), agree_routine=float(((r >= TAU) == (POL['routine'][0] >= TAU)).mean()),
             C=M.cindex(np.log(np.clip(r, 1e-9, 1)), t, e))
    rows.append(m)
tab = pd.DataFrame(rows)[['policy', 'cost', 'share_routine', 'no_lab', 'flagged_per1000', 'deaths_flagged_per1000', 'missed_per1000', 'NB', 'agree_routine', 'C']]

# ---------------- paired bootstrap ----------------
rng = np.random.default_rng(20261108)
PAIRS = [(MAIN, 'routine'), (MAIN, 'current rule (Task 19/21, eps 0.02)'), (MAIN, 'matched (CMP+CBC+phosphorus)'), (MAIN, 'CMP+CBC'),
         ('VOI V=10000', 'routine'), ('VOI V=100000', 'routine'), ('current rule (Task 19/21, eps 0.02)', 'routine')]
KEYS = ['cost', 'share_routine', 'flagged_per1000', 'missed_per1000', 'NB']
bs = {f'{a} - {b}': {k: [] for k in KEYS} for a, b in PAIRS}; share_main = []
for _ in range(NBOOT):
    i = rng.integers(0, n, n); mm = {k: met(POL[k][0], POL[k][1], i) for k in set(p for pr in PAIRS for p in pr)}
    share_main.append(mm[MAIN]['share_routine'])
    for a, b in PAIRS:
        for k in KEYS:
            bs[f'{a} - {b}'][k].append(mm[a][k] - mm[b][k])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if np.all(np.isfinite(v)) else None  # noqa: E731
pt = {k: met(POL[k][0], POL[k][1], ALL) for k in POL}
contrasts = {f'{a} - {b}': {k: dict(point=pt[a][k] - pt[b][k], ci=ci(bs[f'{a} - {b}'][k])) for k in KEYS if not (k == 'NB' and H != 108)} for a, b in PAIRS}
share_ci = ci(share_main)
H4 = dict(share_routine=pt[MAIN]['share_routine'], share_routine_CI=share_ci, share_pass=bool(share_ci[1] <= 0.50))
if H == 108:
    dnb = contrasts[f'{MAIN} - routine']['NB']; H4.update(dNB=dnb['point'], dNB_CI=dnb['ci'], dNB_pass=bool(dnb['ci'][0] > -0.005))
    H4['pass'] = H4['share_pass'] and H4['dNB_pass']
else:
    H4['pass'] = H4['share_pass']

# ---------------- subgroups ----------------
G = M.groups(xc); sub = []
for g, msk in G.items():
    i0 = np.flatnonzero(msk); row = dict(group=g, n=len(i0), deaths=int(e[i0].sum()))
    if e[i0].sum() < 20:
        sub.append(dict(row, note='fewer than 20 deaths')); continue
    a, b = met(POL[MAIN][0], POL[MAIN][1], i0), met(POL['routine'][0], POL['routine'][1], i0)
    d = [met(POL[MAIN][0], POL[MAIN][1], j)['missed_per1000'] - met(POL['routine'][0], POL['routine'][1], j)['missed_per1000']
         for j in (i0[rng.integers(0, len(i0), len(i0))] for _ in range(NSUB))]
    row.update(cost_VOI=a['cost'], missed_VOI=a['missed_per1000'], missed_routine=b['missed_per1000'], dmissed=a['missed_per1000'] - b['missed_per1000'],
               dmissed_lo=ci(d)[0], dmissed_hi=ci(d)[1], flagged_VOI=a['flagged_per1000'], flagged_routine=b['flagged_per1000'])
    sub.append(row)
subd = pd.DataFrame(sub)

# ---------------- output ----------------
end = pd.Series(POL[MAIN][2]).map({j: '+'.join(s) or 'clinical only' for j, s in enumerate(rule.states)}).value_counts(normalize=True).round(4).to_dict()
res = dict(cohort=COH, horizon=H, tau=TAU, train=ctr, test=cte, n=n, deaths=int(e.sum()), V_main=V.V_MAIN, table=tab.to_dict('records'),
           contrasts=contrasts, H4=H4, end_states_main=end, chart_main=rule.chart(V.V_MAIN))
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print('COHORT', COH, 'horizon', H, '| TRAIN', ctr, 'n', len(ta), '| TEST', cte, 'n (complete clinical)', n, 'deaths', int(e.sum()))
print(tab.round(4).to_string())
for k, v in contrasts.items():
    print(k, {kk: (round(vv['point'], 4), [round(x_, 4) for x_ in vv['ci']] if vv['ci'] else None) for kk, vv in v.items()})
print('H4', json.dumps(H4)); print('END STATES', end); print(subd.round(4).to_string())
json.dump(res, open(HERE / (PRE + 'endpoints.json'), 'w'), indent=1, default=float)
subd.to_csv(HERE / (PRE + 'subgroups.csv'), index=False)
person = pd.DataFrame(dict(id=xc.index.astype(str), t=t, e=e.astype(int), **{f'r_{"+".join(s) or "none"}': R[s] for s in rule.states},
                           r_routine=POL['routine'][0], k_main=POL[MAIN][2], cost_main=POL[MAIN][1], r_main=POL[MAIN][0],
                           r_current=POL['current rule (Task 19/21, eps 0.02)'][0], cost_current=POL['current rule (Task 19/21, eps 0.02)'][1]))
person.to_csv(HERE / (PRE + 'person.csv'), index=False)
