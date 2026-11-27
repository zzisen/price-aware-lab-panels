#!/usr/bin/env python3
"""Task 46 (b) -- ADDITIONAL (POST HOC) ANALYSIS, added 2026-11-27 after all planned results were known.
Sequential ordering chart (Task 22, rule_t22.json) and missed deaths among non-Hispanic Black participants.
Usage: b_chart_black.py <NH3|GH|I>
Step 0 re-runs the Task 22 pipeline (same code path, same seeds) and checks the published table and subgroup rows
(task22_adaptive/t22_<cohort>_endpoints.json and _subgroups.csv) exactly. Step 1 diagnoses where people stop and how well the risk at the
stopping step is calibrated, by group. Step 2 evaluates sensitivity variants re-derived on the training data only:
  F2.0          step-1 ordering floor lowered from 3.46% to 2.0% (upper limit and later steps unchanged; V = 30 000)
  V100k         V = $100 000 (the pre-specified sensitivity value; whole chart from the same training gain models)
  V100k+F2.0    both
  Ftrain        step-1 floor chosen in the training data (cross-fitted risks) as the lowest floor on a 0.25-point grid whose training fee
                share stays <= 40% of routine (race not used)
No model uses race. Nobody is dropped. Existing files are only read. Output: b_<cohort>.json, b_<cohort>_groups.csv (this folder)."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task22_adaptive'))
import t22lib as V  # noqa: E402
M = V.M
os.chdir(W / 'task21_final')
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]; NBOOT, NSUB = 1000, 200
H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]
TAU = V.TAU; ROUT_ITEMS = M.STANDARD['CMP+CBC+LIPID+A1C']; C_ROUT = M.cost(ROUT_ITEMS)
out = dict(label='ADDITIONAL (POST HOC) ANALYSIS - not in any frozen plan', date='2026-11-27', cohort=COH, horizon=H)

# ---------------- data and rule: identical to run_t22.py ----------------
xtr, ctr = M.load(list('ABCDEF'))
if COH == 'NH3':
    xte, cte = M.load_nh3()
elif COH == 'GH':
    parts = [T.load_gh(c) for c in 'GH']; xte = pd.concat([p for p, _ in parts])
else:
    xte, cte = X.load_i()
ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr); Z = M.transform(xte, ref=xtr)
cltr = M.clinical(xtr); oktr = cltr.notna().all(axis=1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
cl = M.clinical(xte) if COH == 'NH3' else (T.clinical_gh(xte) if COH == 'GH' else X.clinical_i(xte))
okc = cl.notna().all(axis=1).to_numpy(); Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(okc)]; xc = xte.iloc[np.flatnonzero(okc)]
t, e = T.outcome_at(xc, H); n = len(t); known = e | (t >= H)
ta, ea = ttr[oktr], etr[oktr]; xtrc = xtr.iloc[np.flatnonzero(oktr)]
folds = np.array_split(np.random.default_rng(1).permutation(int(oktr.sum())), 5)
rule = V.VOIRule().fit(Ztrc, ta, ea, folds)
frozen = json.load(open(W / 'task22_adaptive' / 'rule_t22.json'))
for k, v in rule.coef().items():
    assert np.allclose(v['flip'], frozen['gain_models'][k]['flip'], atol=1e-9) and np.allclose(v['magnitude'], frozen['gain_models'][k]['magnitude'], atol=1e-9), k
R = rule.risks(Zc)
MAIN = 'VOI V=30000'
k30, r30, c30 = rule.run(R, V.V_MAIN)
k100, r100, c100 = rule.run(R, 100000.0)
rcols = V.CB + M.features_of(ROUT_ITEMS); mrout = M.Cox(1.0).fit(Ztrc[rcols].to_numpy(), ta, ea)
r_rout = mrout.risk(Zc[rcols].to_numpy()); c_rout = np.full(n, C_ROUT)


def met(r, c, i):
    hi = r[i] >= TAU; ee, kk = e[i], known[i]
    nb = float((hi & ee & kk).sum() / kk.sum() - (hi & ~ee & kk).sum() / kk.sum() * TAU / (1 - TAU)) if H == 108 else np.nan
    return dict(cost=float(c[i].mean()), share_routine=float(c[i].mean() / C_ROUT), flagged_per1000=1000 * float(hi.mean()),
                deaths_flagged_per1000=1000 * float((hi & ee).mean()), missed_per1000=1000 * float((~hi & ee).mean()), NB=nb)


ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
ALL = np.arange(n)
# ---------------- step 0: reproduce ----------------
pub = json.load(open(W / 'task22_adaptive' / f't22_{COH}_endpoints.json')); ptab = {r['policy']: r for r in pub['table']}
chk = {}
for nm, (r, c) in {MAIN: (r30, c30), 'VOI V=100000': (r100, c100), 'routine': (r_rout, c_rout)}.items():
    m = met(r, c, ALL)
    for kk in ['cost', 'share_routine', 'flagged_per1000', 'missed_per1000']:
        chk[f'{nm} {kk}'] = abs(m[kk] - ptab[nm][kk]) < 1e-9
rng = np.random.default_rng(20261108)
for _ in range(NBOOT):                       # advance the generator exactly as the main bootstrap of run_t22.py
    rng.integers(0, n, n)
G = M.groups(xc); psub = pd.read_csv(W / 'task22_adaptive' / f't22_{COH}_subgroups.csv').set_index('group'); rep_sub = {}
for g, msk in G.items():
    i0 = np.flatnonzero(msk)
    if e[i0].sum() < 20:
        continue
    a, b = met(r30, c30, i0), met(r_rout, c_rout, i0)
    d = [met(r30, c30, j)['missed_per1000'] - met(r_rout, c_rout, j)['missed_per1000'] for j in (i0[rng.integers(0, len(i0), len(i0))] for _ in range(NSUB))]
    rep_sub[g] = dict(dmissed=a['missed_per1000'] - b['missed_per1000'], lo=ci(d)[0], hi=ci(d)[1])
    chk[f'sub {g}'] = bool(np.allclose([rep_sub[g]['dmissed'], rep_sub[g]['lo'], rep_sub[g]['hi']],
                                       [psub.loc[g, 'dmissed'], psub.loc[g, 'dmissed_lo'], psub.loc[g, 'dmissed_hi']], atol=1e-9))
out['reproduce'] = dict(overall={nm: met(r, c, ALL) for nm, (r, c) in {MAIN: (r30, c30), 'routine': (r_rout, c_rout)}.items()}, subgroups=rep_sub,
                        checks=chk, all_match=bool(all(chk.values())), chart_V30000=rule.chart(V.V_MAIN))
print('reproduce', json.dumps(out['reproduce']), flush=True)
assert out['reproduce']['all_match'], chk

# ---------------- step 1: diagnosis ----------------
NAMES = ['clinical only', 'CMP', 'CMP+CBC', 'CMP+CBC+acr']
GRP = {'all': np.ones(n, bool), **{g: G[g] for g in ['Non-Hispanic White', 'Non-Hispanic Black', 'Mexican American']}}
lo1, hi1 = rule.chart(V.V_MAIN)['clinical only'][0]
r0 = R[()]


def obs(i):
    kk = known[i]; return float(e[i][kk].mean()) if kk.any() else np.nan


diag = {}
for g, msk in GRP.items():
    i0 = np.flatnonzero(msk); d = dict(n=len(i0), deaths=int(e[i0].sum()))
    d['stop_share'] = {NAMES[s]: float((k30[i0] == s).mean()) for s in range(4)}
    d['stop_before_CMP_below_floor'] = float(((k30[i0] == 0) & (r0[i0] < lo1)).mean())
    d['stop_before_CMP_above_upper'] = float(((k30[i0] == 0) & (r0[i0] > hi1)).mean())
    hv, hr = r30[i0] >= TAU, r_rout[i0] >= TAU; ee = e[i0]
    d['deaths_routine_flag_not_chart_per1000'] = 1000 * float((hr & ~hv & ee).mean())
    d['deaths_chart_flag_not_routine_per1000'] = 1000 * float((hv & ~hr & ee).mean())
    d['excess_per1000'] = d['deaths_routine_flag_not_chart_per1000'] - d['deaths_chart_flag_not_routine_per1000']
    d['excess_by_stop_state_per1000'] = {NAMES[s]: 1000 * float(((k30[i0] == s) & ee & (hr & ~hv)).sum() - ((k30[i0] == s) & ee & (hv & ~hr)).sum()) / len(i0) for s in range(4)}
    d['net_excess_deaths_count_by_stop_state'] = {NAMES[s]: int(((k30[i0] == s) & ee & hr & ~hv).sum() - ((k30[i0] == s) & ee & hv & ~hr).sum()) for s in range(4)}
    cal = {}
    for s in range(4):
        j = i0[k30[i0] == s]
        if len(j) == 0:
            continue
        cal[NAMES[s]] = dict(n=len(j), deaths=int(e[j].sum()), mean_pred_at_stop=float(r30[j].mean()), observed=obs(j),
                             OE=obs(j) / float(r30[j][known[j]].mean()) if known[j].any() else np.nan,
                             share_flagged_at_stop=float((r30[j] >= TAU).mean()), mean_routine_risk=float(r_rout[j].mean()))
    j = i0[(k30[i0] == 0) & (r0[i0] < lo1)]
    cal['clinical only, below floor'] = dict(n=len(j), deaths=int(e[j].sum()), mean_pred=float(r0[j].mean()), observed=obs(j),
                                             OE=obs(j) / float(r0[j][known[j]].mean()), routine_flagged=float((r_rout[j] >= TAU).mean()),
                                             deaths_routine_flagged=int((e[j] & (r_rout[j] >= TAU)).sum()))
    d['calibration_at_stop'] = cal
    d['OE_clinical_only_model'] = obs(i0) / float(r0[i0][known[i0]].mean()); d['OE_routine_model'] = obs(i0) / float(r_rout[i0][known[i0]].mean())
    d['OE_chart_final_risk'] = obs(i0) / float(r30[i0][known[i0]].mean())
    # calibration of the clinical-only model in risk bands (step-1 decision)
    bands = [(0, .02), (.02, lo1), (lo1, .10), (.10, hi1), (hi1, 1.01)]
    d['clinical_only_bands'] = [dict(band=f'{a:.4f}-{b:.4f}', n=int(((r0[i0] >= a) & (r0[i0] < b)).sum()), pred=float(r0[i0][(r0[i0] >= a) & (r0[i0] < b)].mean()) if ((r0[i0] >= a) & (r0[i0] < b)).any() else np.nan,
                                     obs=obs(i0[(r0[i0] >= a) & (r0[i0] < b)]) if ((r0[i0] >= a) & (r0[i0] < b)).any() else np.nan) for a, b in bands]
    diag[g] = d
out['diagnosis'] = diag; print('diagnosis', json.dumps(diag, default=float), flush=True)

# ---------------- step 2: variants (training only) ----------------
# training cross-fitted risks for the floor choice and for the training-data fairness check
rcv = {s: np.zeros(len(ta)) for s in rule.states}; rcv_rout = np.zeros(len(ta))
for te in folds:
    tr = np.setdiff1d(np.arange(len(ta)), te)
    for s in rule.states:
        Xs = Ztrc[rule.cols(s)].to_numpy(); rcv[s][te] = M.Cox(1.0).fit(Xs[tr], ta[tr], ea[tr]).risk(Xs[te])
    Xr = Ztrc[rcols].to_numpy(); rcv_rout[te] = M.Cox(1.0).fit(Xr[tr], ta[tr], ea[tr]).risk(Xr[te])


def run_floor(RR, Vv, floor=None):
    """rule.run with the step-1 go-region replaced by [floor, upper limit of the V-chart] when floor is given."""
    nn = len(RR[rule.final]); k = np.zeros(nn, int); active = np.ones(nn, bool)
    up = rule.chart(Vv)['clinical only'][0][1]
    for j, s in enumerate(rule.states[:-1]):
        idx = np.flatnonzero(active & (k == j)); rr = RR[s][idx]
        g = ((rr >= floor) & (rr <= up)) if (j == 0 and floor is not None) else rule.go(s, rr, Vv)
        k[idx[g]] = j + 1; active[idx[~g]] = False
    risk = np.choose(k, [RR[s] for s in rule.states]); cst = np.array([rule.cost(s) for s in rule.states])[k]
    return k, risk, cst


kt, _, _ = run_floor(rcv, V.V_MAIN); kt2, _, _ = rule.run(rcv, V.V_MAIN); assert (kt == kt2).all()
grid = np.round(np.arange(0.0, 0.0346, 0.0025), 4); tr_share = {}
for f in grid:
    _, _, cc = run_floor(rcv, V.V_MAIN, f); tr_share[float(f)] = float(cc.mean() / C_ROUT)
ok = [f for f, s in tr_share.items() if s <= 0.40]; F_TRAIN = min(ok)
out['training_floor'] = dict(grid_share=tr_share, chosen_floor=F_TRAIN, criterion='lowest floor with training fee share <= 40% of routine (cross-fitted risks)')
# training-data fairness check (cross-fitted, horizon 108, training known status)
tt_, et_ = np.minimum(ta, 108), (ea & (ta <= 108)); Gt = M.groups(xtrc)
trc = {}
for nm, (Vv, fl) in {'main V30000': (V.V_MAIN, None), 'F2.0': (V.V_MAIN, .02), 'V100k': (1e5, None), 'V100k+F2.0': (1e5, .02), 'Ftrain': (V.V_MAIN, F_TRAIN)}.items():
    kk_, rr_, cc_ = run_floor(rcv, Vv, fl); row = dict(share=float(cc_.mean() / C_ROUT))
    for g in ['all', 'Non-Hispanic White', 'Non-Hispanic Black', 'Mexican American']:
        m_ = np.ones(len(ta), bool) if g == 'all' else Gt[g]
        row[g] = 1000 * float(((rr_[m_] < TAU) & et_[m_]).mean() - ((rcv_rout[m_] < TAU) & et_[m_]).mean())
    trc[nm] = row
out['training_crossfit_excess_per1000'] = trc; print('training', json.dumps(out['training_floor']), json.dumps(trc), flush=True)

VAR = {'main V30000 (published)': (V.V_MAIN, None), 'F2.0': (V.V_MAIN, .02), 'V100k': (1e5, None), 'V100k+F2.0': (1e5, .02), f'Ftrain ({F_TRAIN:.4f})': (V.V_MAIN, F_TRAIN)}
POL = {nm: run_floor(R, Vv, fl) for nm, (Vv, fl) in VAR.items()}
assert np.allclose(POL['main V30000 (published)'][1], r30) and np.allclose(POL['V100k'][1], r100)
rng2 = np.random.default_rng(20261127)
GG = {'all': np.ones(n, bool), **{g: m_ for g, m_ in G.items() if e[m_].sum() >= 20}}
res, rows = {}, []
for nm, (kk_, rr_, cc_) in POL.items():
    v = dict(share_routine=float(cc_.mean() / C_ROUT), cost=float(cc_.mean()), no_lab=float((kk_ == 0).mean()), groups={})
    if H == 108:
        v['NB'] = met(rr_, cc_, ALL)['NB']; v['dNB_vs_routine'] = v['NB'] - met(r_rout, c_rout, ALL)['NB']
    for g, msk in GG.items():
        i0 = np.flatnonzero(msk); a, b = met(rr_, cc_, i0), met(r_rout, c_rout, i0)
        bs_d, bs_s = [], []
        for _ in range(NBOOT):
            j = i0[rng2.integers(0, len(i0), len(i0))]
            bs_d.append(met(rr_, cc_, j)['missed_per1000'] - met(r_rout, c_rout, j)['missed_per1000']); bs_s.append(cc_[j].mean() / C_ROUT)
        v['groups'][g] = dict(n=len(i0), deaths=int(e[i0].sum()), share_routine=a['share_routine'], share_CI=ci(bs_s), missed=a['missed_per1000'],
                              missed_routine=b['missed_per1000'], excess=a['missed_per1000'] - b['missed_per1000'], excess_CI=ci(bs_d),
                              flagged=a['flagged_per1000'], flagged_routine=b['flagged_per1000'])
        rows.append(dict(cohort=COH, variant=nm, group=g, **{k_: v_ for k_, v_ in v['groups'][g].items() if not isinstance(v_, list)},
                         share_lo=v['groups'][g]['share_CI'][0], share_hi=v['groups'][g]['share_CI'][1],
                         excess_lo=v['groups'][g]['excess_CI'][0], excess_hi=v['groups'][g]['excess_CI'][1]))
    res[nm] = v
    print(nm, json.dumps(v), flush=True)
out['variants'] = res
pd.DataFrame(rows).to_csv(HERE / f'b_{COH}_groups.csv', index=False)
json.dump(out, open(HERE / f'b_{COH}.json', 'w'), indent=1, default=float)
print('done')
