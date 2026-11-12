#!/usr/bin/env python3
"""Task 27: run the frozen Task 21, 23 and 24 analyses once on Understanding Society (plan_t27_FROZEN.md). Usage: run_t27_ukhls.py [--dry]
Task 22 is not applicable (no full blood count, no urine ACR; plan_t22 section 2).
Training: NHANES 1999-2010 (A-F) restricted to the 13 analytes measured in the UKHLS nurse blood; US Medicare prices; panels from
selected_ukhls.json (training only). Outcome: binary 9-year death (death reported by the wave fielded 9 waves after the nurse visit; known status
only, as CHNS); discrimination by AUC; decision threshold tau = 10% 9-year risk (as NHANES III and CHNS).
Outputs (formats identical to the earlier binary cohort, so the frozen pooling and verification scripts read them unchanged):
  ../task21_final/t21_UKHLS_endpoints.json (+ _panels.csv, _subgroups.csv)
  ../task23_local/t23_UKHLS_reps.csv, t23_UKHLS_summary.json
  ../task24_coai/t24_UKHLS_endpoints.json, _panels.csv, _person.csv
  t27_UKHLS_person.csv (person-level scores for the independent check of the Task 21 contrasts)
--dry: train on NHANES C, 'UKHLS' = NHANES D with known 9-year status and the same menu (development check only; outputs prefixed dry_)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
import ukhls_menu as U  # noqa: E402
import ukhls_lib as K  # noqa: E402
M.L.low_priority(); U.install()
DRY = '--dry' in sys.argv; P_ = 'dry_' if DRY else ''; H = 108; TAU = 0.10
NBOOT, NSUB = (200, 100) if DRY else (1000, 200)
S = json.load(open(HERE / 'selected_ukhls.json'))
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731

# ---------------- data ----------------
if DRY:
    xtr, _ = M.load(['C']); d0, _ = M.load(['D']); td, ed = T.outcome_at(d0, H); keep = ed | (td >= H)
    xte = d0[keep].copy(); xte['died'] = ed[keep].astype(int); xte['weight'] = 1.0; counts = {'pseudo': 'NHANES D, known 9-year status'}
else:
    xtr, _ = M.load(list('ABCDEF')); xte, counts = K.load_ukhls()
ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, U.MENU); Z = T.transform_sub(xte, xtr, U.MENU)
y = xte.died.to_numpy().astype(bool); n = len(y); wt = xte.weight.to_numpy(float)
price = lambda f: M.cost_us(f) if f else 0.0  # noqa: E731
RT = 'routine (shared items)'
P = {'age+sex only': [], 'haemoglobin only': ['hgb'], 'CMP (measured analytes)': U.CMP_UK, RT: S['routine']['features'],
     'matched': S['matched']['features'], 'aware@routine': S['aware_at_routine']['features'], 'all 13 menu': U.MENU}
LP, R9, H0 = {}, {}, {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    H0[k] = T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H); LP[k] = m.lp(Z[cols].to_numpy()); R9[k] = 1 - np.exp(-H0[k] * np.exp(LP[k]))


def slope(r, yy):
    rr = np.clip(r, 1e-4, 1 - 1e-4); u = np.log(rr / (1 - rr)); A = np.column_stack([np.ones(len(u)), u]); b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ b)); Wt = p * (1 - p) + 1e-9; st = np.linalg.solve(A.T @ (Wt[:, None] * A), A.T @ (yy - p)); b += st
        if np.abs(st).max() < 1e-8:
            break
    return float(b[1])


def nb(r, yy, tau, w=None):
    w = np.ones(len(yy)) if w is None else w; hi = r >= tau; Wt = w.sum()
    return float((w * (hi & yy)).sum() / Wt - (w * (hi & ~yy)).sum() / Wt * tau / (1 - tau))


def wauc(s, yy, w):
    sn, wn = s[~yy], w[~yy]; o = np.argsort(sn); sn, wn = sn[o], wn[o]; cw = np.concatenate([[0.0], np.cumsum(wn)])
    lo, hi = np.searchsorted(sn, s[yy], 'left'), np.searchsorted(sn, s[yy], 'right')
    return float((w[yy] * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (w[yy].sum() * wn.sum()))


# ---------------- Task 21 ----------------
hw = np.isfinite(wt) & (wt > 0)
rows = []
for k in P:
    hi = R9[k] >= TAU
    rows.append(dict(panel=k, features=' '.join(P[k]), US_dollars=price(P[k]), AUC=T.auc(LP[k], y), OE_9y=float(y.sum() / R9[k].sum()), slope=slope(R9[k], y.astype(float)),
                     NB=nb(R9[k], y, TAU), NB_half=nb(R9[k], y, TAU / 2), NB_double=nb(R9[k], y, TAU * 2),
                     AUC_weighted=wauc(LP[k][hw], y[hw], wt[hw]), NB_weighted=nb(R9[k][hw], y[hw], TAU, wt[hw]),
                     per1000_cost=1000 * price(P[k]), per1000_flagged=1000 * float(hi.mean()), per1000_deaths_flagged=1000 * float((hi & y).mean()),
                     per1000_deaths_missed=1000 * float((~hi & y).mean())))
panels = pd.DataFrame(rows); pc = panels.set_index('panel')
pairs = [('matched', RT), ('aware@routine', RT), (RT, 'age+sex only')]
rng = np.random.default_rng(20261107); B = {f'dAUC {a} - {b}': [] for a, b in pairs}; B.update({'dNB matched - routine': [], 'dNB aware@routine - routine': []})
for _ in range(NBOOT):
    i = rng.integers(0, n, n)
    if y[i].all() or (~y[i]).all():
        continue
    A = {k: T.auc(LP[k][i], y[i]) for k in set(x_ for p in pairs for x_ in p)}
    for a, b in pairs:
        B[f'dAUC {a} - {b}'].append(A[a] - A[b])
    nr = nb(R9[RT][i], y[i], TAU)
    B['dNB matched - routine'].append(nb(R9['matched'][i], y[i], TAU) - nr); B['dNB aware@routine - routine'].append(nb(R9['aware@routine'][i], y[i], TAU) - nr)
point = {f'dAUC {a} - {b}': float(pc.AUC[a] - pc.AUC[b]) for a, b in pairs}
point['dNB matched - routine'] = float(pc.NB['matched'] - pc.NB[RT]); point['dNB aware@routine - routine'] = float(pc.NB['aware@routine'] - pc.NB[RT])
spl = {}
for k in [RT, 'matched', 'aware@routine']:
    A_, B_ = X.spline_design(Ztr, Z, M.BASE + list(P[k])); spl[k] = M.Cox(1.0).fit(A_, ttr, etr).lp(B_)
sp = {k: T.auc(v, y) for k, v in spl.items()}; d1, d2 = [], []
for _ in range(NSUB):
    i = rng.integers(0, n, n); ar = T.auc(spl[RT][i], y[i])
    d1.append(T.auc(spl['matched'][i], y[i]) - ar); d2.append(T.auc(spl['aware@routine'][i], y[i]) - ar)
sp.update({'dAUC matched - routine': sp['matched'] - sp[RT], 'dAUC matched CI': ci(d1), 'dAUC aware - routine': sp['aware@routine'] - sp[RT], 'dAUC aware CI': ci(d2)})
sub = []
for g, msk in {'female': xte.sex.to_numpy() == 2, 'male': xte.sex.to_numpy() == 1, 'age40-59': xte.age.to_numpy() < 60, 'age60-79': xte.age.to_numpy() >= 60}.items():
    idx = np.flatnonzero(msk); row = dict(group=g, n=len(idx), deaths=int(y[idx].sum()))
    if y[idx].sum() < 20:
        sub.append(dict(row, note='fewer than 20 deaths')); continue
    for k in [RT, 'matched', 'aware@routine']:
        row['AUC ' + k] = T.auc(LP[k][idx], y[idx]); row['OE ' + k] = float(y[idx].sum() / R9[k][idx].sum())
    dd = []
    for _ in range(NSUB):
        j = idx[rng.integers(0, len(idx), len(idx))]
        if y[j].all() or (~y[j]).all():
            continue
        dd.append(T.auc(LP['matched'][j], y[j]) - T.auc(LP[RT][j], y[j]))
    row['dAUC matched - routine'] = row['AUC matched'] - row['AUC ' + RT]; row['dAUC CI lo'], row['dAUC CI hi'] = ci(dd); sub.append(row)
pre = W / 'task21_final' / f'{P_}t21_UKHLS_'
panels.to_csv(f'{pre}panels.csv', index=False); pd.DataFrame(sub).to_csv(f'{pre}subgroups.csv', index=False)
out = dict(cohort='UKHLS', horizon=H, tau=TAU, test=counts, n=int(n), deaths=int(y.sum()), weighted_n=int(hw.sum()), boot={k: ci(v) for k, v in B.items()},
           point=point, spline=sp, subgroups=sub, prices={k: price(v) for k, v in P.items()})
json.dump(out, open(f'{pre}endpoints.json', 'w'), indent=1, default=float)
pd.DataFrame({'y': y.astype(int), **{'cox|' + k: LP[k] for k in P}}).to_csv(HERE / f'{P_}t27_UKHLS_person.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print('UKHLS n', n, 'deaths', int(y.sum())); print(panels.round(4).to_string())
for k in point:
    print(f'  {k:50s} {point[k]:+.4f} {out["boot"][k]}')
print('SPLINE', json.dumps(sp)); print(pd.DataFrame(sub).round(4).to_string())


# ---------------- Task 23 (local recalibration; binary outcome as CHNS: complementary log-log) ----------------
def cloglog(lp, yy, b0):
    Xd = np.column_stack([np.ones_like(lp), lp]); b = np.array(b0, float)
    for _ in range(50):
        eta = Xd @ b; mu = 1 - np.exp(-np.exp(eta)); mu = np.clip(mu, 1e-9, 1 - 1e-9); dmu = np.exp(eta - np.exp(eta))
        wgt = dmu ** 2 / (mu * (1 - mu)); z = (yy - mu) / dmu
        st = np.linalg.solve(Xd.T @ (wgt[:, None] * Xd), Xd.T @ (wgt * z)); b = b + st
        if np.abs(st).max() < 1e-8:
            break
    return b


NS, REPS, SEED, MIN_TEST, MIN_LOCAL_DEATHS = (200, 500, 1000), (50 if DRY else 200), 20261110, 1000, 10
rows, whole = [], {}
for pname, k in [('matched', 'matched'), ('routine', RT), ('aware@routine', 'aware@routine')]:
    lp, rh, H0h = LP[k], R9[k], H0[k]
    O, E_ = y.sum(), rh.sum()
    whole[pname] = dict(OE=float(O / E_), lo=float(O / E_ * np.exp(-1.96 / np.sqrt(O))), hi=float(O / E_ * np.exp(1.96 / np.sqrt(O))), deaths=int(O), n=int(n))
    for nl in NS:
        if n - nl < MIN_TEST:
            continue
        for rep in range(REPS):
            rg = np.random.default_rng(SEED + 1000 * rep + nl); loc = np.zeros(n, bool); loc[rg.choice(n, nl, replace=False)] = True; tst = ~loc
            Ol = int(y[loc].sum())
            if Ol < MIN_LOCAL_DEATHS:
                rows.append(dict(panel=pname, n=nl, rep=rep, local_deaths=Ol, method='skipped')); continue
            El = rh[loc].sum(); recal = bool(abs(np.log(Ol / El)) > 1.96 / np.sqrt(Ol))
            al, be = cloglog(lp[loc], y[loc].astype(float), [np.log(H0h), 1.0]); rB = 1 - np.exp(-np.exp(al + be * lp))
            R = {'A import': rh, 'B recalibrate': rB}; R['T test then recalibrate'] = R['B recalibrate'] if recal else R['A import']
            for mth, r in R.items():
                oe = float(y[tst].sum() / r[tst].sum())
                rows.append(dict(panel=pname, n=nl, rep=rep, local_deaths=Ol, method=mth, OE=oe, abs_log_OE=abs(np.log(oe)), NB=nb(r[tst], y[tst], TAU),
                                 flagged=float((r >= TAU)[tst].mean()), recal=int(recal) if mth[0] == 'T' else np.nan))
D = pd.DataFrame(rows, columns=['panel', 'n', 'rep', 'local_deaths', 'method', 'OE', 'abs_log_OE', 'NB', 'flagged', 'recal'])
D.to_csv(W / 'task23_local' / f'{P_}t23_UKHLS_reps.csv', index=False)
S23 = dict(cohort='UKHLS', n=n, horizon_months=H, tau=TAU, reps=REPS, whole_cohort_import=whole, cells=[])
okr = D[D.method != 'skipped']
for (p, nl), g in okr.groupby(['panel', 'n']):
    w = g.pivot_table(index='rep', columns='method', values=['abs_log_OE', 'NB', 'recal']); A_ = w[('abs_log_OE', 'A import')]
    cell = dict(panel=p, n=int(nl), reps_used=int(len(w)), skipped=int((D[(D.panel == p) & (D.n == nl)].method == 'skipped').sum()),
                abs_log_OE_import=float(A_.mean()), recal_share_T=float(w[('recal', 'T test then recalibrate')].mean()))
    for mth in ['B recalibrate', 'T test then recalibrate']:
        dOE = w[('abs_log_OE', mth)] - A_; dNB = w[('NB', mth)] - w[('NB', 'A import')]
        cell[mth] = dict(abs_log_OE=float(w[('abs_log_OE', mth)].mean()), better_calibrated_share=float((dOE < 0).mean()),
                         dNB_mean=float(dNB.mean()), dNB_p2_5=float(dNB.quantile(.025)), dNB_p97_5=float(dNB.quantile(.975)))
    b = cell['B recalibrate']
    cell['reading_B'] = 'helps' if b['better_calibrated_share'] >= 0.70 and b['dNB_mean'] >= -0.001 else 'not needed' if b['better_calibrated_share'] < 0.50 else 'uncertain'
    S23['cells'].append(cell)
json.dump(S23, open(W / 'task23_local' / f'{P_}t23_UKHLS_summary.json', 'w'), indent=1)
print('T23 whole-cohort O/E', {k: round(v['OE'], 3) for k, v in whole.items()})
for c in S23['cells']:
    print('  T23', c['panel'], c['n'], c['reading_B'], round(c['B recalibrate']['better_calibrated_share'], 2), round(c['B recalibrate']['dNB_mean'], 4))

# ---------------- Task 24 (CoAI; US prices) ----------------
pub = {p['budget']: p['features'] for p in S['coai_published']['panels']}; bun = {p['budget']: p['features'] for p in S['coai_bundled']['panels']}
bm, br = sorted(pub)
P24 = {'routine': S['routine']['features'], 'ours matched': S['matched']['features'], 'ours target-aware': S['aware_at_routine']['features'],
       'CoAI @matched price': pub[bm], 'CoAI @routine price': pub[br], 'CoAI bundled @matched price': bun[bm], 'CoAI bundled @routine price': bun[br]}
PAIRS = [('ours target-aware', 'CoAI @routine price'), ('ours matched', 'CoAI @matched price'),
         ('ours target-aware', 'CoAI bundled @routine price'), ('ours matched', 'CoAI bundled @matched price')]
HYP = S['coai_published']['hyper']; known_tr = etr | (ttr >= M.HORIZON); ytr = (etr & (ttr <= M.HORIZON))[known_tr].astype(int)
S24, Sown, rows = {}, {}, []
for k, f in P24.items():
    cols = M.BASE + list(f); A = Ztr[cols].to_numpy(float); B_ = Z[cols].to_numpy(float)
    S24[k] = M.Cox(1.0).fit(A, ttr, etr).lp(B_); mu, sd = A[known_tr].mean(0), A[known_tr].std(0)
    lr = LogisticRegression(penalty=HYP['penalty'], C=HYP['C'], solver='saga', max_iter=20000, tol=1e-6).fit((A[known_tr] - mu) / sd, ytr)
    Sown[k] = lr.decision_function((B_ - mu) / sd)
    rows.append(dict(panel=k, features=' '.join(f), price=price(f), metric='AUC', cox=T.auc(S24[k], y), coai_own_model=T.auc(Sown[k], y)))
p24 = pd.DataFrame(rows).set_index('panel'); keys = [f'd {a} - {b}' for a, b in PAIRS]
rng = np.random.default_rng(20261110); Bs = {k: [] for k in keys}; Bo = {k: [] for k in keys}
for _ in range(NBOOT):
    i = rng.integers(0, n, n)
    if y[i].all() or (~y[i]).all():
        continue
    for (a, b), kk in zip(PAIRS, keys):
        Bs[kk].append(T.auc(S24[a][i], y[i]) - T.auc(S24[b][i], y[i])); Bo[kk].append(T.auc(Sown[a][i], y[i]) - T.auc(Sown[b][i], y[i]))
out24 = dict(cohort='UKHLS', n=int(n), events=int(y.sum()), metric='AUC', identical_panels={f'{a} | {b}': set(P24[a]) == set(P24[b]) for a, b in PAIRS},
             point={kk: float(p24.cox[a] - p24.cox[b]) for (a, b), kk in zip(PAIRS, keys)}, boot={k: ci(v) for k, v in Bs.items()},
             point_coai_own_model={kk: float(p24.coai_own_model[a] - p24.coai_own_model[b]) for (a, b), kk in zip(PAIRS, keys)},
             boot_coai_own_model={k: ci(v) for k, v in Bo.items()}, prices={k: price(f) for k, f in P24.items()})
pre24 = W / 'task24_coai' / f'{P_}t24_UKHLS_'
p24.reset_index().to_csv(f'{pre24}panels.csv', index=False); json.dump(out24, open(f'{pre24}endpoints.json', 'w'), indent=1)
pd.DataFrame({'y': y.astype(int), **{'cox|' + k: v for k, v in S24.items()}}).to_csv(f'{pre24}person.csv', index=False)
print(p24.round(4).to_string()); print('T24', json.dumps(out24['point']), json.dumps(out24['boot']))
print('counts', json.dumps(counts, default=str))
