#!/usr/bin/env python3
"""Task 25: run the frozen Task 21, 23 and 24 analyses once on CLHLS (plan_t25_FROZEN.md). Usage: run_t25_clhls.py [--dry]
Task 22 (value-of-information ordering) is not applicable: CLHLS has no CMP and no comparable urine ACR in all waves (plan_t22 section 2).
Training: NHANES 1999-2010 (A-F) restricted to the CLHLS menu; panels from selected_clhls.json (training only, Chinese prices).
Outcome: death within 60 months; main analysis ages 65-79, ages 80+ reported separately. tau = 5% 5-year risk (as NHANES 2011-14, 5 years).
Outputs (formats identical to the earlier cohorts, so the frozen pooling and verification scripts read them unchanged):
  ../task21_final/t21_CLHLS_endpoints.json (+ _panels.csv, _subgroups.csv); 80+: t21_CLHLS80_*.
  ../task23_local/t23_CLHLS_reps.csv, t23_CLHLS_summary.json
  ../task24_coai/t24_CLHLS_endpoints.json, _panels.csv, _person.csv
  t25_CLHLS_person.csv (person-level scores for the independent check of Task 21 contrasts)
--dry: train on NHANES C, 'CLHLS' = NHANES D+E aged 65-79 with the same menu and a 60-month outcome (development check only; outputs prefixed dry_)."""
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
import clhls_lib as C  # noqa: E402
M.L.low_priority()
DRY = '--dry' in sys.argv; P_ = 'dry_' if DRY else ''; H = C.H; TAU = 0.05
NBOOT, NSUB = (200, 100) if DRY else (1000, 200)
S = json.load(open(HERE / 'selected_clhls.json'))

# ---------------- data ----------------
if DRY:
    xtr, _ = M.load(['C']); d0, _ = M.load(['D', 'E']); xte = d0[d0.age >= 65].copy(); tt, ee = T.outcome_at(xte, H); xte['months'], xte['died'] = tt, ee.astype(int)
    xte['age_group'] = '65-79'; counts = {'pseudo': 'NHANES D+E aged 65-79'}
else:
    xtr, _ = M.load(list('ABCDEF')); xte, counts = C.load_clhls()
ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, C.MENU); Zall = T.transform_sub(xte, xtr, C.MENU)
tall, eall = C.outcome(xte)
P = {'age+sex only': [], 'CBC only': C.CBC_CL, 'routine': C.ROUTINE_CL, 'matched': S['matched']['features'],
     'aware@routine': S['aware_at_routine']['features'], 'all menu': C.MENU}
MOD = {}
for k, f in P.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr); MOD[k] = (m, cols, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H))


def calib(r, tt, ee):
    kn = ee | (tt >= H); rr, yy = np.clip(r[kn], 1e-4, 1 - 1e-4), ee[kn].astype(float)
    u = np.log(rr / (1 - rr)); A = np.column_stack([np.ones(len(u)), u]); b = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-A @ b)); Wt = p * (1 - p) + 1e-9; st = np.linalg.solve(A.T @ (Wt[:, None] * A), A.T @ (yy - p)); b += st
        if np.abs(st).max() < 1e-8:
            break
    return float(yy.sum() / rr.sum()), float(b[1])


def nb(r, tt, ee, tau):
    kn = ee | (tt >= H); W_ = kn.sum(); hi = r >= tau
    return float((hi & ee & kn).sum() / W_ - (hi & ~ee & kn).sum() / W_ * tau / (1 - tau))


ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731


# ---------------- Task 21 ----------------
def task21(mask, tag):
    Z, t, e = Zall[mask], tall[mask], eall[mask]; n = len(t)
    LP = {k: m.lp(Z[cols].to_numpy()) for k, (m, cols, _) in MOD.items()}
    RH = {k: T.risk_at(m, h0, Z[cols].to_numpy()) for k, (m, cols, h0) in MOD.items()}
    rows = []
    for k in P:
        oe, sl = calib(RH[k], t, e)
        rows.append(dict(panel=k, features=' '.join(P[k]), CN_yuan=(T.cost_cn(P[k]) if P[k] else 0.0), C=M.cindex(LP[k], t, e), OE=oe, slope=sl,
                         NB=nb(RH[k], t, e, TAU), NB_half=nb(RH[k], t, e, TAU / 2), NB_double=nb(RH[k], t, e, TAU * 2)))
    panels = pd.DataFrame(rows); pc = panels.set_index('panel')
    pairs = [('matched', 'routine'), ('aware@routine', 'routine'), ('routine', 'age+sex only')]
    rng = np.random.default_rng(20261107); B = {f'dC {a} - {b}': [] for a, b in pairs}; B.update({'dNB matched - routine': [], 'dNB aware@routine - routine': []})
    for _ in range(NBOOT):
        i = rng.integers(0, n, n); Cb = {k: M.cindex(LP[k][i], t[i], e[i]) for k in ['matched', 'aware@routine', 'routine', 'age+sex only']}
        for a, b in pairs:
            B[f'dC {a} - {b}'].append(Cb[a] - Cb[b])
        nr = nb(RH['routine'][i], t[i], e[i], TAU)
        B['dNB matched - routine'].append(nb(RH['matched'][i], t[i], e[i], TAU) - nr); B['dNB aware@routine - routine'].append(nb(RH['aware@routine'][i], t[i], e[i], TAU) - nr)
    point = {f'dC {a} - {b}': float(pc.C[a] - pc.C[b]) for a, b in pairs}
    point['dNB matched - routine'] = float(pc.NB['matched'] - pc.NB['routine']); point['dNB aware@routine - routine'] = float(pc.NB['aware@routine'] - pc.NB['routine'])
    spl = {}
    for k in ['routine', 'matched', 'aware@routine']:
        A_, B_ = X.spline_design(Ztr, Z, M.BASE + list(P[k])); spl[k] = M.Cox(1.0).fit(A_, ttr, etr).lp(B_)
    sp = {f'C {k}': M.cindex(v, t, e) for k, v in spl.items()}; d1, d2 = [], []
    for _ in range(NSUB):
        i = rng.integers(0, n, n); cr = M.cindex(spl['routine'][i], t[i], e[i])
        d1.append(M.cindex(spl['matched'][i], t[i], e[i]) - cr); d2.append(M.cindex(spl['aware@routine'][i], t[i], e[i]) - cr)
    sp.update({'dC matched - routine': sp['C matched'] - sp['C routine'], 'dC matched CI': ci(d1), 'dC aware - routine': sp['C aware@routine'] - sp['C routine'], 'dC aware CI': ci(d2)})
    sub = []
    sx = xte[mask]
    for g, msk in {'female': (sx.sex == 2).to_numpy(), 'male': (sx.sex == 1).to_numpy()}.items():
        idx = np.flatnonzero(msk); row = dict(group=g, n=len(idx), deaths=int(e[idx].sum()))
        if e[idx].sum() < 20:
            sub.append(dict(row, note='fewer than 20 deaths')); continue
        for k in ['routine', 'matched', 'aware@routine']:
            row['C ' + k] = M.cindex(LP[k][idx], t[idx], e[idx]); row['OE ' + k], row['slope ' + k] = calib(RH[k][idx], t[idx], e[idx])
        dd = [M.cindex(LP['matched'][j], t[j], e[j]) - M.cindex(LP['routine'][j], t[j], e[j]) for j in (idx[rng.integers(0, len(idx), len(idx))] for _ in range(NSUB))]
        row['dC matched - routine'] = row['C matched'] - row['C routine']; row['dC CI lo'], row['dC CI hi'] = ci(dd); sub.append(row)
    pre = W / 'task21_final' / f'{P_}t21_{tag}_'
    panels.to_csv(f'{pre}panels.csv', index=False); pd.DataFrame(sub).to_csv(f'{pre}subgroups.csv', index=False)
    out = dict(cohort=tag, horizon=H, tau=TAU, test=counts, n=int(n), deaths=int(e.sum()), boot={k: ci(v) for k, v in B.items()}, point=point, spline=sp)
    json.dump(out, open(f'{pre}endpoints.json', 'w'), indent=1, default=float)
    pd.DataFrame({'t': t, 'e': e.astype(int), **{'cox|' + k: LP[k] for k in P}}).to_csv(HERE / f'{P_}t25_{tag}_person.csv', index=False)
    print(tag, 'n', n, 'deaths', int(e.sum())); print(panels.round(4).to_string())
    for k in point:
        print(f'  {k:40s} {point[k]:+.4f} {out["boot"][k] if k in out["boot"] else ""}')
    return LP, RH, Z, t, e


main = (xte.age_group == '65-79').to_numpy()
LP, RH, Zm, tm, em = task21(main, 'CLHLS')
if (~main).sum() > 0 and eall[~main].sum() >= 20:
    task21(~main, 'CLHLS80')


# ---------------- Task 23 (local recalibration), main analysis population ----------------
def cox_slope(lp, t, e):
    o = np.argsort(-t, kind='stable'); last = np.searchsorted(-t[o], -t[o], side='right') - 1; x_, d = lp[o], e[o].astype(float); a = 1.0
    for _ in range(50):
        r = np.exp(a * x_ - (a * x_).max()); S0 = np.cumsum(r)[last]; S1 = np.cumsum(r * x_)[last]; S2 = np.cumsum(r * x_ * x_)[last]
        g = (d * (x_ - S1 / S0)).sum(); h = (d * (S2 / S0 - (S1 / S0) ** 2)).sum(); st = g / h; a += st
        if abs(st) < 1e-8:
            break
    return a


def breslow(lp, t, e, h):
    o = np.argsort(-t, kind='stable'); last = np.searchsorted(-t[o], -t[o], side='right') - 1
    return float((e[o] / np.cumsum(np.exp(lp[o]))[last])[t[o] <= h].sum())


NS, REPS, SEED = (200, 500, 1000), (50 if DRY else 200), 20261110
N = len(tm); known = em | (tm >= H); rows, whole = [], {}
for pname in ['matched', 'routine', 'aware@routine']:
    lp = LP[pname]; rh = RH[pname]
    O, E_ = em[known].sum(), rh[known].sum()
    whole[pname] = dict(OE=float(O / E_), lo=float(O / E_ * np.exp(-1.96 / np.sqrt(O))), hi=float(O / E_ * np.exp(1.96 / np.sqrt(O))), deaths=int(O), n=int(known.sum()))
    for n in NS:
        if N - n < 1000:
            continue
        for rep in range(REPS):
            rng = np.random.default_rng(SEED + 1000 * rep + n); loc = np.zeros(N, bool); loc[rng.choice(N, n, replace=False)] = True; tst = ~loc
            Ol = int(em[loc & known].sum())
            if Ol < 10:
                rows.append(dict(panel=pname, n=n, rep=rep, local_deaths=Ol, method='skipped')); continue
            El = rh[loc & known].sum(); recal = bool(abs(np.log(Ol / El)) > 1.96 / np.sqrt(Ol))
            a = cox_slope(lp[loc], tm[loc], em[loc]); H0 = breslow(a * lp[loc], tm[loc], em[loc], H); rB = 1 - np.exp(-H0 * np.exp(a * lp))
            R = {'A import': rh, 'B recalibrate': rB}; R['T test then recalibrate'] = R['B recalibrate'] if recal else R['A import']
            for k, r in R.items():
                oe = float(em[tst & known].sum() / r[tst & known].sum()); hi_ = r >= TAU; Wt = (tst & known).sum()
                nbv = float((hi_ & em & tst & known).sum() / Wt - (hi_ & ~em & tst & known).sum() / Wt * TAU / (1 - TAU))
                rows.append(dict(panel=pname, n=n, rep=rep, local_deaths=Ol, method=k, OE=oe, abs_log_OE=abs(np.log(oe)), NB=nbv,
                                 flagged=float(hi_[tst & known].mean()), recal=int(recal) if k[0] == 'T' else np.nan))
D = pd.DataFrame(rows, columns=['panel', 'n', 'rep', 'local_deaths', 'method', 'OE', 'abs_log_OE', 'NB', 'flagged', 'recal']); D.to_csv(W / 'task23_local' / f'{P_}t23_CLHLS_reps.csv', index=False)
S23 = dict(cohort='CLHLS', n=N, horizon_months=H, tau=TAU, reps=REPS, whole_cohort_import=whole, cells=[])
okr = D[D.method != 'skipped']
for (p, n), g in okr.groupby(['panel', 'n']):
    w = g.pivot_table(index='rep', columns='method', values=['abs_log_OE', 'NB', 'recal']); A_ = w[('abs_log_OE', 'A import')]
    cell = dict(panel=p, n=int(n), reps_used=int(len(w)), skipped=int((D[(D.panel == p) & (D.n == n)].method == 'skipped').sum()),
                abs_log_OE_import=float(A_.mean()), recal_share_T=float(w[('recal', 'T test then recalibrate')].mean()))
    for k in ['B recalibrate', 'T test then recalibrate']:
        dOE = w[('abs_log_OE', k)] - A_; dNB = w[('NB', k)] - w[('NB', 'A import')]
        cell[k] = dict(abs_log_OE=float(w[('abs_log_OE', k)].mean()), better_calibrated_share=float((dOE < 0).mean()),
                       dNB_mean=float(dNB.mean()), dNB_p2_5=float(dNB.quantile(.025)), dNB_p97_5=float(dNB.quantile(.975)))
    b = cell['B recalibrate']
    cell['reading_B'] = 'helps' if b['better_calibrated_share'] >= 0.70 and b['dNB_mean'] >= -0.001 else 'not needed' if b['better_calibrated_share'] < 0.50 else 'uncertain'
    S23['cells'].append(cell)
json.dump(S23, open(W / 'task23_local' / f'{P_}t23_CLHLS_summary.json', 'w'), indent=1)
print('T23 whole-cohort O/E', {k: round(v['OE'], 3) for k, v in whole.items()})

# ---------------- Task 24 (CoAI), main analysis population ----------------
cp = {p['budget']: p['features'] for p in S['coai']['panels']}; bm, br = sorted(cp)
P24 = {'routine': C.ROUTINE_CL, 'ours matched': S['matched']['features'], 'ours target-aware': S['aware_at_routine']['features'],
       'CoAI @matched price': cp[bm], 'CoAI @routine price': cp[br]}
PAIRS = [('ours target-aware', 'CoAI @routine price'), ('ours matched', 'CoAI @matched price')]
HYP = S['coai']['hyper']; known_tr = etr | (ttr >= M.HORIZON); ytr = (etr & (ttr <= M.HORIZON))[known_tr].astype(int)
S24, Sown, rows = {}, {}, []
for k, f in P24.items():
    cols = M.BASE + list(f); A = Ztr[cols].to_numpy(float); B_ = Zm[cols].to_numpy(float)
    S24[k] = M.Cox(1.0).fit(A, ttr, etr).lp(B_); mu, sd = A[known_tr].mean(0), A[known_tr].std(0)
    lr = LogisticRegression(penalty=HYP['penalty'], C=HYP['C'], solver='saga', max_iter=20000, tol=1e-6).fit((A[known_tr] - mu) / sd, ytr)
    Sown[k] = lr.decision_function((B_ - mu) / sd)
    rows.append(dict(panel=k, features=' '.join(f), price=T.cost_cn(f), metric='C', cox=M.cindex(S24[k], tm, em), coai_own_model=M.cindex(Sown[k], tm, em)))
p24 = pd.DataFrame(rows).set_index('panel'); keys = [f'd {a} - {b}' for a, b in PAIRS]
rng = np.random.default_rng(20261110); Bs = {k: [] for k in keys}; Bo = {k: [] for k in keys}
for _ in range(NBOOT):
    i = rng.integers(0, N, N)
    for (a, b), kk in zip(PAIRS, keys):
        Bs[kk].append(M.cindex(S24[a][i], tm[i], em[i]) - M.cindex(S24[b][i], tm[i], em[i])); Bo[kk].append(M.cindex(Sown[a][i], tm[i], em[i]) - M.cindex(Sown[b][i], tm[i], em[i]))
out24 = dict(cohort='CLHLS', n=int(N), events=int(em.sum()), metric='C', identical_panels={f'{a} | {b}': set(P24[a]) == set(P24[b]) for a, b in PAIRS},
             point={kk: float(p24.cox[a] - p24.cox[b]) for (a, b), kk in zip(PAIRS, keys)}, boot={k: ci(v) for k, v in Bs.items()},
             point_coai_own_model={kk: float(p24.coai_own_model[a] - p24.coai_own_model[b]) for (a, b), kk in zip(PAIRS, keys)},
             boot_coai_own_model={k: ci(v) for k, v in Bo.items()}, prices={k: T.cost_cn(f) for k, f in P24.items()})
pre24 = W / 'task24_coai' / f'{P_}t24_CLHLS_'
p24.reset_index().to_csv(f'{pre24}panels.csv', index=False); json.dump(out24, open(f'{pre24}endpoints.json', 'w'), indent=1)
pd.DataFrame({'t': tm, 'e': em.astype(int), **{'cox|' + k: v for k, v in S24.items()}}).to_csv(f'{pre24}person.csv', index=False)
print('T24', json.dumps(out24['point']), json.dumps(out24['boot']))
print('counts', json.dumps(counts, default=str))
