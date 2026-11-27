#!/usr/bin/env python3
"""Task 18 final run (plan_dm_FROZEN_2026-11-06.md). Usage: run_dm.py [--dry]
--dry  development layer only: train on C, test on D, 200 bootstrap draws; checks that the code runs (files dry_*).
plain  train on C+D, test on E+F (primary) and A+B (secondary), 1000 bootstrap draws. Run once; held-out files are read only here.
Panels come from selected_panels.json (chosen on C+D before freezing) and are not re-selected."""
import json
import sys
import numpy as np
import pandas as pd
import dmlib as L
L.low_priority()
DRY = '--dry' in sys.argv
PRE = 'dry_' if DRY else ''
SEL = json.load(open('selected_panels.json'))
TRAIN = ['C'] if DRY else ['C', 'D']
TESTS = {'D': ['D']} if DRY else {'EF': ['E', 'F'], 'AB': ['A', 'B']}
NBOOT = 200 if DRY else 1000
NSUB = 200
TAUS, EPS, EPS0 = (0.10, 0.20), (0.01, 0.02, 0.05), 0.02
BASE = ['age', 'female']
ROUT = 'std:CMP+CBC+LIPID+A1C'

# ---------------- panels ----------------
PAN = {'none': []}
for s, v in L.STANDARD.items():
    PAN['std:' + s] = v
for s, v in L.STANDARD.items():
    PAN['aware@' + s] = SEL['grid'][f'{L.cost(v):.2f}']['items']
PAN['matched'] = SEL['matched']['items'] if SEL['matched'] else None
PAN['stage1'] = SEL['stage1']['items']; PAN['stage2'] = SEL['stage2']['items']
PAN['all'] = list(L.ITEMS)
PAN = {k: v for k, v in PAN.items() if v is not None}
COST = {k: (L.cost(v) if v else 0.0) for k, v in PAN.items()}

# ---------------- training ----------------
xtr, ctr = L.load(TRAIN); ttr, etr = L.outcome(xtr); Ztr = L.transform(xtr)
cols = lambda k, base=BASE: base + L.features_of(PAN[k])  # noqa: E731
M = {k: L.Cox(1.0).fit(Ztr[cols(k)].to_numpy(), ttr, etr) for k in PAN}
folds = np.array_split(np.random.default_rng(1).permutation(len(ttr)), 5)
stages = [L.features_of(PAN[k]) for k in ('none', 'stage1', 'stage2')]
rcv = L.crossfit_risk(Ztr, ttr, etr, stages, folds)
FLIP = {tau: [L.FlipModel(tau).fit(rcv[s], rcv[-1]) for s in range(2)] for tau in TAUS}
SCOST = [0.0, COST['stage1'], COST['stage2']]
# clinical-base sensitivity and CVD (report only)
SENS = ['none', ROUT, 'matched', 'aware@CMP+CBC+LIPID+A1C', 'stage2', 'all']
SENS = [k for k in SENS if k in PAN]
cltr = L.clinical(xtr); oktr = cltr.notna().all(1).to_numpy()
Ztrc = pd.concat([Ztr, cltr], axis=1)
MC = {k: L.Cox(1.0).fit(Ztrc[cols(k, BASE + L.CLIN)].to_numpy()[oktr], ttr[oktr], etr[oktr]) for k in SENS}
tcv, ecv = L.outcome(xtr, 'cvd')
MV = {k: L.Cox(1.0).fit(Ztr[cols(k)].to_numpy(), tcv, ecv) for k in SENS}


def seq(R, tau, eps, n):
    st, risk, cp = L.sequential_policy(np.array([R['none'], R['stage1'], R['stage2']]), FLIP[tau], eps, SCOST)
    return st, risk, cp


def ci(a):
    a = np.asarray(a, float)
    return [float(np.quantile(a, .025)), float(np.quantile(a, .975))]


def evaluate(name, cycles):
    x, counts = L.load(cycles); t, e = L.outcome(x); Z = L.transform(x, ref=xtr)
    n = len(t); w = x.weight.to_numpy(float)
    R = {k: M[k].risk(Z[cols(k)].to_numpy()) for k in PAN}
    LP = {k: M[k].lp(Z[cols(k)].to_numpy()) for k in PAN}
    ll = {k: L.partial_loglik(LP[k], t, e) for k in PAN}
    known = e | (t >= L.HORIZON)
    rows = []
    for k in PAN:
        oe, sl = L.calibration(R[k], t, e)
        rows.append(dict(set=name, panel=k, items=' '.join(PAN[k]), cost=COST[k], C=L.cindex(LP[k], t, e), C_w=L.cindex(LP[k], t, e, w),
                         ll_share=(ll[k] - ll['none']) / (ll['all'] - ll['none']), NB10=L.net_benefit(R[k], t, e, .1), NB20=L.net_benefit(R[k], t, e, .2),
                         NB10_w=L.net_benefit(R[k], t, e, .1, w), OE=oe, slope=sl))
    panels = pd.DataFrame(rows)
    # sequential policies
    srows, SEQ = [], {}
    for tau in TAUS:
        full = R['stage2'] >= tau; rout = R[ROUT] >= tau
        for eps in EPS:
            st, risk, cp = seq(R, tau, eps, n); hi = risk >= tau
            SEQ[(tau, eps)] = (risk, cp)
            srows.append(dict(set=name, tau=tau, eps=eps, mean_cost=float(cp.mean()), cost_share=float(cp.mean() / COST[ROUT]), any_lab=float((st > 0).mean()),
                              stage0=float((st == 0).mean()), stage1=float((st == 1).mean()), stage2=float((st == 2).mean()),
                              agree_stage2=float((hi == full).mean()), agree_routine=float((hi == rout).mean()),
                              sens=float(hi[known & e].mean()), spec=float((~hi[known & ~e]).mean()), flagged=float(hi.mean()),
                              C=L.cindex(np.log(risk), t, e), NB=L.net_benefit(risk, t, e, tau), NB_routine=L.net_benefit(R[ROUT], t, e, tau),
                              NB_stage2=L.net_benefit(R['stage2'], t, e, tau), NB_w=L.net_benefit(risk, t, e, tau, w)))
    seqd = pd.DataFrame(srows)
    # bootstrap (models fixed, test people resampled)
    rng = np.random.default_rng(20261106)
    pairs = [('matched', ROUT), ('aware@CMP+CBC+LIPID+A1C', ROUT)] + [('aware@' + s, 'std:' + s) for s in L.STANDARD]
    pairs = [p for p in pairs if p[0] in PAN]
    uniq = sorted(set(k for p in pairs for k in p))
    B = {f'dC {a} - {b}': [] for a, b in pairs}
    for tau in TAUS:
        B[f'dNB seq{tau} - routine'] = []; B[f'cost_share seq{tau}'] = []; B[f'dC seq{tau} - routine'] = []
    B['dNB10 matched - routine'] = []; B['dNB10 aware@routine - routine'] = []
    for _ in range(NBOOT):
        i = rng.integers(0, n, n); ti, ei = t[i], e[i]
        Cb = {k: L.cindex(LP[k][i], ti, ei) for k in uniq}
        for a, b in pairs:
            B[f'dC {a} - {b}'].append(Cb[a] - Cb[b])
        for tau in TAUS:
            risk, cp = SEQ[(tau, EPS0)]
            B[f'dNB seq{tau} - routine'].append(L.net_benefit(risk[i], ti, ei, tau) - L.net_benefit(R[ROUT][i], ti, ei, tau))
            B[f'cost_share seq{tau}'].append(cp[i].mean() / COST[ROUT])
            B[f'dC seq{tau} - routine'].append(L.cindex(np.log(risk[i]), ti, ei) - Cb[ROUT])
        nbr = L.net_benefit(R[ROUT][i], ti, ei, .1)
        if 'matched' in PAN:
            B['dNB10 matched - routine'].append(L.net_benefit(R['matched'][i], ti, ei, .1) - nbr)
        B['dNB10 aware@routine - routine'].append(L.net_benefit(R['aware@CMP+CBC+LIPID+A1C'][i], ti, ei, .1) - nbr)
    boot = {k: dict(lo=ci(v)[0], hi=ci(v)[1], mean=float(np.mean(v))) for k, v in B.items() if len(v)}
    point = {}
    pc = panels.set_index('panel')
    for a, b in pairs:
        point[f'dC {a} - {b}'] = float(pc.C[a] - pc.C[b])
    s0 = seqd[seqd.eps == EPS0].set_index('tau')
    for tau in TAUS:
        point[f'dNB seq{tau} - routine'] = float(s0.NB[tau] - s0.NB_routine[tau]); point[f'cost_share seq{tau}'] = float(s0.cost_share[tau])
        point[f'dC seq{tau} - routine'] = float(s0.C[tau] - pc.C[ROUT])
    if 'matched' in PAN:
        point['dNB10 matched - routine'] = float(pc.NB10['matched'] - pc.NB10[ROUT])
    point['dNB10 aware@routine - routine'] = float(pc.NB10['aware@CMP+CBC+LIPID+A1C'] - pc.NB10[ROUT])
    # frozen criteria
    g = lambda k: boot[k]  # noqa: E731
    crit = {}
    crit['P1'] = bool('matched' in PAN and COST['matched'] <= 0.8 * COST[ROUT] and g(f'dC matched - {ROUT}')['lo'] > -0.01)
    a1 = s0.loc[0.10]
    crit['P2'] = bool(g('cost_share seq0.1')['hi'] <= 0.60 and g('dNB seq0.1 - routine')['lo'] > -0.005 and a1.agree_stage2 >= 0.97)
    crit['P3'] = bool(all(PAN['aware@' + s] != PAN['std:' + s] and g(f'dC aware@{s} - std:{s}')['lo'] > 0 for s in ('BMP', 'LIPID')))
    crit['S1'] = bool(g(f'dC aware@CMP+CBC+LIPID+A1C - {ROUT}')['lo'] > 0)
    a2 = s0.loc[0.20]
    crit['S2'] = bool(g('cost_share seq0.2')['hi'] <= 0.60 and g('dNB seq0.2 - routine')['lo'] > -0.005 and a2.agree_stage2 >= 0.97)
    crit['WIN'] = crit['P1'] and crit['P2'] and crit['P3']
    # report-only: subgroups, clinical base, CVD, by cycle
    sub = []
    groups = {'female': x.sex.to_numpy() == 2, 'male': x.sex.to_numpy() == 1, 'age40-59': x.age.to_numpy() < 60, 'age60-79': x.age.to_numpy() >= 60}
    for r_, lab in [(1, 'Mexican American'), (2, 'Other Hispanic'), (3, 'Non-Hispanic White'), (4, 'Non-Hispanic Black'), (5, 'Other race')]:
        groups[lab] = x.race.to_numpy() == r_
    groups.update({f'cycle {c}': x.cycle.to_numpy() == c for c in cycles})
    rs = np.random.default_rng(11)
    for gname, mask in groups.items():
        idx = np.flatnonzero(mask)
        if e[idx].sum() < 20:
            sub.append(dict(set=name, group=gname, n=len(idx), deaths=int(e[idx].sum()), note='fewer than 20 deaths; not estimated')); continue
        row = dict(set=name, group=gname, n=len(idx), deaths=int(e[idx].sum()))
        for k in ['none', ROUT, 'matched', 'aware@CMP+CBC+LIPID+A1C']:
            if k in PAN:
                row['C ' + k] = L.cindex(LP[k][idx], t[idx], e[idx])
        for k in ['matched', 'aware@CMP+CBC+LIPID+A1C']:
            if k in PAN:
                d = []
                for _ in range(NSUB):
                    j = idx[rs.integers(0, len(idx), len(idx))]
                    d.append(L.cindex(LP[k][j], t[j], e[j]) - L.cindex(LP[ROUT][j], t[j], e[j]))
                row[f'dC {k} - routine'] = row['C ' + k] - row['C ' + ROUT]; row[f'dC {k} lo'], row[f'dC {k} hi'] = ci(d)
        row['OE routine'], row['slope routine'] = L.calibration(R[ROUT][idx], t[idx], e[idx])
        if 'matched' in PAN:
            row['OE matched'], row['slope matched'] = L.calibration(R['matched'][idx], t[idx], e[idx])
        sub.append(row)
    subd = pd.DataFrame(sub)
    cl = L.clinical(x); ok = cl.notna().all(1).to_numpy(); Zc = pd.concat([Z, cl], axis=1)
    sens = []
    LPc = {k: MC[k].lp(Zc[cols(k, BASE + L.CLIN)].to_numpy()[ok]) for k in SENS}
    tv, ev = L.outcome(x, 'cvd')
    LPv = {k: MV[k].lp(Z[cols(k)].to_numpy()) for k in SENS}
    for k in SENS:
        sens.append(dict(set=name, analysis='clinical base', panel=k, n=int(ok.sum()), C=L.cindex(LPc[k], t[ok], e[ok]),
                         NB10=L.net_benefit(MC[k].risk(Zc[cols(k, BASE + L.CLIN)].to_numpy()[ok]), t[ok], e[ok], .1)))
        sens.append(dict(set=name, analysis='cvd death', panel=k, n=n, events=int(ev.sum()), C=L.cindex(LPv[k], tv, ev)))
    sensd = pd.DataFrame(sens)
    for k in [p for p in SENS if p not in ('none', ROUT, 'all')]:
        for lab, lp, tt, ee in [('clinical base', LPc, t[ok], e[ok]), ('cvd death', LPv, tv, ev)]:
            d = []
            for _ in range(NSUB):
                j = rs.integers(0, len(tt), len(tt))
                d.append(L.cindex(lp[k][j], tt[j], ee[j]) - L.cindex(lp[ROUT][j], tt[j], ee[j]))
            m_ = (sensd.analysis == lab) & (sensd.panel == k)
            sensd.loc[m_, 'dC vs routine'] = float(sensd.C[m_].iloc[0] - sensd.C[(sensd.analysis == lab) & (sensd.panel == ROUT)].iloc[0])
            sensd.loc[m_, 'dC lo'], sensd.loc[m_, 'dC hi'] = ci(d)
    return dict(counts=counts, n=n, deaths_9y=int(e.sum()), panels=panels, seq=seqd, boot=boot, point=point, crit=crit, sub=subd, sens=sensd)


res = {}
for name, cyc in TESTS.items():
    try:
        res[name] = evaluate(name, cyc)
    except Exception as ex:   # a held-out layer that cannot be harmonised is reported, not fixed (plan section 7)
        res[name] = dict(error=repr(ex)); print(name, 'FAILED', repr(ex), flush=True)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_colwidth', 50)
print('TRAIN', TRAIN, ctr, 'n', len(ttr), 'deaths', int(etr.sum()))
print('PANELS', json.dumps({k: [PAN[k], COST[k]] for k in PAN}))
for name, r in res.items():
    if 'error' in r:
        continue
    print('\n=====', name, r['counts'], 'n', r['n'], 'deaths within 9 years', r['deaths_9y'])
    print(r['panels'].round(4).to_string())
    print(r['seq'].round(4).to_string())
    for k in r['boot']:
        print(f"{k:55s} point {r['point'].get(k, float('nan')):+.4f}  95% [{r['boot'][k]['lo']:+.4f}, {r['boot'][k]['hi']:+.4f}]")
    print('CRITERIA', r['crit'])
    print(r['sub'].round(4).to_string()); print(r['sens'].round(4).to_string())
    for part in ['panels', 'seq', 'sub', 'sens']:
        r[part].to_csv(f'{PRE}results_{name}_{part}.csv', index=False)
json.dump({name: ({k: r[k] for k in ['counts', 'n', 'deaths_9y', 'boot', 'point', 'crit']} if 'error' not in r else r) for name, r in res.items()},
          open(f'{PRE}endpoints.json', 'w'), indent=1, default=str)
