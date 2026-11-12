#!/usr/bin/env python3
"""Post hoc data-quality sensitivity analyses (pre-submission code review items A3 and A5; rules fixed before any result was seen). Main analyses unchanged.
CLHLS (A3): values impossible on their own documented scale or contradicted within the same draw are set missing -- haemoglobin above 25 g/dL,
  MPV above 20 fL, HDL cholesterol above total cholesterol. People thereby incomplete drop out of the complete-case analysis.
CHNS (A5): deaths whose recorded death year (DOD_Y) is before the 2009 blood draw (values 1000-2008, i.e. 1009, 2005, 2008) are removed.
Reports the Task 21 contrasts (matched - routine, target-aware - routine) with 1000 bootstrap resamples, and the pooled H1/H2 with the changed cohort
estimate substituted (other cohorts as frozen). Output: sens_data_quality.txt"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
import clhls_lib as C  # noqa: E402
M.L.low_priority()
out = []; ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr)


def contrasts(score, idx_fn, n, P, rng):
    pt = {f'{a} - {b}': idx_fn(score[a], slice(None)) - idx_fn(score[b], slice(None)) for a, b in P}
    bs = {k: [] for k in pt}
    for _ in range(1000):
        i = rng.integers(0, n, n)
        for a, b in P:
            bs[f'{a} - {b}'].append(idx_fn(score[a], i) - idx_fn(score[b], i))
    return {k: (pt[k], ci(v)) for k, v in bs.items()}


# ---- CLHLS ----
S = json.load(open(HERE / 'selected_clhls.json'))
x, _ = C.load_clhls(); n0 = len(x)
bad = (x.hgb > 25) | (x.mpv > 20) | (x.hdl > x.tc)
out.append(f'CLHLS: {int(bad.sum())} of {n0} analysed people have at least one impossible value (hgb>25 g/dL: {int((x.hgb > 25).sum())}, '
           f'mpv>20 fL: {int((x.mpv > 20).sum())}, HDL>TC: {int((x.hdl > x.tc).sum())}); they are removed in this sensitivity analysis.')
xs = x[~bad]
Ztr = T.transform_sub(xtr, xtr, C.MENU); P = {'routine': C.ROUTINE_CL, 'matched': S['matched']['features'], 'aware@routine': S['aware_at_routine']['features']}
for grp in ['65-79', '80+']:
    g = xs[xs.age_group == grp]; Z = T.transform_sub(g, xtr, C.MENU); t, e = C.outcome(g)
    sc = {k: M.Cox(1.0).fit(Ztr[M.BASE + list(f)].to_numpy(), ttr, etr).lp(Z[M.BASE + list(f)].to_numpy()) for k, f in P.items()}
    r = contrasts(sc, lambda s, i: M.cindex(s[i], t[i], e[i]), len(t), [('matched', 'routine'), ('aware@routine', 'routine')], np.random.default_rng(20261107))
    out.append(f'  CLHLS {grp}: n {len(t)}, deaths {int(e.sum())}; ' + '; '.join(f'dC {k} {v[0]:+.4f} [{v[1][0]:+.4f}, {v[1][1]:+.4f}]' for k, v in r.items()))
    if grp == '65-79':
        clhls_main = r

# ---- CHNS ----
SC = json.load(open(W / 'task20_chns' / 'selected_cn.json'))
xc, _ = T.load_chns()
m = __import__('pandas').read_sas(T.CHNS / 'mast_pub_12.sas7bdat', format='sas7bdat'); m.columns = [k.upper() for k in m.columns]; m = m.set_index('IDIND')
dy = m.DOD_Y.reindex(xc.index); conflict = (xc.died == 1) & (dy >= 1000) & (dy <= 2008)
out.append(f'CHNS: {int(conflict.sum())} of {int(xc.died.sum())} deaths have a recorded death year before the 2009 blood draw '
           f'({sorted(dy[conflict].astype(int).tolist())}); removed.')
xc = xc[~conflict]; y = xc.died.to_numpy().astype(bool)
Ztc = T.transform_sub(xtr, xtr, T.SHARED); Zc = T.transform_sub(xc, xtr, T.SHARED)
Pc = {'routine': T.CN_ROUTINE_SHARED, 'matched': SC['matched']['features'], 'aware@routine': SC['aware_at_routine']['features']}
scc = {k: M.Cox(1.0).fit(Ztc[M.BASE + list(f)].to_numpy(), ttr, etr).lp(Zc[M.BASE + list(f)].to_numpy()) for k, f in Pc.items()}
rc = contrasts(scc, lambda s, i: T.auc(s[i], y[i]), len(y), [('matched', 'routine'), ('aware@routine', 'routine')], np.random.default_rng(20261107))
out.append(f'  CHNS: n {len(y)}, deaths {int(y.sum())}; ' + '; '.join(f'dAUC {k} {v[0]:+.4f} [{v[1][0]:+.4f}, {v[1][1]:+.4f}]' for k, v in rc.items()))

# ---- pooled with both substitutions ----
pool = json.load(open(W / 'task21_final' / 't21_pooled_final.json'))
for h, key in [('H1', 'matched - routine'), ('H2', 'aware@routine - routine')]:
    est, lo, hi = [], [], []
    for c, v in pool[h]['per_cohort'].items():
        if c == 'CLHLS':
            v = dict(est=clhls_main[key][0], lo=clhls_main[key][1][0], hi=clhls_main[key][1][1])
        if c == 'CHNS':
            v = dict(est=rc[key][0], lo=rc[key][1][0], hi=rc[key][1][1])
        est.append(v['est']); lo.append(v['lo']); hi.append(v['hi'])
    p = X.pool(est, lo, hi)
    out.append(f'Pooled {h} with both sensitivity substitutions: {p["est"]:+.4f} [{p["lo"]:+.4f}, {p["hi"]:+.4f}], I2 {100 * p["I2"]:.0f}% '
               f'(main: {pool[h]["pooled"]["est"]:+.4f} [{pool[h]["pooled"]["lo"]:+.4f}, {pool[h]["pooled"]["hi"]:+.4f}])')
(HERE / 'sens_data_quality.txt').write_text('\n'.join(out) + '\n', encoding='utf-8'); print('\n'.join(out))
