#!/usr/bin/env python3
"""Task 29 E1-E2 (PLAN.md): pooling sensitivity (DerSimonian-Laird as published; REML with Hartung-Knapp; by metric; leave one out),
non-inferiority at -0.005, and the -0.01 margin as a share of each cohort's routine-over-age-and-sex gain.
Usage: t29_meta.py [published|full]  ('full' replaces the US and UK estimates with the complete-menu panels from valid_<cohort>.json)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, norm
HERE = Path(__file__).resolve().parent; T21 = HERE.parent / 'task21_final'
sys.path.insert(0, str(T21))
import t21lib as X  # noqa: E402
SET = sys.argv[1] if len(sys.argv) > 1 else 'published'
COH = ['NH3', 'GH', 'CHNS', 'UKHLS', 'CLHLS']; METRIC = {'NH3': 'C', 'GH': 'C', 'CHNS': 'AUC', 'UKHLS': 'AUC', 'CLHLS': 'C'}
KEY = {'H1': ('dC matched - routine', 'dAUC matched - routine (shared items)'), 'H2': ('dC aware@routine - routine', 'dAUC aware@routine - routine (shared items)')}
E = {c: json.load(open(T21 / f't21_{c}_endpoints.json')) for c in COH}


def get(c, h):
    if SET == 'full' and c in ('NH3', 'GH', 'UKHLS'):
        v = json.load(open(HERE / f'valid_{c}.json'))
        k = {'H1': 'new matched - routine', 'H2': 'new aware - routine'}[h]
        return v['contrasts'][k]['point'], *v['contrasts'][k]['ci']
    kc, ka = KEY[h]; d = E[c]; k = kc if kc in d['point'] else ka
    return d['point'][k], d['boot'][k][0], d['boot'][k][1]


def reml_hk(y, se):
    y, v = np.asarray(y, float), np.asarray(se, float) ** 2; k = len(y); tau2 = max(0.0, np.var(y, ddof=1) - v.mean())
    for _ in range(500):
        w = 1 / (v + tau2); mu = (w * y).sum() / w.sum()
        new = max(0.0, (w ** 2 * ((y - mu) ** 2 - v)).sum() / (w ** 2).sum() + 1 / w.sum())
        if abs(new - tau2) < 1e-12:
            tau2 = new; break
        tau2 = new
    w = 1 / (v + tau2); mu = (w * y).sum() / w.sum()
    q = (w * (y - mu) ** 2).sum() / (k - 1); se_hk = np.sqrt(max(q, 1e-300) / w.sum()); tq = tdist.ppf(.975, k - 1)
    return dict(est=float(mu), lo=float(mu - tq * se_hk), hi=float(mu + tq * se_hk), tau2=float(tau2), k=k)


def pool_set(names, h):
    rows = [get(c, h) for c in names]; y = [r[0] for r in rows]; lo = [r[1] for r in rows]; hi = [r[2] for r in rows]
    se = (np.array(hi) - np.array(lo)) / (2 * 1.96)
    dl = X.pool(y, lo, hi) if len(names) > 1 else dict(est=y[0], lo=lo[0], hi=hi[0])
    return dict(DL=dl, REML_HK=reml_hk(y, se) if len(names) > 2 else None)


out = dict(set=SET)
for h in ['H1', 'H2']:
    r = dict(per_cohort={c: dict(zip(['est', 'lo', 'hi'], get(c, h))) for c in COH}, all=pool_set(COH, h),
             C_only=pool_set([c for c in COH if METRIC[c] == 'C'], h), AUC_only=pool_set([c for c in COH if METRIC[c] == 'AUC'], h),
             leave_one_out={c: pool_set([d for d in COH if d != c], h) for c in COH})
    out[h] = r
# non-inferiority at -0.005 and the margin relative to each cohort's laboratory gain
ni, gain = {}, {}
for c in COH:
    e, lo, hi = get(c, 'H1'); ni[c] = dict(lower=lo, ni_001=lo > -0.01, ni_0005=lo > -0.005)
    p = pd.read_csv(T21 / f't21_{c}_panels.csv'); m = 'C' if 'C' in p.columns else 'AUC'; pc = p.set_index('panel')[m]
    rt = 'routine' if 'routine' in pc.index else 'routine (shared items)'
    g = float(pc[rt] - pc['age+sex only']); gain[c] = dict(routine_gain=g, margin_share_001=0.01 / g if g > 0 else None)
pl = out['H1']['all']['DL']; ni['pooled'] = dict(lower=pl['lo'], ni_001=pl['lo'] > -0.01, ni_0005=pl['lo'] > -0.005)
out['noninferiority'] = ni; out['margin_vs_gain'] = gain
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(HERE / f'meta_{SET}.json', 'w'), indent=1, default=float)
