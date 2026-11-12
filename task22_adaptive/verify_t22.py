#!/usr/bin/env python3
"""Independent check of Task 22 (written separately from run_t22.py; no project library is imported).
From t22_<cohort>_person.csv: (1) re-derives each person's stopping point by applying the printed rule chart (risk intervals) to the
saved stage risks; (2) recomputes cost with the Medicare 2026 prices typed in here; (3) recomputes deaths missed, share of routine cost
and net benefit, and the H4 decision with its own bootstrap. Usage: verify_t22.py <NH3|GH|I> [--dry]"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
COH = sys.argv[1]; DRY = '--dry' in sys.argv; P = ('dry_' if DRY else '') + f't22_{COH}_'
p = pd.read_csv(HERE / (P + 'person.csv')); ep = json.load(open(HERE / (P + 'endpoints.json')))
H, TAU = ep['horizon'], 0.10
# Medicare CLFS 2026: CMP 10.56 + venipuncture 9.34; CBC 7.77; urine albumin 5.78 + urine creatinine 5.18; routine = CMP+CBC+lipid 13.39+HbA1c 9.71
PRICE = [0.0, 10.56 + 9.34, 10.56 + 9.34 + 7.77, 10.56 + 9.34 + 7.77 + 5.78 + 5.18]; ROUT = 10.56 + 9.34 + 7.77 + 13.39 + 9.71
chart = ep['chart_main']; stage_r = ['r_none', 'r_CMP', 'r_CMP+CBC', 'r_CMP+CBC+acr']; stage_c = ['clinical only', 'CMP', 'CMP+CBC']
ok = True


def check(name, cond, detail=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name, detail); ok &= bool(cond)


def inside(r, ivs):
    return np.any([(r >= a) & (r <= b) for a, b in ivs], axis=0) if ivs else np.zeros(len(r), bool)


k = np.zeros(len(p), int); go = np.ones(len(p), bool)
for j, s in enumerate(stage_c):
    cont = go & inside(p[stage_r[j]].to_numpy(), chart[s]); k[cont] = j + 1; go = cont
agree = (k == p.k_main.to_numpy()).mean()
check('stopping point re-derived from the printed chart', agree >= 0.999, f'agreement {agree:.5f} (grid step 0.00005)')
cost = np.array(PRICE)[p.k_main.to_numpy()]
check('cost recomputed from typed prices', np.allclose(cost, p.cost_main, atol=0.006), f'max diff {np.abs(cost - p.cost_main).max():.4f}')
r = np.choose(p.k_main.to_numpy(), [p[c].to_numpy() for c in stage_r])
check('final risk = risk at the stopping stage', np.allclose(r, p.r_main))
t, e = p.t.to_numpy(), p.e.to_numpy().astype(bool); known = e | (t >= H)


def stats(i):
    a, b = p.r_main.to_numpy()[i] >= TAU, p.r_routine.to_numpy()[i] >= TAU; ee, kk = e[i], known[i]
    nb = lambda h: ((h & ee & kk).sum() - (h & ~ee & kk).sum() * TAU / (1 - TAU)) / kk.sum()  # noqa: E731
    return cost[i].mean() / ROUT, 1000 * ((~a & ee).mean() - (~b & ee).mean()), nb(a) - nb(b)


s0 = stats(np.arange(len(p)))
tab = {d['policy']: d for d in ep['table']}; main = [k_ for k_ in tab if k_.startswith('VOI V=30000')][0]
check('share of routine cost', abs(s0[0] - tab[main]['share_routine']) < 1e-6, f'{s0[0]:.4f}')
check('deaths missed per 1000, VOI - routine', abs(s0[1] - (tab[main]['missed_per1000'] - tab['routine']['missed_per1000'])) < 1e-6, f'{s0[1]:+.4f}')
if H == 108:
    check('net benefit difference', abs(s0[2] - ep['H4']['dNB']) < 1e-9, f'{s0[2]:+.5f}')
rng = np.random.default_rng(99); B = np.array([stats(rng.integers(0, len(p), len(p))) for _ in range(1000)])
share_hi = np.quantile(B[:, 0], .975); dec = share_hi <= 0.50
if H == 108:
    nb_lo = np.quantile(B[:, 2], .025); dec = dec and nb_lo > -0.005
    print(f'own bootstrap: share upper {share_hi:.4f}, dNB lower {nb_lo:+.5f}, deaths missed diff 95% [{np.quantile(B[:, 1], .025):+.3f}, {np.quantile(B[:, 1], .975):+.3f}]')
else:
    print(f'own bootstrap: share upper {share_hi:.4f}, deaths missed diff 95% [{np.quantile(B[:, 1], .025):+.3f}, {np.quantile(B[:, 1], .975):+.3f}]')
check('H4 decision agrees with the main pipeline', dec == ep['H4']['pass'], f'verify {dec} vs main {ep["H4"]["pass"]}')
print('VERIFY ' + ('PASSED' if ok else 'FAILED'))
