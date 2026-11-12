#!/usr/bin/env python3
"""Post hoc diagnostic (after the frozen run; reported as such): is the CLHLS O/E of about 1.8 an artefact of the known-status O/E?
The frozen O/E counts deaths among everyone but drops survivors censored before 60 months; in CLHLS about half of the 2014 blood-wave survivors are
administratively censored before 60 months, which inflates O/E. Here calibration-in-the-large is recomputed with censoring handled:
  O/E_KM      = Kaplan-Meier risk at 60 months / mean predicted 60-month risk (everyone);
  O/E_Poisson = observed deaths / sum of each person's predicted cumulative hazard up to min(t, 60).
Also by blood wave. Same models as the frozen run (ridge Cox on NHANES 1999-2010, CLHLS menu). Output: diag_calibration_km.txt"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final')); sys.path.insert(0, str(HERE))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import clhls_lib as C  # noqa: E402
import json  # noqa: E402
M.L.low_priority()
S = json.load(open(HERE / 'selected_clhls.json'))
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, C.MENU)
x, _ = C.load_clhls(); Z = T.transform_sub(x, xtr, C.MENU); t, e = C.outcome(x)


def km(tt, ee, h):
    o = np.argsort(tt); tt, ee = tt[o], ee[o]; s = 1.0; n = len(tt)
    for u in np.unique(tt[ee & (tt <= h)]):
        at = (tt >= u).sum(); d = (ee & (tt == u)).sum(); s *= 1 - d / at
    return 1 - s


out = []
for pname in ['age+sex only', 'matched', 'routine', 'aware@routine']:
    f = {'age+sex only': [], 'matched': S['matched']['features'], 'routine': C.ROUTINE_CL, 'aware@routine': S['aware_at_routine']['features']}[pname]
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr)
    grid = np.arange(1, 61); H0 = np.array([T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, g) for g in grid])
    lp = m.lp(Z[cols].to_numpy()); r60 = 1 - np.exp(-H0[-1] * np.exp(lp))
    Hi = np.interp(np.minimum(t, 60), np.r_[0, grid], np.r_[0, H0]) * np.exp(lp)
    for grp, msk in [('65-79', (x.age_group == '65-79').to_numpy()), ('80+', (x.age_group == '80+').to_numpy())]:
        kn = e | (t >= 60)
        row = [f'{pname:14s} {grp:6s} n {msk.sum():5d} deaths {int(e[msk].sum()):4d}',
               f'known-status O/E (frozen) {e[msk & kn].sum() / r60[msk & kn].sum():.2f}',
               f'KM risk {km(t[msk], e[msk], 60):.3f} vs mean predicted {r60[msk].mean():.3f} -> O/E_KM {km(t[msk], e[msk], 60) / r60[msk].mean():.2f}',
               f'O/E_Poisson {e[msk].sum() / Hi[msk].sum():.2f}', f'share censored before 60 months alive {(~kn[msk]).mean():.2f}']
        out.append(' | '.join(row))
        if pname == 'matched':
            for wv in [2012, 2014]:
                mw = msk & (x.wave == wv).to_numpy()
                if mw.sum():
                    out.append(f'      wave {wv}: n {mw.sum()} deaths {int(e[mw].sum())} O/E_Poisson {e[mw].sum() / Hi[mw].sum():.2f} median follow-up {np.median(t[mw]):.1f} months')
txt = '\n'.join(out); (HERE / 'diag_calibration_km.txt').write_text(txt, encoding='utf-8'); print(txt)
