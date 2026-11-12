#!/usr/bin/env python3
"""Understanding Society: interval from the nurse visit to the end-point wave (appendix table S19), descriptive.
People classed as alive at the end-point wave (wave 11 for nurse wave 2, wave 12 for nurse wave 3) with a valid nurse-visit year and month
(b_/c_nurdayy, b_/c_nurdaym in indsamp) and a valid sample-issue month at the end-point wave (k_month / l_month in xwaveid; month 1 = January
of the wave's first fieldwork year, 2008 + wave number). Output: followup_interval.txt"""
from pathlib import Path
import pyreadstat
import ukhls_lib as K

x, c = K.load_ukhls(); alive = x[x.died == 0]
xw = pyreadstat.read_dta(str(K.MS / 'xwaveid.dta'), usecols=['pidp', 'k_month', 'l_month'])[0].set_index('pidp')
lines = []
for w, l, endw, col in [(2, 'b', 11, 'k_month'), (3, 'c', 12, 'l_month')]:
    s = pyreadstat.read_dta(str(K.NS / f'{l}_indsamp_ns.dta'), usecols=['pidp', f'{l}_nurdayy', f'{l}_nurdaym'])[0].drop_duplicates('pidp').set_index('pidp')
    ids = alive.index[alive.wave == w]
    ny, nm = s[f'{l}_nurdayy'].reindex(ids), s[f'{l}_nurdaym'].reindex(ids); em = xw[col].reindex(ids)
    ok = (ny > 2000) & (nm.between(1, 12)) & (em.between(1, 24))
    months = ((2008 + endw) * 12 + em[ok] - 1) - (ny[ok] * 12 + nm[ok] - 1)
    lines.append(f'nurse wave {w}: alive {len(ids)}, with valid months {int(ok.sum())}, median interval {float(months.median()):.1f} months '
                 f'(IQR {float(months.quantile(0.25)):.1f}-{float(months.quantile(0.75)):.1f})')
(Path(__file__).parent / 'followup_interval.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
