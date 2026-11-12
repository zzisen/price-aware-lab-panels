#!/usr/bin/env python3
"""Synthetic tests of the UKHLS outcome and analyte rules (no real data). Output: test_ukhls_lib.txt"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import ukhls_lib as K  # noqa: E402
lines, bad = [], 0


def check(name, got, want):
    global bad
    ok = np.array_equal(np.asarray(got), np.asarray(want)) if not isinstance(want, float) else abs(got - want) < 1e-9
    bad += not ok; lines.append(f'{"OK " if ok else "BAD"} {name}: got {got} want {want}')


# outcome: (nurse wave, dcsedfl, dcsedw, lwenum) -> (died, known)
cases = [
    ('wave 2, death reported wave 6', 2, 1, 6, 6, True, True),
    ('wave 2, death reported wave 11 (edge)', 2, 1, 11, 11, True, True),
    ('wave 2, death reported wave 12 (after horizon), enumerated to 11', 2, 1, 12, 11, False, True),
    ('wave 2, death reported wave 12, last enumerated 9 (unknown)', 2, 1, 12, 9, False, False),
    ('wave 3, alive, enumerated to 15', 3, 2, -8, 15, False, True),
    ('wave 3, alive, last enumerated 11 (unknown)', 3, 2, -8, 11, False, False),
    ('wave 3, alive, last enumerated 12 (edge, known)', 3, 2, -8, 12, False, True),
    ('wave 2, missing deceased flag, enumerated 15', 2, -9, -9, 15, False, True),
    ('wave 2, death reported in BHPS code (not 1)', 2, -8, -8, 5, False, False),
]
for name, w, f, dw, lw, d_, k_ in cases:
    d, k = K.outcome_from([w], [f], [dw], [lw]); check(name + ' died', bool(d[0]), d_); check(name + ' known', bool(k[0]), k_)
# analyte codes
v = K.clean_analyte([5.0, -31, 10.0, -32, -9, -22, 0.0])
check('code -31 -> min valid', float(v[1]), 0.0); check('code -32 -> max valid', float(v[3]), 10.0)
check('codes -9, -22 -> missing', [bool(np.isnan(v[4])), bool(np.isnan(v[5]))], [True, True])
# units and CRP floor
raw = pd.DataFrame({'alb': [40.0], 'alkp': [70.0], 'alt': [20.0], 'ast': [25.0], 'ure': [5.0], 'ecre': [88.4], 'ggt': [30.0], 'chol': [5.0],
                    'hdl': [1.5], 'trig': [1.2], 'hscrp': [-31], 'hgb': [140.0], 'hba1c': [48.0]})
raw2 = raw.copy(); raw2['hscrp'] = [12.0]
h = K.harmonise(pd.concat([raw, raw2], ignore_index=True))
check('albumin g/L -> g/dL', float(h.albumin[0]), 4.0); check('urea -> BUN', float(h.bun[0]), 14.0); check('creatinine -> mg/dL', float(h.creatinine[0]), 1.0)
check('cholesterol -> mg/dL', float(h.tc[0]), 5.0 * 38.67); check('HbA1c 48 -> NGSP', float(h.hba1c[0]), 0.09148 * 48 + 2.152)
check('hs-CRP below detection -> floor', float(h.crp[0]), 0.21); check('hs-CRP 12 mg/L -> 1.2 mg/dL', float(h.crp[1]), 1.2)
check('haemoglobin g/L -> g/dL', float(h.hgb[0]), 14.0)
lines.append('ALL OK' if bad == 0 else f'{bad} FAILED')
(HERE / 'test_ukhls_lib.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print('\n'.join(lines))
