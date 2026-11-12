#!/usr/bin/env python3
"""Synthetic unit test of the CLHLS follow-up construction (no real data). Output: test_clhls_lib.txt"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import clhls_lib as C  # noqa: E402
waves = C.WAVES[2008][2]
L = pd.DataFrame({
    'yearin': [2008, 2008, 2008, 2008, 2008, 2009],
    'monthin': [6, 6, 6, 6, 99, 1], 'dayin': [15, 15, 15, 15, 99, 10],
    'dth08_11': [1, 0, 0, -9, 0, 2], 'd11vyear': [2010, -7, -7, -9, -7, 2012], 'd11vmonth': [6, -7, -7, -9, -7, 3], 'd11vday': [15, -7, -7, -9, -7, 99],
    'yearin_11': [-7, 2011, 2011, -9, 2011, 2011], 'monthin_11': [-7, 6, 6, -9, 7, 8], 'dayin_11': [-7, 15, 15, -9, 1, 1],
    'dth11_14': [-8, 1, 0, -8, -9, -8], 'd14vyear': [-8, 2013, -7, -8, -9, -8], 'd14vmonth': [-8, 99, -7, -8, -9, -8], 'd14vday': [-8, 99, -7, -8, -9, -8],
    'yearin_14': [-8, -7, 2014, -8, -9, -8], 'monthin_14': [-8, -7, 6, -8, -9, -8], 'dayin_14': [-8, -7, 15, -8, -9, -8],
    'dth14_18': [-8, -8, 0, -8, -8, -8], 'd18vyear': [-8, -8, -7, -8, -8, -8], 'd18vmonth': [-8, -8, -7, -8, -8, -8], 'd18vday': [-8, -8, -7, -8, -8, -8],
    'yearin_18': [-8, -8, 2018, -8, -8, -8], 'monthin_18': [-8, -8, 6, -8, -8, -8], 'dayin_18': [-8, -8, 15, -8, -8, -8]})
t, e, t0 = C.survival(L, waves)
expect = [(24.0, True), (60.0, False), (60.0, False), (0.0, False), (36.0, False), (38.0, True)]   # case 3: no follow-up -> 0 (removed by the loader, D4)
out, bad = [], 0
for i, (et, ee) in enumerate(expect):
    ok = abs(t[i] - et) < 1.0 and bool(e[i]) == ee; bad += not ok
    out.append(f'case {i}: months {t[i]:.1f} died {bool(e[i])} | expected about {et} {ee} | {"OK" if ok else "FAIL"}')
# D3: valid days 29-31 are kept; an impossible day moves to the month's last day; unknown day -> 15
dd = C.ymd(pd.Series([2012, 2012, 2013, 2012, 2014]), pd.Series([1, 2, 2, 4, 5]), pd.Series([31, 29, 30, 31, 99]))
want = ['2012-01-31', '2012-02-29', '2013-02-28', '2012-04-30', '2014-05-15']
for g, w_ in zip(dd, want):
    ok = str(g.date()) == w_; bad += not ok
    out.append(f'date {g.date()} | expected {w_} | {"OK" if ok else "FAIL"}')
out.append('ALL OK' if bad == 0 else f'{bad} FAILED')
(HERE / 'test_clhls_lib.txt').write_text('\n'.join(out), encoding='utf-8'); print('\n'.join(out))
