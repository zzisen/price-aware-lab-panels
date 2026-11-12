#!/usr/bin/env python3
"""Independent check of Task 23 summaries (imports no project library). Reads t23_<c>_reps.csv, recomputes every summary cell and the
prespecified reading, and compares with t23_<c>_summary.json. Usage: verify_t23.py <cohort> [--dry]"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
c = sys.argv[1]; pre = ('dry_' if '--dry' in sys.argv else '') + f't23_{c}_'
rows = list(csv.DictReader(open(HERE / f'{pre}reps.csv', encoding='utf-8')))
S = json.load(open(HERE / f'{pre}summary.json'))
cell = defaultdict(lambda: defaultdict(dict)); skipped = defaultdict(int)
for r in rows:
    k = (r['panel'], int(r['n']))
    if r['method'] == 'skipped':
        skipped[k] += 1; continue
    oe = float(r['OE']); assert abs(abs(math.log(oe)) - float(r['abs_log_OE'])) < 1e-9
    cell[k][int(r['rep'])][r['method']] = (abs(math.log(oe)), float(r['NB']), r['recal'])
bad = 0
for s in S['cells']:
    k = (s['panel'], s['n']); reps = cell[k]; n = len(reps)
    a = [v['A import'][0] for v in reps.values()]
    chk = dict(reps_used=n, skipped=skipped[k], abs_log_OE_import=sum(a) / n,
               recal_share_T=sum(float(v['T test then recalibrate'][2]) for v in reps.values()) / n)
    for m in ['B recalibrate', 'T test then recalibrate']:
        d_oe = [v[m][0] - v['A import'][0] for v in reps.values()]; d_nb = [v[m][1] - v['A import'][1] for v in reps.values()]
        chk[m + '|abs_log_OE'] = sum(v[m][0] for v in reps.values()) / n
        chk[m + '|better_calibrated_share'] = sum(x < 0 for x in d_oe) / n
        chk[m + '|dNB_mean'] = sum(d_nb) / n
    b = (chk['B recalibrate|better_calibrated_share'], chk['B recalibrate|dNB_mean'])
    chk['reading_B'] = 'helps' if b[0] >= 0.70 and b[1] >= -0.001 else 'not needed' if b[0] < 0.50 else 'uncertain'
    for key, v in chk.items():
        got = s[key.split('|')[0]][key.split('|')[1]] if '|' in key else s[key]
        ok = got == v if isinstance(v, str) else abs(got - v) < 1e-9
        if not ok:
            bad += 1; print('MISMATCH', k, key, got, v)
    print(f"{k[0]:14s} n={k[1]:5d} reps={n:3d} skipped={skipped[k]:3d} |logOE| A {chk['abs_log_OE_import']:.3f} B {chk['B recalibrate|abs_log_OE']:.3f} "
          f"B better {chk['B recalibrate|better_calibrated_share']:.2f} dNB {chk['B recalibrate|dNB_mean']:+.4f} -> {chk['reading_B']}")
print('ALL MATCH' if bad == 0 else f'{bad} MISMATCHES')
