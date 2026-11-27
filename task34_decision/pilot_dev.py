#!/usr/bin/env python3
"""Task 34 pilot on DEVELOPMENT data only (GTEx thyroid and lung, already used in Task 17; RPPA KIRC, already used): checks the code and
times one job per p level. Not part of the locked test; results are not used for any decision."""
import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t34lib as T  # noqa: E402
out = HERE / 'pilot'; out.mkdir(exist_ok=True)
timing = []
x, fam, extra = T.gtex_pool('thyroid', 'gtex_v8_tissues', 1)
tasks = T.make_tasks('DEV_thyroid', x, fam, extra, 900000, (71, 200, 1000))
print('GTEx dev units', len(x), 'tasks', len(tasks), flush=True)
for p in (71, 200, 1000):
    tk = next(t for t in tasks if t['p'] == p)
    s = T.run_job(tk, x, 0, out / f'dev_thyroid_p{p}.csv'); timing.append(dict(source='GTEx thyroid', p=p, seconds=round(s, 1)))
    print(timing[-1], flush=True)
rp = T.rppa_pools(['KIRC'])
for c, (xr, famr, extr) in rp.items():
    tr = T.make_tasks('DEV_' + c, xr, famr, extr, 900100, (71, 71 + len(extr)))
    for p in sorted({t['p'] for t in tr}):
        tk = next(t for t in tr if t['p'] == p)
        s = T.run_job(tk, xr, 0, out / f'dev_{c}_p{p}.csv'); timing.append(dict(source='RPPA ' + c, p=p, seconds=round(s, 1), units=len(xr)))
        print(timing[-1], flush=True)
json.dump(timing, open(out / 'timing.json', 'w'), indent=1)
