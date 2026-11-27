import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t34lib as T
rp = T.rppa_pools(['KIRC']); timing = []
for c, (xr, famr, extr) in rp.items():
    tr = T.make_tasks('DEV_' + c, xr, famr, extr, 900100, (71, 71 + len(extr)))
    print(c, len(xr), 'tasks', len(tr), flush=True)
    for p in sorted({t['p'] for t in tr}):
        tk = next(t for t in tr if t['p'] == p)
        s = T.run_job(tk, xr, 0, HERE / 'pilot' / f'dev_{c}_p{p}.csv'); timing.append(dict(source='RPPA ' + c, p=p, seconds=round(s, 1)))
        print(timing[-1], flush=True)
json.dump(timing, open(HERE / 'pilot' / 'timing_rppa.json', 'w'), indent=1)
