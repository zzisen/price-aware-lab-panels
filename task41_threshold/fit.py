#!/usr/bin/env python3
"""Task 41: fit log(n/k*) = a + b log(ln p) + c log(signal) to the simulation crossovers and to the empirical ones; crossover by k in the simulation."""
import json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent
sim = json.load(open(H / 'sim_summary.json'))
rows = [(int(k.split('_')[0][1:]), float(k.split('_s')[1]), v) for k, v in sim.items() if v]
X = np.array([[1, np.log(np.log(p)), np.log(s)] for p, s, _ in rows]); y = np.log([v for *_, v in rows])
b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
out = {'sim_fit': {'a': b[0], 'b_lnlnp': b[1], 'c_lnsignal': b[2], 'R2': r2}}
d = pd.read_csv(H / 'sim_results.csv'); d['gap'] = d.C - d.A; d['nk'] = d.n / d.k
def cross(h):
    h = h.groupby('nk').gap.mean().reset_index().sort_values('nk'); v, x = h.gap.values, np.log(h.nk.values)
    for i in range(1, len(v)):
        if v[i - 1] < 0 <= v[i]: return float(np.exp(np.interp(0, [v[i - 1], v[i]], [x[i - 1], x[i]])))
    return None
out['sim_cross_by_k'] = {f'p{p}_s{s}_k{k}': cross(h) for (p, s, k), h in d.groupby(['p', 'signal', 'k'])}
# where would the crossover equal 10?  signal* = exp((a + b log ln p - log 10) / -c)
out['signal_at_nk10'] = {p: float(np.exp((np.log(10) - b[0] - b[1] * np.log(np.log(p))) / b[2])) for p in (71, 200, 1000)}
e = json.load(open(H / 'empirical.json'))
er = []
for k, v in e['by_signal_p'].items():
    sb, p = k.split('|'); p = int(p); p = 200 if 200 < p < 300 else p
    lo, hi = [float(x) for x in sb.strip('(]').split(',')]
    if v and v > 0 and lo >= 0.05: er.append((p, (lo + min(hi, 0.8)) / 2, v))
Xe = np.array([[1, np.log(np.log(p)), np.log(s)] for p, s, _ in er]); ye = np.log([v for *_, v in er])
be = np.linalg.lstsq(Xe, ye, rcond=None)[0]; r2e = 1 - ((ye - Xe @ be) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
out['emp_fit'] = {'a': be[0], 'b_lnlnp': be[1], 'c_lnsignal': be[2], 'R2': r2e, 'cells': len(er)}
json.dump(out, open(H / 'fit.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
