"""Development only: load C and D, count people and deaths, check Cox and C-index on simple models."""
import time
import numpy as np
import dmlib as L
x, counts = L.load(['C', 'D'])
for c in counts: print(c)
t, e = L.outcome(x)
print('people', len(x), 'deaths by 9 y', int(e.sum()), 'by cycle', x.assign(e=e).groupby('cycle').e.agg(['size', 'sum']).to_dict())
Z = L.transform(x)
for name, items in [('base', []), ('CMP', ['CMP']), ('CBC', ['CBC']), ('all', list(L.ITEMS))]:
    f = L.features_of(items); cols = ['age', 'female'] + f
    for tr, te in [('C', 'D'), ('D', 'C')]:
        a, b = (x.cycle == tr).to_numpy(), (x.cycle == te).to_numpy()
        t0 = time.time(); m = L.Cox(1.0).fit(Z[cols].to_numpy()[a], t[a], e[a])
        print(name, tr, '->', te, 'cost', L.cost(items) if items else 0, 'C', round(L.cindex(m.lp(Z[cols].to_numpy()[b]), t[b], e[b]), 4), 'pred deaths', round(m.risk(Z[cols].to_numpy()[b]).sum()), 'obs', int(e[b].sum()), f'{time.time()-t0:.1f}s')
for s, items in L.STANDARD.items(): print(s, L.cost(items))
