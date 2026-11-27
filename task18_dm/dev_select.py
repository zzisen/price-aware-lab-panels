"""Development only: target-aware cost-aware panels vs standard bundles at the same price, trained on one dev cycle and tested on the other."""
import sys, time
import numpy as np
import pandas as pd
import dmlib as L
L.low_priority()
cause = sys.argv[1] if len(sys.argv) > 1 else None
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
x, _ = L.load(['C', 'D']); t, e = L.outcome(x, cause); Z = L.transform(x)
rows = []
for tr, te in [('C', 'D'), ('D', 'C')]:
    a, b = np.flatnonzero(x.cycle == tr), np.flatnonzero(x.cycle == te)
    Za, Zb = Z.iloc[a], Z.iloc[b]
    folds = np.array_split(np.random.default_rng(1).permutation(len(a)), 5)
    def ev(items):
        cols = ['age', 'female'] + L.features_of(items)
        m = L.Cox(lam).fit(Za[cols].to_numpy(), t[a], e[a])
        lp = m.lp(Zb[cols].to_numpy())
        return L.cindex(lp, t[b], e[b]), L.partial_loglik(lp, t[b], e[b])
    c0, l0 = ev([]); cf, lf = ev(list(L.ITEMS))
    for s, items in L.STANDARD.items():
        budget = L.cost(items)
        t0 = time.time()
        sel, c = L.select_budget(Za, t[a], e[a], budget, lam, folds)
        cs, ls = ev(items); ct, lt = ev(sel)
        rows.append(dict(train=tr, standard=s, budget=budget, std_C=cs, aware_C=ct, dC=ct - cs, std_ll=(ls - l0) / (lf - l0), aware_ll=(lt - l0) / (lf - l0), aware_cost=c, aware=' '.join(sel), base_C=c0, full_C=cf, sec=round(time.time() - t0)))
        print(rows[-1], flush=True)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(d.drop(columns=['aware']).round(4).to_string())
print(d.groupby('standard', sort=False)[['dC', 'std_ll', 'aware_ll']].mean().round(4).to_string())
