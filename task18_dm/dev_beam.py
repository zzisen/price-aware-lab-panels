"""Development only: beam-search target-aware panels vs standard bundles at the same price, one dev cycle -> the other.
Also the cheapest target-aware panel whose test log-likelihood matches the routine bundle (CMP+CBC+LIPID+A1C)."""
import sys, time
import numpy as np
import pandas as pd
import dmlib as L
L.low_priority()
cause = None if sys.argv[1] == 'all' else sys.argv[1]
x, _ = L.load(['C', 'D']); t, e = L.outcome(x, cause); Z = L.transform(x)
budgets = sorted(set([L.cost(v) for v in L.STANDARD.values()] + [12, 15, 22, 25, 30, 35]))
rows = []
for tr, te in [('C', 'D'), ('D', 'C')]:
    a, b = np.flatnonzero(x.cycle == tr), np.flatnonzero(x.cycle == te)
    Za, Zb = Z.iloc[a], Z.iloc[b]
    folds = np.array_split(np.random.default_rng(1).permutation(len(a)), 5)
    cache = {}
    def ev(items):
        cols = ['age', 'female'] + L.features_of(items)
        m = L.Cox(1.0).fit(Za[cols].to_numpy(), t[a], e[a]); lp = m.lp(Zb[cols].to_numpy())
        return L.cindex(lp, t[b], e[b]), L.partial_loglik(lp, t[b], e[b])
    c0, l0 = ev([]); cf, lf = ev(list(L.ITEMS))
    std = {s: ev(v) for s, v in L.STANDARD.items()}
    for B in budgets:
        t0 = time.time()
        sel, c, _ = L.beam_budget(Za, t[a], e[a], B, 1.0, folds, cache=cache)
        ct, lt = ev(sel)
        same = [s for s, v in L.STANDARD.items() if abs(L.cost(v) - B) < 1e-6]
        r = dict(cause=sys.argv[1], train=tr, budget=B, aware_cost=c, aware_C=ct, aware_ll=(lt - l0) / (lf - l0), aware=' '.join(sel), base_C=c0, full_C=cf, sec=round(time.time() - t0))
        if same:
            cs, ls = std[same[0]]; r.update(standard=same[0], std_C=cs, std_ll=(ls - l0) / (lf - l0), dC=ct - cs)
        rows.append(r); print(r, flush=True)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print(d.round(4).to_string())
d.to_csv(f'dev_beam_{sys.argv[1]}.csv', index=False)
