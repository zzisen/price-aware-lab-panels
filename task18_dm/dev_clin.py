"""Development only: does the lab advantage survive a clinical base (age, sex, BMI, SBP, smoking, diabetes)? Selection stability over bootstrap samples."""
import numpy as np
import pandas as pd
import dmlib as L
L.low_priority()
x, _ = L.load(['C', 'D']); t, e = L.outcome(x); Z = L.transform(x)
cl = L.clinical(x); print('clinical missing share', cl.isna().mean().round(3).to_dict())
ok = cl.notna().all(1).to_numpy(); print('complete clinical', ok.sum(), 'of', len(x))
Z = pd.concat([Z, cl], axis=1)
for tr, te in [('C', 'D'), ('D', 'C')]:
    a = np.flatnonzero((x.cycle == tr).to_numpy() & ok); b = np.flatnonzero((x.cycle == te).to_numpy() & ok)
    for base in (['age', 'female'], ['age', 'female'] + L.CLIN):
        for name, items in [('none', []), ('CMP+CBC', ['CMP', 'CBC']), ('aware CMP+CBC+acr', ['CMP', 'CBC', 'acr']), ('routine', L.STANDARD['CMP+CBC+LIPID+A1C']), ('all', list(L.ITEMS))]:
            cols = base + L.features_of(items)
            m = L.Cox(1.0).fit(Z.iloc[a][cols].to_numpy(), t[a], e[a]); r = m.risk(Z.iloc[b][cols].to_numpy())
            print(tr, '->', te, 'base', len(base), f'{name:20s}', 'C', round(L.cindex(np.log(r), t[b], e[b]), 4), 'NB10', round(L.net_benefit(r, t[b], e[b], .1), 4), 'O/E, slope', np.round(L.calibration(r, t[b], e[b]), 3))
# stability: beam at $40 on 10 bootstrap samples of C+D
rng = np.random.default_rng(7); picks = []
for i in range(10):
    s = rng.choice(len(x), len(x), replace=True)
    Zs, ts, es = Z.iloc[s].reset_index(drop=True), t[s], e[s]
    folds = np.array_split(rng.permutation(len(s)), 5)
    sel, c, _ = L.beam_budget(Zs, ts, es, 40.0, 1.0, folds)
    picks.append(' '.join(sorted(sel))); print('bootstrap', i, sorted(sel), c, flush=True)
print(pd.Series(picks).value_counts().to_string())
