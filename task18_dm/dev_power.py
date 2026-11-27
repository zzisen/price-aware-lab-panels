"""Development only: how wide are bootstrap intervals when models trained on one dev cycle are tested on the other (about 2,400 people)?
Used to choose margins for the frozen plan. Panels are the ones the C-trained beam chose in dev_beam / dev_seq."""
import numpy as np
import dmlib as L
L.low_priority()
x, _ = L.load(['C', 'D']); t, e = L.outcome(x); Z = L.transform(x)
a, b = np.flatnonzero(x.cycle == 'C'), np.flatnonzero(x.cycle == 'D')
fit = lambda items: L.Cox(1.0).fit(Z.iloc[a][['age', 'female'] + L.features_of(items)].to_numpy(), t[a], e[a])  # noqa: E731
pred = lambda m, items: m.risk(Z.iloc[b][['age', 'female'] + L.features_of(items)].to_numpy())  # noqa: E731
P = {'routine': L.STANDARD['CMP+CBC+LIPID+A1C'], 'cmpcbc': ['CMP', 'CBC'], 'aware': ['CBC', 'CMP', 'acr'], 's1': ['CMP'], 'none': []}
R = {k: pred(fit(v), v) for k, v in P.items()}
folds = np.array_split(np.random.default_rng(1).permutation(len(a)), 5)
stages = [[], L.features_of(P['s1']), L.features_of(P['aware'])]
rcv = L.crossfit_risk(Z.iloc[a], t[a], e[a], stages, folds)
flips = [L.FlipModel(0.1).fit(rcv[s], rcv[-1]) for s in range(2)]
st, rseq, cp = L.sequential_policy(np.array([R['none'], R['s1'], R['aware']]), flips, 0.02, [0, L.cost(P['s1']), L.cost(P['aware'])])
tb, eb = t[b], e[b]
rng = np.random.default_rng(3); D = []
for _ in range(300):
    i = rng.integers(0, len(b), len(b))
    C = {k: L.cindex(np.log(r[i]), tb[i], eb[i]) for k, r in R.items()}
    nb = lambda r: L.net_benefit(r[i], tb[i], eb[i], 0.1)  # noqa: E731
    D.append([C['cmpcbc'] - C['routine'], C['aware'] - C['routine'], nb(rseq) - nb(R['routine']), nb(R['aware']) - nb(R['routine']), cp[i].mean() / L.cost(P['routine'])])
D = np.array(D)
for k, col in zip(['dC cmpcbc-routine', 'dC aware-routine', 'dNB10 seq-routine', 'dNB10 aware-routine', 'seq cost share'], D.T):
    print(f'{k:22s} mean {col.mean():+.4f} sd {col.std():.4f} 2.5% {np.quantile(col, .025):+.4f} 97.5% {np.quantile(col, .975):+.4f}')
