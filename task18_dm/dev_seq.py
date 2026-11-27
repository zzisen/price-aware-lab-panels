"""Development only: sequential ordering. Stage 0 age+sex, stage 1 a target-aware panel within b1, stage 2 the stage-1 panel extended within b2.
Flip models from cross-fitted training risks; evaluated on the other dev cycle."""
import sys
import numpy as np
import pandas as pd
import dmlib as L
L.low_priority()
cause = None if sys.argv[1] == 'all' else sys.argv[1]
b1, b2 = float(sys.argv[2]), float(sys.argv[3])
x, _ = L.load(['C', 'D']); t, e = L.outcome(x, cause); Z = L.transform(x)
rows = []
for tr, te in [('C', 'D'), ('D', 'C')]:
    a, b = np.flatnonzero(x.cycle == tr), np.flatnonzero(x.cycle == te)
    Za, Zb, ta, ea, tb, eb = Z.iloc[a], Z.iloc[b], t[a], e[a], t[b], e[b]
    folds = np.array_split(np.random.default_rng(1).permutation(len(a)), 5)
    cache = {}
    s1, c1, _ = L.beam_budget(Za, ta, ea, b1, 1.0, folds, cache=cache)
    s2, c2, _ = L.beam_budget(Za, ta, ea, b2, 1.0, folds, cache=cache, start=tuple(s1))
    stages = [[], L.features_of(s1), L.features_of(s2)]
    costs = [0.0, c1, c2]
    rcv = L.crossfit_risk(Za, ta, ea, stages, folds)
    models = [L.Cox(1.0).fit(Za[['age', 'female'] + f].to_numpy(), ta, ea) for f in stages]
    rt = np.array([m.risk(Zb[['age', 'female'] + f].to_numpy()) for m, f in zip(models, stages)])
    std_items = L.STANDARD['CMP+CBC+LIPID+A1C']; fs = L.features_of(std_items)
    ms = L.Cox(1.0).fit(Za[['age', 'female'] + fs].to_numpy(), ta, ea); rs = ms.risk(Zb[['age', 'female'] + fs].to_numpy())
    known = eb | (tb >= L.HORIZON)              # 9-year status known
    for tau in (0.10, 0.20):
        flips = [L.FlipModel(tau).fit(rcv[s], rcv[-1]) for s in range(2)]
        def summary(name, risk, cost_pp):
            hi = risk >= tau
            return dict(cause=sys.argv[1], train=tr, tau=tau, policy=name, mean_cost=float(np.mean(cost_pp)), any_lab=float(np.mean(cost_pp > 0)),
                        agree_full=float(np.mean(hi == (rt[-1] >= tau))), sens=float(hi[known & eb].mean()), spec=float((~hi[known & ~eb]).mean()),
                        flagged=float(hi.mean()), C=L.cindex(risk, tb, eb),
                        nb=float((hi & eb & known).sum() / known.sum() - (hi & ~eb & known).sum() / known.sum() * tau / (1 - tau)))
        rows.append(summary('standard routine (everyone)', rs, np.full(len(b), L.cost(std_items))))
        rows.append(summary('aware stage 2 (everyone)', rt[-1], np.full(len(b), c2)))
        rows.append(summary('aware stage 1 (everyone)', rt[1], np.full(len(b), c1)))
        rows.append(summary('age+sex only', rt[0], np.zeros(len(b))))
        for eps in (0.01, 0.02, 0.05, 0.1):
            st, risk, cp = L.sequential_policy(rt, flips, eps, costs)
            r = summary(f'sequential eps={eps}', risk, cp); r['stage_share'] = np.bincount(st, minlength=3).round() / len(st)
            rows.append(r)
    print(tr, 'stage1', s1, c1, 'stage2', s2, c2, flush=True)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(d.round(4).to_string())
d.to_csv(f'dev_seq_{sys.argv[1]}_{b1:g}_{b2:g}.csv', index=False)
