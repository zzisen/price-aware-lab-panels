#!/usr/bin/env python3
"""Task 24: our panels versus CoAI panels in one validation cohort (plan_t24_FROZEN.md). Usage: run_t24.py <NH3|GH|I|CHNS> [--dry]
All panels were fixed on NHANES 1999-2010 before freezing (coai_panels.json; ../task19_dm2/selected_all.json; ../task20_chns/selected_cn.json).
Primary model for every panel: ridge Cox (lambda 1) fitted on the training data, as in Task 21, so that only the choice of tests differs.
Secondary: CoAI's own refitted base model (logistic regression for 9-year death, hyperparameters from coai_panels.json).
Outputs: t24_<c>_panels.csv, t24_<c>_endpoints.json. --dry: train on cycle C, test on cycle D."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = Path(__file__).resolve().parent; T21 = HERE.parent / 'task21_final'
sys.path.insert(0, str(T21))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
M.L.low_priority()
COH = sys.argv[1]; DRY = '--dry' in sys.argv; PRE = f"{'dry_' if DRY else ''}t24_{COH}_"; NBOOT = 200 if DRY else 1000; SEED = 20261110
CP = json.load(open(HERE / 'coai_panels.json')); BIN = COH == 'CHNS'

if BIN:
    SCN = json.load(open(T21.parent / 'task20_chns' / 'selected_cn.json'))
    if DRY:
        xtr, _ = M.load(['C']); d0, _ = M.load(['D']); td, ed = T.outcome_at(d0, 72); keep = ed | (td >= 72)
        xte = d0[keep].copy(); xte['died'] = ed[keep].astype(int)
    else:
        xtr, _ = M.load(list('ABCDEF')); xte, _ = T.load_chns()
    ttr, etr = M.outcome(xtr); Ztr = T.transform_sub(xtr, xtr, T.SHARED); Z = T.transform_sub(xte, xtr, T.SHARED)
    y = xte.died.to_numpy().astype(bool); n = len(y); score = lambda s, i=slice(None): T.auc(s[i], y[i])  # noqa: E731
    cn = {p['budget']: p['features'] for p in CP['CN']['panels']}
    P = {'routine': SCN['routine_shared']['features'], 'ours matched': SCN['matched']['features'], 'ours target-aware': SCN['aware_at_routine']['features'],
         'CoAI @matched price': cn[43.8], 'CoAI @routine price': cn[74.8]}
    price = T.cost_cn; HYP = CP['CN']['hyper']
    PAIRS = [('ours target-aware', 'CoAI @routine price'), ('ours matched', 'CoAI @matched price')]
else:
    SEL = json.load(open(T21.parent / 'task19_dm2' / 'selected_all.json'))
    H = {'NH3': 108, 'GH': 60, 'I': 36}[COH]
    if DRY:
        xtr, _ = M.load(['C']); xte, _ = M.load(['D'])
    else:
        xtr, _ = M.load(list('ABCDEF'))
        xte = M.load_nh3()[0] if COH == 'NH3' else (pd.concat([T.load_gh(c)[0] for c in 'GH']) if COH == 'GH' else X.load_i()[0])
    ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr); Z = M.transform(xte, ref=xtr)
    t, e = T.outcome_at(xte, H); n = len(t); score = lambda s, i=slice(None): M.cindex(s[i], t[i], e[i])  # noqa: E731
    pub = {p['budget']: p['features'] for p in CP['US_published']['panels']}; bun = {p['budget']: p['features'] for p in CP['US_bundled']['panels']}
    P = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'CMP+CBC': M.features_of(['CMP', 'CBC']),
         'ours matched': M.features_of(SEL['US']['matched']['items']), 'ours target-aware': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items']),
         'CoAI @matched price': pub[32.41], 'CoAI @routine price': pub[50.77],
         'CoAI bundled @matched price': bun[32.41], 'CoAI bundled @routine price': bun[50.77]}
    price = M.cost_us; HYP = CP['US_published']['hyper']
    PAIRS = [('ours target-aware', 'CoAI @routine price'), ('ours matched', 'CoAI @matched price'),
             ('ours target-aware', 'CoAI bundled @routine price'), ('ours matched', 'CoAI bundled @matched price')]

# training label for CoAI's own model: 9-year death among people with known 9-year status
known_tr = etr | (ttr >= M.HORIZON); ytr = (etr & (ttr <= M.HORIZON))[known_tr].astype(int)
S, S_own, rows = {}, {}, []
for k, f in P.items():
    cols = M.BASE + list(f); A = Ztr[cols].to_numpy(float); B = Z[cols].to_numpy(float)
    S[k] = M.Cox(1.0).fit(A, ttr, etr).lp(B)
    mu, sd = A[known_tr].mean(0), A[known_tr].std(0)
    lr = LogisticRegression(penalty=HYP['penalty'], C=HYP['C'], solver='saga', max_iter=20000, tol=1e-6).fit((A[known_tr] - mu) / sd, ytr)
    S_own[k] = lr.decision_function((B - mu) / sd)
    rows.append(dict(panel=k, features=' '.join(f), price=price(f), metric=('AUC' if BIN else 'C'), cox=score(S[k]), coai_own_model=score(S_own[k])))
panels = pd.DataFrame(rows)
identical = {f'{a} | {b}': set(P[a]) == set(P[b]) for a, b in PAIRS}
rng = np.random.default_rng(SEED); keys = [f'd {a} - {b}' for a, b in PAIRS]; Bs = {k: [] for k in keys}; Bo = {k: [] for k in keys}
for _ in range(NBOOT):
    i = rng.integers(0, n, n)
    if BIN and (y[i].all() or (~y[i]).all()):
        continue
    c = {k: score(S[k], i) for k in set(x_ for p in PAIRS for x_ in p)}; o = {k: score(S_own[k], i) for k in set(x_ for p in PAIRS for x_ in p)}
    for (a, b), kk in zip(PAIRS, keys):
        Bs[kk].append(c[a] - c[b]); Bo[kk].append(o[a] - o[b])
ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]  # noqa: E731
pc = panels.set_index('panel')
point = {kk: float(pc.cox[a] - pc.cox[b]) for (a, b), kk in zip(PAIRS, keys)}
point_own = {kk: float(pc.coai_own_model[a] - pc.coai_own_model[b]) for (a, b), kk in zip(PAIRS, keys)}
out = dict(cohort=COH, n=int(n), events=int(y.sum() if BIN else e.sum()), metric=('AUC' if BIN else 'C'), identical_panels=identical,
           point=point, boot={k: ci(v) for k, v in Bs.items()}, point_coai_own_model=point_own, boot_coai_own_model={k: ci(v) for k, v in Bo.items()},
           prices={k: price(f) for k, f in P.items()})
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(panels.round(4).to_string()); print(json.dumps(out, indent=1))
panels.to_csv(HERE / f'{PRE}panels.csv', index=False); json.dump(out, open(HERE / f'{PRE}endpoints.json', 'w'), indent=1)
person = pd.DataFrame({('y' if BIN else 't'): (y.astype(int) if BIN else t), **({} if BIN else {'e': e.astype(int)}), **{'cox|' + k: v for k, v in S.items()}})
person.to_csv(HERE / f'{PRE}person.csv', index=False)
