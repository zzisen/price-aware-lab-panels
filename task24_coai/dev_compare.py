#!/usr/bin/env python3
"""Task 24 development check (training data only): 5-fold out-of-fold C of CoAI panels and our panels, ridge Cox (lambda 1).
Output: dev_compare.txt"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
M.L.low_priority()
cp = json.load(open(HERE / 'coai_panels.json')); SEL = json.load(open(HERE.parent / 'task19_dm2' / 'selected_all.json'))
SCN = json.load(open(HERE.parent / 'task20_chns' / 'selected_cn.json'))
x, _ = M.load(list('ABCDEF')); Z = M.transform(x); t, e = M.outcome(x); Zc = T.transform_sub(x, x, T.SHARED)
P = {'routine': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'CMP+CBC': M.features_of(['CMP', 'CBC']),
     'ours matched ($32.41)': M.features_of(SEL['US']['matched']['items']), 'ours target-aware ($50.57)': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
for v in ['published', 'bundled']:
    for p in cp['US_' + v]['panels']:
        P[f"CoAI {v} @{p['budget']} (billed {p['billed_cost']})"] = p['features']
folds = np.array_split(np.random.default_rng(1).permutation(len(t)), 5)
def oof(Zm, feats):
    lp = np.zeros(len(t)); cols = M.BASE + list(feats)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(t)), te); lp[te] = M.Cox(1.0).fit(Zm.iloc[tr][cols].to_numpy(), t[tr], e[tr]).lp(Zm.iloc[te][cols].to_numpy())
    return lp
out = []
LP = {k: oof(Z, f) for k, f in P.items()}
for k in P:
    out.append(f"US  {k:48s} billed ${M.cost_us(P[k]):6.2f}  OOF C {M.cindex(LP[k], t, e):.4f}")
rng = np.random.default_rng(5); 
for a, b in [('ours matched ($32.41)', 'CoAI published @32.41 (billed 27.67)'), ('ours target-aware ($50.57)', 'CoAI published @50.77 (billed 27.67)')]:
    d = M.cindex(LP[a], t, e) - M.cindex(LP[b], t, e)
    bs = []
    for _ in range(200):
        i = rng.integers(0, len(t), len(t)); bs.append(M.cindex(LP[a][i], t[i], e[i]) - M.cindex(LP[b][i], t[i], e[i]))
    out.append(f"dC {a} - {b}: {d:+.4f} [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}] (training OOF, 200 bootstraps)")
PC = {'CN routine shared': SCN['routine_shared']['features'], 'CN ours matched': SCN['matched']['features'], 'CN ours aware': SCN['aware_at_routine']['features']}
for p in cp['CN']['panels']:
    PC[f"CN CoAI @{p['budget']}"] = p['features']
for k, f in PC.items():
    out.append(f"{k:40s} yuan {T.cost_cn(f):6.1f}  OOF C {M.cindex(oof(Zc, f), t, e):.4f}  same as: {[j for j, g in PC.items() if j != k and set(g) == set(f)]}")
txt = '\n'.join(out); (HERE / 'dev_compare.txt').write_text(txt, encoding='utf-8'); print(txt)
