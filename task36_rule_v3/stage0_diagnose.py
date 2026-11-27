#!/usr/bin/env python3
"""Task 36 stage 0 (additional, descriptive analyses of Task 34 results; no new runs).
Usage: stage0_diagnose.py  -> stage0_results.json, stage0_print.txt
1. The 74% question: at n/k >= 10, how often and by how much the reusable panel (A) beats the target-aware panel (C), by cell and when averaged
   over splits and repetitions (the aggregation of the Task 16 guide test); tail of losses; by n, k, p, data type, source.
2. Threshold sensitivity of the hybrid rule (C if n/k >= t, else inner CV between A and C), t in {2.5, 5, 10, 20, 40, inf}.
3. Inference: bootstrap stratified by data type; each data type alone; leave-one-data-type-out is the same as each type alone here.
Loss thresholds 0.01 and 0.02 are those used throughout P2 for 'similar' versus 'ahead' (Task 15-17 decisions)."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; T34 = HERE.parent / 'task34_decision'
out, lines = {}, []


def say(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)


def load(folder):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(T34 / folder / '*.csv'))], ignore_index=True)
    d['nk'] = d.n / d.k; d['type'] = np.where(d.source.str.startswith('G_'), 'GTEx', 'RPPA')
    d['best'] = d[['A', 'B', 'C']].max(axis=1); d['CmA'] = d.C - d.A
    return d


def regret(d, t):
    cv = np.where(d.inner_A < d.inner_C, 'A', 'C')
    ch = np.where(d.nk >= t, 'C', cv)
    return d.best - np.where(ch == 'A', d.A, d.C)


def boot(d, f, strat=True, B=2000, seed=20263600):
    rng = np.random.default_rng(seed)
    G = {s: g for s, g in d.groupby('source')}
    types = {t: sorted(d[d.type == t].source.unique()) for t in d.type.unique()}
    v = []
    for _ in range(B):
        if strat:
            pick = [s for t, ss in types.items() for s in rng.choice(ss, len(ss))]
        else:
            ss = sorted(G); pick = list(rng.choice(ss, len(ss)))
        v.append(f(pd.concat([G[s] for s in pick], ignore_index=True)))
    return [float(f(d)), float(np.quantile(v, .025)), float(np.quantile(v, .975))]


for folder, tag in [('dev_results', 'dev'), ('results', 'test')]:
    d = load(folder); R = {}
    hi = d[d.nk >= 10]
    lose = hi[hi.CmA < 0]
    R['cell'] = dict(cells=len(hi), C_ahead=float((hi.CmA > 0).mean()), A_ahead=float((hi.CmA < 0).mean()),
                     A_ahead_by_001=float((hi.CmA < -0.01).mean()), A_ahead_by_002=float((hi.CmA < -0.02).mean()),
                     C_ahead_by_002=float((hi.CmA > 0.02).mean()), mean_gain=float(hi.CmA.mean()),
                     loss_when_A_ahead=dict(mean=float(-lose.CmA.mean()), q50=float(-lose.CmA.quantile(.5)), q90=float(-lose.CmA.quantile(.1)), q95=float(-lose.CmA.quantile(.05))))
    # averaged over outer splits and repetitions within (source, task, p, n, k), and over tasks too (the Task 16 aggregation unit was source x setting)
    a1 = hi.groupby(['source', 'task', 'p', 'n', 'k']).CmA.mean()
    a2 = hi.groupby(['source', 'p', 'n', 'k']).CmA.mean()
    R['averaged'] = dict(task_setting_units=len(a1), task_setting_C_ahead=float((a1 > 0).mean()), task_setting_A_ahead_by_002=float((a1 < -0.02).mean()),
                         source_setting_units=len(a2), source_setting_C_ahead=float((a2 > 0).mean()), source_setting_A_ahead_by_002=float((a2 < -0.02).mean()))
    R['by'] = {c: hi.groupby(c).CmA.agg(lambda s: float((s < 0).mean())).round(3).to_dict() for c in ['n', 'k', 'p', 'type']}
    R['by_type_p'] = {f'{t}_{p}': float((g.CmA < 0).mean()) for (t, p), g in hi.groupby(['type', 'p'])}
    src = hi.groupby('source').CmA.mean()
    R['sources_mean_gain_negative'] = sorted(src[src < 0].round(4).to_dict().items())
    R['source_gain_range'] = [float(src.min()), float(src.max())]
    # where would inner CV have chosen A at n/k >= 10, and was it right?
    cvA = hi.inner_A < hi.inner_C
    R['cv_at_high_nk'] = dict(share_cv_picks_A=float(cvA.mean()), right_when_picks_A=float((hi[cvA].CmA < 0).mean()),
                              regret_C=float((hi.best - hi.C).mean()), regret_cv=float((hi.best - np.where(cvA, hi.A, hi.C)).mean()))
    # threshold sensitivity
    R['threshold'] = {str(t): float(regret(d, t).mean()) for t in [2.5, 5, 10, 20, 40, np.inf]}
    R['threshold_by_type'] = {t: {str(th): float(regret(g, th).mean()) for th in [2.5, 5, 10, 20, np.inf]} for t, g in d.groupby('type')}
    # inference for the Task 34 contrasts, stratified by data type, and per type
    d['rH'] = regret(d, 10); d['rC'] = d.best - d.C
    cvn = np.where(d.nk >= 10, 'C', 'A'); d['rNK'] = d.best - np.where(cvn == 'A', d.A, d.C)
    R['H_minus_C'] = dict(strat=boot(d, lambda x: x.rH.mean() - x.rC.mean()), plain=boot(d, lambda x: x.rH.mean() - x.rC.mean(), strat=False))
    R['H_minus_NK'] = dict(strat=boot(d, lambda x: x.rH.mean() - x.rNK.mean()), plain=boot(d, lambda x: x.rH.mean() - x.rNK.mean(), strat=False))
    R['per_type'] = {t: dict(H_minus_C=boot(g, lambda x: x.rH.mean() - x.rC.mean(), strat=False),
                             H_minus_NK=boot(g, lambda x: x.rH.mean() - x.rNK.mean(), strat=False), sources=int(g.source.nunique()))
                     for t, g in d.groupby('type')}
    out[tag] = R
    say('=====', tag); say(json.dumps(R, indent=1, default=float))
json.dump(out, open(HERE / 'stage0_results.json', 'w'), indent=1, default=float)
open(HERE / 'stage0_print.txt', 'w', encoding='utf-8').write('\n'.join(lines))
