#!/usr/bin/env python3
"""Task 37 stage 2 summary (STAGE2_DEV_PLAN.md). Descriptive development analysis on used data only.
Per row (cohort x outcome x E x rep): test C/AUC of A (target-agnostic), B (history), C (redesigned on the subsample), D (routine),
base (age, sex), ALL (signal reference). Training-only quantities: cv_* (5-fold CV log-likelihood; C apparent), nest_* (nested CV).
Regret = best test value among {A, B, C, D, base} minus the chosen one. Cells (outcome x E) are weighted equally (reps averaged first).
Output stage2_summary.json, stage2_print.txt."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parent
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE / 'results' / '*.csv'))], ignore_index=True)
d['key'] = d.cohort + '_' + d.outcome
P = ['A', 'B', 'C', 'D', 'base']
d['best'] = d[[f'test_{p}' for p in P]].max(axis=1)
d['gain'] = d.test_C - d[['test_A', 'test_D']].max(axis=1)
d['gain_vs_D'] = d.test_C - d.test_D
d['signal'] = (d.cv_ALL - d.cv_base) / d.E                                     # CV log-likelihood gain of all analytes over age+sex, per event
d['signal_D'] = (d.cv_D - d.cv_base) / d.E
fixed = json.load(open(HERE / 'panels_fixed.json'))


def pick_nested(r, opts):
    return max(opts, key=lambda p: r[f'nest_{p}'])


rules = {}
for p in P:
    rules[f'always_{p}'] = lambda r, p=p: p
rules['nested_ACD'] = lambda r: pick_nested(r, ['A', 'C', 'D'])
rules['nested_ACDbase'] = lambda r: pick_nested(r, ['A', 'C', 'D', 'base'])
for t in [1, 2, 5, 10]:
    for dn in ['d_C', 'd_D']:
        rules[f'E/{dn}>={t}:C|D'] = lambda r, t=t, dn=dn: 'C' if (r.E / r[dn] if r[dn] > 0 else np.inf) >= t else 'D'
        rules[f'E/{dn}>={t}:C|nested'] = lambda r, t=t, dn=dn: 'C' if (r.E / r[dn] if r[dn] > 0 else np.inf) >= t else pick_nested(r, ['A', 'C', 'D'])
        rules[f'E/{dn}>={t}:C|nested3'] = lambda r, t=t, dn=dn: 'C' if (r.E / r[dn] if r[dn] > 0 else np.inf) >= t else pick_nested(r, ['A', 'C', 'D', 'base'])
for k, f in rules.items():
    ch = d.apply(f, axis=1)
    d['r_' + k] = d.best - np.array([d.at[i, f'test_{c}'] for i, c in ch.items()])
    d['pick_' + k] = ch
cell = d.groupby(['cohort', 'key', 'E'])
cm = cell[[c for c in d.columns if c.startswith('r_')] + ['gain', 'gain_vs_D', 'signal', 'signal_D', 'test_A', 'test_B', 'test_C', 'test_D',
                                                           'test_base', 'test_ALL', 'd_C', 'k_items_C']].mean().reset_index()
reg = {k: float(cm['r_' + k].mean()) for k in rules}
reg_by = {k: {c: float(g['r_' + k].mean()) for c, g in cm.groupby('cohort')} for k in rules}
ok = [k for k in rules if all(reg_by[k][c] <= reg_by['always_C'][c] + 1e-12 for c in reg_by[k])]
ok_thr = [k for k in ok if k.startswith('E/')]
choice = 'E/d_C>=10:C|nested' if 'E/d_C>=10:C|nested' in ok else (min(ok_thr, key=reg.get) if ok_thr else 'always_C')
best_overall = min(ok, key=reg.get) if ok else 'always_C'
# learning curves; jac = overlap (Jaccard) of the subsample-redesigned items with the full-training redesigned panel
d['jac'] = [len(set(c.split('+')) & set(fixed[k]['C_full'])) / len(set(c.split('+')) | set(fixed[k]['C_full'])) if isinstance(c, str) else 0.0
            for c, k in zip(d.C_items, d.key)]
lc = {}
for key, g in d.groupby('key'):
    lc[key] = {int(E): dict(reps=int(len(h)), gain=float(h.gain.mean()), gain_se=float(h.gain.std(ddof=1) / np.sqrt(len(h))) if len(h) > 1 else None,
                            gain_vs_D=float(h.gain_vs_D.mean()), C=float(h.test_C.mean()), A=float(h.test_A.mean()), B=float(h.test_B.mean()),
                            D=float(h.test_D.mean()), base=float(h.test_base.mean()), ALL=float(h.test_ALL.mean()),
                            C_wins=float((h.gain > 0).mean()), d_C=float(h.d_C.mean()), signal=float(h.signal.mean()),
                            jaccard_C_full=float(h.jac.mean()))
                 for E, h in g.groupby('E')}
sig = spearmanr(cm.signal, cm.gain).statistic
epd = spearmanr(cm.E / cm.d_C.clip(lower=1), cm.gain).statistic
out = dict(rows=len(d), cells=len(cm), regret=dict(sorted(reg.items(), key=lambda kv: kv[1])), regret_by_cohort=reg_by,
           not_worse_than_always_C_in_both=ok, chosen_rule=choice, lowest_regret_qualifying=best_overall,
           spearman_cell_gain_vs_signal=float(sig), spearman_cell_gain_vs_E_over_dC=float(epd), learning_curves=lc,
           pick_share={k: d['pick_' + k].value_counts(normalize=True).round(3).to_dict() for k in ['nested_ACD', 'nested_ACDbase', choice]},
           fixed_panel_keys=list(fixed))
json.dump(out, open(HERE / 'stage2_summary.json', 'w'), indent=1, default=float)
print('rows', len(d), 'cells', len(cm))
print('chosen rule:', choice, '| lowest-regret qualifying:', best_overall)
print('Spearman (cell) gain~signal %.3f, gain~E/d_C %.3f' % (sig, epd))
print('\nregret (cells equal weight), by cohort:')
for k in sorted(reg, key=reg.get):
    print(f'{k:28s} {reg[k]:.4f}  NH {reg_by[k].get("NH", np.nan):.4f}  CL {reg_by[k].get("CL", np.nan):.4f}' + ('  *' if k in ok else ''))
print('\nlearning curves: gain = C - max(A, D); test C/AUC')
for key in lc:
    for E, v in lc[key].items():
        print(f"{key:10s} E={E:4d} reps={v['reps']:2d} gain={v['gain']:+.4f} (se {v['gain_se'] if v['gain_se'] is None else round(v['gain_se'], 4)}) "
              f"C={v['C']:.4f} A={v['A']:.4f} B={v['B']:.4f} D={v['D']:.4f} base={v['base']:.4f} ALL={v['ALL']:.4f} Cwins={v['C_wins']:.2f} dC={v['d_C']:.1f} sig={v['signal']:+.3f} jac={v['jaccard_C_full']:.2f}")
print('\npick shares:', json.dumps(out['pick_share']))
