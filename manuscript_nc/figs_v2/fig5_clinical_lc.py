#!/usr/bin/env python3
"""Figure 5 of NC v0.3: when to redesign laboratory panels — clinical learning curves (Task 37 stage 2, development data: NHANES,
CLHLS) and the clinical rule chosen by the prespecified procedure. Output: figs_v2/out/fig5.*
v0.8: a, x positions of the seven series offset by -5% to +5% so error bars separate; b, c, mean number of analytes in the
redesigned panel (d_C, stage2_summary.json) printed at three event counts; e, 95% hierarchical-bootstrap intervals for each
strategy (task43_clinical_extras/a_rule_ci.json, regret.*.ci_hier) drawn on the pooled ("Both") estimate."""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
OUT = HERE / 'out'; T = W / 'task37_clinical_rule'
d = pd.concat([pd.read_csv(f) for f in glob.glob(str(T / 'results' / '*.csv'))], ignore_index=True)
d['key'] = d.cohort + '_' + d.outcome
S = json.load(open(T / 'stage2_summary.json'))
NAME = {'NH_all': 'NHANES, all-cause death', 'NH_cvd': 'NHANES, CVD death', 'NH_cancer': 'NHANES, cancer death', 'NH_other': 'NHANES, other death',
        'CL_T1': 'CLHLS, death', 'CL_T2': 'CLHLS, ADL disability', 'CL_T3': 'CLHLS, cognitive impairment'}
COL = {'NH_all': INK, 'NH_cvd': REDESIGN, 'NH_cancer': PURPLE, 'NH_other': ORANGE, 'CL_T1': INK, 'CL_T2': SKY, 'CL_T3': RULE}
fig = plt.figure(figsize=(W2, 112 * MM), layout='constrained')
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

ax = fig.add_subplot(gs[0, 0])
ax.axhline(0, color=BASE, lw=0.5)
OFF = dict(zip(NAME, np.linspace(-0.05, 0.05, len(NAME))))   # +-5% horizontal offset per series (log axis)
for k in NAME:
    lc = S['learning_curves'][k]; E = sorted(int(e) for e in lc)
    m = np.array([lc[str(e)]['gain'] for e in E]); se = np.array([lc[str(e)]['gain_se'] or 0 for e in E])
    ls = '-' if k.startswith('NH') else '--'
    ax.errorbar(np.array(E) * (1 + OFF[k]), m, yerr=1.96 * se, color=COL[k], ls=ls, marker='o' if k.startswith('NH') else 's', ms=2.3, lw=0.8, elinewidth=0.35, alpha=0.9, label=NAME[k])
ax.set_xscale('log'); ax.set_xticks([25, 50, 100, 200, 400, 776]); ax.set_xticklabels(['25', '50', '100', '200', '400', '776'])
ax.minorticks_off(); ax.set_xlabel('Training events'); ax.set_ylabel('Redesigned − better of reusable\nand routine panels (C or AUC)')
ax.set_title('Gain of redesign by outcome'); label(ax, 'a', dx=-0.27)


def curves(ax, k, title, lab):
    lc = S['learning_curves'][k]; E = sorted(int(e) for e in lc)
    for p, nm, c, mk in [('base', 'Age and sex only', BASE, None), ('D', 'Routine bundle', INK, 'D'), ('A', 'Reusable (FSCA)', REUSE, 'o'),
                         ('B', 'Historically trained', HIST, '^'), ('C', 'Redesigned', REDESIGN, 'o')]:
        ax.plot(E, [lc[str(e)][p] for e in E], color=c, marker=mk, ms=2.3, lw=0.9 if p == 'C' else 0.7, ls=DASH if p == 'base' else '-', label=nm)
    ax.set_xscale('log'); ax.set_xticks(E); ax.set_xticklabels([str(e) for e in E]); ax.minorticks_off()
    ax.set_xlabel('Training events'); ax.set_title(title); label(ax, lab, dx=-0.22)
    show_e = [E[0], E[len(E) // 2], E[-1]]
    for e in show_e:
        first = e == show_e[0]
        ax.annotate(f"d = {lc[str(e)]['d_C']:.1f}", (e, lc[str(e)]['C']), xytext=(-4, 0) if first else (0, 4), textcoords='offset points',
                    ha='right' if first else 'center', va='center' if first else 'bottom',
                    fontsize=5, color=REDESIGN, fontweight='bold')
    ax.text(0.03, 0.97, 'd = mean analytes in\nredesigned panel', transform=ax.transAxes, ha='left', va='top', fontsize=5, color=REDESIGN)
    lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.12 * (hi_ - lo_)); ax.set_xlim(E[0] * 0.5,E[-1] * 1.22)


ax = fig.add_subplot(gs[0, 1]); curves(ax, 'NH_all', 'NHANES, all-cause death', 'b'); ax.set_ylabel("Harrell's C"); ax.legend(loc='lower right')
ax = fig.add_subplot(gs[0, 2]); curves(ax, 'CL_T1', 'CLHLS, death', 'c'); ax.set_ylabel('AUC')

ax = fig.add_subplot(gs[1, 0])
for k in NAME:
    lc = S['learning_curves'][k]; E = sorted(int(e) for e in lc)
    ax.plot(E, [lc[str(e)]['jaccard_C_full'] for e in E], color=COL[k], ls='-' if k.startswith('NH') else '--', marker='o' if k.startswith('NH') else 's', ms=2.3)
ax.set_xscale('log'); ax.set_xticks([25, 50, 100, 200, 400, 776]); ax.set_xticklabels(['25', '50', '100', '200', '400', '776']); ax.minorticks_off()
ax.set_ylim(0, 1.34); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.spines['left'].set_bounds(0, 1); ax.set_xlabel('Training events'); ax.set_ylabel('Overlap with the panel redesigned\non all training data (Jaccard)')
ax.set_title('Small samples redesign unstably'); label(ax, 'd', dx=-0.27)
ax.legend(handles=[Line2D([], [], color=COL[k], ls='-' if k.startswith('NH') else '--', marker='o' if k.startswith('NH') else 's', ms=2.3, label=NAME[k])
                   for k in NAME], fontsize=5, loc='upper left', handlelength=1.8, labelspacing=0.25, borderaxespad=0.1)

ax = fig.add_subplot(gs[1, 1:])
R = S['regret_by_cohort']
show = [('always_D', 'Always routine bundle'), ('always_A', 'Always reuse (FSCA)'), ('always_base', 'Always age and sex only'),
        ('always_B', 'Always historically trained'), ('nested_ACD', 'Nested CV (A, C, routine)'), ('E/d_C>=10:C|nested', 'Omics rule, events: E/d ≥ 10'),
        ('always_C', 'Always redesign'), ('nested_ACDbase', 'Nested CV incl. no tests'), ('E/d_C>=5:C|nested3', 'Clinical rule')]
y = np.arange(len(show))[::-1]
ax.scatter([R[k]['NH'] for k, _ in show], y + 0.12, marker='o', s=12, color=REUSE, label='NHANES (4 outcomes)', zorder=3, edgecolor='white', lw=0.3)
ax.scatter([R[k]['CL'] for k, _ in show], y - 0.12, marker='s', s=12, color=ORANGE, label='CLHLS (3 outcomes)', zorder=3, edgecolor='white', lw=0.3)
CI = json.load(open(W / 'task43_clinical_extras' / 'a_rule_ci.json'))['regret']
CIK = {'always_D': 'always routine bundle (D)', 'always_A': 'always reuse (A)', 'always_base': 'always age and sex only',
       'always_B': 'always history-trained (B)', 'nested_ACD': 'nested CV only (A, C, D)', 'E/d_C>=10:C|nested': 'omics analogue (E/d_C>=10, else nested A, C, D)',
       'always_C': 'always redesign (C)', 'nested_ACDbase': 'nested CV only (A, C, D, no tests)', 'E/d_C>=5:C|nested3': 'rule (E/d_C>=5, else nested CV incl. no tests)'}
for (k, _), yi in zip(show, y):
    assert abs(CI[CIK[k]]['est'] - S['regret'][k]) < 1e-4, k
    lo_, hi_ = CI[CIK[k]]['ci_hier']; ax.plot([lo_, hi_], [yi, yi], color=INK, lw=0.7, zorder=3, solid_capstyle='butt')
ax.scatter([S['regret'][k] for k, _ in show], y, marker='|', s=40, color=INK, label='Both (95% CI)', zorder=4, lw=1.0)
for yi in y:
    ax.axhline(yi, color=LIGHT, lw=0.4, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in show]); ax.get_yticklabels()[-1].set_color(RULE); ax.get_yticklabels()[-1].set_fontweight('bold')
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlabel('Mean regret (C or AUC lost against the best of five panels; 28 outcome × event cells)'); ax.legend(loc='lower right', title='Clinical rule: redesign if E/d ≥ 5,\notherwise nested CV including no tests', title_fontsize=5)
ax.set_title('Clinical rule chosen by the prespecified procedure (development)'); label(ax, 'e', dx=-0.27)
save(fig, OUT / 'fig5')
print('ok')
