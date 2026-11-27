#!/usr/bin/env python3
"""Supplementary Fig. 1: target resolution and training size in drug-screening cell lines (GDSC, GDSC2, GDSC2 new lines, PRISM).
Top row: mean held-out squared standardised error by measurement resolution against training cell lines; bottom row: differences
between resolutions with 2.5-97.5% ranges over 20 subsamples. Cell-line (row) splits, inner-CV selector, as in the original panels.
Inputs (read only):
  task3_learning_curve/learning_curve_summary.csv, task3_learning_curve/learning_curve_contrasts.csv   (GDSC)
  task3b_external/gdsc2_summary.csv, task3b_external/gdsc2_contrasts.csv                             (GDSC2)
  task3b_external/gdsc2_new_summary.csv, task3b_external/gdsc2_new_contrasts.csv                     (GDSC2, lines not in GDSC)
  task3b_external/prism_summary.csv, task3b_external/prism_contrasts.csv                             (PRISM)
Output: figs_v2/out/figS1.pdf/.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SRC = [('task3_learning_curve/learning_curve', 'GDSC', 'GDSC (row splits)'),
       ('task3b_external/gdsc2', 'GDSC2', 'GDSC2 (independent release)'),
       ('task3b_external/gdsc2_new', 'GDSC2 new lines', 'GDSC2, lines not in GDSC'),
       ('task3b_external/prism', 'PRISM', 'PRISM')]
POL = {'C0': ('C0 (no measurement)', BASE, 'o', ':'), 'Shared': ('Shared', REUSE, 's', '-'), 'Family': ('Family', REDESIGN, 'D', '-'),
       'Exact≤3': ('Exact≤3', PURPLE, 'v', '--'), 'Exact': ('Exact', ORANGE, '^', '-')}
CON = {'Family − Shared': (REDESIGN, 'D', '-', True), 'Exact − Family': (ORANGE, '^', '-', True), 'Exact≤3 − Family': (PURPLE, 'v', '--', False)}
print('Files read:')
fig, ax = plt.subplots(2, 4, figsize=(W2, 112 * MM))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.9, wspace=0.42, hspace=0.42)
for j, (stem, an, title) in enumerate(SRC):
    fs, fc = W / f'{stem}_summary.csv', W / f'{stem}_contrasts.csv'; print(' ', fs); print(' ', fc)
    s = pd.read_csv(fs, encoding='utf-8'); c = pd.read_csv(fc, encoding='utf-8')
    s = s[(s.analysis == f'{an} repeated_random') & (s.selector == 'inner_cv')].sort_values('n_train_mean')
    c = c[(c.analysis == f'{an} repeated_random') & (c.selector == 'inner_cv')]
    n = s.n_train_mean.to_numpy(); xt = [f'{v:.0f}' for v in n]
    a = ax[0, j]
    for p, (lab, col, mk, ls) in POL.items():
        a.plot(n, s[p], ls=ls, marker=mk, color=col, ms=2.4, lw=0.8, label=lab)
    a.set_yscale('log'); a.set_yticks([0.8, 1, 1.5, 2, 3]); a.set_yticklabels(['0.8', '1.0', '1.5', '2', '3']); a.minorticks_off()
    a.set_ylim(0.78, 4.0)
    a.set_title(title, fontsize=6); label(a, 'abcd'[j], -0.3, 1.04)
    if j == 0:
        a.set_ylabel('Held-out squared\nstandardised error (log scale)')
    b = ax[1, j]
    for k, (col, mk, ls, band) in CON.items():
        h = c[c.contrast == k].sort_values('n_train_mean')
        b.plot(h.n_train_mean, h['mean'], ls=ls, marker=mk, color=col, ms=2.4, lw=0.8)
        if band:
            r = h[h.n_reps > 1]
            b.fill_between(r.n_train_mean, r['rep_2.5'], r['rep_97.5'], color=col, alpha=0.18, lw=0)
    b.axhline(0, color=INK, lw=0.5)
    b.set_yscale('symlog', linthresh=0.1, linscale=1.5)
    b.set_yticks([-1, -0.1, -0.05, 0, 0.05, 0.1, 1]); b.set_yticklabels(['−1', '−0.1', '−0.05', '0', '0.05', '0.1', '1']); b.minorticks_off()
    b.set_ylim(-10, 10)
    if j == 0:
        b.set_ylabel('Difference in error\n(symmetric log scale; <0: first better)')
    for z in (a, b):
        z.set_xscale('log'); z.set_xticks(n); z.set_xticklabels(xt); z.minorticks_off()
        z.set_xlim(n[0] / 1.25, n[-1] * 1.25)
        z.set_xlabel('Training cell lines (log scale)')
h1 = [Line2D([], [], ls=ls, marker=mk, color=col, ms=2.4, lw=0.8) for (lab, col, mk, ls) in POL.values()]
fig.legend(h1, [v[0] for v in POL.values()], loc='upper center', ncol=5, bbox_to_anchor=(0.53, 1.0), fontsize=5.5, columnspacing=1.2)
h2 = [Line2D([], [], ls=ls, marker=mk, color=col, ms=2.4, lw=0.8) for (col, mk, ls, _) in CON.values()]
fig.legend(h2, list(CON), loc='lower center', ncol=3, bbox_to_anchor=(0.53, 0.0), fontsize=5.5, columnspacing=1.2)
save(fig, HERE / 'out' / 'figS1'); print('ok')
