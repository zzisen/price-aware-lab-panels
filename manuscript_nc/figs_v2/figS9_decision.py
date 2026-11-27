#!/usr/bin/env python3
"""Supplementary Fig. 9 of NC v0.7: decision curves in NHANES III and NHANES 2011-14. Redrawn in the NC style from the same data as the
Lancet-draft Appendix Figure S3: manuscript_ldh/figures_v2/data/S3_decision_curves.csv (written by export_figure_data.py by refitting the
frozen Task 21 models). Ranges and thresholds as in make_figures_v2.py. Output: figs_v2/out/figS9.*"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
d = pd.read_csv(D / 'S3_decision_curves.csv', encoding='utf-8-sig')
print('Files read:\n ', D / 'S3_decision_curves.csv')
SER = {'Routine bundle': (ROUTINE, '-', 0.8), 'Matched panel': (MATCHED, '-', 0.9), 'Target-aware panel': (AWARE, '-', 0.9), 'Treat all': (INK, DASH, 0.6)}
fig, axs = plt.subplots(1, 2, figsize=(W2, 62 * MM), layout='constrained')
for ax, (c, xmax, thr, hz, yl), lab in zip(axs, [('NHANES III', 0.30, 0.10, '9-year risk', (-0.02, 0.155)), ('NHANES 2011–14', 0.15, 0.05, '5-year risk', (-0.01, 0.05))], 'ab'):
    q = d[(d.cohort == c) & (d.threshold <= xmax + 1e-9)]
    for s, (col, ls, lw) in SER.items():
        r = q[q.strategy == s].sort_values('threshold')
        ax.plot(r.threshold, r.net_benefit, color=col, ls=ls, lw=lw, zorder=3 if s != 'Treat all' else 2)
    ax.axhline(0, color=INK, lw=0.5); ax.axvline(thr, color=BASE, lw=0.6, ls=DASH)
    ax.text(thr + 0.006 * xmax / 0.15, 0.97, 'Prespecified\nthreshold', fontsize=5, va='top', ha='left', color=INK, transform=ax.get_xaxis_transform())
    ax.set_xlim(0.01, xmax); ax.set_ylim(*yl)
    tk = [t / 100 for t in range(5, int(round(xmax * 100)) + 1, 5)]
    ax.set_xticks(tk); ax.set_xticklabels([f'{t * 100:.0f}%' for t in tk])
    ax.set_xlabel(f'Threshold probability ({hz})'); ax.set_ylabel('Net benefit')
    ax.set_title(c, fontsize=6); label(ax, lab, dx=-0.13)
fig.legend(handles=[Line2D([], [], color=c_, ls=l_, lw=w_ + 0.1, label='Flag everyone' if s == 'Treat all' else s) for s, (c_, l_, w_) in SER.items()],
           loc='outside lower center', ncol=4, columnspacing=1.6, handlelength=2.0)
save(fig, OUT / 'figS9')
print('ok')
