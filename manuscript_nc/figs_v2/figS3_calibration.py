#!/usr/bin/env python3
"""Supplementary Fig. 3 of NC v0.7: calibration in the primary cohorts and the effect of local recalibration. Redrawn in the NC style from the
same data as the Lancet-draft Figure 5: manuscript_ldh/figures_v2/data/F5A_calibration_bins.csv, F5A_OE_labels.csv, F5B_recalibration.csv
(written by export_figure_data.py from the frozen Task 21 and Task 23 outputs). Risk horizons as in make_figures_v2.py. Output: figs_v2/out/figS3.*"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
rd = lambda n: pd.read_csv(D / n, encoding='utf-8-sig')
cal, oe, rec = rd('F5A_calibration_bins.csv'), rd('F5A_OE_labels.csv'), rd('F5B_recalibration.csv')
print('Files read:\n ', '\n  '.join(str(D / n) for n in ['F5A_calibration_bins.csv', 'F5A_OE_labels.csv', 'F5B_recalibration.csv']))
COH = ['NHANES III', 'NHANES 2011–14', 'CHNS', 'Understanding Society', 'CLHLS (65–79 y)']
HORIZON = {'NHANES III': '9-year risk', 'NHANES 2011–14': '5-year risk', 'CHNS': '6-year risk', 'Understanding Society': '9-year risk', 'CLHLS (65–79 y)': '5-year risk'}
SER = {'Routine bundle': (ROUTINE, 'D'), 'Matched panel': (MATCHED, 'o'), 'Target-aware panel': (AWARE, 's')}

fig = plt.figure(figsize=(W2, 104 * MM), layout='constrained')
SF = fig.subfigures(2, 1, height_ratios=[1, 1.05])
axs = SF[0].subplots(1, 5)
for i, (ax, c) in enumerate(zip(axs, COH)):
    q = cal[cal.cohort == c]
    up = float(q[['predicted', 'observed']].to_numpy().max()) * 1.08
    ax.plot([0, up], [0, up], color=BASE, lw=0.5, ls=DASH, zorder=1)
    for nm, (col, mk) in SER.items():
        s = q[q.panel == nm].sort_values('bin')
        ax.plot(s.predicted, s.observed, color=col, marker=mk, ms=2.0, mew=0, lw=0.8 if nm != 'Routine bundle' else 0.7, zorder=3)
    ax.set_xlim(0, up); ax.set_ylim(0, up); ax.set_aspect('equal', adjustable='box')
    tk = [t for t in ([0, 0.1, 0.2, 0.3] if up < 0.45 else [0, 0.2, 0.4, 0.6]) if t <= up]
    ax.set_xticks(tk); ax.set_yticks(tk); ax.set_xticklabels([f'{t:g}' for t in tk]); ax.set_yticklabels([f'{t:g}' for t in tk])
    ax.set_title(c.replace(' (65–79 y)', ' (65–79 years)'), loc='center', fontsize=6)
    ax.text(0.04, 0.97, HORIZON[c], transform=ax.transAxes, fontsize=5, color=INK, va='top')
    r = oe[oe.cohort == c].iloc[0]
    ax.text(0.97, 0.03, f'O/E {r.matched_OE:.2f} vs {r.routine_OE:.2f}', transform=ax.transAxes, fontsize=5, ha='right', va='bottom', color=INK)
    ax.set_xlabel('Predicted risk')
    if i == 0:
        ax.set_ylabel('Observed risk')
SF[0].legend(handles=[Line2D([], [], color=c_, marker=m_, ms=2.5, mew=0, lw=0.8, label=nm) for nm, (c_, m_) in SER.items()] +
             [Line2D([], [], color=BASE, ls=DASH, lw=0.5, label='Perfect calibration')],
             loc='outside lower center', ncol=4, columnspacing=1.4, handlelength=1.8)
SF[0].text(0.0, 1.0, 'a', fontsize=8, fontweight='bold', va='top', ha='left')
SF[0].suptitle('Observed against predicted risk by tenth of predicted risk (fifths in CLHLS); O/E, matched panel vs routine bundle', x=0.03, ha='left', fontsize=6)

# b: local recalibration of the matched panel
gs = SF[1].add_gridspec(1, 2, width_ratios=[1, 0.42])
ax = SF[1].add_subplot(gs[0, 0])
m = rec[rec.panel == 'matched']
order = ['NHANES III', 'NHANES 2015–16', 'CHNS', 'NHANES 2011–14', 'Understanding Society']
xs = [200, 500, 1000]
ends = []
for c in order:
    s = m[m.cohort == c].sort_values('n_local')
    over = s.import_OE.iloc[0] < 0.88                       # imported model overpredicted risk (O/E 0.85-0.87)
    col = INK if over else BASE
    ax.plot(range(3), s.improved_share, color=col, marker='o', ms=2.6, mew=0, lw=0.9 if over else 0.7, zorder=3 if over else 2, clip_on=False)
    ends.append((s.improved_share.iloc[-1], f'{c} (O/E {s.import_OE.iloc[0]:.2f})', col))
for lv, t in [(0.5, '50%: recalibration not needed below'), (0.7, '70%: recalibration helps above*')]:
    ax.axhline(lv, color=INK, lw=0.5, ls=DASH, zorder=1)
    ax.text(1.97, lv + 0.012, t, fontsize=5, va='bottom', ha='right')
ends.sort()
ly = []
for v, _, _ in ends:
    ly.append(max(v, ly[-1] + 0.085) if ly else v)
for (v, t, col), yy in zip(ends, ly):
    ax.text(2.08, yy, t, fontsize=5, va='center', ha='left', color=INK, fontweight='normal' if col == BASE else 'bold')
ax.set_xticks(range(3)); ax.set_xticklabels([str(x) for x in xs]); ax.set_xlim(-0.15, 2.0); ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel('Local participants with known outcomes'); ax.set_ylabel('Repetitions in which recalibration\nimproved calibration')
ax.set_title('Local recalibration of the matched panel', fontsize=6)
ax2 = SF[1].add_subplot(gs[0, 1]); ax2.axis('off')
ax2.text(0.62, 0.02, 'Black, bold: imported model\noverpredicted risk.\n*Prespecified; also required\nno mean net-benefit loss\nabove 0.001.', fontsize=5, va='bottom', ha='left', transform=ax2.transAxes)
SF[1].text(0.0, 1.0, 'b', fontsize=8, fontweight='bold', va='top', ha='left')
save(fig, OUT / 'figS3')
print('ok')
