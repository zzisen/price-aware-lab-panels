#!/usr/bin/env python3
"""Supplementary Fig. 2 of NC v0.7: participant flow. Redrawn in the NC style from the same data as the Lancet-draft Appendix Figure S2:
manuscript_ldh/figures_v2/data/S2_flow.csv (written by export_figure_data.py from the frozen cohort outputs). Output: figs_v2/out/figS2.*"""
import sys
import textwrap
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
f = pd.read_csv(D / 'S2_flow.csv', encoding='utf-8-sig')
cohorts = list(dict.fromkeys(f.cohort))
print('Files read:\n ', D / 'S2_flow.csv')

HMM = 108.0
fig = plt.figure(figsize=(W2, HMM * MM))
A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, 183); A.set_ylim(0, HMM); A.axis('off')
P = 183 / len(cohorts); BW = P - 3.2; BH = 9.6; PITCH = 12.6; TOP = HMM - 22.0


def num(n):
    n = int(n)
    return f'{n:,}'.replace(',', ' ') if n >= 10000 else str(n)


def box(x, y, w, h, fc, ec, lw=0.5):
    A.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1):
    A.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.5, color=INK, mutation_scale=4, shrinkA=0, shrinkB=0))


for i, c in enumerate(cohorts):
    x = 1.6 + i * P; xc = x + BW / 2
    head = c.replace(' (training)', '\n(training)').replace(' (supplementary)', '\n(supplementary)').replace('Understanding Society ', 'Understanding\nSociety ')
    A.text(xc, HMM - 6.5, head, ha='center', va='center', fontsize=6, fontweight='bold', linespacing=1.1)
    q = f[f.cohort == c].sort_values('step_order')
    final = q.step_order.max() if 'CLHLS' not in c else 4          # analysed complete cases (CLHLS: before the age split)
    for _, r in q.iterrows():
        k = int(r.step_order)
        if 'CLHLS' in c and k == 6:        # aged 80 and over branches from the complete-case box, beside the 65-79 box
            row = 5
        else:
            row = k
        y = TOP - row * PITCH - (PITCH if ('CLHLS' in c and k == 6) else 0)
        analysed = k == final
        primary = 'CLHLS' in c and k == 5
        fc = '#EAF2EC' if (analysed or primary) else '#F2F2F2'
        ec = RULE if (analysed or primary) else BASE
        box(x, y, BW, BH, fc, ec, lw=0.6 if (analysed or primary) else 0.5)
        lab = '\n'.join(textwrap.wrap(r.step, 21))
        A.text(xc, y + BH / 2 + 1.7, lab, ha='center', va='center', fontsize=5, linespacing=1.1)
        A.text(xc, y + 1.7, num(r.n), ha='center', va='center', fontsize=5.5, fontweight='bold')
        if k > 0 and not ('CLHLS' in c and k == 6):
            arrow(xc, y + BH + PITCH - BH, xc, y + BH)
        if 'CLHLS' in c and k == 6:          # elbow from the complete-case box (row 4) to this box
            yb = TOP - 4 * PITCH
            A.plot([x + BW, x + BW + 1.2, x + BW + 1.2], [yb + BH / 2, yb + BH / 2, y + BH / 2], color=INK, lw=0.5)
            arrow(x + BW + 1.2, y + BH / 2, x + BW, y + BH / 2)
A.text(1.6, 3.0, 'Green outline: analysed complete cases (CLHLS: 65–79 years primary; 80 and over reported separately).', fontsize=5, color=INK, va='center')
save(fig, OUT / 'figS2')
print('ok')
