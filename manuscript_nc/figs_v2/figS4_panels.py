#!/usr/bin/env python3
"""Supplementary Fig. 4 of NC v0.7: analytes in the cheapest panel matching the reference under each fee schedule, menu and outcome.
Redrawn in the NC style from the same data as panel A of the Lancet-draft Figure 3: manuscript_ldh/figures_v2/data/F3A_panel_items.csv
(written by export_figure_data.py from the frozen selected-panel files of Tasks 19, 20, 25, 27 and the CVD selection). Output: figs_v2/out/figS4.*"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
m = pd.read_csv(D / 'F3A_panel_items.csv', encoding='utf-8-sig')
print('Files read:\n ', D / 'F3A_panel_items.csv')
COLS = [('USA', 'USA'), ('Australia', 'Australia'), ('Japan', 'Japan'), ('China, CHNS menu', 'China\n(CHNS)'),
        ('China, CLHLS menu', 'China\n(CLHLS)'), ('UK menu', 'UK menu\n(US prices)'), ('Cardiovascular death (USA)', 'CVD death\n(USA)')]
ITEMS = [('CMP', 'CMP (panel)'), ('CBC', 'CBC'), ('phosphorus', 'Phosphorus'), ('ggt', 'GGT'), ('ldh', 'LDH'), ('hdl', 'HDL cholesterol'),
         ('uric_acid', 'Uric acid'), ('glucose', 'Glucose'), ('bun', 'Urea'), ('creatinine', 'Creatinine'), ('albumin', 'Albumin'), ('protein', 'Total protein')]
assert set(m.item) <= {i for i, _ in ITEMS}, set(m.item) - {i for i, _ in ITEMS}
HMM = 88.0
fig = plt.figure(figsize=(W15, HMM * MM))
A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, 120); A.set_ylim(0, HMM); A.axis('off')
X0, PX, PY, Y0, S = 31.0, 12.6, 5.0, HMM - 5.0, 4.0          # first column centre, column pitch, row pitch, first row centre, cell size (mm)
for i, (it, lab) in enumerate(ITEMS):
    y = Y0 - i * PY
    A.text(X0 - 4.2, y, lab, ha='right', va='center', fontsize=5.5)
    for j, (c, _) in enumerate(COLS):
        x = X0 + j * PX
        hit = m[(m.column == c) & (m.item == it)]
        A.add_patch(Rectangle((x - S / 2, y - S / 2), S, S, fc=MATCHED if len(hit) else '#EDEDED', ec='none'))
        if len(hit) and hit.iloc[0].how.startswith('6 measured CMP'):
            A.text(x, y, '6/14', ha='center', va='center', fontsize=5, color='white', fontweight='bold')
A.plot([X0 - 3.0 - 20, X0 + 6 * PX + S / 2], [Y0 - 1.5 * PY] * 2, color=BASE, lw=0.4)   # bundles above, single analytes below
yl = Y0 - (len(ITEMS) - 1) * PY - S / 2 - 1.2
for j, (c, lab) in enumerate(COLS):
    x = X0 + j * PX; r = m[m.column == c].iloc[0]
    A.text(x, yl, lab, ha='center', va='top', fontsize=5.5, linespacing=1.1)
    A.text(x, yl - 8.6, r.panel_price, ha='center', va='top', fontsize=5.5, fontweight='bold')
    A.text(x, yl - 12.2, r.reference_price.replace(' (PREVENT full set)', ''), ha='center', va='top', fontsize=5.5, color=INK)
A.text(X0 - 6.5, yl - 8.6, 'Panel fee', ha='right', va='top', fontsize=5.5, fontweight='bold')
A.text(X0 - 6.5, yl - 12.2, 'Reference fee', ha='right', va='top', fontsize=5.5)
A.text(2.0, 3.4, "Orange, analyte in the panel. China (CHNS), China (CLHLS): Chinese fees on each cohort's menu.\n"
       'Reference: routine bundle; for CVD death, the PREVENT full laboratory inputs.', fontsize=5, va='center', ha='left', linespacing=1.3)
save(fig, OUT / 'figS4')
print('ok')
