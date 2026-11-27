#!/usr/bin/env python3
"""Supplementary Fig. 5: cardiovascular death against refitted and published guideline equations.
a, Harrell's C of models with the same clinical predictors refitted with each guideline's laboratory inputs and with the target-aware
   cardiovascular panel within the PREVENT price (same data as manuscript_ldh Figure 3C).
b, the US$45.83 panel with clinical variables minus the published (not refitted) pooled cohort equations and PREVENT base equations,
   other deaths censored and competing risks, 95% bootstrap CI.
Inputs (read only):
  manuscript_ldh/figures_v2/data/F3C_cvd.csv     (panel a; exported from task21_final/t21_NH3_endpoints.json and t21_GH_endpoints.json, key 'cvd')
  task32_tier2/t32c_NH3.json, task32_tier2/t32c_GH.json   (panel b)
Output: figs_v2/out/figS5.pdf/.png"""
import json
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FA = W / 'manuscript_ldh/figures_v2/data/F3C_cvd.csv'
FB = {'NH3': W / 'task32_tier2/t32c_NH3.json', 'GH': W / 'task32_tier2/t32c_GH.json'}
print('Files read:'); [print(' ', f) for f in (FA, *FB.values())]
A = pd.read_csv(FA, encoding='utf-8-sig')
B = {c: json.load(open(f, encoding='utf-8')) for c, f in FB.items()}
COH = ['NHANES III', 'NHANES 2011–14']

fig = plt.figure(figsize=(W2, 66 * MM))
POS = [[0.185, 0.27, 0.115, 0.56], [0.325, 0.27, 0.115, 0.56], [0.6, 0.27, 0.2, 0.56]]
# ---- a: C of refitted models
for i, coh in enumerate(COH):
    r = A[A.cohort == coh].reset_index(drop=True)
    ax = fig.add_axes(POS[i])
    for y, row in r.iterrows():
        aware = row.panel == 'Target-aware panel'
        ax.plot(row.C, y, 'o', ms=4 if aware else 3, color=REDESIGN if aware else BASE, mew=0)
    ax.set_ylim(len(r) - 0.5, -0.5); ax.set_xlim(0.76, 0.86); ax.set_xticks([0.78, 0.82, 0.86])
    ax.set_xlabel("Harrell's C")
    ax.set_title(f'{coh}\n({int(r.cvd_deaths[0])} CVD deaths)', loc='center', fontsize=6)
    ax.grid(axis='x', color=LIGHT, lw=0.4); ax.set_axisbelow(True)
    if i == 0:
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels([p if u == 0 else f'{p} (US${u:.2f})' for p, u in zip(r.panel, r.usd)])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        label(ax, 'a', -1.55, 1.1)
    else:
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
# ---- b: panel minus published equations
ax = fig.add_axes(POS[2])
REF = [('PCE (published)', 'pooled cohort equations'), ('PREVENT base (published)', 'PREVENT base equations')]
rows, ylab = [], []
for c, coh in [('NH3', 'NHANES III'), ('GH', 'NHANES 2011–14')]:
    for k, nm in REF:
        rows.append((c, k)); ylab.append(f'{coh}\nvs {nm}')
for y, (c, k) in enumerate(rows):
    key = f'CVD aware@PREVENT-full price - {k}'
    for kind, dy, mfc in [('contrasts_censored', -0.15, INK), ('contrasts_competing', 0.15, 'white')]:
        d = B[c][kind][key]; lo, hi = d['ci']
        ax.plot([lo, hi], [y + dy] * 2, color=INK, lw=0.7)
        ax.plot(d['point'], y + dy, 'o', ms=3, mfc=mfc, mec=INK, mew=0.6)
        ax.text(1.02, y + dy, f"{d['point']:+.4f} ({lo:.4f} to {hi:.4f})".replace('-', '−'), transform=ax.get_yaxis_transform(),
                va='center', fontsize=5, color=INK)
ax.axvline(0, color=INK, lw=0.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(ylab); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_ylim(len(rows) - 0.5, -0.5)
ax.set_xlabel("Difference in Harrell's C")
ax.set_xlim(-0.02, 0.12); ax.set_xticks([0, 0.04, 0.08, 0.12])
ax.legend([Line2D([], [], marker='o', ls='-', color=INK, mfc=INK, ms=3, lw=0.7),
           Line2D([], [], marker='o', ls='-', color=INK, mfc='white', ms=3, lw=0.7, mew=0.6)],
          ['Other deaths censored', 'Competing risks'], loc='upper left', bbox_to_anchor=(-0.02, -0.24), ncol=2, fontsize=5)
n = {c: (B[c]['n'], B[c]['cvd_deaths']) for c in B}
ax.set_title('US$45.83 panel minus published equations\n(not refitted; 95% CI)', loc='left', fontsize=6)
ax.text(0, -0.36, f'NHANES III n = {n["NH3"][0]} ({n["NH3"][1]} CVD deaths); NHANES 2011–14 n = {n["GH"][0]} ({n["GH"][1]})',
        transform=ax.transAxes, fontsize=5, color=INK, va='top')
ax.text(1.02, 1.0, 'Difference (95% CI)', transform=ax.transAxes, fontsize=5, va='bottom', fontweight='bold')
label(ax, 'b', -0.75, 1.1)
save(fig, HERE / 'out' / 'figS5'); print('ok')
