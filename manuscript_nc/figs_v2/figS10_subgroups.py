#!/usr/bin/env python3
"""Supplementary Fig. 10 of NC v0.7: subgroup differences in discrimination, matched panel minus routine bundle. Redrawn in the NC style from the
same data as the Lancet-draft Appendix Figure S1: manuscript_ldh/figures_v2/data/S1_subgroups.csv (written by export_figure_data.py from
task21 t21_*_subgroups.csv). Output: figs_v2/out/figS10.*"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
d = pd.read_csv(D / 'S1_subgroups.csv', encoding='utf-8-sig')
print('Files read:\n ', D / 'S1_subgroups.csv')
GROUPS = [('female', 'Women'), ('male', 'Men'), ('Age 40–59', 'Age 40–59 years'), ('Age 60–79', 'Age 60–79 years'), ('Mexican American', 'Mexican American'),
          ('Non-Hispanic White', 'Non-Hispanic White'), ('Non-Hispanic Black', 'Non-Hispanic Black'), ('Other race', 'Other race'),
          ('Income <1.3× poverty line', 'Income <1.3× poverty line'), ('Income 1.3–3.5×', 'Income 1.3–3.5×'), ('Income ≥3.5×', 'Income ≥3.5×'),
          ('below high school', 'Below high school'), ('high school', 'High school'), ('above high school', 'Above high school')]
assert set(d.group) <= {g for g, _ in GROUPS}, set(d.group) - {g for g, _ in GROUPS}
SEP = [2, 4, 8, 11]                       # boundaries between sex, age, race and ethnicity, income, education
COH = [('NHANES III', "Harrell's C"), ('NHANES 2011–14', "Harrell's C"), ('Understanding Society', 'AUC')]
fig, axs = plt.subplots(1, 3, figsize=(W2, 84 * MM), layout='constrained', sharey=True)
n = len(GROUPS)
for j, (ax, (c, met)) in enumerate(zip(axs, COH)):
    q = d[d.cohort == c].set_index('group')
    for i, (g, _) in enumerate(GROUPS):
        y = n - 1 - i
        if g not in q.index:
            continue
        r = q.loc[g]
        ax.plot([r.lo, r.hi], [y, y], color=MATCHED, lw=0.9, solid_capstyle='butt')
        ax.plot(r.est, y, 'o', ms=2.8, color=MATCHED, mew=0)
        ax.text(1.0, y, f'{int(r.deaths)}', transform=ax.get_yaxis_transform(), fontsize=5, va='center', ha='left', color=INK)
    for b in SEP:
        ax.axhline(n - b - 0.5, color=LIGHT, lw=0.4, zorder=0)
    ax.axvline(0, color=INK, lw=0.5); ax.axvline(-0.01, color=INK, lw=0.5, ls=DASH)
    ax.set_xlim(-0.04, 0.02); ax.set_xticks([-0.04, -0.02, 0, 0.02]); ax.set_xticklabels(['−0.04', '−0.02', '0', '0.02'])
    ax.set_ylim(-0.6, n - 0.4); ax.set_yticks(range(n)); ax.set_yticklabels([lab for _, lab in GROUPS][::-1]); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f'Difference in {met}\n(matched panel − routine bundle)')
    ax.set_title(c, fontsize=6, loc='center')
    ax.text(1.0, n - 0.4, 'deaths', transform=ax.get_yaxis_transform(), fontsize=5, va='bottom', ha='left', color=INK)
save(fig, OUT / 'figS10')
print('ok')
