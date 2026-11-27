#!/usr/bin/env python3
"""Supplementary Fig. 7: the sequential (value-of-information) ordering chart: cost and missed deaths, overall and by group.
a, the chart (risk intervals in which the next test is ordered); b, laboratory fee as a share of the routine bundle (95% CI);
c, excess deaths not flagged at the 10% threshold per 1000 people versus the routine bundle (95% CI), earlier rule and chart;
d, the same for the chart by sex, age and race and ethnicity (groups with fewer than 20 deaths not estimated).
Inputs (read only):
  manuscript_ldh/figures_v2/data/F4A_chart.csv        (panel a; exported from task22_adaptive/rule_t22.json)
  manuscript_ldh/figures_v2/data/F4BC_cost_missed.csv (panels b, c; exported from task22_adaptive/t22_{NH3,GH,I}_endpoints.json)
  task22_adaptive/t22_NH3_subgroups.csv, t22_GH_subgroups.csv, t22_I_subgroups.csv   (panel d)
Output: figs_v2/out/figS7.pdf/.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

D = W / 'manuscript_ldh/figures_v2/data'
FA, FBC = D / 'F4A_chart.csv', D / 'F4BC_cost_missed.csv'
FD = {'NHANES III': W / 'task22_adaptive/t22_NH3_subgroups.csv', 'NHANES 2011–14': W / 'task22_adaptive/t22_GH_subgroups.csv',
      'NHANES 2015–16': W / 'task22_adaptive/t22_I_subgroups.csv'}
print('Files read:'); [print(' ', f) for f in (FA, FBC, *FD.values())]
CH = pd.read_csv(FA, encoding='utf-8-sig'); BC = pd.read_csv(FBC, encoding='utf-8-sig')
SG = {c: pd.read_csv(f, encoding='utf-8') for c, f in FD.items()}
COH = list(FD)
VOI, OLD = 'Value-of-information rule', 'Earlier rule'

fig = plt.figure(figsize=(W2, 130 * MM))
# ---- a: chart
ax = fig.add_axes([0.13, 0.64, 0.3, 0.28])
for y, r in CH.iterrows():
    ax.barh(y, r.risk_high - r.risk_low, left=r.risk_low, height=0.42, color=REDESIGN, alpha=[1, 0.7, 0.45][y], lw=0)
    ax.text(r.risk_high + 0.006, y, f'order {r.next_test} (US${r.next_fee_usd:.2f})\nif risk {r.risk_low:.1%}–{r.risk_high:.1%}', va='center', fontsize=5.3)
ax.axvline(0.10, color=INK, lw=0.5, ls=DASH)
ax.text(0.10, -0.62, '10% decision threshold', ha='center', va='bottom', fontsize=5, bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
ax.set_ylim(2.6, -0.75); ax.set_yticks(range(len(CH))); ax.set_yticklabels(['Clinical information\nonly', 'After CMP', 'After CMP\nand CBC'])
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(0, 0.3); ax.set_xticks(np.arange(0, 0.31, 0.05)); ax.set_xticklabels([f'{v:.0%}' for v in np.arange(0, 0.31, 0.05)])
ax.set_xlabel('Predicted 9-year risk of death (%)')
ax.set_title('When to order the next test'); label(ax, 'a', -0.42, 1.04)
# ---- b, c
x = np.arange(len(COH))
for k, (pos, col, lo, hi, yl, ttl, lab) in enumerate([
        ([0.54, 0.64, 0.19, 0.28], 'cost_share', 'cost_share_lo', 'cost_share_hi', 'Laboratory fee\n(% of routine bundle)', 'Fee relative to routine', 'b'),
        ([0.8, 0.64, 0.19, 0.28], 'extra_missed_per1000', 'extra_missed_lo', 'extra_missed_hi',
         'Excess unflagged deaths per 1000\n(difference from routine bundle)', 'Missed deaths', 'c')]):
    ax = fig.add_axes(pos)
    for rule, dx, fc in [(OLD, -0.15, 'white'), (VOI, 0.15, REDESIGN)]:
        for i, c in enumerate(COH):
            r = BC[(BC.cohort == c) & (BC.rule == rule)].iloc[0]
            ec = BASE if rule == OLD else REDESIGN
            if k == 0:
                ax.bar(i + dx, r[col], width=0.28, color=LIGHT if rule == OLD else REDESIGN, edgecolor=ec, lw=0.5)
                ax.plot([i + dx] * 2, [r[lo], r[hi]], color=INK, lw=0.7)
            else:
                ax.plot([i + dx] * 2, [r[lo], r[hi]], color=ec, lw=0.8)
                ax.plot(i + dx, r[col], 'o', ms=3, mfc=fc, mec=ec, mew=0.7)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('NHANES ', 'NHANES\n') for c in COH]); ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(COH) - 0.5); ax.set_ylabel(yl); ax.set_title(ttl); label(ax, lab, -0.42, 1.04)
    if k == 0:
        ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_yticklabels(['0', '20', '40', '60'])
        ax.axhline(0.5, color=INK, lw=0.5, ls=DASH); ax.text(len(COH) - 0.55, 0.51, 'Prespecified limit (50%)', ha='right', va='bottom', fontsize=5)
    else:
        ax.axhline(0, color=INK, lw=0.5); ax.set_ylim(-1, 6); ax.set_yticks([-1, 0, 2, 4, 6]); ax.set_yticklabels(['−1', '0', '2', '4', '6'])
fig.legend([Patch(facecolor=LIGHT, edgecolor=BASE, lw=0.5), Patch(facecolor=REDESIGN, edgecolor=REDESIGN, lw=0.5)],
           [OLD + ' (open)', 'Value-of-information chart (filled)'], loc='upper center', bbox_to_anchor=(0.77, 0.56), ncol=2, fontsize=5.5)
# ---- d: by group
GL = {'female': 'Women', 'male': 'Men', 'age40-59': 'Age 40–59 years', 'age60-79': 'Age 60–79 years', 'Mexican American': 'Mexican American',
      'Other Hispanic': 'Other Hispanic', 'Non-Hispanic White': 'Non-Hispanic White', 'Non-Hispanic Black': 'Non-Hispanic Black', 'Other race': 'Other race'}
groups = list(SG[COH[0]].group)
for j, c in enumerate(COH):
    ax = fig.add_axes([0.2 + j * 0.27, 0.08, 0.22, 0.38])
    s = SG[c].set_index('group')
    for y, g in enumerate(groups):
        r = s.loc[g]
        if pd.isna(r.dmissed):
            txt = 'not in survey' if r.n == 0 else '<20 deaths'
            ax.text(0.01, y, txt, ha='left', va='center', fontsize=5, color=BASE, transform=ax.get_yaxis_transform(), zorder=5, bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
            continue
        ax.plot([r.dmissed_lo, r.dmissed_hi], [y, y], color=REDESIGN, lw=0.8)
        ax.plot(r.dmissed, y, 'o', ms=3, color=REDESIGN, mew=0)
        ax.text(1.03, y, f'{int(r.deaths)}', ha='left', va='center', fontsize=5, color=BASE, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=0.5)
    for yb in (1.5, 3.5):
        ax.axhline(yb, color=LIGHT, lw=0.5)
    ax.set_ylim(len(groups) - 0.5, -0.5); ax.set_xlim(-3, 11); ax.set_xticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([f'{v:g}'.replace('-', '−') for v in ax.get_xticks()])
    ax.set_xlabel('Excess unflagged deaths per 1000\n(chart minus routine bundle)')
    ax.set_title(c, loc='center', fontsize=6)
    ax.text(1.03, 1.0, 'deaths', ha='left', va='bottom', fontsize=5, color=BASE, transform=ax.transAxes)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    if j == 0:
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([GL[g] for g in groups]); label(ax, 'd', -0.62, 1.04)
    else:
        ax.set_yticks([])
save(fig, HERE / 'out' / 'figS7'); print('ok')
