#!/usr/bin/env python3
"""Figure 7 of NC v0.8 (new main-text figure): local fee schedules reshape the matched panel, and a sequential ordering chart cuts fees.
Combines Supplementary Fig. 4 (panel a) and Supplementary Fig. 7 (panels b-e) on one double-column canvas (183 mm x 168 mm).
a, analytes in the cheapest panel matching the reference under US, Australian, Japanese and Chinese fees and on the CHNS, CLHLS and UK menus
   (+ CVD death, USA); b, the value-of-information ordering chart; c, laboratory fee as a share of the routine bundle (95% CI);
c-d, excess deaths not flagged at the 10% threshold per 1000 versus the routine bundle (95% CI), earlier rule and chart;
e, the same for the chart by sex, age and race and ethnicity (groups with fewer than 20 deaths not estimated).
Inputs (read only; the same files as figS4_panels.py and figS7_ordering.py):
  manuscript_ldh/figures_v2/data/F3A_panel_items.csv, F4A_chart.csv, F4BC_cost_missed.csv
  task22_adaptive/t22_NH3_subgroups.csv, t22_GH_subgroups.csv, t22_I_subgroups.csv
Output: figs_v2/out/fig7.pdf/.png"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
m = pd.read_csv(D / 'F3A_panel_items.csv', encoding='utf-8-sig')
CH = pd.read_csv(D / 'F4A_chart.csv', encoding='utf-8-sig'); BC = pd.read_csv(D / 'F4BC_cost_missed.csv', encoding='utf-8-sig')
FD = {'NHANES III': W / 'task22_adaptive/t22_NH3_subgroups.csv', 'NHANES 2011–14': W / 'task22_adaptive/t22_GH_subgroups.csv',
      'NHANES 2015–16': W / 'task22_adaptive/t22_I_subgroups.csv'}
SG = {c: pd.read_csv(f, encoding='utf-8') for c, f in FD.items()}
COH = list(FD)
VOI, OLD = 'Value-of-information rule', 'Earlier rule'

FW, FH = 183.0, 168.0
fig = plt.figure(figsize=(FW * MM, FH * MM))
A = fig.add_axes([0, 0, 1, 1]); A.set_xlim(0, FW); A.set_ylim(0, FH); A.axis('off')


def axmm(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


# ---------------- a: matched panel under each fee schedule / menu (from figS4_panels.py) ----------------
COLS = [('USA', 'USA'), ('Australia', 'Australia'), ('Japan', 'Japan'), ('China, CHNS menu', 'China\n(CHNS)'),
        ('China, CLHLS menu', 'China\n(CLHLS)'), ('UK menu', 'UK menu\n(US prices)'), ('Cardiovascular death (USA)', 'CVD death\n(USA)')]
ITEMS = [('CMP', 'CMP (panel)'), ('CBC', 'CBC'), ('phosphorus', 'Phosphorus'), ('ggt', 'GGT'), ('ldh', 'LDH'), ('hdl', 'HDL cholesterol'),
         ('uric_acid', 'Uric acid'), ('glucose', 'Glucose'), ('bun', 'Urea'), ('creatinine', 'Creatinine'), ('albumin', 'Albumin'), ('protein', 'Total protein')]
assert set(m.item) <= {i for i, _ in ITEMS}, set(m.item) - {i for i, _ in ITEMS}
fig.text(0.0, 1.0, 'a', fontsize=8, fontweight='bold', va='top', ha='left')
A.text(5.0, 165.2, 'Matched panel under each fee schedule and menu', fontsize=6.5, va='center', ha='left')
X0, PX, PY, Y0, S, SW = 29.0, 10.4, 4.2, 158.0, 3.5, 5.6   # SW: cell width (wide enough for '6/14')
for i, (it, lab) in enumerate(ITEMS):
    y = Y0 - i * PY
    A.text(X0 - 3.6, y, lab, ha='right', va='center', fontsize=5.5)
    for j, (c, _) in enumerate(COLS):
        x = X0 + j * PX
        hit = m[(m.column == c) & (m.item == it)]
        A.add_patch(Rectangle((x - SW / 2, y - S / 2), SW, S, fc=MATCHED if len(hit) else '#EDEDED', ec='none'))
        if len(hit) and hit.iloc[0].how.startswith('6 measured CMP'):
            A.text(x, y, '6/14', ha='center', va='center', fontsize=5, color='white', fontweight='bold')
A.plot([6.0, X0 + 6 * PX + SW / 2], [Y0 - 1.5 * PY] * 2, color=BASE, lw=0.4)   # bundles above, single analytes below
yl = Y0 - (len(ITEMS) - 1) * PY - S / 2 - 1.0
for j, (c, lab) in enumerate(COLS):
    x = X0 + j * PX; r = m[m.column == c].iloc[0]
    A.text(x, yl, lab, ha='center', va='top', fontsize=5, linespacing=1.05)
    A.text(x, yl - 6.6, r.panel_price, ha='center', va='top', fontsize=5, fontweight='bold')
    A.text(x, yl - 9.6, r.reference_price.replace(' (PREVENT full set)', ''), ha='center', va='top', fontsize=5, color=INK)
A.text(X0 - 5.4, yl - 6.6, 'Panel fee', ha='right', va='top', fontsize=5, fontweight='bold')
A.text(X0 - 5.4, yl - 9.6, 'Reference', ha='right', va='top', fontsize=5)
A.text(5.0, yl - 14.4, 'Orange: analyte in the panel (6/14: six measured CMP analytes). Reference: routine bundle;\n'
       'for CVD death, the PREVENT laboratory inputs. China: Chinese fees on each cohort\'s menu.',
       fontsize=5, va='top', ha='left', linespacing=1.25, color='#4B4B4B')

# ---------------- b: ordering chart ----------------
ax = axmm(122.0, 140.0, 38.0, 20.5)
for y, r in CH.iterrows():
    ax.barh(y, r.risk_high - r.risk_low, left=r.risk_low, height=0.45, color=REDESIGN, alpha=[1, 0.7, 0.45][y], lw=0)
    ax.text(r.risk_high + 0.006, y, f'order {r.next_test} (US${r.next_fee_usd:.2f})\nif risk {r.risk_low:.1%}–{r.risk_high:.1%}',
            va='center', fontsize=5, linespacing=1.1)
ax.axvline(0.10, color=INK, lw=0.5, ls=DASH)
ax.text(0.10, -0.62, '10% threshold', ha='center', va='bottom', fontsize=5, bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
ax.set_ylim(2.55, -0.8); ax.set_yticks(range(len(CH))); ax.set_yticklabels(['Clinical\ninformation only', 'After CMP', 'After CMP\nand CBC'], fontsize=5.5)
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlim(0, 0.3); ax.set_xticks(np.arange(0, 0.31, 0.1)); ax.set_xticklabels([f'{v:.0%}' for v in np.arange(0, 0.31, 0.1)])
ax.set_xlabel('Predicted 9-year risk of death', labelpad=1.5)
fig.text(100.0 / FW, 1.0, 'b', fontsize=8, fontweight='bold', va='top', ha='left')
A.text(104.5, 165.2, 'Sequential ordering chart: when to order the next test', fontsize=6.5, va='center', ha='left')

# ---------------- c, d: fee share and missed deaths ----------------
x = np.arange(len(COH))
for k, (xmm, col, lo, hi, yl_, ttl, lab) in enumerate([
        (113.0, 'cost_share', 'cost_share_lo', 'cost_share_hi', 'Laboratory fee\n(% of routine)', 'Fee relative to routine', 'c'),
        (157.0, 'extra_missed_per1000', 'extra_missed_lo', 'extra_missed_hi', 'Excess unflagged\ndeaths per 1000', 'Missed deaths vs routine', 'd')]):
    ax = axmm(xmm, 99.0, 25.0, 18.0)
    for rule, dx, fc in [(OLD, -0.16, 'white'), (VOI, 0.16, REDESIGN)]:
        for i, c in enumerate(COH):
            r = BC[(BC.cohort == c) & (BC.rule == rule)].iloc[0]
            ec = BASE if rule == OLD else REDESIGN
            if k == 0:
                ax.bar(i + dx, r[col], width=0.3, color=LIGHT if rule == OLD else REDESIGN, edgecolor=ec, lw=0.5)
                ax.plot([i + dx] * 2, [r[lo], r[hi]], color=INK, lw=0.7)
            else:
                ax.plot([i + dx] * 2, [r[lo], r[hi]], color=ec, lw=0.8)
                ax.plot(i + dx, r[col], 'o', ms=2.6, mfc=fc, mec=ec, mew=0.7)
    ax.set_xticks(x); ax.set_xticklabels([c.replace('NHANES ', '') for c in COH], fontsize=5); ax.tick_params(axis='x', length=0, pad=1.5)
    ax.set_xlabel('NHANES cohort', fontsize=5.5, labelpad=1.5)
    ax.tick_params(axis='y', labelsize=5)
    ax.set_xlim(-0.5, len(COH) - 0.5); ax.set_ylabel(yl_, fontsize=5.5, labelpad=1.5); ax.set_title(ttl, fontsize=6, pad=2)
    fig.text((xmm - 13.0) / FW, 122.0 / FH, lab, fontsize=8, fontweight='bold', va='top', ha='left')
    if k == 0:
        ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.2, 0.4, 0.6]); ax.set_yticklabels(['0', '20', '40', '60'])
        ax.axhline(0.5, color=INK, lw=0.5, ls=DASH); ax.text(len(COH) - 0.55, 0.51, 'Prespecified limit (50%)', ha='right', va='bottom', fontsize=5)
    else:
        ax.axhline(0, color=INK, lw=0.5); ax.set_ylim(-1, 6); ax.set_yticks([-1, 0, 2, 4, 6]); ax.set_yticklabels(['−1', '0', '2', '4', '6'])
fig.legend([Patch(facecolor=LIGHT, edgecolor=BASE, lw=0.5), Patch(facecolor=REDESIGN, edgecolor=REDESIGN, lw=0.5)],
           ['Earlier rule (open)', 'Ordering chart (filled)'], loc='center', bbox_to_anchor=(146.0 / FW, 127.5 / FH), ncol=2, fontsize=5.5,
           handlelength=1.0, columnspacing=1.2)

# ---------------- e: missed deaths by group ----------------
GL = {'female': 'Women', 'male': 'Men', 'age40-59': 'Age 40–59 years', 'age60-79': 'Age 60–79 years', 'Mexican American': 'Mexican American',
      'Other Hispanic': 'Other Hispanic', 'Non-Hispanic White': 'Non-Hispanic White', 'Non-Hispanic Black': 'Non-Hispanic Black', 'Other race': 'Other race'}
groups = list(SG[COH[0]].group)
A.plot([2.0, 181.0], [86.0, 86.0], color='#E5E5E5', lw=0.5)
fig.text(0.0, 84.5 / FH, 'e', fontsize=8, fontweight='bold', va='top', ha='left')
A.text(5.0, 81.8, 'Ordering chart: missed deaths by group (chart minus routine bundle; 95% CI)', fontsize=6.5, va='center', ha='left')
for j, c in enumerate(COH):
    ax = axmm(31.0 + j * 51.0, 11.0, 40.0, 64.0)
    s = SG[c].set_index('group')
    for y, g in enumerate(groups):
        r = s.loc[g]
        if pd.isna(r.dmissed):
            txt = 'not in survey' if r.n == 0 else '<20 deaths'
            ax.text(0.01, y, txt, ha='left', va='center', fontsize=5, color=BASE, transform=ax.get_yaxis_transform(), zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
            continue
        ax.plot([r.dmissed_lo, r.dmissed_hi], [y, y], color=REDESIGN, lw=0.8)
        ax.plot(r.dmissed, y, 'o', ms=2.6, color=REDESIGN, mew=0)
        ax.text(1.03, y, f'{int(r.deaths)}', ha='left', va='center', fontsize=5, color=BASE, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, lw=0.5)
    for yb in (1.5, 3.5):
        ax.axhline(yb, color=LIGHT, lw=0.5)
    ax.set_ylim(len(groups) - 0.5, -0.5); ax.set_xlim(-3, 11); ax.set_xticks([-2, 0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([f'{v:g}'.replace('-', '−') for v in ax.get_xticks()])
    ax.set_xlabel('Excess unflagged deaths per 1000', labelpad=1.5)
    ax.set_title(c, loc='center', fontsize=6, pad=3)
    ax.text(1.03, 1.0, 'deaths', ha='left', va='bottom', fontsize=5, color=BASE, transform=ax.transAxes)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    if j == 0:
        ax.set_yticks(range(len(groups))); ax.set_yticklabels([GL[g] for g in groups], fontsize=5.5)
    else:
        ax.set_yticks([])
save(fig, OUT / 'fig7'); print('ok')
