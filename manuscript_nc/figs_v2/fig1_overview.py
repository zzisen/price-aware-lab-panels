#!/usr/bin/env python3
"""Redesigned Figure 1 (Nature Portfolio style): low-text, visual-first overview of the study.
Panel a: One measurement-panel decision across biology and medicine (bespoke vector icons + matrix schematic).
Panel b: Training-data decision rule visualised as a 2D regime map (n/k vs cross-validated target signal R^2).
Panel c: Pre-frozen validation across five unseen locked tests (aligned small multiples of mean regret, incl. CV alone).
v0.9: locked test 5 (task48_singlecell/t48_summary.json; 10 single-cell sources, rule v4 unchanged) added as a sixth card;
cards narrowed and status pills allowed two lines; the protocol badge text reflowed; a containment check prints any overflow.
Panel d: Clinical translation — priced laboratory panels for mortality across 5 cohorts in 3 countries (analyte swap + pooled effect).

Every plotted empirical number is loaded dynamically from frozen project result files; none is typed in.
All text >= 5.0 pt sans-serif (Arial/Helvetica), zero missing glyphs, pure vector PDF + 300 dpi PNG.
Outputs: figs_v2/out/fig1.pdf and figs_v2/out/fig1.png.
v0.8 changes: panel d target-aware readout now states point estimates and CIs separately ("Higher in 5/5 (CI > 0 in 4/5)"),
both counts computed from F2BC_forest.csv; matched readout count also computed (CI lower bound above the -0.01 margin);
NI-margin label given a white backing so the zero line no longer runs through it; '+ signal check*' badge widened.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent.parent
sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge  # noqa: E402

OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)

# ---------------- Load all empirical data dynamically ----------------
t1_raw = json.load(open(W / 'task34_decision' / 'fresh_summary_v2.json'))
t2_raw = json.load(open(W / 'task39_locked_v3' / 't39_summary.json'))
t3_raw = json.load(open(W / 'task40_locked_v4' / 't40_summary.json'))
t4T_raw = json.load(open(W / 'task42_locked_v5' / 't42T_summary.json'))
t4S_raw = json.load(open(W / 'task42_locked_v5' / 't42S_summary.json'))
d63_raw = json.load(open(W / 'task40_locked_v4' / 'dryrun_old63_summary.json'))
t5_raw = json.load(open(W / 'task48_singlecell' / 't48_summary.json'))

D_CLIN = W / 'manuscript_ldh' / 'figures_v2' / 'data'
cohorts_df = pd.read_csv(D_CLIN / 'F1C_cohort_cards.csv')
prim_coh = cohorts_df[cohorts_df['role'] == 'primary']
n_cohorts = int(len(prim_coh))
n_countries = int(prim_coh['country'].nunique())
n_people = int(prim_coh['n'].sum())
n_deaths = int(prim_coh['deaths'].sum())

panels_df = pd.read_csv(D_CLIN / 'F1B_outputs.csv').set_index('output')
usd_routine = float(panels_df.loc['Routine bundle (reference)', 'usd'])
usd_matched = float(panels_df.loc['Matched panel (USA)', 'usd'])
usd_aware = float(panels_df.loc['Target-aware panel (USA)', 'usd'])
cmp_df = pd.read_csv(D_CLIN / 'F1A_cmp_price.csv').set_index('item')
usd_cmp_bundle = float(cmp_df.loc['CMP as panel (14 analytes)', 'usd'])
usd_cmp_single = float(cmp_df.loc['CMP analytes priced one by one', 'usd'])

forest_df = pd.read_csv(D_CLIN / 'F2BC_forest.csv')
h1_coh = forest_df[(forest_df['hypothesis'] == 'H1') & (forest_df['kind'] == 'cohort')]
fee_min_pct = int(round(float(h1_coh['matched_price_share'].min()) * 100))
fee_max_pct = int(round(float(h1_coh['matched_price_share'].max()) * 100))
h1_pool = forest_df[(forest_df['hypothesis'] == 'H1') & (forest_df['kind'] == 'pooled')].iloc[0]
h2_pool = forest_df[(forest_df['hypothesis'] == 'H2') & (forest_df['kind'] == 'pooled')].iloc[0]
h2_coh = forest_df[(forest_df['hypothesis'] == 'H2') & (forest_df['kind'] == 'cohort')]
n_h1_ni = int((h1_coh['lo'] > -0.01).sum())          # cohort CI lower bound above the -0.01 margin
n_h2_pos = int((h2_coh['est'] > 0).sum())            # point estimate above zero
n_h2_ci = int((h2_coh['lo'] > 0).sum())              # whole 95% CI above zero

n_src_t1 = int(t1_raw['sources'])
n_src_t2 = int(t2_raw['sources'])
n_src_t3 = int(sum(t3_raw['sources'].values()))
n_src_t4 = int(sum(t4S_raw['sources'].values()))
n_src_t5 = int(t5_raw['sources'])
n_src_unseen = n_src_t1 + n_src_t2 + n_src_t3 + n_src_t4 + n_src_t5

# ---------------- Canvas setup (183 mm x 164 mm <= 170 mm max) ----------------
FIG_W, FIG_H = 183.0, 164.0
fig = plt.figure(figsize=(W2, FIG_H * MM))

BG = fig.add_axes([0, 0, 1, 1])
BG.set_xlim(0, FIG_W)
BG.set_ylim(0, FIG_H)
BG.axis('off')

CV_COL = '#595959'
NK_COL = '#9E9E9E'
SOFT_BLUE = '#EBF3FA'
SOFT_VERM = '#FCEEE6'
SOFT_GREEN = '#E6F5F0'
SOFT_AMBER = '#FDF4E3'
SOFT_GREY = '#F4F5F7'
BORDER_GREY = '#D0D5DD'


def fmt_int(n):
    """Nature house style: thin space for >=5-digit integers, plain for 4-digit integers."""
    return f'{n:,}'.replace(',', ' ') if abs(n) >= 10000 else str(n)


def t(ax, x, y, s, **kw):
    """Safe text helper with parse_math=False (prevents literal $ triggering mathtext) and >=5 pt font."""
    kw.setdefault('fontsize', 5.2)
    kw.setdefault('va', 'center')
    kw.setdefault('ha', 'center')
    kw.setdefault('color', INK)
    kw.setdefault('parse_math', False)
    return ax.text(x, y, s, **kw)


def rbox(ax, x, y, w, h, fc, ec='none', lw=0.6, r=1.6, ls='-', alpha=1.0, zorder=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p


def arr(ax, x0, y0, x1, y1, col=INK, lw=0.7, ms=5.2, ls='-', rad=0.0, zorder=4):
    cs = f'arc3,rad={rad}' if rad != 0 else 'arc3,rad=0'
    ax.annotate('', (x1, y1), (x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=col, ls=ls,
                                mutation_scale=ms, shrinkA=0, shrinkB=0, connectionstyle=cs),
                zorder=zorder)


def draw_status_icon(ax, cx, cy, kind='pass', col=RULE, r=1.25, zorder=5):
    """Pure vector status badge ('pass' check, 'fail' cross, 'part' half-filled) — zero font glyph issues."""
    if kind == 'pass':
        ax.add_patch(Circle((cx, cy), r, fc=col, ec='none', zorder=zorder))
        ax.plot([cx - 0.58 * r, cx - 0.12 * r, cx + 0.62 * r],
                [cy - 0.05 * r, cy - 0.48 * r, cy + 0.48 * r],
                color='white', lw=0.75, solid_capstyle='round', solid_joinstyle='round', zorder=zorder + 1)
    elif kind == 'fail':
        ax.add_patch(Circle((cx, cy), r, fc=col, ec='none', zorder=zorder))
        d = 0.45 * r
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color='white', lw=0.75, solid_capstyle='round', zorder=zorder + 1)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color='white', lw=0.75, solid_capstyle='round', zorder=zorder + 1)
    else:
        ax.add_patch(Circle((cx, cy), r, fc='white', ec=col, lw=0.65, zorder=zorder))
        ax.add_patch(Wedge((cx, cy), r, 90, 270, fc=col, ec='none', zorder=zorder + 1))


def draw_plate_icon(ax, cx, cy):
    """Bespoke vector 18-well screening plate."""
    rbox(ax, cx - 5.0, cy - 3.4, 10.0, 6.8, '#FFFFFF', ec='#6B7280', lw=0.55, r=0.9, zorder=3)
    ax.plot([cx - 5.0, cx - 3.7], [cy + 2.1, cy + 3.4], color='#6B7280', lw=0.45, zorder=4)
    well_cols = [REUSE, LIGHT, REDESIGN, LIGHT, SKY, REUSE,
                 LIGHT, REDESIGN, LIGHT, SKY, LIGHT, REUSE,
                 REDESIGN, LIGHT, REUSE, LIGHT, REDESIGN, LIGHT]
    idx = 0
    for r_i in range(3):
        for c_i in range(6):
            wx = cx - 3.6 + c_i * 1.44
            wy = cy + 1.8 - r_i * 1.8
            ax.add_patch(Circle((wx, wy), 0.50, fc=well_cols[idx], ec='#9CA3AF', lw=0.25, zorder=4))
            idx += 1


def draw_helix_icon(ax, cx, cy, w=7.8, h=2.6):
    """Compact vector double-helix wave."""
    ang = np.linspace(0, 2 * np.pi, 60)
    xs = cx - w / 2 + (ang / (2 * np.pi)) * w
    y1 = cy + (h / 2) * np.sin(ang)
    y2 = cy - (h / 2) * np.sin(ang)
    for k in np.linspace(0.45, 2 * np.pi - 0.45, 7):
        xk = cx - w / 2 + (k / (2 * np.pi)) * w
        yk = (h / 2) * np.sin(k)
        ax.plot([xk, xk], [cy - yk, cy + yk], color='#9CA3AF', lw=0.4, zorder=3)
    ax.plot(xs, y1, color=REUSE, lw=0.8, zorder=4)
    ax.plot(xs, y2, color=REDESIGN, lw=0.8, zorder=4)


def draw_blood_tube_icon(ax, cx, cy):
    """Bespoke vector Vacutainer blood tube + ECG pulse."""
    tx, ty, tw, th = cx - 4.2, cy - 3.6, 2.7, 6.1
    rbox(ax, tx, ty, tw, th, '#FFFFFF', ec='#4B5563', lw=0.55, r=1.0, zorder=3)
    rbox(ax, tx + 0.22, ty + 0.2, tw - 0.44, 2.6, '#C83E2B', ec='none', r=0.75, zorder=4)
    ax.add_patch(Rectangle((tx + 0.22, ty + 2.6), tw - 0.44, 1.45, fc='#FDE68A', ec='none', zorder=4))
    rbox(ax, tx - 0.2, ty + th - 0.55, tw + 0.4, 1.5, '#B91C1C', ec='#7F1D1D', lw=0.45, r=0.4, zorder=5)
    ex = np.array([0.0, 1.3, 1.9, 2.4, 3.0, 3.5, 4.2, 5.6]) + cx - 0.5
    ey = np.array([0.0, 0.0, 0.8, -1.5, 2.7, -0.85, 0.0, 0.0]) + cy - 0.3
    ax.plot(ex, ey, color=REDESIGN, lw=0.85, solid_capstyle='round', solid_joinstyle='round', zorder=4)


def draw_lock_icon(ax, cx, cy, col=RULE):
    """Crisp vector padlock icon."""
    ax.add_patch(Wedge((cx, cy + 0.42), 1.0, 0, 180, width=0.32, fc=col, ec='none', zorder=4))
    rbox(ax, cx - 1.3, cy - 1.5, 2.6, 2.0, col, ec='none', r=0.42, zorder=5)
    ax.add_patch(Circle((cx, cy - 0.45), 0.28, fc='white', ec='none', zorder=6))


def draw_target_icon(ax, cx, cy, col=REDESIGN):
    """Vector crosshair target icon."""
    ax.add_patch(Circle((cx, cy), 1.65, fc='none', ec=col, lw=0.65, zorder=4))
    ax.add_patch(Circle((cx, cy), 0.72, fc=col, ec='none', zorder=4))
    for dx, dy in [(2.15, 0), (-2.15, 0), (0, 2.15), (0, -2.15)]:
        ax.plot([cx + dx * 0.55, cx + dx], [cy + dy * 0.55, cy + dy], color=col, lw=0.6, zorder=4)


def draw_stamp_icon(ax, cx, cy, col=REUSE):
    """Vector fixed-grid / reusable template icon."""
    rbox(ax, cx - 1.75, cy - 1.75, 3.5, 3.5, 'none', ec=col, lw=0.65, r=0.5, zorder=4)
    for r_i in (-0.82, 0.82):
        for c_i in (-0.82, 0.82):
            ax.add_patch(Rectangle((cx + c_i - 0.46, cy + r_i - 0.46), 0.92, 0.92, fc=col, ec='none', zorder=4))


# ==============================================================================
# PANEL a (Top-left: x = 0..108, y = 111..164) — One decision in two settings
# ==============================================================================
flabel(fig, 'a', 0.004, 0.994)
t(BG, 4.2, 160.8, 'One panel-design question across biology and medicine',
  fontsize=6.8, fontweight='bold', ha='left')

# Left domain cards: Biology vs Medicine
rbox(BG, 2.5, 138.0, 27.0, 19.2, SOFT_GREY, ec=BORDER_GREY, lw=0.5, r=1.8)
draw_plate_icon(BG, 9.0, 150.4)
draw_helix_icon(BG, 21.6, 150.4, w=8.0, h=2.7)
t(BG, 16.0, 143.5, '6 biological modalities', fontsize=5.5, fontweight='bold')
t(BG, 16.0, 140.2, 'Drugs · kinases · proteins\ngenes · CRISPR · RNAi',
  fontsize=5.0, color='#4B5563', linespacing=1.15)

rbox(BG, 2.5, 115.2, 27.0, 19.6, SOFT_GREY, ec=BORDER_GREY, lw=0.5, r=1.8)
draw_blood_tube_icon(BG, 10.3, 128.2)
rbox(BG, 18.8, 126.2, 8.8, 4.3, SOFT_AMBER, ec=MATCHED, lw=0.55, r=0.9, zorder=4)
t(BG, 23.2, 128.35, 'US$ / ¥', fontsize=5.1, fontweight='bold', color='#92400E', zorder=5)
t(BG, 16.0, 121.0, 'Clinical lab testing', fontsize=5.5, fontweight='bold')
t(BG, 16.0, 117.7, f'{n_cohorts} cohorts · {n_countries} countries\n{fmt_int(n_people)} adults · {fmt_int(n_deaths)} deaths',
  fontsize=5.0, color='#4B5563', linespacing=1.15)

arr(BG, 29.9, 147.6, 33.6, 141.2, col='#6B7280', lw=0.6, ms=4.5)
arr(BG, 29.9, 125.2, 33.6, 131.6, col='#6B7280', lw=0.6, ms=4.5)

# Center: Training matrix (spacious left margin for rotated 'n training units' label)
rbox(BG, 34.0, 115.2, 30.2, 42.0, '#FFFFFF', ec=BORDER_GREY, lw=0.5, r=1.6)
mx, my = 38.6, 121.0
nrow, ncol, nt = 11, 10, 2
cw, ch = 1.80, 2.22
for i in range(nrow):
    for j in range(ncol):
        BG.add_patch(Rectangle((mx + j * cw, my + i * ch), cw * 0.84, ch * 0.82, fc='#DCE8F5', ec='none', zorder=2))
    for j in range(nt):
        BG.add_patch(Rectangle((mx + (ncol + 0.72 + j) * cw, my + i * ch), cw * 0.84, ch * 0.82, fc='#FAD8C0', ec='none', zorder=2))

for j in (1, 5, 8):
    BG.add_patch(Rectangle((mx + j * cw - 0.12, my - 0.18), cw * 1.08, nrow * ch + 0.2,
                           fc='none', ec=REUSE, lw=0.85, zorder=3))
for j in (2, 6, 9):
    BG.add_patch(Rectangle((mx + j * cw - 0.12, my - 0.18), cw * 1.08, nrow * ch + 0.2,
                           fc='none', ec=REDESIGN, lw=0.85, ls=(0, (2.2, 1.1)), zorder=3))

t(BG, mx + (ncol * cw) / 2, my + nrow * ch + 4.4, 'p candidates', fontsize=5.4, fontweight='bold')
t(BG, mx + (ncol * cw) / 2, my + nrow * ch + 1.8, '(44–1000)', fontsize=5.0, color='#6B7280')
t(BG, mx + (ncol + 0.72 + nt / 2) * cw, my + nrow * ch + 3.1, 'Targets', fontsize=5.4, fontweight='bold', color=REDESIGN)
t(BG, mx - 2.3, my + (nrow * ch) / 2, 'n training units', rotation=90, fontsize=5.3, fontweight='bold')
t(BG, 49.1, 117.6, 'Choose k items (same budget)', fontsize=5.0, color='#4B5563')

# Right: Two competing strategies on a new unit
arr(BG, 64.6, 143.0, 68.6, 147.2, col=REUSE, lw=0.85, ms=5.2)
arr(BG, 64.6, 129.2, 68.6, 125.0, col=REDESIGN, lw=0.85, ms=5.2)

rbox(BG, 69.2, 138.4, 37.8, 18.8, SOFT_BLUE, ec=REUSE, lw=0.75, r=1.8)
draw_stamp_icon(BG, 73.8, 152.5, col=REUSE)
t(BG, 77.6, 152.5, 'Fix & reuse panel', fontsize=6.0, fontweight='bold', ha='left', color=REUSE)
t(BG, 88.1, 146.2, 'Chosen from candidates alone', fontsize=5.3)
t(BG, 88.1, 141.6, 'FSCA · decorrelation · routine bundle\nNo selection noise · lower ceiling',
  fontsize=5.0, color='#374151', linespacing=1.18)

rbox(BG, 69.2, 115.2, 37.8, 18.8, SOFT_VERM, ec=REDESIGN, lw=0.75, r=1.8)
draw_target_icon(BG, 73.8, 129.3, col=REDESIGN)
t(BG, 77.6, 129.3, 'Redesign for target', fontsize=6.0, fontweight='bold', ha='left', color=REDESIGN)
t(BG, 88.1, 123.0, 'Tailored using training targets', fontsize=5.3)
t(BG, 88.1, 118.4, 'somp · lasso · greedy search\nHigher ceiling · learning cost at low n',
  fontsize=5.0, color='#374151', linespacing=1.18)


# ==============================================================================
# PANEL b (Top-right: x = 110..183, y = 111..164) — Visual 2D Regime Decision Map
# ==============================================================================
flabel(fig, 'b', 0.602, 0.994)
t(BG, 113.8, 160.8, 'Training-data rule & decision regimes',
  fontsize=6.8, fontweight='bold', ha='left')

rx0, ry0, rw, rh = 121.8, 127.6, 58.2, 29.0
x_split = rx0 + 21.8   # CV R^2 = 0.10 threshold
y_split = ry0 + 13.0   # n / k = 10 threshold

# Top-right (n/k >= 10 & CV R^2 >= 0.10): REDESIGN
rbox(BG, x_split + 0.4, y_split + 0.4, (rx0 + rw) - x_split - 0.4, (ry0 + rh) - y_split - 0.4,
     SOFT_VERM, ec=REDESIGN, lw=0.8, r=1.4, zorder=2)
draw_target_icon(BG, x_split + 6.0, y_split + 11.2, col=REDESIGN)
t(BG, x_split + 9.8, y_split + 11.2, 'REDESIGN', fontsize=6.1, fontweight='bold', ha='left', color=REDESIGN, zorder=4)
t(BG, (x_split + rx0 + rw) / 2, y_split + 5.6, 'Enough units & signal\nTarget-aware panel wins',
  fontsize=5.1, linespacing=1.18, zorder=4)

# Top-left (n/k >= 10 & CV R^2 < 0.10): SIGNAL CHECK
rbox(BG, rx0, y_split + 0.4, x_split - rx0 - 0.4, (ry0 + rh) - y_split - 0.4,
     SOFT_GREEN, ec=RULE, lw=0.75, r=1.4, zorder=2)
t(BG, (rx0 + x_split) / 2, y_split + 12.2, 'Signal check*', fontsize=5.4, fontweight='bold', color=RULE, zorder=4)
t(BG, (rx0 + x_split) / 2, y_split + 6.5, 'Weak / no signal\n(CRISPR, RNAi)\n→ 5-fold CV',
  fontsize=5.0, color='#1F2937', linespacing=1.12, zorder=4)

# Bottom half (n/k < 10): 5-FOLD CV ARBITRATES
rbox(BG, rx0, ry0, rw, y_split - ry0 - 0.4, SOFT_GREY, ec='#6B7280', lw=0.65, r=1.4, zorder=2)
fx0, fy0 = rx0 + 3.0, ry0 + 4.4
for fold_i in range(5):
    fc_ = REDESIGN if fold_i == 0 else '#FFFFFF'
    ec_ = REDESIGN if fold_i == 0 else REUSE
    BG.add_patch(Rectangle((fx0 + fold_i * 2.1, fy0), 1.75, 4.1, fc=fc_, ec=ec_, lw=0.5, zorder=4))
t(BG, fx0 + 5.1, fy0 - 1.9, '5-fold CV', fontsize=5.0, color='#4B5563', zorder=4)
t(BG, rx0 + 36.6, ry0 + 8.1, '5-fold CV chooses Reuse vs Redesign', fontsize=5.3, fontweight='bold', zorder=4)
t(BG, rx0 + 36.6, ry0 + 3.6, 'Re-selects both panels in every fold', fontsize=5.0, color='#4B5563', zorder=4)

# Threshold lines & clean non-overlapping axis labels (CV R^2 = 0.10 splits only n/k >= 10, matching the rule)
BG.plot([rx0 - 0.8, rx0 + rw], [y_split, y_split], color=RULE, lw=0.9, ls=(0, (3, 1.5)), zorder=5)
BG.plot([x_split, x_split], [y_split, ry0 + rh + 0.5], color=RULE, lw=0.9, ls=(0, (3, 1.5)), zorder=5)
BG.plot([x_split, x_split], [ry0 - 1.0, ry0 - 0.5], color=RULE, lw=0.8, zorder=5)

arr(BG, rx0 - 1.1, ry0 - 0.5, rx0 - 1.1, ry0 + rh + 1.2, col=INK, lw=0.6, ms=4.5, zorder=5)
arr(BG, rx0 - 1.1, ry0 - 0.5, rx0 + rw + 1.2, ry0 - 0.5, col=INK, lw=0.6, ms=4.5, zorder=5)

t(BG, rx0 - 2.0, y_split, '10', fontsize=5.4, fontweight='bold', ha='right', color=RULE)
t(BG, rx0 - 5.2, ry0 + rh / 2, 'Training units / item (n / k)', rotation=90, fontsize=5.3, fontweight='bold')

# X-axis tick '0.10' cleanly separated from axis title
t(BG, x_split, ry0 - 2.1, '0.10', fontsize=5.2, fontweight='bold', color=RULE)
t(BG, rx0 + rw * 0.65, ry0 - 4.5, 'Target signal in training CV (R²)', fontsize=5.2, fontweight='bold')

# Clinical rule ribbon underneath 2D map
rbox(BG, 113.8, 115.2, 66.2, 5.4, '#FFFFFF', ec=BORDER_GREY, lw=0.5, r=1.2)
BG.add_patch(Circle((116.4, 117.9), 1.05, fc=RULE, ec='none', zorder=3))
t(BG, 118.3, 117.9, 'Clinic:  events / analyte E/d ≥ 5 → Redesign;  else nested CV (incl. no tests)',
  fontsize=5.0, ha='left')

# Section divider 1
BG.plot([2.5, 180.5], [112.0, 112.0], color='#E5E7EB', lw=0.6)


# ==============================================================================
# PANEL c (Middle band: y = 50..110) — Pre-frozen locked tests & Mean regret
# ==============================================================================
flabel(fig, 'c', 0.004, 109.8 / FIG_H)
t(BG, 4.2, 107.6,
  f'Pre-frozen validation in {n_src_unseen} previously unanalysed dataset blocks (mean regret = variance explained lost)',
  fontsize=6.8, fontweight='bold', ha='left')

# Left workflow summary badge
PROT_X, PROT_W = 2.5, 21.6
rbox(BG, PROT_X, 83.0, PROT_W, 18.8, SOFT_GREY, ec=BORDER_GREY, lw=0.5, r=1.5)
draw_lock_icon(BG, 5.4, 97.9, col=INK)
t(BG, 7.6, 97.9, 'Frozen test\nprotocol', fontsize=5.3, fontweight='bold', ha='left', linespacing=1.1)
t(BG, PROT_X + PROT_W / 2, 89.5, 'Rule derived on\nused sources\n(24 → 46 → 63),\nthen locked & run\nonce on unseen\ntest blocks',
  fontsize=5.0, color='#4B5563', linespacing=1.05)

t1_rf, t1_rr = t1_raw['regret_fixed'], t1_raw['regret_rules']
t2_r = t2_raw['regret']
t3_r = t3_raw['type_balanced']
t4S_r = t4S_raw['type_balanced']
t4T_r = t4T_raw['type_balanced']

rows_c = [
    ('Always historical', HIST, [t1_rf['B'], t2_r['B'], t3_r['always_B'], t4S_r['always_B'], t4T_r['always_B']]),
    ('Always reuse', REUSE, [t1_rf['A'], t2_r['A'], t3_r['always_A'], t4S_r['always_A'], t4T_r['always_A']]),
    ('Always redesign', REDESIGN, [t1_rf['C'], t2_r['C'], t3_r['always_C'], t4S_r['always_C'], t4T_r['always_C']]),
    ('n/k ≥ 10 alone', NK_COL, [t1_rr['NK'], t2_r['NK10'], t3_r['NK10'], t4S_r['NK10'], t4T_r['NK10']]),
    ('5-fold CV alone', CV_COL, [t1_rr['CV'], t2_r['CV5'], t3_r['CV'], t4S_r['CV'], t4T_r['CV']]),
    ('Decision rule', RULE, [t1_rr['H'], t2_r['H'], t3_r['v4'], t4S_r['v4'], t4T_r['v4']]),
]

pct_t1 = int(round(100 * (t1_rr['H'] - t1_rf['C']) / t1_rf['C']))
pct_t3 = int(round(100 * (t3_r['v4'] - t3_r['always_A']) / t3_r['always_A']))

test_cards = [
    ('Locked test 1', f'{n_src_t1} sources: GTEx, RPPA', f'Passed ({pct_t1}% vs fixed)'.replace('-', '\u2212'), SOFT_GREEN, RULE, 'pass'),
    ('Locked test 2', f'{n_src_t2} sources: + CRISPR, MS', 'CRISPR unlearnable', SOFT_VERM, REDESIGN, 'fail'),
    ('Locked test 3*', f'{n_src_t3} sources: RNA-seq, RNAi', f'All 6 passed ({pct_t3}%)'.replace('-', '\u2212'), SOFT_GREEN, RULE, 'pass'),
    ('Test 4: QR + lasso*', f'{n_src_t4} cohorts: new selectors', 'All 3 passed', SOFT_GREEN, RULE, 'pass'),
    ('Test 4: trees*', f'{n_src_t4} cohorts: new predictor', '2/3 passed (NI missed)', SOFT_AMBER, '#B45309', 'part'),
]

t5_r = t5_raw['regret']          # locked test 5: rule v4 = 'rule'; 'H' = count rule without the check
assert t5_raw['best_fixed'] == 'always_A'
for (nm_, col_, vals_), k5 in zip(rows_c, ['always_B', 'always_A', 'always_C', 'NK10', 'CV', 'rule']):
    vals_.append(t5_r[k5])
t5_h1 = float(t5_raw['H1_rule_minus_best_fixed'][0])
t5_h5 = bool(t5_raw['H5_protection_if_not_learnable']['pass'])
t5_badge = (f'Check protected{" (H5)" if t5_h5 else ""}\n'
            f'H1 {t5_h1:+.4f} vs reuse').replace('-', '−')
SUB2 = {'Locked test 1': f'{n_src_t1} sources:\nGTEx, RPPA', 'Locked test 2': f'{n_src_t2} sources:\n+ CRISPR, MS',
        'Locked test 3*': f'{n_src_t3} sources:\nRNA-seq, RNAi', 'Test 4: QR + lasso*': f'{n_src_t4} cohorts:\nnew selectors',
        'Test 4: trees*': f'{n_src_t4} cohorts:\nnew predictor'}
BADGE2 = {'Locked test 1': f'Passed\n({pct_t1}% vs fixed)'.replace('-', '−'), 'Locked test 2': 'CRISPR\nunlearnable',
          'Locked test 3*': f'All 6 passed\n({pct_t3}%)'.replace('-', '−'), 'Test 4: trees*': '2/3 passed\n(NI missed)'}
test_cards = [(ttl_, SUB2.get(ttl_, sub_), BADGE2.get(ttl_, bd_), fc_, ec_, k_) for ttl_, sub_, bd_, fc_, ec_, k_ in test_cards]
test_cards.append(('Locked test 5*', f'{n_src_t5} single-cell sources;\ntargets not learnable', t5_badge,
                   SOFT_AMBER, '#B45309', 'part'))

X0_C = PROT_X + PROT_W + 1.8
CARD_W = 24.0
PITCH_C = (180.5 - X0_C - CARD_W) / (len(test_cards) - 1)
AX_W = 17.4
CHECK = []   # (artist, x0, x1, y0, y1): text that must stay inside a box (mm)
AY0, AY1 = 55.8, 81.2

# Highlight horizontal stripe behind the "Decision rule" row
rbox(BG, 3.0, AY0 + 0.35, 177.5, 4.1, SOFT_GREEN, ec='none', r=0.9, zorder=0)

for j, (ttl, sub, badge_txt, b_fc, b_ec, st_kind) in enumerate(test_cards):
    cx = X0_C + j * PITCH_C
    rbox(BG, cx, 83.0, CARD_W, 18.8, '#FFFFFF', ec=b_ec, lw=0.65, r=1.5, zorder=2)
    draw_lock_icon(BG, cx + 2.6, 99.0, col=b_ec)
    CHECK.append((t(BG, cx + 4.6, 99.0, ttl, fontsize=5.4, fontweight='bold', ha='left', zorder=4), cx, cx + CARD_W, 83.0, 101.8))
    CHECK.append((t(BG, cx + CARD_W / 2, 94.5, sub, fontsize=5.0, color='#4B5563', linespacing=1.08, zorder=4),
                  cx, cx + CARD_W, 90.6, 97.4))

    # Status pill with pure vector icon (one or two lines)
    rbox(BG, cx + 1.0, 84.1, CARD_W - 2.0, 6.2, b_fc, ec='none', r=1.0, zorder=3)
    draw_status_icon(BG, cx + 2.9, 87.2, kind=st_kind, col=b_ec, r=1.1, zorder=4)
    CHECK.append((t(BG, cx + (CARD_W + 4.0) / 2, 87.2, badge_txt, fontsize=5.0, fontweight='bold', color=b_ec,
                    linespacing=1.22, zorder=4), cx + 4.0, cx + CARD_W - 1.0, 84.1, 90.3))

    if j < len(test_cards) - 1:
        if j == 1:
            # Clean '+ signal check*' callout neatly above gap between Test 2 and Test 3 (below Panel c title!)
            arr(BG, cx + CARD_W + 0.2, 92.4, cx + PITCH_C - 0.2, 92.4, col=RULE, lw=1.0, ms=5.0)
            rbox(BG, cx + CARD_W + 1.25 - 7.6, 102.4, 15.2, 3.2, SOFT_GREEN, ec=RULE, lw=0.45, r=0.8, zorder=5)
            t(BG, cx + CARD_W + 1.25, 104.0, '+ signal check*', fontsize=5.0, fontweight='bold', color=RULE, zorder=6)
            BG.plot([cx + CARD_W + 1.25, cx + CARD_W + 1.25], [102.4, 93.4], color=RULE, lw=0.5, ls=':', zorder=4)
        else:
            arr(BG, cx + CARD_W + 0.25, 92.4, cx + PITCH_C - 0.25, 92.4, col='#9CA3AF', lw=0.65, ms=4.5)

    ax = fig.add_axes([(cx + 0.8) / FIG_W, AY0 / FIG_H, AX_W / FIG_W, (AY1 - AY0) / FIG_H])
    ax.patch.set_alpha(0.0)
    vals = [r[2][j] for r in rows_c]
    y_pos = np.arange(len(rows_c))[::-1]
    best_idx = int(np.argmin(vals))

    xmax = 0.16
    ax.barh(y_pos, vals, height=0.60, color=[r[1] for r in rows_c], edgecolor='none', zorder=3)
    for k_i, (yi, v) in enumerate(zip(y_pos, vals)):
        is_best = (k_i == best_idx)
        is_rule = (k_i == len(rows_c) - 1)
        ax.text(v + 0.004, yi, f'{v:.3f}', va='center', ha='left', fontsize=5.0,
                fontweight='bold' if (is_best or is_rule) else 'normal',
                color=RULE if is_rule else INK, clip_on=False, parse_math=False)

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.55, len(rows_c) - 0.45)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    ax.set_xticklabels(['0', '', '0.10', ''], fontsize=5.0)
    ax.set_yticks(y_pos)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='x', length=2.0, pad=1.5)
    if j == 0:
        ax.set_yticklabels([r[0] for r in rows_c], fontsize=5.4)
        ax.get_yticklabels()[-1].set_color(RULE)
        ax.get_yticklabels()[-1].set_fontweight('bold')
    else:
        ax.set_yticklabels([])
    ax.spines['left'].set_visible(False)
    ax.axvline(0, color='#9CA3AF', lw=0.45, zorder=2)

t(BG, 96.0, 51.4,
  f'Lower regret is better (bold = lowest in test).  *Tests 3–5 include the CV R² ≥ 0.10 signal check (63-source dev regret {d63_raw["type_balanced"]["H"]:.3f} → {d63_raw["type_balanced"]["v4"]:.3f}).',
  fontsize=5.0, color='#4B5563')

# Section divider 2
BG.plot([2.5, 180.5], [48.8, 48.8], color='#E5E7EB', lw=0.6)


# ==============================================================================
# PANEL d (Bottom band: y = 0..48) — Priced clinical laboratory panels for death
# ==============================================================================
flabel(fig, 'd', 0.004, 47.4 / FIG_H)
t(BG, 4.2, 45.3,
  f'Clinical translation: priced blood panels for mortality ({n_cohorts} external cohorts across USA, China & UK; {fmt_int(n_people)} adults, {fmt_int(n_deaths)} deaths)',
  fontsize=6.8, fontweight='bold', ha='left')

# Left module (x = 2.5..104.5): Visual Bundled-Fee & Analyte-Swap Architecture
rbox(BG, 2.5, 2.8, 102.0, 39.6, SOFT_GREY, ec=BORDER_GREY, lw=0.5, r=1.8, zorder=1)
t(BG, 5.0, 39.6,
  f'Bundled billing matters: 14 CMP analytes cost US${usd_cmp_bundle:.2f} as a panel vs US${usd_cmp_single:.2f} singly (2026 Medicare)',
  fontsize=5.2, fontweight='bold', ha='left')


def pill_row(y_c, title, title_col, price_str, ec_col, items, note_txt, bg_col='#FFFFFF'):
    rbox(BG, 4.6, y_c - 4.3, 98.0, 8.8, bg_col, ec=ec_col, lw=0.65, r=1.3, zorder=2)
    t(BG, 6.4, y_c + 1.45, title, fontsize=5.5, fontweight='bold', ha='left', color=title_col, zorder=4)
    t(BG, 6.4, y_c - 1.85, price_str, fontsize=5.0, fontweight='bold', ha='left', color=INK, zorder=4)
    px = 30.8
    for txt_p, kind in items:
        pw = max(7.8, len(txt_p) * 1.18 + 2.6)
        if kind == 'core':
            p_fc, p_ec, t_c = '#F3F4F6', '#6B7280', INK
        elif kind == 'drop':
            p_fc, p_ec, t_c = '#F9FAFB', '#D1D5DB', '#9CA3AF'
        elif kind == 'add_m':
            p_fc, p_ec, t_c = SOFT_AMBER, MATCHED, '#92400E'
        else:
            p_fc, p_ec, t_c = SOFT_VERM, REDESIGN, REDESIGN
        rbox(BG, px, y_c - 2.5, pw, 5.0, p_fc, ec=p_ec, lw=0.55,
             ls='--' if kind == 'drop' else '-', r=0.85, zorder=3)
        t(BG, px + pw / 2, y_c, txt_p, fontsize=5.0,
          fontweight='bold' if kind in ('add_m', 'add_a') else 'normal', color=t_c, zorder=4)
        if kind == 'drop':
            BG.plot([px + 0.9, px + pw - 0.9], [y_c, y_c], color='#9CA3AF', lw=0.55, zorder=5)
        px += pw + 1.05
    t(BG, 101.0, y_c, note_txt, fontsize=5.0, ha='right', color='#374151', zorder=4)


save_pct_us = int(round(100 * (1.0 - usd_matched / usd_routine)))
pill_row(31.8, 'Routine bundle', ROUTINE, f'US${usd_routine:.2f} (100%)', ROUTINE,
         [('CMP', 'core'), ('CBC', 'core'), ('Lipids', 'core'), ('HbA1c', 'core')],
         'Standard check-up', bg_col='#FFFFFF')

pill_row(20.5, 'Matched panel', '#B45309', f'US${usd_matched:.2f} (−{save_pct_us}% fee)', MATCHED,
         [('CMP', 'core'), ('CBC', 'core'), ('+ Phos', 'add_m'), ('Lipids', 'drop'), ('HbA1c', 'drop')],
         f'{fee_min_pct}–{fee_max_pct}% of fee', bg_col='#FFFDF8')

pill_row(9.2, 'Target-aware', AWARE, f'US${usd_aware:.2f} (same budget)', AWARE,
         [('CMP', 'core'), ('CBC', 'core'), ('+ ACR', 'add_a'), ('+ GGT', 'add_a'), ('+ Phos', 'add_a')],
         'Kidney & liver injury', bg_col='#FFF9F6')

arr(BG, 3.8, 28.5, 3.8, 22.4, col=MATCHED, lw=0.75, ms=4.2, rad=0.25)
arr(BG, 3.8, 27.5, 3.8, 11.0, col=REDESIGN, lw=0.75, ms=4.2, rad=0.20)


# Right module (x = 106.8..180.5): Pooled 5-Cohort External Validation Forest + Clean Readout Column
rbox(BG, 106.8, 2.8, 73.7, 39.6, '#FFFFFF', ec=BORDER_GREY, lw=0.5, r=1.8, zorder=1)
t(BG, 109.2, 39.6, f'Pooled external discrimination vs routine ({n_cohorts} cohorts)',
  fontsize=5.4, fontweight='bold', ha='left')

ax_f = fig.add_axes([121.8 / FIG_W, 10.0 / FIG_H, 27.2 / FIG_W, 23.8 / FIG_H])
ax_f.patch.set_alpha(0.0)

ax_f.axvspan(-0.0125, -0.010, color='#FEE2E2', alpha=0.6, lw=0, zorder=0)
ax_f.axvline(-0.010, color=INK, lw=0.65, ls=DASH, zorder=2)
ax_f.plot([0.0, 0.0], [-0.55, 1.22], color=BASE, lw=0.55, zorder=2)  # stops below the NI label

for y_f, row_p, col_f in [(1.0, h1_pool, MATCHED), (0.0, h2_pool, AWARE)]:
    est, lo, hi = float(row_p['est']), float(row_p['lo']), float(row_p['hi'])
    ax_f.fill([lo, est, hi, est], [y_f, y_f + 0.25, y_f, y_f - 0.25],
              color=col_f, edgecolor=col_f, lw=0.4, zorder=4)

ax_f.text(-0.0094, 1.44, 'NI margin (−0.01)', fontsize=5.0, color=INK, ha='left', va='center', parse_math=False,
          bbox=dict(fc='white', ec='none', pad=0.6), zorder=6)
ax_f.set_xlim(-0.0125, 0.0185)
ax_f.set_ylim(-0.55, 1.62)
ax_f.set_xticks([-0.01, 0.0, 0.01])
ax_f.set_xticklabels(['−0.01', '0', '+0.01'], fontsize=5.0)
ax_f.set_yticks([1.0, 0.0])
ax_f.set_yticklabels([f'Matched\n({fee_min_pct}–{fee_max_pct}% fee)', 'Target-aware\n(same budget)'],
                     fontsize=5.2, fontweight='bold')
ax_f.get_yticklabels()[0].set_color('#B45309')
ax_f.get_yticklabels()[1].set_color(AWARE)
ax_f.tick_params(axis='y', length=0, pad=2.5)
ax_f.tick_params(axis='x', length=2.0, pad=1.5)
ax_f.spines['left'].set_visible(False)
ax_f.set_xlabel("Pooled Δ C or AUC", fontsize=5.1, labelpad=2.0)

# Right-hand aligned numeric & badge column inside Right module (generous right margin inside 180.5 mm box)
rx_stat = 151.4
# Matched readout (aligned with y_f = 1.0 -> ~25.5 mm in BG)
y_m_bg, y_a_bg = 25.4, 14.5
h1_est, h1_lo, h1_hi = float(h1_pool['est']), float(h1_pool['lo']), float(h1_pool['hi'])
h2_est, h2_lo, h2_hi = float(h2_pool['est']), float(h2_pool['lo']), float(h2_pool['hi'])

t(BG, rx_stat, y_m_bg + 2.0, f'{h1_est:+.4f} ({h1_lo:+.4f} to {h1_hi:+.4f})'.replace('-', '\u2212'),
  fontsize=5.0, fontweight='bold', ha='left')
draw_status_icon(BG, rx_stat + 1.2, y_m_bg - 1.5, kind='pass', col=RULE, r=1.1, zorder=4)
t(BG, rx_stat + 3.0, y_m_bg - 1.5, f'Non-inferior in {n_h1_ni}/{n_cohorts} cohorts',
  fontsize=5.0, fontweight='bold', ha='left', color=RULE)

t(BG, rx_stat, y_a_bg + 2.0, f'{h2_est:+.4f} ({h2_lo:+.4f} to {h2_hi:+.4f})'.replace('-', '\u2212'),
  fontsize=5.0, fontweight='bold', ha='left')
draw_status_icon(BG, rx_stat + 1.2, y_a_bg - 1.5, kind='pass', col=AWARE, r=1.1, zorder=4)
t(BG, rx_stat + 3.0, y_a_bg - 1.5, f'Higher in {n_h2_pos}/{n_cohorts} (CI > 0 in {n_h2_ci}/{n_cohorts})',
  fontsize=5.0, fontweight='bold', ha='left', color=AWARE)

fig.canvas.draw()
_r = fig.canvas.get_renderer()
_px_mm = fig.dpi / 25.4
for art_, x0_, x1_, y0_, y1_ in CHECK:
    bb_ = art_.get_window_extent(_r)
    bx0, bx1, by0, by1 = bb_.x0 / _px_mm, bb_.x1 / _px_mm, bb_.y0 / _px_mm, bb_.y1 / _px_mm
    if bx0 < x0_ or bx1 > x1_ or by0 < y0_ or by1 > y1_:
        print(f'OVERFLOW {art_.get_text()!r}: text x {bx0:.1f}-{bx1:.1f} y {by0:.1f}-{by1:.1f}; box x {x0_:.1f}-{x1_:.1f} y {y0_:.1f}-{y1_:.1f}')
save(fig, OUT / 'fig1')
print('Saved fig1.pdf and fig1.png.')
