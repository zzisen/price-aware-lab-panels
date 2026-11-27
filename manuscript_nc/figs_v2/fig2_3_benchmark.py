#!/usr/bin/env python3
"""Figures 2 and 3 of NC v0.8 from frozen result files (Tasks 3, 3B, 15, 16, 17). Output: figs_v2/out/fig2.*, fig3.*
v0.8: Fig. 2a neutral grey ramp + target-aware/reusable category bars, NHANES blocks named by cycle/sample
(M = task13 NHANES_I, 665 women aged 20-49, 2015-16; A1 = NH_A1_I, other 2015-16 adults; A2 = NH_A2_H, 2013-14; A3 = NH_A3_G, 2011-12;
see task14_m2/TASK14_M2_REPORT.md), n/k >= 10 shading in Fig. 2c; Fig. 3c shows all training sizes for both GDSC releases on a
symmetric-log y axis; Fig. 3a trees drawn as open squares."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
AWARE = ['R5', 'lasso', 'somp', 'gbt_imp', 'mrmr', 'pls_vip', 'rf', 'univ']
REUS = ['fsca', 'qr', 'medoid', 'pca_loading', 'leverage', 'high_var', 'decorr', 'random']
NAME = {'R5': 'Our search', 'lasso': 'Multitask lasso', 'somp': 'somp', 'gbt_imp': 'Boosted-tree importance', 'mrmr': 'mRMR',
        'pls_vip': 'PLS-VIP', 'rf': 'Random-forest importance', 'univ': 'Univariate correlation', 'fsca': 'FSCA', 'qr': 'Pivoted QR',
        'medoid': 'Medoids', 'pca_loading': 'PC loadings', 'leverage': 'Leverage scores', 'high_var': 'Highest variance',
        'decorr': 'Decorrelation', 'random': 'Random'}
SRC = {'Beat AML': 'Beat AML', 'E1': 'FPMTB', 'E2': 'PACE', 'PKIS2': 'PKIS2', 'E3': 'TCGA RPPA', 'E4': 'GTEx blood', 'M': 'NHANES women',
       'A1': 'NHANES 15–16', 'A2': 'NHANES 13–14', 'A3': 'NHANES 11–12'}

# ================= Figure 2 =================
g = pd.read_csv(W / 'task17_gates/summary/g1_ev.csv'); g = g[g.n.astype(str) == '100']
fig = plt.figure(figsize=(W2, 118 * MM), layout='constrained')
gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], height_ratios=[1, 1])
# a: heatmap, k = 5, ridge
ax = fig.add_subplot(gs[:, 0])
s = g[(g.k == 5) & (g.predictor == 'ridge')].pivot(index='method', columns='source', values='ev')
cols = [c for c in SRC if c in s.columns]
s = s.loc[AWARE + REUS, cols]
cm = LinearSegmentedColormap.from_list('g', ['#FFFFFF', '#E4E1DC', '#A8A29A', '#5E5953', '#2E2B28'])   # neutral warm-grey ramp (no clash with reuse blue)
im = ax.imshow(s.to_numpy(), cmap=cm, vmin=-0.1, vmax=0.6, aspect='auto')
ax.set_xticks(range(len(cols))); ax.set_xticklabels([SRC[c] for c in cols], rotation=55, ha='right')
ax.set_yticks(range(len(s))); ax.set_yticklabels([NAME[m] for m in s.index])
for i, m in enumerate(s.index):
    ax.get_yticklabels()[i].set_color(REDESIGN if m in AWARE else REUSE)
    for j in range(len(cols)):
        v = s.iloc[i, j]; ax.text(j, i, f'{v:.2f}'.replace('-0.', '−.').replace('0.', '.'), ha='center', va='center', fontsize=5,
                                  color='white' if v > 0.38 else INK)
ax.axhline(len(AWARE) - 0.5, color=INK, lw=0.6)
ax.tick_params(length=0); ax.tick_params(axis='y', pad=6); [sp.set_visible(False) for sp in ax.spines.values()]
from matplotlib.patches import Rectangle  # noqa: E402
for y0_, y1_, col_ in [(-0.45, len(AWARE) - 0.55, REDESIGN), (len(AWARE) - 0.45, len(s) - 0.55, REUSE)]:
    ax.add_patch(Rectangle((-0.78, y0_), 0.16, y1_ - y0_, fc=col_, ec='none', clip_on=False))
ax.set_xlim(-0.5, len(cols) - 0.5)
mean = s.mean(1)
for i, v in enumerate(mean):
    ax.text(len(cols) - 0.35, i, f'{v:.2f}', ha='left', va='center', fontsize=5, fontweight='bold' if i in (0, 8) else None)
ax.text(len(cols) - 0.35, -0.9, 'Mean', ha='left', va='center', fontsize=5)
ax.text(-0.5, -1.3, 'Explained variance of held-out targets (n = 100, k = 5, ridge)', fontsize=5.5, ha='left')
cb = fig.colorbar(im, ax=ax, location='bottom', shrink=0.55, aspect=25, pad=0.01); cb.outline.set_linewidth(0.4); cb.ax.tick_params(length=1.5)
label(ax, 'a', dx=-0.62, dy=1.0)
# b: lead of our search over the best reusable panel, every source and setting
ax = fig.add_subplot(gs[0, 1])
sets = [(5, 'ridge'), (10, 'ridge'), (5, 'gbt'), (10, 'gbt')]
for i, (k, p) in enumerate(sets):
    q = g[(g.k == k) & (g.predictor == p)].pivot(index='source', columns='method', values='ev')
    lead = q['R5'] - q[REUS].max(1)
    x = i + np.linspace(-0.22, 0.22, len(lead))
    ax.scatter(x, lead.sort_values(), s=7, color=REDESIGN, edgecolor='white', lw=0.3, zorder=3)
    ax.plot([i - 0.28, i + 0.28], [lead.median()] * 2, color=INK, lw=0.8)
ax.axhline(0, color=BASE, lw=0.5); ax.axhline(0.02, color=BASE, lw=0.5, ls=DASH)
ax.set_xticks(range(4)); ax.set_xticklabels(['k = 5\nridge', 'k = 10\nridge', 'k = 5\ntrees', 'k = 10\ntrees'])
ax.set_ylabel('Lead over best reusable panel\n(explained variance)'); ax.set_ylim(-0.01, None)
ax.text(3.4, 0.022, '0.02', fontsize=5, color=BASE, va='bottom', ha='right')
ax.set_title('40 of 40 block–settings ahead', fontsize=6); label(ax, 'b')
# c: 24 further sources by n/k
ax = fig.add_subplot(gs[0, 2])
c3 = pd.read_csv(W / 'task17_gates/summary/g3_test_cells.csv'); c3 = c3[c3.predictor == 'ridge']; c3['nk'] = c3.n / c3.k
c3['type'] = np.where(c3.source.str.startswith('G_'), 'GTEx tissues', 'TCGA RPPA cancers')
for t, col, mk in [('GTEx tissues', SKY, 'o'), ('TCGA RPPA cancers', PURPLE, 's')]:
    q = c3[c3.type == t]; jit = np.exp(np.random.default_rng(1).normal(0, 0.06, len(q)))
    ax.scatter(q.nk * jit, q.lead_strong, s=4, color=col, marker=mk, alpha=0.75, lw=0, label=t)
med = c3.groupby('nk').lead_strong.median()
ax.plot(med.index, med.values, color=INK, lw=0.9, marker='_', ms=5)
ax.axhline(0, color=BASE, lw=0.5); ax.axvline(10, color=BASE, lw=0.5, ls=DASH)
xl_ = (c3.nk.min() / 1.25, c3.nk.max() * 1.25); ax.axvspan(10, xl_[1], color=LIGHT, alpha=0.45, lw=0, zorder=0)
ax.set_xscale('log'); ax.set_xlim(*xl_); ax.set_xticks([2.5, 5, 10, 20, 40]); ax.set_xticklabels(['2.5', '5', '10', '20', '40'])
ax.set_xlabel('Training units per measurement (n/k)'); ax.set_ylabel('Lead over best of QR, FSCA, medoids')
ax.set_ylim(top=0.145); ax.legend(loc='upper left', handletextpad=0.2, markerscale=1.5, ncol=2, columnspacing=0.8, bbox_to_anchor=(-0.02, 1.03)); ax.set_title('24 further sources (ridge)', fontsize=6); label(ax, 'c')
# d: target switching
ax = fig.add_subplot(gs[1, 1:])
r = pd.read_csv(W / 'task16_route_b/summary_fixed/r_summary.csv'); r = r[r.n == 100]
r['rebuilt_minus_old'] = r.dedicated_on_B - r.wrong_panel_on_B; r['old_minus_decorr'] = r.wrong_panel_on_B - r.decorr_on_B
m = r.groupby(['predictor', 'k'])[['rebuilt_minus_old', 'old_minus_decorr']].mean().reset_index()
for p, ls, mk in [('ridge', '-', 'o'), ('gbt', (0, (2, 1.5)), 's')]:
    q = m[m.predictor == p]
    ax.plot(q.k, q.rebuilt_minus_old, color=REDESIGN, ls=ls, marker=mk, ms=3, label=f'Rebuilt for new targets − old panel ({"ridge" if p == "ridge" else "trees"})')
    ax.plot(q.k, q.old_minus_decorr, color=REUSE, ls=ls, marker=mk, ms=3, label=f'Old panel − decorrelation ({"ridge" if p == "ridge" else "trees"})')
ax.axhline(0, color=BASE, lw=0.5)
ax.set_xticks([5, 10, 20]); ax.set_xlabel('Panel size k (n = 100; mean of six sources)'); ax.set_ylabel('Difference in explained variance')
ax.legend(loc='upper right', ncol=1, bbox_to_anchor=(1.0, 1.02)); ax.set_ylim(-0.03, 0.175); ax.set_title('When every target is replaced', fontsize=6); label(ax, 'd', dx=-0.08)
save(fig, OUT / 'fig2')

# ================= Figure 3 =================
fig = plt.figure(figsize=(W2, 112 * MM), layout='constrained')
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
c = pd.read_csv(W / 'task15_route_b/summary/cells.csv'); c = c[c.source.isin(['Beat AML', 'NHANES', 'PKIS2'])]
for p, col, mk in [('ridge', REDESIGN, 'o'), ('gbt', INK, 's')]:
    q = c[c.predictor == p]; jit = np.exp(np.random.default_rng(2).normal(0, 0.05, len(q)))
    if p == 'ridge':
        ax.scatter(q.n_per_k * jit, q.gain_best, s=6, color=col, marker=mk, lw=0, alpha=0.85, label='ridge')
    else:
        ax.scatter(q.n_per_k * jit, q.gain_best, s=7, facecolor='none', edgecolor=col, marker=mk, lw=0.5, alpha=0.9, label='trees')
ax.axvspan(10, c.n_per_k.max() * 1.3, color=LIGHT, alpha=0.45, lw=0, zorder=0)
ax.axhline(0, color=BASE, lw=0.5); ax.set_xscale('log'); ax.set_xticks([1, 2.5, 5, 10, 20, 50, 100]); ax.set_xticklabels(['1', '2.5', '5', '10', '20', '50', '100']); ax.minorticks_off()
ax.set_xlabel('Training units per measurement (n/k)'); ax.set_ylabel('Target-aware − reusable\n(explained variance)')
ax.legend(loc='upper left', handletextpad=0.2); ax.set_title('Beat AML, NHANES, PKIS2 (120 combinations)', fontsize=6); label(ax, 'a')
ax = fig.add_subplot(gs[0, 1])
v = pd.read_csv(W / 'task16_route_b/summary_fixed/v_cells.csv')
VS = {'E1': ('FPMTB', REDESIGN), 'E2': ('PACE', ORANGE), 'E3': ('TCGA RPPA', PURPLE), 'E4': ('GTEx blood', SKY)}
for s_, (nm, col) in VS.items():
    q = v[v.source == s_]; jit = np.exp(np.random.default_rng(3).normal(0, 0.05, len(q)))
    ax.scatter(q.n_per_k * jit, q.lead, s=5, color=col, lw=0, alpha=0.85, label=nm)
ax.axvspan(10, v.n_per_k.max() * 1.3, color=LIGHT, alpha=0.45, lw=0, zorder=0)
ax.axhline(0, color=BASE, lw=0.5); ax.set_xscale('log'); ax.set_xticks([1, 2.5, 5, 10, 20, 50, 100]); ax.set_xticklabels(['1', '2.5', '5', '10', '20', '50', '100']); ax.minorticks_off()
hi = v[v.n_per_k >= 10]
ax.text(0.97, 0.04, f'n/k ≥ 10: {int((hi.lead > 0).sum())}/{len(hi)} cells ahead', transform=ax.transAxes, ha='right', fontsize=5.5)
ax.set_xlabel('Training units per measurement (n/k)'); ax.set_ylabel('Lead over best reusable panel')
ax.legend(loc='upper left', handletextpad=0.2, ncol=2, columnspacing=0.6); ax.set_title('Frozen test in four new sources', fontsize=6); label(ax, 'b')
# d: simulation, crossover n/k against target signal (Task 41; same content as Supplementary Fig. threshold, panel a)
import json  # noqa: E402
ax = fig.add_subplot(gs[1, 1])
T41 = W / 'task41_threshold'
sim = json.load(open(T41 / 'sim_summary.json')); fit = json.load(open(T41 / 'fit.json'))['sim_fit']
PC = {71: SKY, 200: REUSE, 1000: INK}
for p in (71, 200, 1000):
    pts = [(float(k_.split('_s')[1]), v_) for k_, v_ in sim.items() if k_.startswith(f'p{p}_') and v_]
    s_, v_ = zip(*sorted(pts))
    ax.plot(s_, v_, 'o', color=PC[p], ms=3, label=f'p = {p}')
    gg = np.linspace(0.08, 0.8, 50)
    ax.plot(gg, np.exp(fit['a'] + fit['b_lnlnp'] * np.log(np.log(p)) + fit['c_lnsignal'] * np.log(gg)), color=PC[p], lw=0.6)
ax.axhline(10, color=RULE, ls=DASH, lw=0.7); ax.text(0.79, 10.8, 'rule: 10', color=RULE, ha='right', fontsize=5)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([0.1, 0.25, 0.5, 0.75]); ax.set_xticklabels(['0.1', '0.25', '0.5', '0.75'])
ax.set_yticks([2, 5, 10, 20, 50]); ax.set_yticklabels(['2', '5', '10', '20', '50']); ax.minorticks_off()
ax.set_xlabel('Target signal (population R²)'); ax.set_ylabel('Crossover n/k (redesign = reuse)')
ax.legend(loc='upper right', title='Pool size', title_fontsize=5, fontsize=5)
ax.text(0.03, 0.04, f'fit: n/k* $\\propto$ (ln p)$^{{{fit["b_lnlnp"]:.1f}}}$ / signal$^{{{-fit["c_lnsignal"]:.1f}}}$; R² = {fit["R2"]:.2f}\nsignal 0.02: no crossover', transform=ax.transAxes, fontsize=5)
ax.set_title('Simulation: where redesign overtakes reuse', fontsize=6); label(ax, 'd')
# c: GDSC, the crossover moves with sample size (bottom left)
ax = fig.add_subplot(gs[1, 0])
for f, nm, col in [('task3_learning_curve/learning_curve_contrasts.csv', 'GDSC1', REDESIGN), ('task3b_external/gdsc2_contrasts.csv', 'GDSC2', ORANGE)]:
    q = pd.read_csv(W / f); q = q[(q.selector == 'inner_cv') & (q.contrast == 'Family − Shared') & q.analysis.str.contains('repeated_random')]
    q = q[q.n_label != 'all'].copy(); q['n'] = q.n_train_mean.astype(float)
    off = 1.04 if nm == 'GDSC2' else 1.0
    ax.errorbar(q.n * off, -q['mean'], yerr=[q['rep_97.5'] - q['mean'], q['mean'] - q['rep_2.5']], fmt='o-', ms=3, color=col, lw=0.8, elinewidth=0.6, label=nm)
ax.axhline(0, color=BASE, lw=0.5); ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks([25, 50, 100, 200]); ax.set_xticklabels(['25', '50', '100', '200']); ax.set_xlim(19, 250)
ax.set_yscale('symlog', linthresh=0.08, linscale=2.0); ax.set_ylim(-6, 0.14)
yt_ = [-5, -1, -0.2, -0.05, 0, 0.05, 0.1];ax.set_yticks(yt_); ax.set_yticklabels([('−' if v < 0 else '') + f'{abs(v):g}' for v in yt_]); ax.yaxis.set_minor_locator(plt.NullLocator())
ax.set_xlabel('Training cell lines'); ax.set_ylabel('Error reduction: family-specific\nminus one shared set')
ax.text(0.03, 0.92, 'Specific sets win', transform=ax.transAxes, fontsize=5, color=INK)
ax.text(0.97, 0.05, 'Shared set wins', transform=ax.transAxes, fontsize=5, color=INK, ha='right')
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9), handletextpad=0.3); ax.set_title('The crossover moves with sample size', fontsize=6); label(ax, 'c')
save(fig, OUT / 'fig3')
print('ok')
