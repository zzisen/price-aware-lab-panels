#!/usr/bin/env python3
"""Figure 6 of NC v0.3 (priced laboratory panels for death) from frozen outputs: manuscript_ldh/figures_v2/data (Tasks 21, 24), Task 32d
separate pooling (TASK32_REPORT.md numbers), Task 35 (t35_results.json). Output: figs_v2/out/fig6.*
v0.8: b, "(n; deaths)" after each cohort name (F1C_cohort_cards.csv); d, the two duplicate CoAI callouts replaced by one
annotation, and one pair of endpoints (beam search, matched budget) labelled "single-test pricing" -> "as billed"."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
D = W / 'manuscript_ldh/figures_v2/data'; OUT = HERE / 'out'
fig = plt.figure(figsize=(W2, 128 * MM), layout='constrained')
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])

# a: fee and C in NHANES III
ax = fig.add_subplot(gs[0, 0])
f = pd.read_csv(D / 'F2A_cost_C_NHANES3.csv')
sty = {'standard bundle': (BASE, 'o'), 'routine': (INK, 'D'), 'established score': (BASE, '^'), 'matched': (MATCHED, 'o'),
       'target-aware': (REDESIGN, 'o'), 'lasso': (PURPLE, 's'), 'reference line': (BASE, None)}
ref = f[f.category == 'reference line'].C.iloc[0]
ax.axhline(ref, color=BASE, lw=0.5, ls=DASH); ax.text(108, ref + 0.0012, 'Age and sex only', fontsize=5, ha='right', color=BASE)
for _, r in f[f.category != 'reference line'].iterrows():
    c, m = sty[r.category]
    ax.scatter(r.usd, r.C, s=14 if r.category in ('matched', 'target-aware', 'routine') else 8, color=c, marker=m, edgecolor='white', lw=0.3, zorder=3)
lab = {'Routine bundle': (2.5, -0.0032), 'Matched panel': (-26, 0.0022), 'Target-aware panel': (-14, 0.0025), 'CMP+CBC': (-19, -0.0035), 'All 34 analytes': (-14, -0.0042)}
for nm, (dx, dy) in lab.items():
    r = f[f.panel == nm].iloc[0]; ax.text(r.usd + dx, r.C + dy, nm, fontsize=5, color={'Matched panel': MATCHED, 'Target-aware panel': AWARE}.get(nm, INK))
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=BASE, marker='o', ls='', ms=3, label='Standard bundles'),
                   Line2D([], [], color=BASE, marker='^', ls='', ms=3, label='Published scores, refitted'),
                   Line2D([], [], color=PURPLE, marker='s', ls='', ms=3, label='Price-weighted lasso')], loc='lower right', bbox_to_anchor=(1.03, 0.1), fontsize=5, handletextpad=0.3, borderaxespad=0.2)
rt, mt, aw = [f[f.panel == p].iloc[0] for p in ['Routine bundle', 'Matched panel', 'Target-aware panel']]
ax.annotate('', (mt.usd + 1, mt.C), (rt.usd - 1, rt.C), arrowprops=dict(arrowstyle='->', lw=0.6, color=MATCHED))
ax.annotate('', (aw.usd, aw.C - 0.0015), (rt.usd, rt.C + 0.0015), arrowprops=dict(arrowstyle='->', lw=0.6, color=REDESIGN))
ax.set_xlabel('Laboratory fee per person (US$, 2026 Medicare)'); ax.set_ylabel("Harrell's C, 9-year death")
ax.set_title('NHANES III (893 deaths)', fontsize=6); ax.set_xlim(-3, 110); label(ax, 'a', dx=-0.2)


def forest(ax, hyp, sep, margin=None, title='', col=AWARE):
    q = pd.read_csv(D / 'F2BC_forest.csv'); q = q[q.hypothesis == hyp]
    coh = q[q.kind == 'cohort']; pool = q[q.kind == 'pooled'].iloc[0]; sup = q[q.kind == 'supplementary'].iloc[0]
    rows = list(coh.itertuples())
    cards = pd.read_csv(D / 'F1C_cohort_cards.csv').set_index('cohort')
    nd = (lambda c: f"\n({cards.loc[c, 'n']}; {cards.loc[c, 'deaths']})") if hyp == 'H1' else (lambda c: '')
    y = 0; ys, names = [], []
    share = 'matched_price_share' if hyp == 'H1' else 'aware_price_share'
    for r in rows:
        ax.plot([r.lo, r.hi], [y, y], color=col, lw=0.8); ax.scatter(r.est, y, s=10, color=col, marker='o' if r.metric == 'C' else 's', zorder=3)
        ax.text(1.02, y, f'{getattr(r, share) * 100:.0f}%', transform=ax.get_yaxis_transform(), fontsize=5, va='center') if hyp == 'H1' else None
        ys.append(y); names.append(r.cohort.replace(' (65–79 y)', '') + nd(r.cohort)); y -= 1
    y -= 0.3
    ax.plot([sup.lo, sup.hi], [y, y], color=BASE, lw=0.8); ax.scatter(sup.est, y, s=10, facecolor='white', edgecolor=BASE, zorder=3)
    ys.append(y); names.append('NHANES 2015–16 (suppl.)' + nd('NHANES 2015–16')); y -= 1.3
    for nm, (e, lo, hi) in [('Pooled, all five', (pool.est, pool.lo, pool.hi))] + sep:
        ax.fill([lo, e, hi, e], [y, y + 0.28, y, y - 0.28], color=INK if nm.startswith('Pooled, all') else BASE, lw=0)
        ys.append(y); names.append(nm); y -= 1
    ax.axvline(0, color=BASE, lw=0.5)
    if margin is not None:
        ax.axvline(margin, color=INK, lw=0.6, ls=DASH); ax.text(margin + 0.0003, ys[-1] - 0.05, 'Margin', color=INK, fontsize=5, ha='left', va='center')
    ax.set_yticks(ys); ax.set_yticklabels(names); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    if hyp == 'H1':
        ax.text(1.02, 0.75, 'Fee', transform=ax.get_yaxis_transform(), fontsize=5, va='bottom', fontweight='bold')
    ax.set_title(title, fontsize=6)


ax = fig.add_subplot(gs[0, 1])
forest(ax, 'H1', [('Pooled, C cohorts', (0.0002, -0.0012, 0.0015)), ('Pooled, AUC cohorts', (-0.0004, -0.0040, 0.0033))], margin=-0.01,
       title='Matched panel − routine bundle', col=MATCHED)
ax.set_xlabel('Difference in C or AUC'); ax.set_xlim(-0.012, 0.009); label(ax, 'b', dx=-0.62)
ax = fig.add_subplot(gs[0, 2])
forest(ax, 'H2', [('Pooled, C cohorts', (0.0144, 0.0109, 0.0179)), ('Pooled, AUC cohorts', (0.0053, 0.0021, 0.0085))], title='Target-aware panel − routine bundle')
ax.set_xlabel('Difference in C or AUC'); label(ax, 'c', dx=-0.62)

# d: fee representation x method (Task 35), NHANES III
ax = fig.add_subplot(gs[1, :2])
t = json.load(open(W / 'task35_fee_ablation/t35_results.json')); P, C = t['panels'], t['cohorts']['NH3']['C']
meth = [('beam', 'Beam search', REDESIGN), ('lasso', 'Price-weighted lasso', PURPLE), ('CoAI', 'CoAI', SKY)]
for key, nm, col in meth:
    for b, mk in [('matched', 'o'), ('aware', 's')]:
        s_, u = P[f'{key} single, {b}'], P[f'{key} bundle, {b}']
        xs, ys_ = (s_['billed'], C[f'{key} single, {b}']), (u['billed'], C[f'{key} bundle, {b}'])
        ax.scatter(*xs, s=16, facecolor='white', edgecolor=col, marker=mk, lw=0.8, zorder=3)
        ax.scatter(*ys_, s=16, color=col, marker=mk, edgecolor='white', lw=0.3, zorder=4)
        if abs(xs[0] - ys_[0]) + abs(xs[1] - ys_[1]) > 1e-6:
            ax.annotate('', ys_, xs, arrowprops=dict(arrowstyle='->', lw=0.6, color=col, shrinkA=3, shrinkB=3))
for nm, key in [('Routine bundle', 'routine'), ('CMP+CBC', 'CMP+CBC')]:
    ax.scatter(P[key]['billed'], C[key], s=14, color=INK if key == 'routine' else BASE, marker='D', zorder=3)
    ax.text(P[key]['billed'] + 0.6, C[key] - 0.0012, nm, fontsize=5)
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color=c, marker='o', ls='', ms=3.5, label=n) for _, n, c in meth]
h += [Line2D([], [], color=INK, marker='o', ls='', mfc='white', ms=3.5, label='Matched budget, single-test prices'),
      Line2D([], [], color=INK, marker='o', ls='', ms=3.5, label='Matched budget, fees as billed'),
      Line2D([], [], color=INK, marker='s', ls='', mfc='white', ms=3.5, label='Routine budget, single-test prices'),
      Line2D([], [], color=INK, marker='s', ls='', ms=3.5, label='Routine budget, fees as billed')]
same = [b for b in ('matched', 'aware') if abs(P[f'beam bundle, {b}']['billed'] - P[f'CoAI bundle, {b}']['billed']) < 0.01
        and abs(C[f'beam bundle, {b}'] - C[f'CoAI bundle, {b}']) < 1e-6]
if same:
    ax.text(0.015, 0.97, 'With fees as billed, CoAI selects the same panel as beam search\n(' +
            ('both budgets' if len(same) == 2 else f'{same[0]} budget') + '; blue and vermilion filled markers coincide)',
            transform=ax.transAxes, ha='left', va='top', fontsize=5, color=INK)
bs, bb = P['beam single, matched'], P['beam bundle, matched']
ax.annotate('single-test pricing', (bs['billed'], C['beam single, matched']), xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=5, color=REDESIGN)
ax.annotate('as billed', (bb['billed'], C['beam bundle, matched']), xytext=(-4, 4), textcoords='offset points', ha='right', va='bottom', fontsize=5, color=REDESIGN)
ax.legend(handles=h, loc='lower right', ncol=2, columnspacing=1.0)
ax.set_xlabel('Fee of the selected panel, billed (US$)'); ax.set_ylabel("Harrell's C, NHANES III")
ax.set_title('Same algorithms, two fee representations', fontsize=6); label(ax, 'd', dx=-0.07)

# e: ours minus CoAI (published configuration)
ax = fig.add_subplot(gs[1, 2])
q = pd.read_csv(D / 'F3B_coai.csv'); y = 0; ys, names = [], []; bold = []
for b, hd in [('matched-panel budget', 'Matched-panel budget'), ('routine budget', 'Routine budget')]:
    ys.append(y); names.append(hd); bold.append(len(names) - 1); y -= 1
    for r in q[q.budget == b].itertuples():
        pooled = r.cohort.startswith('Pooled')
        if pooled:
            ax.fill([r.lo, r.est, r.hi, r.est], [y, y + 0.28, y, y - 0.28], color=INK, lw=0)
        else:
            ax.plot([r.lo, r.hi], [y, y], color=REDESIGN, lw=0.8); ax.scatter(r.est, y, s=10, color=REDESIGN, marker='o' if r.metric == 'C' else 's', zorder=3)
        ys.append(y); names.append(('Pooled' if pooled else r.cohort)); y -= 1
    y -= 0.4
ax.axvline(0, color=BASE, lw=0.5); ax.set_yticks(ys); ax.set_yticklabels(names); ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
for i in bold:
    ax.get_yticklabels()[i].set_fontweight('bold')
ax.set_xlabel('Ours − CoAI as published (C or AUC)'); ax.set_title('Against CoAI at equal budgets', fontsize=6); label(ax, 'e', dx=-0.62)
save(fig, OUT / 'fig6')
print('ok')
