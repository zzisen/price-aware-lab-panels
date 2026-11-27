#!/usr/bin/env python3
"""Figure 4 of NC v0.7: when to redesign — the rule, why it works, and its five locked tests (v0.7: terms unified, all text >= 5 pt).
v0.9: locked test 5 (task48_singlecell/t48_summary.json; 10 single-cell sources, rule v4 unchanged) added to panels d and e.
Data: Task 36 stage 1 (46 used sources, outer split 0), Task 34 locked test 1 (22 sources), Task 39 locked test 2 (17), Task 40 locked test 3 (14; rule v4 = count + learnability check), Task 42 locked test 4 (12; same rule with gradient-boosted trees, or with pivoted QR + lasso).
Development value of rule v4: Task 40 dry run on the 63 old sources (type-balanced).
Output: figs_v2/out/fig4.*"""
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
OUT = HERE / 'out'
TYPE = {'G': 'GTEx tissues', 'R': 'TCGA RPPA', 'X': 'CRISPR screens', 'P': 'Proteomics (MS)', 'T': 'TCGA RNA-seq', 'N': 'RNAi screens', 'S': 'Single-cell PBMC'}
T5COL = '#6A3D9A'   # locked test 5 (single cell)
TCOL = {'G': SKY, 'R': PURPLE, 'X': ORANGE, 'P': INK, 'T': '#1F4E79', 'N': '#8C510A', 'S': T5COL}
TORD = 'GRPTXNS'


def load(folder, pat='*_o0.csv', cv=('cv5_A', 'cv5_C')):
    fs = glob.glob(str(folder / pat))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d['nk'] = d.n / d.k; d['t'] = d.source.str[0]; d['best'] = d[['A', 'B', 'C']].max(axis=1)
    cvp = np.where(d[cv[0]] < d[cv[1]], d.A, d.C)
    d['r_A'], d['r_B'], d['r_C'] = d.best - d.A, d.best - d.B, d.best - d.C
    d['r_CV'] = d.best - cvp; d['r_NK'] = d.best - np.where(d.nk >= 10, d.C, d.A); d['r_H'] = d.best - np.where(d.nk >= 10, d.C, cvp)
    d['r_V4'] = d.best - np.where((d.nk >= 10) & ((1 - d.cv5_C) >= 0.10), d.C, cvp) if 'cv5_C' in d else np.nan   # rule v4 (learnability check)
    return d


dev = load(W / 'task36_rule_v3' / 'stage1_results')
lk1 = load(W / 'task34_decision' / 'results', '*.csv', ('inner_A', 'inner_C'))
lk2 = None if 'nolk2' in sys.argv else load(W / 'task39_locked_v3' / 'results')
lk3 = load(W / 'task40_locked_v4' / 'results')
T4 = {nm: json.load(open(W / 'task42_locked_v5' / f)) for nm, f in [('trees', 't42T_summary.json'), ('QR + lasso', 't42S_summary.json')]}
T5 = json.load(open(W / 'task48_singlecell' / 't48_summary.json'))
assert T5['best_fixed'] == 'always_A'
DEV63 = json.load(open(W / 'task40_locked_v4' / 'dryrun_old63_summary.json'))['type_balanced']
K4 = {'r_B': 'always_B', 'r_A': 'always_A', 'r_NK': 'NK10', 'r_CV': 'CV', 'r_C': 'always_C', 'r_H': 'H', 'r_V4': 'v4'}
tb = lambda d, m: float(d.groupby(['t', 'source'])[m].mean().groupby(level=0).mean().mean())       # type-balanced (locked test 3)
fig = plt.figure(figsize=(W2, 166 * MM), layout='constrained')
SF = fig.subfigures(3, 1, height_ratios=[1, 0.92, 1.12])   # separate rows so the wide bottom panels do not distort the top columns
gs = SF[0].add_gridspec(1, 3, width_ratios=[1.05, 1, 0.9])

# a: gain of redesign vs n/k, by data type
ax = SF[0].add_subplot(gs[0, 0])
bins = [2.4, 4, 6, 8.5, 10, 13, 17, 25, 41]; mid = np.sqrt(np.array(bins[:-1]) * np.array(bins[1:]))
ax.axhline(0, color=BASE, lw=0.5); ax.axvline(10, color=RULE, lw=0.6, ls=DASH)
series = [(dev, 'G', '-', 'o'), (dev, 'R', '-', 'o')]
if lk2 is not None:
    series += [(lk2, 'X', '--', 's'), (lk2, 'P', '--', 's')]
if lk3 is not None:
    series += [(lk3, 'T', ':', '^'), (lk3, 'N', ':', '^')]
for d, t, ls, mk in series:
    x = d[d.t == t].assign(b=lambda z: pd.cut(z.nk, bins), g=lambda z: z.C - z.A)
    s = x.groupby(['b', 'source'], observed=True).g.mean().groupby('b', observed=True)
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(s.count())
    ax.fill_between(mid[:len(m)], m - 1.96 * se, m + 1.96 * se, color=TCOL[t], alpha=0.18, lw=0)
    ax.plot(mid[:len(m)], m, ls=ls, marker=mk, ms=2.5, color=TCOL[t], label=f"{TYPE[t]} ({x.source.nunique()})")
ax.set_xscale('log'); ax.set_xticks([3, 5, 10, 20, 40]); ax.set_xticklabels(['3', '5', '10', '20', '40'])
ax.set_xlabel('Training units per measurement, n / k'); ax.set_ylabel('Redesigned − reusable panel\n(test variance explained)')
ax.text(10.5, 0.115, 'n/k = 10', color=RULE, fontsize=5)
ax.legend(loc='lower right', handlelength=1.8, fontsize=5, bbox_to_anchor=(1.02, -0.01), ncol=1); ax.set_title('Redesign pays with data, if targets carry signal'); label(ax, 'a', dx=-0.25)

# b: where cross-validation helps (development data)
ax = SF[0].add_subplot(gs[0, 1])
bb = [0, 5, 7.5, 10, 15, 50]; names = ['<5', '5–7.5', '7.5–10', '10–15', '≥15']
x = dev.assign(b=pd.cut(dev.nk, bb, right=False))
g = x.groupby('b', observed=True)[['r_A', 'r_C', 'r_CV', 'r_H']].mean()
xi = np.arange(len(g))
for col, nm, c, mk in [('r_A', 'Always reuse', REUSE, 'o'), ('r_C', 'Always redesign', REDESIGN, 'o'), ('r_CV', 'Cross-validation alone', BASE, 'D'), ('r_H', 'Rule', RULE, 's')]:
    ax.plot(xi, g[col], color=c, marker=mk, ms=2.8, lw=0.9 if col == 'r_H' else 0.7, label=nm, zorder=4 if col == 'r_H' else 3)
ax.axvspan(2.5, len(g) - 0.5, color=REDESIGN, alpha=0.06, lw=0); ax.text(3.5, 0.086, 'Rule: redesign', color=REDESIGN, fontsize=5, ha='center')
ax.text(1, 0.086, 'Rule: cross-validate', color=INK, fontsize=5, ha='center')
ax.set_xticks(xi); ax.set_xticklabels(names); ax.set_xlabel('n / k'); ax.set_ylabel('Mean regret'); ax.set_ylim(0, 0.092)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=2, columnspacing=0.8); ax.set_title('Development: 46 sources, 82 944 cells'); label(ax, 'b', dx=-0.2)

# c: single test cells vs averaged settings at n/k >= 10
ax = SF[0].add_subplot(gs[0, 2])
hi = dev[dev.nk >= 10].assign(g=lambda z: z.C - z.A)
st = hi.groupby(['source', 'task', 'p', 'n', 'k']).g.mean()
be = np.linspace(-0.15, 0.25, 41)
ax.hist(hi.g.clip(-0.15, 0.25), be, color=LIGHT, density=True, label=f'Single test cell: {100 * (hi.g > 0).mean():.0f}% > 0')
ax.hist(st.clip(-0.15, 0.25), be, histtype='step', color=REDESIGN, lw=0.9, density=True, label=f'Averaged over repeats: {100 * (st > 0).mean():.0f}% > 0')
ax.axvline(0, color=INK, lw=0.5); ax.set_xlabel('Redesigned − reusable at n/k ≥ 10'); ax.set_ylabel('Density')
ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.02), frameon=True, facecolor='white', edgecolor='none', framealpha=1, borderpad=0.2); ax.set_title('At n/k ≥ 10: redesigned − reusable'); label(ax, 'c', dx=-0.22)

# d: regret of strategies in development and locked tests
ax = SF[1].add_subplot(1, 1, 1)
meth = [('r_B', 'Always historically trained'), ('r_A', 'Always reuse'), ('r_NK', 'n/k alone'), ('r_CV', 'Cross-validation alone'), ('r_C', 'Always redesign'), ('r_H', 'Rule (count only)'), ('r_V4', 'Rule + learnability check')]
y = np.arange(len(meth))[::-1]
T4COL = '#1F4E79'
# slot, label, marker, edge colour, face colour, value per row (development of the learnability rule: 63 sources, not 46)
evals = [(0, 'Development (46 sources)', 'o', BASE, BASE, {m: (np.nan if m == 'r_V4' else dev[m].mean()) for m, _ in meth}),
         (0, 'Development (63; types equal)', 'o', BASE, 'white', {m: (DEV63['v4'] if m == 'r_V4' else np.nan) for m, _ in meth}),
         (1, 'Locked test 1 (22)', 's', REUSE, REUSE, {m: (np.nan if m == 'r_V4' else lk1[m].mean()) for m, _ in meth})]
if lk2 is not None:
    evals += [(2, f'Locked test 2 ({lk2.source.nunique()})', 'D', ORANGE, ORANGE, {m: (np.nan if m == 'r_V4' else lk2[m].mean()) for m, _ in meth})]
if lk3 is not None:
    evals += [(3, f'Locked test 3 ({lk3.source.nunique()}; types equal)', '^', RULE, RULE, {m: tb(lk3, m) for m, _ in meth})]
for j, (nm, mk, fc) in enumerate([('trees', 'v', T4COL), ('QR + lasso', 'X', PURPLE)]):
    tbv = T4[nm]['type_balanced']
    evals += [(4 + j, f'Locked test 4, {nm} ({sum(T4[nm]["sources"].values())}; {"predictor" if nm == "trees" else "selectors"} changed)', mk, fc, fc, {m: tbv[K4[m]] for m, _ in meth})]
K5 = dict(K4, r_V4='rule')   # test 5 summary: 'rule' = rule v4, 'H' = count rule without the check
evals += [(6, f'Locked test 5 ({T5["sources"]} single-cell; targets not learnable)', 'P', T5COL, T5COL, {m: T5['regret'][K5[m]] for m, _ in meth})]
nsl = 7; step = 0.12
for sl, nm, mk, c, fc, vals in evals:
    ax.scatter([vals[m] for m, _ in meth], y + (sl - (nsl - 1) / 2) * -step, marker=mk, s=13 if mk in ('X', 'P') else 11, color=fc, edgecolor=c if fc == 'white' else 'white',
               lw=0.6 if fc == 'white' else 0.3, label=nm, zorder=3)
for yi in y:
    ax.axhline(yi, color=LIGHT, lw=0.4, zorder=0)
ax.set_yticks(y); ax.set_yticklabels([n for _, n in meth]); ax.get_yticklabels()[-1].set_color(RULE); ax.get_yticklabels()[-1].set_fontweight('bold'); ax.get_yticklabels()[-2].set_color(RULE)
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False); ax.set_xlim(0, 0.155); ax.set_ylim(-0.6, len(meth) - 0.4)
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14])
ax.set_xlabel('Mean regret (variance explained lost)')
ax.legend(loc='center left', fontsize=5.5, handletextpad=0.3, bbox_to_anchor=(1.01, 0.5), labelspacing=0.5)
ax.text(1.01, 0.02, 'Test 4: rule unchanged; predictor (trees instead of ridge)\nor selectors (QR + lasso instead of FSCA + somp) changed', transform=ax.transAxes, ha='left', va='bottom', fontsize=5, color=T4COL, linespacing=1.15)
ax.set_title('Development and five locked tests'); SF[1].text(0.0, 1.0, 'd', fontsize=8, fontweight='bold', va='top', ha='left')

# e: per source, rule minus always-redesign, locked tests
ax = SF[2].add_subplot(1, 1, 1)
rows = []
for d, nm, rc, bal in [(lk1, 'Locked test 1', 'r_H', False), (lk2, 'Locked test 2', 'r_H', False), (lk3, 'Locked test 3', 'r_V4', True)]:
    if d is None:
        continue
    ref = min(['r_A', 'r_B', 'r_C'], key=lambda c: tb(d, c) if bal else d[c].mean())   # best fixed practice of that test (prespecified comparator)
    for s, g in d.groupby('source'):
        rows.append(dict(test=nm, source=s, t=s[0], diff=g[rc].mean() - g[ref].mean(), ref=ref, rc=rc))
pt = T4['trees']['secondary']['per_source_v4_minus_best_fixed']; ps = T4['QR + lasso']['secondary']['per_source_v4_minus_best_fixed']
assert T4['trees']['best_fixed'] == 'always_C' and T4['QR + lasso']['best_fixed'] == 'always_C'
for s_ in pt:
    rows.append(dict(test='Locked test 4', source=s_, t=s_[0], diff=pt[s_], diff2=ps[s_], ref='r_C', rc='r_V4'))
for s_, v_ in T5['secondary']['per_source_rule_minus_best_fixed'].items():   # best fixed of test 5 = always reuse (asserted above)
    rows.append(dict(test='Locked test 5', source=s_, t='S', diff=v_, ref='r_A', rc='r_V4'))
q = pd.DataFrame(rows).sort_values(['test', 'diff']).reset_index(drop=True)
xpos, xt, gap = [], [], 0
for i, r in q.iterrows():
    if i > 0 and r.test != q.test[i - 1]:
        gap += 1.5
    xpos.append(i + gap)
q['x'] = xpos
ax.axhline(0, color=INK, lw=0.5)
o3 = q[q.test != 'Locked test 4']; o4 = q[q.test == 'Locked test 4']
ax.vlines(o3.x, 0, o3['diff'], color=[TCOL[t] for t in o3.t], lw=1.6)
ax.scatter(o3.x, o3['diff'], s=6, color=[TCOL[t] for t in o3.t], zorder=3)
ax.vlines(o4.x - 0.2, 0, o4['diff'], color=TCOL['T'], lw=1.0); ax.scatter(o4.x - 0.2, o4['diff'], s=9, marker='v', color=TCOL['T'], lw=0, zorder=3)
ax.vlines(o4.x + 0.2, 0, o4['diff2'], color=PURPLE, lw=1.0); ax.scatter(o4.x + 0.2, o4['diff2'], s=11, marker='X', color=PURPLE, lw=0, zorder=3)
SHORT = {'cells_cultured_fibroblasts': 'fibroblasts', 'adipose_visceral_omentum': 'adipose visceral', 'skin_not_sun_exposed_suprapubic': 'skin, unexposed', 'esophagus_gastroesophageal_junction': 'GE junction', 'heart_atrial_appendage': 'heart atrial', 'brain_caudate_basal_ganglia': 'brain caudate', 'brain_spinal_cord_cervical_c-1': 'spinal cord', 'cells_ebv-transformed_lymphocytes': 'lymphocytes (EBV)', 'small_intestine_terminal_ileum': 'ileum'}
SC = {'S_b_cells': 'B cells', 'S_cd14_monocytes': 'CD14 monocytes', 'S_cd34': 'CD34+', 'S_cd4_t_helper': 'CD4 helper T', 'S_cd56_nk': 'CD56 NK', 'S_cytotoxic_t': 'cytotoxic T', 'S_memory_t': 'memory T', 'S_naive_cytotoxic': 'naive cytotoxic T', 'S_naive_t': 'naive T', 'S_regulatory_t': 'regulatory T'}
nice = lambda s: SC[s] if s in SC else SHORT.get(s[2:], s).replace('G_', '').replace('R_', 'RPPA ').replace('X_crispr', 'CRISPR ').replace('P_ccle', 'Proteomics ').replace('N_rnai', 'RNAi ').replace('T_', 'TCGA ').replace('_', ' ')
ax.set_xticks(q.x); ax.set_xticklabels([nice(s) for s in q.source], rotation=90, fontsize=5)
for nm, g in q.groupby('test'):
    rn = 'rule + learnability check' if g.rc.iloc[0] == 'r_V4' else 'rule'
    vs = f'vs always {"redesign" if g.ref.iloc[0] == "r_C" else "reuse"}'
    if nm == 'Locked test 4':
        bt = f'better in {(g["diff"] < 0).sum()}/{len(g)} (trees),\n{(g["diff2"] < 0).sum()}/{len(g)} (QR + lasso)'
    elif nm == 'Locked test 5':
        bt = f'better in {(g["diff"] < 0).sum()}/{len(g)} sources\n(each {g["diff"].min():+.4f} to {g["diff"].max():+.4f})'.replace('-', '\u2212')
    else:
        bt = f'better in {(g["diff"] < 0).sum()}/{len(g)} sources'
    ax.text(g.x.mean(), 0.034, f'{nm}\n{rn}\n{vs}\n{bt}', ha='center', va='bottom', fontsize=5.0, linespacing=1.15)
NI4 = 0.005   # prespecified non-inferiority margin of locked test 4 (task42_locked_v5/PLAN_FROZEN_v5.md, H1); v0.8 addition
ax.plot([o4.x.min() - 0.7, o4.x.max() + 0.7], [NI4, NI4], color=INK, lw=0.7, ls=DASH, zorder=4)
ax.text(o4.x.min() - 0.9, NI4 + 0.0015,'non-inferiority margin (test 4)', ha='right', va='center', fontsize=5.0)
ax.set_ylabel('Rule − best fixed practice\n(mean regret)'); ax.set_xlim(-1, q.x.max() + 1); ax.set_ylim(-0.045, 0.06)
hh = [Line2D([], [], color=TCOL[t], lw=1.6, label=TYPE[t]) for t in sorted(set(q.t), key=TORD.index)]
hh += [Line2D([], [], color=TCOL['T'], lw=1.0, marker='v', ms=3, mew=0, label='Test 4, trees'), Line2D([], [], color=PURPLE, lw=1.0, marker='X', ms=3.3, mew=0, label='Test 4, QR + lasso')]
ax.legend(handles=hh, loc='upper left', ncol=3, bbox_to_anchor=(0.0, 0.735), fontsize=5.5, columnspacing=1.2)
ax.set_title('Locked tests: every source analysed for the first time; rule frozen in advance'); label(ax, 'e', dx=-0.08)
save(fig, OUT / 'fig4')
print('ok', None if lk2 is None else lk2.source.nunique(), None if lk3 is None else lk3.source.nunique())
