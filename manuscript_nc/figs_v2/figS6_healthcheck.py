#!/usr/bin/env python3
"""Supplementary Fig. 6: health-check programme tests compared with panels chosen for death (additional analysis).
Differences in AUC (left) and in net benefit at the cohort's decision threshold (right), 95% bootstrap CI, against the programme
tests and against age and sex alone. a, CHNS (Chinese prices; 2017 and 2025 NBPHS test lists); b, Understanding Society (US prices;
NHS Health Check proxy). Rows without a net-benefit marker: that contrast was not computed in the result file.
Inputs (read only):
  task29_revision/valid_CHNS.json   (CHNS, 2017 NBPHS tests)
  task32_tier2/t32a_CHNS.json       (CHNS, 2025 NBPHS tests)
  task29_revision/valid_UKHLS.json  (Understanding Society, NHS Health Check proxy)
Output: figs_v2/out/figS6.pdf/.png"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402

F17, F25, FUK = W / 'task29_revision/valid_CHNS.json', W / 'task32_tier2/t32a_CHNS.json', W / 'task29_revision/valid_UKHLS.json'
print('Files read:'); [print(' ', f) for f in (F17, F25, FUK)]
C17, C25, UK = (json.load(open(f, encoding='utf-8')) for f in (F17, F25, FUK))
NM = {'CBC': 'CBC', 'CMP': 'CMP', 'LIPID': 'lipid panel', 'Hb': 'haemoglobin', 'hba1c': 'HbA1c', 'ggt': 'GGT', 'alt': 'ALT', 'albumin': 'albumin',
      'protein': 'total protein', 'bun': 'urea', 'creatinine': 'creatinine', 'glucose': 'glucose', 'crp': 'CRP'}
items = lambda p: ' + '.join(NM[t] for t in p['order'])  # noqa: E731
cn = lambda v: f'CN¥{v:.1f}'  # noqa: E731
us = lambda v: f'US${v:.2f}'  # noqa: E731


def row(lab, src, key, color=INK):
    return dict(label=lab, auc=src['contrasts'].get(key), nb=src['nb_contrasts'].get(key), color=color)


P17, P25, PU = C17['panels'], C25['panels'], UK['panels']
A = [dict(header=f"vs 2017 NBPHS tests ({cn(P17['NBPHS check']['price'])}; AUC {P17['NBPHS check']['disc']:.3f})"),
     row('Programme tests minus age and sex alone', C17, 'NBPHS check - age+sex only', BASE),
     row(f"Matched to programme: {items(P17['matched to NBPHS'])} ({cn(P17['matched to NBPHS']['price'])})", C17, 'matched to NBPHS - NBPHS check', MATCHED),
     row(f"Matched panel (matched to routine items; {cn(P17['matched']['price'])})", C17, 'matched - NBPHS check', MATCHED),
     row(f"Best panel within programme price ({cn(P17['NBPHS budget']['price'])})", C17, 'NBPHS budget - NBPHS check', REDESIGN),
     row(f"Study's routine items ({cn(P17['routine']['price'])})", C17, 'routine - NBPHS check', INK),
     row(f"Target-aware panel at routine price ({cn(P17['aware@routine']['price'])})", C17, 'aware@routine - NBPHS check', REDESIGN),
     dict(header=f"vs 2025 NBPHS tests ({cn(P25['NBPHS 2025']['price'])}; AUC {P25['NBPHS 2025']['disc']:.3f})"),
     row('Programme tests minus age and sex alone', C25, 'NBPHS 2025 - age+sex only', BASE),
     row('2025 minus 2017 programme tests', C25, 'NBPHS 2025 - NBPHS 2017', BASE),
     row(f"Matched to programme: {items(P25['matched to NBPHS 2025'])} ({cn(P25['matched to NBPHS 2025']['price'])})", C25,
         'matched to NBPHS 2025 - NBPHS 2025', MATCHED),
     row(f"Best panel within programme price ({cn(P25['NBPHS 2025 budget']['price'])})", C25, 'NBPHS 2025 budget - NBPHS 2025', REDESIGN),
     row(f"Target-aware panel at routine price ({cn(P25['aware@routine']['price'])})", C25, 'aware@routine - NBPHS 2025', REDESIGN)]
hc = PU['NHS Health Check']
B = [dict(header=f"vs NHS Health Check proxy: {items(hc)} ({us(hc['price'])}; AUC {hc['disc']:.3f})"),
     row('Programme tests minus age and sex alone', UK, 'NHS Health Check - age+sex only', BASE),
     row(f"Matched to programme in training: {items(PU['matched to Health Check'])} ({us(PU['matched to Health Check']['price'])})", UK,
         'matched to Health Check - NHS Health Check', MATCHED),
     row(f"Best panel within programme price: {items(PU['Health Check budget'])} ({us(PU['Health Check budget']['price'])})", UK,
         'Health Check budget - NHS Health Check', REDESIGN),
     dict(header=f"vs routine items billed as measured ({us(PU['routine']['price'])}; AUC {PU['routine']['disc']:.3f})"),
     row(f"Best panel within Health Check price ({us(PU['Health Check budget']['price'])})", UK, 'Health Check budget - routine', REDESIGN)]

fig = plt.figure(figsize=(W2, 122 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[len(A), len(B)], width_ratios=[1.35, 1], left=0.4, right=0.985, bottom=0.08, top=0.92,
                      hspace=0.42, wspace=0.12)
for i, (rows, title, src) in enumerate([(A, f"CHNS ({C17['n']} participants, {C17['deaths']} deaths; {C17['horizon'] // 12}-year horizon)", C17),
                                        (B, f"Understanding Society ({UK['n']} participants, {UK['deaths']} deaths; {UK['horizon'] // 12}-year horizon)", UK)]):
    ya = range(len(rows))
    for j, (kind, xl) in enumerate([('auc', 'Difference in AUC'), ('nb', f"Difference in net benefit at {src['tau']:.0%} risk")]):
        ax = fig.add_subplot(gs[i, j])
        for y, r in zip(ya, rows):
            if 'header' in r:
                continue
            d = r[kind]
            if d is None:
                if kind == 'nb':
                    ax.text(0, y, 'not computed', ha='center', va='center', fontsize=5, color=BASE, zorder=5,
                            bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
                continue
            lo, hi = d['ci']
            ax.plot([lo, hi], [y, y], color=r['color'], lw=0.8, solid_capstyle='butt')
            ax.plot(d['point'], y, 'o', ms=3, color=r['color'], mew=0)
        ax.axvline(0, color=INK, lw=0.5)
        ax.set_ylim(len(rows) - 0.5, -0.7)
        for y, r in zip(ya, rows):
            if 'header' in r:
                ax.axhspan(y - 0.5, y + 0.5, color=LIGHT, alpha=0.5, lw=0, zorder=0)
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.set_xlabel(xl)
        if kind == 'auc':
            ax.set_xlim(-0.03, 0.04); ax.set_xticks([-0.02, 0, 0.02, 0.04])
            for y, r in zip(ya, rows):
                ax.text(-0.02, y, r.get('header', r.get('label')), transform=ax.get_yaxis_transform(), ha='right', va='center',
                        fontsize=5.5 if 'header' in r else 5.3, fontweight='bold' if 'header' in r else 'normal')
            yt = ax.get_position().y1 + 0.035
            flabel(fig, 'ab'[i], 0.01, yt); fig.text(0.04, yt - 0.002, title, fontsize=6.5, va='top', ha='left')
        else:
            ax.set_xlim(-0.005, 0.01); ax.set_xticks([-0.004, 0, 0.004, 0.008])
        ax.set_xticklabels([f'{v:g}'.replace('-', '−') for v in ax.get_xticks()])
save(fig, HERE / 'out' / 'figS6'); print('ok')
