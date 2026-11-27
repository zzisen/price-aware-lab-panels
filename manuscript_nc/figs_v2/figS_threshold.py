#!/usr/bin/env python3
"""Supplementary Fig. 8: where redesign overtakes reuse in real data (Task 41, post hoc).
v0.8: the simulation panel was removed (it duplicates Fig. 3d); the two empirical panels are relettered a, b.
a, real data (77 sources already analysed): redesigned minus reusable explained variance against n/k by task signal; b, by pool size.
Input: task41_threshold/empirical.json. Output: figs_v2/out/figS8.* (and the same file under the old name figS_threshold.*)"""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent.parent; sys.path.insert(0, str(HERE))
from nstyle import *  # noqa: F401,F403,E402
import matplotlib.pyplot as plt  # noqa: E402
T = W / 'task41_threshold'
emp = json.load(open(T / 'empirical.json'))
fig, ax = plt.subplots(1, 2, figsize=(W15, 58 * MM)); fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.88, wspace=0.38)


def curve(c):
    xs, ys = [], []
    for b, v in c.items():
        lo, hi = [float(x) for x in b.strip('[)').split(',')]
        xs.append(np.sqrt(lo * hi)); ys.append(v)
    return xs, ys


SC = {'(-1.0, 0.05]': ('< 0.05', '#BDBDBD'), '(0.05, 0.15]': ('0.05–0.15', ORANGE), '(0.15, 0.3]': ('0.15–0.3', PURPLE),
      '(0.3, 0.45]': ('0.3–0.45', SKY), '(0.45, 0.6]': ('0.45–0.6', REUSE), '(0.6, 1.0]': ('> 0.6', INK)}
b = ax[0]
for k, (lab, col) in SC.items():
    x, y = curve(emp['curve_by_signal'][k]); b.plot(x, y, '-o', color=col, ms=2, label=lab)
b.set_title('By task signal'); b.legend(title='signal', title_fontsize=5, fontsize=5, loc='lower right', ncol=2, frameon=True, facecolor='white', edgecolor='none', framealpha=1, borderpad=0.3)
label(b, 'a', -0.3)
c = ax[1]
cols = {71: SKY, 200: REUSE, 1000: INK}
for pp in (71, 200, 1000):
    x, y = curve(emp['curve_by_p_nonRPPA'][str(pp)]); c.plot(x, y, '-o', color=cols[pp], ms=2, label=f'p = {pp}')
c.set_title('By pool size (same sources*)'); c.legend(loc='lower right', fontsize=5); label(c, 'b', -0.3)
for z in (b, c):
    z.axhline(0, color=BASE, lw=0.5); z.axvline(10, color=RULE, ls=DASH, lw=0.7); z.set_xscale('log')
    z.set_xticks([3, 5, 10, 20, 30]); z.set_xticklabels(['3', '5', '10', '20', '30']); z.minorticks_off()
    z.set_xlabel('n / k'); z.set_ylabel('Redesigned − reusable\nexplained variance')
c.text(0.97, 0.30, '*All types except RPPA;\ntasks with signal > 0.05', transform=c.transAxes, fontsize=5, va='bottom', ha='right')
fig.savefig(str(HERE / 'out' / 'figS_threshold') + '.pdf'); fig.savefig(str(HERE / 'out' / 'figS_threshold') + '.png', dpi=300)
save(fig, HERE / 'out' / 'figS8'); print('ok')
