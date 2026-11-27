#!/usr/bin/env python3
"""Task 3 summaries, crossover flags and figure, following plan_learning_curve.md (frozen 2026-10-28 20:55 NZDT).

Inputs: gdsc_runs.csv, radbill_cells.csv (this folder). Outputs: learning_curve_summary.csv,
learning_curve_contrasts.csv, learning_curve_flags.json, learning_curve.png/.pdf. Run with .env312.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLORS = {'C0': '#ABB3BD', 'Shared': '#3D8EAD', 'Family': '#65AC99', 'Exact≤3': '#9A4F2C', 'Exact': '#D58A68',
          'Conventional': '#7A8190'}
MARKERS = {'C0': 'o', 'Shared': 's', 'Family': 'D', 'Exact≤3': 'v', 'Exact': '^', 'Conventional': 'x'}
STYLE = {'C0': ':', 'Shared': '-', 'Family': '-', 'Exact≤3': '--', 'Exact': '-', 'Conventional': ':'}
GDSC_P = {'C0': 'C0', 'Shared': 'Shared', 'Family': 'Family', 'Exact≤3': 'Global3', 'Exact': 'Exact'}
RAD_P = {'C0': 'C0', 'Shared': 'SHARED', 'Family': 'FAMILY', 'Exact': 'EXACT', 'Conventional': 'CONVENTIONAL'}
CONTRASTS_GDSC = [('Exact', 'Family'), ('Exact≤3', 'Family'), ('Family', 'Shared'), ('Exact', 'Shared')]
CONTRASTS_RAD = [('Exact', 'Family'), ('Exact', 'Shared'), ('Family', 'Shared')]
CANDIDATES_GDSC = ['Shared', 'Family', 'Exact≤3', 'Exact']
CANDIDATES_RAD = ['Shared', 'Family', 'Exact']


def gdsc_tables():
    runs = pd.read_csv(HERE / 'gdsc_runs.csv')
    summ, cons = [], []
    for (ev, sel), g in runs.groupby(['evaluation', 'selector']):
        for n_label, h in g.groupby('n_label', sort=False):
            cells = h.n_test_cells.sum()
            risk = {p: h[f'loss_sum_{c}'].sum() / cells for p, c in GDSC_P.items()}
            best = min(CANDIDATES_GDSC, key=lambda p: risk[p])
            summ.append({'analysis': f'GDSC {ev}', 'selector': sel, 'n_label': str(n_label),
                         'n_train_mean': h.n_train.mean(), 'runs': len(h), **risk, 'best_policy': best,
                         'global3_equals_family_share': h.global3_equals_family.mean(),
                         'exact_distinct_features_mean': h.exact_distinct_features.mean(),
                         'redraws': int(h.redraws.sum())})
            for a, b in CONTRASTS_GDSC:
                diff = h[f'loss_sum_{GDSC_P[a]}'] - h[f'loss_sum_{GDSC_P[b]}']
                per_rep = (diff.groupby(h.rep).sum() / h.n_test_cells.groupby(h.rep).sum())
                per_fold = (diff / h.n_test_cells).groupby([h.outer_seed, h.fold]).mean()
                cons.append({'analysis': f'GDSC {ev}', 'selector': sel, 'n_label': str(n_label),
                             'n_train_mean': h.n_train.mean(), 'contrast': f'{a} − {b}',
                             'mean': diff.sum() / cells, 'rep_2.5': per_rep.quantile(.025), 'rep_97.5': per_rep.quantile(.975),
                             'n_reps': len(per_rep), 'folds_negative': int((per_fold < 0).sum()), 'n_folds': len(per_fold)})
    return summ, cons


def radbill_tables():
    cells = pd.read_csv(HERE / 'radbill_cells.csv')
    summ, cons = [], []
    for mtr, h in cells.groupby('m_train'):
        risk = {p: h[c].mean() for p, c in RAD_P.items()}
        summ.append({'analysis': 'Radbill', 'selector': 'inner LOPO', 'n_label': 'all' if mtr == 16 else str(mtr),
                     'n_train_mean': float(mtr), 'runs': h.rep.nunique(), **risk,
                     'best_policy': min(CANDIDATES_RAD, key=lambda p: risk[p]),
                     'redraws': int(h.drop_duplicates(['outer_participant', 'rep']).redraws.sum())})
        for a, b in CONTRASTS_RAD:
            diff = h[RAD_P[a]] - h[RAD_P[b]]
            per_rep = diff.groupby(h.rep).mean()
            per_fold = diff.groupby(h.outer_participant).mean()
            cons.append({'analysis': 'Radbill', 'selector': 'inner LOPO', 'n_label': 'all' if mtr == 16 else str(mtr),
                         'n_train_mean': float(mtr), 'contrast': f'{a} − {b}', 'mean': diff.mean(),
                         'rep_2.5': per_rep.quantile(.025), 'rep_97.5': per_rep.quantile(.975), 'n_reps': len(per_rep),
                         'folds_negative': int((per_fold < 0).sum()), 'n_folds': len(per_fold)})
    return summ, cons


def flags(summary, contrasts):
    out = {}
    for (an, sel), g in contrasts.groupby(['analysis', 'selector']):
        for c, h in g.groupby('contrast'):
            h = h.sort_values('n_train_mean')
            signs = np.sign(h['mean'].to_numpy())
            observed = bool(len(set(signs[signs != 0])) > 1)
            # the full-n point is a single run with no replicate interval, so it cannot count as excluding zero
            excl = [0 if k == 1 else (-1 if hi < 0 else 1 if lo > 0 else 0)
                    for lo, hi, k in zip(h['rep_2.5'], h['rep_97.5'], h['n_reps'])]
            clear = any(excl[i] != 0 and excl[j] == -excl[i] for i in range(len(excl)) for j in range(i + 1, len(excl)))
            out[f'{an} | {sel} | {c}'] = {'signs_by_n': dict(zip(h.n_label, signs.astype(int).tolist())),
                                          'crossover_observed': observed, 'clear_crossover': clear}
        s = summary[(summary.analysis == an) & (summary.selector == sel)].sort_values('n_train_mean')
        out[f'{an} | {sel} | best policy'] = {'by_n': dict(zip(s.n_label, s.best_policy)),
                                              'changes_between_smallest_and_all': s.best_policy.iloc[0] != s.best_policy.iloc[-1]}
    return out


def figure(summary, contrasts):
    panels = [('GDSC repeated_random', 'inner_cv', 'GDSC, cell-line splits', list(GDSC_P), 'Squared standardized error'),
              ('GDSC held_out_tissue', 'inner_cv', 'GDSC, tissue held out', list(GDSC_P), 'Squared standardized error'),
              ('Radbill', 'inner LOPO', 'Radbill pacing', list(RAD_P), 'Standardized absolute error')]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for k, (an, sel, title, pols, ylab) in enumerate(panels):
        s = summary[(summary.analysis == an) & (summary.selector == sel)].sort_values('n_train_mean')
        ax = axes[0, k]
        for p in pols:
            ax.plot(s.n_train_mean, s[p], STYLE[p], marker=MARKERS[p], color=COLORS[p], lw=1.6, ms=5, label=p)
        ax.set_xscale('log'); ax.set_title(title, loc='left', fontsize=10, fontweight='bold')
        ax.set_xticks(s.n_train_mean); ax.set_xticklabels([f'{v:.0f}' for v in s.n_train_mean], fontsize=8)
        ax.minorticks_off(); ax.set_ylabel(ylab, fontsize=8.5); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, loc='upper right')
        ax2 = axes[1, k]
        cs = contrasts[(contrasts.analysis == an) & (contrasts.selector == sel)]
        shown = {'Family − Shared': COLORS['Family'], 'Exact − Family': COLORS['Exact'], 'Exact≤3 − Family': COLORS['Exact≤3']}
        for c, col in shown.items():
            h = cs[cs.contrast == c].sort_values('n_train_mean')
            if h.empty:
                continue
            a = c.split(' − ')[0]
            ax2.plot(h.n_train_mean, h['mean'], STYLE[a], marker=MARKERS[a], color=col, lw=1.5, ms=4.5, label=c)
            if c != 'Exact≤3 − Family':
                ax2.fill_between(h.n_train_mean, h['rep_2.5'], h['rep_97.5'], color=col, alpha=.15, lw=0)
        ax2.axhline(0, color='#888', lw=.8)
        ax2.set_xscale('log'); ax2.set_xticks(s.n_train_mean)
        ax2.set_xticklabels([f'{v:.0f}' for v in s.n_train_mean], fontsize=8); ax2.minorticks_off()
        ax2.set_xlabel('Training units (log scale)', fontsize=8.5); ax2.tick_params(labelsize=8)
        ax2.set_ylabel('Risk difference (<0: first policy better)', fontsize=8)
        ax2.legend(fontsize=7, frameon=False)
        for a_ in (ax, ax2):
            a_.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.005, 'Post hoc. C0 = no added measurement. Bands: 2.5–97.5% across subsample replicates (descriptive; folds overlap). '
             'Rightmost point = full training set (V3.2 values).', fontsize=7.5, color='#555')
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / f'learning_curve.{ext}', dpi=300)


def main():
    gs, gc = gdsc_tables(); rs, rc = radbill_tables()
    summary = pd.DataFrame(gs + rs); contrasts = pd.DataFrame(gc + rc)
    summary.to_csv(HERE / 'learning_curve_summary.csv', index=False)
    contrasts.to_csv(HERE / 'learning_curve_contrasts.csv', index=False)
    fl = flags(summary, contrasts)
    (HERE / 'learning_curve_flags.json').write_text(json.dumps(fl, indent=1, ensure_ascii=False), encoding='utf-8')
    figure(summary, contrasts)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summary.drop(columns=['runs']).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print(contrasts.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(json.dumps(fl, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
