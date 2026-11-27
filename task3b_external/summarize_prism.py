#!/usr/bin/env python3
"""Task 3b summaries: E1 learning curves in PRISM (with the replication criteria fixed in plan_external_prism.md),
E2 transport of GDSC-chosen measurements, and a figure. Run with .env312 after prism_analysis.py."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task3_learning_curve'))
import summarize_learning_curves as S  # noqa: E402  (same colours, contrasts and crossover rules as Task 3)

DS = sys.argv[1] if len(sys.argv) > 1 else 'prism'
LABEL = {'prism': 'PRISM', 'gdsc2': 'GDSC2', 'gdsc2_new': 'GDSC2 new lines'}[DS]


def e1_tables():
    runs = pd.read_csv(HERE / f'{DS}_runs.csv')
    summ, cons = [], []
    for (ev, sel), g in runs.groupby(['evaluation', 'selector']):
        for n_label, h in g.groupby('n_label', sort=False):
            cells = h.n_test_cells.sum()
            risk = {p: h[f'loss_sum_{c}'].sum() / cells for p, c in S.GDSC_P.items()}
            summ.append({'analysis': f'{LABEL} {ev}', 'selector': sel, 'n_label': str(n_label), 'n_train_mean': h.n_train.mean(),
                         'runs': len(h), **risk, 'best_policy': min(S.CANDIDATES_GDSC, key=lambda p: risk[p]),
                         'global3_equals_family_share': h.global3_equals_family.mean(),
                         'exact_distinct_features_mean': h.exact_distinct_features.mean(), 'redraws': int(h.redraws.sum())})
            for a, b in S.CONTRASTS_GDSC:
                diff = h[f'loss_sum_{S.GDSC_P[a]}'] - h[f'loss_sum_{S.GDSC_P[b]}']
                per_rep = diff.groupby(h.rep).sum() / h.n_test_cells.groupby(h.rep).sum()
                per_fold = (diff / h.n_test_cells).groupby([h.outer_seed, h.fold]).mean()
                cons.append({'analysis': f'{LABEL} {ev}', 'selector': sel, 'n_label': str(n_label), 'n_train_mean': h.n_train.mean(),
                             'contrast': f'{a} − {b}', 'mean': diff.sum() / cells, 'rep_2.5': per_rep.quantile(.025),
                             'rep_97.5': per_rep.quantile(.975), 'n_reps': len(per_rep),
                             'folds_negative': int((per_fold < 0).sum()), 'n_folds': len(per_fold)})
    return pd.DataFrame(summ), pd.DataFrame(cons)


def criteria(summary, contrasts):
    s = summary[(summary.analysis == f'{LABEL} repeated_random') & (summary.selector == 'inner_cv')].set_index('n_label')
    c = contrasts[(contrasts.analysis == f'{LABEL} repeated_random') & (contrasts.selector == 'inner_cv')
                  & (contrasts.contrast == 'Family − Shared')].sort_values('n_train_mean')
    sub = c[c.n_reps > 1]
    pos = [i for i, lo in enumerate(sub['rep_2.5']) if lo > 0]
    neg = [i for i, hi in enumerate(sub['rep_97.5']) if hi < 0]
    return {
        '1 full-n ordering (Family < Shared at n=all)': bool(s.loc['all', 'Family'] < s.loc['all', 'Shared']),
        '2 small-n reversal (Family − Shared > 0 at n=25)': bool(c[c.n_label == '25']['mean'].iloc[0] > 0),
        '3 clear crossover (interval >0 at smaller n and <0 at larger n, n<all)': bool(any(p < q for p in pos for q in neg)),
        'Family − Shared by n': dict(zip(c.n_label, c['mean'].round(4))),
        'Exact − Family by n': dict(zip(*[contrasts[(contrasts.analysis == f'{LABEL} repeated_random') & (contrasts.selector == 'inner_cv')
                                                  & (contrasts.contrast == 'Exact − Family')].sort_values('n_train_mean')[k]
                                       for k in ('n_label', 'mean')])),
    }


def e2_table():
    t = pd.read_csv(HERE / f'{DS}_transport_runs.csv')
    p = pd.read_csv(HERE / f'{DS}_runs.csv')
    p = p[(p.evaluation == 'repeated_random') & (p.selector == 'inner_cv')]
    rows = []
    for n_label, h in t.groupby('n_label', sort=False):
        q = p[p.n_label.astype(str) == str(n_label)]
        r = {'n_label': str(n_label), 'n_train_mean': h.n_train.mean(), 'C0': h.loss_sum_C0.sum() / h.n_test_cells.sum()}
        for pol, col in S.GDSC_P.items():
            if pol == 'C0':
                continue
            r[f'{pol} (GDSC-chosen)'] = h[f'loss_sum_{col}'].sum() / h.n_test_cells.sum()
            r[f'{pol} (locally chosen)'] = q[f'loss_sum_{col}'].sum() / q.n_test_cells.sum()
            r[f'{pol} (GDSC-chosen, new lines)'] = h[f'loss_sum_{col}_new_units'].sum() / h.n_test_cells_new_units.sum()
        r['C0 (new lines)'] = h.loss_sum_C0_new_units.sum() / h.n_test_cells_new_units.sum()
        rows.append(r)
    return pd.DataFrame(rows)


def figure(summary, contrasts, e2):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for k, (an, title) in enumerate([(f'{LABEL} repeated_random', f'{LABEL}, cell-line splits'), (f'{LABEL} held_out_tissue', f'{LABEL}, tissue held out')]):
        s = summary[(summary.analysis == an) & (summary.selector == 'inner_cv')].sort_values('n_train_mean')
        ax = axes[k]
        if s.empty:
            ax.axis('off'); continue
        for p in S.GDSC_P:
            ax.plot(s.n_train_mean, s[p], S.STYLE[p], marker=S.MARKERS[p], color=S.COLORS[p], lw=1.6, ms=5, label=p)
        ax.set_xscale('log'); ax.set_xticks(s.n_train_mean); ax.set_xticklabels([f'{v:.0f}' for v in s.n_train_mean]); ax.minorticks_off()
        ax.set_title(title, loc='left', fontsize=10, fontweight='bold'); ax.set_xlabel('Training cell lines (log scale)', fontsize=8.5)
        ax.set_ylabel('Squared standardized error', fontsize=8.5); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, frameon=False)
    ax = axes[2]
    e = e2.sort_values('n_train_mean')
    ax.plot(e.n_train_mean, e['C0'], ':', marker='o', color=S.COLORS['C0'], label='C0')
    for pol in ('Family', 'Exact'):
        ax.plot(e.n_train_mean, e[f'{pol} (locally chosen)'], '-', marker=S.MARKERS[pol], color=S.COLORS[pol], label=f'{pol}, chosen locally')
        ax.plot(e.n_train_mean, e[f'{pol} (GDSC-chosen)'], '--', marker=S.MARKERS[pol], mfc='white', color=S.COLORS[pol], label=f'{pol}, chosen in GDSC')
    ax.set_xscale('log'); ax.set_xticks(e.n_train_mean); ax.set_xticklabels([f'{v:.0f}' for v in e.n_train_mean]); ax.minorticks_off()
    ax.set_title('Transport of measurement choices', loc='left', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{LABEL} training cell lines (log scale)', fontsize=8.5); ax.tick_params(labelsize=8); ax.legend(fontsize=7, frameon=False)
    for a in axes:
        a.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.005, 'Inner-CV selector. Measurements chosen in the GDSC example are fixed; only intercept and slope are refitted in the external data.',
             fontsize=7.5, color='#555')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / f'{DS}_external.{ext}', dpi=300)


def main():
    summary, contrasts = e1_tables()
    summary.to_csv(HERE / f'{DS}_summary.csv', index=False); contrasts.to_csv(HERE / f'{DS}_contrasts.csv', index=False)
    flags = S.flags(summary, contrasts)
    crit = criteria(summary, contrasts)
    e2 = e2_table(); e2.to_csv(HERE / f'{DS}_transport_summary.csv', index=False)
    (HERE / f'{DS}_flags.json').write_text(json.dumps({'replication_criteria': crit, 'flags': flags}, indent=1, ensure_ascii=False,
                                                      default=str), encoding='utf-8')
    figure(summary, contrasts, e2)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(summary.drop(columns=['runs']).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print(contrasts.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print(e2.T.to_string(float_format=lambda v: f'{v:.4g}'))
    print(json.dumps(crit, indent=1, ensure_ascii=False, default=str))
    for k, v in flags.items():
        print(k, v.get('by_n', v.get('signs_by_n')), v.get('crossover_observed', ''), v.get('clear_crossover', ''))


if __name__ == '__main__':
    main()
