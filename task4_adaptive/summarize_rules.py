#!/usr/bin/env python3
"""Task 4 summaries: risks, regret against the hindsight-best fixed resolution, choice frequencies, the prespecified
criteria K1-K3 (plan_adaptive_rules.md) and the figure. Run with .env312 after adaptive_rules.py and radbill_rules.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIXED = {'Shared': 'Shared', 'Family': 'Family', 'Exact≤3': 'Global3', 'Exact': 'Exact'}
RULES = {'R1 nested CV': 'R1', 'R2 corrected': 'R2', 'R2 + C0': 'R2c0', 'S nested': 'Snest', 'S boot': 'Sboot'}
ANALYSES = [('gdsc1', 'repeated_random', 'GDSC1, cell-line splits'), ('gdsc1', 'held_out_tissue', 'GDSC1, tissue held out'),
            ('gdsc2', 'repeated_random', 'GDSC2, cell-line splits'), ('gdsc2', 'held_out_tissue', 'GDSC2, tissue held out'),
            ('gdsc2_new', 'repeated_random', 'GDSC2 new lines'), ('prism', 'repeated_random', 'PRISM, cell-line splits'),
            ('prism', 'held_out_tissue', 'PRISM, tissue held out'), ('radbill', None, 'Radbill pacing')]
COLORS = {'C0': '#8c8c8c', 'Shared': '#1b9e77', 'Family': '#d95f02', 'Exact≤3': '#7570b3', 'Exact': '#e7298a',
          'R1 nested CV': '#444444', 'R2 corrected': '#000000', 'S nested': '#a6761d'}


def load(dataset, design):
    """Long frame: one row per (n_label, rep, fold-unit) with loss sums and cells, plus choices."""
    if dataset == 'radbill':
        cells = pd.read_csv(HERE / 'radbill_rules_cells.csv')
        ch = pd.read_csv(HERE / 'radbill_rules_choices.csv')
        cols = {'C0': 'C0', 'Shared': 'SHARED', 'Family': 'FAMILY', 'Exact': 'EXACT', 'R1': 'R1', 'R2': 'R2', 'R2c0': 'R2c0'}
        g = cells.groupby(['outer_participant', 'm_train', 'rep'])
        f = g.size().rename('cells').reset_index()
        for k, c in cols.items():
            f[f'loss_sum_{k}'] = g[c].apply(lambda s: s.sum(min_count=1)).values
        f = f.merge(ch, on=['outer_participant', 'm_train', 'rep'])
        f = f.rename(columns={'m_train': 'n'})
        f['n_label'] = f.n.astype(str); f.loc[f.rep < 0, 'n_label'] = 'all'
        f['choice_R2'] = f.choice_R2.str.title(); f['choice_R1'] = f.choice_R1.str.title(); f['choice_R2c0'] = f.choice_R2c0.str.title()
        return f, {'Shared': 'Shared', 'Family': 'Family', 'Exact': 'Exact'}, ['R1', 'R2', 'R2c0']
    tag = f'{dataset}_{design}' if dataset != 'gdsc2_new' else 'gdsc2_new'
    f = pd.read_csv(HERE / f'{tag}_rules.csv')
    f = f.rename(columns={'n_test_cells': 'cells'})
    f['n'] = f.n_train
    f['n_label'] = f.n_label.astype(str)
    return f, FIXED, ['R1', 'R2', 'R2c0', 'Snest', 'Sboot']


def order(f):
    return f.groupby('n_label').n.mean().sort_values().index.tolist()


def risk(h, col):
    ok = h[f'loss_sum_{col}'].notna()
    return h.loc[ok, f'loss_sum_{col}'].sum() / h.loc[ok, 'cells'].sum()


def summarize(dataset, design, label):
    f, fixed, rules = load(dataset, design)
    rows, cons, freq = [], [], []
    for n_label in order(f):
        h = f[f.n_label == n_label]
        r = {'analysis': label, 'n_label': n_label, 'n_mean': h.n.mean(), 'C0': risk(h, 'C0')}
        r.update({p: risk(h, c) for p, c in fixed.items()})
        best = min(fixed, key=lambda p: r[p]); r['hindsight_best'] = best
        for rule in rules:
            r[rule] = risk(h, rule)
        for p in list(fixed) + rules:
            r[f'regret_{p}'] = r[p] - r[best]
        if dataset == 'radbill':   # R1 exists only on the first 20 reps (and the full set): paired comparison there
            h20 = h[h.loss_sum_R1.notna()]
            r['R2_on_R1_reps'] = risk(h20, 'R2'); r['best_on_R1_reps'] = min(risk(h20, c) for c in fixed.values())
            r['regret_R1_paired'] = risk(h20, 'R1') - r['best_on_R1_reps']; r['regret_R2_paired'] = r['R2_on_R1_reps'] - r['best_on_R1_reps']
        rows.append(r)
        for a, b in [('R2', 'Exact'), ('R2', 'R1'), ('R2', 'Shared'), ('R2', 'Family'), ('R1', 'Exact'), ('Snest', 'R2'), ('Snest', 'Family'), ('R2c0', 'C0')]:
            ca, cb = (fixed.get(a, a), fixed.get(b, b))
            if f'loss_sum_{ca}' not in h or f'loss_sum_{cb}' not in h:
                continue
            hh = h[h[f'loss_sum_{ca}'].notna() & h[f'loss_sum_{cb}'].notna()]
            d = hh[f'loss_sum_{ca}'] - hh[f'loss_sum_{cb}']
            per_rep = d.groupby(hh.rep).sum() / hh.cells.groupby(hh.rep).sum()
            cons.append({'analysis': label, 'n_label': n_label, 'contrast': f'{a} − {b}', 'mean': d.sum() / hh.cells.sum(),
                         'rep_2.5': per_rep.quantile(.025) if len(per_rep) > 1 else np.nan,
                         'rep_97.5': per_rep.quantile(.975) if len(per_rep) > 1 else np.nan, 'n_reps': len(per_rep)})
        for rule in ['R1', 'R2', 'R2c0'] + (['Snest', 'Sboot'] if 'Snest' in rules else []):
            vc = h[f'choice_{rule}'].dropna().value_counts(normalize=True)
            freq.append({'analysis': label, 'n_label': n_label, 'rule': rule, **vc.round(4).to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(cons), pd.DataFrame(freq), fixed


def criteria(s, c, fq, fixed):
    labels = s.n_label.tolist()
    r2 = fq[fq.rule == 'R2'].set_index('n_label').reindex(labels)
    sh = r2.get('Shared', pd.Series(0, index=labels)).fillna(0)
    modal = r2.drop(columns=['analysis', 'rule']).fillna(0).idxmax(axis=1)
    k1 = bool(modal.iloc[0] == 'Shared' and modal.iloc[-1] != 'Shared' and all(np.diff(sh.values) <= 1e-12))
    ce = c[c.contrast == 'R2 − Exact'].set_index('n_label').reindex(labels)
    k2 = bool(ce['rep_97.5'].iloc[0] < 0 and (s.R2 <= s.Exact + 1e-12).all())
    maxreg = {p: float(s[f'regret_{p}'].max()) for p in list(fixed) + ['R1', 'R2']}
    if 'regret_R1_paired' in s:
        maxreg['R1'] = float(s.regret_R1_paired.max()); maxreg['R2 (R1 reps)'] = float(s.regret_R2_paired.max())
    others = [p for p in list(fixed) + ['R1']]
    k3 = bool(all(maxreg['R2'] < maxreg[p] for p in others))
    return {'K1 tracks the crossover': k1, 'K2 beats naive Exact at small n': k2, 'K3 minimax regret': k3,
            'R2 Shared share by n': dict(zip(labels, sh.round(3))), 'R2 modal choice by n': dict(zip(labels, modal)),
            'max regret': {k: round(v, 4) for k, v in maxreg.items()}}


def figure(results):
    show = ['GDSC1, cell-line splits', 'GDSC1, tissue held out', 'GDSC2, cell-line splits', 'Radbill pacing']
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for j, label in enumerate(show):
        s, c, fq, fixed = results[label]
        x = np.arange(len(s))
        ax = axes[0, j]
        for p in fixed:
            ax.plot(x, s[f'regret_{p}'], '-', marker='o', ms=4, lw=1.2, color=COLORS[p], alpha=.75, label=p)
        ax.plot(x, s['regret_R1'] if 'regret_R1_paired' not in s else s['regret_R1_paired'], '--', marker='s', ms=4, lw=1.6, color=COLORS['R1 nested CV'], label='R1 nested CV')
        ax.plot(x, s['regret_R2'], '-', marker='D', ms=5, lw=2.4, color=COLORS['R2 corrected'], label='R2 corrected')
        if 'regret_Snest' in s:
            ax.plot(x, s['regret_Snest'], ':', marker='^', ms=4, lw=1.6, color=COLORS['S nested'], label='S partial pooling')
        ax.axhline(0, color='#bbbbbb', lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:.0f}' for v in s.n_mean], fontsize=8)
        ax.set_title(label, loc='left', fontsize=9.5, fontweight='bold')
        ax.set_ylabel('Regret vs best fixed resolution' if j == 0 else '', fontsize=8.5); ax.tick_params(labelsize=8)
        ax.set_ylim(bottom=-0.01)
        if j == 0:
            ax.legend(fontsize=7, frameon=False)
        ax2 = axes[1, j]
        for rule, dx, hatch in (('R1', -0.2, None), ('R2', 0.2, '///')):
            share = fq[fq.rule == rule].set_index('n_label').reindex(s.n_label).fillna(0)
            bottom = np.zeros(len(s))
            for p, col in fixed.items():
                v = share[col].values if col in share else np.zeros(len(s))
                ax2.bar(x + dx, v, bottom=bottom, color=COLORS[p], width=.38, label=p if rule == 'R1' else None,
                        edgecolor='white', linewidth=.8, hatch=hatch)
                bottom += v
        for xi in x:
            ax2.text(xi - 0.2, 1.01, 'R1', ha='center', fontsize=6, color='#555'); ax2.text(xi + 0.2, 1.01, 'R2', ha='center', fontsize=6, color='#555')
        ax2.set_xticks(x); ax2.set_xticklabels([f'{v:.0f}' for v in s.n_mean], fontsize=8)
        ax2.set_xlabel('Training units', fontsize=8.5); ax2.set_ylim(0, 1.06)
        ax2.set_ylabel('Choice share (R1 left, R2 right)' if j == 0 else '', fontsize=8.5); ax2.tick_params(labelsize=8)
        if j == 0:
            ax2.legend(fontsize=7, frameon=False, loc='upper right')
    for a in axes.flat:
        a.spines[['top', 'right']].set_visible(False)
    fig.text(0.01, 0.005, 'Regret = outer-test risk minus the lowest risk among fixed resolutions at that n (chosen after evaluation). '
             'Radbill R1 is on the first 20 draws per size.', fontsize=7.5, color='#555')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(HERE / f'adaptive_rules.{ext}', dpi=300)


def main():
    results, allcrit = {}, {}
    S, C, F = [], [], []
    for dataset, design, label in ANALYSES:
        tag = f'{dataset}_{design}' if dataset not in ('gdsc2_new', 'radbill') else dataset
        if not (HERE / ('radbill_rules_cells.csv' if dataset == 'radbill' else f'{tag}_rules.csv')).exists():
            print('missing', label); continue
        s, c, fq, fixed = summarize(dataset, design, label)
        results[label] = (s, c, fq, fixed)
        allcrit[label] = criteria(s, c, fq, fixed)
        S.append(s); C.append(c); F.append(fq)
    pd.concat(S).to_csv(HERE / 'rules_summary.csv', index=False)
    pd.concat(C).to_csv(HERE / 'rules_contrasts.csv', index=False)
    pd.concat(F).to_csv(HERE / 'rules_choice_frequencies.csv', index=False)
    (HERE / 'rules_criteria.json').write_text(json.dumps(allcrit, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    if all(k in results for k in ['GDSC1, cell-line splits', 'GDSC1, tissue held out', 'GDSC2, cell-line splits', 'Radbill pacing']):
        figure(results)
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        for s in S:
            print(s.drop(columns=[k for k in s if k.startswith('regret_') or k.endswith('_reps')]).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print(pd.concat(C).to_string(index=False, float_format=lambda v: f'{v:.4g}'))
        print(pd.concat(F).to_string(index=False))
    print(json.dumps(allcrit, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
