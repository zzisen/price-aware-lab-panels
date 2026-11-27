"""CiPA-28 summary against the frozen criteria (plan_cipa28.md)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
USD = (16 * 8.76, 28 * 8.76)
out = {}
for name in ['CIPA28', 'CIPA28_T12']:
    pt = pd.read_csv(HERE / 'per_target' / f'{name}.csv')
    k = np.array([len(f) for f in json.loads((HERE / 'per_target' / f'{name}_r4_features.json').read_text())], float)
    feats = json.loads((HERE / 'per_target' / f'{name}_r4_features.json').read_text())
    v = pd.read_csv(HERE / 'validation' / f'{name}.csv'); s = pd.read_csv(HERE / 'standard' / f'{name}.csv')
    ev = lambda col, f=v: float(1 - f[col].sum() / v.C0.sum())
    ev4, evr = float(1 - pt.sse_R4.sum() / pt.sse_C0.sum()), float(1 - pt.sse_Ridge.sum() / pt.sse_C0.sum())
    crit = [ev4 > 0, (ev4 >= 0.5 * evr) if evr > 0 else (ev4 >= evr), k.mean() <= 2]
    out[name] = {'drugs': len(k), 'ev_R4': ev4, 'ev_Ridge': evr,
                 'ev_other_from_standard': {c: ev(c, s) for c in ['Lasso', 'ElasticNet', 'MultiTaskLasso', 'Ridge'] if c in s},
                 'mean_channels_R4': float(k.mean()), 'choices_R4': v.choice_R4.value_counts().to_dict(),
                 'channel_sets': pd.Series([','.join(f) for f in feats]).value_counts().to_dict(),
                 'criteria': crit, 'value_seen': bool(all(crit)),
                 'usd_R4': [float(k.mean() * u) for u in USD], 'usd_all4': [4 * u for u in USD],
                 'targets': pt[['target', 'ev_R4', 'ev_Ridge']].round(4).to_dict('records')}
perm = np.array([1 - (lambda v: v.R4.sum() / v.C0.sum())(pd.read_csv(HERE / 'validation' / f'CIPA28_P{k:03d}.csv')) for k in range(200)])
obs = 1 - (lambda v: v.R4.sum() / v.C0.sum())(pd.read_csv(HERE / 'validation' / 'CIPA28.csv'))
out['permutation'] = {'n': 200, 'observed_ev_R4_validation': float(obs), 'p_one_sided': float((1 + (perm >= obs).sum()) / 201),
                      'perm_quantiles': np.quantile(perm, [0.5, 0.9, 0.95, 0.99]).round(4).tolist()}
(HERE / 'cipa28_endpoints.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
print(json.dumps(out, indent=1, default=float))
