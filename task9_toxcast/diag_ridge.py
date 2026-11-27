"""POST HOC technical diagnosis (after the frozen decision rule was computed): why do the multi-measurement
methods return explained variance around -1e8? Identifies the near-constant candidate and recomputes Ridge
without it. Does NOT replace the pre-specified decision; it only explains an implausible number."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L
L.HERE = HERE
import validate_generic as VG
from validate_standard import RIDGE_ALPHAS
from target_resolution import group_folds

ann = pd.read_excel(HERE.parent / 'external_data' / 'toxcast_v4_3' / 'assay_annotations_invitrodb_v4_3_AUG2024.xlsx',
                    usecols=['aeid', 'aid', 'assay_component_endpoint_name'])
nm = dict(zip(ann.aeid.astype(int), ann.assay_component_endpoint_name))
rows = []
for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')):
    name = t['task']
    L.GRID = [None]
    task, x, y, fam = L.load_task(name)
    feats = [c for c in pd.read_csv(HERE / 'matrices' / f'{name}.csv', nrows=0).columns
             if c not in task['targets'] + ['unit']]
    sd = x.std(axis=0, ddof=1)
    worst = int(np.argmin(sd))
    aeid = int(feats[worst].removeprefix('cand_'))
    groups = np.arange(len(x))
    keep = np.array([i for i in range(x.shape[1]) if i != worst])
    sse = {'C0': 0.0, 'Ridge_all': 0.0, 'Ridge_drop1': 0.0}
    for ev, seed, fold, n, r, sub, held, gr in VG.subsamples(task, x):
        xs, ys = x[sub], y[sub]
        xm = xs.mean(0); xsd = np.where(xs.std(0, ddof=1) == 0, 1.0, xs.std(0, ddof=1))
        ym = ys.mean(0); ysd = np.where(ys.std(0, ddof=1) == 0, 1.0, ys.std(0, ddof=1))
        yt = (y[held] - ym) / ysd
        idx = np.arange(len(xs)); cv = [(np.setdiff1d(idx, h), h) for h in group_folds(gr[sub], 5, seed + fold + 1)]
        sse['C0'] += float((yt ** 2).sum())
        for lab, cols in (('Ridge_all', np.arange(x.shape[1])), ('Ridge_drop1', keep)):
            m = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv).fit(((xs - xm) / xsd)[:, cols], (ys - ym) / ysd)
            sse[lab] += float(((m.predict(((x[held] - xm) / xsd)[:, cols]) - yt) ** 2).sum())
    rows.append({'task': name, 'worst_aeid': aeid, 'endpoint': nm.get(aeid, '?'), 'sd': float(sd[worst]),
                 'activity_rate': float((x[:, worst] >= 0.9).mean()),
                 'ev_Ridge_all': 1 - sse['Ridge_all'] / sse['C0'],
                 'ev_Ridge_drop_that_one': 1 - sse['Ridge_drop1'] / sse['C0']})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(HERE / 'summary' / 'posthoc_ridge_diagnosis.csv', index=False)
