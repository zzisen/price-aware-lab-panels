#!/usr/bin/env python3
"""Task 8 (plan_real_cost.md): build the real-cost tasks in the Task 5 task format.
Writes tasks.json, task_counts.csv and matrices/<task>.csv into this folder. Run with .env312."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
EXT = WORK / 'external_data'
T5 = WORK / 'task5_validation'
NHANES_GRID = [25, 50, 100, 200, 500, 1000, None]


# ---- Part 1: drug responses as candidate measurements ---------------------------------------------------------------
DRUG_TASKS = ['GDSC1_T1', 'GDSC1_T2', 'GDSC1_T3', 'GDSC2_T1', 'GDSC2_T2', 'GDSC2_T3', 'GDSC2_T4',
              'PRISM_T1', 'PRISM_T2', 'PRISM_T3', 'CTRPv2_T1', 'CTRPv2_T2', 'NCI60_T1', 'NCI60_T2', 'NCI60_T3', 'NCI60_T4']
COVERAGE, MIN_ROWS = 0.95, 100


def drug_tasks():
    long = pd.read_csv(EXT / 'pharmacodb' / 'experiments_long.csv').dropna(subset=['AAC'])
    mapping = pd.read_csv(T5 / 'cell_line_mapping.csv').dropna(subset=['unit'])
    long = long.merge(mapping, on=['dataset', 'cell_line'])
    tasks5 = {t['task']: t for t in json.loads((T5 / 'tasks.json').read_text())}
    out, counts = [], []
    for i, name in enumerate(DRUG_TASKS):
        t = tasks5[name]
        m = pd.read_csv(T5 / 'matrices' / f'{name}.csv').set_index('unit')
        sub = long[(long.dataset == name.split('_')[0]) & long.unit.isin(m.index)]
        w = sub.pivot_table(index='unit', columns='compound', values='AAC', aggfunc='mean')
        w = w.drop(columns=[c for c in t['targets'] if c in w.columns])
        keep = w.columns[w.notna().mean() >= COVERAGE]
        x = w[keep].dropna()
        x.columns = [f'cand_{c}' for c in x.columns]
        units = x.index.intersection(m.index)
        frame = pd.concat([x.loc[units], m.loc[units, t['targets']]], axis=1).dropna()
        counts.append({'task': f'DRUG_{name}', 'units': len(frame), 'candidates': len(keep), 'included': len(frame) >= MIN_ROWS})
        if len(frame) < MIN_ROWS:
            continue
        frame.reset_index().to_csv(HERE / 'matrices' / f'DRUG_{name}.csv', index=False)
        out.append({'task': f'DRUG_{name}', 'targets': t['targets'], 'families': t['families'], 'code': 800 + i,
                    'cost_unit': 'drug', 'source_task': name})
    return out, counts


# ---- Part 2: NHANES -------------------------------------------------------------------------------------------------
NH_TARGETS = {'LBXGH': 'metabolic', 'LBDHDD': 'metabolic', 'LBXFER': 'iron_inflammation', 'LBXHSCRP': 'iron_inflammation',
              'LBDTIB': 'iron_inflammation', 'LBXVIDMS': 'nutrition', 'LBXBPB': 'exposure', 'LBXBCD': 'exposure', 'LBXTHG': 'exposure'}
NH_TARGET_FILE = {'LBXGH': 'GHB_J', 'LBDHDD': 'HDL_J', 'LBXFER': 'FERTIN_J', 'LBXHSCRP': 'HSCRP_J', 'LBDTIB': 'FETIB_J',
                  'LBXVIDMS': 'VID_J', 'LBXBPB': 'PBCD_J', 'LBXBCD': 'PBCD_J', 'LBXTHG': 'PBCD_J'}


def nhanes_task():
    d = EXT / 'nhanes_2017_2018'
    read = lambda f: pd.read_sas(d / f'{f}.xpt', format='xport').set_index('SEQN')
    demo = read('DEMO_J')[['RIDAGEYR']]
    cbc = read('CBC_J').drop(columns=['LBXNRBC'], errors='ignore')
    bio = read('BIOPRO_J'); bio = bio[[c for c in bio if c.startswith('LBX')]]
    tg = pd.concat([read(NH_TARGET_FILE[k])[[k]] for k in NH_TARGETS], axis=1)
    f = demo[demo.RIDAGEYR >= 20].join(cbc, how='inner').join(bio, how='inner').join(tg, how='inner').drop(columns='RIDAGEYR').dropna()
    for c in f:
        pos = f[c][f[c] > 0]
        f[c] = np.log(f[c] + pos.min() / 2)
    f.index = f.index.astype(int).astype(str); f.index.name = 'unit'
    f.reset_index().to_csv(HERE / 'matrices' / 'NHANES.csv', index=False)
    task = {'task': 'NHANES', 'targets': list(NH_TARGETS), 'families': list(NH_TARGETS.values()), 'code': 900,
            'grid': NHANES_GRID, 'cost_unit': 'clfs', 'candidates_cbc': list(cbc.columns), 'candidates_bio': list(bio.columns)}
    return task, {'task': 'NHANES', 'units': len(f), 'candidates': cbc.shape[1] + bio.shape[1], 'included': True}


# ---- Part 3: Saad 2022 human LV trabeculae --------------------------------------------------------------------------
SAAD_INDEX_SHEETS = ['Fdev', 'TTP', 'Fdia', 'dF dt', 'dF.dt.Fdev', 'neg dF.dt', 'neg dF.dt.Fdev', 'RT50', 'RT90']
SAAD_TARGET_IDX = ['Fdev', 'TTP', 'RT50']


def saad_block(path, sheet):
    """First data block of a wide sheet: {condition label: {heart: value}}."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    hdr = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.fullmatch(r'NF1').any())
    hearts = {j: str(v) for j, v in raw.iloc[hdr].items() if re.fullmatch(r'(NF|FNH|FH)\d+', str(v))}
    label_col = next(j for j in range(raw.shape[1]) if raw.iloc[hdr + 1:hdr + 3, j].astype(str).str.contains('Hz').any())
    out = {}
    for i in range(hdr + 1, len(raw)):
        lab = raw.iat[i, label_col]
        if not isinstance(lab, str) or 'Hz' not in lab:
            break
        out[lab] = {h: pd.to_numeric(raw.iat[i, j], errors='coerce') for j, h in hearts.items()}
    return out


def saad_task():
    d = EXT / 'human_lv_trabeculae_saad2022'
    iso = {s: saad_block(d / 'S1_Dataset.xlsx', s) for s in SAAD_INDEX_SHEETS}
    freq = {s: saad_block(d / 'S2_Dataset.xlsx', s) for s in SAAD_TARGET_IDX}
    base_label = '0.5HzIso0'
    cols = {}
    for s in SAAD_INDEX_SHEETS:
        cols[f'base_{s}'] = iso[s][base_label]
    targets, families = [], []
    for s in SAAD_TARGET_IDX:
        for lab, v in freq[s].items():
            if lab != '0.5HzL100':
                cols[f'{s}@{lab}'] = v; targets.append(f'{s}@{lab}'); families.append('frequency')
        for lab, v in iso[s].items():
            if lab != base_label:
                cols[f'{s}@{lab}'] = v; targets.append(f'{s}@{lab}'); families.append('isoproterenol')
    f = pd.DataFrame(cols).dropna()
    f.index.name = 'unit'
    f.reset_index().to_csv(HERE / 'matrices' / 'SAAD_LV.csv', index=False)
    task = {'task': 'SAAD_LV', 'targets': targets, 'families': families, 'code': 910, 'design': 'leave_one_out',
            'cost_unit': 'minutes', 'minutes_per_condition': 2.0}
    return task, {'task': 'SAAD_LV', 'units': len(f), 'candidates': len(SAAD_INDEX_SHEETS), 'included': True}


# ---- Part 4: CiPA ---------------------------------------------------------------------------------------------------
CHANNELS = ['hERG', 'INa', 'INaL', 'ICaL', 'Ito', 'IKs', 'IK1']


def hill(c, ic50, h):
    return 100.0 / (1.0 + (ic50 / c) ** h)


def log_ic50(g):
    c, b = g.conc.to_numpy(float), g.block.to_numpy(float)
    top = c.max()
    if np.nanmean(b[c == top]) < 50:          # plan: <50% block at the highest tested concentration -> 10 x top
        return np.log10(10 * top)
    try:
        p, _ = curve_fit(hill, c, b, p0=[np.median(c), 1.0], bounds=([1e-6, 0.1], [1e9, 10]), maxfev=20000)
        return np.log10(p[0])
    except RuntimeError:
        return np.nan


def cipa_task():
    b = pd.read_csv(EXT / 'cipa' / 'drug_block.csv').dropna(subset=['block'])
    x = b.groupby(['drug', 'channel']).apply(log_ic50, include_groups=False).unstack()
    x = x[[c for c in CHANNELS if c in x.columns]]
    x.columns = [f'log10IC50_{c}' for c in x.columns]
    d = pd.read_excel(EXT / 'cipa' / 'blinova2018' / 'Blinova_etal_2018_data.xlsx')
    d['drug'] = d.Drug_Name.str.lower().str.replace(r'^d,\s*l[ ,]*', '', regex=True).str.strip()
    y = d.groupby(['drug', 'Cell_type', 'conc']).ddFPDc.median().unstack(['Cell_type', 'conc'])
    y.columns = [f'ddFPDc_{ct}_c{c}' for ct, c in y.columns]
    f = x.join(y, how='inner')
    targets = sorted(c for c in y.columns if f[c].notna().all())
    f = f[list(x.columns) + targets].dropna()
    f.index.name = 'unit'
    f.reset_index().to_csv(HERE / 'matrices' / 'CIPA.csv', index=False)
    task = {'task': 'CIPA', 'targets': targets, 'families': [t.split('_')[1] for t in targets], 'code': 920,
            'design': 'leave_one_out', 'cost_unit': 'channel'}
    return task, {'task': 'CIPA', 'units': len(f), 'candidates': x.shape[1], 'included': True}


def main():
    (HERE / 'matrices').mkdir(exist_ok=True)
    tasks, counts = drug_tasks()
    for fn in (nhanes_task, saad_task, cipa_task):
        t, c = fn(); tasks.append(t); counts.append(c)
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1), encoding='utf-8')
    pd.DataFrame(counts).to_csv(HERE / 'task_counts.csv', index=False)
    print(pd.DataFrame(counts).to_string())


if __name__ == '__main__':
    main()
