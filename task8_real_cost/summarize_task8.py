#!/usr/bin/env python3
"""Task 8 summary (plan_real_cost.md, COST_SOURCES.md). Run with .env312 after run_all.sh and per_target.py.
Writes summary/*.csv and summary/task8_endpoints.json."""
import itertools
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
T5 = HERE.parent / 'task5_validation'
sys.path.insert(0, str(T5))
import dev_eval as D  # noqa: E402

OUT = HERE / 'summary'
FIXED = ['Shared', 'Family', 'Global3', 'Exact']
KEYS = ['evaluation', 'outer_seed', 'fold', 'n_label', 'rep']
THRESHOLDS = (0.5, 0.75)
TASKS = json.loads((HERE / 'tasks.json').read_text(encoding='utf-8'))


def joined(name, base=HERE):
    v = pd.read_csv(base / 'validation' / f'{name}.csv'); s = pd.read_csv(base / 'standard' / f'{name}.csv')
    b = pd.read_csv(base / 'budget' / f'{name}.csv')
    for f in (v, s, b):
        f['n_label'] = f.n_label.astype(str)
    assert (v[KEYS].to_numpy() == s[KEYS].to_numpy()).all() and (v[KEYS].to_numpy() == b[KEYS].to_numpy()).all()
    return pd.concat([v, s[[c for c in s if c not in v]], b[[c for c in b if c not in v and c not in s]]], axis=1)


def by_n(f, cols):
    rows = []
    for n_label, h in f.groupby('n_label'):
        risk = {k: h[k].sum() / h.cells.sum() for k in cols}
        best = min(risk[k] for k in FIXED)
        rows.append({'n_label': n_label, 'n_mean': h.n_train.mean(), 'k_R4': h.k_R4.mean(),
                     **{f'risk_{k}': x for k, x in risk.items()}, **{f'regret_{k}': risk[k] - best for k in cols},
                     **{f'ev_{k}': 1 - risk[k] / risk['C0'] for k in cols},
                     **{f'nmeas_{k}': h[f'nmeas_{k}'].mean() for k in ['Lasso', 'MultiTaskLasso', 'Ridge', 'MTLrelaxed_matched']}})
    return pd.DataFrame(rows)


# ---- Part 1 ---------------------------------------------------------------------------------------------------------
def part1():
    cols = ['C0'] + FIXED + ['R4', 'Ridge', 'MultiTaskLasso', 'Lasso', 'g_MTLrelaxed_matched']
    per_n, crit = [], []
    for t in [t for t in TASKS if t['task'].startswith('DRUG_')]:
        f = joined(t['task'])
        p = by_n(f, cols); p['task'] = t['task']; per_n.append(p)
        a = p[p.n_label == 'all'].iloc[0]
        n_cand = int(f.nmeas_Ridge.iloc[0])
        ev4, evr = a.ev_R4, a.ev_Ridge
        ratio_ok = (ev4 >= 0.5 * evr) if evr > 0 else (ev4 >= evr)    # Ridge explaining nothing: R4 must be no worse
        crit.append({'task': t['task'], 'candidates': n_cand, 'ev_R4': ev4, 'ev_Ridge': evr, 'ev_MTL': a.ev_MultiTaskLasso,
                     'k_R4': a.k_R4, 'k_MTL': a.nmeas_MultiTaskLasso, 'ev_ratio': ev4 / evr if evr > 0 else np.nan,
                     'meas_fraction': a.k_R4 / n_cand, 'value_met': bool(ratio_ok and a.k_R4 <= 0.1 * n_cand)})
    per_n = pd.concat(per_n); crit = pd.DataFrame(crit)
    mx = per_n.groupby('task')[['regret_R4', 'regret_g_MTLrelaxed_matched']].max()
    d = mx.regret_R4 - mx.regret_g_MTLrelaxed_matched; d = d[np.abs(d) > 1e-12]
    sign = {'wins_R4': int((d < 0).sum()), 'losses_R4': int((d > 0).sum()), 'p_two_sided': float(binomtest(int((d < 0).sum()), len(d), 0.5).pvalue)}
    per_n.to_csv(OUT / 'part1_drug_per_n.csv', index=False); crit.to_csv(OUT / 'part1_drug_criterion.csv', index=False)
    curve = per_n.groupby('n_label')[['ev_R4', 'ev_Ridge', 'ev_MultiTaskLasso', 'ev_Lasso', 'k_R4', 'nmeas_MultiTaskLasso', 'nmeas_Lasso', 'nmeas_Ridge']].mean()
    return {'tasks': len(crit), 'tasks_value_met': int(crit.value_met.sum()), 'needed': int(np.ceil(2 / 3 * len(crit))),
            'value_established': bool(crit.value_met.sum() >= np.ceil(2 / 3 * len(crit))),
            'R4_vs_matched_relaxed_MTL_sign_test': sign, 'curve_by_n': curve.round(4).to_dict()}


# ---- Part 2: NHANES costs -------------------------------------------------------------------------------------------
CLFS = pd.read_csv(HERE.parent / 'external_data' / 'clfs_2026q4' / 'unz' / 'PUF_CLFS_CY2026_Q4V1.csv', skiprows=4, dtype=str)
CLFS = CLFS[CLFS.MOD.isna()]; PRICE = dict(zip(CLFS.HCPCS, CLFS.RATE.astype(float)))
BIO_CPT = {'LBXSATSI': ['84460'], 'LBXSAL': ['82040'], 'LBXSAPSI': ['84075'], 'LBXSASSI': ['84450'], 'LBXSC3SI': ['82374'],
           'LBXSBU': ['84520'], 'LBXSCLSI': ['82435'], 'LBXSCK': ['82550'], 'LBXSCR': ['82565'], 'LBXSGB': ['84155', '82040'],
           'LBXSGL': ['82947'], 'LBXSGTSI': ['82977'], 'LBXSIR': ['83540'], 'LBXSLDSI': ['83615'],
           'LBXSOSSI': ['84295', '82947', '84520'], 'LBXSPH': ['84100'], 'LBXSKSI': ['84132'], 'LBXSNASI': ['84295'],
           'LBXSTB': ['82247'], 'LBXSCA': ['82310'], 'LBXSCH': ['82465'], 'LBXSTP': ['84155'], 'LBXSTR': ['84478'], 'LBXSUA': ['84550']}
PANELS = {'80053': ['82040', '82247', '82310', '82374', '82435', '82565', '82947', '84075', '84132', '84155', '84295', '84460', '84450', '84520'],
          '80048': ['82310', '82374', '82435', '82565', '82947', '84132', '84295', '84520'],
          '80076': ['82040', '82247', '82248', '84075', '84155', '84460', '84450'],
          '80069': ['82040', '82310', '82374', '82435', '82565', '82947', '84100', '84132', '84295', '84520']}
CBC_DIFF = ('LBXLYPCT', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT', 'LBXBAPCT', 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO')
TARGET_CPT = {'LBXGH': '83036', 'LBDHDD': '83718', 'LBXFER': '82728', 'LBXHSCRP': '86141', 'LBDTIB': '83550',
              'LBXVIDMS': '82306', 'LBXBPB': '83655', 'LBXBCD': '82300', 'LBXTHG': '83825'}
VENIPUNCTURE = PRICE['36415']


def billing(features, cbc):
    """Cheapest CLFS billing that covers the analytes behind the chosen features (panels or single codes)."""
    cost = 0.0
    chosen_cbc = [f for f in features if f in cbc]
    if chosen_cbc:
        cost += PRICE['85025'] if any(f in CBC_DIFF for f in chosen_cbc) else PRICE['85027']
    need = sorted({c for f in features if f in BIO_CPT for c in BIO_CPT[f]})
    best = sum(PRICE[c] for c in need)
    for k in range(1, len(PANELS) + 1):
        for combo in itertools.combinations(PANELS, k):
            covered = set().union(*(PANELS[p] for p in combo))
            best = min(best, sum(PRICE[p] for p in combo) + sum(PRICE[c] for c in need if c not in covered))
    return cost + best


def part2():
    t = next(t for t in TASKS if t['task'] == 'NHANES')
    pt = pd.read_csv(HERE / 'per_target' / 'NHANES.csv')
    pt['price'] = pt.target.map(lambda k: PRICE[TARGET_CPT[k]])
    feats = json.loads((HERE / 'per_target' / 'NHANES_r4_features.json').read_text(encoding='utf-8'))
    r4_cost = [VENIPUNCTURE + billing(f, t['candidates_cbc']) if f else 0.0 for f in feats]
    ridge_cost = VENIPUNCTURE + billing(t['candidates_cbc'] + t['candidates_bio'], t['candidates_cbc'])
    util = {}
    for f in (HERE.parent / 'external_data' / 'cms_utilization_2024').glob('nat_*.json'):
        for r in json.loads(f.read_text()):
            if r['Place_Of_Srvc'] == 'O':
                util[r['HCPCS_Cd']] = (float(r['Tot_Srvcs']), float(r['Avg_Mdcr_Pymt_Amt']))
    pt['medicare_2024_services'] = pt.target.map(lambda k: util.get(TARGET_CPT[k], (np.nan, np.nan))[0])
    pt['medicare_2024_payment_usd'] = pt.target.map(lambda k: np.prod(util.get(TARGET_CPT[k], (np.nan, np.nan))))
    for th in THRESHOLDS:
        pt[f'R4_ge_{th}'] = pt.ev_R4 >= th; pt[f'Ridge_ge_{th}'] = pt.ev_Ridge >= th
    pt.to_csv(OUT / 'part2_nhanes_targets.csv', index=False)
    f = joined('NHANES'); per_n = by_n(f, ['C0'] + FIXED + ['R4', 'Ridge', 'MultiTaskLasso', 'Lasso', 'g_MTLrelaxed_matched'])
    per_n.to_csv(OUT / 'part2_nhanes_per_n.csv', index=False)
    return {'participants': int(len(pd.read_csv(HERE / 'matrices' / 'NHANES.csv', usecols=['unit']))),
            'scenarioB_R4_mean_cost_usd': float(np.mean(r4_cost)), 'scenarioB_ridge_cost_usd': ridge_cost,
            'venipuncture_usd': VENIPUNCTURE,
            'targets': pt[['target', 'price', 'ev_R4', 'ev_Ridge', 'medicare_2024_services', 'medicare_2024_payment_usd']].round(4).to_dict('records')}


# ---- Part 3: cardiac ------------------------------------------------------------------------------------------------
MINUTES = {'HumanAtrial_T1': 15.0, 'AWINDA_MOUSE': 7.0, 'RLC1_RAT': 2.0, 'SAAD_LV': 2.0, 'TANNER_RAT_TRABECULA': 0.2,
           'RADBILL_HUMAN_PACING': 1.0}


def condition_of(system, target):
    if system == 'HumanAtrial_T1':
        return str(target)[str(target).index('(') + 1:str(target).index(')')]       # 'SS F (0.1 ATP)' -> '0.1 ATP'
    if system == 'SAAD_LV':
        return target.split('@', 1)[1]
    if system == 'RLC1_RAT':
        return target.split('_', 1)[0]
    if system == 'RADBILL_HUMAN_PACING':
        return target.split('_', 1)[0]
    return str(target)


def native_cardiac(system):
    f = pd.read_csv(T5 / 'validation' / f'cardiac_{system}.csv')
    g = f.groupby(['target', 'policy']).loss.mean().unstack() if system != 'AWINDA_MOUSE' else \
        f.groupby(['target', 'unit', 'policy']).loss.mean().groupby(['target', 'policy']).mean().unstack()
    r4 = f[f.R4].groupby('target').loss.mean() if system != 'AWINDA_MOUSE' else f[f.R4].groupby(['target', 'unit']).loss.mean().groupby('target').mean()
    return pd.DataFrame({'target': g.index.astype(str), 'reduction_R4': (1 - r4 / g['C0']).to_numpy(), 'metric': 'native'})


def radbill():
    recs = [r for r in pickle.load(open(T5 / 'cache' / 'radbill.pkl', 'rb')) if r['n_label'] == 'all']
    choice = {r['outer_unit']: D.argmin_rule(D.averaged_units(r), D.candidates(r, True)) for r in recs}
    c = pd.read_csv(HERE.parent / 'task4_adaptive' / 'radbill_rules_cells.csv')
    c = c[c.m_train == c.m_train.max()]
    c['R4'] = [getattr(row, {'C0': 'C0', 'Shared': 'SHARED', 'Family': 'FAMILY', 'Exact': 'EXACT'}[choice[row.outer_participant]]) for row in c.itertuples()]
    g = c.groupby('target')[['C0', 'R4']].mean()
    return pd.DataFrame({'target': g.index, 'reduction_R4': (1 - g.R4 / g.C0).to_numpy(), 'metric': 'native'})


def part3():
    rows = []
    for sysname, frame in [('HumanAtrial_T1', pd.read_csv(HERE / 'per_target' / 'HumanAtrial_T1.csv')),
                           ('SAAD_LV', pd.read_csv(HERE / 'per_target' / 'SAAD_LV.csv'))]:
        for r in frame.itertuples():
            rows.append({'system': sysname, 'target': r.target, 'reduction_R4': r.ev_R4, 'reduction_Ridge': r.ev_Ridge, 'metric': 'explained variance'})
    for sysname in ['AWINDA_MOUSE', 'RLC1_RAT', 'TANNER_RAT_TRABECULA']:
        for r in native_cardiac(sysname).itertuples():
            rows.append({'system': sysname, 'target': r.target, 'reduction_R4': r.reduction_R4, 'metric': 'error reduction (native)'})
    for r in radbill().itertuples():
        rows.append({'system': 'RADBILL_HUMAN_PACING', 'target': r.target, 'reduction_R4': r.reduction_R4,
                     'metric': 'error reduction (native); development data'})
    pt = pd.DataFrame(rows)
    pt['condition'] = [condition_of(s, t) for s, t in zip(pt.system, pt.target)]
    pt.to_csv(OUT / 'part3_cardiac_targets.csv', index=False)
    summ = []
    for s, h in pt.groupby('system'):
        cond = h.groupby('condition').reduction_R4.min()
        summ.append({'system': s, 'targets': len(h), 'conditions': len(cond), 'minutes_per_condition': MINUTES[s],
                     'minutes_all_conditions': len(cond) * MINUTES[s],
                     **{f'conditions_all_targets_ge_{th}': int((cond >= th).sum()) for th in THRESHOLDS},
                     **{f'minutes_saved_ge_{th}': float((cond >= th).sum() * MINUTES[s]) for th in THRESHOLDS},
                     'median_reduction_R4': float(h.reduction_R4.median())})
    summ = pd.DataFrame(summ); summ.to_csv(OUT / 'part3_cardiac_summary.csv', index=False)
    return summ.round(3).to_dict('records')


# ---- Part 4: CiPA ---------------------------------------------------------------------------------------------------
USD_PER_CHANNEL = (16 * 8.76, 28 * 8.76)


def part4():
    pt = pd.read_csv(HERE / 'per_target' / 'CIPA.csv')
    feats = json.loads((HERE / 'per_target' / 'CIPA_r4_features.json').read_text(encoding='utf-8'))
    k = np.array([len(f) for f in feats], float)
    pt.to_csv(OUT / 'part4_cipa_targets.csv', index=False)
    return {'drugs': len(feats), 'mean_channels_R4': float(k.mean()), 'channels_all': 7,
            'usd_per_channel_low_high': USD_PER_CHANNEL,
            'usd_R4_low_high': (float(k.mean() * USD_PER_CHANNEL[0]), float(k.mean() * USD_PER_CHANNEL[1])),
            'usd_all_low_high': (7 * USD_PER_CHANNEL[0], 7 * USD_PER_CHANNEL[1]),
            'ev_R4_overall': float(1 - pt.sse_R4.sum() / pt.sse_C0.sum()), 'ev_Ridge_overall': float(1 - pt.sse_Ridge.sum() / pt.sse_C0.sum()),
            'targets': pt[['target', 'ev_R4', 'ev_Ridge']].round(4).to_dict('records')}


def main():
    OUT.mkdir(exist_ok=True)
    out = {'part1_drug': part1(), 'part2_nhanes': part2(), 'part3_cardiac': part3(), 'part4_cipa': part4()}
    (OUT / 'task8_endpoints.json').write_text(json.dumps(out, indent=1, default=float), encoding='utf-8')
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
