#!/usr/bin/env python3
"""Export every number the redrawn figures need (figures_v2) into plain CSV files, from the frozen outputs only.
The plotting code (written separately) must read ONLY these CSV files, so that no figure recomputes or retypes a number.
Output: manuscript_ldh/figures_v2/data/*.csv  (one file per panel; a README lists the columns)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; MS = HERE.parent; W = MS.parent; R = W / 'task21_final'; OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
for d in ['task21_final', 'task25_newdata', 'task27_ukhls']:
    sys.path.insert(0, str(W / d))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
import clhls_lib as C  # noqa: E402
import ukhls_menu as U  # noqa: E402
import ukhls_lib as K  # noqa: E402
M.L.low_priority()
L = M.L
SEL = json.load(open(W / 'task19_dm2/selected_all.json')); SCN = json.load(open(W / 'task20_chns/selected_cn.json'))
SCL = json.load(open(W / 'task25_newdata/selected_clhls.json')); SUK = json.load(open(W / 'task27_ukhls/selected_ukhls.json'))
CVD = json.load(open(R / 'selected_cvd.json'))
E = {c: json.load(open(R / f't21_{c}_endpoints.json')) for c in ['NH3', 'GH', 'I', 'CHNS', 'CLHLS', 'CLHLS80', 'UKHLS']}
POOL = json.load(open(R / 't21_pooled_final.json'))
NAME = {'NH3': 'NHANES III', 'GH': 'NHANES 2011–14', 'I': 'NHANES 2015–16', 'CHNS': 'CHNS', 'CLHLS': 'CLHLS (65–79 y)', 'UKHLS': 'Understanding Society'}
w = lambda df, f: df.to_csv(OUT / f, index=False)  # noqa: E731

# ---------- F1: cohort cards (study design) ----------
cards = [
    dict(order=0, role='training', cohort='NHANES 1999–2010', country='USA', years='1999–2010', n=16153, deaths=2081, menu_analytes=34, prices='US Medicare 2026',
         death_ascertainment='National Death Index linkage', follow_up='9 years', analysed='development'),
    dict(order=1, role='primary', cohort='NHANES III', country='USA', years='1988–94', n=E['NH3']['n'] if 'n' in E['NH3'] else 5792, deaths=893, menu_analytes=34,
         prices='US Medicare 2026', death_ascertainment='National Death Index linkage', follow_up='9 years', analysed='preliminary validation, then final plan'),
    dict(order=2, role='primary', cohort='NHANES 2011–14', country='USA', years='2011–14', n=5917, deaths=300, menu_analytes=33, prices='US Medicare 2026',
         death_ascertainment='National Death Index linkage', follow_up='5 years', analysed='preliminary validation, then final plan'),
    dict(order=3, role='primary', cohort='CHNS', country='China', years='2009', n=4542, deaths=227, menu_analytes=15, prices='Shandong public prices 2026',
         death_ascertainment='Death year or survey wave of first report', follow_up='about 6 years', analysed='preliminary validation, then final plan'),
    dict(order=4, role='primary', cohort='CLHLS (65–79 y)', country='China', years='2012, 2014', n=E['CLHLS']['n'], deaths=E['CLHLS']['deaths'], menu_analytes=14,
         prices='Shandong public prices 2026', death_ascertainment='Verified death dates', follow_up='5 years', analysed='new cohort, final plan'),
    dict(order=5, role='primary', cohort='Understanding Society', country='UK', years='2010–12', n=E['UKHLS']['n'], deaths=E['UKHLS']['deaths'], menu_analytes=13,
         prices='US Medicare 2026 (scenario)', death_ascertainment='Survey wave in which death was reported', follow_up='about 9 years', analysed='new cohort, final plan'),
    dict(order=6, role='supplementary', cohort='NHANES 2015–16', country='USA', years='2015–16', n=3017, deaths=100, menu_analytes=34, prices='US Medicare 2026',
         death_ascertainment='National Death Index linkage', follow_up='3 years', analysed='final plan (supplementary, not pooled)')]
w(pd.DataFrame(cards), 'F1C_cohort_cards.csv')
w(pd.DataFrame([dict(item='CMP as panel (14 analytes)', usd=L.PANELS['CMP'][1]),
                dict(item='CMP analytes priced one by one', usd=round(sum(L.CHEM[a][2] for a in L.PANELS['CMP'][0]), 2))]), 'F1A_cmp_price.csv')
w(pd.DataFrame([dict(schedule='Australia (A$)', n_items=k, fee=v) for k, v in zip([1, 2, 3, 4, 5], [9.70, 11.65, 13.65, 15.65, 17.70])]
               + [dict(schedule='Japan (points)', n_items=k, fee=v) for k, v in [(5, 93), (6, 93), (7, 93), (8, 99), (9, 99), (10, 103), (11, 103), (12, 103)]]),
  'F1A_tiers.csv')
w(pd.DataFrame([dict(output='Matched panel (USA)', items='CMP + CBC + phosphorus', usd=32.41, share_of_routine=0.64),
                dict(output='Target-aware panel (USA)', items='CMP + CBC + urine ACR + GGT + phosphorus', usd=50.57, share_of_routine=0.996),
                dict(output='Routine bundle (reference)', items='CMP + CBC + lipids + HbA1c', usd=50.77, share_of_routine=1.0),
                dict(output='Ordering chart', items='clinical information -> CMP -> CBC -> urine ACR', usd=None, share_of_routine=None)]), 'F1B_outputs.csv')

# ---------- F2A: cost and C of panels ----------
KEEP = {'age+sex only': ('Age and sex only', 'reference line'), 'BMP': ('BMP', 'standard bundle'), 'CBC': ('CBC', 'standard bundle'), 'CMP': ('CMP', 'standard bundle'),
        'LIPID': ('Lipids', 'standard bundle'), 'CMP+CBC': ('CMP+CBC', 'standard bundle'), 'routine': ('Routine bundle', 'routine'),
        'CBC+BMP (IMRS-type)': ('CBC+BMP (Intermountain-type)', 'established score'), 'PhenoAge analytes': ('PhenoAge analytes (refitted)', 'established score'),
        'matched': ('Matched panel', 'matched'), 'aware@routine': ('Target-aware panel', 'target-aware'), 'all analytes': ('All 34 analytes', 'standard bundle')}
for c, f in [('NH3', 'F2A_cost_C_NHANES3.csv')]:
    p = pd.read_csv(R / f't21_{c}_panels.csv').set_index('panel')
    w(pd.DataFrame([dict(panel=lab, category=cat, usd=p.US[k], C=p.C[k]) for k, (lab, cat) in KEEP.items() if k in p.index]), f)

# ---------- F2B/C: forest plots ----------
share = {'NH3': 32.41 / 50.77, 'GH': 32.41 / 50.77, 'CHNS': 43.8 / 74.8, 'UKHLS': 27.10 / 50.77, 'CLHLS': 44.8 / 62.3}
ashare = {'NH3': 50.57 / 50.77, 'GH': 50.57 / 50.77, 'CHNS': 74.8 / 74.8, 'UKHLS': 49.76 / 50.77, 'CLHLS': 60.8 / 62.3}
rows = []
for h in ['H1', 'H2']:
    for c, v in POOL[h]['per_cohort'].items():
        rows.append(dict(hypothesis=h, cohort=NAME[c], kind='cohort', metric='AUC' if c in ('CHNS', 'UKHLS') else 'C', est=v['est'], lo=v['lo'], hi=v['hi'],
                         matched_price_share=share[c], aware_price_share=ashare[c]))
    p = POOL[h]['pooled']; rows.append(dict(hypothesis=h, cohort='Pooled (random effects)', kind='pooled', metric='', est=p['est'], lo=p['lo'], hi=p['hi'],
                                             I2=p['I2'], pi_lo=p['pi_lo'], pi_hi=p['pi_hi']))
    k = 'dC matched - routine' if h == 'H1' else 'dC aware@routine - routine'
    rows.append(dict(hypothesis=h, cohort='NHANES 2015–16 (supplementary)', kind='supplementary', metric='C', est=E['I']['point'][k], lo=E['I']['boot'][k][0],
                     hi=E['I']['boot'][k][1], matched_price_share=32.41 / 50.77, aware_price_share=50.57 / 50.77))
w(pd.DataFrame(rows), 'F2BC_forest.csv')

# ---------- F3A: items chosen by each matched panel ----------
CMP = set(L.PANELS['CMP'][0]); CBC = set(M.CBC2)
def items(feats, cmp_panel=True):
    f = set(feats); out = {}
    if cmp_panel and f & CMP:
        k = len(f & CMP); out['CMP'] = 'full panel' if k == len(CMP) else f'{k} measured CMP analytes'
        f -= CMP
    if f & CBC:
        out['CBC'] = 'full count' if CBC <= f else 'blood count indices measured'
        f -= CBC
    for a in sorted(f):
        out[a] = 'single test'
    return out
cols = [('USA', 'US Medicare', SEL['US']['matched']['items'], None), ('Australia', 'Australian MBS', SEL['AU']['matched']['items'], None),
        ('Japan', 'Japanese fee schedule', SEL['JP']['matched']['items'], None), ('China, CHNS menu', 'Shandong prices', None, SCN['matched']['features']),
        ('China, CLHLS menu', 'Shandong prices', None, SCL['matched']['features']), ('UK menu', 'US Medicare (scenario)', None, SUK['matched']['features']),
        ('Cardiovascular death (USA)', 'US Medicare', CVD['matched_prevent_full']['items'], None)]
PRICE = {'USA': ('US$32.41', 'US$50.77'), 'Australia': ('A$41.45', 'A$69.30'), 'Japan': ('¥4960', '¥5450'), 'China, CHNS menu': ('CN¥43.8', 'CN¥74.8'),
         'China, CLHLS menu': ('CN¥44.8', 'CN¥62.3'), 'UK menu': ('US$27.10', 'US$50.77'), 'Cardiovascular death (USA)': ('US$21.63', 'US$47.67 (PREVENT full set)')}
mat = []
for col, sched, it, feats in cols:
    fs = M.features_of(it) if it is not None else feats
    for item, how in items(fs, cmp_panel=not col.startswith('China')).items():
        mat.append(dict(column=col, price_schedule=sched, item=item, how=how, panel_price=PRICE[col][0], reference_price=PRICE[col][1]))
w(pd.DataFrame(mat), 'F3A_panel_items.csv')

# ---------- F3B: CoAI ----------
P24 = json.load(open(W / 'task24_coai/t24_pooled.json')); rows = []
for c in ['NH3', 'GH', 'UKHLS']:
    e = json.load(open(W / f'task24_coai/t24_{c}_endpoints.json')); pan = pd.read_csv(W / f'task24_coai/t24_{c}_panels.csv').set_index('panel')
    for budget, ours, coai in [('matched-panel budget', 'ours matched', 'CoAI @matched price'), ('routine budget', 'ours target-aware', 'CoAI @routine price')]:
        k = f'd {ours} - {coai}'
        rows.append(dict(budget=budget, cohort=NAME[c], metric=e['metric'], est=e['point'][k], lo=e['boot'][k][0], hi=e['boot'][k][1], ours_price=pan.price[ours],
                         coai_price_as_panels=pan.price[coai], coai_tests=pan.features[coai]))
for h, budget in [('H5', 'matched-panel budget'), ('H6', 'routine budget')]:
    p = P24[h]['pooled']; rows.append(dict(budget=budget, cohort='Pooled (random effects)', metric='', est=p['est'], lo=p['lo'], hi=p['hi'], I2=p['I2']))
w(pd.DataFrame(rows), 'F3B_coai.csv')
w(pd.DataFrame([dict(menu='US menu', cmp_analytes=14, singles_usd=66.51, panel_usd=10.56, ratio=round(66.51 / 10.56, 1)),
                dict(menu='UK menu', cmp_analytes=6, singles_usd=29.68, panel_usd=10.56, ratio=round(29.68 / 10.56, 1))]), 'F3B_pricing_error.csv')

# ---------- F3C: cardiovascular death ----------
LAB = [('none', 'Clinical variables only'), ('PCE/SCORE2 lipids', 'PCE/SCORE2 lipids'), ('PREVENT base', 'PREVENT base'), ('PREVENT full', 'PREVENT full'),
       ('CVD aware@PREVENT-full price', 'Target-aware panel')]
rows = []
for c in ['NH3', 'GH']:
    d = E[c]['cvd']
    for k, lab in LAB:
        rows.append(dict(cohort=NAME[c], cvd_deaths=d['events'], panel=lab, usd=d['US'][k], C=d['C'][k]))
w(pd.DataFrame(rows), 'F3C_cvd.csv')

# ---------- F4: ordering rule ----------
rule = json.load(open(W / 'task22_adaptive/rule_t22.json'))
w(pd.DataFrame([dict(step='Clinical information only', next_test='CMP', next_fee_usd=19.90, risk_low=rule['chart']['30000']['clinical only'][0][0],
                     risk_high=rule['chart']['30000']['clinical only'][0][1]),
                dict(step='After CMP', next_test='CBC', next_fee_usd=7.77, risk_low=rule['chart']['30000']['CMP'][0][0], risk_high=rule['chart']['30000']['CMP'][0][1]),
                dict(step='After CMP and CBC', next_test='urine ACR', next_fee_usd=10.96, risk_low=rule['chart']['30000']['CMP+CBC'][0][0],
                     risk_high=rule['chart']['30000']['CMP+CBC'][0][1])]), 'F4A_chart.csv')
rows, stop = [], []
for c in ['NH3', 'GH', 'I']:
    e = json.load(open(W / f'task22_adaptive/t22_{c}_endpoints.json')); ct = e['contrasts']
    for rule_name, key in [('Value-of-information rule', 'VOI V=30000 - routine'), ('Earlier rule', 'current rule (Task 19/21, eps 0.02) - routine')]:
        s_ = ct[key]['share_routine']; m_ = ct[key]['missed_per1000']
        rows.append(dict(cohort=NAME[c], rule=rule_name, cost_share=1 + s_['point'], cost_share_lo=1 + s_['ci'][0], cost_share_hi=1 + s_['ci'][1],
                         extra_missed_per1000=m_['point'], extra_missed_lo=m_['ci'][0], extra_missed_hi=m_['ci'][1],
                         net_benefit_diff=ct[key]['NB']['point'] if c == 'NH3' else None,
                         net_benefit_lo=ct[key]['NB']['ci'][0] if c == 'NH3' else None, net_benefit_hi=ct[key]['NB']['ci'][1] if c == 'NH3' else None))
    for st, v in e['end_states_main'].items():
        stop.append(dict(cohort=NAME[c], stopped_after=st, share=v))
w(pd.DataFrame(rows), 'F4BC_cost_missed.csv'); w(pd.DataFrame(stop), 'F4A_stop_shares.csv')

# ---------- F5B: local recalibration ----------
rows = []
for c in ['NH3', 'I', 'CHNS', 'GH', 'UKHLS']:
    s = json.load(open(W / f'task23_local/t23_{c}_summary.json'))
    for x in s['cells']:
        rows.append(dict(cohort=NAME[c], import_OE=s['whole_cohort_import']['matched']['OE'], panel=x['panel'], n_local=x['n'],
                         improved_share=x['B recalibrate']['better_calibrated_share'], reps_used=x['reps_used'], reading=x['reading_B']))
w(pd.DataFrame(rows), 'F5B_recalibration.csv')

# ---------- S1: subgroups ----------
rows = []
for c in ['NH3', 'GH']:
    d = pd.read_csv(R / f't21_{c}_subgroups.csv')
    for _, r in d.iterrows():
        if pd.notna(r.get('dC matched - routine')):
            rows.append(dict(cohort=NAME[c], group=r['group'], n=r['n'], deaths=r['deaths'], metric='C', est=r['dC matched - routine'], lo=r['dC CI lo'], hi=r['dC CI hi']))
for _, r in pd.read_csv(R / 't21_UKHLS_subgroups.csv').iterrows():
    rows.append(dict(cohort=NAME['UKHLS'], group=r['group'], n=r['n'], deaths=r['deaths'], metric='AUC', est=r['dAUC matched - routine'], lo=r['dAUC CI lo'], hi=r['dAUC CI hi']))
GROUP = {'age40-59': 'Age 40–59', 'age60-79': 'Age 60–79', 'income <1.3x poverty': 'Income <1.3× poverty line',
         'income 1.3-3.5x': 'Income 1.3–3.5×', 'income >=3.5x': 'Income ≥3.5×'}
for r_ in rows: r_['group'] = GROUP.get(r_['group'], r_['group'])
w(pd.DataFrame(rows), 'S1_subgroups.csv')

# ---------- S4: cross-domain benchmark ----------
g = pd.read_csv(W / 'task17_gates/summary/g1_by_source.csv'); g['aware_ev'] = g.reuse_ev + g.aware_minus_reuse
w(g[['k', 'predictor', 'source', 'aware_side', 'aware_ev', 'best_reuse', 'reuse_ev', 'aware_minus_reuse']], 'S4_benchmark.csv')

# ---------- S2: participant flow ----------
tr = E['NH3']['train']; gh = E['GH']['test']; cl = E['CLHLS']['test']; uk = E['UKHLS']['test']
flow = [('NHANES 1999–2010 (training)', [('In survey files', sum(c['in_file'] for c in tr)), ('Aged 40–79', sum(c['age_40_79'] for c in tr)),
                                         ('Eligible for mortality linkage', sum(c['mortality_eligible'] for c in tr)), ('All 34 analytes measured', sum(c['complete'] for c in tr))]),
        ('NHANES III', [('In survey files', E['NH3']['test']['in_file']), ('Aged 40–79', E['NH3']['test']['age_40_79']),
                        ('Eligible for mortality linkage', E['NH3']['test']['mortality_eligible']), ('All analytes measured', E['NH3']['test']['complete'])]),
        ('NHANES 2011–14', [('In survey files', sum(c['in_file'] for c in gh)), ('Aged 40–79', sum(c['age_40_79'] for c in gh)),
                            ('Eligible for mortality linkage', sum(c['mortality_eligible'] for c in gh)), ('All analytes measured (no CRP in these cycles)', sum(c['complete'] for c in gh))]),
        ('NHANES 2015–16 (supplementary)', [('In survey files', E['I']['test']['in_file']), ('Aged 40–79', E['I']['test']['age_40_79']),
                            ('Eligible for mortality linkage', E['I']['test']['mortality_eligible']), ('All analytes measured', E['I']['test']['complete'])]),
        ('CHNS 2009', [('With 2009 blood sample', E['CHNS']['test']['blood_2009']), ('Aged 40–79', E['CHNS']['test']['age_40_79']),
                       ('Death or 2015 survey status known', E['CHNS']['test']['known_status']), ('All 15 shared analytes measured', E['CHNS']['test']['complete'])]),
        ('CLHLS 2012/2014', [('With blood sample, linked to survey (2012, 2014)', cl['linked_2012'] + cl['linked_2014']), ('Distinct people (earliest draw)', cl['persons']),
                             ('Aged 65 and over', cl['age_65plus']), ('With follow-up after baseline', cl['dated_follow_up']), ('All 14 shared analytes measured', cl['complete']),
                             ('Aged 65–79 (primary)', E['CLHLS']['n']), ('Aged 80 and over (reported separately)', cl['complete'] - E['CLHLS']['n'])]),
        ('Understanding Society 2010–12', [('With nurse blood sample', uk['blood_records']), ('Aged 40–79', uk['age_40_79']),
                                           ('9-year status known', uk['known_9y_status']), ('All 13 shared analytes measured', uk['complete'])])]
w(pd.DataFrame([dict(cohort=c, step_order=i, step=s_, n=int(v)) for c, steps in flow for i, (s_, v) in enumerate(steps)]), 'S2_flow.csv')

# ---------- F5A calibration bins and S3 decision curves (refit the frozen models; no re-selection) ----------
def km(tt, ee, h):
    o = np.argsort(tt); tt, ee = tt[o], ee[o]; s = 1.0
    for v in np.unique(tt[ee & (tt <= h)]):
        s *= 1 - (ee & (tt == v)).sum() / (tt >= v).sum()
    return 1 - s


xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr)
US = {'Routine bundle': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'Matched panel': M.features_of(SEL['US']['matched']['items']),
      'Target-aware panel': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
cal, dca = [], []
def fit_risk(Zt, f, Zv, H):
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Zt[cols].to_numpy(), ttr, etr)
    return T.risk_at(m, T.h0_at(m, Zt[cols].to_numpy(), ttr, etr, H), Zv[cols].to_numpy())
def add_cal(name, risks, t, e, H, binary, groups=10):
    for k, r in risks.items():
        q = pd.qcut(r, groups, labels=False, duplicates='drop')
        for b in np.unique(q):
            s_ = q == b
            cal.append(dict(cohort=name, panel=k, bin=int(b), predicted=float(r[s_].mean()), observed=float(e[s_].mean() if binary else km(t[s_], e[s_], H)), n=int(s_.sum())))
for c, H, load in [('NH3', 108, lambda: M.load_nh3()[0]), ('GH', 60, lambda: pd.concat([T.load_gh(cc)[0] for cc in 'GH']))]:
    x = load(); Z = M.transform(x, ref=xtr); t, e = T.outcome_at(x, H); risks = {k: fit_risk(Ztr, f, Z, H) for k, f in US.items()}
    add_cal(NAME[c], risks, t, e, H, False)
    kn = e | (t >= H); prev = e[kn].mean(); W_ = kn.sum()
    for thr in np.round(np.linspace(0.01, 0.30, 59), 4):
        for k, r in risks.items():
            hi = r >= thr; dca.append(dict(cohort=NAME[c], threshold=thr, strategy=k, net_benefit=((hi & e & kn).sum() - (hi & ~e & kn).sum() * thr / (1 - thr)) / W_))
        dca.append(dict(cohort=NAME[c], threshold=thr, strategy='Treat all', net_benefit=prev - (1 - prev) * thr / (1 - thr)))
    print(c, flush=True)
xc, _ = T.load_chns(); Zcn = T.transform_sub(xtr, xtr, T.SHARED); Zc = T.transform_sub(xc, xtr, T.SHARED)
add_cal('CHNS', {k: fit_risk(Zcn, f, Zc, 72) for k, f in {'Routine bundle': T.CN_ROUTINE_SHARED, 'Matched panel': SCN['matched']['features'],
                                                         'Target-aware panel': SCN['aware_at_routine']['features']}.items()},
        None, xc.died.to_numpy().astype(bool), 72, True)
xu, _ = K.load_ukhls(); Zuk = T.transform_sub(xtr, xtr, U.MENU); Zu = T.transform_sub(xu, xtr, U.MENU)
add_cal(NAME['UKHLS'], {k: fit_risk(Zuk, f, Zu, 108) for k, f in {'Routine bundle': SUK['routine']['features'], 'Matched panel': SUK['matched']['features'],
                                                                  'Target-aware panel': SUK['aware_at_routine']['features']}.items()},
        None, xu.died.to_numpy().astype(bool), 108, True)
xl, _ = C.load_clhls(); kk = (xl.age_group == '65-79').to_numpy(); Zcl = T.transform_sub(xtr, xtr, C.MENU); Zl = T.transform_sub(xl[kk], xtr, C.MENU); tl, el = C.outcome(xl[kk])
add_cal(NAME['CLHLS'], {k: fit_risk(Zcl, f, Zl, 60) for k, f in {'Routine bundle': C.ROUTINE_CL, 'Matched panel': SCL['matched']['features'],
                                                                 'Target-aware panel': SCL['aware_at_routine']['features']}.items()}, tl, el, 60, False, groups=5)
w(pd.DataFrame(cal), 'F5A_calibration_bins.csv'); w(pd.DataFrame(dca), 'S3_decision_curves.csv')
oe = [dict(cohort=NAME[c], matched_OE=v1, routine_OE=v2) for c, v1, v2 in
      [('NH3', 0.97, 0.94), ('GH', 0.87, 0.88), ('CHNS', 1.05, 1.02), ('UKHLS', 0.85, 0.84), ('CLHLS', 1.32, 1.31)]]
w(pd.DataFrame(oe), 'F5A_OE_labels.csv')
print('written', sorted(p.name for p in OUT.glob('*.csv')))
