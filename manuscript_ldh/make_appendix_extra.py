#!/usr/bin/env python3
"""Appendix items required for submission (TRIPOD+AI 20a, 22; calibration plots; decision curves). Uses the frozen models and the cohorts already
analysed (NHANES III, 2011-14, 2015-16, CHNS, CLHLS, Understanding Society); nothing is re-selected. Outputs in manuscript_ldh/:
  figures/figureS2_flow.(png|pdf), appendix_flow.md            participant flow per cohort
  appendix_coefficients.md, model_coefficients.json            full model specification (TRIPOD+AI 22)
  figures/figureS3_calibration.(png|pdf)                       calibration by tenth of predicted risk (Kaplan-Meier observed)
  figures/figureS4_decision.(png|pdf)                          decision curves, NHANES III and NHANES 2011-14"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
HERE = Path(__file__).resolve().parent; W = HERE.parent
for d in ['task21_final', 'task25_newdata', 'task27_ukhls']:
    sys.path.insert(0, str(W / d))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
import t21lib as X  # noqa: E402
import clhls_lib as C  # noqa: E402
import ukhls_menu as U  # noqa: E402
import ukhls_lib as K  # noqa: E402
M.L.low_priority()
FIG = HERE / 'figures'; R = W / 'task21_final'
BLUE, ORANGE, GREEN, GREY, INK, INK2 = '#2a78d6', '#d9822b', '#2e8b57', '#8a8a85', '#1d1d1b', '#55554f'
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
SEL = json.load(open(W / 'task19_dm2' / 'selected_all.json')); SCN = json.load(open(W / 'task20_chns' / 'selected_cn.json'))
SCL = json.load(open(W / 'task25_newdata' / 'selected_clhls.json')); SUK = json.load(open(W / 'task27_ukhls' / 'selected_ukhls.json'))


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------- 1. participant flow ----------------
E = {c: json.load(open(R / f't21_{c}_endpoints.json')) for c in ['NH3', 'GH', 'I', 'CHNS', 'CLHLS']}
tr = E['NH3']['train']; gh = E['GH']['test']
flow = [('NHANES 1999–2010 (training)', [('In survey files', sum(c['in_file'] for c in tr)), ('Aged 40–79', sum(c['age_40_79'] for c in tr)),
                                         ('Eligible for mortality linkage', sum(c['mortality_eligible'] for c in tr)), ('All 34 analytes measured', sum(c['complete'] for c in tr))]),
        ('NHANES III', [('In survey files', E['NH3']['test']['in_file']), ('Aged 40–79', E['NH3']['test']['age_40_79']),
                        ('Eligible for mortality linkage', E['NH3']['test']['mortality_eligible']), ('All analytes measured', E['NH3']['test']['complete'])]),
        ('NHANES 2011–14', [('In survey files', sum(c['in_file'] for c in gh)), ('Aged 40–79', sum(c['age_40_79'] for c in gh)),
                            ('Eligible for mortality linkage', sum(c['mortality_eligible'] for c in gh)), ('All analytes measured*', sum(c['complete'] for c in gh))]),
        ('NHANES 2015–16', [('In survey files', E['I']['test']['in_file']), ('Aged 40–79', E['I']['test']['age_40_79']),
                            ('Eligible for mortality linkage', E['I']['test']['mortality_eligible']), ('All analytes measured', E['I']['test']['complete'])]),
        ('CHNS 2009', [('With 2009 blood sample', E['CHNS']['test']['blood_2009']), ('Aged 40–79', E['CHNS']['test']['age_40_79']),
                       ('Death or 2015 survey status known', E['CHNS']['test']['known_status']), ('All 15 shared analytes measured', E['CHNS']['test']['complete'])])]
cl = E['CLHLS']['test']
flow.append(('CLHLS 2012/2014', [('With blood sample, linked to survey†', cl['linked_2012'] + cl['linked_2014']), ('Distinct people (earliest draw)', cl['persons']),
                                 ('Aged 65 and over', cl['age_65plus']), ('With positive follow-up‡', cl['dated_follow_up']),
                                 ('All 14 shared analytes measured', cl['complete']),
                                 ('Aged 65–79 (primary) / 80 and over', f"{E['CLHLS']['n']} / {cl['complete'] - E['CLHLS']['n']}")]))
uk = json.load(open(R / 't21_UKHLS_endpoints.json'))['test']
flow.append(('Understanding Society 2010–13', [('With nurse blood sample', uk['blood_records']), ('Aged 40–79', uk['age_40_79']),
                                               ('9-year status known§', uk['known_9y_status']), ('All 13 shared analytes measured', uk['complete'])]))
md_ = ['# Participant flow (TRIPOD+AI item 20a)\n']
for name, steps in flow:
    sp = lambda v: f'{v:,}'.replace(',', ' ') if v >= 10000 else str(v)
    md_.append(f'**{name}**: ' + ' → '.join(f'{s} {sp(v)}' if isinstance(v, int) else f'{s} {v}' for s, v in steps) + '\n')
md_.append('\n*C-reactive protein was not measured in NHANES 2011–14, and panels containing it were not evaluated there. †The 2008 blood wave was not used, because its hs-CRP had '
           'no documented unit; people first sampled in 2008 entered at their 2012 or 2014 draw if they had one. ‡Follow-up after the baseline interview '
           '(people lost before the first follow-up wave, or with a death date before baseline, had none). §Death reported by the wave '
           'fielded nine waves after the nurse visit, or enumerated alive at that wave.\n')
(HERE / 'appendix_flow.md').write_text('\n'.join(md_), encoding='utf-8')
fig, axs = plt.subplots(1, len(flow), figsize=(17, 3.6))
for ax, (name, steps) in zip(axs, flow):
    ax.axis('off'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(name.replace(' 2010–13', '\n2010–13').replace(' (training)', '\n(training)'), fontsize=7.2, loc='center', color=INK)
    ys = np.linspace(0.88, 0.1, len(steps))
    for k, (y, (s, v)) in enumerate(zip(ys, steps)):
        ax.text(0.5, y, f'{s}\n{v:,}' if isinstance(v, int) else f'{s}\n{v}', ha='center', va='center', fontsize=6.2, color=INK,
                bbox=dict(boxstyle='round,pad=0.3', fc='#f4f3ef', ec='#c9c7bf', lw=0.6))
        if k < len(steps) - 1:
            ax.annotate('', (0.5, ys[k + 1] + 0.07), (0.5, y - 0.07), arrowprops=dict(arrowstyle='->', color=INK2, lw=0.7))
save(fig, 'figureS2_flow')

# ---------------- 2. coefficients ----------------
xtr, _ = M.load(list('ABCDEF')); ttr, etr = M.outcome(xtr); Ztr = M.transform(xtr)
off = {a: float(0.01 * xtr[a][xtr[a] > 0].median()) for a in M.FEATURES}
MODELS = {'US matched panel': (Ztr, M.features_of(SEL['US']['matched']['items'])), 'US target-aware panel': (Ztr, M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])),
          'US routine bundle': (Ztr, M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']))}
Zcn = T.transform_sub(xtr, xtr, T.SHARED); Zcl = T.transform_sub(xtr, xtr, C.MENU)
MODELS.update({'CHNS matched panel': (Zcn, SCN['matched']['features']), 'CHNS target-aware panel': (Zcn, SCN['aware_at_routine']['features']),
               'CHNS routine items': (Zcn, T.CN_ROUTINE_SHARED), 'CLHLS matched panel': (Zcl, SCL['matched']['features']),
               'CLHLS target-aware panel': (Zcl, SCL['aware_at_routine']['features']), 'CLHLS routine items': (Zcl, C.ROUTINE_CL)})
Zuk = T.transform_sub(xtr, xtr, U.MENU)
MODELS.update({'UK matched panel': (Zuk, SUK['matched']['features']), 'UK target-aware panel': (Zuk, SUK['aware_at_routine']['features']),
               'UK routine items': (Zuk, SUK['routine']['features'])})
VNAME = {'age': 'Age (years)', 'female': 'Female sex', 'acr': 'Urine albumin-to-creatinine ratio', 'albumin': 'Albumin', 'alp': 'Alkaline phosphatase',
         'alt': 'ALT', 'ast': 'AST', 'bilirubin': 'Total bilirubin', 'bun': 'Urea nitrogen', 'calcium': 'Calcium', 'chloride': 'Chloride',
         'co2': 'Bicarbonate', 'creatinine': 'Creatinine', 'crp': 'C-reactive protein', 'ggt': 'GGT', 'glucose': 'Glucose', 'gran_pct': 'Granulocytes (%)',
         'hba1c': 'HbA1c', 'hdl': 'HDL cholesterol', 'hgb': 'Haemoglobin', 'lymph_pct': 'Lymphocytes (%)', 'mcv': 'Mean cell volume',
         'mono_pct': 'Monocytes (%)', 'mpv': 'Mean platelet volume', 'phosphorus': 'Phosphorus', 'plt': 'Platelets', 'potassium': 'Potassium',
         'protein': 'Total protein', 'rdw': 'Red cell distribution width', 'sodium': 'Sodium', 'tc': 'Total cholesterol', 'tg': 'Triglycerides',
         'uric_acid': 'Uric acid', 'wbc': 'White cell count', 'ldh': 'LDH', 'iron': 'Iron'}
coef, md_ = {}, ['# Model coefficients (TRIPOD+AI item 22)\n',
                 'Ridge Cox models (penalty 1) fitted on NHANES 1999–2010 (n 16 153; 2081 deaths). Each analyte enters as z = ln(x + offset), with x in US conventional units (urine albumin-to-creatinine ratio: mg/g divided by 100) '
                 'and offset = 1% of the training median (listed). The linear predictor is the sum over variables of coefficient × (z − mean) / SD, with age in '
                 'years and female = 1 for women. Risk at t months = 1 − exp(−H0(t) × exp(linear predictor)); H0 is given at 3, 5 and 9 years (36, 60 and '
                 '108 months), the horizons used in the validation cohorts. Units are US conventional units (appendix A2); C-reactive protein is in mg/dL with '
                 'values below 0.25 set to 0.21. In CHNS the 9-year risk was used for an outcome observed over about 6 years. The sequential-ordering '
                 'models and chart are in the calculator file (model_v2.json) in the code repository. Variable codes match the code and the calculator.\n']


def H0_at(m, X, t, e, h):
    """Breslow baseline cumulative hazard at h months, computed as in Cox.fit (which stores it at 108 months)."""
    Z = (np.asarray(X, float) - m.m) / m.s; o = np.argsort(-t, kind='stable'); Z, t, e = Z[o], np.asarray(t)[o], np.asarray(e)[o]
    r = np.exp(Z @ m.b); S0 = np.cumsum(r)[np.searchsorted(-t, -t, side='right') - 1]
    return float((e / S0)[t <= h].sum())


for name, (Zf, f) in MODELS.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Zf[cols].to_numpy(), ttr, etr)
    H36, H60, H108 = (H0_at(m, Zf[cols].to_numpy(), ttr, etr, h) for h in (36, 60, 108))
    assert abs(H108 - m.H0) < 1e-10, (name, H108, m.H0)
    coef[name] = dict(columns=cols, offset={c: off[c] for c in cols if c in off}, mean=m.m.tolist(), sd=m.s.tolist(), coef=m.b.tolist(),
                      H0_36=H36, H0_60=H60, H0_108=float(m.H0))
    md_.append(f'## {name} (H0 at 3, 5 and 9 years = {H36:.5f}, {H60:.5f}, {m.H0:.5f})\n\n| Variable | Code | Offset | Mean | SD | Coefficient per SD |\n|---|---|---|---|---|---|')
    for c, mu, sd, b in zip(cols, m.m, m.s, m.b):
        o = f'{off[c]:.4g}' if (c in off and c not in M.BASE) else '—'
        md_.append(f"| {VNAME.get(c, c)} | {c} | {o} | {mu:.4f} | {sd:.4f} | {b:+.4f} |")
    md_.append('')
(HERE / 'appendix_coefficients.md').write_text('\n'.join(md_), encoding='utf-8'); json.dump(coef, open(HERE / 'model_coefficients.json', 'w'), indent=1)
print('coefficients written', flush=True)


# ---------------- 3. calibration and decision curves ----------------
def km(tt, ee, h):
    o = np.argsort(tt); tt, ee = tt[o], ee[o]; s = 1.0
    for u in np.unique(tt[ee & (tt <= h)]):
        s *= 1 - (ee & (tt == u)).sum() / (tt >= u).sum()
    return 1 - s


def nb(r, tt, ee, tau, H):
    kn = ee | (tt >= H); W_ = kn.sum(); hi = r >= tau
    return (hi & ee & kn).sum() / W_ - (hi & ~ee & kn).sum() / W_ * tau / (1 - tau)


US = {'Routine bundle': M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']), 'Matched panel': M.features_of(SEL['US']['matched']['items']),
      'Target-aware panel': M.features_of(SEL['US']['aware']['CMP+CBC+LIPID+A1C']['items'])}
COL = {'Routine bundle': GREY, 'Matched panel': BLUE, 'Target-aware panel': ORANGE}
cohorts = []
for name, H, load in [('NHANES III (9 y)', 108, lambda: M.load_nh3()[0]), ('NHANES 2011–14 (5 y)', 60, lambda: pd.concat([T.load_gh(c)[0] for c in 'GH'])),
                      ('NHANES 2015–16 (3 y)', 36, lambda: X.load_i()[0])]:
    x = load(); Z = M.transform(x, ref=xtr); t, e = T.outcome_at(x, H); risks = {}
    for k, f in US.items():
        cols = M.BASE + f; m = M.Cox(1.0).fit(Ztr[cols].to_numpy(), ttr, etr); risks[k] = T.risk_at(m, T.h0_at(m, Ztr[cols].to_numpy(), ttr, etr, H), Z[cols].to_numpy())
    cohorts.append((name, H, t, e, risks, False)); print(name, flush=True)
xc, _ = T.load_chns(); Zc = T.transform_sub(xc, xtr, T.SHARED); y = xc.died.to_numpy().astype(bool); risks = {}
for k, f in {'Routine bundle': T.CN_ROUTINE_SHARED, 'Matched panel': SCN['matched']['features'], 'Target-aware panel': SCN['aware_at_routine']['features']}.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Zcn[cols].to_numpy(), ttr, etr); risks[k] = T.risk_at(m, T.h0_at(m, Zcn[cols].to_numpy(), ttr, etr, 72), Zc[cols].to_numpy())
cohorts.append(('CHNS (~6 y; observed proportion)', 72, None, y, risks, True))
xl, _ = C.load_clhls(); k_ = (xl.age_group == '65-79').to_numpy(); Zl = T.transform_sub(xl[k_], xtr, C.MENU); tl, el = C.outcome(xl[k_]); risks = {}
for k, f in {'Routine bundle': C.ROUTINE_CL, 'Matched panel': SCL['matched']['features'], 'Target-aware panel': SCL['aware_at_routine']['features']}.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Zcl[cols].to_numpy(), ttr, etr); risks[k] = T.risk_at(m, T.h0_at(m, Zcl[cols].to_numpy(), ttr, etr, 60), Zl[cols].to_numpy())
cohorts.append(('CLHLS 65–79 (5 y; fifths)', 60, tl, el, risks, False))
xu, _ = K.load_ukhls(); Zu = T.transform_sub(xu, xtr, U.MENU); yu = xu.died.to_numpy().astype(bool); risks = {}
for k, f in {'Routine bundle': SUK['routine']['features'], 'Matched panel': SUK['matched']['features'], 'Target-aware panel': SUK['aware_at_routine']['features']}.items():
    cols = M.BASE + list(f); m = M.Cox(1.0).fit(Zuk[cols].to_numpy(), ttr, etr); risks[k] = T.risk_at(m, T.h0_at(m, Zuk[cols].to_numpy(), ttr, etr, 108), Zu[cols].to_numpy())
cohorts.append(('Understanding Society (~9 y; observed proportion)', 108, None, yu, risks, True))
fig, axs = plt.subplots(1, len(cohorts), figsize=(14.5, 2.9))
for ax, (name, H, t, e, risks, binary) in zip(axs, cohorts):
    top = 0
    for k, r in risks.items():
        g = 5 if name.startswith('CLHLS') else 10; q = pd.qcut(r, g, labels=False, duplicates='drop'); px, py = [], []
        for b in np.unique(q):
            s = q == b; px.append(r[s].mean()); py.append(e[s].mean() if binary else km(t[s], e[s], H))
        ax.plot(px, py, 'o-', ms=2.5, lw=1, color=COL[k], label=k); top = max(top, max(px), max(py))
    ax.plot([0, top * 1.05], [0, top * 1.05], color=INK2, lw=0.7, ls=(0, (3, 3)))
    ax.set_title(name, fontsize=7.5, loc='left'); ax.set_xlabel('Predicted risk'); ax.grid(color='#e5e3dc', lw=0.5)
axs[0].set_ylabel('Observed risk (Kaplan–Meier)'); axs[0].legend(frameon=False, fontsize=6)
save(fig, 'figureS3_calibration')
fig, axs = plt.subplots(1, 2, figsize=(7.5, 2.8))
for ax, (name, H, t, e, risks, _), grid, tau in zip(axs, cohorts[:2], [np.linspace(0.02, 0.30, 57), np.linspace(0.01, 0.15, 57)], [0.10, 0.05]):
    kn = e | (t >= H); prev = e[kn].mean()
    for k, r in risks.items():
        ax.plot(grid, [nb(r, t, e, g_, H) for g_ in grid], color=COL[k], lw=1.1, label=k)
    ax.plot(grid, [prev - (1 - prev) * g_ / (1 - g_) for g_ in grid], color=INK2, lw=0.8, ls=(0, (4, 2)), label='Test and treat all')
    ax.axhline(0, color=INK, lw=0.6); ax.axvline(tau, color=GREEN, lw=0.7, ls=':')
    ax.set_ylim(-0.02, prev * 1.1); ax.set_title(name, fontsize=7.5, loc='left'); ax.set_xlabel('Risk threshold'); ax.grid(color='#e5e3dc', lw=0.5)
axs[0].set_ylabel('Net benefit'); axs[0].legend(frameon=False, fontsize=6)
save(fig, 'figureS4_decision')
print('figures written')
