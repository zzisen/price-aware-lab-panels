#!/usr/bin/env python3
"""Appendix tables S1-S19 from the analysis outputs. Output: manuscript_ldh/appendix_tables.md
Numbers are copied from the result files unchanged; only labels and layout are made readable."""
import json
import re
from pathlib import Path
import pandas as pd
W = Path(__file__).resolve().parent.parent; R = W / 'task21_final'; OUT = Path(__file__).resolve().parent / 'appendix_tables.md'
T22, T23, T24, T25, T26 = (W / d for d in ['task22_adaptive', 'task23_local', 'task24_coai', 'task25_newdata', 'task26_transport'])
NAME = {'NH3': 'NHANES III (9-year death)', 'GH': 'NHANES 2011–14 (5-year death)', 'I': 'NHANES 2015–16 (3-year death; supplementary)'}
L = ['# Appendix tables\n']

# ---------------- labels ----------------
EXACT = {
    'sequential': 'earlier ordering rule', 'routine': 'routine bundle', 'matched': 'matched panel', 'aware@routine': 'target-aware panel',
    'none': 'clinical variables only', 'clinical only': 'clinical variables only', 'no labs (clinical only)': 'no laboratory tests',
    'age+sex only': 'age and sex only', 'all 13 menu': 'all 13 analytes', 'all 15 shared': 'all 15 shared analytes', 'all menu': 'all menu analytes',
    'all analytes': 'all analytes', 'not needed': 'unnecessary', 'helps': 'helpful', 'CMP+CBC+acr': 'CMP+CBC+ACR', 'LIPID': 'lipids', 'CMP+CBC+LIPID': 'CMP+CBC+lipids',
    'PhenoAge analytes': 'PhenoAge analytes (refitted)', 'PhenoAge published': 'PhenoAge (published formula)',
    'age40-59': 'age 40–59', 'age60-79': 'age 60–79', 'income <1.3x poverty': 'income <1.3× poverty line', 'income 1.3-3.5x': 'income 1.3–3.5×',
    'income >=3.5x': 'income ≥3.5×', 'True': 'yes', 'False': 'no',
}
SUBS = [
    (r'current rule \(Task 19/21, eps 0\.02\)', 'earlier rule'),
    (r'VOI V=(\d+)', lambda m: 'value-of-information rule (V=$' + f"{int(m.group(1)):,}".replace(',', ' ') + ')'),
    (r'CVD aware@PREVENT-full price', 'CVD target-aware panel (PREVENT price)'),
    (r'CVD aware@routine price', 'CVD target-aware panel (routine budget)'),
    (r'CVD aware@routine', 'CVD target-aware panel (routine budget)'),
    (r'CVD matched to PREVENT full', 'CVD panel matched to the PREVENT full set'),
    (r'CVD matched to routine', 'CVD panel matched to the routine bundle'),
    (r'aware@routine \(beam\)', 'target-aware panel (beam search)'), (r'matched \(beam\)', 'matched panel (beam search)'),
    (r'CoAI bundled @matched price', 'CoAI with panel prices, matched-panel budget'), (r'CoAI bundled @routine price', 'CoAI with panel prices, routine budget'),
    (r'CoAI @matched price', 'CoAI, matched-panel budget'), (r'CoAI @routine price', 'CoAI, routine budget'),
    (r'^d ours ', 'our '), (r'^ours matched$', 'our matched panel'), (r'^ours target-aware$', 'our target-aware panel'),
    (r'lasso-pw matched', 'price-weighted lasso, matched'), (r'lasso-uw matched', 'unweighted lasso, matched'),
    (r'lasso-pw within routine', 'price-weighted lasso, within routine budget'), (r'lasso-uw within routine', 'unweighted lasso, within routine budget'),
    (r'within routine price', 'within routine budget'),
    (r'aware@routine', 'target-aware'), (r'(?<![-\w])aware\b', 'target-aware'), (r'age\+sex only', 'age and sex only'), (r'\bacr\b', 'ACR'),
    (r'PCE/SCORE2 lipids - none', 'PCE/SCORE2 lipids − clinical variables only'),
    (r'^dC ', 'ΔC, '), (r'^dAUC ', 'ΔAUC, '), (r'^dNB ', 'ΔNB, '), (r' - ', ' − '), (r'\(shared items\)', '(shared analytes)'),
]


def lab(x):
    s = str(x)
    if s in EXACT:
        return EXACT[s]
    for a, b in SUBS:
        s = re.sub(a, b, s)
    return s


COL = {
    'panel': 'Panel', 'cohort': 'Cohort', 'contrast': 'Contrast', 'estimate': 'Estimate', 'lo': '95% CI, lower', 'hi': '95% CI, upper',
    'group': 'Subgroup', 'n': 'n', 'deaths': 'Deaths', 'items': 'Items', 'strategy': 'Strategy', 'policy': 'Strategy', 'measure': 'Measure',
    'US': 'Fee (US$)', 'US_dollars': 'Fee (US$)', 'CN_yuan': 'Fee (CN¥)', 'price': 'Fee (US$)', 'cost': 'Cost (US$)', 'mean_cost': 'Mean cost (US$)',
    'C': 'C', 'C_w': 'C, survey-weighted', 'OE': 'O/E', 'OE_6y': 'O/E', 'OE_9y': 'O/E', 'slope': 'Calibration slope',
    'NB': 'Net benefit', 'NB_w': 'Net benefit, survey-weighted', 'NB_half': 'Net benefit at ½× threshold', 'NB_double': 'Net benefit at 2× threshold',
    'AUC_weighted': 'AUC, survey-weighted', 'NB_weighted': 'Net benefit, survey-weighted', 'NB_routine': 'Net benefit, routine bundle',
    'per1000_cost': 'Cost per 1000 (US$)', 'per1000_flagged': 'Flagged per 1000', 'per1000_deaths_flagged': 'Deaths flagged per 1000',
    'per1000_deaths_missed': 'Deaths not flagged per 1000', 'flagged': 'Flagged per 1000', 'deaths_flagged': 'Deaths flagged per 1000',
    'deaths_missed': 'Deaths not flagged per 1000', 'flagged_per1000': 'Flagged per 1000', 'deaths_flagged_per1000': 'Deaths flagged per 1000',
    'missed_per1000': 'Deaths not flagged per 1000', 'share_cmpcbc': 'Share given CMP+CBC', 'share_routine': 'Cost, share of routine bundle',
    'no_lab': 'Share with no laboratory test', 'agree_final': 'Agreement with full-panel decision', 'agree_routine': 'Agreement with routine-bundle decision',
    'C_seq': 'C, ordering rule', 'C_routine_clin': 'C, routine bundle', 'share_CI': 'Share given CMP+CBC, 95% CI', 'agree_CI': 'Agreement, 95% CI',
    'dNB': 'ΔNB (mean)', 'dNB_CI': 'ΔNB, 95% CI', 'sample': 'Subsample', 'matched_cost': 'Matched panel fee (US$)', 'aware': 'Target-aware panel',
    'aware_cost': 'Target-aware panel fee (US$)', 'n_features': 'Number of analytes', 'C_oof': 'C (out-of-fold)', 'C_oof_linear': 'C (out-of-fold), linear',
    'C_oof_spline': 'C (out-of-fold), spline', 'hypothesis': 'Hypothesis', 'k': 'Cohorts', 'est': 'Pooled estimate', 'tau2': 'τ²', 'I2': 'I²',
    'pi_lo': 'Prediction interval, lower', 'pi_hi': 'Prediction interval, upper', 'passed': 'Met', 'cost_VOI': 'Cost, value-of-information rule (US$)',
    'missed_VOI': 'Deaths not flagged per 1000, value-of-information rule', 'missed_routine': 'Deaths not flagged per 1000, routine bundle',
    'dmissed': 'Difference', 'dmissed_lo': 'Difference, 95% CI lower', 'dmissed_hi': 'Difference, 95% CI upper',
    'flagged_VOI': 'Flagged per 1000, value-of-information rule', 'flagged_routine': 'Flagged per 1000, routine bundle', 'note': 'Note',
    'n_local': 'Local sample', 'reps': 'Repetitions', 'skipped': 'Repetitions skipped', 'abs_log_OE_import': 'Mean absolute log O/E, imported',
    'abs_log_OE_recal': 'Mean absolute log O/E, recalibrated', 'better_share': 'Share of repetitions improved', 'dNB_p2_5': 'ΔNB, 2.5th percentile',
    'dNB_p97_5': 'ΔNB, 97.5th percentile', 'T_recal': 'Share recalibrated after testing first', 'reading': 'Reading', 'metric': 'Metric',
    'cox': 'Ridge Cox model', 'coai_own_model': "CoAI's logistic model", 'estimate_coai_model': "Estimate, CoAI's logistic model",
    'identical_panels': 'Same panel as ours', 'imp': 'Imputation',
}
TWO = re.compile(r'^(US|US_dollars|CN_yuan|price|cost|mean_cost|per1000_.*|matched_cost|aware_cost|cost_VOI|.*per1000.*|flagged.*|deaths_.*|missed_.*|dmissed.*)$')


def head(c):
    if c in COL:
        return COL[c]
    s = c
    for a, b in [(r'aware@routine', 'target-aware'), (r'(?<![-\w])aware\b', 'target-aware'), (r'\bOE\b', 'O/E'), (r'\bsens\b', 'sensitivity'), (r'\bspec\b', 'specificity'),
                 (r'^dC CI lo$', 'ΔC, 95% CI lower'), (r'^dC CI hi$', 'ΔC, 95% CI upper'), (r'^dAUC CI lo$', 'ΔAUC, 95% CI lower'),
                 (r'^dAUC CI hi$', 'ΔAUC, 95% CI upper'), (r'^dC ', 'ΔC, '), (r'^dAUC ', 'ΔAUC, '), (r' - ', ' − '), (r'\(shared items\)', '(shared analytes)'),
                 (r'\bmatched\b(?! panel)', 'matched panel'), (r'\broutine\b(?! bundle)', 'routine bundle'), (r'\btarget-aware\b(?! panel)', 'target-aware panel')]:
        s = re.sub(a, b, s)
    return s[:1].upper() + s[1:]


ITEM = {'alt': 'ALT', 'ast': 'AST', 'bun': 'urea nitrogen', 'hdl': 'HDL cholesterol', 'tc': 'total cholesterol', 'tg': 'triglycerides',
        'uric_acid': 'uric acid', 'protein': 'total protein', 'crp': 'CRP', 'hba1c': 'HbA1c', 'ggt': 'GGT', 'ldh': 'LDH', 'acr': 'urine ACR',
        'ACR': 'urine ACR', 'iron': 'iron', 'phosphorus': 'phosphorus', 'albumin': 'albumin', 'creatinine': 'creatinine', 'glucose': 'glucose',
        'LIPID': 'lipids', 'CMP': 'CMP', 'CBC': 'CBC', 'haemoglobin': 'haemoglobin', 'hgb': 'haemoglobin'}


def items(v):
    toks = str(v).split()
    if toks and all(t in ITEM for t in toks):
        return '+'.join(ITEM[t] for t in toks)
    return None


def fmt(col, v):
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, list):
        return ' to '.join(f'{x:.4f}' for x in v)
    if isinstance(v, float):
        if pd.isna(v):
            return ''
        if col == 'I2':
            return f'{v * 100:.0f}%'
        if col == 'tau2':
            return '0' if v < 5e-10 else f'{v:.2e}'
        return f'{v:.2f}' if TWO.match(col) else f'{v:.4f}'
    if col == 'measure':
        return COL.get(str(v), lab(v))
    it = items(v)
    if it:
        return it
    return lab(v)


def md(df, cn=False):
    cols = list(df.columns); hh = [head(str(c)).replace('US$', 'CN¥') if cn else head(str(c)) for c in cols]
    out = ['| ' + ' | '.join(hh) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        out.append('| ' + ' | '.join(fmt(str(c), v) for c, v in zip(cols, r)) + ' |')
    return '\n'.join(out) + '\n'


def table(rows, cols):
    return '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n' + '\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows) + '\n'


# ---------------- S1-S11 ----------------
L.append('## Table S1. All panels in each US validation cohort\n\nNet benefit at the cohort threshold (10% for 9-year, 5% for 5-year and 3% for '
         '3-year risk), and at half and double that threshold. O/E: observed-to-expected deaths.\n')
for c in ['NH3', 'GH', 'I']:
    p = pd.read_csv(R / f't21_{c}_panels.csv').drop(columns=['features'])
    L.append(f'### {NAME[c]}\n'); L.append(md(p))
E = {c: json.load(open(R / f't21_{c}_endpoints.json')) for c in ['NH3', 'GH', 'I']}
L.append('## Table S2. Prespecified contrasts (bootstrap 95% CI, 1000 resamples)\n')
rows = []
for c, e in E.items():
    for k, v in e['point'].items():
        if k in e['boot']:
            rows.append(dict(cohort=NAME[c], contrast=k, estimate=v, lo=e['boot'][k][0], hi=e['boot'][k][1]))
ch = json.load(open(R / 't21_CHNS_endpoints.json'))
for k, v in ch['point'].items():
    rows.append(dict(cohort='CHNS (AUC, Chinese prices)', contrast=k, estimate=v, lo=ch['boot'][k][0], hi=ch['boot'][k][1]))
L.append(md(pd.DataFrame(rows)))
L.append('## Table S3. CHNS panels (Chinese prices, CN¥)\n'); L.append(md(pd.read_csv(R / 't21_CHNS_panels.csv'), cn=True))
L.append('## Table S4. Restricted cubic spline Cox models\n')
L.append(md(pd.DataFrame([dict(cohort=NAME[c], **E[c]['spline']) for c in E])))
L.append('## Table S5. Cardiovascular death (clinical base: age, sex, BMI, systolic blood pressure, smoking, diabetes)\n')
for c, e in E.items():
    d = e['cvd']; t = pd.DataFrame(dict(panel=list(d['C']), US=[d['US'][k] for k in d['C']], C=list(d['C'].values())))
    lab_ = ('heart-disease deaths (the 2015–16 public file has no separate cerebrovascular code; the models were trained on heart disease plus '
            'cerebrovascular death)') if c == 'I' else 'cardiovascular deaths (heart disease and cerebrovascular disease)'
    L.append(f"### {NAME[c]}: {d['events']} {lab_}\n"); L.append(md(t))
    L.append(md(pd.DataFrame([dict(contrast=k, estimate=v['point'], lo=v['ci'][0], hi=v['ci'][1]) for k, v in d['contrasts'].items()])))
L.append('## Table S6. Sequential ordering, earlier rule (stop when the probability of a changed decision is below 2%)\n')
L.append(md(pd.DataFrame([dict(cohort=NAME[c], **E[c]['seq']) for c in E])))
L.append('## Table S7. Subgroups (sensitivity and specificity at the cohort threshold; ΔC, matched panel − routine bundle)\n')
for c in ['NH3', 'GH', 'I']:
    L.append(f'### {NAME[c]}\n'); L.append(md(pd.read_csv(R / f't21_{c}_subgroups.csv')))
L.append('### CHNS\n'); L.append(md(pd.DataFrame(ch['subgroups'])))
he = E['NH3']['he']
L.append('## Table S8. Per 1000 people, NHANES III (9 years, 10% risk threshold; people with complete clinical variables)\n\n'
         'Cost: laboratory fees (US$). Flagged: people at or above 10% risk. The value-of-information rule is in table S13.\n')
L.append(md(pd.DataFrame([dict(strategy=k, **v) for k, v in he['per1000'].items()])))
L.append('## Table S9. Multiple imputation (NHANES III; C in each imputed dataset)\n')
L.append(md(pd.DataFrame(E['NH3']['mi']).rename(columns={'routine': 'C, routine bundle', 'matched': 'C, matched panel', 'aware@routine': 'C, target-aware panel'})))
L.append('## Table S10. Selection stability (20 random 80% subsamples of training data)\n'); L.append(md(pd.read_csv(R / 'stability.csv')))
dv = json.load(open(R / 'dev_train_report.json'))
L.append('## Table S11. Training-data performance (5-fold out-of-fold C)\n\n### All-cause death\n'); L.append(md(pd.DataFrame(dv['all_cause']).drop(columns=['features'])))
L.append('### Linear and spline models\n'); L.append(md(pd.DataFrame(dv['spline'])))
L.append('### Cardiovascular death\n'); L.append(md(pd.DataFrame(dv['cvd']).drop(columns=['features'])))
pr = dv['price']
L.append(f"### Price perturbation ({pr['draws']} random perturbations of the US fees)\n")
rows = [('Matched panel unchanged', f"{pr['matched_unchanged']:.1%}"), ('Target-aware panel unchanged', f"{pr['aware_unchanged']:.1%}"),
        ('Matched panel fee ≤80% of the routine bundle', f"{pr['matched_cost_le_80pct']:.1%}")]
rows += [(f'Matched panel chosen: {lab(k).replace("CBC CMP ", "CMP+CBC+").replace("ldh", "LDH").replace("ggt", "GGT")}', f'{v:.1%}')
         for k, v in pr['matched_choices'].items()]
rows += [('Australian fees with the coning rule (three most expensive items paid, plus the episode fee): routine bundle', f"A${pr['AU_coned']['routine']:.2f}"),
         ('Australian fees with the coning rule: Australian matched panel', f"A${pr['AU_coned']['matched_AU']:.2f}")]
L.append(table(rows, ['Quantity', 'Value']))

# ---------------- S12-S16 ----------------
P = json.load(open(R / 't21_pooled_final.json'))
L.append('## Table S12. Random-effects pooling (NHANES III, NHANES 2011–14, CHNS, Understanding Society, CLHLS ages 65–79)\n\n'
         'H1: matched panel − routine bundle (non-inferiority margin −0.01). H2: target-aware panel − routine bundle.\n')
L.append(md(pd.DataFrame([dict(hypothesis=h, **P[h]['pooled']) for h in ['H1', 'H2']])))
L.append('## Table S13. Sequential ordering rules (people with complete clinical variables; 9-year risk threshold 10%)\n\n'
         'Value-of-information rule: main V = $30 000; sensitivity analyses V = $10 000 and $100 000. Earlier rule: stop when the predicted probability '
         'of a changed decision is below 2%. Counts per 1000 people. Net benefit only for the 9-year cohort.\n')
for c in ['NH3', 'GH', 'I']:
    e22 = json.load(open(T22 / f't22_{c}_endpoints.json'))
    L.append(f"### {NAME[c]}: n {e22['n']}, deaths {e22['deaths']}\n"); L.append(md(pd.DataFrame(e22['table'])))
    L.append(md(pd.DataFrame([dict(contrast=k, measure=m_, estimate=v['point'], lo=v['ci'][0], hi=v['ci'][1])
                              for k, d in e22['contrasts'].items() for m_, v in d.items() if v['ci']])))
L.append('## Table S14. Sequential ordering by subgroup (deaths not flagged per 1000: value-of-information rule − routine bundle)\n')
for c in ['NH3', 'GH', 'I']:
    L.append(f'### {NAME[c]}\n'); L.append(md(pd.read_csv(T22 / f't22_{c}_subgroups.csv')))
L.append('## Table S15. Local recalibration (up to 200 attempts per local sample size; the rest of each cohort is the test set)\n\n'
         'Repetitions: usable attempts; attempts were skipped when the local sample or the test set had too few deaths. CLHLS was not analysed, because too '
         'few deaths would have remained for testing. Imported O/E uses the participants with the follow-up needed for the cohort horizon.\n\n'
         'Imported O/E: observed-to-expected ratio of the imported model in the whole cohort (95% CI). Mean absolute log O/E: mean absolute log '
         'observed-to-expected ratio in the test set. Share of repetitions improved: recalibration improved calibration on the same test set. ΔNB: net '
         'benefit, recalibrated − imported (mean, and 2.5th–97.5th percentiles across repetitions, the range a site might see). Share recalibrated after '
         'testing first: a variant that recalibrates only when the local O/E differs significantly from 1. Reading: helpful if the share improved is at '
         'least 0.70 and mean ΔNB is at least −0.001; unnecessary if it is below 0.50; otherwise uncertain.\n')
NM23 = dict(NAME, CHNS='CHNS (death by the 2015 survey; Chinese panels)', UKHLS='Understanding Society (death reported by about 9 years; UK menu, US prices)')
for c in ['NH3', 'GH', 'I', 'CHNS', 'UKHLS']:
    s23 = json.load(open(T23 / f't23_{c}_summary.json'))
    oe = '; '.join(f"{lab(p)} {v['OE']:.2f} ({v['lo']:.2f} to {v['hi']:.2f})" for p, v in s23['whole_cohort_import'].items())
    L.append(f"### {NM23[c]}: n {s23['n']}; imported O/E: {oe}\n")
    L.append(md(pd.DataFrame([dict(panel=x['panel'], n_local=x['n'], reps=x['reps_used'], skipped=x['skipped'], abs_log_OE_import=x['abs_log_OE_import'],
                                   abs_log_OE_recal=x['B recalibrate']['abs_log_OE'], better_share=x['B recalibrate']['better_calibrated_share'],
                                   dNB=x['B recalibrate']['dNB_mean'], dNB_p2_5=x['B recalibrate']['dNB_p2_5'], dNB_p97_5=x['B recalibrate']['dNB_p97_5'],
                                   T_recal=x['recal_share_T'], reading=x['reading_B']) for x in s23['cells']])))
L.append('## Table S16. Head-to-head comparison with CoAI (same training data and budgets; every panel refitted with ridge Cox)\n\n'
         'CoAI as published prices its outpatient features as fixed groups of single Medicare tests (for example the blood count as one group), with no panel '
         'pricing for chemistry tests. The variant with panel prices is given the CMP and lipid panels as orderable groups at their schedule prices (which also '
         'removes the single CMP and lipid tests from its choices) and the CBC. Fee: the set priced under the fee schedule, ordering each panel in full. Ridge '
         "Cox model: C (AUC in CHNS and Understanding Society); CoAI's logistic model: CoAI's own model refitted. Contrasts: 1000 paired bootstrap resamples. "
         'Pooled: DerSimonian–Laird over the US-priced primary cohorts (NHANES III, NHANES 2011–14 and Understanding Society); both pooled rows test the '
         'main-text hypothesis H4.\n')
NM24 = dict(NAME, CHNS='CHNS (Chinese prices; AUC)', UKHLS='Understanding Society (US prices; AUC)')
for c in ['NH3', 'GH', 'I', 'CHNS', 'UKHLS']:
    e24 = json.load(open(T24 / f't24_{c}_endpoints.json'))
    L.append(f"### {NM24[c]}: n {e24['n']}, deaths {e24['events']}\n"); L.append(md(pd.read_csv(T24 / f't24_{c}_panels.csv').drop(columns=['features']), cn=(c == 'CHNS')))
    L.append(md(pd.DataFrame([dict(contrast=k, estimate=v, lo=e24['boot'][k][0], hi=e24['boot'][k][1], estimate_coai_model=e24['point_coai_own_model'][k],
                                   identical_panels=e24['identical_panels'][k[2:].replace(' - ', ' | ')]) for k, v in e24['point'].items()])))
p24 = json.load(open(T24 / 't24_pooled.json'))
H4N = {'H5': 'H4, matched panel − CoAI (matched-panel budget)', 'H6': 'H4, target-aware panel − CoAI (routine budget)'}
L.append('### Pooled\n'); L.append(md(pd.DataFrame([dict(hypothesis=H4N[h], **p24[h]['pooled'], passed=p24[h]['pass']) for h in ['H5', 'H6']])))

# ---------------- S17 CLHLS ----------------
L.append('## Table S17. CLHLS (5-year death; Chinese prices; menu of 14 analytes measured in every blood wave)\n\n'
         'Data-reading rules were written from the survey documentation. Two data-format problems were found after the first analysis and corrected: (a) in '
         'the 2012 file the lymphocyte count and percentage columns were swapped (100 × count / white cells = 30.8%); (b) the 2008 hs-CRP values had no '
         'documented unit (median about 14 times the 2012 and 2014 values), so the 2008 blood wave could not be used. Three coding errors were then found and '
         'corrected before the final analysis: (c) valid days 29–31 had been set to 28; (d) people without any follow-up after baseline had been kept as '
         '0.5-month observations instead of being excluded as planned; (e) the unusable 2008 records still occupied each person\'s earliest draw, blocking '
         'their 2012 or 2014 samples. Estimates before these corrections are listed at the end of this table. O/E, calibration slope and net benefit in the panel tables use '
         'people with known 5-year status; because 30% of the 65–79 group were censored alive before 5 years, the censoring-aware estimates (Kaplan–Meier '
         'O/E below; inverse-probability-of-censoring-weighted net benefit and slope in table S23) are the ones reported in the main text.\n')
for tag, name in [('CLHLS', 'Ages 65–79 (primary)'), ('CLHLS80', 'Ages 80 and over')]:
    e25 = json.load(open(R / f't21_{tag}_endpoints.json'))
    L.append(f"### {name}: n {e25['n']}, deaths {e25['deaths']}\n"); L.append(md(pd.read_csv(R / f't21_{tag}_panels.csv').drop(columns=['features'])))
    L.append(md(pd.DataFrame([dict(contrast=k, estimate=v, lo=e25['boot'][k][0], hi=e25['boot'][k][1]) for k, v in e25['point'].items()])))
    sp = e25['spline']
    L.append(f"Restricted cubic spline Cox models (3 knots per analyte; {e25['n']} people, {e25['deaths']} deaths; 200 bootstrap resamples): C routine "
             f"{sp['C routine']:.4f}, matched {sp['C matched']:.4f}, target-aware {sp['C aware@routine']:.4f}; matched − routine "
             f"{sp['dC matched - routine']:.4f} ({sp['dC matched CI'][0]:.4f} to {sp['dC matched CI'][1]:.4f}); target-aware − routine "
             f"{sp['dC aware - routine']:.4f} ({sp['dC aware CI'][0]:.4f} to {sp['dC aware CI'][1]:.4f}).\n")
L.append('### Calibration accounting for censoring (post hoc)\n\nKnown-status O/E: people with known 5-year status only. Kaplan–Meier O/E: '
         'Kaplan–Meier 5-year risk divided by the mean predicted risk. Poisson O/E: observed deaths divided by the sum of predicted hazards over each '
         "person's follow-up.\n")
rows, waves = [], []
cur = None
for ln in (T25 / 'diag_calibration_km.txt').read_text(encoding='utf-8').splitlines():
    m = re.match(r'\s*(.+?)\s+(65-79|80\+)\s+n\s+(\d+) deaths\s+(\d+) \| known-status O/E \(frozen\) ([\d.]+) \| KM risk ([\d.]+) vs mean predicted ([\d.]+) '
                 r'-> O/E_KM ([\d.]+) \| O/E_Poisson ([\d.]+) \| share censored before 60 months alive ([\d.]+)', ln)
    if m:
        g = m.groups(); cur = (lab(g[0]), g[1].replace('65-79', '65–79').replace('80+', '80 and over'))
        rows.append([cur[0], cur[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9]]); continue
    w = re.match(r'\s*wave (\d+): n (\d+) deaths (\d+) O/E_Poisson ([\d.]+) median follow-up ([\d.]+) months', ln)
    if w and cur:
        waves.append([cur[0], cur[1], *w.groups()])
L.append(table(rows, ['Model', 'Ages', 'n', 'Deaths', 'Known-status O/E', 'Kaplan–Meier 5-year risk', 'Mean predicted risk', 'Kaplan–Meier O/E',
                      'Poisson O/E', 'Share censored alive before 5 years']))
L.append(table(waves, ['Model', 'Ages', 'Blood wave', 'n', 'Deaths', 'Poisson O/E', 'Median follow-up (months)']))
r1 = json.load(open(R / 't21_CLHLS_endpoints_run1.json'))
L.append('### Before correcting the data-format problems (a) and (b) (ages 65–79)\n')
L.append(md(pd.DataFrame([dict(contrast=k, estimate=v, lo=r1['boot'][k][0], hi=r1['boot'][k][1]) for k, v in r1['point'].items()])))
r2 = json.load(open(T25 / 'run2_outputs' / 't21_CLHLS_endpoints.json'))
L.append(f"### After correcting (a) and (b), before correcting the coding errors (c)–(e) (ages 65–79; n {r2['n']}, deaths {r2['deaths']})\n")
L.append(md(pd.DataFrame([dict(contrast=k, estimate=v, lo=r2['boot'][k][0], hi=r2['boot'][k][1]) for k, v in r2['point'].items()])))
L.append('### Data-quality sensitivity analyses (post hoc)\n')
sq = (T25 / 'sens_data_quality.txt').read_text(encoding='utf-8')
note = [ln.strip() for ln in sq.splitlines() if ln.startswith(('CLHLS:', 'CHNS:'))]
L.append(' '.join(n.replace('CLHLS: ', 'CLHLS: ').replace('hgb>25 g/dL', 'haemoglobin >25 g/dL').replace('mpv>20 fL', 'mean platelet volume >20 fL')
                  .replace('HDL>TC', 'HDL cholesterol above total cholesterol').replace('they are removed in this sensitivity analysis', 'they were removed')
                  .replace(' ([1009, 2005, 2008]); removed.', '; they were removed.') for n in note) + '\n')
rows = []
for ln in sq.splitlines():
    m = re.match(r'\s+(CLHLS 65-79|CLHLS 80\+|CHNS): n (\d+), deaths (\d+); (dC|dAUC) matched - routine ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\]; '
                 r'(?:dC|dAUC) aware@routine - routine ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\]', ln)
    if m:
        g = m.groups(); met = 'C' if g[3] == 'dC' else 'AUC'
        rows.append([g[0].replace('65-79', 'ages 65–79').replace('80+', 'ages 80 and over'), g[1], g[2], met,
                     f'{float(g[4]):.4f} ({float(g[5]):.4f} to {float(g[6]):.4f})', f'{float(g[7]):.4f} ({float(g[8]):.4f} to {float(g[9]):.4f})'])
    m = re.match(r'Pooled (H[12]) with both sensitivity substitutions: ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\], I2 (\d+)% \(main: ([-+\d.]+) \[([-+\d.]+), ([-+\d.]+)\]\)', ln)
    if m:
        g = m.groups()
        rows.append([f'Pooled {g[0]} (main analysis {float(g[5]):.4f}, {float(g[6]):.4f} to {float(g[7]):.4f})', '', '', '',
                     f'{float(g[1]):.4f} ({float(g[2]):.4f} to {float(g[3]):.4f}); I² {g[4]}%' if g[0] == 'H1' else '',
                     f'{float(g[1]):.4f} ({float(g[2]):.4f} to {float(g[3]):.4f}); I² {g[4]}%' if g[0] == 'H2' else ''])
L.append(table(rows, ['Analysis', 'n', 'Deaths', 'Metric', 'Matched panel − routine bundle', 'Target-aware panel − routine bundle']))

# ---------------- S18 re-centring ----------------
L.append('## Table S18. Re-centring the laboratory terms on local laboratory values (exploratory; no outcome data used for the shift)\n\n'
         'Simple: laboratory terms re-centred on the local means. Age- and sex-adjusted: re-centred on local means adjusted for age and sex (declared the '
         'primary variant before results). A shift is one constant per cohort, so discrimination is unchanged. O/E by Kaplan–Meier (CHNS: binary, known '
         'status) and by Poisson ratio. CLHLS rows use the final CLHLS sample (rerun after the corrections listed in table S17).\n')
blocks = re.split(r'\n\s*\n', (W / 'task29_revision' / 'dev_recenter.txt').read_text(encoding='utf-8').strip())
for b in blocks:
    lines = b.splitlines(); title = lines[0].strip()
    if title.startswith('O/E'):
        L.append('### ' + ('Kaplan–Meier O/E' if 'KM' in title else 'Poisson O/E') + '\n')
        rows, coh = [], ''
        for ln in lines[3:]:
            m = re.match(r'(.*?)\s*\b(routine|matched|aware@routine|age\+sex only)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$', ln)
            if m:
                coh = m.group(1).strip() or coh
                rows.append([coh.replace('2011-14', '2011–14').replace('2015-16', '2015–16').replace('65-79', 'ages 65–79').replace('80+', 'ages 80 and over'),
                             lab(m.group(2)), m.group(3), m.group(4), m.group(5)])
        L.append(table(rows, ['Cohort', 'Model', 'Imported', 'Simple re-centring', 'Age- and sex-adjusted re-centring']))
    elif title.startswith('mean / max'):
        rows = []
        for ln in lines[3:]:
            m = re.match(r'(C1 simple|C2 age-sex adjusted|original)\s+([\d.]+)\s+([\d.]+)', ln.strip())
            if m:
                rows.append([{'C1 simple': 'Simple re-centring', 'C2 age-sex adjusted': 'Age- and sex-adjusted re-centring', 'original': 'Imported'}[m.group(1)],
                             m.group(2), m.group(3)])
        L.append('### Mean and maximum absolute log O/E over laboratory panels and cohorts\n'); L.append(table(rows, ['Method', 'Mean', 'Maximum']))

# ---------------- S19 Understanding Society ----------------
e27 = json.load(open(R / 't21_UKHLS_endpoints.json')); c27 = e27['test']
L.append('## Table S19. Understanding Society (UK Household Longitudinal Study; nurse blood 2010–12; US prices; menu of 13 analytes)\n\n'
         'The End User Licence gives only the survey wave in which a death was reported, so the outcome is binary: death reported by the wave fielded nine '
         'waves after the nurse visit (waves 11–12, 2019–22), against being enumerated alive at that wave; people with neither were not analysed (as in CHNS). '
         'Discrimination: AUC. Decision threshold: 10% 9-year risk. Weighted columns use the nurse-blood cross-sectional weight.\n')
sp = lambda v: f'{v:,}'.replace(',', ' ') if v >= 10000 else str(v)
L.append(f"Flow: blood records {sp(c27['blood_records'])}; aged 40–79 {sp(c27['age_40_79'])}; 9-year status known {sp(c27['known_9y_status'])}; all 13 analytes "
         f"{sp(c27['complete'])} (wave 2 {sp(c27['by_wave']['2'])}, wave 3 {sp(c27['by_wave']['3'])}); deaths {e27['deaths']}. Among people classed as alive "
         'with a known nurse-visit month and end-point sample-issue month, the median interval from the nurse visit to the end-point wave was 103 months '
         "(wave 2; 4056 people) and 102 months (wave 3; 1507 people). O/E uses the imported model's 108-month risk as an approximate comparator. Death "
         'status, the reporting wave and the last enumerated wave are from the cross-wave identifier file; variable names are in the code repository.\n')
L.append(md(pd.read_csv(R / 't21_UKHLS_panels.csv').drop(columns=['features'])))
L.append(md(pd.DataFrame([dict(contrast=k, estimate=v, lo=e27['boot'][k][0], hi=e27['boot'][k][1]) for k, v in e27['point'].items()])))
L.append('### Subgroups\n'); L.append(md(pd.read_csv(R / 't21_UKHLS_subgroups.csv')))
s = e27['spline']
L.append('### Restricted cubic spline models (AUC)\n')
L.append(table([[f"{s['routine (shared items)']:.4f}", f"{s['matched']:.4f}", f"{s['aware@routine']:.4f}",
                 f"{s['dAUC matched - routine']:.4f} ({s['dAUC matched CI'][0]:.4f} to {s['dAUC matched CI'][1]:.4f})",
                 f"{s['dAUC aware - routine']:.4f} ({s['dAUC aware CI'][0]:.4f} to {s['dAUC aware CI'][1]:.4f})"]],
               ['Routine bundle (shared analytes)', 'Matched panel', 'Target-aware panel', 'ΔAUC, matched − routine (95% CI)', 'ΔAUC, target-aware − routine (95% CI)']))
OUT.write_text('\n'.join(L), encoding='utf-8'); print('written', OUT)

# ---------------- S20 missing inputs (post hoc) ----------------
T28 = W / 'task28_missing'
s28 = pd.read_csv(T28 / 't28_summary.csv'); g28 = pd.read_csv(T28 / 't28_single.csv')
NM28 = {'NH3': 'NHANES III', 'GH': 'NHANES 2011–14'}
MD28 = {'matched': 'matched panel', 'target-aware': 'target-aware panel', 'routine': 'routine bundle'}
L2 = ['## Table S20. Missing inputs in use (post hoc)\n\n'
      'Missing inputs were replaced by their conditional mean given age, sex and the results present, from the mean vector and covariance of the '
      'transformed inputs in the training data, as the web calculator does; the 80% range comes from the conditional variance of the linear predictor. '
      'Decisions use 10% predicted 9-year risk. Same decision: share of people with the same decision as with complete inputs. Flagged: share whose 80% '
      'range crosses 10%. Same if not flagged: same decision among people not flagged. Changes flagged: share of changed decisions that were flagged. '
      f"One analyte missing: each analyte of each model removed in turn ({len(g28)} cases); median change in C {g28.dC.median():.4f}, "
      f"{(g28.dC > -0.005).mean():.0%} of cases within 0.005, largest {g28.dC.min():.4f}. In the rows of column-wise extremes each column is the "
      "extreme over the analytes removed one at a time, so different columns may come from different analytes.\n"]
rows = []
for _, r in s28.iterrows():
    rows.append([NM28[r.cohort], MD28[r.model], r.scenario.replace('one analyte missing: worst of each', 'one analyte missing: column-wise extremes over analytes'),
                 f'{r.C_complete:.4f}', f'{r.C:.4f}', f'{r.dC:.4f}', f'{r.same_decision:.1%}', f'{r.flagged:.1%}', f'{r.same_if_not_flagged:.1%}',
                 '' if pd.isna(r.changed_and_flagged) else f'{r.changed_and_flagged:.1%}'])
L2.append(table(rows, ['Cohort', 'Model', 'Scenario', 'C, complete', 'C, imputed', 'ΔC', 'Same decision', 'Flagged', 'Same if not flagged', 'Changes flagged']))
OUT.write_text(OUT.read_text(encoding='utf-8') + '\n' + '\n'.join(L2), encoding='utf-8'); print('S20 added')

# ---------------- S21-S24: revision analyses (post hoc; task29_revision/PLAN.md) ----------------
T29 = W / 'task29_revision'
NM29 = {'NH3': 'NHANES III', 'GH': 'NHANES 2011–14', 'CHNS': 'CHNS', 'UKHLS': 'Understanding Society', 'CLHLS': 'CLHLS'}
J = lambda n: json.load(open(T29 / n))  # noqa: E731
f4 = lambda v: f'{v:.4f}'  # noqa: E731
ci4 = lambda d: f"{d['point']:.4f} ({d['ci'][0]:.4f} to {d['ci'][1]:.4f})"  # noqa: E731
ORD = {'CBC without differential': 'CBC without differential', 'Hb': 'haemoglobin', 'HEPATIC': 'hepatic function panel', 'RENAL': 'renal function panel',
       'ELEC': 'electrolyte panel', 'LIPID': 'lipid panel', 'hba1c': 'HbA1c', 'acr': 'urine ACR', 'crp': 'CRP', 'CBC': 'CBC', 'CMP': 'CMP', 'BMP': 'BMP'}
od = lambda o: ' + '.join(ORD.get(x, AN.get(x, x)) for x in o) if o else 'none'  # noqa: E731
PN = {'age+sex only': 'Age and sex only', 'routine': 'Routine items', 'old matched': 'Matched panel (original menu)', 'old aware': 'Target-aware panel (original menu)',
      'new matched': 'Matched panel (expanded menu)', 'new aware': 'Target-aware panel (expanded menu)', 'CMP+CBC': 'CMP+CBC',
      'new aware (as-ordered budget)': 'Target-aware panel, as-ordered budget', 'CoAI full @matched': 'CoAI at the matched-panel price',
      'CoAI full @routine': 'CoAI at the routine price', 'CoAI full @measured routine': 'CoAI at the as-measured routine price',
      'CoAI full @ordered routine': 'CoAI at the as-ordered routine price', 'lasso-pw within routine (full)': 'Price-weighted lasso within the routine price',
      'lasso-pw matched (full)': 'Price-weighted lasso, matched', 'NHS Health Check': 'NHS Health Check laboratory tests', 'matched to Health Check': 'Matched to the Health Check',
      'Health Check budget': 'Best panel within the Health Check price', 'NBPHS check': 'NBPHS check laboratory tests', 'matched to NBPHS': 'Matched to the NBPHS check',
      'NBPHS budget': 'Best panel within the NBPHS price', 'matched': 'Matched panel (main analysis)', 'aware@routine': 'Target-aware panel (main analysis)'}


def pn(k):
    return ' − '.join(PN.get(x, x) for x in k.split(' - '))


AN = {'acr': 'urine ACR', 'alp': 'ALP', 'alt': 'ALT', 'ast': 'AST', 'ggt': 'GGT', 'hgb': 'haemoglobin', 'lymph_pct': 'lymphocyte %', 'gran_pct': 'granulocyte %',
      'mcv': 'MCV', 'mpv': 'MPV', 'plt': 'platelets', 'rdw': 'RDW', 'wbc': 'white cells', 'protein': 'total protein', 'tc': 'total cholesterol',
      'hdl': 'HDL cholesterol', 'tg': 'triglycerides', 'bun': 'urea', 'crp': 'CRP', 'hba1c': 'HbA1c'}
an = lambda a: AN.get(a, a)  # noqa: E731


L3 = ['## Table S21. Expanded US fee menu (post hoc)\n\n'
      'The original US menu billed chemistry singly or as the BMP, CMP or lipid panel, and the blood count only as a CBC with differential. The expanded '
      'menu adds the electrolyte panel (CPT 80051, $7.01), renal function panel (80069, $8.68), hepatic function panel (80076, $8.17), haemoglobin alone '
      '(85018, $2.37) and a CBC without differential (85027, $6.47: white cells, haemoglobin, MCV, RDW, platelets, MPV), all from the 2026 Q4 Clinical '
      'Laboratory Fee Schedule. Each feature set is billed at the cheapest billable combination. Selection was repeated in training data with the same score '
      'and search, reusing the cross-validated scores already computed (the score does not depend on prices). CoAI used the same feature importances and '
      'was run within the expanded candidate menu, in every compatible disjoint grouping of its orders (chemistry singly or through one panel family; lipids as a panel or singly; full CBC, CBC without '
      'differential or haemoglobin alone), keeping the grouping with the highest total importance within budget. The lasso candidates were repriced. '
      'Understanding Society: the routine reference was priced as ordered (CMP, CBC, lipids and HbA1c: $50.77) and as the cheapest billing of the '
      'analytes measured ($45.37, with haemoglobin alone).\n']
if (T29 / 'sel_us.json').exists():
    su = J('sel_us.json'); sk = J('sel_uk.json')
    rows = [['US', 'Routine bundle', od(su['routine']['order']), f"${su['routine']['price_old']:.2f}", f"${su['routine']['price_full']:.2f}"],
            ['US', 'Matched panel, expanded menu', od(su['matched']['order']), '', f"${su['matched']['price']:.2f}"],
            ['US', 'Target-aware panel, expanded menu', od(su['aware']['order']), '', f"${su['aware']['price']:.2f}"]]
    rows += [['US', f"CoAI, expanded menu, budget ${p['budget']:.2f}", od(p['order']), '', f"${p['billed_cost']:.2f}"] for p in su['coai_full']['panels']]
    for k, nm in [('matched', 'matched'), ('within', 'within the routine budget')]:
        v = su['lasso_full']['price-weighted'][k]
        if v:
            rows.append(['US', f'Price-weighted lasso, {nm}', 'analytes used (not orders): ' + ', '.join(an(a) for a in v['features']), '', f"${v['price']:.2f}"])
    rows += [['UK', 'Routine items', 'as ordered / as measured', f"${sk['routine']['price_old']:.2f}",
              f"${sk['routine']['price_as_ordered']:.2f} / ${sk['routine']['price_as_measured']:.2f}"],
             ['UK', 'Matched panel, expanded menu', od(sk['matched']['order']), '', f"${sk['matched']['price']:.2f}"],
             ['UK', 'Target-aware panel, as-ordered budget', od(sk['aware_ordered']['order']), '', f"${sk['aware_ordered']['price']:.2f}"],
             ['UK', 'Target-aware panel, as-measured budget', od(sk['aware_measured']['order']), '', f"${sk['aware_measured']['price']:.2f}"]]
    rows += [['UK', f"CoAI, expanded menu, budget ${p['budget']:.2f}", od(p['order']), '', f"${p['billed_cost']:.2f}"] for p in sk['coai_full']['panels']]
    L3.append('### Panels chosen in training data\n'); L3.append(table(rows, ['Menu', 'Panel', 'Orders', 'Price, original menu', 'Price, expanded menu']))
    L3.append('### External validation\n')
    for c, nm in [('NH3', 'NHANES III (C)'), ('GH', 'NHANES 2011–14 (C)'), ('I', 'NHANES 2015–16 (C; supplementary)'), ('UKHLS', 'Understanding Society (AUC)')]:
        if not (T29 / f'valid_{c}.json').exists():
            continue
        v = J(f'valid_{c}.json'); L3.append(f'**{nm}**\n')
        keep = [k for k in v['panels'] if c != 'UKHLS' or 'Health Check' not in k]
        L3.append(table([[pn(k), f"${v['panels'][k]['price']:.2f}", f4(v['panels'][k]['disc']), f4(v['panels'][k]['NB'])] for k in keep],
                        ['Panel', 'Price (expanded menu)', 'C or AUC', 'Net benefit']))
        L3.append(table([[pn(k), ci4(d)] for k, d in v['contrasts'].items() if 'Health Check' not in k] +
                        [['Net benefit: ' + pn(k), ci4(d)] for k, d in v['nb_contrasts'].items() if 'Health Check' not in k],
                        ['Contrast', 'Difference (95% CI)']))
    if (T29 / 'meta_full.json').exists():
        mf = J('meta_full.json'); rows = []
        for h_, nm in [('H1', 'Matched panel − routine bundle'), ('H2', 'Target-aware panel − routine bundle')]:
            d = mf[h_]['all']['DL']; rows.append([nm, f"{d['est']:.4f} ({d['lo']:.4f} to {d['hi']:.4f})", f"{d['I2']:.0%}"])
        L3.append('### Pooled over the five primary cohorts with the expanded-menu panels in the US and UK cohorts\n')
        L3.append(table(rows, ['Contrast', 'Pooled (95% CI)', 'I²']))
# S22 programmes
CNM = {'CBC': 'blood count', 'alt': 'ALT', 'bun': 'urea', 'hdl': 'HDL cholesterol', 'tc': 'total cholesterol', 'tg': 'triglycerides', 'protein': 'total protein',
       'crp': 'CRP', 'uric_acid': 'uric acid', 'hba1c': 'HbA1c'}


def cn_tests(o):
    o = [x for x in o if x not in ('lymph_pct', 'mcv', 'mpv', 'wbc', 'hgb', 'plt')]
    return ', '.join(CNM.get(x, x) for x in o) or 'none'


L3.append('## Table S22. Programme-based laboratory references from two national health checks (post hoc)\n\n'
          'China, National Basic Public Health Service annual check for people aged 65 and over (NBPHS; 2017 specification): blood count, liver function (ALT, AST, total '
          'bilirubin), kidney function (creatinine, urea), fasting glucose and lipids (total, HDL and LDL cholesterol, triglycerides); urinalysis, AST, total bilirubin and LDL cholesterol were not in the analysis menus and are not '
          'included (CHNS measured LDL cholesterol, but it was not on the training menu). Price of the full 2017 blood list at our Shandong prices, including one blood draw, CN¥79.8 (AST CN¥3.5, total bilirubin CN¥4.5 and LDL cholesterol CN¥11, from the Jining list in appendix A1); the reference is the subset of the 2017 programme tests included in each cohort\'s analysis menu, at Chinese '
          'prices. England, NHS Health Check (2019 guidance): simulated as the lipid panel and HbA1c for everyone, priced under the expanded US menu as a proxy (haemoglobin alone costs $2.37 plus one $9.34 draw). Cohort age ranges (CHNS and Understanding Society 40–79) differ from those of the programmes. '
          'Panels were chosen in training data against each reference: the cheapest set at least as informative (matched to the programme) and the most '
          'informative set within its price (programme budget). CLHLS: ages 65–79, 5-year death, inverse-probability-of-censoring-weighted net benefit at '
          '5%; CHNS: 72-month risk, net benefit at 6%; Understanding Society: 9-year risk, net benefit at 10%.\n')
for c, nm, cur in [('CHNS', 'CHNS (AUC; Chinese prices)', 'CN¥'), ('CLHLS', 'CLHLS, ages 65–79 (C; Chinese prices)', 'CN¥'),
                   ('UKHLS', 'Understanding Society (AUC; US prices, expanded menu)', '$')]:
    if not (T29 / f'valid_{c}.json').exists():
        continue
    v = J(f'valid_{c}.json'); L3.append(f'**{nm}**\n')
    keep = [k for k in v['panels'] if c != 'UKHLS' or k in ('age+sex only', 'routine', 'NHS Health Check', 'matched to Health Check', 'Health Check budget')]
    L3.append(table([[pn(k), od(v['panels'][k]['order']) if c == 'UKHLS' else cn_tests(v['panels'][k]['order']),
                      f"{cur}{v['panels'][k]['price']:.2f}", f4(v['panels'][k]['disc']), f4(v['panels'][k]['NB'])] for k in keep],
                    ['Panel', 'Tests', 'Price', 'C or AUC', 'Net benefit']))
    pr = [k for k in v['contrasts'] if c != 'UKHLS' or 'Health Check' in k]
    L3.append(table([[pn(k), ci4(v['contrasts'][k])] for k in pr] + [['Net benefit: ' + pn(k), ci4(d)] for k, d in v['nb_contrasts'].items()
                                                                     if c != 'UKHLS' or 'Health Check' in k], ['Contrast', 'Difference (95% CI)']))
# S23 censoring and horizons
cz = J('censor.json')
L3.append('## Table S23. Censoring, unknown status and prediction horizons (post hoc)\n\n'
          'CLHLS: net benefit and calibration slope recomputed with inverse probability of censoring weights (reverse Kaplan–Meier; people censored alive '
          'before 5 years carry weight 0 and the denominator is everyone); Kaplan–Meier net benefit as a check. Understanding Society: people with unknown '
          '9-year status were represented by weighting those with known status by the inverse of their probability of known status (logistic model on age, '
          'sex, wave and the 13 analytes; weights truncated at the 1st and 99th percentiles). CHNS: 72-month risk and a 6% threshold (about 1% a year, as for '
          'the other horizons). Understanding Society: 102-month risk, the typical interval to the last wave for people known to be alive.\n')
rows = []
for tag, nm in [('CLHLS', 'ages 65–79'), ('CLHLS80', 'ages 80 and over')]:
    for k, p in cz[tag]['panels'].items():
        rows.append([nm, lab(k), f4(p['NB_known_status']), f4(p['NB_ipcw']), f4(p['NB_km']), f4(p['NB_treat_all_ipcw']), f"{p['slope_known_status']:.2f}",
                     f"{p['slope_ipcw']:.2f}", f"{p['OE_ipcw']:.2f}"])
L3.append(f"### CLHLS (censored alive before 5 years: {cz['CLHLS']['censored_alive_before_H']:.0%} at ages 65–79, "
          f"{cz['CLHLS80']['censored_alive_before_H']:.0%} at 80 and over)\n")
L3.append(table(rows, ['Ages', 'Model', 'Net benefit, known status', 'Net benefit, IPCW', 'Net benefit, Kaplan–Meier', 'Treat all, IPCW',
                       'Slope, known status', 'Slope, IPCW', 'O/E, IPCW']))
L3.append(table([[pn(k.replace('dNB ', '').replace('aware@routine', 'aware@routine').replace(' - routine', ' - routine')), ci4(d)] for k, d in cz['CLHLS']['contrasts_ipcw'].items()],
                ['Contrast, ages 65–79 (IPCW net benefit)', 'Difference (95% CI)']))
L3.append('### NHANES cohorts: share censored alive before the horizon\n')
L3.append(table([[k.replace('2011-14', '2011–14').replace('2015-16', '2015–16'), v['n'], f"{v['share']:.2%}"] for k, v in cz['nhanes_censoring'].items()],
                ['Cohort', 'n', 'Censored alive before the horizon']))
u = cz['UKHLS']
L3.append(f"### Understanding Society (complete laboratory data {u['n_complete']}; known status {u['n_known']}; unknown {u['unknown_share']:.1%})\n")
L3.append(table([[lab(k), f4(p['AUC_known']), f4(p['AUC_ipw']), f"{p['OE_108_known']:.2f}", f"{p['OE_108_ipw']:.2f}", f"{p['OE_102_known']:.2f}",
                  f"{p['slope_108']:.2f}", f"{p['slope_102']:.2f}", f4(p['NB_108_known']), f4(p['NB_108_ipw']), f4(p['NB_102_known'])] for k, p in u['panels'].items()],
                ['Model', 'AUC', 'AUC, IPW', 'O/E 108 m', 'O/E 108 m, IPW', 'O/E 102 m', 'Slope 108 m', 'Slope 102 m', 'NB 108 m', 'NB 108 m, IPW', 'NB 102 m']))
L3.append(table([[('AUC: ' if k.startswith('dAUC') else 'Net benefit: ') + pn(k.split(' ', 1)[1]), ci4(d)] for k, d in u['contrasts_ipw'].items()],
                ['Contrast (inverse-probability weighted)', 'Difference (95% CI)']))
h = cz['CHNS']
L3.append(f"### CHNS (n {h['n']}, deaths {h['deaths']}; treat-all net benefit at 6%: {h['NB_treat_all_72_tau6']:.4f})\n")
L3.append(table([[lab(k), f"{p['OE_72']:.2f}", f"{p['slope_72']:.2f}", f4(p['NB_72_tau6']), f4(p['NB_9y_tau10_old']), f"{p['flagged_72_tau6']:.1%}",
                  f"{p['flagged_9y_tau10']:.1%}", f"{p['threshold_6y_equivalent_to_9y_10pct']:.1%}"] for k, p in h['panels'].items()],
                ['Model', 'O/E, 72 m', 'Slope, 72 m', 'NB, 72 m at 6%', 'NB, 9-year risk at 10%', 'Flagged, 72 m ≥6%', 'Flagged, 9-year ≥10%',
                 '6-year risk equivalent to 9-year 10%']))
L3.append(table([[pn(k.replace('dNB ', '')), ci4(d)] for k, d in h['contrasts_72'].items()], ['Contrast (72-month net benefit at 6%)', 'Difference (95% CI)']))
# S24 pooling sensitivity
mp = J('meta_published.json')
L3.append('## Table S24. Pooling sensitivity and the non-inferiority margin (post hoc)\n\n'
          'DL: DerSimonian–Laird random effects (main analysis). REML–HK: restricted maximum likelihood with the Hartung–Knapp interval (t distribution, '
          'k−1 degrees of freedom). C cohorts: NHANES III, NHANES 2011–14, CLHLS; AUC cohorts: CHNS, Understanding Society.\n')


def fx(d):
    return f"{d['est']:.4f} ({d['lo']:.4f} to {d['hi']:.4f})"


rows = []
for h_, nm in [('H1', 'Matched − routine'), ('H2', 'Target-aware − routine')]:
    a = mp[h_]
    rows += [[nm, 'All five', fx(a['all']['DL']), fx(a['all']['REML_HK'])], [nm, 'C cohorts', fx(a['C_only']['DL']), fx(a['C_only']['REML_HK'])],
             [nm, 'AUC cohorts', fx(a['AUC_only']['DL']), '—']]
    rows += [[nm, f'Without {NM29[c]}', fx(v['DL']), fx(v['REML_HK'])] for c, v in a['leave_one_out'].items()]
L3.append(table(rows, ['Contrast', 'Cohorts', 'DL (95% CI)', 'REML–HK (95% CI)']))
ni, gn = mp['noninferiority'], mp['margin_vs_gain']
rows = [[NM29[c], f"{ni[c]['lower']:.4f}", 'yes' if ni[c]['ni_001'] else 'no', 'yes' if ni[c]['ni_0005'] else 'no',
         f"{gn[c]['routine_gain']:.4f}", f"{gn[c]['margin_share_001']:.0%}"] for c in ['NH3', 'GH', 'CHNS', 'UKHLS', 'CLHLS']]
rows.append(['Pooled', f"{ni['pooled']['lower']:.4f}", 'yes' if ni['pooled']['ni_001'] else 'no', 'yes' if ni['pooled']['ni_0005'] else 'no', '', ''])
L3.append("### Matched panel: lower 95% bound against two margins, and the −0.01 margin relative to the routine bundle's gain over age and sex\n")
L3.append(table(rows, ['Cohort', 'Lower bound', 'Above −0.01', 'Above −0.005', 'Routine minus age and sex', '0.01 as share of that gain']))
OUT.write_text(OUT.read_text(encoding='utf-8') + '\n' + '\n'.join(L3), encoding='utf-8'); print('S21-S24 added')

# ---------- S25-S27: further post hoc analyses (task30_extra) ----------
T30 = W / 'task30_extra'
L4 = []
mi = pd.read_csv(T30 / 't30_mi_pooled.csv'); MIQ = {'dC matched': 'C: matched − routine', 'dC aware': 'C: target-aware − routine',
                                                    'dNB matched': 'Net benefit at 10%: matched − routine', 'dNB aware': 'Net benefit at 10%: target-aware − routine'}
L4.append('## Table S25. Multiple imputation in NHANES III: paired differences (post hoc)\n\n'
          '8002 participants with at least half of the analytes measured (1307 deaths by 9 years); five imputed datasets (chained equations, as in table S9); '
          '200 participant bootstrap resamples within each dataset; Rubin\'s rules with Barnard–Rubin degrees of freedom.\n')
fm = lambda v: f'{v:.5f}' if abs(v) < 0.0001 and v != 0 else f'{v:.4f}'  # noqa: E731
L4.append(table([[MIQ[r.quantity], f"{fm(r.estimate)} ({fm(r.low)} to {fm(r.high)})"] for r in mi.itertuples()], ['Contrast', 'Pooled difference (95% CI)']))
cz = pd.read_csv(T30 / 't30_calc.csv'); CN = {'NH3': 'NHANES III (9-year risk, 10%)', 'GH': 'NHANES 2011–14 (5-year risk, 5%)'}
L4.append("## Table S26. The calculator's 80% risk range with missing results (post hoc)\n\n"
          'Fixed panels, values removed by simulation. Coverage: share of people whose complete-data risk (not true risk) lies within the 80% range; coverage and width are on the 9-year risk scale of the calculator (nominal coverage 80%). Width: upper minus lower limit, percentage points. One analyte missing: median over analytes, and the column-wise extreme over analytes (lowest C and coverage, widest range, largest ΔNB loss; not necessarily the same analyte). '
          'ΔNB: net benefit with imputed risk minus net benefit with complete data, at the cohort threshold. Same cohorts, models and missing-data scenarios as table S20.\n')
for c in ['NH3', 'GH']:
    L4.append(f'### {CN[c]}\n')
    L4.append(table([[r.model, r.scenario.replace('(worst analyte)', '(column-wise extremes)'), f'{r.C_complete:.4f}', f'{r.C:.4f}', f'{r.coverage:.0%}', f'{r.width_median_pp:.1f}', f'{r.width_p90_pp:.1f}', f'{r.dNB:+.4f}']
                     for r in cz[cz.cohort == c].itertuples()],
                    ['Model', 'Scenario', 'C, complete', 'C, imputed', 'Coverage', 'Median width', '90th centile width', 'ΔNB']))
bz = pd.read_csv(T30 / 't30_bench_summary.csv')
L4.append('## Table S27. Cross-domain benchmark: split-level spread (appendix figure S4; post hoc)\n\n'
          'Training subsample n=100; ten data blocks (from nine sources) × five outer splits per setting. Lead: explained variance of the better target-aware approach minus the best reusable panel (FSCA), '
          'per block and outer split. Interval: 2000 bootstrap resamples of blocks. The method on each side was chosen on the same held-out scores, so the table is descriptive.\n')
L4.append(table([[f'k={int(r.k)}, ' + ('ridge' if r.predictor == 'ridge' else 'boosted trees'), f'{r.lead_positive}/{r.cells}', f'{r.lead_ge_002}/{r.cells}',
                  f'{r.min_diff:.3f}', f'{r.mean_lead:.3f} ({r.low:.3f} to {r.high:.3f})'] for r in bz.itertuples()],
                ['Setting', 'Splits with a lead', 'Lead ≥0.02', 'Smallest lead', 'Mean lead (95% interval)']))
OUT.write_text(OUT.read_text(encoding='utf-8') + '\n' + '\n'.join(L4), encoding='utf-8'); print('S25-S27 added')
import re as _re  # noqa: E402
OUT.write_text(_re.sub(r'-0\.0000(?!\d)', '0.0000', OUT.read_text(encoding='utf-8')), encoding='utf-8')   # no negative zero in printed tables
