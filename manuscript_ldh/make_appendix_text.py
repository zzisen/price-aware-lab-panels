#!/usr/bin/env python3
"""Appendix text: fee schedules and sources, harmonisation across survey eras and countries, PROBAST self-assessment.
Price tables are generated from the price code actually used (dmlib, dm2lib, t20lib), so text and analysis cannot disagree. Output: appendix_text.md"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
L = M.L
NAME = {'glucose': 'Glucose', 'bun': 'Urea nitrogen', 'creatinine': 'Creatinine', 'sodium': 'Sodium', 'potassium': 'Potassium', 'chloride': 'Chloride',
        'co2': 'Bicarbonate', 'calcium': 'Calcium', 'albumin': 'Albumin', 'protein': 'Total protein', 'alp': 'Alkaline phosphatase', 'alt': 'ALT', 'ast': 'AST',
        'bilirubin': 'Total bilirubin', 'ggt': 'GGT', 'uric_acid': 'Uric acid', 'ldh': 'LDH', 'phosphorus': 'Phosphorus', 'iron': 'Iron', 'tg': 'Triglycerides',
        'tc': 'Total cholesterol', 'hdl': 'HDL cholesterol', 'hba1c': 'HbA1c', 'crp': 'C-reactive protein', 'acr': 'Urine albumin/creatinine ratio'}
out = ['# Appendix text\n', '## A1. Fee schedules and sources\n',
       'Prices are the listed schedule fees for the laboratory tests alone, under a scenario in which every panel that is ordered is performed in full '
       '(as US coding rules require for billing a panel code), whatever subset of its results a model uses. Bundling, tier and episode rules are '
       'implemented in the released code. The tables below are generated from that code. "Cheapest" '
       'everywhere means cheapest among the bundles and single tests modelled here, among the sets the search evaluated.\n',
       '### United States: Medicare Clinical Laboratory Fee Schedule, 2026 quarter 4 (US$)\n',
       'Source: Centers for Medicare & Medicaid Services, CLFS file CY2026 Q4 version 1 (national limitation amounts). A set of chemistry analytes is priced '
       'at the cheapest combination of single-test codes and the basic metabolic panel (CPT 80048), comprehensive metabolic panel (80053) and lipid panel '
       '(80061) codes that covers it; the electrolyte (80051), renal function (80069) and hepatic function (80076) panels were not modelled. CBC with '
       'differential 85025; HbA1c 83036; high-sensitivity CRP 86141 (standard CRP 86140 is US$5.18); urine albumin 82043 and urine creatinine 82570; '
       'venipuncture 36415, added once when any blood test is ordered. In Understanding Society, haemoglobin was the only blood-count index measured; '
       'it is priced as an ordered CBC (the single haemoglobin code 85018, US$2.37, was not modelled).\n',
       '| Item | Price (US$) |', '|---|---|']
out += [f'| {NAME[a]} (single test) | {v[2]:.2f} |' for a, v in L.CHEM.items()]
out += [f"| {p} panel ({', '.join(NAME[a] for a in m)}) | {pr:.2f} |" for p, (m, pr) in L.PANELS.items()]
out += [f'| Complete blood count with differential | {L.CBC_PRICE:.2f} |', '| HbA1c | 9.71 |', '| C-reactive protein | 12.95 |',
        '| Urine albumin + urine creatinine (ACR) | 10.96 |', '| Venipuncture (once per draw) | 9.34 |',
        f"\nRoutine bundle (CMP + CBC + lipid panel + HbA1c): US${M.cost(M.STANDARD['CMP+CBC+LIPID+A1C']):.2f}. The 14 CMP analytes priced one by one: "
        f"US${sum(L.CHEM[a][2] for a in L.PANELS['CMP'][0]):.2f}; as the panel: US${L.PANELS['CMP'][1]:.2f}.\n",
        '### Australia: Medicare Benefits Schedule pathology (A$)\n',
        'Chemistry items in group 66500 are priced by the number of items requested (1 to 5 or more: A$9.70, 11.65, 13.65, 15.65, 17.70), whatever the items. '
        'HDL cholesterol (A$11.05), iron studies (A$32.55), full blood count (A$17.80), HbA1c (A$16.80) and urine ACR (A$20.10) '
        'are separate items; a specimen-collection episode fee of A$5.95 (73928, approved collection centre) is added. Listed schedule fees are used, not '
        'Medicare benefits, and HbA1c has clinical eligibility conditions. The routine bundle costs A$69.30 without and A$58.25 with the episode-coning rule '
        '(at most three items paid per episode), which was examined as a sensitivity scenario; the matched panel costs A$41.45 either way.\n',
        '### Japan: medical fee schedule, 2026 (¥; 1 point = ¥10)\n',
        'Biochemistry (I) items are priced individually (11 or 17 points) up to four items, and as a band for five or more (93 points for 5–7 items, 99 for 8–9, '
        '103 for 10 or more), plus a biochemistry judgement fee (144 points). Bicarbonate is priced separately (48 points). Haematology: complete blood count '
        '36 points, HbA1c 49 points, judgement fee 125 points. CRP 16 points plus immunology judgement 144 points; urine albumin 99 + creatinine 11 + urine '
        'judgement 34 points; blood collection 40 points (outpatient venous collection, D400, per day). Judgement fees (D026) are charged once per '
        'category per month; the single-visit scenario is assumed.\n',
        '### China: public primary-care price lists, Shandong, 2026 (CN¥)\n',
        'Sources: two public price lists from Shandong primary care, combined as one price scenario. (1) The laboratory price list published on 28 April 2026 by '
        'a township health centre in Jining High-tech Zone (prices set under the Jining Healthcare Security Bureau documents [2025] No. 10 and [2026] No. 6): '
        'all analytes below except potassium, sodium, chloride, calcium and bicarbonate, and the five-part blood count (CN¥19). (2) The 2026 second-quarter '
        'price list of a township health centre in Jinling, Zhaoyuan (Yantai): potassium, sodium and chloride (CN¥3 each), calcium (CN¥4; the Jining list gives '
        f'CN¥15), bicarbonate (CN¥5) and venipuncture (CN¥{T.CN_VENI:.1f}). Each analyte is billed singly; the blood count is one item.\n',
        '| Item | Price (CN¥) |', '|---|---|']
out += [f'| {NAME.get(a, a)} | {v:.1f} |' for a, v in T.CN_PRICE.items() if a in NAME]
out += [f'| Blood count (five-part differential) | {T.CN_CBC:.1f} |',
        '\nRoutine reference in China: the routine check-up items measured in each cohort (CHNS CN¥74.8; CLHLS CN¥62.3).\n',
        'Understanding Society (UK) was evaluated under the US fee schedule, as prespecified; its routine '
        'reference is the routine bundle restricted to the 11 analytes measured (US$50.77: the CMP, lipid panel, CBC and HbA1c billed as such).\n',
        '## A2. Harmonisation across survey eras and countries\n',
        '- **Granulocyte percentage** = neutrophil + eosinophil + basophil percentage, because NHANES III reports granulocytes only.',
        '- **C-reactive protein**: NHANES III used a standard (not high-sensitivity) assay with an analytical detection limit of 0.3 mg/dL and recorded '
        'values below it as 0.21 mg/dL. To harmonise, values below 0.25 mg/dL were set to 0.21 mg/dL in every cohort (hs-CRP in mg/L was first divided by 10). '
        'CRP was not measured in NHANES 2011–14, so panels containing CRP were not evaluated there.',
        '- **Creatinine**: NHANES 1999–2000 and 2005–06 recalibrated as recommended by NHANES (1.013x + 0.147 and 0.978x − 0.016); NHANES III standardised '
        'as 0.960x − 0.184 (NHANES III laboratory documentation).',
        '- **Logarithms**: every analyte enters as ln(x + offset), with offset = 1% of the training median. Because the offset is small, a constant unit '
        'factor in a validation cohort shifts ln(x + offset) by approximately a constant, which changes mainly the intercept (calibration) and hardly the '
        'ranking of people (discrimination); the shift is exact only when the offset is rescaled with the unit.',
        '- **Urine albumin-to-creatinine ratio**: computed as urine albumin (mg/L, equal to µg/mL) divided by urine creatinine (mg/dL) in every cohort, which '
        'is the conventional ratio in mg/g divided by 100. Published coefficients use this scale; users with mg/g should divide by 100 (the web calculator '
        'does this automatically).',
        '- **CHNS**: columns named *_MG are mg/dL; urea (mmol/L) × 2.8 = BUN mg/dL; albumin, total protein and haemoglobin g/L ÷ 10; hs-CRP mg/L ÷ 10.',
        '- **CLHLS**: glucose mmol/L × 18.016; urea × 2.8; creatinine ÷ 88.4; uric acid µmol/L ÷ 59.48; cholesterol and HDL mmol/L × 38.67; '
        'triglycerides mmol/L × 88.57; hs-CRP mg/L ÷ 10; haemoglobin g/L ÷ 10. The biomarker documents label creatinine "mmol/L" but give a reference range '
        'of 50–120, which is the µmol/L scale; it was treated as µmol/L. Data-format problems and coding errors that were found and corrected are '
        'described in table S17.',
        '- **Understanding Society**: albumin g/L ÷ 10; urea × 2.8; creatinine µmol/L ÷ 88.4; cholesterol and HDL mmol/L × 38.67; triglycerides mmol/L × 88.57; '
        'hs-CRP mg/L ÷ 10 (below the detection limit set to the CRP floor); haemoglobin g/L ÷ 10; HbA1c IFCC mmol/mol to NGSP % as 0.09148x + 2.152. Values '
        'coded below the detection limit take the smallest valid value of that analyte, above the linear range the largest. Details are in '
        'table S19.\n',
        '## A3. Risk of bias (PROBAST self-assessment)\n',
        '| Domain | Development (NHANES 1999–2010) | External validation | Comment |', '|---|---|---|---|',
        '| Participants | Low | Low | Nationally representative surveys (NHANES), population cohorts (CHNS, CLHLS, Understanding Society); all adults in the age range with the required tests; no exclusions on other grounds. |',
        '| Predictors | Low | Low (US); unclear (China, UK) | Standardised laboratory assays measured without knowledge of outcome; era harmonisation as above. Chinese and UK cohorts used different laboratories and units; a constant unit factor mainly affects calibration. |',
        '| Outcome | Low | Low (NHANES); unclear (CHNS, CLHLS, Understanding Society) | NCHS public-use linked mortality in NHANES (follow-up months from the examination; for some records follow-up time or underlying cause was replaced with synthetic values, with vital status preserved). CHNS deaths from the recorded death year or the wave of first report (binary outcome); CLHLS deaths validated with dates but 30% of the 65–79 group censored alive before 5 years. Understanding Society: death known only by the survey wave in which it was reported (binary outcome); 26% of eligible people had unknown 9-year status and were not analysed. |',
        '| Analysis | Low | Low | 2081 training deaths (about 60 per candidate parameter for the largest panel); ridge penalty; all selection on training data; prespecified analysis plan (appendix A7); discrimination, calibration and net benefit reported; bootstrap intervals; independent re-implementation. CLHLS main analysis small (92 deaths); Understanding Society 479 deaths. |',
        '| Overall | Low | Low for US cohorts; unclear for Chinese and UK cohorts | Applicability: population surveys rather than care records (see Limitations). |\n']
QD = W / 'external_data' / 'pubmed_search'
out += ['## A5. Literature search (Research in context)\n',
        'PubMed via NCBI E-utilities; records published 1 January 1990 to 1 November 2026; search run on 8 November 2026; no language restriction; 52 unique '
        'records screened by title and abstract. Supplementary targeted searches covered named laboratory scores (PhenoAge, Intermountain Risk Score, '
        'PCAge, LinAge, ENABL Age), guideline equations (PREVENT, SCORE2, pooled cohort equations), machine-learning mortality scores from routine blood '
        'tests, and cost-aware learning (PubMed and arXiv). The four PubMed strings, exactly as run:\n']
out += [f'{i}. `{(QD / f"q{i}.txt").read_text(encoding="utf-8").strip()}`\n' for i in range(1, 5)]
out += ['## A6. Ethics\n',
        'This study is a secondary analysis of de-identified data obtained from the data providers under their standard access conditions; no new data were '
        'collected and no participant was contacted, so this secondary analysis did not require ethics approval. Each '
        'source study had ethical approval and informed consent:\n',
        '- **NHANES**: protocols approved by the NCHS Research Ethics Review Board (continuous NHANES protocols #98-12 for 1999–2004, #2005-06 for 2005–10 '
        'and #2011-17 for 2011–16; NHANES III approved by the NCHS institutional review board); written informed consent from all participants.',
        '- **CHNS**: approved by the institutional review boards of the University of North Carolina at Chapel Hill and the National Institute of Nutrition '
        'and Food Safety (now the National Institute for Nutrition and Health), Chinese Center for Disease Control and Prevention; written informed consent.',
        '- **CLHLS**: approved by the Research Ethics Committees of Peking University (IRB00001052-13074); informed consent from participants or proxies.',
        '- **Understanding Society**: the University of Essex Ethics Committee approved all main-survey data collection; collection of biosocial data by '
        'trained nurses in waves 2 and 3 was approved by the National Research Ethics Service (Understanding Society – UK Household Longitudinal Study: '
        'A Biosocial Component, Oxfordshire A REC, reference 10/H0604/2) (Understanding Society main survey user guide, ethics section).\n',
        '## A7. Analysis timeline\n',
        'All panels, models and decision rules were fixed in NHANES 1999–2010 before the cohorts below were analysed with them. Order of analysis:\n',
        '1. **Initial validations (7 November 2026).** NHANES III, and then NHANES 2011–14 and CHNS, were analysed with the same panels under two '
        'prespecified plans.',
        '2. **Final analysis plan (7 November 2026).** A single plan (hypotheses H1–H3, margins and secondary analyses) was fixed and dated, and '
        'NHANES III, NHANES 2011–14, CHNS and NHANES 2015–16 were analysed under it.',
        '3. **Protocol amendment: value-of-information ordering rule (8 November 2026).** Developed in training data only, after the earlier stopping '
        'rule had been evaluated; each NHANES cohort was then analysed once more (main-text H3).',
        '4. **Protocol amendments: local recalibration and the CoAI comparison (10 November 2026).** Specified before either analysis was run (CoAI '
        'comparison: main-text H4).',
        '5. **Additional cohorts (11 November 2026).** CLHLS and Understanding Society were analysed under the same plan. In CLHLS, two data-format '
        'problems and three coding errors were corrected before the final analysis (table S17).',
        '6. **Post hoc: missing results in use (12 November 2026).** The handling of missing results in the web calculator was evaluated in '
        'NHANES III and NHANES 2011–14 after all other analyses (table S20).\n']
(HERE / 'appendix_text.md').write_text('\n'.join(out), encoding='utf-8'); print('written', HERE / 'appendix_text.md')
