# Data dictionary (derived variables)

Every model uses the transformed inputs below; offsets are 1% of the NHANES 1999–2010 training median. Harmonisation of units across survey eras and countries is described in appendix A2 of the article and implemented in the loaders of each task folder.

| Code | Variable | Unit | Transformation |
|---|---|---|---|
| `age` | Age (years) | years | as recorded |
| `female` | Female sex (1 = female, 0 = male) | — | as recorded |
| `bmi` | Body-mass index | kg/m² | as recorded |
| `sbp` | Systolic blood pressure | mmHg | as recorded |
| `smoker` | Current smoker (1/0) | — | as recorded |
| `diabetes` | Diabetes told by a doctor (1/0) | — | as recorded |
| `glucose` | Glucose | mg/dL | ln(x + 0.94) |
| `bun` | Urea nitrogen | mg/dL | ln(x + 0.13) |
| `creatinine` | Creatinine (IDMS-standardised) | mg/dL | ln(x + 0.008642) |
| `sodium` | Sodium | mmol/L | ln(x + 1.39) |
| `potassium` | Potassium | mmol/L | ln(x + 0.04) |
| `chloride` | Chloride | mmol/L | ln(x + 1.035) |
| `co2` | Bicarbonate | mmol/L | ln(x + 0.25) |
| `calcium` | Calcium | mg/dL | ln(x + 0.094) |
| `albumin` | Albumin | g/dL | ln(x + 0.042) |
| `protein` | Total protein | g/dL | ln(x + 0.072) |
| `alp` | Alkaline phosphatase | U/L | ln(x + 0.7) |
| `alt` | ALT | U/L | ln(x + 0.22) |
| `ast` | AST | U/L | ln(x + 0.23) |
| `bilirubin` | Total bilirubin | mg/dL | ln(x + 0.007) |
| `ggt` | GGT | U/L | ln(x + 0.23) |
| `uric_acid` | Uric acid | mg/dL | ln(x + 0.054) |
| `ldh` | LDH | U/L | ln(x + 1.34) |
| `phosphorus` | Phosphorus | mg/dL | ln(x + 0.037) |
| `iron` | Iron | µg/dL | ln(x + 0.81) |
| `tg` | Triglycerides | mg/dL | ln(x + 1.29) |
| `tc` | Total cholesterol | mg/dL | ln(x + 2.03) |
| `hdl` | HDL cholesterol | mg/dL | ln(x + 0.5) |
| `wbc` | White cell count | 10³/µL | ln(x + 0.069) |
| `lymph_pct` | Lymphocytes | % | ln(x + 0.298) |
| `mono_pct` | Monocytes | % | ln(x + 0.078) |
| `gran_pct` | Granulocytes (neutrophils + eosinophils + basophils) | % | ln(x + 0.622) |
| `hgb` | Haemoglobin | g/dL | ln(x + 0.143) |
| `mcv` | Mean cell volume | fL | ln(x + 0.905) |
| `rdw` | Red cell distribution width | % | ln(x + 0.126) |
| `plt` | Platelets | 10³/µL | ln(x + 2.53) |
| `mpv` | Mean platelet volume | fL | ln(x + 0.08) |
| `hba1c` | HbA1c | % | ln(x + 0.056) |
| `crp` | C-reactive protein | mg/dL | ln(x + 0.0021) |
| `acr` | Urine albumin-to-creatinine ratio | mg/g | ln(x + 0.0007217) with x = albumin (mg/L) / creatinine (mg/dL), i.e. mg/g ÷ 100 |

## Outcomes

| Cohort | Outcome | Source |
|---|---|---|
| NHANES 1999–2010 (training), NHANES III, NHANES 2011–14, NHANES 2015–16 | All-cause death within 9, 9, 5 and 3 years of the examination (time to event); cardiovascular death by underlying cause | NCHS public-use linked mortality files, 2019 |
| CHNS | Death by the 2015 survey (binary), from the recorded year of death or the wave of first report | CHNS survey files |
| CLHLS | Death within 5 years of the blood draw (time to event), from dated death reports | CLHLS longitudinal data 1998–2018 |
| Understanding Society | Death reported by the wave fielded nine waves after the nurse visit (binary); unknown status not analysed | Cross-wave identifier file (dcsedfl_dv, dcsedw_dv, lwenum_dv), SN 6614 |
