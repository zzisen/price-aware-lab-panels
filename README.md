# Price-aware laboratory panels for mortality risk: code and models

Code, fitted models, the ordering chart and the offline calculator for the article "Price-aware selection of laboratory panels for mortality risk: development and external validation in five cohorts from three countries" (Zhou Z, Zhang Y, Pang H; submitted). Please cite this repository with the DOI shown on its Zenodo record (see `CITATION.cff`).
Research software; not a medical device.

## Contents

| Folder | What it does |
|---|---|
| `tool/` | `labpanel` package (price-aware selection, fee schedules, sequential ordering) and the offline web calculator (`tool/calculator/index.html`). |
| `task19_dm2/` | Training-data selection of the US panels (`final_select.py` -> `selected_all.json`). |
| `task20_chns/` | Selection within the analytes measured in CHNS, Chinese prices (`selected_cn.json`). |
| `task21_final/` | Validation in each cohort (`run_t21.py`, `run_t21_chns.py`), pooling (`pool_t21.py`) and an independent re-implementation (`verify_t21.py`); cardiovascular panels and lasso comparators. |
| `task22_adaptive/` | Value-of-information ordering rule (`select_t22.py` -> `rule_t22.json`), validation and independent check. |
| `task23_local/` | Local recalibration analyses. |
| `task24_coai/` | Head-to-head comparison with CoAI (re-implementation of github.com/suinleelab/coai, MIT licence). |
| `task25_newdata/` | CLHLS data reading, selection within its menu, validation and checks. |
| `task26_transport/` | Exploratory re-centring analysis. |
| `task27_ukhls/` | Understanding Society data reading, selection within its menu (US prices), validation and checks. |
| `task28_missing/` | Post hoc evaluation of the calculator's handling of missing results (appendix table S20). |
| `manuscript_ldh/` | Scripts that produce the tables and appendix; `model_coefficients.json` (all panel models). |
| `manuscript_ldh/figures_v2/` | `export_figure_data.py` writes the plotted values to `data/` from the analysis outputs; `make_figures_v2.py` draws Figures 1–5 and appendix figures S1–S4 from those files. |
| `DATA_DICTIONARY.md` | Derived variables: names, units, transformations and outcome definitions in each cohort. |

All panels, rules and model coefficients were fixed on NHANES 1999–2010 only. The statistical analysis plan is available from the corresponding author on request.

## Data

No survey data are included. Place the original files under `external_data/` next to these folders:
- NHANES 1999–2016 and NHANES III with the NCHS public-use linked mortality files: https://www.cdc.gov/nchs/nhanes/
- CHNS: https://www.cpc.unc.edu/projects/china (registration required)
- CLHLS: Peking University Open Research Data Platform (application required)
- Understanding Society: UK Data Service, SN 6614 and SN 7251 (End User Licence)
- US Medicare Clinical Laboratory Fee Schedule 2026 Q4: https://www.cms.gov/medicare/payment/fee-schedules/clinical-laboratory-fee-schedule-clfs

## Requirements

Python 3.12 with numpy, pandas, scikit-learn, matplotlib and pyreadstat. Analyses are single-threaded and run on a laptop.

## Licence

MIT (see `LICENSE`).
