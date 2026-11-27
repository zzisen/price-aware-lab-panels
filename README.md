# When to redesign measurement panels in biology and medicine: code and models

Code, fitted models, the ordering chart and the offline calculator for the article "When to redesign measurement panels in biology and medicine" (Zhou Z, Zhang Y, Pang H; submitted). Earlier versions (1.0.0–1.1.0) accompanied a clinical-only version of this work. Please cite this repository as https://doi.org/10.5281/zenodo.24149614 (all versions; see `CITATION.cff`).
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
| `task29_revision/` | Post hoc analyses (appendix A4.4, tables S21–S24): selection and validation with the expanded US fee menu (our search and CoAI), programme-based health-check references (CHNS, CLHLS, Understanding Society), censoring and unknown-status weighting, prediction horizons, and pooling sensitivity. Selections and aggregate results are included. |
| `task30_extra/` | Further post hoc analyses (tables S25–S27): paired differences after multiple imputation in NHANES III, coverage of the calculator's 80% risk range, and the split-level spread of the cross-domain benchmark. Aggregate results are included. |
| `manuscript_ldh/` | Scripts that produce the tables and appendix; `model_coefficients.json` (all panel models). |
| `manuscript_ldh/figures_v2/` | `export_figure_data.py` writes the plotted values to `data/` from the analysis outputs; `make_figures_v2.py` draws Figures 1–5 and appendix figures S1–S4 from those files. |
| `base_v32/`, `task3_learning_curve/`, `task3b_external/`, `task4_adaptive/` | GDSC data preparation, learning curves in GDSC1 and GDSC2, PRISM, and the first adaptive rules. |
| `task5_validation/` to `task11_transfer/` | Multi-source benchmark (Beat AML, NHANES, PKIS2, FPMTB, PACE, TCGA RPPA, GTEx): standard selectors, budget-matched comparisons, measured costs, ToxCast, reversal by training size, transfer across sources. |
| `task12_r5/` to `task17_gates/` | The target-resolution method (R5) and its confirmatory and gate analyses. `task12_r5/r5lib.py` (and its copy in `task13_confirm/frozen_code/`) includes a fix: an empty inner dimension in one einsum is taken as zero (its mathematical value); without it, some processes returned inf or NaN for single-target groups. |
| `task34_decision/`, `task36_rule_v3/` | Development of the molecular decision rule (n/k and cross-validation). |
| `task39_locked_v3/`, `task40_locked_v4/`, `task42_locked_v5/`, `task48_singlecell/` | Locked tests 2–5 (CRISPR, proteomics and new tissues; TCGA RNA-seq and RNAi; trees and QR + lasso in 12 TCGA cohorts; single-cell metacells), with pool builders, run scripts and summaries. |
| `task41_threshold/`, `task44_coessential/`, `task45_independence/`, `task47_learnability_scale/` | Simulation of the threshold, co-essential gene groups, overlap of units between sources, and the scale of the learnability check. |
| `task18_dm/`, `task32_tier2/`, `task33_clhls_ageing/`, `task35_fee_ablation/`, `task37_clinical_rule/`, `task38_clinical_stats/`, `task43_clinical_extras/`, `task46_clinical_post/` | Clinical analyses added for this article: development of the clinical decision rule, fee representation, additional statistics, comparisons and subgroup analyses (code only; outputs come from survey data). |
| `manuscript_nc/figs_v2/` | Scripts that draw Figures 1–7 and Supplementary Figures 1–8. |
| `v3_release_copy/analysis/v3/code/adaptive_resolution.py` | Module from our earlier release (https://doi.org/10.5281/zenodo.23113843), imported by some cross-domain scripts. |
| `DATA_DICTIONARY.md` | Derived variables: names, units, transformations and outcome definitions in each cohort. |

All clinical panels, rules and model coefficients were fixed on NHANES 1999–2010 only.

## Data

No data are included. Cross-domain data are public: Beat AML, FPMTB (Zenodo 7370747), PACE (Bioconductor BloodCancerMultiOmics2017), PKIS2, TCGA RPPA and RNA-seq (UCSC Xena), GTEx v8, DepMap 24Q4 CRISPR, CCLE proteomics (Nusinow et al. 2020), DEMETER2 RNAi (figshare 6025238), 10x Genomics purified blood-cell populations (Zheng et al. 2017), GDSC1 and GDSC2, PRISM and ToxCast. The `download_*.sh` scripts fetch the files used in the locked tests.

No survey data are included. Place the original files under `external_data/` next to these folders:
- NHANES 1999–2016 and NHANES III with the NCHS public-use linked mortality files: https://www.cdc.gov/nchs/nhanes/
- CHNS: https://www.cpc.unc.edu/projects/china (registration required)
- CLHLS: Peking University Open Research Data Platform (application required)
- Understanding Society: UK Data Service, SN 6614 and SN 7251 (End User Licence)
- US Medicare Clinical Laboratory Fee Schedule 2026 Q4: https://www.cms.gov/medicare/payment/fee-schedules/clinical-laboratory-fee-schedule-clfs

## Reproducing the figures and appendix

Figures 1–5 and appendix figures S1–S4 can be redrawn from the CSV files in `manuscript_ldh/figures_v2/data/` with `make_figures_v2.py`. Regenerating those CSV files with `export_figure_data.py` needs the outputs of each task folder, which are produced from the survey data; appendix figure S4 (cross-domain benchmark) needs benchmark summaries that are not part of this release, so its plotted values are provided as `S4_benchmark.csv`. The literature-search strings are listed in appendix A5 of the article.

## Versions

- 1.2.0: accompanies the article above. Adds the cross-domain analyses, the molecular and clinical decision rules, locked tests 2–5 and the figure scripts; fixes the einsum fault in `task12_r5/r5lib.py`.
- 1.1.0: adds the post hoc analyses reported in the submitted article (`task29_revision/`, `task30_extra/`; appendix A4.4 and tables S21–S27), the appendix-table scripts that print them, and the final figure scripts and plotted values (including the flag-everyone curve in appendix figure S3 and the separate development path of the ordering chart in figure 1B).
- 1.0.1: the stepwise calculator now flags missing values whenever the 80% risk range is not entirely inside or entirely outside an ordering band; citation metadata include the DOI.
- 1.0.0: first release.

## Requirements

Python 3.12 with numpy, pandas, scikit-learn, matplotlib and pyreadstat. Analyses are single-threaded and run on a laptop.

## Licence

MIT (see `LICENSE`).
