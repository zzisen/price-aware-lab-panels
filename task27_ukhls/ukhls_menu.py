#!/usr/bin/env python3
"""Understanding Society (UKHLS) menu, units and prices (Task 27; written from the SN 7251 documentation and variable labels only).
Blood file xlabblood_ns (nurse health assessment, waves 2-3, 2010-12). Analytes shared with the NHANES menu (13):
  albumin (alb, g/L / 10), alkaline phosphatase (alkp, U/L), ALT (alt, U/L), AST (ast, U/L), urea (ure, mmol/L x 2.8 = BUN mg/dL),
  creatinine (ecre, umol/L / 88.4), GGT (ggt, U/L), total cholesterol (chol, mmol/L x 38.67), HDL (hdl, mmol/L x 38.67), triglycerides
  (trig, mmol/L x 88.57), HbA1c (hba1c, IFCC mmol/mol -> NGSP % = 0.09148 x IFCC + 2.152), hs-CRP (hscrp, mg/L / 10, then the Task 19 CRP floor),
  haemoglobin (hgb, g/L / 10). Negative values are Understanding Society missing codes. A unit factor changes calibration only, not discrimination.
Prices: US Medicare 2026 (dm2lib.cost_us) - there is no public UK laboratory fee schedule in the analysis; the Task 24 plan anticipated US prices.
Orderable items: each chemistry singly; the CMP (billed as the panel) for the six CMP analytes measured; the lipid panel; the blood count (only
haemoglobin measured); HbA1c; hs-CRP. Routine reference: the routine bundle (CMP + CBC + lipids + HbA1c) restricted to the analytes measured."""
import sys
from pathlib import Path
W = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
CMP_UK = ['albumin', 'alp', 'alt', 'ast', 'bun', 'creatinine']
MENU = CMP_UK + ['ggt', 'tc', 'hdl', 'tg', 'hba1c', 'crp', 'hgb']
ITEMS_UK = {a: [a] for a in CMP_UK + ['ggt', 'tc', 'hdl', 'tg']}
ITEMS_UK.update({'CMP_UK': CMP_UK, 'LIPID': ['tc', 'hdl', 'tg'], 'CBC_UK': ['hgb'], 'hba1c': ['hba1c'], 'crp': ['crp']})
ROUTINE_UK = [a for a in M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']) if a in MENU]
UNITS = {'albumin': ('alb', 0.1), 'alp': ('alkp', 1.0), 'alt': ('alt', 1.0), 'ast': ('ast', 1.0), 'bun': ('ure', 2.8), 'creatinine': ('ecre', 1 / 88.4),
         'ggt': ('ggt', 1.0), 'tc': ('chol', 38.67), 'hdl': ('hdl', 38.67), 'tg': ('trig', 88.57), 'crp': ('hscrp', 0.1), 'hgb': ('hgb', 0.1)}


def install():
    M.ITEMS.update(ITEMS_UK)


def hba1c_ngsp(ifcc):
    return 0.09148 * ifcc + 2.152
