#!/usr/bin/env python3
"""Build calculator/index.html from model_v2.json. The models are embedded, so the page works offline and sends nothing anywhere.
Missing inputs are replaced by their conditional mean (see missing.py), with an 80% risk range.
Two tools: (A) stepwise ordering with the value-of-information chart (manuscript v0.4, Task 22); (B) 9-year risk from a fixed panel.
The page re-computes embedded Python test cases on load and reports whether its arithmetic matches."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
model = json.load(open(HERE / 'model_v2.json', encoding='utf-8'))
LABEL = {'glucose': 'Glucose', 'bun': 'Urea nitrogen (BUN)', 'creatinine': 'Creatinine', 'sodium': 'Sodium', 'potassium': 'Potassium',
         'chloride': 'Chloride', 'co2': 'Bicarbonate (CO₂)', 'calcium': 'Calcium', 'albumin': 'Albumin', 'protein': 'Total protein',
         'alp': 'Alkaline phosphatase', 'alt': 'ALT', 'ast': 'AST', 'bilirubin': 'Total bilirubin', 'phosphorus': 'Phosphorus', 'ggt': 'GGT',
         'wbc': 'White cells', 'lymph_pct': 'Lymphocytes', 'mono_pct': 'Monocytes', 'gran_pct': 'Granulocytes (neutrophils + eosinophils + basophils)',
         'hgb': 'Haemoglobin', 'mcv': 'MCV', 'rdw': 'RDW', 'plt': 'Platelets', 'mpv': 'MPV', 'acr': 'Urine albumin/creatinine ratio',
         'tc': 'Total cholesterol', 'hdl': 'HDL cholesterol', 'tg': 'Triglycerides', 'hba1c': 'HbA1c'}
CMP = ['glucose', 'bun', 'creatinine', 'sodium', 'potassium', 'chloride', 'co2', 'calcium', 'albumin', 'protein', 'alp', 'alt', 'ast', 'bilirubin']
CBC = ['wbc', 'lymph_pct', 'mono_pct', 'gran_pct', 'hgb', 'mcv', 'rdw', 'plt', 'mpv']
ALL = list(dict.fromkeys(CMP + CBC + ['phosphorus', 'ggt', 'acr', 'tc', 'hdl', 'tg', 'hba1c']))
U = model['units']; C = model['step_costs_US']; P = {k: model['models'][k] for k in ['matched', 'target-aware', 'routine']}


def fields(names, prefix):
    return '\n'.join(f'<label data-a="{a}"><span>{LABEL[a]} <em>{U.get(a, "")}</em></span><input type="number" step="any" inputmode="decimal" '
                     f'data-f="{prefix}{a}"></label>' for a in names)


def clin(prefix, with_clinical):
    s = (f'<label><span>Age <em>years, 40–79</em></span><input type="number" min="40" max="79" data-f="{prefix}age"></label>'
         f'<label><span>Sex</span><select data-f="{prefix}female"><option value="">–</option><option value="0">Male</option><option value="1">Female</option></select></label>')
    if with_clinical:
        s += (f'<label><span>BMI <em>kg/m²</em></span><input type="number" step="any" data-f="{prefix}bmi"></label>'
              f'<label><span>Systolic blood pressure <em>mmHg</em></span><input type="number" step="any" data-f="{prefix}sbp"></label>'
              f'<label><span>Current smoker</span><select data-f="{prefix}smoker"><option value="">–</option><option value="0">No</option><option value="1">Yes</option></select></label>'
              f'<label><span>Diabetes (told by a doctor)</span><select data-f="{prefix}diabetes"><option value="">–</option><option value="0">No</option><option value="1">Yes</option></select></label>')
    return s


panel_opts = ''.join(f'<option value="{k}">{k.capitalize()} panel — {" + ".join(v["items"])} (US ${v["price_US"]:.2f})</option>' for k, v in P.items())
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Price-Aware Lab Panels</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#66655f; --card:#ffffff; --line:#e3e1da; --accent:#1f5f8b; --go:#8a4b00; --stop:#2e6b3a; --hi:#a1262b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#161615; --fg:#ecebe6; --muted:#a3a29c; --card:#1f1f1d; --line:#353532; --accent:#7fb6dd; --go:#f0b061; --stop:#86c995; --hi:#f08a8e; }} }}
:root[data-theme="dark"] {{ --bg:#161615; --fg:#ecebe6; --muted:#a3a29c; --card:#1f1f1d; --line:#353532; --accent:#7fb6dd; --go:#f0b061; --stop:#86c995; --hi:#f08a8e; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:780px; margin:0 auto; padding:24px 16px 64px; }} h1 {{ font-size:1.5rem; margin:0 0 4px; }} p.lead {{ color:var(--muted); margin:0 0 16px; }}
nav {{ display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }} nav button {{ font:inherit; padding:8px 14px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }}
nav button[aria-selected="true"] {{ border-color:var(--accent); color:var(--accent); font-weight:600; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:14px 0; }} section.off {{ opacity:.45; }}
h2 {{ font-size:1.05rem; margin:0 0 10px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:10px 14px; }}
label {{ display:flex; flex-direction:column; gap:3px; font-size:.9rem; }} label em {{ color:var(--muted); font-style:normal; font-size:.8rem; }}
input, select {{ font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); width:100%; }}
.out {{ margin-top:12px; padding:10px 12px; border-radius:8px; border:1px solid var(--line); }} b.go {{ color:var(--go); }} b.stop {{ color:var(--stop); }} b.hi {{ color:var(--hi); }}
small, .note {{ color:var(--muted); }} .note {{ font-size:.88rem; }} [hidden] {{ display:none !important; }}
</style></head><body><main>
<h1>Price-aware laboratory panels for 9-year mortality risk</h1>
<p class="lead">Research tool accompanying the manuscript. Not a medical device and not medical advice. Models fitted on NHANES 1999–2010 (adults aged 40–79;
{model['n']:,} people, {model['deaths']:,} deaths). Everything runs in this page; nothing you enter is sent anywhere.</p>
<nav role="tablist"><button role="tab" aria-selected="true" data-tab="seq">Stepwise ordering</button><button role="tab" aria-selected="false" data-tab="fix">Fixed panel</button></nav>

<div id="tab-seq">
<section><h2>How it works</h2><p class="note">Start with free clinical information. After each step the page estimates 9-year risk and orders the next test only while
that risk lies in the printed interval, where finishing the tests is worth their price (value V per correctly flagged death; decision threshold
{int(100 * model['tau'])}% 9-year risk). Order: CMP (${C[1]:.2f}), then CBC (${C[2]:.2f} in total), then urine albumin/creatinine ratio (${C[3]:.2f} in total), US Medicare 2026.</p>
<label style="max-width:300px"><span>Value per correctly flagged death (V)</span><select id="V"><option value="30000" selected>$30 000 (main analysis)</option>
<option value="10000">$10 000 (fewer tests)</option><option value="100000">$100 000 (more tests)</option></select></label></section>
<section id="s0"><h2>Step 0 · Free clinical information</h2><div class="grid">{clin('s_', True)}</div><div class="out" id="o0"></div></section>
<section id="s1" class="off"><h2>Step 1 · Comprehensive metabolic panel (CMP)</h2><div class="grid">{fields(CMP, 's_')}</div><div class="out" id="o1"></div></section>
<section id="s2" class="off"><h2>Step 2 · Complete blood count (CBC)</h2><div class="grid">{fields(CBC, 's_')}</div><div class="out" id="o2"></div></section>
<section id="s3" class="off"><h2>Step 3 · Urine albumin/creatinine ratio</h2><div class="grid">{fields(['acr'], 's_')}</div><div class="out" id="o3"></div></section>
</div>

<div id="tab-fix" hidden>
<section><h2>Panel</h2><select id="panel">{panel_opts}</select>
<p class="note">Matched panel: the cheapest set at least as informative as the routine bundle. Target-aware panel: the most informative set within the routine price.</p></section>
<section><h2>Age, sex and test results</h2><div class="grid">{clin('f_', False)}{fields(ALL, 'f_')}</div><div class="out" id="of"></div></section>
</div>

<section><h2>Before using it elsewhere</h2><p class="note">Units are US conventional units, as listed beside each field (creatinine IDMS-standardised; granulocytes = neutrophils +
eosinophils + basophils). Missing results: leave the field empty. The page estimates each missing value from the results entered (its conditional mean in the
training data) and shows an 80% range for the risk; if the range crosses the decision threshold, it names the tests to measure. Age and sex are always required.
Calibration: where the observed-to-expected ratio of deaths is near 1, use the models unchanged; where it is clearly off (about 0.85 in the UK), recalibrate
on about 1000 local people with known outcomes.</p>
<p class="note" id="selftest"></p></section>
</main>
"""
LBL_ALL = dict(LABEL, age='Age', female='Sex', bmi='BMI', sbp='Systolic blood pressure', smoker='Smoking', diabetes='Diabetes')
JS = (HERE / 'calculator_script.js').read_text(encoding='utf-8')
html += (JS.replace('__MODEL__', json.dumps(model, ensure_ascii=False)).replace('__LABELS__', json.dumps(LBL_ALL, ensure_ascii=False))
         .replace('__CMP__', json.dumps(CMP)).replace('__CBC__', json.dumps(CBC)))
(HERE / 'index.html').write_text(html, encoding='utf-8')
print('wrote', HERE / 'index.html', len(html), 'bytes')
