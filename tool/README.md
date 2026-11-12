# labpanel

Choose the cheapest laboratory panel that carries the information needed for one prediction target, under a given price schedule, and order tests one block at a time only for people whose decision could still change.

Research software. It is not a medical device and its output is not medical advice.

## What it does

1. **Selection.** `Selector` scores sets of analytes by cross-validated Cox partial log-likelihood for your outcome. It then searches over orderable items, either single analytes or panels such as BMP, CMP, lipid panel and CBC, under a budget.
   - `best_affordable(budget, schedule)` returns the best set found that fits within the budget.
   - `cheapest_matching(reference, schedule)` returns the cheapest set that is at least as informative as a reference bundle.
2. **Prices.** Three schedules are built in: US Medicare CLFS 2026 Q4 (`US`), Australian MBS pathology (`AU`) and the Japanese 2026 fee schedule (`JP`).
   - A schedule is just a function from a set of analyte names to a bill, so you can supply your own.
   - The three schedules bill panels very differently, so the best panel for the same budget differs between them.
3. **Sequential ordering.** Tests are added one block at a time after free information (age, sex, BMI, blood pressure, smoking and diabetes).
   - **Rule used in the manuscript and the calculator:** value of information. A person goes on to the next block only if the expected value of what the remaining tests could change (value per net true positive, $30 000 in the main analysis) exceeds their cost. The fitted rule is frozen as a printable chart of risk intervals in `task22_adaptive/rule_t22.json`; `apply_chart(rule, risks)` applies it.
   - `SequentialPolicy` is a simpler generic alternative for new targets: a person stops when the predicted chance that the remaining tests would change the decision at threshold `tau` is below `eps`. Each stop rule is again a risk interval (`FlipModel.interval`). It was an earlier rule in this project and is not the one reported in the manuscript.

## Quick start

```python
import labpanel as lp
S = lp.Selector(Z, t, e)                       # Z: DataFrame with 'age', 'female' and log analytes; t months; e event
items, cost, score = S.beam(50.77, 'US')       # best panel within $50.77
cheap = S.cheapest_matching(['CMP', 'CBC', 'LIPID', 'hba1c'], 'US')
P = lp.SequentialPolicy(['age', 'female', 'bmi', 'sbp', 'smoker', 'diabetes'],
                        [lp.ORDERABLE['CMP'], lp.ORDERABLE['CBC'], ['acr']], tau=0.10).fit(Z, t, e)
step, risk, paid = P.run(Z_new, eps=0.02, step_costs=[0, 19.90, 27.67, 38.63])
```

Applying the manuscript's chart:

```python
import json
rule = json.load(open('task22_adaptive/rule_t22.json'))
step, risk, paid = lp.apply_chart(rule, {'clinical only': r0, 'CMP': r1, 'CMP+CBC': r2, 'CMP+CBC+acr': r3})   # 9-year risks at each step
```

Self-test: `python tests/test_basic.py`.

## Calculator

`calculator/index.html` is a single-file web page. It runs entirely in the browser and sends nothing anywhere. It has two tools:

- **Stepwise ordering.** It applies the value-of-information chart from the manuscript (CMP, then CBC, then urine ACR; decision threshold 10% 9-year risk). The value per net true positive can be set to $10 000, $30 000 (main) or $100 000.
- **Fixed panel.** It gives the 9-year risk from the matched panel, the target-aware panel at the routine price, or the routine bundle.

Missing results can be left empty (age and sex are required). The page replaces each missing input by its conditional mean given the inputs entered, using the training means and covariances of the model inputs (`calculator/missing.py` is the reference implementation), reports an 80% range for the risk, and names the tests to measure when that range crosses the decision threshold. Values outside the 0.5th–99.5th training percentiles are flagged for a unit check.

The page re-computes embedded test cases from the fitted models when it opens and reports whether its arithmetic matches.

Rebuilding:
1. `export_model_v2.py` fits the models on NHANES 1999-2010, adds the training means, covariances and ranges used for missing results, and copies the chart from `task22_adaptive/rule_t22.json` into `model_v2.json`.
2. `build_calculator_v2.py` writes `index.html`.

`model.json` and `build_calculator.py` are the earlier version, built with the earlier (generic) stopping rule; they are not used by the calculator.

## Licence

MIT.
