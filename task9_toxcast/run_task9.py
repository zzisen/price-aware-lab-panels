#!/usr/bin/env python3
"""Task 9 (plan_toxcast.md): run the frozen Task 5/6/7 pipelines unchanged on one Tox21 task.
Same entry point as task8_real_cost/run_task8.py; only the task folder differs, so rulelib and the three
validate_* modules are imported and called exactly as frozen.
Run with .env312:  python run_task9.py <task> [--time]"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L  # noqa: E402
import validate_generic  # noqa: E402
import validate_standard  # noqa: E402
import validate_budget  # noqa: E402


def main(name, timing=False, all_only=False):
    task = next(t for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    L.HERE = HERE
    if 'grid' in task:
        L.GRID = task['grid']
    if all_only:
        # Only the full training size, the one the plan's decision rule uses. At the small sizes a target whose
        # activity rate is a fraction of a percent is constant within the subsample, so its training SD is ~0, the
        # standardised losses reach 1e22 and explained variance is undefined (diag_scale.py).
        L.GRID = [None]
    if timing:
        validate_standard.main(name, True)
        validate_budget.main(name, True)
        return
    validate_generic.main(name)
    validate_standard.main(name)
    validate_budget.main(name)


if __name__ == '__main__':
    main(sys.argv[1], '--time' in sys.argv, '--all-only' in sys.argv)
