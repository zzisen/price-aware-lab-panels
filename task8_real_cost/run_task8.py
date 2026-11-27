#!/usr/bin/env python3
"""Task 8 (plan_real_cost.md): run the frozen Task 5/6/7 pipelines unchanged on one Task 8 task.
Only the task folder (tasks.json, matrices, outputs) and, for NHANES, the size grid differ.
Writes validation/, standard/ and budget/ under this folder. Run with .env312:  python run_task8.py <task>"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L  # noqa: E402
import validate_generic  # noqa: E402
import validate_standard  # noqa: E402
import validate_budget  # noqa: E402


def main(name):
    task = next(t for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')) if t['task'] == name)
    L.HERE = HERE
    if 'grid' in task:
        L.GRID = task['grid']
    validate_generic.main(name)
    validate_standard.main(name)
    validate_budget.main(name)


if __name__ == '__main__':
    main(sys.argv[1])
