"""Run the frozen per_target.py on the CiPA-28 folder (only the task folder differs)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import per_target as P  # noqa: E402
P.HERE = HERE
for name in ['CIPA28', 'CIPA28_T12']:
    import rulelib as L
    P.main('task8', name)
