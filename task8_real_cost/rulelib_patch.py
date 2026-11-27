"""Swap the Global3 solver in rulelib for the equivalent global3_fast version (imported for its side effect)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'task5_validation'))
import rulelib as L  # noqa: E402
from global3_fast import globally_optimal_actions  # noqa: E402
L.globally_optimal_actions = globally_optimal_actions
