#!/usr/bin/env python3
"""Runs the frozen summarize_v.py on the rerun (DEVIATIONS D3). Only its two folder names change: results_fixed/ in, summary_fixed/ out. Its rules and code are not edited.
Run with .env312 after rerun_fixed.py."""
from pathlib import Path
import summarize_v as S

HERE = Path(__file__).resolve().parent
S.RES = HERE / 'results_fixed'
S.OUT = HERE / 'summary_fixed'

if __name__ == '__main__':
    S.main()
