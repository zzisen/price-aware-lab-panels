#!/usr/bin/env python3
"""Task 8: run_task8.py with the memory-safe, equivalent Global3 solver (global3_fast.py; equality checked in
check_equal.py, 210/210 identical). Used only for SAAD_LV (36 targets), where the frozen enumerator runs out of memory."""
import sys
import rulelib_patch  # noqa: F401
import run_task8

if __name__ == '__main__':
    run_task8.main(sys.argv[1])
