#!/bin/bash
# Runs validate_budget.py on the 19 validation tasks, 8 at a time, single-threaded BLAS.
cd "$(dirname "$0")/../task5_validation"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
cat validation_tasks.txt | xargs -P 8 -I{} sh -c '../.env312/python.exe validate_budget.py {} > logs/budget_{}.txt 2>&1'
echo ALLDONE
