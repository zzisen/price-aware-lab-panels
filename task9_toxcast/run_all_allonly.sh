#!/bin/bash
# Task 9: the six Tox21 tasks at n = all (the size the frozen decision rule uses), 6 at a time, single-threaded BLAS.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
for t in TOX21_T1 TOX21_T2 TOX21_T3 TOX21_T4 TOX21_T5 TOX21_T6; do echo $t; done \
  | xargs -P 6 -I{} sh -c '../.env312/python.exe run_task9.py {} --all-only > logs/{}.txt 2>&1'
echo ALLDONE
