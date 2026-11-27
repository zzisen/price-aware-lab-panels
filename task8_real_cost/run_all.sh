#!/bin/bash
# Task 8: all remaining tasks, 8 at a time, single-threaded BLAS (CIPA was run first as a smoke run of the same code).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
for t in NHANES DRUG_GDSC2_T1 DRUG_GDSC2_T2 DRUG_GDSC2_T3 DRUG_GDSC2_T4 DRUG_PRISM_T1 DRUG_PRISM_T2 DRUG_PRISM_T3 DRUG_CTRPv2_T1 DRUG_GDSC1_T1 DRUG_GDSC1_T2 DRUG_CTRPv2_T2 DRUG_GDSC1_T3 SAAD_LV; do echo $t; done | xargs -P 8 -I{} sh -c '../.env312/python.exe run_task8.py {} > logs/{}.txt 2>&1'
echo ALLDONE
