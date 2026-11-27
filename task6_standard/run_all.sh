#!/bin/bash
# Runs validate_standard.py on the 19 validation tasks, 16 at a time, single-threaded BLAS.
cd "$(dirname "$0")/../task5_validation"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
# Largest tasks first
for t in GDSC1_T1 GDSC1_T2 GDSC1_T3 GDSC2_T1 GDSC2_T2 GDSC2_T3 GDSC2_T4 PRISM_T1 PRISM_T2 PRISM_T3 CTRPv2_T1 CTRPv2_T2 BeatAML_T1 BeatAML_T2 NCI60_T1 NCI60_T2 NCI60_T3 NCI60_T4 HumanAtrial_T1; do echo $t; done | xargs -P 16 -I{} sh -c '../.env312/python.exe validate_standard.py {} > logs/std_{}.txt 2>&1'
echo ALLDONE
