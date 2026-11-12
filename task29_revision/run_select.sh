#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY=../.env312/python.exe
for m in us uk cn; do $PY -I t29_select.py $m > sel_${m}_print.txt 2>&1; echo "$m exit $?" >> run_select_log.txt; done
echo done >> run_select_log.txt
