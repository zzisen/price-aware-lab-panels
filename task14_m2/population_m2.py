#!/usr/bin/env python3
"""Task 14, plan section 6: sex and age of the three NHANES cohorts actually analysed. Reads only DEMO and the unit column.
Run with .env312:  python -I -X utf8 population_m2.py"""
from pathlib import Path
import pandas as pd

W = Path(__file__).resolve().parent.parent
for name, folder, sfx in [('NH_A1_I', 'nhanes_2015_2016', 'I'), ('NH_A2_H', 'nhanes_2013_2014', 'H'), ('NH_A3_G', 'nhanes_2011_2012', 'G')]:
    demo = pd.read_sas(W / 'external_data' / folder / f'DEMO_{sfx}.xpt', format='xport').set_index('SEQN')
    unit = pd.read_csv(W / 'task14_m2' / 'matrices' / f'{name}.csv', usecols=['unit']).unit.astype(float)
    p = demo.loc[unit, ['RIDAGEYR', 'RIAGENDR']]
    print(f'{name}: n {len(p)} | women {(p.RIAGENDR == 2).mean():.3f} | age {p.RIDAGEYR.min():.0f}-{p.RIDAGEYR.max():.0f} | median age {p.RIDAGEYR.median():.0f} | age>=50 {(p.RIDAGEYR >= 50).mean():.3f}')
