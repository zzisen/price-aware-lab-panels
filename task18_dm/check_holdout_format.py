"""Allowed pre-freeze check (DATA_SPLIT.md section 4): held-out files' variable names and row counts only. Computes no statistics,
prints no values, and does not parse the mortality files beyond counting lines."""
import pandas as pd
import dmlib as L
need = {'demo': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'WTMEC2YR'],
        'bio': ['SEQN'] + [v for s, v, _ in L.CHEM.values() if s == 'bio'], 'tc': ['SEQN', 'LBXTC'], 'hdl': ['SEQN', 'LBDHDD|LBXHDD|LBDHDL'],
        'cbc': ['SEQN'] + list(L.CBC.values()), 'ghb': ['SEQN', 'LBXGH'], 'crp': ['SEQN', 'LBXCRP'], 'acr': ['SEQN', 'URXUMA', 'URXUCR']}
clin = {'BMX': ['BMXBMI'], 'BPX': ['BPXSY1'], 'SMQ': ['SMQ020', 'SMQ040'], 'DIQ': ['DIQ010']}
for c in ['E', 'F', 'A', 'B']:
    f, d = L.FILES[c], L.HOLD / c
    for k, vs in need.items():
        cols = set(pd.read_sas(d / f'{f[k]}.xpt', format='xport', chunksize=1).columns)   # header only
        miss = [v for v in vs if not any(a in cols for a in v.split('|'))]
        print(c, k, f[k], 'missing:', miss or 'none', ('hdl names present: ' + str(sorted(x for x in cols if 'HD' in x))) if k == 'hdl' else '')
    for k, vs in clin.items():
        cols = set(pd.read_sas(d / f'{k}{L.CLIN_SFX[c]}.xpt', format='xport', chunksize=1).columns)
        print(c, k, 'missing:', [v for v in vs if v not in cols] or 'none')
    n = sum(1 for _ in open(d / f"NHANES_{f['mort']}_MORT_2019_PUBLIC.dat"))
    print(c, 'mortality file lines', n)
