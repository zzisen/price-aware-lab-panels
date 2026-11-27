"""Pre-plan coverage check for cross-platform transfer (counts only; no drug-drug relations computed)."""
from itertools import permutations
from pathlib import Path
import pandas as pd
W = Path(__file__).resolve().parent.parent
long = pd.read_csv(W / 'external_data' / 'pharmacodb' / 'experiments_long.csv').dropna(subset=['AAC'])
mp = pd.read_csv(W / 'task5_validation' / 'cell_line_mapping.csv').dropna(subset=['unit'])
long = long.merge(mp, on=['dataset', 'cell_line'])
print('rows with mapped unit', len(long))
per = long.groupby('dataset').agg(units=('unit', 'nunique'), compounds=('compound', 'nunique'))
print(per.to_string())
sets_u = {d: set(g.unit) for d, g in long.groupby('dataset')}
sets_c = {d: set(g.compound) for d, g in long.groupby('dataset')}
ds = sorted(sets_u)
print('\nshared compounds (upper) / shared units (lower):')
rows = []
for a in ds:
    rows.append([len(sets_c[a] & sets_c[b]) if a < b else (len(sets_u[a] & sets_u[b]) if a > b else '-') for b in ds])
print(pd.DataFrame(rows, index=ds, columns=ds).to_string())

# For each ordered pair (A train, B test): compounds well covered in BOTH (>=90% of each platform's units),
# and B units NOT in A (fully independent units on the test platform).
print('\nordered pairs: compounds covered >=90% in both | B-only units | B units total')
for a, b in permutations(ds, 2):
    wa = long[long.dataset == a].pivot_table(index='unit', columns='compound', values='AAC', aggfunc='mean')
    wb = long[long.dataset == b].pivot_table(index='unit', columns='compound', values='AAC', aggfunc='mean')
    shared = [c for c in wa.columns.intersection(wb.columns)
              if wa[c].notna().mean() >= 0.9 and wb[c].notna().mean() >= 0.9]
    if len(shared) < 20:
        continue
    bonly = set(wb.index) - set(wa.index)
    full_b = wb.loc[list(bonly), shared].dropna() if bonly else wb.iloc[:0]
    full_a = wa[shared].dropna()
    print(f'{a:>7} -> {b:<7} shared {len(shared):4d} | A complete {len(full_a):4d} | B-only units {len(bonly):4d}, complete {len(full_b):4d} | B total {len(wb)}')
