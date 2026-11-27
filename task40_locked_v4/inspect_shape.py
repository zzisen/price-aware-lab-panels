"""Structure only (shapes, missingness, sample counts); no values relating candidates and targets are examined."""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
E = Path(sys.argv[1])
d = pd.read_csv(E / 'demeter2_rnai' / 'D2_combined_gene_dep_scores.csv', index_col=0)
print('D2 genes x lines', d.shape, 'first idx', d.index[:2].tolist(), 'cols', d.columns[:2].tolist())
na = d.isna()
print('genes complete in all lines', int((~na.any(axis=1)).sum()))
for q in (0.9, 0.95, 0.99):
    g = na.mean(axis=1) <= 1 - q
    lines_complete = int((~na[g].any(axis=0)).sum())
    print(f'genes measured in >= {q:.0%} lines: {int(g.sum())}; lines complete on those: {lines_complete}')
for f in sorted((E / 'tcga_rnaseq_xena').glob('*.gz')):
    with gzip.open(f, 'rt') as h:
        hdr = h.readline().rstrip('\n').split('\t')[1:]
    s = pd.Series(hdr)
    print(f.name, 'samples', len(s), 'primary(-01)', int(s.str.endswith('-01').sum()))
