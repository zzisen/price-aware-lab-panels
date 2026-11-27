"""Before freezing: structure only (matrix shapes, families available, pool sizes, number of tasks). No model is fitted and no
relation between candidates and targets is computed."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t40lib as L
for c in L.TCGA10:
    x = L.tcga_matrix(c); xp, fam, extra = L.generic_pool(x, 0)
    print('T', c, 'units', x.shape[0], 'genes', x.shape[1], 'pool', xp.shape[1], 'families', len(set(fam.values())), 'extra', len(extra))
for n, x in L.rnai_sources().items():
    xp, fam, extra = L.generic_pool(x, 0)
    print('N', n, 'units', x.shape[0], 'genes', x.shape[1], 'pool', xp.shape[1], 'families', len(set(fam.values())), 'extra', len(extra))
