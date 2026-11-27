"""Technical check before running: any zero-variance column would make explained variance undefined."""
import json
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
for t in json.loads((HERE / 'tasks.json').read_text(encoding='utf-8')):
    m = pd.read_csv(HERE / 'matrices' / f"{t['task']}.csv").set_index('unit')
    sd = m.std(ddof=1)
    tg = sd[t['targets']]
    cd = sd.drop(index=t['targets'])
    print(t['task'], 'target sd min', round(float(tg.min()), 5), 'cand sd min', round(float(cd.min()), 5),
          'zero-var targets', int((tg == 0).sum()), 'zero-var cands', int((cd == 0).sum()),
          '| target sd quantiles', tg.quantile([0, .25, .5, 1]).round(4).to_dict())
