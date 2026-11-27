#!/usr/bin/env python3
"""Task 34 DEVELOPMENT run on sources already used in Task 17 (12 GTEx tissues in external_data/gtex_v8_tissues, 12 RPPA cancers).
Same code path as run_t34.py (t34lib.run_job, same p levels, n, k, splits); only the sources, seeds and output folder differ.
Output: dev_results/, dev_run_log.txt, dev_sources.json"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t34lib as T  # noqa: E402
T.GL.low_priority()
OUT = HERE / 'dev_results'; OUT.mkdir(exist_ok=True)
TISSUES = ['muscle_skeletal', 'skin_sun_exposed_lower_leg', 'adipose_subcutaneous', 'artery_tibial', 'thyroid', 'nerve_tibial', 'lung',
           'esophagus_mucosa', 'heart_left_ventricle', 'colon_transverse', 'breast_mammary_tissue', 'pancreas']
DEV_CANCERS = ['KIRC', 'LGG', 'OV', 'UCEC', 'THCA', 'LUAD', 'STAD', 'COAD', 'PRAD', 'HNSC', 'BLCA', 'LUSC']
log = open(HERE / 'dev_run_log.txt', 'a', encoding='utf-8')


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


def run(source, x, tasks):
    for tk in tasks:
        for o in range(T.OUTER):
            f = OUT / f"{tk['task']}_p{tk['p']}_o{o}.csv"
            if not f.exists():
                say(tk['task'], tk['p'], o, round(T.run_job(tk, x, o, f), 1), 's')


meta = []
for i, tissue in enumerate(TISSUES):
    x, fam, extra = T.gtex_pool(tissue, 'gtex_v8_tissues', 20262400 + i)
    tasks = T.make_tasks('G_' + tissue, x, fam, extra, 2400000 + i * 10, (71, 200, 1000))
    meta.append(dict(source='G_' + tissue, units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks}))); say('pool', meta[-1])
    run('G_' + tissue, x, tasks)
rp = T.rppa_pools(DEV_CANCERS, min_n=150)
for i, (c, (x, fam, extra)) in enumerate(sorted(rp.items())):
    tasks = T.make_tasks('R_' + c, x, fam, extra, 2500000 + i * 10, (71, 71 + len(extra)))
    meta.append(dict(source='R_' + c, units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks}))); say('pool', meta[-1])
    run('R_' + c, x, tasks)
json.dump(meta, open(HERE / 'dev_sources.json', 'w'), indent=1)
say('done')
