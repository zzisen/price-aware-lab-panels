#!/usr/bin/env python3
"""Task 34 locked test (PLAN_FROZEN.md). Builds the fresh pools (GTEx tissues never used before; RPPA cancers never analysed) and runs every job
once. One process, one numeric thread, low priority. Finished jobs are skipped on restart. Output: results/<task>_p<p>_o<o>.csv, run_log.txt"""
import json
import sys
import time
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t34lib as T  # noqa: E402
T.GL.low_priority()
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)
TISSUES = ['skin_not_sun_exposed_suprapubic', 'adipose_visceral_omentum', 'artery_aorta', 'artery_coronary', 'esophagus_muscularis',
           'esophagus_gastroesophageal_junction', 'colon_sigmoid', 'stomach', 'testis', 'heart_atrial_appendage', 'cells_cultured_fibroblasts',
           'adrenal_gland', 'pituitary', 'spleen', 'prostate', 'brain_cerebellum', 'liver']
USED = {'KIRC', 'LGG', 'OV', 'UCEC', 'THCA', 'LUAD', 'STAD', 'COAD', 'PRAD', 'HNSC', 'BLCA', 'LUSC', 'BRCA'}
log = open(HERE / 'run_log.txt', 'a', encoding='utf-8')


def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()


meta = []
for i, tissue in enumerate(TISSUES):
    x, fam, extra = T.gtex_pool(tissue, 'gtex_v8_tissues_fresh', 20263400 + i)
    tasks = T.make_tasks('G_' + tissue, x, fam, extra, 3400000 + i * 10, (71, 200, 1000))
    meta.append(dict(source='G_' + tissue, units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks})))
    say('pool', meta[-1])
    for tk in tasks:
        for o in range(T.OUTER):
            f = OUT / f"{tk['task']}_p{tk['p']}_o{o}.csv"
            if not f.exists():
                say(tk['task'], tk['p'], o, round(T.run_job(tk, x, o, f), 1), 's')
p = pd.read_csv(T.EXT / 'tcga_rppa_xena' / 'Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', usecols=['sample', 'cancer type abbreviation'])
cands = sorted(set(p['cancer type abbreviation'].dropna()) - USED)
rp = T.rppa_pools(cands, min_n=150)
for i, (c, (x, fam, extra)) in enumerate(sorted(rp.items())):
    tasks = T.make_tasks('R_' + c, x, fam, extra, 3500000 + i * 10, (71, 71 + len(extra)))
    meta.append(dict(source='R_' + c, units=len(x), tasks=len({t['task'] for t in tasks}), p=sorted({t['p'] for t in tasks})))
    say('pool', meta[-1])
    for tk in tasks:
        for o in range(T.OUTER):
            f = OUT / f"{tk['task']}_p{tk['p']}_o{o}.csv"
            if not f.exists():
                say(tk['task'], tk['p'], o, round(T.run_job(tk, x, o, f), 1), 's')
json.dump(meta, open(HERE / 'sources.json', 'w'), indent=1)
say('done')
