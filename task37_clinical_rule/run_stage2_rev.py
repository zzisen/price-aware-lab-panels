#!/usr/bin/env python3
"""Second runner for stage 2: same jobs as run_stage2.py, smallest first, to use cores freed after stage 1. Existing outputs are skipped;
results are deterministic per job, so a job run twice writes the same row."""
import sys
import time
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import run_stage2 as R  # noqa: E402
if __name__ == '__main__':
    R.OUT.mkdir(exist_ok=True); J = R.jobs()[::-1]
    log = open(HERE / 'stage2_log_rev.txt', 'a', encoding='utf-8'); log.write(f'start {time.ctime()} jobs {len(J)}\n'); log.flush(); t0 = time.time()
    with Pool(int(sys.argv[1])) as pl:
        for i, (name, sec) in enumerate(pl.imap(R.run_job, J)):
            log.write(f'{i + 1}/{len(J)} {name} {sec}s {round((time.time() - t0) / 60, 1)}min\n'); log.flush()
    log.write(f'done {time.ctime()}\n')
