#!/usr/bin/env python3
"""Acquire the pinned public R example and verify a lossless CSV conversion."""
from pathlib import Path
import argparse,hashlib
from urllib.request import urlopen
import numpy as np
import pandas as pd
import rdata

ROOT=Path(__file__).resolve().parent
URL='https://raw.githubusercontent.com/cran/BayesSUR/2.3-1/data/exampleGDSC.rda'
RDA_SHA='7b812ae9267322cc57997dc09821c228fb5610f59e1d88b33865ab8025b9163f'
def main():
    p=argparse.ArgumentParser();p.add_argument('--rda',type=Path);p.add_argument('--output-dir',type=Path,default=ROOT/'inputs'/'GDSC')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    rda=args.rda or args.output_dir/'exampleGDSC_2.3-1.rda'
    if not rda.exists():
        with urlopen(URL,timeout=60) as response: payload=response.read()
        if hashlib.sha256(payload).hexdigest()!=RDA_SHA:raise RuntimeError('Downloaded RDA hash mismatch')
        rda.write_bytes(payload)
    if hashlib.sha256(rda.read_bytes()).hexdigest()!=RDA_SHA:raise RuntimeError('RDA hash mismatch')
    arr=rdata.read_rda(rda)['exampleGDSC']['data']
    frame=pd.DataFrame(arr.values,columns=arr.coords['dim_1'].values.astype(str))
    if frame.shape!=(499,857) or not np.isfinite(frame.values).all():raise RuntimeError('Unexpected R data')
    out=args.output_dir/'exampleGDSC_2.3-1_readable.csv'
    if not out.exists():frame.to_csv(out,index=False,float_format='%.17g',lineterminator='\n')
    replay=pd.read_csv(out,float_precision='round_trip')
    if list(replay.columns)!=list(frame.columns) or not np.array_equal(replay.values,frame.values):
        raise RuntimeError('Existing or generated CSV is not a lossless conversion')
    print('Verified pinned RDA and lossless 499 x 857 CSV:',out)
    print('CSV SHA-256:',hashlib.sha256(out.read_bytes()).hexdigest())

if __name__=='__main__':main()
