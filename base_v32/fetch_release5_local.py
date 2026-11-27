#!/usr/bin/env python3
"""Download official GDSC release-5 source files for local pilot only."""
from urllib.request import Request,urlopen
from pathlib import Path
import hashlib,json,time
ROOT=Path(__file__).resolve().parent
out=ROOT/'source_local_only';out.mkdir(exist_ok=True)
base='https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-5.0/'
files=['gdsc_en_input_w5.csv','gdsc_tissue_output_w5.csv','gdsc_cell_lines_w5.csv','gdsc_drug_sensitivity_fitted_data_w5.csv']
EXPECTED={
 'gdsc_en_input_w5.csv':'89e97de69e776668f94156fdd4163eb83be4161c63775d8dcebab6579126ff0e',
 'gdsc_tissue_output_w5.csv':'f56e520aec4519939ebed49cc384c51924b3f0303dda231d89460792701495bc',
 'gdsc_cell_lines_w5.csv':'eb8445b649cecd35d4af3a6e1dd87c902ee986694c05988da405535dd4116604',
 'gdsc_drug_sensitivity_fitted_data_w5.csv':'c8d4edd4f149d0817cf978bb17ef089106d3fa2d9618e00772f74a0b416a34ea'}
report={}
for name in files:
    path=out/name;start=time.perf_counter();digest=hashlib.sha256();total=0
    if path.exists():
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):digest.update(chunk);total+=len(chunk)
        if digest.hexdigest()!=EXPECTED[name]:
            raise RuntimeError(f'Existing file hash differs; inspect {path} rather than overwriting it')
        print('REUSED verified',name,flush=True)
    else:
        with urlopen(Request(base+name,headers={'User-Agent':'P2-academic-methods-pilot/1.0'}),timeout=120) as response, path.open('xb') as f:
            while True:
                chunk=response.read(1<<20)
                if not chunk:break
                f.write(chunk);digest.update(chunk);total+=len(chunk)
                if total//(50<<20)!=(total-len(chunk))//(50<<20):print(name,round(total/1048576),'MiB',flush=True)
        if digest.hexdigest()!=EXPECTED[name]:raise RuntimeError(f'Download hash differs; inspect {path}')
    report[name]={'url':base+name,'bytes':total,'sha256':digest.hexdigest(),'seconds':time.perf_counter()-start}
    print('DONE',name,total,round(time.perf_counter()-start,1),'s',flush=True)
(out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
