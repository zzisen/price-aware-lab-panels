#!/usr/bin/env python3
"""Reconstruct locally the release-5 molecular matrix and seven outcomes.

The source has precomputed expression and fitted IC50 values. It is not raw
assay-level processing. No complete-case filtering is used. Repeated fitted
records for one cell/target are collapsed by their median within that cell.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent; local=ROOT/'source_local_only'
PREV=ROOT if (ROOT/'target_resolution.py').exists() else ROOT.parent/'P2_V3.1_EXACT_2026-10-28'
TARGETS=['Methotrexate','RDEA119','PD-0325901','CI-1040','AZD6244','Nilotinib','Axitinib']
IDS=['1008','1014','1060','1015','1062','1013','1021']
t0=time.perf_counter()
expected={
 'gdsc_en_input_w5.csv':'89e97de69e776668f94156fdd4163eb83be4161c63775d8dcebab6579126ff0e',
 'gdsc_cell_lines_w5.csv':'eb8445b649cecd35d4af3a6e1dd87c902ee986694c05988da405535dd4116604',
 'gdsc_drug_sensitivity_fitted_data_w5.csv':'c8d4edd4f149d0817cf978bb17ef089106d3fa2d9618e00772f74a0b416a34ea'}
for name,sha in expected.items():
    h=hashlib.sha256()
    with (local/name).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    assert h.hexdigest()==sha, f'Source hash differs: {name}'
src=pd.read_csv(local/'gdsc_en_input_w5.csv',index_col=0)
assert src.shape==(13831,624) and src.index.is_unique and src.columns.is_unique
expr=src.index[:13321]; cnv=src.index[13321:13747];mut=src.index[13747:13818];tissues=src.index[13818:]
genes=set([s[:-3] for s in cnv]+[s[:-4] for s in mut]);expr=[s for s in expr if s in genes]
assert len(expr)==343 and len(cnv)==426 and len(mut)==71 and len(tissues)==13
X=np.concatenate([np.log(src.loc[expr].to_numpy().T),src.loc[cnv].to_numpy().T,src.loc[mut].to_numpy().T],axis=1)
assert X.shape==(624,840) and np.isfinite(X).all()
feature_names=expr+[s[:-3]+'.CNV' for s in cnv]+[s[:-4]+'.MUT' for s in mut]
tissue=src.loc[tissues].to_numpy().T;assert (tissue.sum(1)==1).all()
metadata=pd.read_csv(local/'gdsc_cell_lines_w5.csv',dtype=str)
metadata=metadata[['CELL_LINE_NAME','COSMIC_ID']].drop_duplicates()
assert not (metadata.groupby('CELL_LINE_NAME').COSMIC_ID.nunique()>1).any()
meta=metadata.drop_duplicates('CELL_LINE_NAME').set_index('CELL_LINE_NAME')
assert all(name in meta.index for name in src.columns)
ids=meta.loc[src.columns,'COSMIC_ID'].tolist();assert len(set(ids))==624
cosmic_to_cell=dict(zip(ids,src.columns));parts=[];nlines=0
for chunk in pd.read_csv(local/'gdsc_drug_sensitivity_fitted_data_w5.csv',
                         usecols=['cell_line_name','cosmic_id','drug_id','ic_50_est'],
                         dtype={'cell_line_name':str,'cosmic_id':str,'drug_id':str},chunksize=30000):
    nlines+=len(chunk);part=chunk[chunk.cosmic_id.isin(cosmic_to_cell)&chunk.drug_id.isin(IDS)].copy()
    if len(part):parts.append(part)
df=pd.concat(parts,ignore_index=True);df=df[df.cell_line_name==df.cosmic_id.map(cosmic_to_cell)].copy()
df=df[np.isfinite(df.ic_50_est)].copy()
counts=df.groupby(['cell_line_name','drug_id']).size()
outcome=df.groupby(['cell_line_name','drug_id']).ic_50_est.median().unstack()
outcome=outcome.reindex(index=src.columns,columns=IDS);Y=outcome.to_numpy(float)
assert Y.shape==(624,7)
example=pd.read_csv(PREV/'inputs/GDSC/exampleGDSC_2.3-1_readable.csv')
map_path=PREV/'inputs/GDSC/exampleGDSC_2.3-1_row_to_source_cell_line.csv'
if map_path.exists():
    mapfile=pd.read_csv(map_path)
    indices=pd.Index(src.columns).get_indexer(mapfile.gdsc_cell_line_name)
else:
    # The distributed example omits line identities. Recover them locally by
    # unique matching on ten source expression features; never publish them.
    anchor_cols=pd.Index(feature_names).get_indexer(example.columns[20:30])
    assert (anchor_cols>=0).all()
    source_anchors=X[:,anchor_cols]
    indices=[]
    for row in example.iloc[:,20:30].to_numpy(float):
        candidates=np.flatnonzero(np.max(np.abs(source_anchors-row),axis=1)<1e-10)
        assert len(candidates)==1, 'Ambiguous source identity; inspect locally'
        indices.append(int(candidates[0]))
    indices=np.asarray(indices,dtype=int)
assert (indices>=0).all()
matched_diff=np.abs(Y[indices]-example[TARGETS].to_numpy(float))
report={'source_cell_lines':624,'source_features':840,'target_names':TARGETS,
        'n_cell_lines_each_target':dict(zip(TARGETS,np.isfinite(Y).sum(0).astype(int).tolist())),
        'lines_with_at_least_one_observed_target':int((np.isfinite(Y).sum(1)>0).sum()),
        'lines_with_all_seven_targets':int((np.isfinite(Y).sum(1)==7).sum()),
        'n_observed_cells_targets':int(np.isfinite(Y).sum()),
        'replicate_cell_target_pairs':int((counts>1).sum()),
        'max_replicate_records_per_pair':int(counts.max()),
        'example_499_responses_matching_median':int((matched_diff<1e-10).sum()),
        'example_499_responses_total':int(matched_diff.size),
        'example_499_max_median_difference':float(np.nanmax(matched_diff)),
        'tissue_cell_counts':dict(zip(tissues.tolist(),tissue.sum(0).astype(int).tolist())),
        'candidate_mutations_original':71,
        'upstream_status':'Source already supplies processed expression and fitted IC50; this is not raw assay-level data.',
        'outcome_rule':'median of each cell-drug repeated fitted records; not across-cell imputation',
        'source_rows_scanned':nlines,'runtime_seconds':time.perf_counter()-t0}
np.savez_compressed(local/'release5_local_624x840.npz',X=X,Y=Y,tissue=tissue,
                    cells=src.columns.to_numpy(str),features=np.array(feature_names),
                    targets=np.array(TARGETS),tissue_names=tissues.to_numpy(str))
out=ROOT/'results'/'upstream_rebuild';out.mkdir(parents=True,exist_ok=True)
(out/'source_inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
