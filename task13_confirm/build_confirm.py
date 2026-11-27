#!/usr/bin/env python3
"""Task 13 (plan_r5_confirm FROZEN 20261101T2045, section 3): build the confirmation tasks.
BeatAML main tasks (6), BeatAML large-target tasks for S1 (3), NHANES 2015-2016 (1; LBDTIB dropped, DEVIATIONS D1).
Prints task definitions and sample sizes only. Writes tasks.json (field "set": main / s1 / nhanes) and matrices/.
Run with .env312."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE.parent
EXT = W / 'external_data'
NH_TARGETS = {'LBXGH': 'metabolic', 'LBDHDD': 'metabolic', 'LBXFER': 'iron_inflammation', 'LBXHSCRP': 'iron_inflammation',
              'LBXVIDMS': 'nutrition', 'LBXBPB': 'exposure', 'LBXBCD': 'exposure', 'LBXTHG': 'exposure'}   # LBDTIB: D1
NH_TARGET_FILE = {'LBXGH': 'GHB_I', 'LBDHDD': 'HDL_I', 'LBXFER': 'FERTIN_I', 'LBXHSCRP': 'HSCRP_I',
                  'LBXVIDMS': 'VID_I', 'LBXBPB': 'PBCD_I', 'LBXBCD': 'PBCD_I', 'LBXTHG': 'PBCD_I'}


def save(name, x, y):
    frame = pd.concat([x.add_prefix('cand_'), y], axis=1)
    frame.index.name = 'unit'
    frame.reset_index().to_csv(HERE / 'matrices' / f'{name}.csv', index=False)
    return len(frame)


def beataml():
    d = EXT / 'beataml2'
    f = pd.read_csv(d / 'beataml_probit_curve_fits_v4_dbgap.txt', sep='\t')
    f = f[(f.type == 'single-agent') & (f.paper_inclusion == True) & np.isfinite(f.auc)]  # noqa: E712
    w = f.pivot_table(index='dbgap_subject_id', columns='inhibitor', values='auc', aggfunc='mean')
    cov = w.notna().mean()
    order = sorted(cov.index, key=lambda c: (-cov[c], c))
    top = order[:80]
    w = w[top].dropna()
    dg = pd.read_excel(d / 'beataml_drug_families.xlsx', sheet_name='drug_gene')
    dg = dg[dg.inhibitor.isin(top)].dropna(subset=['family']).reset_index(drop=True)
    prim = dg.sort_values(['inhibitor', 'db_min_kd'], kind='stable').drop_duplicates('inhibitor').set_index('inhibitor').family
    members = {}
    for i in top:
        if i in prim.index:
            members.setdefault(prim[i], []).append(i)
    elig = sorted(g for g, m in members.items() if len(m) >= 3)
    print('BeatAML: patients complete on top 80 =', len(w), '| boundary coverage', round(cov[top[-1]], 3),
          'next', round(cov[order[80]], 3), '| families with >=3 members:', len(elig))
    main, s1 = [], []
    for t in range(6):
        rng = np.random.default_rng(20261104 + t)
        fams = list(rng.choice(elig, 3, replace=False))
        targets = [str(i) for g in fams for i in rng.choice(sorted(members[g]), 3, replace=False)]
        cands = [i for i in top if i not in targets]
        name = f'BEAT_T{t + 1}'
        n = save(name, w[cands], w[targets])
        main.append({'task': name, 'targets': targets, 'families': [g for g in fams for _ in range(3)], 'code': 1300 + t})
        print(' ', name, 'units', n, 'candidates', len(cands), '| families', [g.split(':')[0][:28] for g in fams])
    for t in range(3):
        rng = np.random.default_rng(20261105 + t)
        perm = list(rng.permutation(sorted(top)))
        targets, cands = perm[:40], perm[40:]
        name = f'BEAT_S1_T{t + 1}'
        n = save(name, w[cands], w[targets])
        fam = [prim[i] if i in prim.index else 'unannotated' for i in targets]
        s1.append({'task': name, 'targets': targets, 'families': fam, 'code': 1320 + t})
        print(' ', name, 'units', n, 'targets', len(targets), 'candidates', len(cands), 'target families', len(set(fam)))
    dup = len({tuple(sorted(t['targets'])) for t in main}) < len(main)
    print('  duplicate main tasks:', dup)
    return main, s1


def nhanes():
    d = EXT / 'nhanes_2015_2016'
    read = lambda f: pd.read_sas(d / f'{f}.xpt', format='xport').set_index('SEQN')  # noqa: E731
    demo = read('DEMO_I')[['RIDAGEYR']]
    cbc = read('CBC_I').drop(columns=['LBXNRBC'], errors='ignore')
    bio = read('BIOPRO_I'); bio = bio[[c for c in bio if c.startswith('LBX')]]
    tg = pd.concat([read(NH_TARGET_FILE[k])[[k]] for k in NH_TARGETS], axis=1)
    f = demo[demo.RIDAGEYR >= 20].join(cbc, how='inner').join(bio, how='inner').join(tg, how='inner').drop(columns='RIDAGEYR').dropna()
    for c in f:
        pos = f[c][f[c] > 0]
        f[c] = np.log(f[c] + pos.min() / 2)
    f.index = f.index.astype(int).astype(str)
    cands = list(cbc.columns) + list(bio.columns)
    n = save('NHANES_I', f[cands], f[list(NH_TARGETS)])
    print('NHANES_I: adults complete', n, '| candidates', len(cands), '| targets', len(NH_TARGETS))
    return [{'task': 'NHANES_I', 'targets': list(NH_TARGETS), 'families': list(NH_TARGETS.values()), 'code': 1330}]


def main():
    (HERE / 'matrices').mkdir(exist_ok=True)
    m, s1 = beataml()
    nh = nhanes()
    tasks = [{**t, 'set': s} for s, ts in [('main', m), ('s1', s1), ('nhanes', nh)] for t in ts]
    (HERE / 'tasks.json').write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
