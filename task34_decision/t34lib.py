#!/usr/bin/env python3
"""Task 34 (PLAN_DRAFT.md): pools, tasks and one job = (source, p level, task, outer split).
Panels: A = FSCA (target-agnostic), B = somp on 18 historical targets from six other families (chosen among candidates that are not historical
targets, refitted for the current targets), C = somp on the current targets. Paths to k = 20 (greedy, so k = 5 and 10 are prefixes).
Prediction: guarded RidgeCV from the frozen R5 library (fix applied). Score: explained variance of held-out targets on the training-standardised
scale, 1 - SSE / sum(yt^2). Decision D1: 5-fold CV inside the training subsample, panels re-selected in every inner fold."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent; EXT = W / 'external_data'
sys.path.insert(0, str(W / 'task17_gates')); sys.path.insert(0, str(W / 'task16_route_b'))
import gatelib as GL  # noqa: E402
from rppa_families import FAMILY as RPPA_FAMILY  # noqa: E402
R, V = GL.r5_libs()
KMAX, KS, NS, OUTER, REPS, INNER = 20, (5, 10, 20), (50, 100, 200), 3, 4, 5
NOT_FAMILY = {'Other', 'other', 'unannotated'}


# ---------------- pools ----------------
def gtex_pool(tissue, folder, seed, p_levels=(71, 200, 1000)):
    h = pd.read_csv(EXT / 'hgnc' / 'hgnc_complete_set.txt', sep='\t', low_memory=False, usecols=['symbol', 'locus_group', 'gene_group', 'ensembl_gene_id'])
    h = h[(h.locus_group == 'protein-coding gene') & h.ensembl_gene_id.notna()].drop_duplicates('ensembl_gene_id').set_index('ensembl_gene_id')
    t = pd.read_csv(EXT / folder / f'gene_tpm_2017-06-05_v8_{tissue}.gct.gz', sep='\t', skiprows=2)
    t['ens'] = t.Name.str.split('.').str[0]
    t = t[t.ens.isin(h.index)].drop_duplicates('ens').set_index('ens').drop(columns=['id', 'Name', 'Description'])
    expressed = sorted(t.index[t.median(axis=1) >= 1])
    members = {}
    for e in expressed:
        if isinstance(h.gene_group.get(e), str):
            for g in h.gene_group[e].split('|'):
                members.setdefault(g, []).append(e)
    elig = sorted(g for g, m in members.items() if len(m) >= 10)
    rng = np.random.default_rng(seed)
    used, fam, n = [], {}, 0
    for g in rng.permutation(elig):
        free = sorted(set(members[g]) - set(used))
        if len(free) < 4:
            continue
        for e in rng.choice(free, 4, replace=False):
            used.append(e); fam[e] = str(g)
        n += 1
        if n == 20:
            break
    rest = [e for e in expressed if e not in set(used)]
    extra = list(rng.permutation(rest)[:max(p_levels) - 71])                  # nested extension: p = 71 + first (p - 71) extra genes
    cols = used + extra
    x = np.log2(t.loc[cols] + 1).T
    x.columns = [h.symbol[e] for e in cols]
    famsym = {h.symbol[e]: f for e, f in fam.items()}
    return x, famsym, [h.symbol[e] for e in extra]


def rppa_pools(cancers, min_n=150):
    t = pd.read_csv(EXT / 'tcga_rppa_xena' / 'TCGA-RPPA-pancan-clean.xena.gz', sep='\t', index_col=0, na_values=['NA', 'NaN', '']).T
    p = pd.read_csv(EXT / 'tcga_rppa_xena' / 'Survival_SupplementalTable_S1_20171025_xena_sp', sep='\t', usecols=['sample', 'cancer type abbreviation']).set_index('sample')
    out = {}
    for c in cancers:
        ids = [s for s in t.index if s.endswith('-01') and s in p.index and p.loc[s, 'cancer type abbreviation'] == c]
        if len(ids) < min_n:
            continue
        x = t.loc[ids]; done = sorted(x.columns[x.notna().all()])
        var = x[done].var()
        top = sorted(done, key=lambda a: (-var[a], a))[:80]
        extra = [a for a in done if a not in top]
        out[c] = (x[top + extra], {a: RPPA_FAMILY.get(a) for a in top}, extra)
    return out


def make_tasks(source, x, fam, extra, code0, p_levels):
    base = [c for c in x.columns if c not in set(extra)]                     # the 80-measurement pool
    mem = {}
    for c in base:
        if fam.get(c) is not None and fam[c] not in NOT_FAMILY:
            mem.setdefault(fam[c], []).append(c)
    mem = {g: sorted(m) for g, m in mem.items() if len(m) >= 3}
    tasks = []
    for ti in range(6):
        rng = np.random.default_rng(code0 + ti)
        fams = list(rng.permutation(sorted(mem)))
        cur, hist = fams[:3], fams[3:9]
        if len(hist) < 4:                                                    # at least four historical families (RPPA has fewer than GTEx)
            continue
        tg = [str(c) for g in cur for c in rng.choice(mem[g], 3, replace=False)]
        hg = [str(c) for g in hist for c in rng.choice(mem[g], 3, replace=False)]
        cand71 = [c for c in base if c not in tg]
        for p in p_levels:
            if p - 71 > len(extra):
                continue
            cand = cand71 + list(extra[:p - 71])
            tasks.append(dict(source=source, task=f'{source}_T{ti + 1}', p=p, code=code0 + ti, targets=tg, hist_targets=hg, candidates=cand))
    return tasks


# ---------------- panels and scoring ----------------
def paths(xs, ys, yh, hist_idx):
    C = GL.somp(xs, ys, KMAX, 0)
    A = GL.fsca(xs, KMAX, 0)
    keep = np.setdiff1d(np.arange(xs.shape[1]), hist_idx)
    B = keep[GL.somp(xs[:, keep], yh, KMAX, 0)]
    return dict(A=A, B=B, C=C)


def ev(xs, ys, xt, yt, cols):
    pred = R.ridge_predict(xs, ys, xt, np.asarray(cols, int))
    return 1 - float(((pred - yt) ** 2).sum()) / float((yt ** 2).sum())


def run_job(task, frame, o, out_csv):
    t0 = time.time()
    cand, tg = task['candidates'], task['targets']
    X = frame[cand].to_numpy(float); Y = frame[tg].to_numpy(float); YH = frame[task['hist_targets']].to_numpy(float)
    hist_idx = np.array([cand.index(c) for c in task['hist_targets'] if c in cand], int)
    rng = np.random.default_rng([task['code'], o])
    perm = rng.permutation(len(X)); nte = int(round(0.2 * len(X))); te, trn = perm[:nte], perm[nte:]
    rows = []
    for n in NS:
        if n > len(trn):
            continue
        for r in range(REPS):
            sub = np.random.default_rng([task['code'], o, n, r]).choice(trn, n, replace=False)
            ym, ysd = Y[sub].mean(0), Y[sub].std(0, ddof=1); ysd = np.where(ysd == 0, 1, ysd)
            ys, yt = (Y[sub] - ym) / ysd, (Y[te] - ym) / ysd
            hm, hsd = YH[sub].mean(0), YH[sub].std(0, ddof=1); yh = (YH[sub] - hm) / np.where(hsd == 0, 1, hsd)
            P = paths(X[sub], ys, yh, hist_idx)
            outer = {(m, k): ev(X[sub], ys, X[te], yt, P[m][:k]) for m in 'ABC' for k in KS}
            folds = np.random.default_rng([task['code'], o, n, r, 7]).permutation(n) % INNER
            inner = {(m, k): 0.0 for m in 'ABC' for k in KS}
            for f in range(INNER):
                a, b = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
                xa, xb = X[sub][a], X[sub][b]
                m2, s2 = ys[a].mean(0), ys[a].std(0, ddof=1); s2 = np.where(s2 == 0, 1, s2)
                ya, yb = (ys[a] - m2) / s2, (ys[b] - m2) / s2
                h2 = (yh[a] - yh[a].mean(0)) / np.where(yh[a].std(0, ddof=1) == 0, 1, yh[a].std(0, ddof=1))
                Pi = paths(xa, ya, h2, hist_idx)
                for m in 'ABC':
                    for k in KS:
                        pred = R.ridge_predict(xa, ya, xb, np.asarray(Pi[m][:k], int))
                        inner[(m, k)] += float(((pred - yb) ** 2).sum())
            for k in KS:
                d1 = min('ABC', key=lambda m: (inner[(m, k)], 'ABC'.index(m)))
                rows.append(dict(source=task['source'], task=task['task'], p=task['p'], outer=o, n=n, rep=r, k=k,
                                 A=outer[('A', k)], B=outer[('B', k)], C=outer[('C', k)], D1=d1,
                                 inner_A=inner[('A', k)], inner_B=inner[('B', k)], inner_C=inner[('C', k)]))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return time.time() - t0
