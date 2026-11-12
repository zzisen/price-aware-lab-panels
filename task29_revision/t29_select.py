#!/usr/bin/env python3
"""Task 29 C and D: selection on the TRAINING layer only (NHANES 1999-2010, A-F). Usage: t29_select.py <us|uk|cn>
us : complete US menu; routine reference repriced; matched and target-aware panels re-selected; CoAI over all legal groupings; lasso repriced.
uk : Understanding Society menu, complete US menu; routine reference as ordered ($50.77) and as measured; NHS Health Check reference.
cn : NBPHS (China, 65+ annual check) reference on the CHNS and CLHLS menus, Chinese prices.
Same score and search as Tasks 19/20/25/27 (beam width 4, lambda 1, 5-fold CV Cox log-likelihood, seed 1). Output: sel_<mode>.json"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t29lib as Q  # noqa: E402
M, T, L = Q.M, Q.T, Q.L
import coai_select as CO  # noqa: E402
M.L.low_priority()
MODE = sys.argv[1]
x, counts = M.load(list('ABCDEF')); t, e = M.outcome(x); Z = M.transform(x)
folds = np.array_split(np.random.default_rng(1).permutation(len(x)), 5)
cache = Q.load_cache(); n0 = len(cache)
sc = lambda f: cache.setdefault(tuple(sorted(f)), M.gain_ll(Z, t, e, sorted(f), 1.0, folds))  # noqa: E731
known = e | (t >= M.HORIZON); y = (e & (t <= M.HORIZON))[known].astype(int)
out = dict(mode=MODE, cache_start=n0)


def search(price, budgets, pool, sched):
    for B in sorted(set(round(b, 2) for b in budgets)):
        t0 = time.time(); it, c, v = M.beam(Z, t, e, B, folds, sched, pool=pool, cache=cache)
        print(sched, B, it, c, round(v, 2), round(time.time() - t0), 's', len(cache), flush=True)
        Q.save_cache(cache)


def pick(price, ref_cv, budget, menu):
    ok = [k for k in cache if set(k) <= set(menu)]
    m_ = sorted(((price(k), -cache[k]), k) for k in ok if cache[k] >= ref_cv - 1e-9)[0][1]
    a_ = max(((cache[k], -price(k)), k) for k in ok if price(k) <= budget + 1e-9)[1]
    return (dict(features=list(m_), price=price(m_), cv=cache[m_]), dict(features=list(a_), price=price(a_), cv=cache[a_]))


if MODE == 'us':
    menu = list(M.FEATURES)
    # (i) old menu, merged cache: does the larger cache alone change the published panels?
    rout = M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C']); rcv = sc(rout)
    mo, ao = pick(M.cost_us, rcv, M.cost_us(rout), menu)
    out['old_menu_merged_cache'] = dict(matched=mo, aware=ao)
    print('old menu, merged cache', mo, ao, flush=True)
    # (ii) complete menu
    Q.install_full(); pool = list(M.ITEMS)
    std = {k: Q.cost_full(M.features_of(v)) for k, v in M.STANDARD.items()}
    for v in M.STANDARD.values():
        sc(M.features_of(v))
    sc([]); sc(menu)
    grid = list(std.values()) + list(np.round(np.quantile(list(std.values()), [0.1, 0.3, 0.5, 0.7, 0.9]), 2))
    search(Q.cost_full, grid, pool, 'USFULL')
    rp = std['CMP+CBC+LIPID+A1C']
    mf, af = pick(Q.cost_full, rcv, rp, menu)
    for d in (mf, af):
        d['order'] = Q.order_full(d['features'])
    out.update(routine=dict(features=rout, price_old=M.cost_us(rout), price_full=rp, cv=rcv, order=Q.order_full(rout)), standard_full=std, matched=mf, aware=af)
    print('FULL routine', rp, 'matched', mf, 'aware', af, flush=True)
    # CoAI on the complete menu; budgets = our new panels' prices
    Zk = Z[known]
    out['coai_full'] = Q.coai_full(Zk, y, menu, [mf['price'], rp], CO)
    for p in out['coai_full']['panels']:
        p['order'] = Q.order_full(p['features']); p['cv'] = sc(p['features'])
        print('CoAI full', p['budget'], p['order'], p['billed_cost'], flush=True)
    # price-weighted and unweighted lasso candidates repriced
    LC = pd.read_csv(Q.W / 'task21_final' / 'lasso_candidates.csv', keep_default_na=False)
    LC['full'] = [Q.cost_full(str(f).split()) if f else 0.0 for f in LC.features]
    lz = {}
    for kind in ['price-weighted', 'unweighted']:
        s = LC[LC.kind == kind]
        dm = s[s.cv >= rcv].sort_values(['full', 'cv'], ascending=[True, False]); dw = s[s.full <= rp + 1e-9].sort_values('cv', ascending=False)
        lz[kind] = dict(matched=dict(features=str(dm.features.iloc[0]).split(), price=float(dm.full.iloc[0])) if len(dm) else None,
                        within=dict(features=str(dw.features.iloc[0]).split(), price=float(dw.full.iloc[0])))
    out['lasso_full'] = lz; print('lasso', lz, flush=True)
    out['cmpcbc_full'] = Q.cost_full(M.features_of(['CMP', 'CBC']))

elif MODE == 'uk':
    import ukhls_menu as U
    U.install(); Q.install_full(U.MENU)
    pool = list(U.ITEMS_UK) + [k for k in Q.ITEMS_FULL if k in M.ITEMS and set(M.ITEMS[k]) <= set(U.MENU)]
    menu = U.MENU; rout = U.ROUTINE_UK; rcv = sc(rout); sc([]); sc(menu)
    hc = Q.NHSHC; hcv = sc(hc)
    p_ord, p_meas, p_hc = M.cost_us(M.features_of(M.STANDARD['CMP+CBC+LIPID+A1C'])), Q.cost_full(rout), Q.cost_full(hc)
    search(Q.cost_full, [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, p_ord, p_meas, p_hc], pool, 'USFULL')
    m1, a1 = pick(Q.cost_full, rcv, p_ord, menu); _, a2 = pick(Q.cost_full, rcv, p_meas, menu); m3, a3 = pick(Q.cost_full, hcv, p_hc, menu)
    for d in (m1, a1, a2, m3, a3):
        d['order'] = Q.order_full(d['features'])
    out.update(routine=dict(features=rout, price_as_ordered=p_ord, price_as_measured=p_meas, price_old=M.cost_us(rout), cv=rcv),
               matched=m1, aware_ordered=a1, aware_measured=a2, nhshc=dict(features=hc, price=p_hc, cv=hcv, order=Q.order_full(hc)), matched_hc=m3, aware_hc=a3)
    out['coai_full'] = Q.coai_full(Z[known], y, menu, [m1['price'], p_meas, p_ord], CO)
    for p in out['coai_full']['panels']:
        p['order'] = Q.order_full(p['features']); p['cv'] = sc(p['features'])
    print(json.dumps({k: out[k] for k in ['routine', 'matched', 'aware_ordered', 'aware_measured', 'nhshc', 'matched_hc', 'aware_hc']}, indent=1), flush=True)
    for p in out['coai_full']['panels']:
        print('CoAI full', p['budget'], p['order'], p['billed_cost'])

elif MODE == 'cn':
    import clhls_lib as C
    T.install_cn(); C.install()
    full_price = Q.nbphs_price_full()
    for name, menu, pool in [('CHNS', T.SHARED, list(T.CN_ITEMS)), ('CLHLS', C.MENU, list(C.ITEMS_CL))]:
        cbc = [a for a in menu if a in M.CBC2]
        ref = [a for a in Q.NBPHS if a in menu and a not in M.CBC2] + cbc
        rcv = sc(ref); pr = T.cost_cn(ref)
        search(T.cost_cn, [15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, pr], pool, 'CN')
        m_, a_ = pick(T.cost_cn, rcv, pr, menu)
        out[name] = dict(nbphs=dict(features=ref, price_as_measured=pr, price_as_ordered=full_price, cv=rcv), matched=m_, aware=a_)
        print(name, json.dumps(out[name]), flush=True)

out['cache_end'] = len(cache); Q.save_cache(cache)
json.dump(out, open(Q.HERE / f'sel_{MODE}.json', 'w'), indent=1, default=str)
print('done', MODE)
