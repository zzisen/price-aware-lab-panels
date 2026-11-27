#!/usr/bin/env python3
"""Task 32c (PLAN.md): original guideline equations versus our panels for cardiovascular death, NHANES III.
PCE 2013 (Goff et al.; Table A coefficients) and PREVENT 2024 base model, 10-year total CVD (Khan et al.; coefficients from the CRAN package
preventr 0.12.2, checked against the paper's worked examples). Each is used as published (no refitting); discrimination only.
People: those of run_t21.py's cardiovascular analysis (complete clinical variables) who also have treatment and laboratory inputs.
Treatment: HAE5A = 1 (now taking prescribed medicine for high blood pressure) and HAE9D = 1 (prescribed medicine to lower cholesterol; used for the
PREVENT statin term); not asked when the preceding question was answered no, which is coded 0. eGFR: CKD-EPI 2021 (race-free).
Outputs: Harrell C with other deaths censored and Wolbers' competing-risk C at 108 months; paired bootstrap 1000 (seed 20261128). t32c_NH3.json"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(W / 'task21_final'))
import dm2lib as M  # noqa: E402
import t20lib as T  # noqa: E402
M.L.low_priority()


def wolbers(s, t, e, c):
    """Competing-risk concordance (same function as t32b_cvd_compete.py)."""
    num = den = 0.0
    for i in np.flatnonzero(e):
        ok = (t > t[i]) | (c & (t <= t[i])); ok[i] = False
        d = s[i] - s[ok]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += ok.sum()
    return num / den if den else np.nan


# ---------------- equations ----------------
PCE = {  # (lnage, lnage2, lntc, lnage*lntc, lnhdl, lnage*lnhdl, lntrsbp, lnage*lntrsbp, lnuntrsbp, lnage*lnuntrsbp, smoker, lnage*smoker, dm, S10, mean)
    'WF': (-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 2.019, 0, 1.957, 0, 7.574, -1.665, 0.661, 0.9665, -29.18),
    'BF': (17.114, 0, 0.940, 0, -18.920, 4.475, 29.291, -6.432, 27.820, -6.087, 0.691, 0, 0.874, 0.9533, 86.61),
    'WM': (12.344, 0, 11.853, -2.664, -7.990, 1.769, 1.797, 0, 1.764, 0, 7.837, -1.795, 0.658, 0.9144, 61.18),
    'BM': (2.469, 0, 0.302, 0, -0.307, 0, 1.916, 0, 1.809, 0, 0.549, 0, 0.645, 0.8954, 19.54)}


def pce(age, female, black, tc, hdl, sbp, bptx, smoker, dm):
    out = np.empty(len(age))
    for i in range(len(age)):
        c = PCE[('B' if black[i] else 'W') + ('F' if female[i] else 'M')]
        la, lt, lh, ls = np.log(age[i]), np.log(tc[i]), np.log(hdl[i]), np.log(sbp[i])
        s = (c[0] * la + c[1] * la ** 2 + c[2] * lt + c[3] * la * lt + c[4] * lh + c[5] * la * lh
             + (c[6] * ls + c[7] * la * ls if bptx[i] else c[8] * ls + c[9] * la * ls) + c[10] * smoker[i] + c[11] * la * smoker[i] + c[12] * dm[i])
        out[i] = 1 - c[13] ** np.exp(s - c[14])
    return out


PV = {  # PREVENT base, 10-year total CVD: female, male
    'F': dict(age=0.7939329, nonhdl=0.0305239, hdl=-0.1606857, sbp_lt=-0.2394003, sbp_ge=0.3600781, dm=0.8667604, smk=0.5360739, egfr_lt=0.6045917,
              egfr_ge=0.0433769, bptx=0.3151672, statin=-0.1477655, bptx_sbp=-0.0663612, statin_nonhdl=0.1197879, age_nonhdl=-0.0819715,
              age_hdl=0.0306769, age_sbp=-0.0946348, age_dm=-0.27057, age_smk=-0.078715, age_egfr=-0.1637806, const=-3.307728),
    'M': dict(age=0.7688528, nonhdl=0.0736174, hdl=-0.0954431, sbp_lt=-0.4347345, sbp_ge=0.3362658, dm=0.7692857, smk=0.4386871, egfr_lt=0.5378979,
              egfr_ge=0.0164827, bptx=0.288879, statin=-0.1337349, bptx_sbp=-0.0475924, statin_nonhdl=0.150273, age_nonhdl=-0.0517874,
              age_hdl=0.0191169, age_sbp=-0.1049477, age_dm=-0.2251948, age_smk=-0.0895067, age_egfr=-0.1543702, const=-3.031168)}


def egfr_ckdepi2021(scr, age, female):
    k = np.where(female, 0.7, 0.9); a = np.where(female, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female, 1.012, 1.0)


def prevent(age, female, tc, hdl, sbp, bptx, statin, smoker, dm, egfr):
    ac = (age - 55) / 10; nh = (tc - hdl) * 0.02586 - 3.5; hc = (hdl * 0.02586 - 1.3) / 0.3
    sl, sg = (np.minimum(sbp, 110) - 110) / 20, (np.maximum(sbp, 110) - 130) / 20
    el, eg = (np.minimum(egfr, 60) - 60) / -15, (np.maximum(egfr, 60) - 90) / -15
    out = np.empty(len(age))
    for sx, msk in (('F', female), ('M', ~female)):
        c = PV[sx]; i = msk
        lo = (c['age'] * ac[i] + c['nonhdl'] * nh[i] + c['hdl'] * hc[i] + c['sbp_lt'] * sl[i] + c['sbp_ge'] * sg[i] + c['dm'] * dm[i] + c['smk'] * smoker[i]
              + c['egfr_lt'] * el[i] + c['egfr_ge'] * eg[i] + c['bptx'] * bptx[i] + c['statin'] * statin[i] + c['bptx_sbp'] * bptx[i] * sg[i]
              + c['statin_nonhdl'] * statin[i] * nh[i] + c['age_nonhdl'] * ac[i] * nh[i] + c['age_hdl'] * ac[i] * hc[i] + c['age_sbp'] * ac[i] * sg[i]
              + c['age_dm'] * ac[i] * dm[i] + c['age_smk'] * ac[i] * smoker[i] + c['age_egfr'] * ac[i] * el[i] + c['const'])
        out[i] = 1 / (1 + np.exp(-lo))
    return out


def selftest():
    # PCE Table A example (age 55, TC 213, HDL 50, untreated SBP 120, non-smoker, no diabetes): 2.1%, 3.0%, 5.3-5.4%, 6.1%
    a = np.full(4, 55.0); r = pce(a, [1, 1, 0, 0], [0, 1, 0, 1], np.full(4, 213.0), np.full(4, 50.0), np.full(4, 120.0), [0] * 4, [0] * 4, [0] * 4)
    assert np.allclose(r, [0.02052, 0.03026, 0.05384, 0.06073], atol=2e-4), r
    # PREVENT main-text example: woman 50, TC 240, HDL 55, SBP 160 treated, no statin, no diabetes, eGFR 90: 5.4%; smoker 9.3%
    f = np.array([True, True]); r = prevent(np.array([50.0, 50.0]), f, np.array([240.0, 240.0]), np.array([55.0, 55.0]), np.array([160.0, 160.0]),
                                            np.array([1, 1]), np.array([0, 0]), np.array([0, 1]), np.array([0, 0]), np.array([90.0, 90.0]))
    assert np.allclose(r, [0.05427, 0.09258], atol=2e-4), r
    # preventr test, female: woman 50, TC 200, HDL 45, SBP 160 treated, diabetes, eGFR 90: 0.147
    r = prevent(np.array([50.0]), np.array([True]), np.array([200.0]), np.array([45.0]), np.array([160.0]), np.array([1]), np.array([0]),
                np.array([0]), np.array([1]), np.array([90.0]))
    assert abs(r[0] - 0.14684) < 2e-4, r
    assert abs(egfr_ckdepi2021(np.array([1.0]), np.array([50.0]), np.array([True]))[0] - 68.6) < 0.1
    print('self-test passed')


if __name__ == '__main__':
    selftest()
    H = 108
    CVD = json.load(open(W / 'task21_final' / 'selected_cvd.json'))
    CB = M.BASE + M.CLIN
    xtr, _ = M.load(list('ABCDEF')); Ztr = M.transform(xtr)
    xte, _ = M.load_nh3(); Z = M.transform(xte, ref=xtr)
    cltr = M.clinical(xtr); oktr = cltr.notna().all(1).to_numpy(); Ztrc = pd.concat([Ztr, cltr], axis=1).iloc[np.flatnonzero(oktr)]
    cl = M.clinical(xte)
    ad = M.read_fixed('adult', ['SEQN', 'HAE2', 'HAE4A', 'HAE5A', 'HAE6', 'HAE7', 'HAE8D', 'HAE9D'])
    seqn = pd.Index([int(i[2:]) for i in xte.index])
    g = lambda v: ad[v].reindex(seqn).to_numpy()  # noqa: E731
    bptx = np.where(g('HAE5A') == 1, 1.0, np.where((g('HAE5A') == 2) | (g('HAE2') == 2) | (g('HAE4A') == 2), 0.0, np.nan))
    statin = np.where(g('HAE9D') == 1, 1.0, np.where((g('HAE9D') == 2) | (g('HAE6') == 2) | (g('HAE7') == 2) | (g('HAE8D') == 2), 0.0, np.nan))
    print('race codes', xte.race.value_counts().to_dict(), 'tc median', float(xte.tc.median()), 'hdl median', float(xte.hdl.median()),
          'creatinine median', float(xte.creatinine.median()), 'bptx missing', int(np.isnan(bptx).sum()), 'statin missing', int(np.isnan(statin).sum()))
    ok = cl.notna().all(1).to_numpy() & ~np.isnan(bptx) & ~np.isnan(statin)
    xo, clo = xte.iloc[np.flatnonzero(ok)], cl.iloc[np.flatnonzero(ok)]
    Zc = pd.concat([Z, cl], axis=1).iloc[np.flatnonzero(ok)]
    tv, eall = T.outcome_at(xo, H); cvd = xo.ucod.isin(['001', '005']).to_numpy(); ev = eall & cvd; comp = eall & ~cvd
    age = xo.age.to_numpy(float); fem = (xo.sex == 2).to_numpy(); black = (xo.race == 4).to_numpy() if xo.race.dtype != object else (xo.race == 'Non-Hispanic Black').to_numpy()
    tc, hdl, scr = xo.tc.to_numpy(float), xo.hdl.to_numpy(float), xo.creatinine.to_numpy(float)
    sbp, smk, dm = clo.sbp.to_numpy(float), clo.smoker.to_numpy(float), clo.diabetes.to_numpy(float)
    bt, st = bptx[ok], statin[ok]
    S = {'PCE (published)': pce(age, fem, black, tc, hdl, sbp, bt, smk, dm),
         'PREVENT base (published)': prevent(age, fem, tc, hdl, sbp, bt, st, smk, dm, egfr_ckdepi2021(scr, age, fem))}
    cv_tr_t, cv_tr_e = M.outcome(xtr, 'cvd'); cv_tr_t, cv_tr_e = cv_tr_t[oktr], cv_tr_e[oktr]
    CV = {k: v['features'] for k, v in CVD['guideline'].items() if k in ('PREVENT full', 'routine', 'none')}
    CV.update({'CVD aware@PREVENT-full price': CVD['aware_at_prevent_full']['features'], 'CVD aware@routine': CVD['aware_at_routine']['features']})
    for k, f in CV.items():
        S[k] = M.Cox(1.0).fit(Ztrc[CB + list(f)].to_numpy(), cv_tr_t, cv_tr_e).lp(Zc[CB + list(f)].to_numpy())
    pairs = [('CVD aware@PREVENT-full price', 'PREVENT base (published)'), ('CVD aware@PREVENT-full price', 'PCE (published)'),
             ('PREVENT full', 'PREVENT base (published)'), ('none', 'PREVENT base (published)'), ('PREVENT base (published)', 'PCE (published)')]
    out = dict(cohort='NH3', n=len(tv), cvd_deaths=int(ev.sum()), other_deaths=int(comp.sum()), bptx_share=float(bt.mean()), lipid_med_share=float(st.mean()),
               C_censored={k: M.cindex(v, tv, ev) for k, v in S.items()}, C_competing={k: wolbers(v, tv, ev, comp) for k, v in S.items()})
    rng = np.random.default_rng(20261128); n = len(tv); bc = {f'{a} - {b}': [] for a, b in pairs}; bw = {f'{a} - {b}': [] for a, b in pairs}
    for _ in range(1000):
        j = rng.integers(0, n, n)
        Cc = {k: M.cindex(S[k][j], tv[j], ev[j]) for k in S}; Cw = {k: wolbers(S[k][j], tv[j], ev[j], comp[j]) for k in S}
        for a, b in pairs:
            bc[f'{a} - {b}'].append(Cc[a] - Cc[b]); bw[f'{a} - {b}'].append(Cw[a] - Cw[b])
    ci = lambda v: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))]  # noqa: E731
    out['contrasts_censored'] = {f'{a} - {b}': dict(point=out['C_censored'][a] - out['C_censored'][b], ci=ci(bc[f'{a} - {b}'])) for a, b in pairs}
    out['contrasts_competing'] = {f'{a} - {b}': dict(point=out['C_competing'][a] - out['C_competing'][b], ci=ci(bw[f'{a} - {b}'])) for a, b in pairs}
    json.dump(out, open(HERE / 't32c_NH3.json', 'w'), indent=1, default=float)
    print('n', n, 'CVD', int(ev.sum()), 'other', int(comp.sum()), 'BP meds', round(float(bt.mean()), 3), 'lipid meds', round(float(st.mean()), 3))
    for k in S:
        print(f"{k:34s} C(censored) {out['C_censored'][k]:.4f}  C(competing) {out['C_competing'][k]:.4f}")
    for k in out['contrasts_censored']:
        a, b = out['contrasts_censored'][k], out['contrasts_competing'][k]
        print(f"{k:62s} cens {a['point']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]  comp {b['point']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
