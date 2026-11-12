"""Quick self-test on simulated data: python -m pytest tests (or python tests/test_basic.py)."""
import numpy as np
import pandas as pd
import labpanel as lp


def simulate(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cols = list(dict.fromkeys(f for v in lp.ORDERABLE.values() for f in v))
    Z = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols); Z['age'] = rng.normal(size=n); Z['female'] = rng.integers(0, 2, n).astype(float)
    T = rng.exponential(1 / (0.003 * np.exp(0.8 * Z.age + Z.albumin + 0.7 * Z.wbc))); return Z, np.minimum(T, 108), T <= 108


def test_prices():
    assert lp.SCHEDULES['US'](lp.features_of(['CMP', 'CBC'])) == 27.67
    assert lp.SCHEDULES['US'](lp.features_of(['CMP', 'CBC', 'LIPID', 'hba1c'])) == 50.77
    assert lp.SCHEDULES['AU'](lp.features_of(['CMP', 'crp'])) == lp.SCHEDULES['AU'](lp.features_of(['CMP']))


def test_selection_finds_signal():
    Z, t, e = simulate()
    S = lp.Selector(Z, t, e)
    items, c, _ = S.beam(30.0, 'US', pool=['albumin', 'CBC', 'glucose', 'hdl', 'acr'])
    assert 'albumin' in items and 'CBC' in items


def test_sequential_extremes():
    Z, t, e = simulate()
    P = lp.SequentialPolicy(['age', 'female'], [['albumin'], lp.ORDERABLE['CBC']], tau=0.1).fit(Z, t, e)
    st, _, c = P.run(Z, 1.01, [0, 13, 30]); assert (st == 0).all()
    st, _, c = P.run(Z, 0.0, [0, 13, 30]); assert (st == 2).all()


if __name__ == '__main__':
    test_prices(); test_selection_finds_signal(); test_sequential_extremes(); print('all tests passed')
