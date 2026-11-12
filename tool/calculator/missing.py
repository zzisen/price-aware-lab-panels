"""Risk with missing inputs, as computed by the web calculator (reference implementation).

Each model is a ridge Cox model on transformed inputs z (age, female, ln(analyte + offset), clinical items). When some inputs are missing,
they are replaced by their conditional mean given the inputs that are present, using the mean vector and covariance matrix of z in the
training data (NHANES 1999-2010). Because the linear predictor is linear in z, its conditional variance is w_m' S_m|o w_m, where w are the
coefficients per unit of z of the missing inputs and S_m|o is their conditional covariance. The page reports the point risk and an 80% range
(linear predictor +/- 1.2816 conditional SD). With no missing inputs the range collapses to the point risk.
"""
import numpy as np

Q80 = 1.2815515655446004


def predict(model, joint, z):
    """model: dict(columns, mean, sd, coef, H0); joint: dict(names, mean, cov); z: dict column -> transformed value or None.
    Returns (risk, low, high) or None if age or sex is missing."""
    cols = model['columns']
    if z.get('age') is None or z.get('female') is None:
        return None
    pos = {n: i for i, n in enumerate(joint['names'])}
    mu, S = np.asarray(joint['mean'], float), np.asarray(joint['cov'], float)
    w = np.asarray(model['coef'], float) / np.asarray(model['sd'], float)
    zv = np.array([np.nan if z.get(c) is None else float(z[c]) for c in cols])
    mis = [i for i, c in enumerate(cols) if z.get(c) is None]
    obs = [i for i, c in enumerate(cols) if z.get(c) is not None]
    var = 0.0
    if mis:
        jo, jm = [pos[cols[i]] for i in obs], [pos[cols[i]] for i in mis]
        K = np.linalg.solve(S[np.ix_(jo, jo)], S[np.ix_(jo, jm)]).T          # S_mo S_oo^-1
        zv[mis] = mu[jm] + K @ (zv[obs] - mu[jo])
        C = S[np.ix_(jm, jm)] - K @ S[np.ix_(jo, jm)]
        var = float(w[mis] @ C @ w[mis])
    s = float(w @ (zv - np.asarray(model['mean'], float)))
    sd = float(np.sqrt(max(var, 0.0)))
    f = lambda v: float(1 - np.exp(-model['H0'] * np.exp(v)))
    return f(s), f(s - Q80 * sd), f(s + Q80 * sd)


def predict_pattern(model, joint, Z, missing):
    """Vectorised version for many people who share one missing pattern. Z: array (n, len(columns)); missing: list of column names.
    Returns arrays (risk, low, high)."""
    cols = model['columns']; pos = {n: i for i, n in enumerate(joint['names'])}
    mu, S = np.asarray(joint['mean'], float), np.asarray(joint['cov'], float)
    w = np.asarray(model['coef'], float) / np.asarray(model['sd'], float)
    Z = np.array(Z, float)
    mis = [cols.index(c) for c in missing]; obs = [i for i in range(len(cols)) if i not in mis]
    var = 0.0
    if mis:
        jo, jm = [pos[cols[i]] for i in obs], [pos[cols[i]] for i in mis]
        K = np.linalg.solve(S[np.ix_(jo, jo)], S[np.ix_(jo, jm)]).T
        Z[:, mis] = mu[jm] + (Z[:, obs] - mu[jo]) @ K.T
        C = S[np.ix_(jm, jm)] - K @ S[np.ix_(jo, jm)]
        var = float(w[mis] @ C @ w[mis])
    s = (Z - np.asarray(model['mean'], float)) @ w
    sd = np.sqrt(max(var, 0.0))
    f = lambda v: 1 - np.exp(-model['H0'] * np.exp(v))
    return f(s), f(s - Q80 * sd), f(s + Q80 * sd)
