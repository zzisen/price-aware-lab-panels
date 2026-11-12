"""labpanel core: ridge Cox model, Harrell's C, net benefit and calibration for a fixed prediction horizon."""
import numpy as np


class Cox:
    """Ridge-penalised Cox proportional hazards model (Breslow ties) on standardised inputs, fitted by damped Newton steps.
    risk(X) returns the predicted probability of the event by `horizon` (same time unit as t)."""

    def __init__(self, lam=1.0, horizon=108.0):
        self.lam, self.horizon = lam, horizon

    def fit(self, X, t, e, w=None):
        X = np.asarray(X, float); t = np.asarray(t, float); e = np.asarray(e, bool)
        self.m = X.mean(0); self.s = X.std(0); self.s[self.s == 0] = 1
        Z = (X - self.m) / self.s
        w = np.ones(len(t)) if w is None else np.asarray(w, float) / np.mean(w)
        o = np.argsort(-t, kind='stable'); Z, t, e, w = Z[o], t[o], e[o], w[o]
        n, p = Z.shape; b = np.zeros(p)
        last = np.searchsorted(-t, -t, side='right') - 1
        for _ in range(30):
            eta = Z @ b; r = w * np.exp(eta - eta.max())
            S0 = np.cumsum(r)[last]; S1 = np.cumsum(r[:, None] * Z, 0)[last]
            S2 = np.cumsum(r[:, None, None] * Z[:, :, None] * Z[:, None, :], 0)[last]
            mu = S1 / S0[:, None]; we = w * e
            g = (we[:, None] * (Z - mu)).sum(0) - self.lam * b
            H = (we[:, None, None] * (S2 / S0[:, None, None] - mu[:, :, None] * mu[:, None, :])).sum(0) + self.lam * np.eye(p)
            step = np.linalg.solve(H, g); step *= min(1.0, 1.0 / max(np.abs(step).max(), 1e-12))
            b = b + step
            if np.abs(step).max() < 1e-6:
                break
        self.b = b
        eta = Z @ b; r = w * np.exp(eta); S0 = np.cumsum(r)[last]
        self.H0 = float(((w * e) / S0)[t <= self.horizon].sum())
        return self

    def lp(self, X):
        return ((np.asarray(X, float) - self.m) / self.s) @ self.b

    def risk(self, X):
        return 1 - np.exp(-self.H0 * np.exp(self.lp(X)))

    def to_dict(self, columns):
        return dict(columns=list(columns), mean=self.m.tolist(), sd=self.s.tolist(), coef=self.b.tolist(), H0=self.H0, horizon=self.horizon, lam=self.lam)


def partial_loglik(lp, t, e):
    lp, t, e = np.asarray(lp, float), np.asarray(t, float), np.asarray(e, bool)
    o = np.argsort(-t, kind='stable'); lp, t, e = lp[o], t[o], e[o]
    S0 = np.cumsum(np.exp(lp - lp.max()))[np.searchsorted(-t, -t, side='right') - 1]
    return float((e * (lp - lp.max() - np.log(S0))).sum())


def cindex(score, t, e):
    """Harrell's C: among pairs where the earlier time is an event, the share where the earlier one has the higher score (ties count 1/2)."""
    s, t, e = np.asarray(score, float), np.asarray(t, float), np.asarray(e, bool)
    num = den = 0.0
    idx = np.flatnonzero(e)
    for ch in np.array_split(idx, max(1, len(idx) // 200)):
        later = t[None, :] > t[ch, None]; d = s[ch, None] - s[None, :]
        num += float((later * ((d > 0) + 0.5 * (d == 0))).sum()); den += float(later.sum())
    return num / den


def net_benefit(risk, t, e, tau, horizon=108.0):
    known = np.asarray(e, bool) | (np.asarray(t) >= horizon); hi = np.asarray(risk) >= tau; e = np.asarray(e, bool)
    k = known.sum()
    return float((hi & e & known).sum() / k - (hi & ~e & known).sum() / k * tau / (1 - tau))
