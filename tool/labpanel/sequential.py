"""Sequential ordering: add one block of tests at a time and stop as soon as the remaining tests are unlikely to change the decision."""
import numpy as np
from .core import Cox


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class FlipModel:
    """P(decision at the final step differs from the current decision | current risk); logistic in d, d^2, |d| with d = logit(risk) - logit(tau)."""

    def __init__(self, tau):
        self.tau = tau

    def _X(self, r):
        d = _logit(r) - _logit(self.tau)
        return np.column_stack([np.ones(len(r)), d, d ** 2, np.abs(d)])

    def fit(self, r_now, r_final):
        X = self._X(r_now); y = ((r_now >= self.tau) != (r_final >= self.tau)).astype(float); b = np.zeros(4)
        for _ in range(50):
            p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p) + 1e-9
            step = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(4), X.T @ (y - p)); b += step
            if np.abs(step).max() < 1e-8:
                break
        self.b = b
        return self

    def prob(self, r):
        return 1 / (1 + np.exp(-self._X(np.atleast_1d(r)) @ self.b))

    def interval(self, eps, grid=np.linspace(0.0005, 0.9995, 19991)):
        """Risk ranges in which the rule says 'add the next block' (probability >= eps)."""
        go = self.prob(grid) >= eps; out, start, prev = [], None, None
        for r, g in zip(grid, go):
            if g and start is None:
                start = r
            if not g and start is not None:
                out.append([float(start), float(prev)]); start = None
            prev = r
        if start is not None:
            out.append([float(start), float(prev)])
        return out


class SequentialPolicy:
    """Fixed order of feature blocks after a free base. fit() uses cross-fitted training risks for the stop rules."""

    def __init__(self, base, blocks, tau=0.10, lam=1.0, horizon=108.0):
        self.base, self.blocks, self.tau, self.lam, self.horizon = list(base), [list(b) for b in blocks], tau, lam, horizon
        self.cols = [self.base + [f for b in self.blocks[:k] for f in b] for k in range(len(self.blocks) + 1)]

    def fit(self, Z, t, e, folds=5, seed=1):
        t, e = np.asarray(t, float), np.asarray(e, bool)
        idx = np.array_split(np.random.default_rng(seed).permutation(len(t)), folds)
        rcv = np.zeros((len(self.cols), len(t)))
        for te in idx:
            tr = np.setdiff1d(np.arange(len(t)), te)
            for s, c in enumerate(self.cols):
                X = Z[c].to_numpy(); rcv[s, te] = Cox(self.lam, self.horizon).fit(X[tr], t[tr], e[tr]).risk(X[te])
        self.models = [Cox(self.lam, self.horizon).fit(Z[c].to_numpy(), t, e) for c in self.cols]
        self.stop = [FlipModel(self.tau).fit(rcv[s], rcv[-1]) for s in range(len(self.cols) - 1)]
        return self

    def run(self, Z, eps, step_costs):
        """step_costs[k] = total cost after k blocks. Returns (step reached, risk used, cost) per person."""
        R = np.array([m.risk(Z[c].to_numpy()) for m, c in zip(self.models, self.cols)])
        n = R.shape[1]; step = np.zeros(n, int); active = np.ones(n, bool)
        for s in range(len(self.cols) - 1):
            go = active & (self.stop[s].prob(R[s]) >= eps); step[go] = s + 1; active = go
        return step, R[step, np.arange(n)], np.asarray(step_costs)[step]


def apply_chart(rule, risks, value=30000):
    """Apply the frozen value-of-information chart (the rule used in the manuscript and the calculator; task22_adaptive/rule_t22.json).
    rule: the loaded JSON; risks: dict step name -> array of predicted risks at that step ('clinical only', 'CMP', 'CMP+CBC', 'CMP+CBC+acr').
    A person goes on to the next block while the current risk lies inside one of the chart's intervals for that step.
    Returns (step reached, risk used, cost paid) per person."""
    names = list(rule['costs']); chart = rule['chart'][str(int(value))]
    n = len(np.asarray(risks[names[0]])); step = np.zeros(n, int); active = np.ones(n, bool)
    for s, nm in enumerate(names[:-1]):
        r = np.asarray(risks[nm], float)
        go = np.zeros(n, bool)
        for lo, hi in chart.get(nm, []):
            go |= (r >= lo) & (r <= hi)
        go &= active; step[go] = s + 1; active = go
    R = np.array([np.asarray(risks[nm], float) for nm in names])
    return step, R[step, np.arange(n)], np.array([rule['costs'][nm] for nm in names])[step]
