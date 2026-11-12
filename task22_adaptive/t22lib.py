#!/usr/bin/env python3
"""Task 22 library: value-of-information (VOI) stopping rule for sequential laboratory ordering.
Blocks are ordered in a fixed sequence (CMP -> CBC -> urine ACR). Clinical information (age, sex, BMI, SBP, smoking, diabetes) is free.
At each state s (blocks done so far) the expected net-benefit gain of finishing all blocks is
    g_s(r) = P(decision at the final state differs | r) x E[|r_final - tau| / (1 - tau) | decision differs, r],
both parts fitted on 5-fold cross-fitted training risks. A person continues to the next block while V x g_s(r_s) >= cost(final) - cost(s)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'task21_final'))
import dm2lib as M  # noqa: E402

ORDER = ['CMP', 'CBC', 'acr']
TAU = 0.10
V_MAIN, V_SENS = 30000.0, (10000.0, 100000.0)
CB = M.BASE + M.CLIN


def logit(r):
    r = np.clip(r, 1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


def design(r, tau=TAU):
    d = logit(r) - logit(tau)
    return np.column_stack([np.ones(len(r)), d, d ** 2, np.abs(d)])


class Gain:
    def fit(self, r0, r1, tau=TAU):
        self.tau = tau
        self.flip = M.FlipModel(tau).fit(r0, r1)
        fl = (r0 >= tau) != (r1 >= tau); m = np.abs(r1 - tau) / (1 - tau)
        X = design(r0[fl], tau)
        self.bm = np.linalg.lstsq(X.T @ X + 1e-6 * np.eye(4), X.T @ m[fl], rcond=None)[0]
        self.n_flip = int(fl.sum())
        return self

    def g(self, r):
        return self.flip.prob(r) * np.clip(design(r, self.tau) @ self.bm, 0, None)

    def coef(self):
        return dict(flip=[float(v) for v in self.flip.b], magnitude=[float(v) for v in self.bm], n_flip=self.n_flip)


class VOIRule:
    def __init__(self, order=ORDER, tau=TAU):
        self.order, self.tau = list(order), tau
        self.states = [tuple(self.order[:k]) for k in range(len(self.order) + 1)]   # (), (CMP,), (CMP, CBC), (CMP, CBC, acr)
        self.final = self.states[-1]

    def cols(self, s):
        return CB + M.features_of(list(s))

    def cost(self, s):
        return M.cost(list(s)) if s else 0.0

    def fit(self, Z, t, e, folds):
        self.models = {s: M.Cox(1.0).fit(Z[self.cols(s)].to_numpy(), t, e) for s in self.states}
        rcv = {s: np.zeros(len(t)) for s in self.states}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(t)), te)
            for s in self.states:
                X = Z[self.cols(s)].to_numpy(); rcv[s][te] = M.Cox(1.0).fit(X[tr], t[tr], e[tr]).risk(X[te])
        self.gain = {s: Gain().fit(rcv[s], rcv[self.final], self.tau) for s in self.states[:-1]}
        return self

    def risks(self, Z):
        return {s: self.models[s].risk(Z[self.cols(s)].to_numpy()) for s in self.states}

    def go(self, s, r, V):
        return V * self.gain[s].g(r) >= self.cost(self.final) - self.cost(s)

    def run(self, R, V):
        n = len(R[self.final]); k = np.zeros(n, int); active = np.ones(n, bool)
        for j, s in enumerate(self.states[:-1]):
            idx = np.flatnonzero(active & (k == j))
            g = self.go(s, R[s][idx], V)
            k[idx[g]] = j + 1; active[idx[~g]] = False
        risk = np.array([R[self.states[k[i]]][i] for i in range(n)])
        cst = np.array([self.cost(self.states[k[i]]) for i in range(n)])
        return k, risk, cst

    def chart(self, V, grid=np.linspace(0.0005, 0.9995, 19991)):
        """Risk intervals in which the rule says 'order the next block', for each state."""
        out = {}
        for s in self.states[:-1]:
            g = self.go(s, grid, V); iv, start, prev = [], None, None
            for r, x in zip(grid, g):
                if x and start is None:
                    start = r
                if not x and start is not None:
                    iv.append([float(start), float(prev)]); start = None
                prev = r
            if start is not None:
                iv.append([float(start), float(prev)])
            out['+'.join(s) or 'clinical only'] = iv
        return out

    def coef(self):
        return {'+'.join(s) or 'clinical only': self.gain[s].coef() for s in self.states[:-1]}
