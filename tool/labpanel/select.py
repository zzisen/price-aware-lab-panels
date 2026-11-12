"""Target-aware, cost-aware panel selection: beam search over orderable items scored by cross-validated Cox partial log-likelihood."""
import numpy as np
from .core import Cox, partial_loglik
from .prices import SCHEDULES, ORDERABLE, features_of


class Selector:
    """Z: pandas DataFrame with base columns and log-transformed analytes; t, e: follow-up time and event indicator.
    Scores are cached per analyte set, so searches under several price schedules or budgets reuse them."""

    def __init__(self, Z, t, e, base=('age', 'female'), lam=1.0, folds=5, seed=1, horizon=108.0, orderable=ORDERABLE):
        self.Z, self.t, self.e = Z, np.asarray(t, float), np.asarray(e, bool)
        self.base, self.lam, self.horizon, self.orderable = list(base), lam, horizon, orderable
        self.folds = np.array_split(np.random.default_rng(seed).permutation(len(self.t)), folds)
        self.cache = {}

    def score(self, items):
        key = tuple(sorted(features_of(items, self.orderable)))
        if key not in self.cache:
            X = self.Z[self.base + list(key)].to_numpy(); s = 0.0
            for te in self.folds:
                tr = np.setdiff1d(np.arange(len(self.t)), te)
                m = Cox(self.lam, self.horizon).fit(X[tr], self.t[tr], self.e[tr])
                s += partial_loglik(m.lp(X[te]), self.t[te], self.e[te])
            self.cache[key] = s
        return self.cache[key]

    def beam(self, budget, schedule='US', width=4, pool=None, start=()):
        price = SCHEDULES[schedule] if isinstance(schedule, str) else schedule
        cost = lambda items: price(features_of(items, self.orderable)) if items else 0.0  # noqa: E731
        pool = list(self.orderable) if pool is None else pool
        beam = [(tuple(sorted(start)), self.score(list(start)))]; best = beam[0]
        while True:
            cand = {}
            for items, _ in beam:
                have = set(features_of(items, self.orderable))
                for it in pool:
                    if it in items or set(self.orderable[it]) <= have:
                        continue
                    new = tuple(sorted(items + (it,)))
                    if new not in cand and cost(new) <= budget + 1e-9:
                        cand[new] = self.score(list(new))
            if not cand:
                break
            beam = sorted(cand.items(), key=lambda kv: -kv[1])[:width]
            if beam[0][1] > best[1] + 1e-9:
                best = beam[0]
            elif all(s <= best[1] + 1e-9 for _, s in beam):
                break
        return list(best[0]), cost(best[0]), best[1]

    def best_affordable(self, budget, schedule='US'):
        """Best analyte set among everything scored so far whose price is within budget (monotone in budget)."""
        price = SCHEDULES[schedule] if isinstance(schedule, str) else schedule
        s, k = max((v, k) for k, v in self.cache.items() if price(k) <= budget + 1e-9)
        return list(k), price(k), s

    def cheapest_matching(self, reference_items, schedule='US'):
        """Cheapest analyte set scored so far whose cross-validated score is at least that of reference_items."""
        price = SCHEDULES[schedule] if isinstance(schedule, str) else schedule
        thr = self.score(reference_items)
        (c, negs), k = min(((price(k), -v), k) for k, v in self.cache.items() if v >= thr)
        return list(k), c, -negs
