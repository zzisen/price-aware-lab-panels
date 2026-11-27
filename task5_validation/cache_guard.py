#!/usr/bin/env python3
"""Task 5: guarded cache (extrapolation guard in selection, middle layer and outer scoring) for R4 development.
Same records and order as cache_generic.py. Run with .env312, same arguments as cache_generic.py."""
import pickle
import sys
import rulelib as L
from cache_generic import LEARNING


def main(args):
    if args[0] == 'task':
        assert args[1] in LEARNING
        task, x, y, fam = L.load_task(args[1]); tag = args[1]
        it = ((sub, held, groups, x, y, seed, fold) for ev, seed, fold, n, r, sub, held, groups in L.task_subsamples(task, x))
    else:
        import adaptive_rules as A
        fam = A.FAM; tag = f'{args[0]}_{args[1]}' if len(args) > 1 else args[0]
        it = ((sub, held, groups, x, y, seed, fold) for ev, seed, fold, n, r, sub, held, groups, x, y, _ in A.subsamples(args[0], args[1] if len(args) > 1 else None))
    out = [L.cache_subsample_g(x, y, groups, fam, sub, held, seed + fold + 1) for sub, held, groups, x, y, seed, fold in it]
    with open(L.HERE / 'cache' / f'{tag}_guard.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print(tag, 'guard records', len(out))


if __name__ == '__main__':
    main(sys.argv[1:])
