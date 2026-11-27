"""Task 5: exploratory comparison of rule variants on development and learning data (development only)."""
import sys
import pandas as pd
import dev_eval as D

V = dict(D.RULES)
for c in (0.5, 1.0, 1.5, 2.0):
    V[f'SE{c}+C0'] = (lambda c: lambda rec: D.se_rule(D.units(rec), D.candidates(rec, True), c))(c)
    V[f'avgSE{c}+C0'] = (lambda c: lambda rec: D.se_rule(D.averaged_units(rec), D.candidates(rec, True), c))(c)
V['avgSE1'] = lambda rec: D.se_rule(D.averaged_units(rec), D.candidates(rec, False), 1.0)
V['argmin+C0'] = lambda rec: D.argmin_rule(D.units(rec), D.candidates(rec, True))
reg, mx = D.evaluate(V)
with pd.option_context('display.width', 300, 'display.max_columns', 40):
    print(mx.round(3).T)
    print(mx.mean().round(4).sort_values())
