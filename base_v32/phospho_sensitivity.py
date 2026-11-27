#!/usr/bin/env python3
"""Reproduce P2's frozen phospho analysis and run post-hoc sensitivities.

Inputs are frozen, de-identified pre-2026-10-10 summaries and recorded losses.
The alternative regressors are exploratory: all choices were made after the
original results were known. Each outer fold repeats inner action selection,
training-only scaling, fitting, and evaluation. Deletion re-fits every fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS = ['pS6', 'p4EBP1', 'pSTAT5', 'pSYK', 'pPLCγ2', 'pAKT', 'pERK1/2', 'pIKAROS']
TARGETS = ['BCR-Crosslink', 'BEZ-235', 'Dasatinib', 'IL-7', 'Pervanadate', 'TSLP', 'Tofacitinib']
STANFORD_TARGETS = ['Dasatinib', 'IL-7']
POLICIES = ['C0', 'SHARED', 'EXACT']
CONFIGS = [('ols', 0.0, False), ('bounded_ols', 0.0, True),
           ('ridge10', 10.0, False), ('bounded_ridge10', 10.0, True),
           ('ridge40', 40.0, False), ('bounded_ridge40', 40.0, True)]
CONFIGS = [(label, alpha, clip, 'standardized') for label, alpha, clip in CONFIGS] + [
    (label + '_native', alpha, clip, 'native') for label, alpha, clip in CONFIGS]


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f'No output rows for {path}')
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_matrices(summary_path, target_names):
    rows = read_rows(summary_path)
    patients = list(dict.fromkeys(r['patient_id'] for r in rows))
    values = {}
    for r in rows:
        k = (r['patient_id'], r['condition'], r['channel'])
        if k in values:
            raise ValueError(f'Duplicate measurement: {k}')
        value = float(r['positive_fraction'])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid positive fraction: {k}, {value}')
        values[k] = value
    expected = {(p, condition, c) for p in patients
                for condition in ['Basal'] + target_names for c in CHANNELS}
    if set(values) != expected:
        raise ValueError(f'Missing/extra measurements in {summary_path.name}: '
                         f'{len(expected - set(values))}/{len(set(values) - expected)}')
    x = np.array([[values[(p, 'Basal', c)] for c in CHANNELS] for p in patients])
    y = np.array([[[values[(p, t, c)] for c in CHANNELS] for t in target_names] for p in patients])
    return patients, x, y


def predictions(x_train, y_train, x_test, alpha, clip):
    """Predictions for every candidate action, in training-standardized coordinates.

    x: training patients x channels. y: training patients x targets x channels.
    Ridge penalizes the slope only, using sum-squared-error scale; intercept is
    unpenalized. Zero sample SD is replaced by 1 as in the frozen analysis.
    """
    xm = x_train.mean(axis=0); xs = x_train.std(axis=0, ddof=1)
    ym = y_train.mean(axis=0); ys = y_train.std(axis=0, ddof=1)
    xs = np.where(xs == 0, 1.0, xs); ys = np.where(ys == 0, 1.0, ys)
    xz = (x_train - xm) / xs; yz = (y_train - ym) / ys
    slope = np.einsum('ic,ito->cto', xz, yz) / ((xz * xz).sum(axis=0)[:, None, None] + alpha)
    pred = ((x_test - xm) / xs)[:, None, None] * slope
    if clip:
        pred = (np.clip(ym[None, :, :] + ys[None, :, :] * pred, 0, 1) - ym[None, :, :]) / ys[None, :, :]
    return pred, ym, ys


def choose_actions(x, y, training_indices, alpha, clip, loss_scale):
    inner_losses = []
    for val in training_indices:
        tr = training_indices[training_indices != val]
        assert val not in tr and len(tr) == len(training_indices) - 1
        pred, ym, ys = predictions(x[tr], y[tr], x[val], alpha, clip)
        observed_z = (y[val] - ym) / ys
        error = pred - observed_z[None, :, :]
        if loss_scale == 'native': error = error * ys[None, :, :]
        inner_losses.append(np.mean(error**2, axis=2))
    by_action_target = np.mean(inner_losses, axis=0)
    return int(np.argmin(by_action_target.mean(axis=1))), np.argmin(by_action_target, axis=0)


def outer_loo(patients, x, y, alpha, clip, loss_scale):
    n, nt = len(patients), y.shape[1]
    losses = np.empty((n, nt, len(POLICIES))); shared_actions = []; exact_actions = []
    all_idx = np.arange(n)
    for val in all_idx:
        train = all_idx[all_idx != val]
        shared, exact = choose_actions(x, y, train, alpha, clip, loss_scale)
        pred, ym, ys = predictions(x[train], y[train], x[val], alpha, clip)
        yz = (y[val] - ym) / ys
        scale = ys if loss_scale == 'native' else 1.0
        losses[val, :, 0] = np.mean((yz*scale)**2, axis=1)
        losses[val, :, 1] = np.mean(((pred[shared] - yz)*scale)**2, axis=1)
        losses[val, :, 2] = np.mean(((pred[exact, np.arange(nt)] - yz)*scale)**2, axis=1)
        shared_actions.append(CHANNELS[shared])
        exact_actions.append([CHANNELS[k] for k in exact])
    return losses, shared_actions, exact_actions


def transport(x, y, xt, yt, alpha, clip, loss_scale):
    training = np.arange(len(x))
    shared, exact = choose_actions(x, y, training, alpha, clip, loss_scale)
    tidx = [TARGETS.index(t) for t in STANFORD_TARGETS]
    losses = np.empty((len(xt), len(tidx), len(POLICIES)))
    for j in range(len(xt)):
        pred, ym, ys = predictions(x, y, xt[j], alpha, clip)
        yz = (yt[j] - ym[tidx]) / ys[tidx]
        scale = ys[tidx] if loss_scale == 'native' else 1.0
        losses[j, :, 0] = np.mean((yz*scale)**2, axis=1)
        losses[j, :, 1] = np.mean(((pred[shared, tidx] - yz)*scale)**2, axis=1)
        losses[j, :, 2] = np.mean(((pred[exact[tidx], tidx] - yz)*scale)**2, axis=1)
    return losses, CHANNELS[shared], [CHANNELS[k] for k in exact[tidx]]


def recorded_loss_tensor(path, patients, target_names):
    records = read_rows(path)
    lookup = {(r['patient_id'], r['target']): r for r in records}
    assert len(lookup) == len(patients) * len(target_names)
    return np.array([[[float(lookup[(p, t)]['loss_' + policy]) for policy in POLICIES]
                      for t in target_names] for p in patients])


def summary(loss, patient_ids, name, config, draws=50000):
    means = loss.mean(axis=(0, 1))
    d = (loss[:, :, 1] - loss[:, :, 2]).mean(axis=1)
    rng = np.random.default_rng(20261027)
    bs = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {'evaluation': name, 'config': config, 'n_patients': len(patient_ids),
            'C0': means[0], 'Shared': means[1], 'Exact': means[2],
            'Shared_minus_Exact': means[1] - means[2],
            'bootstrap_2.5pct': lo, 'bootstrap_97.5pct': hi,
            'patient_contrast_median': np.median(d),
            'n_patients_Shared_lower': int(np.sum(d < 0)),
            'n_patients_Exact_lower': int(np.sum(d > 0)),
            'UPN21_patient_contrast': float(d[patient_ids.index('UPN21')]) if 'UPN21' in patient_ids else ''}


def append_losses(rows, losses, patients, targets, config, evaluation):
    for i, patient in enumerate(patients):
        for t, target in enumerate(targets):
            rows.append({'evaluation': evaluation, 'config': config, 'patient_id': patient,
                         'target': target, **{f'loss_{p}': losses[i, t, j] for j, p in enumerate(POLICIES)}})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input-dir', type=Path, default=Path(__file__).parent/'inputs')
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).parent/'results')
    args = ap.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    dev, x, y = load_matrices(args.input_dir/'B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv', TARGETS)
    stan, xt, yt = load_matrices(args.input_dir/'B2_STANFORD_PATIENT_CONDITION_CHANNEL_SUMMARY.csv', STANFORD_TARGETS)
    assert len(dev) == 44 and len(stan) == 9 and 'UPN21' in dev and not set(dev) & set(stan)
    expected_dev = recorded_loss_tensor(args.input_dir/'B1_OUTER_PATIENT_TARGET_LOSSES.csv', dev, TARGETS)
    expected_stan = recorded_loss_tensor(args.input_dir/'B2_STANFORD_PATIENT_TARGET_LOSSES.csv', stan, STANFORD_TARGETS)
    keep = np.array([p != 'UPN21' for p in dev]); dev43 = [p for p in dev if p != 'UPN21']
    keep42 = np.array([p not in {'UPN21', 'UPN68'} for p in dev])
    dev42 = [p for p in dev if p not in {'UPN21', 'UPN68'}]
    summaries = []; all_losses = []; action_rows = []; validation = {}
    for label, alpha, clip, loss_scale in CONFIGS:
        loss, shared, exact = outer_loo(dev, x, y, alpha, clip, loss_scale)
        if label == 'ols':
            validation['dev_original_max_abs_error'] = float(np.max(np.abs(loss - expected_dev)))
            if not np.allclose(loss, expected_dev, rtol=1e-9, atol=1e-9):
                raise AssertionError('Original 44-patient analysis did not reproduce')
        summaries.append(summary(loss, dev, 'dev_44_nested_cv', label))
        summaries.append(summary(loss[keep], dev43, 'dev_43_frozen_losses_only', label))
        append_losses(all_losses, loss, dev, TARGETS, label, 'dev_44_nested_cv')
        for i, p in enumerate(dev):
            for ti, target in enumerate(TARGETS):
                action_rows.append({'evaluation':'dev_44_nested_cv', 'config':label, 'patient_id':p,
                                    'target':target, 'selected_shared':shared[i], 'selected_exact':exact[i][ti]})
        loss43, shared43, exact43 = outer_loo(dev43, x[keep], y[keep], alpha, clip, loss_scale)
        summaries.append(summary(loss43, dev43, 'dev_43_full_refit', label))
        append_losses(all_losses, loss43, dev43, TARGETS, label, 'dev_43_full_refit')
        for i, p in enumerate(dev43):
            for ti, target in enumerate(TARGETS):
                action_rows.append({'evaluation':'dev_43_full_refit', 'config':label, 'patient_id':p,
                                    'target':target, 'selected_shared':shared43[i], 'selected_exact':exact43[i][ti]})
        # Explicitly post-hoc: both patients were selected after examining losses.
        loss42, shared42, exact42 = outer_loo(dev42, x[keep42], y[keep42], alpha, clip, loss_scale)
        summaries.append(summary(loss42, dev42, 'dev_42_both_removed_full_refit', label))
        append_losses(all_losses, loss42, dev42, TARGETS, label, 'dev_42_both_removed_full_refit')
        for i, p in enumerate(dev42):
            for ti, target in enumerate(TARGETS):
                action_rows.append({'evaluation':'dev_42_both_removed_full_refit', 'config':label, 'patient_id':p,
                                    'target':target, 'selected_shared':shared42[i], 'selected_exact':exact42[i][ti]})
        test_loss, shared_t, exact_t = transport(x, y, xt, yt, alpha, clip, loss_scale)
        if label == 'ols':
            validation['stanford_original_max_abs_error'] = float(np.max(np.abs(test_loss - expected_stan)))
            if not np.allclose(test_loss, expected_stan, rtol=1e-9, atol=1e-9):
                raise AssertionError('Original Stanford analysis did not reproduce')
        summaries.append(summary(test_loss, stan, 'stanford_9_held_out_transport', label))
        append_losses(all_losses, test_loss, stan, STANFORD_TARGETS, label, 'stanford_9_held_out_transport')
        for i, p in enumerate(stan):
            for ti, target in enumerate(STANFORD_TARGETS):
                action_rows.append({'evaluation':'stanford_9_held_out_transport', 'config':label,
                                    'patient_id':p, 'target':target, 'selected_shared':shared_t,
                                    'selected_exact':exact_t[ti]})
        print(label, 'completed', flush=True)
    write_rows(args.out_dir/'sensitivity_summary.csv', summaries)
    write_rows(args.out_dir/'held_out_losses.csv', all_losses)
    write_rows(args.out_dir/'action_selections.csv', action_rows)
    validation['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input_dir.glob('*.csv')}
    validation['method'] = 'post-hoc exploratory; standardized and native-fraction losses target different estimands; paired patient resampling of fixed held-out losses, 50000 draws; not independent validation'
    (args.out_dir/'reproduction_and_provenance.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
