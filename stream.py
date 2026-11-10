"""Steps 06-12,14: MMD warning + ADWIN error trigger + replay + distill.

Protocol: for each chronological batch, predict -> score with verified labels
-> add to observed history -> optionally adapt. Labels arrive at end of batch.
Do not use future test labels to make current predictions.
"""
import argparse
from collections import deque
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import psutil
from river import drift
from model import DEVICE, fit, probabilities, score, clone, load_model
from replay import BalancedReservoir



def rbf_mmd2(A, B, gamma):
    """Biased RBF MMD squared. Nonnegative up to numerical error."""
    def kernel(x, z):
        aa = np.sum(x*x, axis=1, keepdims=True)
        zz = np.sum(z*z, axis=1, keepdims=True).T
        dist2 = np.maximum(aa + zz - 2 * (x @ z.T), 0.)
        return np.exp(-gamma * dist2)
    return float(max(0., kernel(A, A).mean() + kernel(B, B).mean()
                     - 2*kernel(A, B).mean()))


def calibrate_mmd(X_initial, window=96):
    """Calibrate using disjoint reference samples exclusively from training period."""
    if len(X_initial) < 4 * window:
        raise ValueError('Not enough training rows for MMD window.')
    rng = np.random.default_rng(11)
    refs = rng.choice(len(X_initial), size=window, replace=False)
    ref = X_initial[refs].astype('float32')
    # RBF gamma based on median pairwise train-reference distance.
    d2 = np.sum((ref[:48, None, :] - ref[None, :48, :])**2, axis=2)
    nonzero = d2[d2 > 0]
    gamma = 1. / float(np.median(nonzero)) if len(nonzero) else 1.
    values = []
    for _ in range(15):
        ids = rng.choice(len(X_initial), size=window, replace=False)
        values.append(rbf_mmd2(ref, X_initial[ids], gamma))
    threshold = float(np.quantile(values, 0.95) * 1.15 + 1e-6)
    return ref, gamma, threshold



def adaptive_update(current, recent, buffer, retention_X, retention_y):
    """Candidate train on prior observed batches, validate on last observed batch.

    Latest batch is held out entirely from candidate training and replay until
    decision has been made. Historical retention is also never used for updates.
    """
    if len(recent) < 2:
        return current, False, {'reason': 'need_2_observed_batches'}
    prior = list(recent)[:-1]
    val_X, val_y = recent[-1]
    if len(np.unique(val_y)) < 2:
        return current, False, {'reason': 'latest_validation_batch_has_one_class'}
    new_X = np.concatenate([x for x, _ in prior])
    new_y = np.concatenate([y for _, y in prior])
    if len(new_y) < 100:
        return current, False, {'reason': 'not_enough_recent_verified_samples'}
    replay_X, replay_y = buffer.sample(128)
    if replay_X is not None:
        new_X = np.concatenate([new_X, replay_X])
        new_y = np.concatenate([new_y, replay_y])
    teacher = clone(current).eval()
    candidate = clone(current)
    fit(candidate, new_X, new_y, epochs=4, lr=0.0003, teacher=teacher,
        distill_weight=0.20)
    incumbent_val = score(val_y, probabilities(current, val_X))
    candidate_val = score(val_y, probabilities(candidate, val_X))
    old_hist = score(retention_y, probabilities(current, retention_X))
    new_hist = score(retention_y, probabilities(candidate, retention_X))
    # Predefined policy: recent macro-F1 must improve; historical retention may
    # decline by no more than .03, and benign FPR by no more than .03.
    improved = candidate_val['macro_f1'] > incumbent_val['macro_f1'] + 0.005
    retained = new_hist['macro_f1'] >= old_hist['macro_f1'] - 0.03
    old_fpr = old_hist['attack_fpr'] or 0.
    new_fpr = new_hist['attack_fpr'] or 0.
    fpr_ok = new_fpr <= old_fpr + 0.03
    accept = bool(improved and retained and fpr_ok)
    info = {'reason': 'accepted' if accept else 'failed_validation',
            'old_recent_macro_f1': incumbent_val['macro_f1'],
            'candidate_recent_macro_f1': candidate_val['macro_f1'],
            'old_historical_macro_f1': old_hist['macro_f1'],
            'candidate_historical_macro_f1': new_hist['macro_f1'],
            'accepted': accept}
    if accept:
        # Save rollback before replacing the serving model.
        torch.save(current.cpu().state_dict(), 'models/rollback_mlp.pt')
        current.to(DEVICE)
        torch.save(candidate.cpu().state_dict(), 'models/updated_mlp.pt')
        candidate.to(DEVICE)
        return candidate, True, info
    return current, False, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-prefix', default='hybrid')
    args = parser.parse_args()
    if args.batch_size < 64:
        parser.error('--batch-size should be at least 64 for MMD and error monitoring.')
    arr = np.load('data/processed/scaled_data.npz')
    X, y = arr['X'], arr['y'].astype(int)
    fit_end, train_end, dev_end = [int(arr[k]) for k in ('fit_end','train_end','dev_end')]
    model = load_model()
    # Full initial fit contains both labels. No use of later data to calibrate.
    X_train, y_train = X[:fit_end], y[:fit_end]
    ref, gamma, threshold = calibrate_mmd(X_train)
    print(f'MMD calibrated: gamma={gamma:.5f}, warning threshold={threshold:.5f}')
    adwin = drift.ADWIN(delta=0.002)
    error_hist = deque(maxlen=512)
    recent = deque(maxlen=4)
    buffer = BalancedReservoir(capacity_per_class=250)
    buffer.add(X_train, y_train)
    retention_X, retention_y = X[fit_end:train_end], y[fit_end:train_end]
    rows = []
    process = psutil.Process()
    full_y, full_probs = [], []
    last_mmd_warning_batch = -999
    last_adwin_increase_batch = -999
    mmd_warnings, adwin_alarms, adaptations = 0, 0, 0
    results = Path('results')
    results.mkdir(exist_ok=True)
    for batch_idx, start in enumerate(range(dev_end, len(y), args.batch_size)):
        stop = min(start + args.batch_size, len(y))
        xb, yb = X[start:stop], y[start:stop]
        t0 = time.perf_counter()
        prob = probabilities(model, xb)
        prediction_time = time.perf_counter()-t0
        metrics = score(yb, prob)
        full_probs.extend(prob.tolist())
        full_y.extend(yb.tolist())
        # MMD is input distribution change ONLY, not evidence of harmful drift.
        window = min(96, len(xb))
        mm = rbf_mmd2(ref, xb[:window], gamma) if window >= 32 else float('nan')
        warning = bool(mm > threshold)
        if warning:
            mmd_warnings += 1
            last_mmd_warning_batch = batch_idx
        # ADWIN gets verified 0/1 prediction errors only AFTER scoring batch.
        error_increase_detected = False
        for error in (prob >= 0.5).astype(int) != yb:
            error_hist.append(int(error))
            adwin.update(int(error))
            if adwin.drift_detected:
                adwin_alarms += 1
                if len(error_hist) >= 256:
                    series = np.asarray(error_hist)
                    before = series[:len(series)//2].mean()
                    after = series[len(series)//2:].mean()
                    if after > before + 0.03:
                        error_increase_detected = True
        if error_increase_detected:
            last_adwin_increase_batch = batch_idx
        recent.append((xb.copy(), yb.copy()))
        updated = False
        update_details = {'reason': 'no_hybrid_trigger'}
        update_seconds = 0.
        triggered = bool(
            (error_increase_detected and batch_idx - last_mmd_warning_batch <= 4)
            or (warning and batch_idx - last_adwin_increase_batch <= 4)
        )
        if triggered:
            t1 = time.perf_counter()
            model, updated, update_details = adaptive_update(
                model, recent, buffer, retention_X, retention_y)
            update_seconds = time.perf_counter() - t1
            if updated:
                adaptations += 1
                # Reset error monitoring; old error process is no longer comparable.
                adwin = drift.ADWIN(delta=0.002)
                error_hist.clear()
                recent.clear()
                last_adwin_increase_batch = -999
                last_mmd_warning_batch = -999
        # The latest batch joins replay only AFTER it is scored and validated.
        buffer.add(xb, yb)
        row = {'batch': batch_idx, 'stream_start': start, 'stream_stop': stop,
               'mmd2': mm, 'mmd_warning': warning,
               'adwin_error_increase': error_increase_detected,
               'adaptation_attempted': triggered,
               'adaptation_accepted': updated,
               'update_reason': update_details['reason'],
               'prediction_seconds': prediction_time,
               'prediction_rows_per_second': float(len(xb) / max(prediction_time, 1e-9)),
               'update_seconds': update_seconds,
               'ram_mb': process.memory_info().rss / 1e6,
               'replay_rows': sum(len(v) for v in buffer.items.values()),
               **metrics}
        rows.append(row)
        print(f'Batch {batch_idx:03} | F1={metrics["f1"]:.3f} | MMD={mm:.4f}'
              f' | warning={int(warning)} | ADWIN increase={int(error_increase_detected)}'
              f' | update={int(updated)}')
    summary = score(np.array(full_y), np.array(full_probs))
    summary.update({'mmd_warnings': mmd_warnings, 'adwin_alarms': adwin_alarms,
                    'accepted_updates': adaptations, 'batches': len(rows),
                    'mean_prediction_seconds': float(np.mean([r['prediction_seconds'] for r in rows])),
                    'total_update_seconds': float(np.sum([r['update_seconds'] for r in rows])),
                    'peak_observed_ram_mb': float(max(r['ram_mb'] for r in rows)),
                    'batch_size': args.batch_size,
                    'labels': 'verified at end of each batch; no artificial label delay',
                    'drift_false_alarm_rate': None,
                    'drift_detection_delay': None,
                    'note': 'True concept-drift locations unknown for CICIDS2017.'})
    pd.DataFrame(rows).to_csv(results / f'{args.output_prefix}_batches.csv', index=False)
    (results / f'{args.output_prefix}_summary.json').write_text(json.dumps(summary, indent=2))
    print('Final PREQUENTIAL summary:', json.dumps(summary, indent=2))
    print('Saved:', results / f'{args.output_prefix}_summary.json')


if __name__ == '__main__':
    main()
