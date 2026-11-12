"""Exploratory CICIDS2018 hybrid: train-calibrated MMD + ADWIN error-rise
+ balanced replay + teacher distillation + forward-only candidate validation.

Run in ~/adaptive_ids_project/experiments/cicids2018:
    python hybrid_balanced_kd2018.py

Protocol: predict ALL flows in a chronological batch before revealing labels;
then update ADWIN / candidate validation, and possibly train a new candidate
using verified labels from batches already seen. Candidate is checked on
strictly later batches before replacing the active model.

This is an exploratory experiment on ALREADY INSPECTED Thursday data, not an
independent generalization claim or a production-ready IDS.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from river import drift

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, clone, fit, load_model, probabilities, score
from replay import BalancedReservoir
from stream import calibrate_mmd, rbf_mmd2

# Freeze these exploratory settings for this run; don't retune on Thursday/Friday.
BATCH_SIZE = 256
MMD_WINDOW = 96
MMD_CALIBRATION_QUANTILE = 0.99
MMD_RECENT_BATCHES = 4
ADWIN_DELTA = 0.002
MIN_ERROR_INCREASE = 0.03
MIN_MISSED_ATTACKS = 8
COOLDOWN_BATCHES = 3
RECENT_BATCHES = 4
REPLAY_SAMPLES = 256
BALANCED_PER_CLASS = 384
RECENT_SAMPLING_WEIGHT = 3.0
CANDIDATE_EPOCHS = 6
CANDIDATE_LR = 0.0003
DISTILLATION_WEIGHT = 0.20
MIN_VALIDATION_ATTACKS = 8
MIN_VALIDATION_BENIGN = 64
MAX_VALIDATION_BATCHES = 12
MIN_RECALL_GAIN = 0.05
MAX_VALIDATION_FPR_INCREASE = 0.01
MAX_HISTORICAL_FPR_INCREASE = 0.02
MAX_HISTORICAL_MACRO_F1_DROP = 0.02
OUTPUT_PREFIX = 'hybrid_balanced_kd2018'


def get_fpr(metrics):
    return float(metrics['attack_fpr'] or 0.0)


def train_only_mmd_calibration(train_X):
    """Fit MMD baseline entirely within pre-stream training data.

    Do NOT calibrate from Thursday or Friday distributions/labels.
    The original helper chooses the RBF gamma and 96-sample reference only
    from training data. We replace its independent-random-window threshold
    with a high quantile of CHRONOLOGICAL training windows.
    """
    reference, gamma, _ = calibrate_mmd(train_X, window=MMD_WINDOW)
    max_start = len(train_X) - MMD_WINDOW
    if max_start < MMD_WINDOW:
        raise ValueError('Not enough training data for chronological MMD calibration')
    # Fixed, chronological positions. Reference and windows are all training-only.
    starts = np.linspace(0, max_start, num=min(96, max_start + 1), dtype=int)
    scores = np.asarray([
        rbf_mmd2(reference, train_X[s:s + MMD_WINDOW], gamma)
        for s in np.unique(starts)
    ], dtype=np.float64)
    threshold = float(np.quantile(scores, MMD_CALIBRATION_QUANTILE))
    return reference, gamma, threshold, scores


def balanced_candidate(active, recent, replay, batch):
    """Train using observed labels only, with recent+historical class balance."""
    xs = np.concatenate([a for a, _ in recent], axis=0)
    ys = np.concatenate([b for _, b in recent], axis=0)
    n_recent = len(ys)

    rx, ry = replay.sample(REPLAY_SAMPLES)
    if rx is not None:
        xs = np.concatenate([xs, rx], axis=0)
        ys = np.concatenate([ys, ry], axis=0)

    rng = np.random.default_rng(1000 + batch)
    selections = []
    for klass in (0, 1):
        ids = np.flatnonzero(ys == klass)
        if len(ids) == 0:
            return None, f'missing_training_class_{klass}'
        weights = np.where(ids < n_recent, RECENT_SAMPLING_WEIGHT, 1.0)
        weights = weights / weights.sum()
        selections.append(rng.choice(ids, size=BALANCED_PER_CLASS,
                                     replace=True, p=weights))
    keep = np.concatenate(selections)
    rng.shuffle(keep)
    xfit, yfit = xs[keep], ys[keep]

    teacher = clone(active).eval()
    candidate = clone(active)
    fit(candidate, xfit, yfit, epochs=CANDIDATE_EPOCHS,
        lr=CANDIDATE_LR, teacher=teacher,
        distill_weight=DISTILLATION_WEIGHT, batch_size=256)
    return candidate, 'trained'


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    path = Path('data/processed/scaled_data.npz')
    if not path.is_file():
        raise SystemExit('Missing processed data. Run in experiments/cicids2018.')
    data = np.load(path)
    X = data['X']
    y = data['y'].astype(np.int8)
    fit_end, train_end, dev_end = (int(data[k]) for k in
                                   ('fit_end', 'train_end', 'dev_end'))
    if not (0 < fit_end < train_end < dev_end < len(y)):
        raise ValueError('Invalid chronological split boundaries')
    if len(np.unique(y[:fit_end])) != 2 or len(np.unique(y[fit_end:train_end])) != 2:
        raise ValueError('Training and historical guard periods must have both classes')

    static = load_model()
    active = clone(static)
    historical_X, historical_y = X[fit_end:train_end], y[fit_end:train_end]
    historical_baseline = score(historical_y, probabilities(static, historical_X))

    reference, gamma, mmd_threshold, train_mmd = train_only_mmd_calibration(X[:fit_end])
    print('MMD threshold (train-only chronological 99th percentile):',
          round(mmd_threshold, 6), 'gamma:', round(gamma, 6), flush=True)

    adwin = drift.ADWIN(delta=ADWIN_DELTA)
    error_history = deque(maxlen=512)
    recent = deque(maxlen=RECENT_BATCHES)
    replay = BalancedReservoir(capacity_per_class=250, seed=42)
    replay.add(X[:fit_end], y[:fit_end])

    pending = None
    accepted_at, proposed_at, decisions, rows = [], [], [], []
    last_attempt = -COOLDOWN_BATCHES - 1
    last_mmd_warning = -10**9
    mmd_warnings = 0
    adwin_alarms = 0
    rising_error_alarms = 0
    hybrid_triggers = 0
    all_proba, all_y = [], []
    process = psutil.Process()
    prediction_seconds = 0.0
    update_seconds = 0.0

    for batch, start in enumerate(range(dev_end, len(y), BATCH_SIZE)):
        stop = min(start + BATCH_SIZE, len(y))
        xb, yb = X[start:stop], y[start:stop]

        # Prequential predictions: labels NOT used yet.
        t0 = time.perf_counter()
        active_probs = probabilities(active, xb)
        prediction_seconds += time.perf_counter() - t0
        all_proba.extend(active_probs.tolist())
        all_y.extend(yb.tolist())  # Stored for OFFLINE scoring, not fed to prediction.

        # Unlabeled distribution monitoring; reference and threshold train-only.
        if len(xb) >= 32:
            mmd_value = rbf_mmd2(reference, xb[:min(MMD_WINDOW, len(xb))], gamma)
            mmd_warning = bool(mmd_value > mmd_threshold)
        else:
            mmd_value, mmd_warning = float('nan'), False
        if mmd_warning:
            mmd_warnings += 1
            last_mmd_warning = batch

        # Current verified labels become available AFTER predictions.
        missed = int(((yb == 1) & (active_probs < 0.5)).sum())
        batch_error = (active_probs >= 0.5).astype(np.int8) != yb
        error_rise = False
        for e in batch_error:
            value = int(e)
            error_history.append(value)
            adwin.update(value)
            if adwin.drift_detected:
                adwin_alarms += 1
                if len(error_history) >= 256:
                    arr = np.asarray(error_history)
                    old = arr[:len(arr) // 2].mean()
                    new = arr[len(arr) // 2:].mean()
                    if new > old + MIN_ERROR_INCREASE:
                        error_rise = True
                        rising_error_alarms += 1

        accepted_now = False
        if pending is not None:
            # Candidate hasn't seen this batch's labels in training.
            candidate_probs = probabilities(pending['model'], xb)
            pending['ys'].append(yb.copy())
            pending['incumbent_probs'].append(active_probs.copy())
            pending['candidate_probs'].append(candidate_probs.copy())
            pending['batches'] += 1
            vy = np.concatenate(pending['ys'])
            enough = (int((vy == 1).sum()) >= MIN_VALIDATION_ATTACKS
                      and int((vy == 0).sum()) >= MIN_VALIDATION_BENIGN)
            exhausted = pending['batches'] >= MAX_VALIDATION_BATCHES
            if enough or exhausted:
                result = {
                    'proposal_batch': pending['proposal_batch'],
                    'decision_batch': batch,
                    'validation_flows': int(len(vy)),
                    'validation_attacks': int((vy == 1).sum()),
                    'validation_benign': int((vy == 0).sum()),
                    'candidate_trained_on_validation': False,
                }
                if enough:
                    incumbent_metrics = score(vy, np.concatenate(pending['incumbent_probs']))
                    candidate_metrics = score(vy, np.concatenate(pending['candidate_probs']))
                    history_metrics = score(
                        historical_y, probabilities(pending['model'], historical_X))
                    recall_ok = (candidate_metrics['recall'] >=
                                 incumbent_metrics['recall'] + MIN_RECALL_GAIN
                                 and candidate_metrics['f1'] >= incumbent_metrics['f1'])
                    fpr_ok = (get_fpr(candidate_metrics) <=
                              get_fpr(incumbent_metrics) + MAX_VALIDATION_FPR_INCREASE)
                    historical_ok = (
                        history_metrics['macro_f1'] >= historical_baseline['macro_f1']
                        - MAX_HISTORICAL_MACRO_F1_DROP
                        and get_fpr(history_metrics) <= get_fpr(historical_baseline)
                        + MAX_HISTORICAL_FPR_INCREASE)
                    accept = bool(recall_ok and fpr_ok and historical_ok)
                    result.update({
                        'status': 'accepted' if accept else 'rejected',
                        'active_validation': incumbent_metrics,
                        'candidate_validation': candidate_metrics,
                        'candidate_historical': history_metrics,
                        'passes_recall_f1': bool(recall_ok),
                        'passes_fpr_guard': bool(fpr_ok),
                        'passes_retention_guard': bool(historical_ok),
                    })
                    if accept:
                        active = pending['model']
                        accepted_at.append(batch)
                        accepted_now = True
                        # Previous error stream no longer describes this new model.
                        adwin = drift.ADWIN(delta=ADWIN_DELTA)
                        error_history.clear()
                else:
                    result['status'] = 'insufficient_validation_classes'
                decisions.append(result)
                print(f"Decision batch {batch:03}: {result['status']} "
                      f"(attacks={result['validation_attacks']})", flush=True)
                pending = None

        # Current labeled batch joins recent training context now.
        recent.append((xb.copy(), yb.copy()))
        hybrid_alert = bool(error_rise and (batch - last_mmd_warning <= MMD_RECENT_BATCHES))
        if hybrid_alert:
            hybrid_triggers += 1

        trained_now = False
        if (pending is None and not accepted_now and hybrid_alert
                and missed >= MIN_MISSED_ATTACKS
                and batch - last_attempt >= COOLDOWN_BATCHES):
            t1 = time.perf_counter()
            candidate, reason = balanced_candidate(active, recent, replay, batch)
            update_seconds += time.perf_counter() - t1
            if candidate is not None:
                pending = {'model': candidate, 'proposal_batch': batch, 'ys': [],
                           'incumbent_probs': [], 'candidate_probs': [], 'batches': 0}
                proposed_at.append(batch)
                last_attempt = batch
                trained_now = True
                print(f'Proposed hybrid candidate batch {batch:03} '
                      f'(missed={missed}, MMD={mmd_value:.4f})', flush=True)

        # Replay receives current labeled data AFTER candidate generation and
        # after candidate has been scored on this batch, preventing label leakage.
        replay.add(xb, yb)
        rows.append({
            'batch': batch, 'stream_start': start, 'stream_stop': stop,
            'attacks': int((yb == 1).sum()), 'missed_before_learn': missed,
            'mmd2': float(mmd_value), 'mmd_warning': mmd_warning,
            'adwin_error_increase': error_rise, 'hybrid_trigger': hybrid_alert,
            'candidate_proposed': trained_now, 'accepted_update': accepted_now,
            'pending_candidate': bool(pending is not None),
            'ram_mb': process.memory_info().rss / 1e6,
        })
        if batch % 50 == 0:
            print(f'Batch {batch:03} | attacks={int((yb == 1).sum())}'
                  f' | missed={missed} | MMD={int(mmd_warning)}'
                  f' | ADWIN rise={int(error_rise)} | accepted={len(accepted_at)}',
                  flush=True)

    static_metrics = score(y[dev_end:], probabilities(static, X[dev_end:]))
    hybrid_metrics = score(np.asarray(all_y), np.asarray(all_proba))
    historical_after = score(historical_y, probabilities(active, historical_X))

    summary = {
        'static_stream': static_metrics,
        'integrated_hybrid_stream': hybrid_metrics,
        'historical_before': historical_baseline,
        'historical_after': historical_after,
        'batches': len(rows),
        'mmd_threshold_train_only': mmd_threshold,
        'mmd_training_window_q50': float(np.median(train_mmd)),
        'mmd_training_window_q99': float(np.quantile(train_mmd, .99)),
        'mmd_warnings': mmd_warnings,
        'adwin_alarms': adwin_alarms,
        'adwin_positive_error_alarms': rising_error_alarms,
        'hybrid_triggers': hybrid_triggers,
        'proposed_updates': len(proposed_at),
        'accepted_updates': len(accepted_at),
        'proposal_batches': proposed_at,
        'accepted_batches': accepted_at,
        'unresolved_candidate_at_end': bool(pending is not None),
        'candidate_decisions': decisions,
        'mean_prediction_seconds_per_batch': prediction_seconds / max(len(rows), 1),
        'total_training_seconds': update_seconds,
        'peak_observed_ram_mb': float(max(r['ram_mb'] for r in rows)),
        'settings': {
            'batch_size': BATCH_SIZE, 'mmd_quantile': MMD_CALIBRATION_QUANTILE,
            'mmd_recent_batches': MMD_RECENT_BATCHES,
            'adwin_delta': ADWIN_DELTA,
            'candidate_epochs': CANDIDATE_EPOCHS,
            'distillation_weight': DISTILLATION_WEIGHT,
            'balanced_per_class': BALANCED_PER_CLASS,
            'candidate_validation_recall_gain': MIN_RECALL_GAIN,
        },
        'protocol': 'predict batch -> receive verified labels -> MMD/ADWIN gate '
                    '-> balanced replay+KD candidate -> validate on later batches '
                    '-> accept/reject; no Friday training',
        'warning': 'Exploratory Thursday 2018, previously inspected/tuned; historical '
                   'holdout reused in update decisions. MMD alarms are not proven '
                   'concept drift, and true drift detection delay/FAR unknown.',
    }
    results_dir = Path('results')
    models_dir = Path('models')
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    summary_file = results_dir / f'{OUTPUT_PREFIX}_summary.json'
    summary_file.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(results_dir / f'{OUTPUT_PREFIX}_batches.csv', index=False)
    torch.save({'n_features': int(X.shape[1]),
                'state_dict': active.cpu().state_dict()},
               models_dir / f'{OUTPUT_PREFIX}.pt')

    print('\nSTATIC   Macro-F1=', round(static_metrics['macro_f1'], 4),
          'Recall=', round(static_metrics['recall'], 4))
    print('INTEGRATED Macro-F1=', round(hybrid_metrics['macro_f1'], 4),
          'Recall=', round(hybrid_metrics['recall'], 4))
    print('MMD warnings:', mmd_warnings,
          'ADWIN rises:', rising_error_alarms,
          'Proposed:', len(proposed_at), 'Accepted:', len(accepted_at))
    print('Saved:', summary_file)


if __name__ == '__main__':
    main()
