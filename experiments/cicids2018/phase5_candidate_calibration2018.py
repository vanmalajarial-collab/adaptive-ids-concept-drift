"""Phase 5 exploratory conservative replay with past-only candidate thresholds.

Run in ~/adaptive_ids_project/experiments/cicids2018:
    python phase5_candidate_calibration2018.py

Uses the *existing* phase5_conservative_replay2018.py trainer unchanged.
Each candidate's operating threshold is selected on FEB-14 DEVELOPMENT BENIGN
scores ONLY, before any later Feb-23 validation labels are observed.
The initial OR checkpoint starts at its existing 0.5 threshold. A candidate
may replace it only after later labeled stream validation AND historical
retention checks, both evaluated at the models' own operating thresholds.

Important limitations: Feb-23 is already-inspected DEVELOPMENT data; the
Feb-14 development set and historical guard have been repeatedly consulted
in prior work, so this is NOT an independent estimate. Training assumes
verified batch labels available at batch end. Not an MMD/ADWIN-gated run.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import phase5_conservative_replay2018 as base

OUT_STEM = 'phase5_feb23_candidate_calibrated2018'
TARGET_DEV_BENIGN_FPR = 0.001  # exploratory fixed target; never optimize on Feb-23
INITIAL_THRESHOLD = 0.5


def choose_threshold(benign_probs: np.ndarray, target_fpr: float = TARGET_DEV_BENIGN_FPR):
    """Low as possible threshold without exceeding empirical benign FP budget.

    Uses strictly earlier Feb-14 development benign labels, not later
    candidate-validation labels or Feb-23 attack labels.
    """
    values = np.sort(np.asarray(benign_probs, dtype=np.float64).reshape(-1))
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Missing / invalid benign calibration scores')
    budget = int(math.floor(target_fpr * len(values)))
    pivot = values[len(values) - budget - 1]
    threshold = float(np.nextafter(pivot, np.inf))
    fp = int((values >= threshold).sum())
    if fp > budget:
        raise AssertionError('Calibration FP budget exceeded')
    return threshold, budget, fp


def metrics(y, predictions, probabilities=None):
    """All metrics computed from actual deployed threshold decisions."""
    yy = np.asarray(y, dtype=np.int8)
    pred = np.asarray(predictions, dtype=bool)
    if len(yy) != len(pred):
        raise ValueError('Target / prediction length mismatch')
    tp = int(((yy == 1) & pred).sum())
    fn = int(((yy == 1) & ~pred).sum())
    fp = int(((yy == 0) & pred).sum())
    tn = int(((yy == 0) & ~pred).sum())
    n = len(yy)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    benign_f1 = 2 * tn / max(1, 2 * tn + fp + fn)
    ap = None
    if probabilities is not None and len(np.unique(yy)) == 2:
        ap = float(average_precision_score(yy, probabilities))
    return {
        'accuracy': (tp + tn) / max(1, n),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'macro_f1': (f1 + benign_f1) / 2,
        'pr_auc': ap,
        'attack_fpr': fp / max(1, fp + tn),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


def main():
    for file in (base.CSV, base.PREP, base.SOURCE_MODEL, base.PROCESSED):
        if not file.is_file():
            raise SystemExit(f'Missing file: {file}')
    torch.manual_seed(42)
    np.random.seed(42)

    with np.load(base.PROCESSED) as saved:
        Xold = saved['X']
        yold = saved['y'].astype(np.int8)
        fit_end = int(saved['fit_end'])
        train_end = int(saved['train_end'])
        dev_end = int(saved['dev_end'])
    if not (0 < fit_end < train_end < dev_end <= len(yold)):
        raise ValueError('Unexpected chronological cutoffs for Feb-14')
    historical_X = Xold[fit_end:train_end]
    historical_y = yold[fit_end:train_end]
    dev_X = Xold[train_end:dev_end]
    dev_y = yold[train_end:dev_end]
    cal_benign_X = dev_X[dev_y == 0]
    if len(np.unique(historical_y)) != 2 or len(cal_benign_X) < 1000:
        raise ValueError('Need historical guard both classes and >=1000 development benign')

    X, y, labels, raw_seen, invalid, was_sorted = base.load_feb23()
    static = base.load_checkpoint(base.SOURCE_MODEL, X.shape[1])
    active = base.clone(static)
    active_threshold = INITIAL_THRESHOLD
    historical_baseline = metrics(
        historical_y,
        base.probabilities(static, historical_X) >= INITIAL_THRESHOLD)

    replay = base.BalancedReservoir(capacity_per_class=250, seed=42)
    replay.add(Xold[:fit_end], yold[:fit_end])
    del Xold, yold, dev_X, dev_y  # calibration set is never added to replay
    recent = deque(maxlen=base.RECENT_BATCHES)
    new_attack_pool = deque(maxlen=base.NEW_ATTACK_POOL_CAPACITY)

    candidate_pending = None
    proposals, acceptances, decisions, traces = [], [], [], []
    static_probabilities, active_probabilities, active_decisions = [], [], []
    last_attempt = -base.COOLDOWN_BATCHES - 1
    cumulative_missed = cumulative_attacks = 0
    training_seconds = prediction_seconds = 0.0

    print('Calibration benign flows (Feb-14 development):', len(cal_benign_X))
    print('Target empirical Feb-14 benign FPR:', TARGET_DEV_BENIGN_FPR)
    print('Starting frozen OR decision threshold:', active_threshold, flush=True)

    for batch_id, start in enumerate(range(0, len(y), base.BATCH_SIZE)):
        end = min(start + base.BATCH_SIZE, len(y))
        xb, yb = X[start:end], y[start:end]
        begin = time.perf_counter()
        original_p = base.probabilities(static, xb)
        current_p = base.probabilities(active, xb)
        prediction_seconds += time.perf_counter() - begin
        original_pred = original_p >= INITIAL_THRESHOLD
        threshold_at_prediction = active_threshold
        current_pred = current_p >= threshold_at_prediction
        static_probabilities.append(original_p)
        active_probabilities.append(current_p)
        active_decisions.append(current_pred)

        # No batch's label is used before its prediction is captured.
        attack_count = int(yb.sum())
        missed = int(((yb == 1) & ~current_pred).sum())
        cumulative_missed += missed
        cumulative_attacks += attack_count
        accepted_now = False

        if candidate_pending is not None:
            cand_p = base.probabilities(candidate_pending['model'], xb)
            candidate_pending['ys'].append(yb.copy())
            candidate_pending['active_scores'].append(current_p.copy())
            candidate_pending['active_preds'].append(current_pred.copy())
            candidate_pending['candidate_scores'].append(cand_p.copy())
            candidate_pending['n_batches'] += 1
            vy = np.concatenate(candidate_pending['ys'])
            positives = int(vy.sum())
            negatives = int(len(vy) - positives)
            sufficient = (positives >= base.MIN_VALIDATION_ATTACKS and
                          negatives >= base.MIN_VALIDATION_BENIGN)
            exhausted = candidate_pending['n_batches'] >= base.MAX_VALIDATION_BATCHES
            if sufficient or exhausted:
                d = {
                    'proposal_batch': candidate_pending['proposal_batch'],
                    'decision_batch': batch_id,
                    'validation_flows': int(len(vy)),
                    'validation_attacks': positives,
                    'validation_benign': negatives,
                    'candidate_trained_on_validation': False,
                    'active_threshold': candidate_pending['active_threshold'],
                    'candidate_threshold': candidate_pending['threshold'],
                    'calibration_feb14_benign': len(cal_benign_X),
                    'calibration_fp_budget': candidate_pending['calibration_fp_budget'],
                    'calibration_fp': candidate_pending['calibration_fp'],
                }
                if sufficient:
                    active_pooled = np.concatenate(candidate_pending['active_scores'])
                    active_pred_pooled = np.concatenate(candidate_pending['active_preds'])
                    candidate_pooled = np.concatenate(candidate_pending['candidate_scores'])
                    inc_m = metrics(vy, active_pred_pooled, active_pooled)
                    cand_m = metrics(
                        vy, candidate_pooled >= candidate_pending['threshold'],
                        candidate_pooled)
                    hist_probs = base.probabilities(candidate_pending['model'], historical_X)
                    hist_m = metrics(
                        historical_y, hist_probs >= candidate_pending['threshold'],
                        hist_probs)
                    recall_good = (
                        cand_m['recall'] >= inc_m['recall'] + base.MIN_VALIDATION_RECALL_GAIN
                        and cand_m['f1'] >= inc_m['f1'])
                    fpr_good = (
                        cand_m['attack_fpr'] <= inc_m['attack_fpr']
                        + base.MAX_VALIDATION_FPR_INCREASE)
                    hist_good = (
                        hist_m['macro_f1'] >= historical_baseline['macro_f1']
                        - base.MAX_HISTORICAL_MACRO_F1_DROP
                        and hist_m['attack_fpr'] <= historical_baseline['attack_fpr']
                        + base.MAX_HISTORICAL_FPR_INCREASE)
                    ok = bool(recall_good and fpr_good and hist_good)
                    d.update({
                        'status': 'accepted' if ok else 'rejected',
                        'active_validation': inc_m,
                        'candidate_validation': cand_m,
                        'candidate_historical': hist_m,
                        'passes_recall_f1': bool(recall_good),
                        'passes_fpr_guard': bool(fpr_good),
                        'passes_retention_guard': bool(hist_good),
                    })
                    if ok:
                        active = candidate_pending['model']
                        active_threshold = candidate_pending['threshold']
                        acceptances.append(batch_id)
                        accepted_now = True
                else:
                    d['status'] = 'insufficient_validation_classes'
                decisions.append(d)
                print(f'Validation batch {batch_id}: {d["status"]} | '
                      f'threshold={d["candidate_threshold"]:.6g} | '
                      f'positives={positives} negatives={negatives}', flush=True)
                candidate_pending = None

        # This batch's labels become eligible for strictly LATER training.
        recent.append((xb.copy(), yb.copy()))
        for row in xb[yb == 1]:
            new_attack_pool.append(np.array(row, copy=True))

        proposed = False
        if (candidate_pending is None and not accepted_now
                and cumulative_missed >= base.MIN_CUMULATIVE_MISSES
                and len(new_attack_pool) >= base.MIN_NEW_ATTACK_EXAMPLES
                and batch_id - last_attempt >= base.COOLDOWN_BATCHES):
            begin = time.perf_counter()
            candidate, reason = base.build_candidate(
                active, recent, new_attack_pool, replay, batch_id)
            if candidate is not None:
                cand_dev_scores = base.probabilities(candidate, cal_benign_X)
                new_threshold, budget, calibration_fp = choose_threshold(cand_dev_scores)
                candidate_pending = {
                    'model': candidate,
                    'proposal_batch': batch_id,
                    'threshold': new_threshold,
                    'active_threshold': active_threshold,
                    'calibration_fp_budget': budget,
                    'calibration_fp': calibration_fp,
                    'ys': [], 'active_scores': [], 'active_preds': [],
                    'candidate_scores': [], 'n_batches': 0,
                }
                proposed = True
                proposals.append(batch_id)
                last_attempt = batch_id
                cumulative_missed = 0
                print(f'Candidate batch {batch_id}: calibrated threshold '
                      f'{new_threshold:.6g} (Feb-14 budget {budget}, fp {calibration_fp})',
                      flush=True)
            training_seconds += time.perf_counter() - begin

        replay.add(xb, yb)
        traces.append({
            'batch': batch_id, 'attacks': attack_count,
            'missed_before_learning': missed,
            'active_threshold_at_prediction': threshold_at_prediction,
            'candidate_proposed': proposed,
            'candidate_accepted': accepted_now,
            'candidate_pending': candidate_pending is not None,
        })
        if batch_id % 500 == 0:
            print(f'Batch {batch_id:04} | attacks={attack_count} missed={missed} '
                  f'proposed={len(proposals)} accepted={len(acceptances)}', flush=True)

    # Summary uses predictions made at the threshold ACTIVE AT THAT TIME.
    # Raw AP across model changes is reported for context only.
    predictions = np.concatenate(active_decisions)
    probs = np.concatenate(active_probabilities)
    after_probs = base.probabilities(active, historical_X)
    summary = {
        'dataset': base.CSV.name,
        'purpose': 'Exploratory per-candidate threshold calibration, conservative replay, and future-batch guards',
        'protocol': 'Feb-23 predict batch first; verified labels then candidate training; '
                    'candidate threshold selected on Feb-14 dev BENIGN only; '
                    'later Feb-23 batches used only for acceptance checks',
        'calibration': {
            'source': 'Feb-14 development only, excluded from candidate replay/training',
            'benign_count': len(cal_benign_X),
            'target_empirical_fpr': TARGET_DEV_BENIGN_FPR,
            'initial_threshold': INITIAL_THRESHOLD,
            'final_active_threshold': active_threshold,
        },
        'raw_rows_seen': int(raw_seen),
        'evaluated_rows': int(len(y)),
        'invalid_rows': int(invalid),
        'csv_already_chronological': bool(was_sorted),
        'label_counts': dict(Counter(labels)),
        'attack_rows': int(y.sum()),
        'benign_rows': int((y == 0).sum()),
        'static_or_stream': base.score(y, np.concatenate(static_probabilities)),
        'calibrated_candidate_stream': metrics(y, predictions, probs),
        'historical_before': historical_baseline,
        'historical_after': metrics(
            historical_y, after_probs >= active_threshold, after_probs),
        'proposed_updates': len(proposals),
        'accepted_updates': len(acceptances),
        'proposal_batches': proposals,
        'accepted_batches': acceptances,
        'unresolved_candidate_at_end': candidate_pending is not None,
        'decisions': decisions,
        'training_plus_calibration_seconds': training_seconds,
        'mean_prediction_seconds_per_batch': prediction_seconds / max(1, len(traces)),
        'settings': {
            'batch_size': base.BATCH_SIZE,
            'candidate_epochs': base.CANDIDATE_EPOCHS,
            'sampled_benign_per_candidate': base.NEGATIVE_TRAIN_SAMPLES,
            'sampled_attack_per_candidate': base.POSITIVE_TRAIN_SAMPLES,
            'distillation_weight': base.DISTILL_WEIGHT,
            'calibration_target_dev_fpr': TARGET_DEV_BENIGN_FPR,
            'validation_recall_gain': base.MIN_VALIDATION_RECALL_GAIN,
            'max_validation_fpr_increase': base.MAX_VALIDATION_FPR_INCREASE,
            'max_historical_fpr_increase': base.MAX_HISTORICAL_FPR_INCREASE,
            'max_historical_macro_f1_drop': base.MAX_HISTORICAL_MACRO_F1_DROP,
        },
        'warning': 'Already-inspected Feb-23 development. Threshold/validation protocol '
                   'designed after earlier tests; historical guard reused for selection. '
                   'A calibration FPR constraint on Feb-14 does NOT guarantee future FPR. '
                   'Not independent evidence; no MMD/ADWIN gating in this run.',
    }
    outdir = Path('results')
    outdir.mkdir(exist_ok=True)
    path = outdir / f'{OUT_STEM}_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame(traces).to_csv(outdir / f'{OUT_STEM}_batches.csv', index=False)
    Path('models').mkdir(exist_ok=True)
    torch.save({'n_features': int(X.shape[1]),
                'state_dict': active.cpu().state_dict(),
                'operating_threshold': float(active_threshold)},
               Path('models') / f'{OUT_STEM}.pt')
    print('\n=== PHASE 5 CANDIDATE THRESHOLD CALIBRATION ===')
    print('STATIC recall:', round(summary['static_or_stream']['recall'], 4),
          '| macro-F1:', round(summary['static_or_stream']['macro_f1'], 4))
    print('CALIBRATED recall:', round(summary['calibrated_candidate_stream']['recall'], 4),
          '| macro-F1:', round(summary['calibrated_candidate_stream']['macro_f1'], 4))
    print('Proposals:', len(proposals), '| Accepted:', len(acceptances),
          '| Final threshold:', round(active_threshold, 8))
    print('Saved:', path)


if __name__ == '__main__':
    main()
