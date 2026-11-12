"""Exploratory supervised replay with prospective candidate validation (CICIDS2018).

Evaluate the active model on each full chronological batch before using its labels.
Train candidates on already-labeled data only. Validate each candidate against
future batches that the candidate has never trained on, before deployment.
Not an MMD/ADWIN hybrid, and not an independent test (Thursday already inspected).
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import load_model, probabilities, score, clone, fit
from replay import BalancedReservoir

BATCH_SIZE = 256
MIN_MISSED_ATTACKS = 8
COOLDOWN_BATCHES = 3
VALIDATION_MIN_ATTACKS = 8
VALIDATION_MIN_BENIGN = 64
MAX_VALIDATION_BATCHES = 12
MIN_RECALL_GAIN = 0.05
MAX_FPR_INCREASE = 0.01
MAX_HISTORICAL_FPR_INCREASE = 0.02
MAX_HISTORICAL_MACRO_F1_DROP = 0.02


def fpr(metrics):
    return float(metrics.get('attack_fpr') or 0.0)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    path = Path('data/processed/scaled_data.npz')
    if not path.is_file():
        raise SystemExit('Missing scaled data: run training in experiments/cicids2018 first.')
    data = np.load(path)
    X, y = data['X'], data['y'].astype(int)
    fit_end, train_end, dev_end = (int(data[k]) for k in ('fit_end', 'train_end', 'dev_end'))
    if not (0 < fit_end < train_end < dev_end < len(y)):
        raise ValueError('Invalid chronological split indices')

    initial = load_model()
    active = clone(initial)
    holdout_X, holdout_y = X[fit_end:train_end], y[fit_end:train_end]
    historical_baseline = score(holdout_y, probabilities(initial, holdout_X))

    replay = BalancedReservoir(capacity_per_class=250, seed=42)
    replay.add(X[:fit_end], y[:fit_end])
    recent = deque(maxlen=4)
    prequential_probs, labels_seen = [], []
    rows, decisions = [], []
    pending = None
    accepted, attempted = [], []
    last_attempt = -10

    for batch, start in enumerate(range(dev_end, len(y), BATCH_SIZE)):
        stop = min(start + BATCH_SIZE, len(y))
        xb, yb = X[start:stop], y[start:stop]

        # Predictions are fixed BEFORE the batch labels are revealed.
        active_prob = probabilities(active, xb)
        prequential_probs.extend(active_prob.tolist())
        labels_seen.extend(yb.tolist())
        missed = int(((yb == 1) & (active_prob < 0.5)).sum())
        accepted_now = False

        # The candidate is frozen. This batch has never been used to train it.
        if pending is not None:
            candidate_prob = probabilities(pending['model'], xb)
            pending['labels'].append(yb.copy())
            pending['active_probs'].append(active_prob.copy())
            pending['candidate_probs'].append(candidate_prob.copy())
            pending['batches'] += 1
            vy = np.concatenate(pending['labels'])

            enough = (int((vy == 1).sum()) >= VALIDATION_MIN_ATTACKS and
                      int((vy == 0).sum()) >= VALIDATION_MIN_BENIGN)
            exhausted = pending['batches'] >= MAX_VALIDATION_BATCHES
            if enough or exhausted:
                verdict = {
                    'proposal_batch': pending['proposal_batch'],
                    'decision_batch': batch,
                    'validation_flows': int(len(vy)),
                    'validation_attacks': int((vy == 1).sum()),
                    'validation_benign': int((vy == 0).sum()),
                    'candidate_trained_on_validation': False,
                }
                if enough:
                    live = score(vy, np.concatenate(pending['active_probs']))
                    cand = score(vy, np.concatenate(pending['candidate_probs']))
                    hist = score(holdout_y, probabilities(pending['model'], holdout_X))
                    improves = (cand['recall'] >= live['recall'] + MIN_RECALL_GAIN and
                                cand['f1'] >= live['f1'])
                    alarm_ok = fpr(cand) <= fpr(live) + MAX_FPR_INCREASE
                    retention_ok = (
                        hist['macro_f1'] >= historical_baseline['macro_f1'] - MAX_HISTORICAL_MACRO_F1_DROP
                        and fpr(hist) <= fpr(historical_baseline) + MAX_HISTORICAL_FPR_INCREASE
                    )
                    accept = bool(improves and alarm_ok and retention_ok)
                    verdict.update({
                        'status': 'accepted' if accept else 'rejected',
                        'active_validation': live,
                        'candidate_validation': cand,
                        'candidate_historical': hist,
                        'passes_recall_f1': bool(improves),
                        'passes_fpr_guard': bool(alarm_ok),
                        'passes_retention_guard': bool(retention_ok),
                    })
                    if accept:
                        active = pending['model']
                        accepted.append(batch)
                        accepted_now = True
                else:
                    verdict['status'] = 'insufficient_validation_classes'
                decisions.append(verdict)
                print(f"Decision batch {batch:03}: {verdict['status']} "
                      f"(validation attacks={verdict['validation_attacks']})", flush=True)
                pending = None

        # Now current batch labels are available, and may be used for learning.
        recent.append((xb.copy(), yb.copy()))
        if (pending is None and not accepted_now and missed >= MIN_MISSED_ATTACKS
                and batch - last_attempt >= COOLDOWN_BATCHES):
            xs = np.concatenate([a for a, _ in recent])
            ys = np.concatenate([b for _, b in recent])
            replay_x, replay_y = replay.sample(256)
            if replay_x is not None:
                xs = np.concatenate([xs, replay_x])
                ys = np.concatenate([ys, replay_y])
            candidate = clone(active)
            fit(candidate, xs, ys, epochs=6, lr=0.0003, batch_size=256)
            pending = {
                'model': candidate,
                'proposal_batch': batch,
                'labels': [], 'active_probs': [], 'candidate_probs': [],
                'batches': 0,
            }
            attempted.append(batch)
            last_attempt = batch
            print(f'Proposed candidate at batch {batch:03} '
                  f'(missed attacks={missed})', flush=True)

        # Only historical labeled flows enter replay; current validation batch
        # is added AFTER the pending candidate was validated.
        replay.add(xb, yb)
        rows.append({
            'batch': batch, 'stream_start': start, 'stream_stop': stop,
            'attacks': int((yb == 1).sum()),
            'missed_before_learn': missed,
            'candidate_pending': bool(pending is not None),
            'accepted_update_after_batch': bool(accepted_now),
        })
        if batch % 50 == 0:
            print(f'Batch {batch:03} | attacks={int((yb == 1).sum())} '
                  f'| missed={missed} | accepted={len(accepted)}', flush=True)

    static_metrics = score(y[dev_end:], probabilities(initial, X[dev_end:]))
    guarded_metrics = score(np.asarray(labels_seen), np.asarray(prequential_probs))
    historical_after = score(holdout_y, probabilities(active, holdout_X))
    summary = {
        'static_stream': static_metrics,
        'candidate_guarded_replay_stream': guarded_metrics,
        'historical_before': historical_baseline,
        'historical_after': historical_after,
        'proposed_updates': len(attempted),
        'accepted_updates': len(accepted),
        'accepted_decision_batches': accepted,
        'incomplete_candidate_at_stream_end': bool(pending is not None),
        'candidate_decisions': decisions,
        'criteria': {
            'min_validation_attacks': VALIDATION_MIN_ATTACKS,
            'min_validation_benign': VALIDATION_MIN_BENIGN,
            'min_recall_gain': MIN_RECALL_GAIN,
            'max_validation_fpr_increase': MAX_FPR_INCREASE,
            'max_historical_fpr_increase': MAX_HISTORICAL_FPR_INCREASE,
            'max_historical_macro_f1_drop': MAX_HISTORICAL_MACRO_F1_DROP,
        },
        'protocol': 'active predict full batch -> verified labels -> prospective candidate validation -> possible deployment',
        'warning': 'Exploratory Thursday 2018 test already inspected; no MMD/ADWIN triggers and historical holdout reused for candidate acceptance.',
    }
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    Path('results/replay_guarded6ep2018_summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv('results/replay_guarded6ep2018_batches.csv', index=False)
    torch.save({'n_features': int(X.shape[1]), 'state_dict': active.cpu().state_dict()},
               'models/replay_guarded6ep2018.pt')
    print('\nSTATIC:  Macro-F1=', round(static_metrics['macro_f1'], 4),
          'Recall=', round(static_metrics['recall'], 4))
    print('GUARDED: Macro-F1=', round(guarded_metrics['macro_f1'], 4),
          'Recall=', round(guarded_metrics['recall'], 4))
    print('Proposed:', len(attempted), 'Accepted:', len(accepted))
    print('Saved: results/replay_guarded6ep2018_summary.json')


if __name__ == '__main__':
    main()
