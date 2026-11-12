"""Exploratory supervised replay benchmark for CICIDS2018.

Predict each Thursday batch before revealing labels; after verification, optionally
fine-tune on recent labeled batches + past reservoir. Not an independently
validated IDS; updates are not subject to candidate acceptance gates.
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


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    data = np.load('data/processed/scaled_data.npz')
    X = data['X']
    y = data['y'].astype(int)
    fit_end, train_end, dev_end = (
        int(data[k]) for k in ('fit_end', 'train_end', 'dev_end')
    )

    static = load_model()
    adaptive = clone(static)
    old_hist = score(y[fit_end:train_end],
                     probabilities(static, X[fit_end:train_end]))
    static_probs = probabilities(static, X[dev_end:])

    replay = BalancedReservoir(capacity_per_class=250, seed=42)
    replay.add(X[:fit_end], y[:fit_end])
    recent = deque(maxlen=4)
    all_probs = []
    batch_rows = []
    update_batches = []
    last_update = -10
    batch_size = 256

    for batch, start in enumerate(range(dev_end, len(y), batch_size)):
        stop = min(start + batch_size, len(y))
        xb, yb = X[start:stop], y[start:stop]

        # Predict BEFORE the current batch's verified labels are used for learning.
        prob = probabilities(adaptive, xb)
        all_probs.extend(prob.tolist())
        missed = int(((yb == 1) & (prob < 0.5)).sum())
        attacks = int((yb == 1).sum())
        benign = int((yb == 0).sum())
        recent.append((xb.copy(), yb.copy()))

        # Labels are now assumed verified at END of this batch.
        # This is a supervised scheduled-update baseline, NOT a hybrid MMD/ADWIN trigger.
        do_update = (missed >= 8 and batch - last_update >= 3)
        if do_update:
            xx = np.concatenate([part_x for part_x, _ in recent])
            yy = np.concatenate([part_y for _, part_y in recent])
            replay_x, replay_y = replay.sample(256)
            if replay_x is not None:
                xx = np.concatenate([xx, replay_x])
                yy = np.concatenate([yy, replay_y])
            candidate = clone(adaptive)
            fit(candidate, xx, yy, epochs=2, lr=0.0003, batch_size=256)
            adaptive = candidate
            last_update = batch
            update_batches.append(batch)

        # The current batch joins replay only after prediction and optional update.
        replay.add(xb, yb)
        batch_rows.append({
            'batch': batch,
            'stream_start': start,
            'stream_stop': stop,
            'attacks': attacks,
            'benign': benign,
            'missed_attacks_before_update': missed,
            'update_after_batch': bool(do_update),
        })
        if batch % 50 == 0 or do_update:
            print(f'Batch {batch:03} | attacks={attacks} | '
                  f'missed={missed} | updated_after_batch={int(do_update)}',
                  flush=True)

    replay_metrics = score(y[dev_end:], np.asarray(all_probs))
    static_metrics = score(y[dev_end:], static_probs)
    updated_hist = score(y[fit_end:train_end],
                         probabilities(adaptive, X[fit_end:train_end]))
    output = {
        'static_stream': static_metrics,
        'supervised_replay_stream': replay_metrics,
        'historical_before': old_hist,
        'historical_after': updated_hist,
        'accepted_updates': None,
        'scheduled_updates_without_candidate_validation': len(update_batches),
        'update_batches': update_batches,
        'protocol': 'predict full batch -> receive verified labels -> optional learn',
        'warning': 'Exploratory supervised replay baseline on already-inspected Thursday; no MMD/ADWIN gate or candidate validation, not independent test.'
    }
    Path('results').mkdir(exist_ok=True)
    pd.DataFrame(batch_rows).to_csv('results/replay2018_probe_batches.csv', index=False)
    Path('results/replay2018_probe_summary.json').write_text(
        json.dumps(output, indent=2))
    torch.save({
        'n_features': int(X.shape[1]),
        'state_dict': adaptive.cpu().state_dict(),
    }, 'models/replay2018_probe.pt')
    print('\nSTATIC:      F1=', round(static_metrics['f1'], 4),
          'Recall=', round(static_metrics['recall'], 4))
    print('REPLAY:      F1=', round(replay_metrics['f1'], 4),
          'Recall=', round(replay_metrics['recall'], 4))
    print('HISTORICAL:  Macro-F1 before=', round(old_hist['macro_f1'], 4),
          'after=', round(updated_hist['macro_f1'], 4))
    print('Updates:', len(update_batches))
    print('Saved: results/replay2018_probe_summary.json')


if __name__ == '__main__':
    main()
