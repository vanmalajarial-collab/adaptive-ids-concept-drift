"""Step 13: four ablation comparisons on the SAME chronological evaluation stream.

Modes: static, regular fine-tuning, replay, replay+distillation.
Updates happen AFTER scoring their batch (prequential). These deliberately do
NOT use candidate-validation gates; the full hybrid model in stream.py does.
"""
import argparse
import json
from collections import deque
from pathlib import Path
import numpy as np
from model import fit, probabilities, score, clone
from replay import BalancedReservoir
from model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--interval', type=int, default=5)
    args = parser.parse_args()
    np.random.seed(42)
    arr = np.load('data/processed/scaled_data.npz')
    X, y = arr['X'], arr['y'].astype(int)
    fit_end, dev_end = int(arr['fit_end']), int(arr['dev_end'])
    all_results = {}
    for mode in ('static', 'fine_tuning', 'replay', 'replay_distillation'):
        model = load_model()
        buffer = BalancedReservoir(capacity_per_class=250)
        buffer.add(X[:fit_end], y[:fit_end])
        past = deque(maxlen=4)
        predictions, labels = [], []
        updates = 0
        for idx, start in enumerate(range(dev_end, len(y), args.batch_size)):
            stop = min(start + args.batch_size, len(y))
            xb, yb = X[start:stop], y[start:stop]
            # Prediction/metric evaluated BEFORE current batch is used for learning.
            prediction = probabilities(model, xb)
            predictions.extend(prediction.tolist())
            labels.extend(yb.tolist())
            past.append((xb.copy(), yb.copy()))
            if mode != 'static' and (idx+1) % args.interval == 0:
                recent_x = np.concatenate([xx for xx, _ in past])
                recent_y = np.concatenate([yy for _, yy in past])
                if mode in ('replay','replay_distillation'):
                    xr, yr = buffer.sample(128)
                    if xr is not None:
                        recent_x = np.concatenate((recent_x, xr))
                        recent_y = np.concatenate((recent_y, yr))
                teacher = clone(model) if mode == 'replay_distillation' else None
                model = fit(clone(model), recent_x, recent_y,
                            epochs=4, lr=0.0003, teacher=teacher)
                updates += 1
            # Current batch may enter replay only AFTER its prediction/update.
            buffer.add(xb, yb)
        summary = score(np.asarray(labels), np.asarray(predictions))
        summary['update_count'] = updates
        all_results[mode] = summary
        print(mode, 'macro-F1:', round(summary['macro_f1'], 4), 'updates:', updates)
    # Hybrid runs independently via stream.py; its update rule is drift-triggered.
    Path('results').mkdir(exist_ok=True)
    Path('results/ablation.json').write_text(json.dumps(all_results, indent=2))
    print('Saved results/ablation.json')


if __name__ == '__main__':
    main()
