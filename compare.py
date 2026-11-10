"""Step 13: static MLP vs Adaptive Random Forest streaming baseline.
Run --arf-limit 0 for full data, slow for large datasets on CPU.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from river import forest
from model import probabilities, score
from model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arf-train-limit', type=int, default=12000,
                        help='For faster trial. 0 uses all available initial training data.')
    parser.add_argument('--arf-test-limit', type=int, default=10000,
                        help='For faster trial. 0 uses whole evaluation stream.')
    args = parser.parse_args()
    arr = np.load('data/processed/scaled_data.npz')
    X, y = arr['X'], arr['y'].astype(int)
    fit_end, dev_end = int(arr['fit_end']), int(arr['dev_end'])
    xeval, yeval = X[dev_end:], y[dev_end:]
    static = score(yeval, probabilities(load_model(), xeval))
    n_train = fit_end if args.arf_train_limit <= 0 else min(args.arf_train_limit, fit_end)
    n_eval = len(yeval) if args.arf_test_limit <= 0 else min(args.arf_test_limit, len(yeval))
    # Train on chronologically FIRST n_train observations, not a future sample.
    arf = forest.ARFClassifier(n_models=5, seed=42)
    print(f'ARF initial samples={n_train}, evaluation samples={n_eval}')
    for i in range(n_train):
        arf.learn_one(dict(enumerate(map(float, X[i]))), int(y[i]))
        if (i+1) % 5000 == 0:
            print('ARF initial fit:', i+1)
    probs = []
    for i in range(n_eval):
        xi = dict(enumerate(map(float, xeval[i])))
        pred = arf.predict_proba_one(xi).get(1, 0.0)
        probs.append(float(pred))
        # Online prequential learning: true label AFTER its own prediction.
        arf.learn_one(xi, int(yeval[i]))
        if (i+1) % 5000 == 0:
            print('ARF prequential:', i+1)
    comparison = {'static_mlp_full_stream': static,
                  'arf_prequential_subset': score(yeval[:n_eval], np.asarray(probs)),
                  'arf_train_rows': n_train, 'arf_test_rows': n_eval,
                  'warning': ('Not strictly comparable if ARF uses fewer initial training rows '
                              'or test rows. Set both limits 0 for same data budget.')}
    Path('results').mkdir(exist_ok=True)
    Path('results/comparison.json').write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
