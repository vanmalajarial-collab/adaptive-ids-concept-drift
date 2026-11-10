"""05: Chronological split, training-only median imputation/scaling, MLP checkpoint."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from model import MLP, DEVICE, fit, probabilities, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-fraction', type=float, default=0.60)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    arr = np.load('data/processed/raw_data.npz')
    X, y = arr['X'], arr['y'].astype(int)
    names = arr['columns'].tolist()
    n = len(y)
    split_train = int(n * args.train_fraction)
    split_dev = int(n * min(0.90, args.train_fraction + 0.15))
    # Retain last 10% of training period as a historical holdout, NEVER train on it.
    split_fit = int(split_train * 0.90)
    if min(split_fit, split_dev-split_train, n-split_dev) < 50:
        raise ValueError('Dataset too small. Need at least ~1000 timestamped flows.')
    if len(np.unique(y[:split_fit])) < 2:
        raise ValueError('Initial training has one class. Include earlier attack data, '
                         'or increase --train-fraction while preserving enough test traffic.')
    if len(np.unique(y[split_fit:split_train])) < 2:
        print('WARNING: historical holdout has one class; macro-F1 can be less informative.')
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    x_fit = imputer.fit_transform(X[:split_fit])
    scaler.fit(x_fit)
    X_scaled = scaler.transform(imputer.transform(X)).astype('float32')
    np.clip(X_scaled, -10, 10, out=X_scaled)
    del X, x_fit
    out = Path('data/processed')
    np.savez_compressed(out / 'scaled_data.npz', X=X_scaled, y=y,
                        fit_end=split_fit, train_end=split_train, dev_end=split_dev)
    models = Path('models')
    models.mkdir(exist_ok=True)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'features': names}, models / 'preprocessor.joblib')
    model = MLP(X_scaled.shape[1]).to(DEVICE)
    fit(model, X_scaled[:split_fit], y[:split_fit], epochs=args.epochs)
    torch.save({'state_dict': model.cpu().state_dict(), 'n_features': X_scaled.shape[1],
                'features': names}, models / 'initial_mlp.pt')
    model.to(DEVICE)
    dev_metrics = score(y[split_train:split_dev],
                        probabilities(model, X_scaled[split_train:split_dev]))
    print('Train/holdout/dev/stream sizes:', split_fit, split_train-split_fit,
          split_dev-split_train, n-split_dev)
    print('Initial model development period:', json.dumps(dev_metrics, indent=2))
    (models / 'initial_metrics.json').write_text(json.dumps(dev_metrics, indent=2))
    print('Saved: models/initial_mlp.pt; data/processed/scaled_data.npz')


if __name__ == '__main__':
    main()
