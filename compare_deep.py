"""Optional Step 13: CNN/LSTM/GRU temporal sequence baselines.

Each non-overlapping length-L window of consecutive flows predicts the label
of its LAST flow. Evaluate MLP on the same end-of-window flows for fairness.
WARNING: consecutive flows may be from different hosts; this is a sequence
baseline, not proof of host-specific temporal semantics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import DEVICE, probabilities, score
from model import load_model


class SequenceClassifier(torch.nn.Module):
    def __init__(self, n_features, name):
        super().__init__()
        self.name = name
        if name == 'CNN':
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )
        elif name == 'LSTM':
            self.encoder = torch.nn.LSTM(n_features, 32, batch_first=True)
        elif name == 'GRU':
            self.encoder = torch.nn.GRU(n_features, 32, batch_first=True)
        self.head = torch.nn.Linear(32, 1)

    def forward(self, x):
        if self.name == 'CNN':
            feat = self.encoder(x.transpose(1, 2)).squeeze(-1)
        else:
            output, _ = self.encoder(x)
            feat = output[:, -1, :]
        return self.head(feat).flatten()


def sequences(X, y, length):
    end = np.arange(length - 1, len(X), length)
    windows = np.stack([X[i-length+1:i+1] for i in end])
    return windows, y[end], X[end]


def train_seq(model, X, y, epochs):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    y = y.astype('float32')
    for epoch in range(epochs):
        rng = np.random.default_rng(42 + epoch)
        order = rng.permutation(len(y))
        for start in range(0, len(y), 256):
            ids = order[start:start+256]
            xb = torch.tensor(X[ids], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(y[ids], dtype=torch.float32, device=DEVICE)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def get_probs(model, X):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(X), 256):
            xx = torch.tensor(X[start:start+256], device=DEVICE)
            output.extend(torch.sigmoid(model(xx)).cpu().numpy().tolist())
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    if args.length < 2:
        parser.error('--length must be >= 2')
    torch.manual_seed(42)
    arr = np.load('data/processed/scaled_data.npz')
    X, y = arr['X'], arr['y'].astype(int)
    fit_end, dev_end = int(arr['fit_end']), int(arr['dev_end'])
    train_x, train_y, _ = sequences(X[:fit_end], y[:fit_end], args.length)
    test_x, test_y, test_end_flows = sequences(X[dev_end:], y[dev_end:], args.length)
    output = {'MLP_same_endpoints': score(test_y, probabilities(load_model(), test_end_flows))}
    for name in ('CNN', 'LSTM', 'GRU'):
        model = SequenceClassifier(X.shape[1], name)
        train_seq(model, train_x, train_y, args.epochs)
        output[name] = score(test_y, get_probs(model, test_x))
        print(name, 'macro-F1 =', round(output[name]['macro_f1'], 4))
    Path('results').mkdir(exist_ok=True)
    Path('results/deep_sequence_comparison.json').write_text(json.dumps(output, indent=2))
    print('Saved results/deep_sequence_comparison.json')


if __name__ == '__main__':
    main()
