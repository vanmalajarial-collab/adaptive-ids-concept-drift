"""Frozen, single-use held-out day evaluation for the CICIDS2018 trigger ablation.

Run in ~/adaptive_ids_project/experiments/cicids2018.
Prerequisite CSV (not in raw training folder):
  ../../data/cicids2018/holdout_21feb/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv

Safety/protocol:
* Loads preprocessor fit on Feb 14 only; does NOT refit or train.
* Every checkpoint is frozen; threshold fixed at 0.5 from earlier experiments.
* A fixed 1-in-10 row sample is taken without using labels for selection.
* Writes only a new result JSON, never checkpoints.
* Once labels have been inspected, do NOT tune on this day and call it independent.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, probabilities, score

CSV = Path('../../data/cicids2018/holdout_21feb/Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv')
PREP = Path('models/preprocessor.joblib')
STRIDE = 10
OUTPUT = Path('results/heldout_21feb2018_frozen_summary.json')
CHECKPOINTS = {
    'static_wednesday': Path('models/initial_mlp.pt'),
    'guarded_balanced_6ep': Path('models/replay_balanced6ep2018.pt'),
    'hybrid_AND': Path('models/hybrid_ablation_and_2018.pt'),
    'hybrid_ADWIN': Path('models/hybrid_ablation_adwin_2018.pt'),
    'hybrid_MMD': Path('models/hybrid_ablation_mmd_2018.pt'),
    'hybrid_OR': Path('models/hybrid_ablation_or_2018.pt'),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_frozen_model(path, n_features):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if int(checkpoint['n_features']) != n_features:
        raise ValueError(f'Feature mismatch for {path}: '
                         f"{checkpoint['n_features']} vs {n_features}")
    model = MLP(n_features).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def main():
    for path in (CSV, PREP, *CHECKPOINTS.values()):
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')

    prep = joblib.load(PREP)
    features = list(prep['features'])
    if len(features) != len(set(features)):
        raise ValueError('Duplicate feature names in preprocessing')
    if {'timestamp', 'label'} & {str(c).lower() for c in features}:
        raise ValueError('Leaky feature name in preprocessing')
    required = set(features) | {'Timestamp', 'Label'}
    header = pd.read_csv(CSV, nrows=0, encoding='latin1')
    actual = set(header.columns.str.strip())
    absent = sorted(required - actual)
    if absent:
        raise SystemExit('Missing columns in heldout CSV: ' + ', '.join(absent))

    models = {name: load_frozen_model(path, len(features))
              for name, path in CHECKPOINTS.items()}
    labels_buffer = []
    probabilities_buffer = {name: [] for name in models}
    raw_seen, invalid_rows = 0, 0

    for chunk in pd.read_csv(
        CSV, chunksize=20000, encoding='latin1', low_memory=False,
        usecols=lambda column: column.strip() in required,
    ):
        chunk.columns = chunk.columns.str.strip()
        # Deterministic downsampling, independent of labels and timestamps.
        keep = (np.arange(len(chunk)) + raw_seen) % STRIDE == 0
        raw_seen += len(chunk)
        chunk = chunk.iloc[keep]
        if chunk.empty:
            continue
        times = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(),
                               format='mixed', dayfirst=True, errors='coerce')
        labels = chunk['Label'].astype('string').str.strip()
        valid = times.notna() & labels.notna() & labels.ne('')
        invalid_rows += int((~valid).sum())
        chunk = chunk.loc[valid]
        labels = labels.loc[valid]
        if chunk.empty:
            continue
        numeric = chunk[features].apply(pd.to_numeric, errors='coerce')
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        X = prep['imputer'].transform(numeric.to_numpy(dtype=np.float64))
        X = prep['scaler'].transform(X).astype(np.float32)
        np.clip(X, -10, 10, out=X)
        labels_buffer.append(labels.to_numpy(dtype=str))
        for name, model in models.items():
            probabilities_buffer[name].append(
                np.asarray(probabilities(model, X)).reshape(-1))

    if not labels_buffer:
        raise SystemExit('No valid sampled flows in held-out day')
    labels = np.concatenate(labels_buffer)
    y = (np.char.upper(labels) != 'BENIGN').astype(np.int8)

    report = {
        'dataset': CSV.name,
        'protocol': 'Frozen checkpoints and Feb-14-fitted preprocessing; '
                    'no heldout training/refit/threshold selection; fixed threshold 0.5',
        'caution': 'This day is holdout only if previously unseen. After viewing '
                   'these results, do not tune on this day and call it independent.',
        'raw_rows_seen': raw_seen,
        'sample_stride': STRIDE,
        'invalid_sampled_rows': invalid_rows,
        'evaluated_rows': int(len(y)),
        'attack_rows': int(y.sum()),
        'benign_rows': int((y == 0).sum()),
        'label_counts': dict(sorted(Counter(labels).items())),
        'n_features': len(features),
        'preprocessor_sha256': sha256(PREP),
        'checkpoint_sha256': {name: sha256(path) for name, path in CHECKPOINTS.items()},
        'models': {},
    }

    print('\n=== HELDOUT DAY: FROZEN MODEL COMPARISON ===')
    print('Rows:', len(y), '| attacks:', int(y.sum()), '| benign:', int((y == 0).sum()))
    print(f"{'Model':<24} {'Recall':>8} {'Macro-F1':>9} {'FPR':>9} {'AP':>9}")
    for name, parts in probabilities_buffer.items():
        probs = np.concatenate(parts)
        metrics = score(y, probs)
        per_label = {}
        flagged = probs >= 0.5
        for label, count in sorted(Counter(labels).items()):
            mask = labels == label
            flagged_n = int((flagged & mask).sum())
            per_label[label] = {
                'samples': int(count),
                'kind': 'benign' if label.upper() == 'BENIGN' else 'attack',
                'predicted_attack': flagged_n,
                'recall_or_fpr': float(flagged_n / count),
            }
        report['models'][name] = {'metrics': metrics, 'per_label': per_label}
        print(f"{name:<24} {metrics['recall']:8.4f} {metrics['macro_f1']:9.4f} "
              f"{metrics['attack_fpr'] if metrics['attack_fpr'] is not None else float('nan'):9.4f} "
              f"{metrics['pr_auc'] if metrics['pr_auc'] is not None else float('nan'):9.4f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSaved:', OUTPUT)
    print('Per-attack labels available in the JSON; checkpoints were not modified.')


if __name__ == '__main__':
    main()
