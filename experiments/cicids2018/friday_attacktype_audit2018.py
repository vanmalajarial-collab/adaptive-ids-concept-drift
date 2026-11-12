"""Friday CSE-CIC-IDS2018 per-attack-class, frozen-model audit.

Run from ~/adaptive_ids_project/experiments/cicids2018.
Uses already-inspected Friday labels for descriptive analysis only.
Never trains, refits preprocessing, changes models, or selects a threshold.
"""
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
from model import DEVICE, MLP, load_model, probabilities

CSV = Path('../../data/cicids2018/holdout_friday/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')
ADAPTED_PATH = Path('models/replay_balanced6ep2018.pt')
INITIAL_PATH = Path('models/initial_mlp.pt')
STRIDE = 10


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    for path in (CSV, ADAPTED_PATH, INITIAL_PATH, Path('models/preprocessor.joblib')):
        if not path.is_file():
            raise SystemExit(f'Missing file: {path}')

    prep = joblib.load('models/preprocessor.joblib')
    features = list(prep['features'])
    required = set(features) | {'Timestamp', 'Label'}
    if len(set(features)) != len(features):
        raise ValueError('Duplicate feature names')
    if any(c.lower().strip() in {'label', 'timestamp'} for c in features):
        raise ValueError('Label/timestamp in input features')

    static = load_model()
    checkpoint = torch.load(ADAPTED_PATH, map_location='cpu', weights_only=True)
    if int(checkpoint['n_features']) != len(features):
        raise ValueError('Checkpoint/preprocessor dimension mismatch')
    adapted = MLP(len(features)).to(DEVICE)
    adapted.load_state_dict(checkpoint['state_dict'])
    static.eval()
    adapted.eval()

    label_chunks, static_chunks, adapted_chunks = [], [], []
    offset = 0
    for chunk in pd.read_csv(CSV, chunksize=20000, encoding='latin1', low_memory=False,
                             usecols=lambda name: name.strip() in required):
        chunk.columns = chunk.columns.str.strip()
        if required.difference(chunk.columns):
            raise ValueError('Required columns missing')
        keep = (np.arange(len(chunk)) + offset) % STRIDE == 0
        offset += len(chunk)
        chunk = chunk.iloc[keep]
        if chunk.empty:
            continue
        timestamps = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(),
                                    format='mixed', dayfirst=True, errors='coerce')
        labels = chunk['Label'].astype('string').str.strip()
        valid = timestamps.notna() & labels.notna() & labels.ne('')
        chunk = chunk.loc[valid]
        labels = labels.loc[valid]
        if chunk.empty:
            continue

        raw_x = chunk[features].apply(pd.to_numeric, errors='coerce')
        raw_x = raw_x.replace([np.inf, -np.inf], np.nan)
        x = prep['imputer'].transform(raw_x.to_numpy(dtype=np.float64))
        x = prep['scaler'].transform(x).astype(np.float32)
        np.clip(x, -10, 10, out=x)

        label_chunks.append(labels.to_numpy(dtype=str))
        static_chunks.append(np.asarray(probabilities(static, x)).reshape(-1))
        adapted_chunks.append(np.asarray(probabilities(adapted, x)).reshape(-1))

    if not label_chunks:
        raise SystemExit('No valid sampled flows')
    labels = np.concatenate(label_chunks)
    y = np.char.upper(labels) != 'BENIGN'
    static_probs = np.concatenate(static_chunks)
    adapted_probs = np.concatenate(adapted_chunks)
    prediction_sets = {
        'static_at_0.5': static_probs >= 0.5,
        'static_at_0.001_diagnostic_only': static_probs >= 0.001,
        'adapted_at_0.5': adapted_probs >= 0.5,
    }
    report = {
        'purpose': 'Per-attack-label descriptive audit on ALREADY INSPECTED Friday',
        'warning': '0.001 static threshold was inspected using Friday labels; not a deployable or independently tested threshold',
        'total_rows': int(len(labels)),
        'features': len(features),
        'checkpoint_sha256': {
            'static': file_sha256(INITIAL_PATH),
            'adapted': file_sha256(ADAPTED_PATH),
        },
        'per_label': {},
    }

    for raw_label, n in sorted(Counter(labels).items(), key=lambda v: -v[1]):
        mask = labels == raw_label
        is_benign = raw_label.upper() == 'BENIGN'
        row = {'samples': int(n), 'kind': 'benign' if is_benign else 'attack'}
        for model_key, predictions in prediction_sets.items():
            predicted_attack = int((predictions & mask).sum())
            row[model_key + ('_false_positives' if is_benign else '_true_positives')] = predicted_attack
            row[model_key + ('_fpr' if is_benign else '_recall')] = float(predicted_attack / n)
        report['per_label'][raw_label] = row

    for key, predicted in prediction_sets.items():
        print('\n' + key)
        for label, stats in report['per_label'].items():
            rate_field = key + ('_fpr' if stats['kind'] == 'benign' else '_recall')
            print(f"  {label[:30]:30s} n={stats['samples']:6d} rate={stats[rate_field]:.4f}")
    out = Path('results/friday_attacktype_audit2018.json')
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSaved:', out)


if __name__ == '__main__':
    main()
