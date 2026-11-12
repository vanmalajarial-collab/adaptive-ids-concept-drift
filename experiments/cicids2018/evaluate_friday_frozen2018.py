"""Frozen evaluation on an unseen CSE-CIC-IDS2018 Friday CSV.

Run from ~/adaptive_ids_project/experiments/cicids2018.
Does not retrain, refit preprocessing, or overwrite existing checkpoints.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, load_model, probabilities, score

CSV = Path('../../data/cicids2018/holdout_friday/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')
CHECKPOINT = Path('models/replay_balanced6ep2018.pt')
STRIDE = 10

if not CSV.is_file():
    raise SystemExit(f'Missing Friday CSV: {CSV}')
if not CHECKPOINT.is_file():
    raise SystemExit(f'Missing balanced model checkpoint: {CHECKPOINT}')

prep = joblib.load('models/preprocessor.joblib')
features = list(prep['features'])
required = set(features) | {'Timestamp', 'Label'}
header = pd.read_csv(CSV, nrows=0, encoding='latin1')
actual = set(c.strip() for c in header.columns)
missing = required - actual
if missing:
    raise SystemExit('Missing required columns: ' + ', '.join(sorted(missing)))

static = load_model()
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if ckpt['n_features'] != len(features):
    raise ValueError('Feature count mismatch between balanced checkpoint and fitted preprocessing')
balanced = MLP(len(features)).to(DEVICE)
balanced.load_state_dict(ckpt['state_dict'])

all_y, static_probs, balanced_probs = [], [], []
processed = 0
sampled = 0
for chunk in pd.read_csv(CSV, chunksize=20000, encoding='latin1', low_memory=False,
                         usecols=lambda col: col.strip() in required):
    chunk.columns = chunk.columns.str.strip()
    keep = (np.arange(len(chunk)) + processed) % STRIDE == 0
    processed += len(chunk)
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
    y = (labels.str.upper() != 'BENIGN').to_numpy(dtype=np.int8)
    raw_x = chunk[features].apply(pd.to_numeric, errors='coerce')
    raw_x = raw_x.replace([np.inf, -np.inf], np.nan)
    x = prep['imputer'].transform(raw_x.to_numpy(dtype=np.float64))
    x = prep['scaler'].transform(x).astype(np.float32)
    np.clip(x, -10, 10, out=x)

    all_y.append(y)
    static_probs.append(probabilities(static, x))
    balanced_probs.append(probabilities(balanced, x))
    sampled += len(y)
    if processed % 200000 == 0:
        print(f'Read {processed:,} Friday rows; evaluated {sampled:,} samples', flush=True)

if not all_y:
    raise SystemExit('No valid samples in Friday CSV')
y = np.concatenate(all_y)
static_metrics = score(y, np.concatenate(static_probs))
balanced_metrics = score(y, np.concatenate(balanced_probs))
results = {
    'dataset': CSV.name,
    'training_policy': 'Frozen models; no Friday learning or threshold tuning',
    'raw_rows_seen': processed,
    'sample_stride': STRIDE,
    'evaluation_rows': len(y),
    'attack_rows': int(y.sum()),
    'benign_rows': int(len(y) - y.sum()),
    'static_wednesday_model': static_metrics,
    'balanced_guarded_thursday_adapted_model': balanced_metrics,
}
Path('results').mkdir(exist_ok=True)
out = Path('results/friday_frozen_2018_summary.json')
out.write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
print('Saved:', out)
