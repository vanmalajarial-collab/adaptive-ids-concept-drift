"""Audit frozen CICIDS2018 Friday predictions (NO training or parameter changes).

Run from ~/adaptive_ids_project/experiments/cicids2018
The printed Friday threshold sweep is DIAGNOSTIC ONLY: do not use Friday
labels to select a deployable threshold or claim an independent tuned test.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, load_model, probabilities

CSV = Path('../../data/cicids2018/holdout_friday/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv')
CHECKPOINT = Path('models/replay_balanced6ep2018.pt')
STRIDE = 10

if not CSV.exists() or not CHECKPOINT.exists():
    raise SystemExit('Missing Friday CSV or balanced checkpoint. Check working directory.')

prep = joblib.load('models/preprocessor.joblib')
features = list(prep['features'])
if len(features) != len(set(features)):
    raise ValueError('Duplicate input feature names')
if any(name.strip().lower() in ('label', 'timestamp') for name in features):
    raise ValueError('Target or timestamp found in model feature list!')

initial = load_model()
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if int(ckpt['n_features']) != len(features):
    raise ValueError('Checkpoint feature count does not match the preprocessor')
adapted = MLP(len(features)).to(DEVICE)
adapted.load_state_dict(ckpt['state_dict'])
initial.eval()
adapted.eval()

labels_all, initial_all, adapted_all = [], [], []
row_offset = 0
for chunk in pd.read_csv(CSV, chunksize=20_000, encoding='latin1', low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    # Parenthesize explicitly: both features and required columns must exist.
    missing = (set(features) | {'Timestamp', 'Label'}) - set(chunk.columns)
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    sampled = ((np.arange(len(chunk)) + row_offset) % STRIDE == 0)
    row_offset += len(chunk)
    chunk = chunk.iloc[sampled].copy()
    if chunk.empty:
        continue
    timestamps = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(), format='mixed', dayfirst=True, errors='coerce')
    labels = chunk['Label'].astype('string').str.strip()
    valid = timestamps.notna() & labels.notna() & labels.ne('')
    chunk = chunk.loc[valid]
    labels = labels.loc[valid]
    if chunk.empty:
        continue
    yy = (labels.str.upper() != 'BENIGN').to_numpy(dtype=np.int8)
    features_numeric = chunk[features].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    xx = prep['imputer'].transform(features_numeric.to_numpy(dtype=np.float64))
    xx = prep['scaler'].transform(xx).astype(np.float32)
    np.clip(xx, -10, 10, out=xx)
    labels_all.append(yy)
    initial_all.append(np.asarray(probabilities(initial, xx)).reshape(-1))
    adapted_all.append(np.asarray(probabilities(adapted, xx)).reshape(-1))

if not labels_all:
    raise SystemExit('No valid sampled rows')
y = np.concatenate(labels_all)
preds = {'static': np.concatenate(initial_all), 'thursday_adapted': np.concatenate(adapted_all)}

results = {'friday_test': 'already inspected; diagnostic only',
           'sampled_rows': int(len(y)), 'feature_count': len(features),
           'feature_label_leakage_by_name': False, 'models': {}}

for name, probs in preds.items():
    metrics = {'average_precision': float(average_precision_score(y, probs)),
               'roc_auc': float(roc_auc_score(y, probs)),
               'probability_quantiles': {}, 'fixed_threshold_diagnostics': []}
    print(f'\n=== {name} ===')
    print(f'Average precision: {metrics["average_precision"]:.6f}, ROC-AUC: {metrics["roc_auc"]:.6f}')
    for label_id, label_name in [(0, 'benign'), (1, 'attack')]:
        vals = probs[y == label_id]
        quantiles = {str(q): float(np.quantile(vals, q)) for q in (0.1, 0.5, 0.9, 0.99)}
        metrics['probability_quantiles'][label_name] = quantiles
        print(f'{label_name} score quantiles: {quantiles}')
    print('Threshold | Recall | FPR | TP | FP')
    for threshold in (0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5):
        pred = probs >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = int(np.sum(~pred & (y == 0)))
        record = {'threshold': threshold, 'recall': tp / (tp + fn),
                  'fpr': fp / (fp + tn), 'tp': tp, 'fp': fp}
        metrics['fixed_threshold_diagnostics'].append(record)
        print(f'{threshold:>9g} | {record["recall"]:.4f} | {record["fpr"]:.4f} | {tp} | {fp}')
    results['models'][name] = metrics

Path('results').mkdir(exist_ok=True)
Path('results/friday_probability_audit.json').write_text(json.dumps(results, indent=2))
print('\nSaved results/friday_probability_audit.json')
print('Do not select a final threshold using these already-viewed Friday labels.')
