"""Predeclared, frozen threshold comparison on CSE-CIC-IDS2018 February 22.

Run in ~/adaptive_ids_project/experiments/cicids2018.
Use the full CSV (stride=1): this capture day has very few attack flows.

Reads saved 14-Feb development-selected operating thresholds for six frozen checkpoints.
Reports both thresholds (0.5 and development-selected) without changing either.
Never trains/refits, and never uses February 22 labels to choose thresholds.

WARNING: Development threshold protocol was designed after examining other days.
Do not use February 22 outcomes to adjust thresholds and continue calling it held out.
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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, probabilities

CSV = Path('../../data/cicids2018/holdout_22feb/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv')
PREP = Path('models/preprocessor.joblib')
CALIB = Path('results/feb14_dev_calibrated_thresholds.json')
OUT = Path('results/heldout_22feb_calibrated2018_summary.json')
CHECKPOINTS = {
    'static_wednesday': Path('models/initial_mlp.pt'),
    'guarded_balanced_6ep': Path('models/replay_balanced6ep2018.pt'),
    'hybrid_AND': Path('models/hybrid_ablation_and_2018.pt'),
    'hybrid_ADWIN': Path('models/hybrid_ablation_adwin_2018.pt'),
    'hybrid_MMD': Path('models/hybrid_ablation_mmd_2018.pt'),
    'hybrid_OR': Path('models/hybrid_ablation_or_2018.pt'),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: Path, n_features: int):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if int(ckpt['n_features']) != n_features:
        raise ValueError(f'Checkpoint feature count mismatch: {path}')
    model = MLP(n_features).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def metrics(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    predictions = probs >= threshold
    benign = y == 0
    attack = y == 1
    tn = int((benign & ~predictions).sum())
    fp = int((benign & predictions).sum())
    fn = int((attack & ~predictions).sum())
    tp = int((attack & predictions).sum())
    return {
        'threshold': float(threshold),
        'accuracy': float((predictions == y).mean()),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'attack_f1': float(f1_score(y, predictions, zero_division=0)),
        'macro_f1': float(f1_score(y, predictions, average='macro', zero_division=0)),
        'fpr': float(fp / (fp + tn)) if fp + tn else None,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
    }


def main():
    required_files = [CSV, PREP, CALIB, *CHECKPOINTS.values()]
    for path in required_files:
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')

    calibration = json.loads(CALIB.read_text(encoding='utf-8'))
    if set(calibration['models']) != set(CHECKPOINTS):
        raise SystemExit('Calibration and checkpoint model names do not match')
    if float(calibration['target_development_fpr']) != 0.001:
        raise SystemExit('Unexpected development FPR policy')
    thresholds = {}
    for name, path in CHECKPOINTS.items():
        actual = digest(path)
        expected = calibration['models'][name]['checkpoint_sha256']
        if actual != expected:
            raise SystemExit(f'Frozen checkpoint changed after calibration: {name}')
        threshold = float(calibration['models'][name]['selected_threshold'])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise SystemExit(f'Invalid calibration threshold: {name}')
        thresholds[name] = threshold

    prep = joblib.load(PREP)
    features = list(prep['features'])
    if not features or len(features) != len(set(features)):
        raise SystemExit('Missing or duplicate preprocessing features')
    if {'label', 'timestamp'} & {str(c).strip().lower() for c in features}:
        raise SystemExit('Label/timestamp leaked into preprocessing features')
    required_cols = set(features) | {'Timestamp', 'Label'}
    header = pd.read_csv(CSV, nrows=0, encoding='latin1')
    missing = sorted(required_cols - set(header.columns.str.strip()))
    if missing:
        raise SystemExit('CSV missing training features: ' + ', '.join(missing))

    models = {name: load_checkpoint(path, len(features))
              for name, path in CHECKPOINTS.items()}
    all_labels = []
    all_probs = {name: [] for name in models}
    raw_seen, invalid_rows = 0, 0

    print('=== FEB 22 FULL-ROW FROZEN EVALUATION ===', flush=True)
    print('Thresholds loaded from Feb 14; no refit, training, or threshold search.', flush=True)
    for chunk in pd.read_csv(
        CSV, chunksize=20000, encoding='latin1', low_memory=False,
        usecols=lambda col: col.strip() in required_cols,
    ):
        chunk.columns = chunk.columns.str.strip()
        raw_seen += len(chunk)
        timestamps = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(),
                                    format='mixed', dayfirst=True, errors='coerce')
        labels = chunk['Label'].astype('string').str.strip()
        valid = timestamps.notna() & labels.notna() & labels.ne('')
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
        all_labels.append(labels.to_numpy(dtype=str))
        for name, model in models.items():
            p = np.asarray(probabilities(model, X), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(p)):
                raise SystemExit(f'Nonfinite scores: {name}')
            all_probs[name].append(p)
        if raw_seen // 200000 > (raw_seen - len(chunk)) // 200000:
            print(f'Read {raw_seen:,} rows', flush=True)

    if not all_labels:
        raise SystemExit('No valid rows loaded')
    labels = np.concatenate(all_labels)
    y = (np.char.upper(labels) != 'BENIGN').astype(np.int8)
    counts = dict(sorted(Counter(labels).items()))
    result = {
        'dataset': CSV.name,
        'protocol': 'Frozen models and preprocessing; thresholds fixed on Feb-14 development; no Feb-22 training/refit/selection',
        'caution': 'Selection protocol designed after earlier days inspected; Feb-22 must not be reused for tuning. Sparse attacks make estimates unstable.',
        'raw_rows_seen': raw_seen,
        'rows_evaluated': int(len(y)),
        'invalid_rows': invalid_rows,
        'sample_stride': 1,
        'attack_rows': int(y.sum()),
        'benign_rows': int((y == 0).sum()),
        'label_counts': counts,
        'n_features': len(features),
        'preprocessor_sha256': digest(PREP),
        'calibration_json_sha256': digest(CALIB),
        'checkpoint_sha256': {name: digest(path) for name, path in CHECKPOINTS.items()},
        'models': {},
    }

    print(f'Rows {len(y):,} | attacks {int(y.sum()):,} | benign {int((y == 0).sum()):,}')
    print(f"{'Model':<22} {'Thr':>9} {'Recall@0.5':>11} {'Recall@cal':>11} {'FPR@cal':>10} {'TP@cal':>8} {'FP@cal':>8}")
    for name, parts in all_probs.items():
        probs = np.concatenate(parts)
        baseline = metrics(y, probs, 0.5)
        calibrated = metrics(y, probs, thresholds[name])
        per_label = {}
        for label, n in counts.items():
            mask = labels == label
            per_label[label] = {
                'samples': int(n),
                'kind': 'benign' if label.upper() == 'BENIGN' else 'attack',
                'flagged_at_0_5': int(((probs >= 0.5) & mask).sum()),
                'flagged_at_dev_threshold': int(((probs >= thresholds[name]) & mask).sum()),
            }
        result['models'][name] = {
            'development_selected_threshold': thresholds[name],
            'at_0_5': baseline,
            'at_development_threshold': calibrated,
            'average_precision': (float(average_precision_score(y, probs))
                                  if len(np.unique(y)) == 2 else None),
            'per_label': per_label,
        }
        print(f'{name:<22} {thresholds[name]:9.4f} {baseline["recall"]:11.4f} '
              f'{calibrated["recall"]:11.4f} {calibrated["fpr"]:10.4f} '
              f'{calibrated["tp"]:8d} {calibrated["fp"]:8d}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', OUT)
    print('Check per_label and both fixed operating points in the JSON.')


if __name__ == '__main__':
    main()
