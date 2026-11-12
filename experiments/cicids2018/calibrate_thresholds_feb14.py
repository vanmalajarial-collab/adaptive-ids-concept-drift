"""Select per-checkpoint operating thresholds on Feb 14 development traffic ONLY.

Run from ~/adaptive_ids_project/experiments/cicids2018.
No Friday/Feb 21/Feb 22 data are loaded. No model is trained/refitted.
The target development benign FPR is predeclared at 0.001 (0.1%).
Threshold selection uses only benign development scores; attack results are
reported for context and do not affect threshold selection.
This is an exploratory protocol motivated by prior inspected test days;
use an entirely new test day for future out-of-sample claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, probabilities

DATA = Path('data/processed/scaled_data.npz')
OUT = Path('results/feb14_dev_calibrated_thresholds.json')
CHECKPOINTS = {
    'static_wednesday': Path('models/initial_mlp.pt'),
    'guarded_balanced_6ep': Path('models/replay_balanced6ep2018.pt'),
    'hybrid_AND': Path('models/hybrid_ablation_and_2018.pt'),
    'hybrid_ADWIN': Path('models/hybrid_ablation_adwin_2018.pt'),
    'hybrid_MMD': Path('models/hybrid_ablation_mmd_2018.pt'),
    'hybrid_OR': Path('models/hybrid_ablation_or_2018.pt'),
}
TARGET_DEV_FPR = 0.001  # fixed in advance, not selected on Feb 21


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def load_frozen(path: Path, features: int):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if int(checkpoint['n_features']) != features:
        raise ValueError(f'Feature mismatch: {path}')
    model = MLP(features).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def select_threshold_from_benign(benign_probs: np.ndarray, allowed_fpr: float):
    """Smallest strict boundary allowing <= floor(allowed_fpr*n) training FP.

    Tie-safe: bump the boundary infinitesimally above the score at rank.
    This is an empirical development-set guarantee only, not a future guarantee.
    """
    s = np.sort(np.asarray(benign_probs, dtype=np.float64))
    if len(s) == 0 or not np.all(np.isfinite(s)):
        raise ValueError('Benign development scores are missing or nonfinite')
    budget = int(math.floor(len(s) * allowed_fpr))
    pivot = s[len(s) - budget - 1]
    return float(np.nextafter(pivot, np.inf)), budget


def evaluate(y, p, threshold):
    pred = p >= threshold
    benign = y == 0
    attack = y == 1
    tn = int(np.sum(benign & ~pred))
    fp = int(np.sum(benign & pred))
    fn = int(np.sum(attack & ~pred))
    tp = int(np.sum(attack & pred))
    return {
        'threshold': threshold,
        'recall': float(recall_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'attack_f1': float(f1_score(y, pred, zero_division=0)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'fpr': float(fp / (fp + tn)),
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
    }


def main():
    if not DATA.is_file():
        raise SystemExit(f'Missing {DATA}')
    for path in CHECKPOINTS.values():
        if not path.is_file():
            raise SystemExit(f'Missing checkpoint: {path}')

    with np.load(DATA) as raw:
        train_end = int(raw['train_end'])
        dev_end = int(raw['dev_end'])
        X = raw['X'][train_end:dev_end]
        y = raw['y'][train_end:dev_end].astype(np.int8)
    if not len(X) or np.unique(y).tolist() != [0, 1]:
        raise SystemExit('Feb 14 development slice must contain benign and attack flows')

    result = {
        'protocol': 'Frozen model operating thresholds selected on Feb-14 development only; '
                    'no Friday or Feb-21 scores/labels used',
        'selection_method': 'Allow at most floor(0.001 * dev_benign_count) '
                            'benign development false positives',
        'target_development_fpr': TARGET_DEV_FPR,
        'note': 'Development FPR is not a guarantee for future traffic. '
                'This policy was conceived after inspecting Friday and Feb-21; '
                'only a later untouched day can serve as a new test.',
        'dataset_sha256': sha256(DATA),
        'dev_index_range': [train_end, dev_end],
        'dev_rows': int(len(y)),
        'dev_benign': int(np.sum(y == 0)),
        'dev_attacks': int(np.sum(y == 1)),
        'models': {},
    }
    print('FEB 14 DEVELOPMENT-ONLY THRESHOLD SELECTION')
    print(f'Rows: {len(y)}; benign: {(y == 0).sum()}; attacks: {(y == 1).sum()}')
    print(f'{"Model":<24} {"Threshold":>12} {"Dev recall":>11} {"Dev FPR":>10} {"FP":>6}')

    for name, checkpoint in CHECKPOINTS.items():
        model = load_frozen(checkpoint, X.shape[1])
        probs = np.asarray(probabilities(model, X), dtype=np.float64).reshape(-1)
        if len(probs) != len(y) or not np.all(np.isfinite(probs)):
            raise ValueError(f'Invalid scores from {checkpoint}')
        threshold, fp_budget = select_threshold_from_benign(probs[y == 0], TARGET_DEV_FPR)
        metrics = evaluate(y, probs, threshold)
        baseline = evaluate(y, probs, 0.5)
        if metrics['fp'] > fp_budget:
            raise AssertionError('Development FPR budget exceeded')
        result['models'][name] = {
            'checkpoint_sha256': sha256(checkpoint),
            'selected_threshold': threshold,
            'dev_fp_budget': fp_budget,
            'development_at_selected_threshold': metrics,
            'development_at_0_5': baseline,
        }
        print(f'{name:<24} {threshold:12.6g} {metrics["recall"]:11.4f} '
              f'{metrics["fpr"]:10.5f} {metrics["fp"]:6d}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Saved:', OUT)
    print('No model was trained and no threshold was chosen on test days.')


if __name__ == '__main__':
    main()
