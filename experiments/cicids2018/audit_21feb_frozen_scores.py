"""Descriptive audit of already-inspected CICIDS2018 Feb 21 scores.

Run in ~/adaptive_ids_project/experiments/cicids2018 alongside
'evaluate_untouched_day2018.py'. Uses identical row sampling, feature order,
Feb-14-fitted preprocessing and frozen checkpoints.

NO fitting, NO threshold selection. The 0.001 threshold was previously
inspected on Friday and is shown for diagnosis only; Feb 21 is now inspected.
"""
from __future__ import annotations

from collections import defaultdict
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluate_untouched_day2018 as base

OUTPUT = base.Path('results/heldout_21feb2018_score_audit.json')
THRESHOLDS = (0.001, 0.5)  # 0.001: inspected on Friday; diagnostic only
QUANTILES = (0.1, 0.5, 0.9, 0.99)


def main():
    prep = joblib.load(base.PREP)
    features = list(prep['features'])
    required = set(features) | {'Timestamp', 'Label'}

    header = pd.read_csv(base.CSV, nrows=0, encoding='latin1')
    missing_headers = required - set(header.columns.str.strip())
    if missing_headers:
        raise SystemExit(f'Missing CSV columns: {sorted(missing_headers)}')

    models = {name: base.load_frozen_model(path, len(features))
              for name, path in base.CHECKPOINTS.items()}
    probs_parts = {name: [] for name in models}
    label_parts = []
    raw_seen = 0
    invalid_rows = 0
    group_checks = defaultdict(lambda: {'n_features': 0, 'nan': 0, 'clipped': 0})

    from model import probabilities

    for chunk in pd.read_csv(base.CSV, chunksize=20000,
                             encoding='latin1', low_memory=False,
                             usecols=lambda col: col.strip() in required):
        chunk.columns = chunk.columns.str.strip()
        select = (np.arange(len(chunk)) + raw_seen) % base.STRIDE == 0
        raw_seen += len(chunk)
        chunk = chunk.iloc[select]
        if chunk.empty:
            continue

        times = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(),
                               format='mixed', dayfirst=True, errors='coerce')
        labels = chunk['Label'].astype('string').str.strip()
        valid = times.notna() & labels.notna() & labels.ne('')
        invalid_rows += int((~valid).sum())
        labels = labels.loc[valid]
        chunk = chunk.loc[valid]
        if chunk.empty:
            continue

        numeric = chunk[features].apply(pd.to_numeric, errors='coerce')
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        raw_x = numeric.to_numpy(dtype=np.float64)
        x = prep['imputer'].transform(raw_x)
        x = prep['scaler'].transform(x).astype(np.float32)
        label_np = labels.to_numpy(dtype=str)

        for label in np.unique(label_np):
            mask = label_np == label
            gs = group_checks[str(label)]
            gs['n_features'] += int(mask.sum() * len(features))
            gs['nan'] += int(np.isnan(raw_x[mask]).sum())
            gs['clipped'] += int((np.abs(x[mask]) > 10).sum())

        np.clip(x, -10, 10, out=x)
        label_parts.append(label_np)
        for name, model in models.items():
            probs_parts[name].append(np.asarray(probabilities(model, x)).reshape(-1))

    if not label_parts:
        raise SystemExit('No valid rows in Feb 21 dataset')

    labels = np.concatenate(label_parts)
    y = (np.char.upper(labels) != 'BENIGN').astype(np.int8)
    groups = sorted(np.unique(labels))
    report = {
        'warning': 'Feb 21 labels already inspected; this audit is descriptive ONLY. '
                   'Do not use Feb 21 results for threshold/model selection and '
                   'then claim independent test performance.',
        'preselected_thresholds_for_diagnostics': [*THRESHOLDS],
        'threshold_0_001_origin': 'Previously inspected Friday data; not independent',
        'rows': len(y),
        'invalid_sampled_rows': invalid_rows,
        'feature_count': len(features),
        'preprocessing_fingerprint': base.sha256(base.PREP),
        'raw_features_diagnostics': {
            k: {'nan_fraction_before_imputation': v['nan'] / v['n_features'],
                'clipped_fraction_after_scaling': v['clipped'] / v['n_features']}
            for k, v in group_checks.items()
        },
        'models': {},
    }

    print('=== FEB 21 SCORE AUDIT (DESCRIPTIVE ONLY) ===')
    print('Rows:', len(y), 'Labels:', {v: int((labels == v).sum()) for v in groups})

    for name, parts in probs_parts.items():
        p = np.concatenate(parts)
        result = {
            'average_precision_all_attacks': float(average_precision_score(y, p)),
            'per_label': {},
            'hoic_vs_benign_ranking': None,
        }
        print('\nMODEL:', name, 'AP:', f'{result["average_precision_all_attacks"]:.6f}')

        for label in groups:
            mask = labels == label
            q = np.quantile(p[mask], QUANTILES)
            detail = {
                'count': int(mask.sum()),
                'score_quantiles': {str(key): float(val)
                                    for key, val in zip(QUANTILES, q)},
                'at_fixed_thresholds': {},
            }
            for threshold in THRESHOLDS:
                pred_positive = int(np.sum(p[mask] >= threshold))
                detail['at_fixed_thresholds'][str(threshold)] = {
                    'count_flagged': pred_positive,
                    'fraction_flagged': pred_positive / int(mask.sum()),
                }
            result['per_label'][str(label)] = detail
            print(f'  {label}: n={detail["count"]}, '
                  f'quantiles={dict(zip(QUANTILES, np.round(q, 7)))}; '
                  f'flagged@0.001={detail["at_fixed_thresholds"]["0.001"]["count_flagged"]}, '
                  f'flagged@0.5={detail["at_fixed_thresholds"]["0.5"]["count_flagged"]}')

        hoic = labels == 'DDOS attack-HOIC'
        benign = np.char.upper(labels) == 'BENIGN'
        selected = hoic | benign
        if int(hoic.sum()) > 0 and int(benign.sum()) > 0:
            binary = hoic[selected].astype(np.int8)
            ranking = {
                'average_precision': float(average_precision_score(binary, p[selected])),
                'roc_auc': float(roc_auc_score(binary, p[selected])),
            }
            result['hoic_vs_benign_ranking'] = ranking
            print('  HOIC vs benign ranking:', ranking)
        report['models'][name] = result

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSaved:', OUTPUT)
    print('Diagnostic only. No checkpoint/preprocessor was modified.')


if __name__ == '__main__':
    main()
