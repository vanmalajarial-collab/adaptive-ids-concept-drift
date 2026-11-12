"""Phase 5 exploratory CICIDS2018 Feb-23 conservative replay ablation.

Run in ~/adaptive_ids_project/experiments/cicids2018:
  python phase5_accumulated_replay2018.py

Protocol: use ALL Feb-23 rows (attacks are very rare), sort chronologically
when needed; predict each batch before its labels become available; collect
verified missed attacks across multiple batches; train balanced replay +
knowledge-distillation candidate; validate candidate on STRICTLY LATER labeled
batches before optional model replacement. The source checkpoint and original
preprocessor remain unchanged. This is a supervised feedback baseline, NOT a
full MMD/ADWIN policy and NOT an independent final test.

The candidate validation uses a previously reused historical guard set. Feb-23
is intended for development only. Results depend on next-batch verified labels.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model import DEVICE, MLP, clone, probabilities, score
from replay import BalancedReservoir

CSV = Path('../../data/cicids2018/phase5_development/Friday-23-02-2018_TrafficForML_CICFlowMeter.csv')
PREP = Path('models/preprocessor.joblib')
SOURCE_MODEL = Path('models/hybrid_ablation_or_2018.pt')
PROCESSED = Path('data/processed/scaled_data.npz')
OUT_STEM = 'phase5_feb23_conservative_replay2018'

BATCH_SIZE = 256
MIN_CUMULATIVE_MISSES = 24       # verified missed attacks, not current-batch only
MIN_NEW_ATTACK_EXAMPLES = 24     # require truly new attack examples for candidate
COOLDOWN_BATCHES = 8
RECENT_BATCHES = 8
NEW_ATTACK_POOL_CAPACITY = 512
REPLAY_SAMPLE_SIZE = 256
NEGATIVE_TRAIN_SAMPLES = 576
POSITIVE_TRAIN_SAMPLES = 192
FIXED_POS_WEIGHT = 1.0     # no automatic reweighting that undoes 3:1 sampling
CANDIDATE_EPOCHS = 6
CANDIDATE_LR = 0.0003
DISTILL_WEIGHT = 0.20
MIN_VALIDATION_ATTACKS = 8
MIN_VALIDATION_BENIGN = 512
MAX_VALIDATION_BATCHES = 512
MIN_VALIDATION_RECALL_GAIN = 0.03
MAX_VALIDATION_FPR_INCREASE = 0.0005
MAX_HISTORICAL_FPR_INCREASE = 0.002
MAX_HISTORICAL_MACRO_F1_DROP = 0.02


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
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


def load_feb23():
    prep = joblib.load(PREP)
    features = list(prep['features'])
    if len(features) != 78 or len(set(features)) != len(features):
        raise ValueError('Unexpected feature mapping. Stop and inspect preprocessing.')
    if {'label', 'timestamp'} & {str(c).strip().lower() for c in features}:
        raise ValueError('Leakage: Label/Timestamp included in feature list.')
    needed = set(features) | {'Label', 'Timestamp'}
    header = pd.read_csv(CSV, nrows=0, encoding='latin1')
    missing = needed - set(header.columns.str.strip())
    if missing:
        raise ValueError(f'Missing feature columns: {sorted(missing)}')

    chunks_x, chunks_y, chunks_t, chunks_label = [], [], [], []
    seen = invalid = 0
    print('Reading full Feb-23 data, reusing Feb-14-fitted preprocessing.', flush=True)
    for chunk in pd.read_csv(CSV, chunksize=20000, encoding='latin1',
                             low_memory=False,
                             usecols=lambda col: col.strip() in needed):
        chunk.columns = chunk.columns.str.strip()
        seen += len(chunk)
        times = pd.to_datetime(chunk['Timestamp'].astype(str).str.strip(),
                               format='mixed', dayfirst=True, errors='coerce')
        labels = chunk['Label'].astype('string').str.strip()
        valid = times.notna() & labels.notna() & labels.ne('')
        invalid += int((~valid).sum())
        chunk = chunk.loc[valid]
        if chunk.empty:
            continue
        numeric = chunk[features].apply(pd.to_numeric, errors='coerce')
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        x = prep['imputer'].transform(numeric.to_numpy(dtype=np.float64))
        x = prep['scaler'].transform(x).astype(np.float32)
        np.clip(x, -10, 10, out=x)
        str_labels = labels.loc[valid].to_numpy(dtype=str)
        y = (np.char.upper(str_labels) != 'BENIGN').astype(np.int8)
        chunks_x.append(x)
        chunks_y.append(y)
        chunks_t.append(times.loc[valid].to_numpy(dtype='datetime64[ns]'))
        chunks_label.append(str_labels)
        if seen % 200000 < 20000:
            print(f'Read {seen:,} rows', flush=True)

    if not chunks_x:
        raise ValueError('No valid rows found')
    X = np.concatenate(chunks_x)
    y = np.concatenate(chunks_y)
    ts = np.concatenate(chunks_t)
    labels = np.concatenate(chunks_label)
    was_sorted = bool(np.all(ts[1:] >= ts[:-1]))
    if not was_sorted:
        idx = np.argsort(ts, kind='stable')
        X, y, ts, labels = X[idx], y[idx], ts[idx], labels[idx]
        print('Input CSV was not chronological; sorted by Timestamp.', flush=True)
    print('Rows', len(y), '| attacks', int(y.sum()), '| invalid', invalid,
          '| input chronological:', was_sorted, flush=True)
    return X, y, labels, seen, invalid, was_sorted


def fit_conservative(model, X, y, *, epochs, lr, teacher, distill_weight, batch_size):
    """Original training objective with fixed positive-class weight of 1.0.

    IMPORTANT: model.fit automatically computes pos_weight=benign/attack.
    That would undo the intended conservative 3:1 sampling ratio, so use
    this local trainer rather than silently falling back to model.fit.
    """
    model = model.to(DEVICE)
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(FIXED_POS_WEIGHT, device=DEVICE))
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = torch.as_tensor(X[idx], dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            supervised = criterion(logits, yb)
            temperature = 2.0
            with torch.no_grad():
                soft_target = torch.sigmoid(teacher(xb) / temperature)
            distill = torch.nn.functional.binary_cross_entropy_with_logits(
                logits / temperature, soft_target) * temperature ** 2
            loss = (1 - distill_weight) * supervised + distill_weight * distill
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def build_candidate(active, recent_batches, new_attack_pool,
                    replay, attempt_batch):
    if len(new_attack_pool) < MIN_NEW_ATTACK_EXAMPLES:
        return None, 'not_enough_new_attack_examples'
    recent_x = np.concatenate([x for x, _ in recent_batches])
    recent_y = np.concatenate([y for _, y in recent_batches])
    new_x = np.stack(new_attack_pool).astype(np.float32)
    new_y = np.ones(len(new_x), dtype=np.int8)
    hist_x, hist_y = replay.sample(REPLAY_SAMPLE_SIZE)
    parts_x = [new_x, recent_x]
    parts_y = [new_y, recent_y]
    parts_weight = [np.full(len(new_x), 8., dtype=np.float64),
                    np.full(len(recent_y), 3., dtype=np.float64)]
    if hist_x is not None:
        parts_x.append(hist_x)
        parts_y.append(hist_y)
        parts_weight.append(np.ones(len(hist_y), dtype=np.float64))
    sx = np.concatenate(parts_x)
    sy = np.concatenate(parts_y)
    weights = np.concatenate(parts_weight)
    rng = np.random.default_rng(7000 + attempt_batch)
    chosen = []
    for c in (0, 1):
        ids = np.flatnonzero(sy == c)
        if len(ids) == 0:
            return None, f'no_class_{c}_training_data'
        p = weights[ids] / weights[ids].sum()
        chosen.append(rng.choice(ids, size=(NEGATIVE_TRAIN_SAMPLES if c == 0
                                            else POSITIVE_TRAIN_SAMPLES),
                                 replace=True, p=p))
    selected = np.concatenate(chosen)
    rng.shuffle(selected)
    teacher = clone(active).eval()
    candidate = clone(active)
    fit_conservative(candidate, sx[selected], sy[selected], epochs=CANDIDATE_EPOCHS,
        lr=CANDIDATE_LR, teacher=teacher,
        distill_weight=DISTILL_WEIGHT, batch_size=256)
    return candidate, 'trained'


def main():
    for path in (CSV, PREP, SOURCE_MODEL, PROCESSED):
        if not path.is_file():
            raise SystemExit(f'Missing required file: {path}')
    torch.manual_seed(42)
    np.random.seed(42)

    original = np.load(PROCESSED)
    Xold, yold = original['X'], original['y'].astype(np.int8)
    fit_end, train_end = int(original['fit_end']), int(original['train_end'])
    historical_X = Xold[fit_end:train_end]
    historical_y = yold[fit_end:train_end]
    if len(np.unique(historical_y)) != 2:
        raise ValueError('Historical guard requires both classes')

    X, y, labels, raw_seen, invalid, was_sorted = load_feb23()
    static = load_checkpoint(SOURCE_MODEL, X.shape[1])
    active = clone(static)
    historical_baseline = score(historical_y,
                                probabilities(static, historical_X))

    # Historical training examples never include Feb-23 (the current stream).
    replay = BalancedReservoir(capacity_per_class=250, seed=42)
    replay.add(Xold[:fit_end], yold[:fit_end])
    del Xold, yold
    recent = deque(maxlen=RECENT_BATCHES)
    new_attack_pool = deque(maxlen=NEW_ATTACK_POOL_CAPACITY)

    candidate_pending = None
    attempted_batches = []
    accepted_batches = []
    decisions = []
    traces = []
    probs_baseline = []
    probs_prequential = []
    last_attempt = -COOLDOWN_BATCHES - 1
    cumulative_missed = 0
    cumulative_attacks = 0
    train_seconds = 0.
    prediction_seconds = 0.

    for batch_id, start in enumerate(range(0, len(y), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(y))
        xb = X[start:end]
        yb = y[start:end]
        begin = time.perf_counter()
        base_p = probabilities(static, xb)
        active_p = probabilities(active, xb)
        prediction_seconds += time.perf_counter() - begin
        probs_baseline.append(base_p)
        probs_prequential.append(active_p)

        # At this point labels are only used for post-prediction verification.
        n_attack = int(yb.sum())
        n_missed = int(((yb == 1) & (active_p < 0.5)).sum())
        cumulative_missed += n_missed
        cumulative_attacks += n_attack
        accepted_now = False

        # Train/validate separation: candidate never trains on these batches.
        if candidate_pending is not None:
            pp = probabilities(candidate_pending['model'], xb)
            candidate_pending['ys'].append(yb.copy())
            candidate_pending['active'].append(active_p.copy())
            candidate_pending['candidate'].append(pp.copy())
            candidate_pending['n_batches'] += 1
            vy = np.concatenate(candidate_pending['ys'])
            positives = int(vy.sum())
            negatives = int(len(vy) - positives)
            sufficient = (positives >= MIN_VALIDATION_ATTACKS and
                          negatives >= MIN_VALIDATION_BENIGN)
            exhausted = candidate_pending['n_batches'] >= MAX_VALIDATION_BATCHES
            if sufficient or exhausted:
                decision = {
                    'proposal_batch': candidate_pending['proposal_batch'],
                    'decision_batch': batch_id,
                    'validation_flows': int(len(vy)),
                    'validation_attacks': positives,
                    'validation_benign': negatives,
                    'candidate_trained_on_validation': False,
                }
                if sufficient:
                    inc_m = score(vy, np.concatenate(candidate_pending['active']))
                    cand_m = score(vy, np.concatenate(candidate_pending['candidate']))
                    hist_m = score(historical_y, probabilities(
                        candidate_pending['model'], historical_X))
                    recall_good = (cand_m['recall'] >=
                                   inc_m['recall'] + MIN_VALIDATION_RECALL_GAIN
                                   and cand_m['f1'] >= inc_m['f1'])
                    fpr_good = (cand_m['attack_fpr'] <= inc_m['attack_fpr']
                                + MAX_VALIDATION_FPR_INCREASE)
                    hist_good = (
                        hist_m['macro_f1'] >= historical_baseline['macro_f1']
                        - MAX_HISTORICAL_MACRO_F1_DROP and
                        hist_m['attack_fpr'] <= historical_baseline['attack_fpr']
                        + MAX_HISTORICAL_FPR_INCREASE)
                    ok = bool(recall_good and fpr_good and hist_good)
                    decision.update({
                        'status': 'accepted' if ok else 'rejected',
                        'active_validation': inc_m,
                        'candidate_validation': cand_m,
                        'candidate_historical': hist_m,
                        'passes_recall_f1': bool(recall_good),
                        'passes_fpr_guard': bool(fpr_good),
                        'passes_retention_guard': bool(hist_good),
                    })
                    if ok:
                        active = candidate_pending['model']
                        accepted_batches.append(batch_id)
                        accepted_now = True
                else:
                    decision['status'] = 'insufficient_validation_classes'
                decisions.append(decision)
                print(f'Validation batch {batch_id}: {decision["status"]} '
                      f'({positives} attacks / {negatives} benign)', flush=True)
                candidate_pending = None

        # Verified batch is now eligible for any later model training.
        recent.append((xb.copy(), yb.copy()))
        for row in xb[yb == 1]:
            new_attack_pool.append(np.array(row, copy=True))

        proposed_now = False
        if (candidate_pending is None and not accepted_now
                and cumulative_missed >= MIN_CUMULATIVE_MISSES
                and len(new_attack_pool) >= MIN_NEW_ATTACK_EXAMPLES
                and batch_id - last_attempt >= COOLDOWN_BATCHES):
            begin = time.perf_counter()
            model, reason = build_candidate(active, recent, new_attack_pool,
                                            replay, batch_id)
            train_seconds += time.perf_counter() - begin
            if model is not None:
                candidate_pending = {'model': model, 'proposal_batch': batch_id,
                                     'ys': [], 'active': [], 'candidate': [],
                                     'n_batches': 0}
                proposed_now = True
                attempted_batches.append(batch_id)
                last_attempt = batch_id
                cumulative_missed = 0
                print(f'Proposed batch {batch_id}: total observed attacks '
                      f'{cumulative_attacks}; available new-attack pool '
                      f'{len(new_attack_pool)}', flush=True)

        # Keep the bounded replay cache strictly behind time (observed labels).
        replay.add(xb, yb)
        traces.append({
            'batch': batch_id, 'attacks': n_attack,
            'missed_before_learning': n_missed,
            'missed_accumulated_end': cumulative_missed,
            'candidate_proposed': proposed_now,
            'candidate_accepted': accepted_now,
            'candidate_pending': candidate_pending is not None,
        })
        if batch_id % 500 == 0:
            print(f'Batch {batch_id:04} | attacks={n_attack} | missed={n_missed} '
                  f'| proposed={len(attempted_batches)} '
                  f'| accepted={len(accepted_batches)}', flush=True)

    results_path = Path('results')
    model_path = Path('models')
    results_path.mkdir(exist_ok=True)
    model_path.mkdir(exist_ok=True)
    by_label = Counter(labels)
    summary = {
        'dataset': CSV.name,
        'protocol': 'Full-row Feb-23; timestamp sort if needed; predict '
                    'batch before verified labels; accumulate confirmed '
                    'misses across batches; supervised balanced replay + KD; '
                    'validate on later labeled batches before acceptance',
        'status': 'Exploratory conservative 3:1 replay ablation; not MMD/ADWIN-gated',
        'raw_rows_seen': raw_seen,
        'evaluated_rows': len(y),
        'invalid_rows': invalid,
        'csv_already_chronological': was_sorted,
        'attack_rows': int(y.sum()),
        'benign_rows': int((y == 0).sum()),
        'label_counts': dict(by_label),
        'batches': len(traces),
        'source_or_model_sha256': sha256(SOURCE_MODEL),
        'feb14_preprocessor_sha256': sha256(PREP),
        'static_or_stream': score(y, np.concatenate(probs_baseline)),
        'accumulated_replay_stream': score(y, np.concatenate(probs_prequential)),
        'historical_before': historical_baseline,
        'historical_after': score(historical_y,
                                  probabilities(active, historical_X)),
        'proposed_updates': len(attempted_batches),
        'accepted_updates': len(accepted_batches),
        'proposal_batches': attempted_batches,
        'accepted_batches': accepted_batches,
        'unresolved_candidate_at_end': candidate_pending is not None,
        'decisions': decisions,
        'total_candidate_training_seconds': train_seconds,
        'mean_model_prediction_seconds_per_batch': prediction_seconds / max(len(traces), 1),
        'settings': {
            'batch_size': BATCH_SIZE,
            'min_cumulative_misses': MIN_CUMULATIVE_MISSES,
            'min_new_attack_examples': MIN_NEW_ATTACK_EXAMPLES,
            'candidate_epochs': CANDIDATE_EPOCHS,
            'negative_train_samples': NEGATIVE_TRAIN_SAMPLES,
            'positive_train_samples': POSITIVE_TRAIN_SAMPLES,
            'fixed_pos_weight': FIXED_POS_WEIGHT,
            'distillation_weight': DISTILL_WEIGHT,
            'min_validation_attacks': MIN_VALIDATION_ATTACKS,
            'min_validation_benign': MIN_VALIDATION_BENIGN,
            'max_validation_batches': MAX_VALIDATION_BATCHES,
            'max_validation_fpr_increase': MAX_VALIDATION_FPR_INCREASE,
        },
        'warning': 'Feb-23 is development; verified labels assumed available '
                   'at each batch end. Historical guard has been reused in '
                   'earlier candidate selection. Further retuning on Feb-23 '
                   'would not be independent evaluation.',
    }
    outfile = results_path / f'{OUT_STEM}_summary.json'
    outfile.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame(traces).to_csv(results_path / f'{OUT_STEM}_batches.csv', index=False)
    torch.save({'n_features': int(X.shape[1]),
                'state_dict': active.cpu().state_dict()},
               model_path / f'{OUT_STEM}.pt')
    print('\n=== PHASE 5 SPARSE-ATTACK BASELINE ===')
    print('Frozen OR recall:', round(summary['static_or_stream']['recall'], 4),
          '| macro-F1:', round(summary['static_or_stream']['macro_f1'], 4))
    print('Accumulated recall:', round(summary['accumulated_replay_stream']['recall'], 4),
          '| macro-F1:', round(summary['accumulated_replay_stream']['macro_f1'], 4))
    print('Proposed:', len(attempted_batches), '| Accepted:', len(accepted_batches))
    print('Saved:', outfile)


if __name__ == '__main__':
    main()
