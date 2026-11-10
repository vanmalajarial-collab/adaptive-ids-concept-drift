# Adaptive Intrusion Detection System — step-by-step Python project

**Goal:** CICIDS2017 (time ordered) → initial MLP → RBF-MMD warning and ADWIN error monitoring → candidate fine-tuning → class-balanced historical replay + knowledge distillation → gate on fresh held-out labels and historical retention → continued prequential scoring. Optional comparisons: static MLP, ordinary fine-tuning, replay, replay+distillation, Adaptive Random Forest, temporal CNN/LSTM/GRU.

> This is a **research prototype**, not an operational IDS that can block attacks. The Python scripts analyze labeled network flow CSVs offline as a replayed stream. Do not report synthetic-data numbers as real research results.

## A. Environment (Step 1)

Recommend **Python 3.11+**. On Kali/Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip unzip
cd adaptive_ids_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd adaptive_ids_project
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check packages:

```bash
python -c "import torch, river, sklearn, pandas; print('Setup OK', torch.__version__, river.__version__)"
```

## B. Before actual dataset, run the TEST demo

**The demo is synthetic only**, with a controlled traffic/label change around row 8400. It does not replace CICIDS2017.

```bash
python demo_data.py
python prepare.py --raw data/demo --stride 1
python train.py --epochs 8
python stream.py --batch-size 256
python ablation.py
python compare.py
python compare_deep.py --epochs 4
python plot.py
python known_drift_eval.py --points 8400
```

`results/hybrid_summary.json`, `results/hybrid_batches.csv`, and `results/performance.png` are produced. If no update is accepted, this is possible under the chosen alarm and validation criteria and is not automatically a software error.

## C. Now train on REAL CICIDS2017 (Steps 2–4)

1. Get `GeneratedLabelledFlows.zip` (NOT `MachineLearningCSV.zip`) via the [official CICIDS2017 site](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Extract the timestamped CSV files into `data/raw/` (nested folders also work).
3. Include files from Monday **and** the later days, otherwise the starting training period might contain only benign traffic.
4. Inspect the printed time range. For original capture, dates should fall in **2017-07-03 through 2017-07-07**. The parser assumes `dayfirst=True`, as with the original dd/mm/yyyy timestamp format. If your dataset is transformed or uses a different date format, fix parsing first.

For a quick starter sample (systematic every 10th row per file, not random shuffle):

```bash
python prepare.py --raw data/raw --stride 10
```

For full data, if your RAM is sufficient:

```bash
python prepare.py --raw data/raw --stride 1
```

`data/processed/data_report.json` records the input paths, byte sizes, sample stride, rows, class counts and time range. Keep a copy for your report. Data and feature metadata from other datasets must be preprocessed in their **own** experiments — do not directly concatenate CSE-CIC-IDS2018 or TON_IoT without schema/label alignment.

## D. Train baseline detector (Step 5)

```bash
python train.py --epochs 8 --train-fraction 0.60 --seed 42
```

Chronological protocol:

- First **60%** of stream = initial period; first 90% of it used to fit imputation, scaling and model. Its last 10% is a **historical holdout** never used in training.
- Next **15%** = development period for one-time initial validation, not used as future evaluation samples.
- Last **25%** = **evaluation stream**; never shuffled and never used to fit the initial scaler or initial model.

Save paths: `models/initial_mlp.pt`, `models/preprocessor.joblib`, `models/initial_metrics.json`. If it errors **one class in initial training**, increase the training fraction slightly (e.g. `--train-fraction 0.70`) only while leaving a substantial evaluation stream; inspect the class distribution and report the selected split.

## E. Run MMD + ADWIN + adaptation (Steps 6–12)

```bash
python stream.py --batch-size 256
```

Algorithm (each consecutive batch):

1. Predict with the incumbent model, store predictions and prediction latency.
2. Evaluate those predictions with **verified labels available at the end of each batch** (simulated immediate batch-end feedback; not delayed-label evaluation).
3. RBF-MMD uses 96 flows from a frozen initial reference versus 96 current flows, gamma = reciprocal median squared initial distance; cutoff calibrated from **15 early training samples** (95th percentile × 1.15). MMD alone is **warning only**, not proof of harmful concept drift.
4. ADWIN (delta `0.002`) consumes 0=correct or 1=incorrect errors one-by-one. A statistically detected change with a **rising** recent error rate counts as a possible concept drift.
5. Trigger a candidate update only if the ADWIN rising-error event happens within four batches of an MMD warning.
6. Train on the preceding observed batches + class-balanced **bounded reservoir replay** (at most 250 stored samples per class). Knowledge distillation uses the previous frozen MLP as teacher, student initialized from that model, `T=2`, `20%` distillation loss.
7. Validate against the latest batch, which is held out of *that* update, and the untouched historical retention set. Require `recent_macro_F1 > incumbent+0.005`, `historical_macro_F1 >= incumbent-0.03`, and historical benign false-positive rate not worse by >0.03. Reject otherwise.
8. On acceptance, write `models/rollback_mlp.pt` and `models/updated_mlp.pt`. Continue to next batch. Log resource use, prediction and update times.

**Reference policy:** the MMD reference is frozen for the first prototype. **Model policy:** a rejected candidate is discarded. **No ground-truth fabrication:** predicted labels are never treated as verified labels.

## F. Compare methods (Step 13)

```bash
python ablation.py --batch-size 256 --interval 5
python compare.py --arf-train-limit 0 --arf-test-limit 0
python compare_deep.py --length 8 --epochs 4
```

- `ablation.py`: static MLP vs periodic fine-tuning vs replay vs replay+distillation. **These baselines deliberately lack candidate validation**; the full hybrid method additionally requires drift alarms and acceptance checks. They all use prequential evaluation.
- `compare.py`: River `forest.ARFClassifier`, `learn_one` and `predict_proba_one`. The ARF option is computationally expensive on full datasets. The provided default limits are a fast trial; to report a comparable full-data experiment use both limits 0.
- `compare_deep.py`: CNN, LSTM, GRU trained on non-overlapping chronological windows of 8 flows, scored on the last flow in each window. An initial MLP is scored on the **same endpoints**. Neighboring flows may belong to different hosts — interpret this as a temporal baseline, not necessarily a host-session model.
- `stream.py` is the **full hybrid**; compare its `results/hybrid_summary.json` to ablation results, reporting where training budgets/update frequency differ.

## G. Produce evaluation outputs (Step 14)

```bash
python plot.py
```

- `results/hybrid_batches.csv`: per-batch precision, recall, F1, macro-F1, PR-AUC (if both labels occur), benign false-positive rate, MMD, ADWIN warnings, attempted/accepted updates, throughput, peak-observed process memory and replay rows.
- `results/hybrid_summary.json`: whole-stream **prequential** classification metrics, total alarm counts and update count.
- `results/ablation.json`, `results/comparison.json`, `results/deep_sequence_comparison.json`: extra baselines.
- `results/performance.png`: performance over time; orange lines are MMD warnings and green dashed lines are accepted updates.

**Very important:** an attack false alarm (benign classified as attack) is not the same as a false **drift** alarm. On unmodified CICIDS2017, the true underlying concept-drift change locations are not annotated, so honest `drift_false_alarm_rate`, `missed_drifts`, `drift_detection_delay`, and `recovery_time` cannot automatically be determined from filename/day changes. Use controlled scenarios with externally known change points. A synthetic smoke-test example:

```bash
python known_drift_eval.py --points 8400
```

That last command is only valid after the included synthetic demo, or when you supply an actual documented change point for your own controlled stream. Repeat real experiments with multiple random seeds (e.g. 42/43/44), record variability and compute drift delay/recovery against **known** drift points only.

## File-to-step map

| Steps in `Steps.docx` | Relevant script |
|---|---|
| 1 | `requirements.txt`, `README.md` |
| 2,3,4 | `prepare.py`, `data/processed/` |
| 5 | `model.py`, `train.py` |
| 6,7,8 | `stream.py` (`rbf_mmd2`, `calibrate_mmd`, ADWIN, trigger rule) |
| 9,10 | `stream.py`, `replay.py`, `model.py` |
| 11 | `stream.py` (`adaptive_update`, acceptance gate, rollback) |
| 12 | `stream.py` (predict/evaluate/learn order) |
| 13 | `ablation.py`, `compare.py`, `compare_deep.py` |
| 14 | `plot.py`, `known_drift_eval.py`, streamed CSV/JSON metrics |

## Cautions and limitations

- Offline flow-based Python replay, **not live packet sniffing**.
- Current label feedback is assumed available at the **end of a batch**; for realistic delayed labels, add a label-arrival queue before feeding ADWIN/updating the model.
- MMD estimates **input distribution change**, whereas ADWIN catches **error-rate change**. They are complementary, not interchangeable.
- Results on different dataset schemas should first be compared as **separate experiments**.
- The sample stride is intended for rapid prototyping; re-run at stride 1 for a research-quality experiment if practical.
- CICIDS2017 datasets contain known artifacts/label issues; report the exact source and preprocessing policy and do not overclaim real-world generalization.
- The provided implementation has a basic test with synthetic data; check all code on your own environment with the real River package and original dataset before publication.
