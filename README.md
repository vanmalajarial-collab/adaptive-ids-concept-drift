# Adaptive Intrusion Detection System — Concept Drift & Continual Learning

**Status:** Completed research prototype and exploratory evaluation. **Not** a live intrusion-prevention product.

This project replays chronological, labeled network-flow data through an initial multilayer perceptron (MLP) IDS, monitors traffic distribution and prediction errors with **RBF-MMD** and **ADWIN**, and investigates validated model updates using **class-balanced experience replay**, **knowledge distillation**, and historical retention checks. The code analyzes CSV network flows offline; it does not sniff live packets or block traffic.

## Research question

Can drift-triggered continual learning improve attack detection under changing traffic while controlling false positives and retaining previous knowledge? The experiments show gains on one exploratory Thursday stream, but also reveal serious limitations on unseen attack families and rare-attack periods.

## Stack

- **OS tested:** Kali Linux; Python virtual environment.
- **Python tools:** pandas, NumPy, PyTorch, scikit-learn, River (ADWIN / Adaptive Random Forest), matplotlib, joblib, psutil.
- **Datasets:** CICIDS2017 and CSE-CIC-IDS2018, analyzed as separate experiments because feature schemas differ.
- **Data handling:** time-ordered samples, train-only imputation/scaling, binary benign/attack labels, prequential batch scoring (predict before labels become available).

## Repository layout

```text
adaptive_ids_project/
├── prepare.py                # CICIDS2017 parsing, cleaning and chronological ordering
├── train.py                  # Initial MLP and train-only preprocessing
├── model.py                  # MLP, scoring and training helpers
├── stream.py                 # Initial hybrid MMD + ADWIN IDS
├── replay.py                 # Class-balanced experience replay buffer
├── ablation.py               # Static / fine-tune / replay / distillation comparisons
├── compare.py                # Adaptive Random Forest baseline
├── compare_deep.py           # CNN / LSTM / GRU baselines
├── plot.py                   # Initial performance plots
├── requirements.txt
└── experiments/cicids2018/
    ├── prepare2018.py        # Date-specific preparation, separate schema
    ├── train2018_time.py     # Feb-14 chronological splits and initial model
    ├── hybrid_trigger_ablation2018.py
    ├── calibrate_thresholds_feb14.py
    ├── evaluate_feb22_calibrated2018.py
    ├── phase5_accumulated_replay2018.py
    ├── phase5_conservative_replay2018.py
    └── phase5_candidate_calibration2018.py
```

Some later scripts above were created as separate experimental files; confirm they are in `experiments/cicids2018/` before running the corresponding commands. `data/`, `models/`, `results/` and the Python virtual environment are local artifacts rather than prerequisites to commit.

## Setup and reproducibility

```bash
cd ~/adaptive_ids_project
source venv/bin/activate
python --version
python -c "import torch, river, pandas, sklearn; print('Environment OK')"
```

The original starter `requirements.txt` pins `river>=0.22,<0.24`, while later experiments were run with **River 0.26.1**. Preserve your actual working environment and capture `python -m pip freeze > environment_freeze.txt` before choosing a public dependency pin. Do not imply the original requirements were verified for every later script.

### CICIDS2017 starter pipeline

```bash
python prepare.py --raw data/raw --stride 10
python train.py --epochs 8
python stream.py --batch-size 256
python ablation.py
python compare.py
python compare_deep.py --epochs 4
python plot.py
```

The `data/raw/` directory must contain the appropriate legitimate CICIDS2017 CSVs first. The starter CICIDS2017 experiment sampled 283,072 labeled flows at stride 10. The original time-series and deep-learning baselines have different training budgets / sample definitions; comparisons are exploratory unless those budgets are controlled.

### CSE-CIC-IDS2018 experiment

From the repo root:

```bash
cd experiments/cicids2018
source ../../venv/bin/activate
python hybrid_trigger_ablation2018.py --all
python calibrate_thresholds_feb14.py
python evaluate_feb22_calibrated2018.py
python phase5_accumulated_replay2018.py
python phase5_conservative_replay2018.py
python phase5_candidate_calibration2018.py
```

**Prerequisites:** date-specific CSV files, Feb-14-trained `models/preprocessor.joblib`, saved frozen checkpoints, and `data/processed/scaled_data.npz`. The 2018 scripts assume they exist; they cannot be run on a new checkout without preparing data and training or restoring approved checkpoints. In particular, the Feb-22 evaluation expects the Feb-14 development calibration JSON and six saved checkpoints.

## Dataset split and usage ledger

| Source | Date / use | Scientific status |
|---|---|---|
| CICIDS2017 | July 3–7, sampled chronology | Initial proof-of-concept and exploratory baselines |
| CSE-CIC-IDS2018 | Feb 14, 2018 | Initial training, historical retention and development calibration |
| CSE-CIC-IDS2018 | Feb 15, 2018 | Explored streaming, replay, trigger policy and candidate validation |
| CSE-CIC-IDS2018 | Feb 16 and Feb 21 | Frozen-model diagnostics; subsequently inspected (not suitable as new tests) |
| CSE-CIC-IDS2018 | Feb 22, 2018 | One-time frozen-model evaluation with preselected Feb-14 thresholds; now inspected |
| CSE-CIC-IDS2018 | Feb 23, 2018 | Sparse-attack developmental adaptation experiment; now inspected |

No additional unseen-day performance claim is made here. TON_IoT appears in the original research plan but was **not completed as a separate evaluated experiment** in the reported results.

## Selected experimental results

**15 February, exploratory prequential stream (5,250 attacks; 410 batches):**

| Trigger | Attack recall | Macro-F1 | Benign FPR | Proposals | Accepted updates |
|---|---:|---:|---:|---:|---:|
| AND | 3.37% | 52.01% | ~0.01% | 2 | 1 |
| ADWIN-only | 24.82% | 66.50% | 0.82% | 7 | 3 |
| MMD-only | 18.76% | 63.54% | 0.47% | 13 | 3 |
| OR | **28.51%** | **69.54%** | 0.52% | 17 | 4 |

**22 February, previously uninspected at run time, frozen-model evaluation** (1,048,575 flows; 362 attacks; threshold fixed using Feb-14 development):

| Model | Recall | Attack F1 | Precision | False positives |
|---|---:|---:|---:|---:|
| Static MLP | 18.78% | 8.00% | 5.08% | 1,270 |
| Balanced guarded | 16.30% | 7.46% | 4.84% | 1,160 |
| AND | 18.78% | 7.73% | 4.87% | 1,329 |
| ADWIN | 18.78% | 6.84% | 4.18% | 1,559 |
| MMD | 18.78% | 8.18% | 5.23% | 1,232 |
| OR | 18.51% | 8.72% | 5.71% | 1,107 |

**23 February, sparse attack development:** 1,048,575 flows; 566 attacks. The proposed accumulation/conservative/calibration experiments each attempted 19 candidate proposals; none were accepted. The frozen OR and adapted streams therefore had the same observed recall **13.60%**, attack F1 **2.00%**, and benign FPR **0.6743%**. Rejected candidates improved ranking in many windows but had unacceptable false-positive/retention trade-offs.

## Interpretation and limitations

- **Distribution shift ≠ confirmed harmful concept drift.** MMD indicates feature-distribution differences; ADWIN monitors observed prediction-error changes after true labels arrive. No ground-truth drift points were supplied for the real datasets, so drift detection delay and drift false-alarm rate are not verified.
- **Accuracy is misleading on rare attacks.** Report class-specific recall, precision, PR-AUC, F1, benign FPR and confusion counts.
- **Prequential feedback is optimistic for real operations.** The simulator assumes verified labels at the end of each batch; real networks may have delayed labels.
- **Threshold drift matters.** Feb-21 HOIC scores were ranked well but often below the fixed 0.50 threshold. Threshold `0.001` was inspected using an earlier test day and is *diagnostic only*, not deployable proof.
- **Selection bias.** Thursday was repeatedly inspected/tuned. Feb-14 threshold-calibration protocol was designed after other days were inspected. Feb-22 was evaluated once under the fixed protocol, but further tuning on it would remove its independent-test role.
- **No production guarantee.** No claims of live intrusion prevention, all-attack coverage, or validated novelty over every prior paper.

## Results, integrity and GitHub hygiene

The Kali experiments generated `results/hybrid_trigger_ablation2018_comparison.json`, `results/heldout_22feb_calibrated2018_summary.json`, `results/phase5_feb23_*summary.json`, and charts in `results/final_charts/`. The user verified a snapshot using SHA-256 manifests with all files reporting `OK`.

Keep raw data, model binaries, credentials, virtual environments and oversized artifacts out of Git. Keep a compact `results/README.md` and optionally the sanitized chart PNGs / summary JSONs if their licensing and privacy have been checked. Avoid committing data or results whose provenance cannot be independently verified.

For reproducibility, save: (1) dataset dates, hashes and labels, (2) environment freeze, (3) scripts and configuration values, (4) seed and model checkpoint hashes, and (5) explicit notes for explored-versus-reserved dates.

## References

1. CICIDS2017 — https://www.unb.ca/cic/datasets/ids-2017.html
2. CSE-CIC-IDS2018 — https://www.unb.ca/cic/datasets/ids-2018.html
3. Gretton et al. (2012), *A Kernel Two-Sample Test*, JMLR — https://www.jmlr.org/papers/v13/gretton12a.html
4. Bifet & Gavaldà (2007), *Learning from Time-Changing Data with Adaptive Windowing*, SDM — https://doi.org/10.1137/1.9781611972771.42
5. Hinton et al. (2015), *Distilling the Knowledge in a Neural Network* — https://arxiv.org/abs/1503.02531

**Research scope:** an auditable educational prototype to study adaptive detection and failure cases, not a deployed security control.
