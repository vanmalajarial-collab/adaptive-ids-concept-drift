import json
from pathlib import Path

import numpy as np
import pandas as pd

from stream import calibrate_mmd, rbf_mmd2

# Load existing processed CICIDS2017
data = np.load("data/processed/scaled_data.npz")

X = data["X"]
fit_end = int(data["fit_end"])

# Training data only for calibration
X_train = X[:fit_end]

# Reproduce existing calibration
ref, gamma, old_threshold = calibrate_mmd(
    X_train, window=96
)

print("\n=== ORIGINAL MMD SETTINGS ===")
print("Gamma:", gamma)
print("Original threshold:", old_threshold)

# Sample chronological training windows
window = 96
n_windows = 150

starts = np.linspace(
    0,
    len(X_train) - window,
    num=n_windows,
    dtype=int
)

values = []

for start in starts:
    current = X_train[start:start + window]
    mmd = rbf_mmd2(ref, current, gamma)
    values.append(mmd)

values = np.asarray(values)

# Candidate thresholds from training only
threshold_95 = float(np.quantile(values, 0.95))
threshold_99 = float(np.quantile(values, 0.99))

print("\n=== TRAINING WINDOW ANALYSIS ===")
print("Mean MMD:", values.mean())
print("Median MMD:", np.median(values))
print("95th percentile:", threshold_95)
print("99th percentile:", threshold_99)

# Read the recorded stream measurements
df = pd.read_csv("results/hybrid_batches.csv")
stream_values = df["mmd2"].dropna().to_numpy()

print("\n=== STREAM WARNING COMPARISON ===")

comparison = {}

for name, threshold in {
    "original": old_threshold,
    "training_95": threshold_95,
    "training_99": threshold_99,
}.items():

    warnings = int(np.sum(stream_values > threshold))
    rate = warnings / len(stream_values) * 100

    comparison[name] = {
        "threshold": float(threshold),
        "warnings": warnings,
        "warning_percentage": round(rate, 2)
    }

    print(
        name,
        "| warnings:", warnings,
        "| rate:", round(rate, 2), "%"
    )

Path("results").mkdir(exist_ok=True)

Path("results/mmd_diagnostic.json").write_text(
    json.dumps(comparison, indent=2)
)

print("\nSaved results/mmd_diagnostic.json")
