import sys
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Import existing project model
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import MLP, DEVICE, fit, probabilities, score

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load CICIDS2018 data
raw = np.load("data/processed/raw_data.npz")

X = raw["X"]
y = raw["y"].astype(int)
times = raw["times"].astype("datetime64[ns]")
features = raw["columns"].tolist()

# Ensure chronological order
if np.any(times[1:] < times[:-1]):
    raise ValueError("Timestamps are not sorted!")

# Explicit time boundaries
fit_end = int(np.searchsorted(
    times, np.datetime64("2018-02-14T10:00:00")
))

train_end = int(np.searchsorted(
    times, np.datetime64("2018-02-14T11:00:00")
))

dev_end = int(np.searchsorted(
    times, np.datetime64("2018-02-15T00:00:00")
))

splits = {
    "Training": (0, fit_end),
    "Holdout": (fit_end, train_end),
    "Development": (train_end, dev_end),
    "Streaming": (dev_end, len(y))
}

print("\n=== NEW CHRONOLOGICAL SPLITS ===")

for name, (start, end) in splits.items():
    labels = y[start:end]
    if len(labels) == 0:
        raise ValueError(f"{name} is empty")

    benign = int((labels == 0).sum())
    attacks = int((labels == 1).sum())

    print(f"{name}: {len(labels)} rows")
    print(f"  Benign: {benign}")
    print(f"  Attacks: {attacks}")

    if benign == 0 or attacks == 0:
        raise ValueError(
            f"{name} contains only one class"
        )

# Fit preprocessing ONLY on training period
imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True
)

scaler = StandardScaler()

X_fit = imputer.fit_transform(X[:fit_end])
scaler.fit(X_fit)

X_scaled = scaler.transform(
    imputer.transform(X)
).astype("float32")

np.clip(X_scaled, -10, 10, out=X_scaled)

# Save scaled dataset
Path("data/processed").mkdir(
    parents=True, exist_ok=True
)

np.savez_compressed(
    "data/processed/scaled_data.npz",
    X=X_scaled,
    y=y,
    fit_end=fit_end,
    train_end=train_end,
    dev_end=dev_end
)

# Train MLP
print("\n=== TRAINING MLP ===")

model = MLP(X_scaled.shape[1]).to(DEVICE)

fit(
    model,
    X_scaled[:fit_end],
    y[:fit_end],
    epochs=8
)

# Evaluate only the pre-stream periods
holdout_metrics = score(
    y[fit_end:train_end],
    probabilities(
        model, X_scaled[fit_end:train_end]
    )
)

dev_metrics = score(
    y[train_end:dev_end],
    probabilities(
        model, X_scaled[train_end:dev_end]
    )
)

print("\nHOLDOUT RESULTS:")
print(json.dumps(holdout_metrics, indent=2))

print("\nDEVELOPMENT RESULTS:")
print(json.dumps(dev_metrics, indent=2))

# Save checkpoint and preprocessing
Path("models").mkdir(exist_ok=True)

torch.save(
    {
        "state_dict": model.cpu().state_dict(),
        "n_features": X_scaled.shape[1],
        "features": features
    },
    "models/initial_mlp.pt"
)

joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "features": features
    },
    "models/preprocessor.joblib"
)

Path("models/initial_metrics.json").write_text(
    json.dumps(dev_metrics, indent=2)
)

Path("models/split_report.json").write_text(
    json.dumps({
        "splits": {
            name: {
                "start": start,
                "end": end,
                "benign": int((y[start:end] == 0).sum()),
                "attacks": int((y[start:end] == 1).sum())
            }
            for name, (start, end) in splits.items()
        },
        "holdout_metrics": holdout_metrics,
        "development_metrics": dev_metrics
    }, indent=2)
)

print("\nTraining completed successfully!")
print("Model: models/initial_mlp.pt")
print("Data: data/processed/scaled_data.npz")
