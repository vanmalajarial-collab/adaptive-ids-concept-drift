import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import load_model, probabilities

data = np.load("data/processed/scaled_data.npz")
raw = np.load("data/processed/raw_data.npz")

X = data["X"]
y = data["y"].astype(int)

fit_end = int(data["fit_end"])
train_end = int(data["train_end"])
dev_end = int(data["dev_end"])

model = load_model()

print("\n=== MODEL PROBABILITY ANALYSIS ===")

periods = {
    "Development": (train_end, dev_end),
    "Thursday Streaming": (dev_end, len(y))
}

for name, (start, end) in periods.items():

    x_part = X[start:end]
    y_part = y[start:end]

    probs = np.asarray(
        probabilities(model, x_part)
    ).reshape(-1)

    print(f"\n{name}")
    print("Total samples:", len(y_part))

    for label, class_name in [
        (0, "Benign"),
        (1, "Attack")
    ]:
        values = probs[y_part == label]

        if len(values) == 0:
            continue

        print(f"\n{class_name}:")
        print("Samples:", len(values))
        print("Median probability:",
              round(float(np.median(values)), 6))
        print("90th percentile:",
              round(float(np.percentile(values, 90)), 6))
        print("99th percentile:",
              round(float(np.percentile(values, 99)), 6))
        print("Predicted attack at 0.5:",
              int(np.sum(values >= 0.5)))

print("\n=== FEATURE DISTRIBUTION SHIFT ===")

# Compare standardized features, without changing them
train_median = np.median(X[:fit_end], axis=0)
stream_median = np.median(X[dev_end:], axis=0)

differences = np.abs(
    stream_median - train_median
)

feature_names = raw["columns"]

top_indices = np.argsort(differences)[-10:][::-1]

print("\nTop 10 shifted feature medians:")

for i in top_indices:
    print(
        f"{feature_names[i]}: "
        f"{differences[i]:.4f}"
    )

print("\nDiagnosis completed!")
