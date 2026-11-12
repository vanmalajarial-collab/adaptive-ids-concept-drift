import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from model import load_model, probabilities

# Load the preprocessed real dataset
data = np.load("data/processed/scaled_data.npz")
X = data["X"]
y = data["y"]

train_end = int(data["train_end"])
dev_end = int(data["dev_end"])

# Development data only; streaming data stays untouched
X_dev = X[train_end:dev_end]
y_dev = y[train_end:dev_end]

model = load_model()
probs = probabilities(model, X_dev)

def evaluate(threshold):
    pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_dev, pred, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (2 * precision * recall /
          (precision + recall)) if precision + recall else 0
    fpr = fp / (fp + tn) if fp + tn else 0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

results = [
    evaluate(float(t))
    for t in np.linspace(0.01, 0.99, 99)
]

# Example constraint: at most 2% benign false positives
max_fpr = 0.02
valid = [r for r in results if r["fpr"] <= max_fpr]

if not valid:
    raise RuntimeError("No threshold satisfies the FPR limit.")

best = max(
    valid,
    key=lambda r: (r["f1"], r["recall"], -r["fpr"])
)

baseline = evaluate(0.5)

Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

Path("results/threshold_tuning.json").write_text(
    json.dumps({
        "baseline": baseline,
        "selected": best,
        "max_fpr": max_fpr,
        "split": "development"
    }, indent=2)
)

Path("models/decision_threshold.json").write_text(
    json.dumps({
        "threshold": best["threshold"]
    }, indent=2)
)

print("\nOriginal threshold:", baseline["threshold"])
print("Original F1:", round(baseline["f1"], 4))

print("\nSelected threshold:", best["threshold"])
print("Selected F1:", round(best["f1"], 4))
print("Precision:", round(best["precision"], 4))
print("Recall:", round(best["recall"], 4))
print("False Positive Rate:", round(best["fpr"], 4))

print("\nResults saved successfully!")
