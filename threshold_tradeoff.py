import numpy as np
import torch
from model import load_model, clone, probabilities

data = np.load("data/processed/scaled_data.npz")

X = data["X"]
y = data["y"].astype(int)

fit_end = int(data["fit_end"])
train_end = int(data["train_end"])

# Historical holdout data
X_test = X[fit_end:train_end]
y_test = y[fit_end:train_end]

# Load updated MLP
model = clone(load_model())

weights = torch.load(
    "models/updated_mlp_mmd95.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(weights)

probs = probabilities(model, X_test)

print("\n=== THRESHOLD COMPARISON ===")
print("Threshold | Precision | Recall | F1 | FPR")

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:

    pred = (probs >= threshold).astype(int)

    tp = np.sum((pred == 1) & (y_test == 1))
    tn = np.sum((pred == 0) & (y_test == 0))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    fpr = fp / (fp + tn) if fp + tn else 0

    print(
        f"{threshold:.2f}      | "
        f"{precision:.4f}    | "
        f"{recall:.4f} | "
        f"{f1:.4f} | "
        f"{fpr:.4f}"
    )
