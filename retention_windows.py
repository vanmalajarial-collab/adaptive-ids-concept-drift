import numpy as np
import pandas as pd
import torch

from model import (
    load_model, clone, probabilities, score
)

data = np.load("data/processed/scaled_data.npz")

X = data["X"]
y = data["y"].astype(int)

fit_end = int(data["fit_end"])
train_end = int(data["train_end"])

initial = load_model()
updated = clone(initial)

weights = torch.load(
    "models/updated_mlp_mmd95.pt",
    map_location="cpu",
    weights_only=True
)
updated.load_state_dict(weights)

# Split historical holdout into 5 chronological windows
windows = np.array_split(
    np.arange(fit_end, train_end), 5
)

results = []

for i, indices in enumerate(windows, start=1):
    x_window = X[indices]
    y_window = y[indices]

    old = score(
        y_window,
        probabilities(initial, x_window)
    )

    new = score(
        y_window,
        probabilities(updated, x_window)
    )

    both_classes = len(np.unique(y_window)) == 2

    result = {
        "window": i,
        "flows": len(indices),
        "attacks": int(np.sum(y_window == 1)),
        "initial_recall": (
            old["recall"] if both_classes else None
        ),
        "updated_recall": (
            new["recall"] if both_classes else None
        ),
        "initial_macro_f1": (
            old["macro_f1"] if both_classes else None
        ),
        "updated_macro_f1": (
            new["macro_f1"] if both_classes else None
        ),
        "initial_false_positives": old["fp"],
        "updated_false_positives": new["fp"],
        "initial_missed_attacks": old["fn"],
        "updated_missed_attacks": new["fn"]
    }

    results.append(result)

df = pd.DataFrame(results)

print("\n=== HISTORICAL RETENTION BY WINDOW ===")
print(df.to_string(index=False))

df.to_csv(
    "results/retention_windows.csv",
    index=False
)

print("\nSaved results/retention_windows.csv")
