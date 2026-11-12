import json
import numpy as np
import torch
from pathlib import Path
from model import load_model, clone, probabilities, score

data = np.load("data/processed/scaled_data.npz")
X = data["X"]
y = data["y"].astype(int)

fit_end = int(data["fit_end"])
train_end = int(data["train_end"])

# Historical retention set
X_hist = X[fit_end:train_end]
y_hist = y[fit_end:train_end]

initial = load_model()

updated = clone(initial)
weights = torch.load(
    "models/updated_mlp_mmd95.pt",
    map_location="cpu",
    weights_only=True
)
updated.load_state_dict(weights)

initial_metrics = score(
    y_hist, probabilities(initial, X_hist)
)

updated_metrics = score(
    y_hist, probabilities(updated, X_hist)
)

difference = (
    updated_metrics["macro_f1"]
    - initial_metrics["macro_f1"]
)

results = {
    "initial": initial_metrics,
    "updated": updated_metrics,
    "macro_f1_change": difference
}

Path("results/forgetting_diagnostic.json").write_text(
    json.dumps(results, indent=2)
)

print("Initial historical Macro-F1:",
      round(initial_metrics["macro_f1"], 4))

print("Updated historical Macro-F1:",
      round(updated_metrics["macro_f1"], 4))

print("Change:", round(difference, 4))

print("Saved results/forgetting_diagnostic.json")
