import json
from pathlib import Path

import numpy as np
from river import forest
from model import score

data = np.load("data/processed/scaled_data.npz")
X = data["X"]
y = data["y"].astype(int)

fit_end = int(data["fit_end"])
dev_end = int(data["dev_end"])

batch_size = 256

model = forest.ARFClassifier(n_models=5, seed=42)

print("Training initial ARF model...")

for i in range(fit_end):
    features = dict(enumerate(map(float, X[i])))
    model.learn_one(features, int(y[i]))

    if (i + 1) % 10000 == 0:
        print("Trained:", i + 1)

print("Starting batch-wise streaming evaluation...")

all_probs = []
all_labels = []

for start in range(dev_end, len(y), batch_size):
    stop = min(start + batch_size, len(y))

    xb = X[start:stop]
    yb = y[start:stop]

    # Predict the entire batch BEFORE learning its labels
    probs = []

    for row in xb:
        features = dict(enumerate(map(float, row)))
        probability = model.predict_proba_one(
            features
        ).get(1, 0.0)
        probs.append(float(probability))

    all_probs.extend(probs)
    all_labels.extend(yb.tolist())

    # Labels become available at end of batch
    for row, label in zip(xb, yb):
        features = dict(enumerate(map(float, row)))
        model.learn_one(features, int(label))

results = score(
    np.asarray(all_labels),
    np.asarray(all_probs)
)

results["batch_size"] = batch_size
results["initial_train_rows"] = fit_end
results["evaluation_rows"] = len(all_labels)
results["label_policy"] = "end_of_batch"

Path("results").mkdir(exist_ok=True)

Path("results/arf_batch256_summary.json").write_text(
    json.dumps(results, indent=2)
)

print("\nARF Batch-256 Results:")
print(json.dumps(results, indent=2))
