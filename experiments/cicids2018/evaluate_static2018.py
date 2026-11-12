import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import load_model, probabilities, score

# Load CICIDS2018 processed data
data = np.load("data/processed/scaled_data.npz")

X = data["X"]
y = data["y"].astype(int)

dev_end = int(data["dev_end"])

# Untouched Thursday streaming period
X_test = X[dev_end:]
y_test = y[dev_end:]

print("\n=== STATIC MLP EVALUATION ===")
print("Evaluation samples:", len(y_test))
print("Actual attacks:", int((y_test == 1).sum()))

# Load original model without adaptation
model = load_model()

# Predict, but DO NOT retrain
probs = probabilities(model, X_test)

results = score(y_test, probs)

# Inspect probability distributions
attack_probs = probs[y_test == 1]
benign_probs = probs[y_test == 0]

results["attack_probability_median"] = float(
    np.median(attack_probs)
)
results["benign_probability_median"] = float(
    np.median(benign_probs)
)
results["model_type"] = "Static MLP"
results["updates"] = 0

print("\n=== RESULTS ===")
print(json.dumps(results, indent=2))

Path("results").mkdir(exist_ok=True)

Path("results/static2018_summary.json").write_text(
    json.dumps(results, indent=2)
)

print("\nSaved results/static2018_summary.json")
