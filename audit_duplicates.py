import numpy as np
import pandas as pd
from collections import defaultdict

raw = np.load("data/processed/raw_data.npz")
scaled = np.load("data/processed/scaled_data.npz")

X_raw = raw["X"]
X_scaled = scaled["X"]
y = scaled["y"]

fit_end = int(scaled["fit_end"])
dev_end = int(scaled["dev_end"])

assert len(X_raw) == len(X_scaled)
assert np.array_equal(raw["y"], y)

def row_hashes(X):
    return pd.util.hash_pandas_object(
        pd.DataFrame(X), index=False
    ).to_numpy()

def check_overlap(X):
    train_hash = row_hashes(X[:fit_end])
    stream_hash = row_hashes(X[dev_end:])

    matched = np.isin(stream_hash, train_hash)
    return train_hash, stream_hash, matched

raw_train, raw_stream, raw_match = check_overlap(X_raw)
_, _, scaled_match = check_overlap(X_scaled)

print("\n=== DUPLICATE VERIFICATION ===")

print("Raw feature matches:", int(raw_match.sum()))
print("Scaled feature matches:", int(scaled_match.sum()))

print("Matches introduced after preprocessing:",
      int((scaled_match & ~raw_match).sum()))

print("Raw overlap percentage:",
      round(raw_match.mean() * 100, 4), "%")

# Check label consistency for matching raw features
train_labels = defaultdict(set)

for h, label in zip(raw_train, y[:fit_end]):
    train_labels[int(h)].add(int(label))

conflicts = 0

for h, label, matched in zip(
    raw_stream, y[dev_end:], raw_match
):
    if matched and int(label) not in train_labels[int(h)]:
        conflicts += 1

print("Raw matches with conflicting labels:", conflicts)

# Check chronological ordering
times = raw["times"]

print("\n=== TIME ORDER CHECK ===")
print("Backward timestamp jumps:",
      int(np.sum(np.diff(times) < 0)))

print("\n=== FEATURE AUDIT ===")
print("Feature count:", len(raw["columns"]))
print("Feature names:")
for name in raw["columns"]:
    print(" -", name)

print("\nAudit completed!")
