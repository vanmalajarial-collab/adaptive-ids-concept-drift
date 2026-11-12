import numpy as np
import pandas as pd

raw = np.load("data/processed/raw_data.npz")
scaled = np.load("data/processed/scaled_data.npz")

y = scaled["y"].astype(int)
times = raw["times"]

if np.issubdtype(times.dtype, np.number):
    dates = pd.to_datetime(times, unit="ns", errors="coerce")
else:
    dates = pd.to_datetime(times, errors="coerce")

df = pd.DataFrame({
    "Timestamp": dates,
    "Attack": y
})

print("\n=== DATA QUALITY ===")
print("Missing timestamps:", df["Timestamp"].isna().sum())
print("Chronological:", df["Timestamp"].is_monotonic_increasing)

print("\n=== HOURLY ATTACK DISTRIBUTION ===")

df["Hour"] = df["Timestamp"].dt.floor("h")

hourly = df.groupby("Hour").agg(
    Total=("Attack", "size"),
    Attacks=("Attack", "sum")
)

hourly["Benign"] = hourly["Total"] - hourly["Attacks"]

print(hourly.to_string())

print("\n=== SPLIT TIME RANGES ===")

fit_end = int(scaled["fit_end"])
train_end = int(scaled["train_end"])
dev_end = int(scaled["dev_end"])

splits = {
    "Training": (0, fit_end),
    "Holdout": (fit_end, train_end),
    "Development": (train_end, dev_end),
    "Streaming": (dev_end, len(df))
}

for name, (start, end) in splits.items():
    section = df.iloc[start:end]

    print(f"\n{name}")
    print("Start:", section["Timestamp"].iloc[0])
    print("End:", section["Timestamp"].iloc[-1])
    print("Total:", len(section))
    print("Attacks:", int(section["Attack"].sum()))

print("\nAnalysis completed!")
