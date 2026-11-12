import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

folder = Path("data/cicids2018/raw")

old = np.load("data/processed/raw_data.npz")
old_features = set(old["columns"].tolist())

files = sorted(folder.glob("*.csv"))

if not files:
    raise SystemExit("No CSV files found!")

for file in files:

    print("\n" + "=" * 55)
    print("FILE:", file.name)

    header = pd.read_csv(file, nrows=0)
    columns = [c.strip() for c in header.columns]

    print("Total Columns:", len(columns))
    print("Timestamp Available:", "Timestamp" in columns)
    print("Label Available:", "Label" in columns)

    new_features = set(columns) - {
        "Timestamp", "Label"
    }

    common = old_features & new_features

    print("\n=== FEATURE COMPARISON ===")
    print("2017 Features:", len(old_features))
    print("2018 Candidate Features:", len(new_features))
    print("Exact Name Matches:", len(common))

    print("\n2017 Features Missing in 2018:")
    print(sorted(old_features - new_features)[:15])

    print("\n=== ATTACK LABEL DISTRIBUTION ===")

    counts = Counter()
    total = 0

    for chunk in pd.read_csv(
        file,
        usecols=lambda c: c.strip() == "Label",
        chunksize=50000,
        low_memory=False
    ):
        chunk.columns = chunk.columns.str.strip()
        labels = chunk["Label"].dropna().astype(str).str.strip()

        counts.update(labels)
        total += len(chunk)

    benign = sum(
        count for label, count in counts.items()
        if label.upper() == "BENIGN"
    )

    attacks = total - benign

    print("Total Rows:", total)
    print("Benign:", benign)
    print("Attacks:", attacks)
    print("Attack Percentage:",
          round(attacks / total * 100, 2) if total else 0)

    print("\nAttack Types:")
    for label, count in counts.most_common():
        print(label, ":", count)

print("\nDataset inspection completed!")
