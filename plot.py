"""Step 14: visualize temporally measured quality and drift alarms."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

path = Path('results/hybrid_batches.csv')
if not path.exists():
    raise SystemExit('Run python stream.py first.')
df = pd.read_csv(path)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['batch'], df['f1'], marker='.', label='Batch attack F1')
ax.plot(df['batch'], df['macro_f1'], marker='.', label='Batch macro-F1')
for index, row in df.iterrows():
    if bool(row['mmd_warning']):
        ax.axvline(row['batch'], color='orange', alpha=.12)
    if bool(row['adaptation_accepted']):
        ax.axvline(row['batch'], color='green', alpha=.8, linestyle='--')
ax.set(xlabel='Chronological stream batch', ylabel='Score', title='Adaptive IDS: performance over time')
ax.legend()
fig.tight_layout()
fig.savefig('results/performance.png', dpi=170)
print('Saved results/performance.png')
