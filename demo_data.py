"""Create SYNTHETIC labeled flows to smoke-test the code before CICIDS2017.
Not a research result or substitute for real attacks.
"""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 11000
n_features = 16
X = rng.normal(0, 1, (n, n_features)).astype('float32')
X[8400:] += np.array([1.2, -0.7, 1.0, -0.8] + [0.0] * 12)
y = (X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] + rng.normal(0, 0.55, n) > 0.5)
y[8400:] = (X[8400:, 0] - X[8400:, 1] + 0.7*X[8400:, 2]
            + rng.normal(0, 0.55, n-8400) > 0.4)
t = pd.date_range('2017-07-03 08:00:00', periods=n, freq='s')
df = pd.DataFrame(X, columns=[f'Feature_{i:02d}' for i in range(n_features)])
df['Timestamp'] = t.strftime('%d/%m/%Y %H:%M:%S')
df['Label'] = np.where(y, 'DoS', 'BENIGN')
folder = Path('data/demo')
folder.mkdir(parents=True, exist_ok=True)
df.to_csv(folder / 'synthetic_flows.csv', index=False)
print('Created synthetic example:', folder / 'synthetic_flows.csv')
print('Controlled concept change begins near row 8400 (this is only a smoke test).')
