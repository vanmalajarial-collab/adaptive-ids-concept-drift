"""Bounded class-balanced experience replay across historical periods."""
import numpy as np

class BalancedReservoir:
    """Bounded, per-class reservoir over past labeled data (retains old periods)."""
    def __init__(self, capacity_per_class=250, seed=42):
        self.cap = capacity_per_class
        self.rng = np.random.default_rng(seed)
        self.items = {0: [], 1: []}
        self.counts = {0: 0, 1: 0}

    def add(self, X, y):
        for x, label in zip(X, y):
            label = int(label)
            self.counts[label] += 1
            buf = self.items[label]
            if len(buf) < self.cap:
                buf.append(np.array(x, dtype=np.float32, copy=True))
            else:
                chosen = int(self.rng.integers(0, self.counts[label]))
                if chosen < self.cap:
                    buf[chosen] = np.array(x, dtype=np.float32, copy=True)

    def sample(self, n=128):
        pairs = []
        for label in (0, 1):
            buf = self.items[label]
            if not buf:
                continue
            ids = self.rng.choice(len(buf), size=min(n//2, len(buf)), replace=False)
            pairs.extend((buf[i], label) for i in ids)
        if not pairs:
            return None, None
        self.rng.shuffle(pairs)
        return np.stack([x for x, _ in pairs]), np.array([y for _, y in pairs])
