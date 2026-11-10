"""Reusable PyTorch MLP, predictions, metrics, and fine-tuning."""
import copy
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, average_precision_score, confusion_matrix)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x).flatten()


def probabilities(model, X, chunk=2048):
    model.eval()
    ans = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            xx = torch.as_tensor(X[i:i+chunk], dtype=torch.float32, device=DEVICE)
            ans.append(torch.sigmoid(model(xx)).cpu().numpy())
    return np.concatenate(ans) if ans else np.array([], dtype=float)


def score(y, probs):
    y = np.asarray(y)
    predicted = np.asarray(probs) >= 0.5
    cm = confusion_matrix(y, predicted.astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'macro_f1': float(f1_score(y, predicted, average='macro', zero_division=0)),
        'pr_auc': (float(average_precision_score(y, probs)) if len(np.unique(y)) == 2 else None),
        'attack_fpr': float(fp / (fp + tn)) if fp + tn else None,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def fit(model, X, y, *, epochs=6, lr=0.001, teacher=None, distill_weight=0.20, batch_size=512):
    """BCE + optional binary knowledge distillation from previous frozen teacher."""
    model = model.to(DEVICE)
    if teacher is not None:
        teacher = teacher.to(DEVICE).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # Keep weights modest and use both classes when possible.
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    weight = min(10., float(counts[0] / max(counts[1], 1)))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=DEVICE))
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(y))  # shuffle ONLY known labeled training data
        for start in range(0, len(order), batch_size):
            idx = order[start:start+batch_size]
            xb = torch.as_tensor(X[idx], dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(y[idx], dtype=torch.float32, device=DEVICE)
            logit = model(xb)
            supervised = criterion(logit, yb)
            if teacher is not None:
                T = 2.0
                with torch.no_grad():
                    soft_target = torch.sigmoid(teacher(xb) / T)
                distill = torch.nn.functional.binary_cross_entropy_with_logits(
                    logit / T, soft_target) * T * T
                loss = (1 - distill_weight)*supervised + distill_weight*distill
            else:
                loss = supervised
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return model


def clone(model):
    return copy.deepcopy(model).to(DEVICE)


def load_model():
    ckpt = torch.load("models/initial_mlp.pt", map_location="cpu", weights_only=True)
    model = MLP(ckpt["n_features"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    return model
