"""A compact PyTorch CNN for spatial EEG images.

Input  : (batch, F, H, W) float32 - F feature channels (here mu / low-beta / high-beta band power),
         H x W pixels on the scalp grid (default 16 x 16).
Output : (batch, 2) logits for [left hand, right hand]. Logits are NOT probabilities; we use the
         difference logit_right - logit_left as a score (> 0 means "right").

Relation to EEGLearn
    Kept: VGG-like 3x3 convolution stacks with max-pooling, a dense layer with dropout 0.5, Adam
    (lr 1e-3) and an L2 penalty of 1e-4 (implemented as weight decay).
    Changed: smaller network (fewer filters, input is 16 x 16 not 32 x 32), batch normalisation,
    no recurrent / multi-window branch (the EEGLearn recurrent-convolutional model needs a
    sequence of images per trial; this project uses a single window per trial), binary output.
    This is NOT a reproduction of the paper's architecture.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import balanced_accuracy_simple


class ModelInputError(ValueError):
    """Raised for invalid model inputs (shape, labels, non-finite values)."""


class EEGSpatialCNN(nn.Module):
    def __init__(self, in_channels: int = 3, grid_size: int = 16, channels=(16, 32, 64),
                 dropout: float = 0.5, n_classes: int = 2, hidden: int = 64):
        super().__init__()
        if in_channels < 1 or grid_size < 4 or n_classes < 2:
            raise ModelInputError("in_channels >= 1, grid_size >= 4 and n_classes >= 2 are required")
        layers: list[nn.Module] = []
        prev, size = in_channels, grid_size
        for i, ch in enumerate(channels):
            layers += [nn.Conv2d(prev, ch, kernel_size=3, padding=1), nn.BatchNorm2d(ch), nn.ELU()]
            if i >= 1 and size >= 4:          # pool after every block except the first
                layers.append(nn.MaxPool2d(2))
                size //= 2
            prev = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev * size * size, hidden), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes),
        )
        self.init_args = dict(in_channels=in_channels, grid_size=grid_size, channels=list(channels),
                              dropout=dropout, n_classes=n_classes, hidden=hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _check_inputs(X: np.ndarray, y: np.ndarray | None, in_channels: int | None = None) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 4:
        raise ModelInputError(f"X must be a 4-D array (trials, feature channels, H, W); got {getattr(X, 'shape', None)}")
    if X.shape[0] == 0:
        raise ModelInputError("X contains no trials")
    if in_channels is not None and X.shape[1] != in_channels:
        raise ModelInputError(f"Model expects {in_channels} feature channels, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ModelInputError("X contains NaN or infinite values")
    if y is not None:
        y = np.asarray(y)
        if y.shape != (X.shape[0],):
            raise ModelInputError(f"y must have shape ({X.shape[0]},); got {y.shape}")
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ModelInputError(f"Labels must be 0 or 1; found {sorted(set(np.unique(y).tolist()))}")


def predict_scores(model: EEGSpatialCNN, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Score = logit(right) - logit(left), computed in eval mode (dropout off, BN uses running stats)."""
    _check_inputs(X, None, model.init_args["in_channels"])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.from_numpy(X[i:i + batch_size].astype(np.float32)))
            out.append((logits[:, 1] - logits[:, 0]).numpy())
    return np.concatenate(out)


def train_cnn(X_train, y_train, X_val, y_val, cfg, seed: int = 0, verbose: bool = False):
    """Train with Adam; select the epoch with the best VALIDATION balanced accuracy (ties: lower loss).

    The test set is never seen here. Returns (model with best weights restored, history dict).
    """
    _check_inputs(X_train, y_train)
    _check_inputs(X_val, y_val, X_train.shape[1])
    if len(np.unique(y_train)) < 2:
        raise ModelInputError("Training labels contain a single class")
    c = cfg.models["cnn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = EEGSpatialCNN(X_train.shape[1], X_train.shape[2], tuple(c["channels"]), c["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    Xv = torch.from_numpy(X_val.astype(np.float32))
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.int64))

    best = {"score": -1.0, "loss": np.inf, "state": None, "epoch": -1}
    history = {"train_loss": [], "val_loss": [], "val_balanced_accuracy": []}
    bad_epochs = 0
    for epoch in range(c["max_epochs"]):
        model.train()
        order = rng.permutation(len(Xt))
        total, n_batches = 0.0, 0
        for i in range(0, len(order), c["batch_size"]):
            idx = torch.from_numpy(order[i:i + c["batch_size"]])
            if len(idx) < 2:          # BatchNorm cannot train on a single sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = float(loss_fn(logits, yv))
            val_pred = logits.argmax(1).numpy()
        val_score = balanced_accuracy_simple(np.asarray(y_val), val_pred)
        history["train_loss"].append(total / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_balanced_accuracy"].append(val_score)
        if verbose:
            print(f"epoch {epoch + 1:3d} train {history['train_loss'][-1]:.4f} val {val_loss:.4f} bacc {val_score:.3f}")
        if (val_score > best["score"] + 1e-9) or (abs(val_score - best["score"]) <= 1e-9 and val_loss < best["loss"]):
            best.update(score=val_score, loss=val_loss, state=copy.deepcopy(model.state_dict()), epoch=epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= c["patience"]:
                break
    model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"] + 1
    history["best_val_balanced_accuracy"] = best["score"]
    return model, history


def save_cnn(directory: str | Path, model: EEGSpatialCNN, metadata: dict) -> None:
    """Save weights (``cnn.pt``, a plain state_dict) + JSON metadata needed to rebuild and use the model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "cnn.pt")
    meta = dict(metadata, architecture=model.init_args)
    (directory / "cnn_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_cnn(directory: str | Path) -> tuple[EEGSpatialCNN, dict]:
    """Load a model saved by ``save_cnn``. Uses ``weights_only=True`` so no arbitrary code can run."""
    directory = Path(directory)
    meta_path, weights_path = directory / "cnn_meta.json", directory / "cnn.pt"
    if not meta_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"No CNN artifact in {directory} (need cnn.pt and cnn_meta.json)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    a = meta["architecture"]
    model = EEGSpatialCNN(a["in_channels"], a["grid_size"], tuple(a["channels"]), a["dropout"], a["n_classes"], a["hidden"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, meta
