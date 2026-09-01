"""
src/train.py
────────────
Reusable training loop for both Stage 3 (baseline CNN) and Stage 4
(transfer learning). The same function is called with a different model
object — all other training conditions remain identical for a fair comparison.

Design decisions
────────────────
Optimizer: Adam with lr=1e-3
  - Adam adapts the learning rate per-parameter, which makes it more
    forgiving of the learning-rate choice than SGD.  For a from-scratch CNN
    on a small imbalanced dataset, this matters — SGD with a bad LR stalls.
  - SGD + momentum can outperform Adam on very large datasets, but for
    ~7k training images Adam is the pragmatic choice.
  - lr=1e-3 is Adam's canonical default and a safe starting point.

Scheduler: ReduceLROnPlateau(patience=3, factor=0.5)
  - Halves the LR when val loss stops improving for 3 epochs.
  - This avoids manually tuning a step schedule and works well with early
    stopping: the model gets a chance to escape plateaus before we give up.

Early stopping: patience=7 epochs
  - Free Colab T4 gives ~4 hours; ~30-60 epochs is realistic.
  - patience=7 (combined with LR reduction at 3) gives the model two
    LR reductions before stopping — enough to confirm a real plateau.

Best checkpoint: saved by *val loss* (not val accuracy)
  - Val accuracy is misleading under class imbalance: a model predicting
    mostly 'nv' can have high accuracy with poor loss.
  - Val loss directly reflects the weighted CrossEntropyLoss and is more
    sensitive to whether minority classes are being learned.

Weight decay: 1e-4 (L2 regularisation)
  - Light regularisation on top of dropout to further discourage overfitting
    on the small dataset.
"""

import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim


def train(
    model,
    train_loader,
    val_loader,
    class_weights: torch.Tensor,
    device,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    checkpoint_path: str = None,
    param_groups: list = None,
):
    """
    Train a model and return the best weights (by lowest val loss).

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model whose forward() accepts (B, 3, 224, 224) tensors.
    train_loader, val_loader : DataLoader
        Outputs of make_dataloaders() from preprocessing.py.
    class_weights : torch.Tensor  shape (n_classes,)
        From compute_class_weights() — passed to CrossEntropyLoss.
    device : torch.device
        'cuda' or 'cpu'.
    num_epochs : int
        Maximum epochs. Early stopping may end training sooner.
    lr : float
        Initial learning rate for Adam. Ignored if param_groups is provided.
    weight_decay : float
        L2 regularisation coefficient.
    early_stopping_patience : int
        Stop if val loss doesn't improve for this many consecutive epochs.
    checkpoint_path : str, optional
        If provided, save the best model weights to this path.
    param_groups : list of dicts, optional
        If provided, passed directly to Adam instead of model.parameters().
        Use this for two-speed learning rates in transfer learning
        (e.g. backbone lr=1e-4, head lr=1e-3).

    Returns
    -------
    history : dict
        Keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc' — each a list
        of per-epoch values. Useful for plotting learning curves.
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # Use param_groups for two-speed LR (transfer learning), else all params
    opt_params = param_groups if param_groups is not None else model.parameters()
    optimizer  = optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss   = float("inf")
    best_weights    = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            preds          = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * images.size(0)
                preds        = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        # ── LR scheduler ─────────────────────────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ───────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>8.2e}  ({elapsed:.0f}s)")

        # ── Checkpoint best model ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_weights, checkpoint_path)
                print(f"         ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement in val loss for {early_stopping_patience} epochs.")
            break

    # Restore best weights into model
    model.load_state_dict(best_weights)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return history
