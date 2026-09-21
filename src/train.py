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

    Returns history dict with keys: train_loss, val_loss, train_acc, val_acc.
    Pass param_groups for two-speed learning rates (transfer learning).
    """
    model = model.to(device)

    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt_params = param_groups if param_groups is not None else model.parameters()
    optimizer  = optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss     = float("inf")
    best_weights      = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Training
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

        # Validation
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

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>8.4f} | {val_acc:>6.2%} | {current_lr:>8.2e}  ({elapsed:.0f}s)")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_weights      = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(best_weights, checkpoint_path)
                print(f"         ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch} — no improvement for {early_stopping_patience} epochs.")
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return history
