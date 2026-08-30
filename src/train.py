"""
src/train.py
────────────
Shared training loop used by both the baseline CNN (Stage 3) and the
transfer learning models (Stage 4).

Keeping one training loop avoids duplicating logic and makes the
baseline-vs-transfer comparison fair: identical training conditions.

STAGE STATUS: stub — implemented in Stage 3.
"""


def train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs: int, device, checkpoint_path: str = None):
    """
    Train `model` for `num_epochs`, evaluate on val_loader each epoch,
    and (optionally) save the best checkpoint to `checkpoint_path`.

    Implemented in Stage 3.
    """
    raise NotImplementedError("Implemented in Stage 3.")
