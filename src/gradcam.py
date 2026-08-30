"""
src/gradcam.py
──────────────
Grad-CAM visualizations using the pytorch-grad-cam library.
Applied to both the baseline CNN and the best transfer model to show
which image regions drive each prediction.

We use the `pytorch-grad-cam` library rather than implementing Grad-CAM
from scratch because:
  (a) its implementation is battle-tested and handles edge cases correctly,
  (b) it is free, open-source, and supports multiple CAM methods in one API,
  (c) re-implementing the matrix calculus from scratch adds complexity
      without adding portfolio value — the insight is in the interpretation,
      not the backprop bookkeeping.

STAGE STATUS: stub — implemented in Stage 5.
"""


def visualize_gradcam(model, target_layer, image_tensor,
                      true_label: int, pred_label: int,
                      class_names: list, save_path: str = None):
    """Generate and display a Grad-CAM overlay for one image. Implemented in Stage 5."""
    raise NotImplementedError("Implemented in Stage 5.")
