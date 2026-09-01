"""
src/gradcam.py
──────────────
Grad-CAM visualizations using the pytorch-grad-cam library.

What Grad-CAM shows (mathematically)
──────────────────────────────────────
Grad-CAM computes the gradient of the class score y^c with respect to the
feature maps A^k of a target convolutional layer:

    α_k^c  =  (1/Z) Σ_{i,j} ∂y^c / ∂A^k_{ij}       (global average of gradients)

    L^c_Grad-CAM  =  ReLU( Σ_k α_k^c · A^k )         (weighted sum, ReLU clips negatives)

The ReLU ensures we only highlight regions that INCREASE the class score.
The result is a coarse heatmap (same spatial size as the target layer's
feature maps) that is upsampled to the original image resolution and overlaid.

Why pytorch-grad-cam (not from scratch)?
  The library is battle-tested, handles edge cases (e.g. batch norm in eval
  mode, inplace ReLU), and supports EigenCAM, GradCAM++, etc. in one API.
  The portfolio value here is in the interpretation, not reimplementing backprop.

Target layer selection — why these layers?
──────────────────────────────────────────
The target layer must be:
  - Late enough: has learned semantically meaningful features (not raw edges)
  - Early enough: still has spatial resolution > 1×1 (GAP collapses to 1×1)

  BaselineCNN   → features[3].block[0]   (last Conv2d before MaxPool in Block 4)
    Spatial size: 28×28 → after maxpool: 14×14
    Features: highest-level semantic features before GAP collapses them.
    We target the Conv before MaxPool so we retain 28×28 spatial resolution.

  ResNet18      → layer4[1].conv2        (last conv in the last residual block)
    Spatial size: 7×7 (standard ResNet18 feature map at the end)
    This is the conventional target for ResNet Grad-CAM in the literature.

  EfficientNet-B0 → features[8][0]       (last MBConv block's first conv)
    Spatial size: 7×7
    features[8] is the last MBConv stage before the global pooling head.
    Targeting its first conv gives the best balance of semantics and resolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data_loader import LABEL_MAP, CLASSES


# ── Target layer registry ─────────────────────────────────────────────────────

def get_target_layer(model, model_name: str):
    """
    Return the correct target convolutional layer for Grad-CAM, given the
    model architecture.

    Parameters
    ----------
    model      : nn.Module  — the trained model
    model_name : str        — one of 'baseline', 'resnet18', 'efficientnet_b0'

    Returns
    -------
    list containing the single target layer (pytorch-grad-cam expects a list)
    """
    if model_name == "baseline":
        # Last conv in block 4, before MaxPool — retains 28×28 spatial resolution
        return [model.features[3].block[0]]
    elif model_name == "resnet18":
        # Last conv in last residual block — 7×7, conventional target
        return [model.layer4[1].conv2]
    elif model_name == "efficientnet_b0":
        # Last MBConv block's first conv — 7×7
        return [model.features[8][0]]
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Choose from: 'baseline', 'resnet18', 'efficientnet_b0'")


# ── Single image Grad-CAM ─────────────────────────────────────────────────────

def visualize_gradcam(model, model_name: str, image_tensor: torch.Tensor,
                      true_label: int, pred_label: int,
                      ax=None, title_prefix: str = "") -> np.ndarray:
    """
    Generate a Grad-CAM heatmap overlay for a single image.

    Parameters
    ----------
    model        : nn.Module       — trained model in eval mode, on correct device
    model_name   : str             — 'baseline', 'resnet18', or 'efficientnet_b0'
    image_tensor : torch.Tensor    — shape (3, 224, 224), ImageNet-normalised
    true_label   : int             — ground-truth class index
    pred_label   : int             — model's predicted class index
    ax           : matplotlib Axes — if provided, draws onto this axes
    title_prefix : str             — prepended to the axes title

    Returns
    -------
    np.ndarray  — shape (224, 224, 3) float32 RGB overlay image [0, 1]
    """
    target_layers = get_target_layer(model, model_name)
    device = next(model.parameters()).device

    # pytorch-grad-cam expects the model in eval mode
    model.eval()

    input_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Denormalise for overlay display (Grad-CAM needs the original pixel values)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    rgb_img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Target the predicted class — shows what drove THIS prediction
        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # (224, 224)

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if ax is not None:
        correct = (true_label == pred_label)
        true_name = LABEL_MAP[CLASSES[true_label]]
        pred_name = LABEL_MAP[CLASSES[pred_label]]
        color = "green" if correct else "red"
        status = "✓" if correct else "✗"
        ax.imshow(overlay)
        ax.set_title(
            f"{title_prefix}\n"
            f"True: {true_name}\n"
            f"Pred: {pred_name} {status}",
            fontsize=7, color=color
        )
        ax.axis("off")

    return overlay


# ── Grid: Grad-CAM for multiple images ───────────────────────────────────────

def gradcam_grid(model, model_name: str, samples: list,
                 title: str = "", save_path: str = None) -> None:
    """
    Generate a grid of Grad-CAM overlays for a list of images.

    Parameters
    ----------
    model      : nn.Module
    model_name : str — 'baseline', 'resnet18', or 'efficientnet_b0'
    samples    : list of dicts, each with keys:
                   'image_tensor' : torch.Tensor (3, 224, 224)
                   'true_label'   : int
                   'pred_label'   : int
                   'image_id'     : str  (optional, for subplot title)
    title      : str — figure suptitle
    save_path  : str — if provided, saves figure to this path
    """
    import os
    n = len(samples)
    # Two columns: original | Grad-CAM overlay
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing works for n=1

    fig.suptitle(title, fontsize=11, y=1.01)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, s in enumerate(samples):
        img_t      = s["image_tensor"]
        true_label = s["true_label"]
        pred_label = s["pred_label"]
        img_id     = s.get("image_id", "")

        # Denormalise for display
        rgb = img_t.cpu().numpy().transpose(1, 2, 0)
        rgb = std * rgb + mean
        rgb = np.clip(rgb, 0, 1)

        correct   = (true_label == pred_label)
        true_name = LABEL_MAP[CLASSES[true_label]]
        pred_name = LABEL_MAP[CLASSES[pred_label]]
        color     = "green" if correct else "red"
        status    = "✓" if correct else "✗"

        # Left: original image
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Original\n{img_id}", fontsize=7)
        axes[i, 0].axis("off")

        # Right: Grad-CAM overlay
        overlay = visualize_gradcam(
            model, model_name, img_t,
            true_label, pred_label,
            ax=axes[i, 1],
            title_prefix=img_id,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()


# ── Side-by-side across models ────────────────────────────────────────────────

def gradcam_model_comparison(models_dict: dict, samples: list,
                              title: str = "", save_path: str = None) -> None:
    """
    For each sample, show the Grad-CAM overlay from multiple models side by side.
    Useful for comparing what the baseline CNN vs. transfer model "looks at".

    Parameters
    ----------
    models_dict : dict  {model_name: model}  — e.g. {'baseline': m1, 'resnet18': m2}
    samples     : list of dicts (same format as gradcam_grid)
    title       : str
    save_path   : str
    """
    import os
    model_names = list(models_dict.keys())
    n_models    = len(model_names)
    n_samples   = len(samples)

    # Columns: original + one per model
    n_cols = 1 + n_models
    fig, axes = plt.subplots(n_samples, n_cols,
                             figsize=(3 * n_cols, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=11, y=1.01)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, s in enumerate(samples):
        img_t      = s["image_tensor"]
        true_label = s["true_label"]
        pred_label = s["pred_label"]
        img_id     = s.get("image_id", "")

        # Column 0: original
        rgb = img_t.cpu().numpy().transpose(1, 2, 0)
        rgb = std * rgb + mean
        rgb = np.clip(rgb, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"Original\n{LABEL_MAP[CLASSES[true_label]]}\n{img_id}",
            fontsize=7
        )
        axes[i, 0].axis("off")

        # Remaining columns: one per model
        for j, mname in enumerate(model_names):
            visualize_gradcam(
                models_dict[mname], mname, img_t,
                true_label, pred_label,
                ax=axes[i, j + 1],
                title_prefix=mname,
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()
