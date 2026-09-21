import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data_loader import LABEL_MAP, CLASSES


def get_target_layer(model, model_name: str):
    """Return the target conv layer for Grad-CAM by model name."""
    if model_name == "baseline":
        return [model.features[3].block[0]]
    elif model_name == "resnet18":
        return [model.layer4[1].conv2]
    elif model_name == "efficientnet_b0":
        return [model.features[8][0]]
    else:
        raise ValueError(f"Unknown model_name '{model_name}'. "
                         f"Choose from: 'baseline', 'resnet18', 'efficientnet_b0'")


def visualize_gradcam(model, model_name: str, image_tensor: torch.Tensor,
                      true_label: int, pred_label: int,
                      ax=None, title_prefix: str = "") -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay for a single image."""
    target_layers = get_target_layer(model, model_name)
    device        = next(model.parameters()).device

    model.eval()
    input_tensor = image_tensor.unsqueeze(0).to(device)

    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    rgb_img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    rgb_img = std * rgb_img + mean
    rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets       = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]

    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    if ax is not None:
        correct   = (true_label == pred_label)
        true_name = LABEL_MAP[CLASSES[true_label]]
        pred_name = LABEL_MAP[CLASSES[pred_label]]
        color     = "green" if correct else "red"
        status    = "✓" if correct else "✗"
        ax.imshow(overlay)
        ax.set_title(
            f"{title_prefix}\n"
            f"True: {true_name}\n"
            f"Pred: {pred_name} {status}",
            fontsize=7, color=color
        )
        ax.axis("off")

    return overlay


def gradcam_grid(model, model_name: str, samples: list,
                 title: str = "", save_path: str = None) -> None:
    """Generate a grid of Grad-CAM overlays for a list of images."""
    import os
    n = len(samples)
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(title, fontsize=11, y=1.01)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i, s in enumerate(samples):
        img_t      = s["image_tensor"]
        true_label = s["true_label"]
        pred_label = s["pred_label"]
        img_id     = s.get("image_id", "")

        rgb = img_t.cpu().numpy().transpose(1, 2, 0)
        rgb = std * rgb + mean
        rgb = np.clip(rgb, 0, 1)

        correct   = (true_label == pred_label)
        true_name = LABEL_MAP[CLASSES[true_label]]
        pred_name = LABEL_MAP[CLASSES[pred_label]]
        color     = "green" if correct else "red"
        status    = "✓" if correct else "✗"

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Original\n{img_id}", fontsize=7)
        axes[i, 0].axis("off")

        visualize_gradcam(
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


def gradcam_model_comparison(models_dict: dict, samples: list,
                              title: str = "", save_path: str = None) -> None:
    """Show Grad-CAM overlays from multiple models side by side for each sample."""
    import os
    model_names = list(models_dict.keys())
    n_models    = len(model_names)
    n_samples   = len(samples)

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

        rgb = img_t.cpu().numpy().transpose(1, 2, 0)
        rgb = std * rgb + mean
        rgb = np.clip(rgb, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(
            f"Original\n{LABEL_MAP[CLASSES[true_label]]}\n{img_id}",
            fontsize=7
        )
        axes[i, 0].axis("off")

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
