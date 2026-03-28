import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.ml.predict import load_model_from_checkpoint, predict_single, compute_metrics
from src.core.ml.dataset import LithographyDataset
import src.core.ml.models  # noqa: F401


def mirror_quadrant_to_full(quadrant):
    """Mirror a bottom-right quadrant into a full symmetric illumination pattern."""
    top_half  = np.concatenate([quadrant[:, ::-1], quadrant], axis=1)
    full      = np.concatenate([top_half[::-1, :], top_half], axis=0)
    return full


def plot_predictions(model, dataset, device, n=6, save_dir=None, show=True):
    """Plot n random samples with columns: Mask | Illumination | Intensity Pred | Intensity True | Resist Pred | Resist True."""
    model.eval()
    indices     = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    col_titles  = ["Mask", "Illumination", "Intensity pred", "Intensity true", "Resist pred", "Resist true"]
    cmaps       = ["gray", "hot", "gray", "gray", "gray", "gray"]

    fig, axes = plt.subplots(n, 6, figsize=(20, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Model predictions", fontsize=14, y=1.001)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    all_pred_int, all_pred_res = [], []
    all_gt_int,   all_gt_res   = [], []

    for row, idx in enumerate(indices):
        mask, illum_q, gt_int, gt_res = dataset[idx]
        pred_int_np, pred_res_np      = predict_single(model, mask, illum_q, device)

        mask_np      = mask.squeeze().numpy()
        illum_q_np   = illum_q.squeeze().numpy()
        gt_int_np    = gt_int.squeeze().numpy()
        gt_res_np    = gt_res.squeeze().numpy()
        illum_full   = mirror_quadrant_to_full(illum_q_np)

        all_pred_int.append(pred_int_np)
        all_pred_res.append(pred_res_np)
        all_gt_int.append(gt_int_np)
        all_gt_res.append(gt_res_np)

        images = [mask_np, illum_full, pred_int_np, gt_int_np, pred_res_np, gt_res_np]
        for col, (img, cmap) in enumerate(zip(images, cmaps)):
            im = axes[row, col].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    # Print metrics over plotted samples
    metrics = compute_metrics(
        np.stack(all_pred_int), np.stack(all_pred_res),
        np.stack(all_gt_int),   np.stack(all_gt_res),
    )
    fig.text(0.01, -0.01,
             f"MSE intensity: {metrics['mse_intensity']:.6f}    BCE resist: {metrics['bce_resist']:.6f}",
             fontsize=10)

    plt.tight_layout()
    if save_dir:
        save_path = Path(save_dir) / "predictions.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    CHECKPOINT = "checkpoints/exp001_baseline/epoch_0050.pt"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(CHECKPOINT)
    model = model.to(device)

    config  = checkpoint["config"]
    dataset = LithographyDataset(config["data"]["data_dir"], split="test")

    plot_predictions(model, dataset, device, n=6, save_dir="outputs", show=True)