import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.core.ml.predict import load_model_from_checkpoint, predict_single, compute_metrics
from src.core.ml.dataset import LithographyDataset
import src.core.simulator.illuminator as illuminator
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.ml.models  # noqa: F401


def plot_predictions(model, dataset, device, n=6, save_dir=None, show=True):
    """Plot n random samples with columns: Mask | Illumination | Intensity Pred | Intensity True | Resist Pred | Resist True."""
    model.eval()

    indices    = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    col_titles = ["Mask", "Illumination", "Intensity pred", "Intensity true", "Resist pred", "Resist true"]
    col_fields = ["mask", "illumination", "wafer_intensity", "wafer_intensity", "resist_profile", "resist_profile"]

    fig, axes = plt.subplots(n, 6, figsize=(6 * 4, n * 2.25))
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Model predictions", fontsize=14, y=1.001)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    all_pred_int, all_pred_res = [], []
    all_gt_int,   all_gt_res   = [], []

    for row, idx in enumerate(indices):
        mask, illum_q, gt_int, gt_res = dataset[idx]
        pred_int_np, pred_res_np      = predict_single(model, mask, illum_q, device)

        mask_np    = mask.squeeze().numpy()
        illum_np   = illuminator.upsample_illumination(illuminator.quadrant_to_full(illum_q.squeeze().numpy()), target_size=mask_np.shape[0])
        gt_int_np  = gt_int.squeeze().numpy()
        gt_res_np  = gt_res.squeeze().numpy()

        all_pred_int.append(pred_int_np)
        all_pred_res.append(pred_res_np)
        all_gt_int.append(gt_int_np)
        all_gt_res.append(gt_res_np)

        images = [mask_np, illum_np, pred_int_np, gt_int_np, pred_res_np, gt_res_np]
        for col, (img, field) in enumerate(zip(images, col_fields)):
            cfg = simulation_visualizer.FIELD_CONFIG[field]
            ax  = axes[row, col]
            simulation_visualizer._plot_field(ax, img, cfg, title="" , extent=[-500, 500, -500, 500])

    metrics = compute_metrics(
        np.stack(all_pred_int), np.stack(all_pred_res),
        np.stack(all_gt_int),   np.stack(all_gt_res),
    )
    fig.text(
        0.01, -0.01,
        f"MSE intensity: {metrics['mse_intensity']:.6f}    BCE resist: {metrics['bce_resist']:.6f}",
        fontsize=10
    )

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
    CHECKPOINT = "checkpoints/exp008_baseline/epoch_0050.pt"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(CHECKPOINT)
    model = model.to(device)

    config  = checkpoint["config"]
    dataset = LithographyDataset(config["data"]["data_dir"], split="train")

    while True:
        plot_predictions(model, dataset, device, n=4, save_dir="outputs", show=True)