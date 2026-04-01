# src/core/ml/predict.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from src.core.ml.registry import build_model
import src.core.ml.models  # noqa: F401


def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint path, using the config stored inside the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model      = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def predict_single(model, mask, illum_q, device):
    """Run inference on a single sample, returning (pred_intensity, pred_resist) as numpy arrays."""
    model.eval()
    with torch.no_grad():
        mask_batch  = mask.unsqueeze(0).to(device)
        illum_batch = illum_q.unsqueeze(0).to(device)
        pred_int, pred_res = model(mask_batch, illum_batch)
    return pred_int.squeeze().cpu().numpy(), pred_res.squeeze().cpu().numpy()


def predict_batch(model, dataloader, device):
    """Run inference over a full dataloader, returning predictions and targets as numpy arrays."""
    model.eval()
    all_pred_int, all_pred_res = [], []
    all_gt_int,   all_gt_res   = [], []

    with torch.no_grad():
        for mask, illum_q, intensity, resist in dataloader:
            mask, illum_q = mask.to(device), illum_q.to(device)
            pred_int, pred_res = model(mask, illum_q)

            all_pred_int.append(pred_int.cpu().numpy())
            all_pred_res.append(pred_res.cpu().numpy())
            all_gt_int.append(intensity.numpy())
            all_gt_res.append(resist.numpy())

    return (
        np.concatenate(all_pred_int),
        np.concatenate(all_pred_res),
        np.concatenate(all_gt_int),
        np.concatenate(all_gt_res),
    )


def compute_metrics(pred_int, pred_res, gt_int, gt_res):
    """Compute mean MSE on intensity and BCE on resist, returning a metrics dict."""
    mse = float(np.mean((pred_int - gt_int) ** 2))

    # Clip predictions to avoid log(0) in BCE
    eps       = 1e-7
    pred_res  = np.clip(pred_res, eps, 1 - eps)
    bce       = float(-np.mean(gt_res * np.log(pred_res) + (1 - gt_res) * np.log(1 - pred_res)))

    return {"mse_intensity": mse, "bce_resist": bce}


if __name__ == "__main__":
    from src.core.ml.dataset import LithographyDataset
    from torch.utils.data import DataLoader

    CHECKPOINT = "checkpoints/exp020/epoch_0100.pt"
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model_from_checkpoint(CHECKPOINT)
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss {checkpoint['val_loss']:.4f})")

    config   = checkpoint["config"]
    dataset  = LithographyDataset(config["data"]["data_dir"], split="test")
    loader   = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    pred_int, pred_res, gt_int, gt_res = predict_batch(model, loader, device)
    metrics = compute_metrics(pred_int, pred_res, gt_int, gt_res)

    print(f"\nTest metrics:")
    print(f"  MSE intensity : {metrics['mse_intensity']:.6f}")
    print(f"  BCE resist    : {metrics['bce_resist']:.6f}")