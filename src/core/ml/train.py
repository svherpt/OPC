import torch
import torch.nn as nn
import wandb
import argparse
from pathlib import Path
from tqdm import tqdm

from src.core.ml.config import load_config
from src.core.ml.dataset import build_dataloaders
from src.core.ml.registry import build_model
import src.core.ml.models  # noqa: F401 — triggers model registration


def compute_loss(pred_intensity, pred_resist, gt_intensity, gt_resist, lambda_resist):
    """Compute combined MSE loss on intensity and BCE loss on resist."""
    loss_intensity = nn.functional.mse_loss(pred_intensity, gt_intensity)
    loss_resist    = nn.functional.binary_cross_entropy(pred_resist, gt_resist)
    return loss_intensity + lambda_resist * loss_resist, loss_intensity, loss_resist


def run_epoch(model, loader, optimizer, lambda_resist, training, device):
    """Run one full epoch, returning mean total, intensity and resist losses."""
    model.train() if training else model.eval()

    total_loss     = 0.0
    total_int_loss = 0.0
    total_res_loss = 0.0

    desc    = "train" if training else "val"
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc=f"  {desc}", leave=False, dynamic_ncols=True, disable=True)
        for mask, illum_q, intensity, resist in pbar:
            mask, illum_q, intensity, resist = (
                mask.to(device), illum_q.to(device),
                intensity.to(device), resist.to(device)
            )
            pred_intensity, pred_resist = model(mask, illum_q)
            loss, int_loss, res_loss = compute_loss(
                pred_intensity, pred_resist, intensity, resist, lambda_resist
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss     += loss.item()
            total_int_loss += int_loss.item()
            total_res_loss += res_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "int":  f"{int_loss.item():.4f}",
                "res":  f"{res_loss.item():.4f}",
            })

    n = len(loader)
    return total_loss / n, total_int_loss / n, total_res_loss / n


def save_run_id(checkpoint_dir, run_id):
    """Save wandb run ID to file so it can be recovered on resume."""
    with open(checkpoint_dir / "wandb_run_id.txt", "w") as f:
        f.write(run_id)


def load_run_id(checkpoint_dir):
    """Load wandb run ID from file if it exists."""
    path = checkpoint_dir / "wandb_run_id.txt"
    if path.exists():
        return path.read_text().strip()
    return None


def load_latest_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    """Load the most recent checkpoint, returning the next epoch and wandb run id."""
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        return 1, None

    path       = checkpoints[-1]
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    run_id      = load_run_id(checkpoint_dir) or checkpoint.get("wandb_run_id")

    print(f"Resumed from {path.name} (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    return start_epoch, run_id


def train(config_path, resume=False):
    """Run full training loop from a YAML config path, optionally resuming from latest checkpoint."""
    config = load_config(config_path)
    tcfg   = config["training"]
    wcfg   = config["wandb"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader = build_dataloaders(config)
    model     = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    checkpoint_dir = Path("checkpoints") / wcfg["run_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch   = 1
    resume_run_id = None
    if resume:
        start_epoch, resume_run_id = load_latest_checkpoint(
            checkpoint_dir, model, optimizer, scheduler, device
        )
        if start_epoch > tcfg["epochs"]:
            print("Training already complete.")
            return

    run = wandb.init(
        project=wcfg["project"],
        name=wcfg["run_name"],
        id=resume_run_id,
        resume="allow" if resume else None,
        config={
            "model":         config["model"]["name"],
            "channels":      config["model"].get("channels", 16),
            "illum_dim":     config["model"].get("illum_dim", 128),
            "lr":            tcfg["lr"],
            "epochs":        tcfg["epochs"],
            "lambda_resist": tcfg["lambda_resist"],
            "batch_size":    config["data"]["batch_size"],
            "device":        str(device),
        }
    )
    save_run_id(checkpoint_dir, run.id)

    epoch_pbar = tqdm(
        range(start_epoch, tcfg["epochs"] + 1),
        desc="Epochs", dynamic_ncols=True,
        initial=start_epoch - 1, total=tcfg["epochs"]
    )
    for epoch in epoch_pbar:
        train_loss, train_int, train_res = run_epoch(
            model, train_loader, optimizer, tcfg["lambda_resist"],
            training=True, device=device
        )
        val_loss, val_int, val_res = run_epoch(
            model, test_loader, optimizer, tcfg["lambda_resist"],
            training=False, device=device
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log({
            "train_loss":     train_loss,
            "train_int_loss": train_int,
            "train_res_loss": train_res,
            "val_loss":       val_loss,
            "val_int_loss":   val_int,
            "val_res_loss":   val_res,
            "lr":             current_lr,
        }, step=epoch)

        torch.save({
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "val_loss":     val_loss,
            "config":       config,
            "wandb_run_id": run.id,
        }, checkpoint_dir / f"epoch_{epoch:04d}.pt")

        epoch_pbar.set_postfix({
            "train": f"{train_loss:.4f}",
            "val":   f"{val_loss:.4f}",
            "lr":    f"{current_lr:.2e}",
        })

    wandb.finish()
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/exp001_baseline.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    train(args.config, resume=args.resume)