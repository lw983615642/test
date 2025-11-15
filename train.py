import argparse
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.polar_dataset import PolarReflectionDataset
from src.models.polar_reflection_net import PolarReflectionRemovalNet
from src.modules.losses import SSIMLoss, gradient_loss, reflection_consistency_loss
from src.utils.train_utils import MetricTracker, count_parameters, save_checkpoint, seed_everything, torch_autocast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Polarized Reflection Removal network")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to glass input images")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to ground truth images")
    parser.add_argument("--reflection_dir", type=str, required=True, help="Path to reflection maps")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def create_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    dataset = PolarReflectionDataset(
        args.input_dir,
        args.target_dir,
        args.reflection_dir,
        augment=True,
    )
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    val_size = max(1, int(num_samples * args.val_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(dataset, train_indices.tolist())
    val_dataset = Subset(
        PolarReflectionDataset(
            args.input_dir,
            args.target_dir,
            args.reflection_dir,
            augment=False,
        ),
        val_indices.tolist(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    ssim_loss: SSIMLoss,
    use_amp: bool,
    log_interval: int,
) -> MetricTracker:
    model.train()
    tracker = MetricTracker()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        reflections = batch["reflection"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch_autocast(device.type, enabled=use_amp):
            clean, enhanced, predicted_reflection = model(inputs, reflections)
            loss_l1 = F.l1_loss(clean, targets)
            loss_ssim = ssim_loss(clean, targets)
            loss_grad = gradient_loss(clean, targets)
            loss_enh = F.l1_loss(enhanced, targets)
            loss_reflection = reflection_consistency_loss(predicted_reflection, reflections)
            loss = loss_l1 + 0.4 * loss_ssim + 0.1 * loss_grad + 0.05 * loss_reflection + 0.2 * loss_enh

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        tracker.update("loss", float(loss.item()), inputs.size(0))
        tracker.update("l1", float(loss_l1.item()), inputs.size(0))
        tracker.update("psnr", float(psnr(clean.detach(), targets).item()), inputs.size(0))

        if batch_idx % log_interval == 0:
            tqdm.write(
                f"Batch {batch_idx}: loss={loss.item():.4f}, L1={loss_l1.item():.4f}, PSNR={tracker.average('psnr'):.2f}"
            )

    return tracker


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ssim_loss: SSIMLoss,
    use_amp: bool,
) -> MetricTracker:
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Val", leave=False):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            reflections = batch["reflection"].to(device)

            with torch_autocast(device.type, enabled=use_amp):
                clean, enhanced, predicted_reflection = model(inputs, reflections)
                loss_l1 = F.l1_loss(clean, targets)
                loss_ssim = ssim_loss(clean, targets)
                loss_grad = gradient_loss(clean, targets)
                loss_enh = F.l1_loss(enhanced, targets)
                loss_reflection = reflection_consistency_loss(predicted_reflection, reflections)
                loss = loss_l1 + 0.4 * loss_ssim + 0.1 * loss_grad + 0.05 * loss_reflection + 0.2 * loss_enh

            tracker.update("loss", float(loss.item()), inputs.size(0))
            tracker.update("l1", float(loss_l1.item()), inputs.size(0))
            tracker.update("psnr", float(psnr(clean, targets).item()), inputs.size(0))
    return tracker


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(args)

    model = PolarReflectionRemovalNet().to(device)
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ssim_loss = SSIMLoss(channel=1).to(device)

    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = checkpoint.get("epoch", 0)
        best_val = checkpoint.get("best_val", best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            ssim_loss,
            args.amp,
            args.log_interval,
        )
        val_metrics = validate(model, val_loader, device, ssim_loss, args.amp)
        scheduler.step()

        train_avg = train_metrics.averages()
        val_avg = val_metrics.averages()

        print(
            f"Train Loss: {train_avg['loss']:.4f}, Val Loss: {val_avg['loss']:.4f}, "
            f"Val PSNR: {val_avg['psnr']:.2f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        is_best = val_avg["loss"] < best_val
        if is_best:
            best_val = val_avg["loss"]

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }
        save_checkpoint(checkpoint, is_best=is_best, checkpoint_dir=args.output_dir, filename=f"epoch_{epoch+1:04d}.pth")

    print("Training finished. Best validation loss:", best_val)


if __name__ == "__main__":
    main()
