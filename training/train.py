"""
Training script for MTG card embedding model.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import create_dataloaders
from model import CardEmbeddingModel, count_parameters


def train_epoch(model, loader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast(enabled=device.type == "cuda"):
            logits = model(images, labels)
            loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.1f}%"
        })

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validating"):
        images = images.to(device)
        labels = labels.to(device)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images, labels)
            loss = nn.CrossEntropyLoss()(logits, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MTG card embedding model")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data" / "images")
    parser.add_argument("--metadata", type=Path, default=Path(__file__).parent / "data" / "cards_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Number of classes: {num_classes}")

    # Create model
    print("\nCreating model...")
    model = CardEmbeddingModel(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        pretrained=True,
    )
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%")

        # Update learning rate
        scheduler.step()
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = args.output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes,
                "embedding_dim": args.embedding_dim,
            }, checkpoint_path)
            print(f"Saved best model (val_acc: {100*val_acc:.2f}%)")

    # Save final model
    final_path = args.output_dir / "final_model.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "embedding_dim": args.embedding_dim,
    }, final_path)

    # Save training history
    history_path = args.output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Generate embeddings: python generate_embeddings.py")
    print("  2. Export to ONNX: python export_onnx.py")


if __name__ == "__main__":
    main()
