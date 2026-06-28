# train.py
# Author: Cong
# Streamlined Training Engine with Custom Profiling & Target Metrics Strategy
# Adapted for Level 5 Dual-Tower LoRA and Hybrid Temporal Modeling

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

try:
    import wandb
except ImportError:
    wandb = None

from model import Siglip2FullLoRATemporalBridge
from datasets import UCF101VideoDataset
import test

def run_train_epoch(epoch, model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch optimization pass."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Training (Epoch {epoch})")
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label_id"].to(device)
        
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "Acc": f"{100 * correct / total:.2f}%"
        })
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Level 5 VideoSiglip2 Dual-Tower Co-Alignment Optimizer")
    
    # Dataset path updates
    parser.add_argument("--base_dir", type=str, default="/root/data/UCF-101", help="Path to extracted UCF101 video folders")
    parser.add_argument("--annotation_dir", type=str, default="/root/data/ucfTrainTestlist", help="Path to UCF101 train/test split .txt files")
    parser.add_argument("--split", type=int, default=1, help="Official UCF101 split to use (1, 2, or 3)")
    
    # Standard training params
    parser.add_argument("--num_epochs", type=int, default=30, help="Total training epochs loop ceiling")
    parser.add_argument("--batch_size", type=int, default=16, help="Data layout pipeline constraints")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate applied to trainable weights")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="L2 weight normalization decay coefficient")
    parser.add_argument("--num_frames", type=int, default=8, help="Uniformly sampled frames count")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Target backup weights directory")
    parser.add_argument("--resume", type=str, default=None, help="Path parameter state file to resume progress")
    
    # Level 5 custom parameter additions
    parser.add_argument("--lora_r", type=int, default=8, help="Rank value configuration for adapter weights matrix blocks")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Scaling denominator coefficient for adapter layers mapping")
    
    args = parser.parse_args()

    # Enforce standard asset validation guards early
    if not os.path.exists(args.annotation_dir):
        raise FileNotFoundError(f"Missing core storage index alignment folder metadata at: {args.annotation_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing optimization loop target context device state: {device}")

    model_name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)

    if wandb is not None:
        wandb.init(
            project="action-siglip",
            name="level-5-co-alignment",
            config=vars(args)
        )

    print("Setting up Training Split and Data Loaders pipeline configurations...")
    train_dataset = UCF101VideoDataset(
        base_dir=args.base_dir,
        annotation_dir=args.annotation_dir,
        processor=processor,
        split=args.split,
        num_frames=args.num_frames,
        mode='train'
    )
    val_dataset = UCF101VideoDataset(
        base_dir=args.base_dir,
        annotation_dir=args.annotation_dir,
        processor=processor,
        split=args.split,
        num_frames=args.num_frames,
        mode='val'
    )

    num_workers = 0 if os.name == 'nt' else 16
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    class_names_list = train_dataset.unique_labels
    print(f"Extracted unique tokens count elements targets: {len(class_names_list)}")

    model = Siglip2FullLoRATemporalBridge(
        model_name=model_name,
        class_names=class_names_list,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Active Trainable Tensors layers identified for backpropagation processing: {len(trainable_params)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_epoch = 1
    best_val_acc = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"Restoring optimization execution checkpoint data state vectors from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)

    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f"\n--- Starting Epoch {epoch}/{args.num_epochs} ---")
        train_loss, train_acc = run_train_epoch(epoch, model, train_loader, criterion, optimizer, device)
        
        metrics = test.validate(epoch, val_loader, model, device)
        
        print(f"Epoch Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Summary   -> Top-1: {metrics['top1']:.2f}% | Top-5: {metrics['top5']:.2f}% | F1: {metrics['f1']:.2f}%")

        if wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": metrics["top1"],
                "val_top5": metrics["top5"],
                "val_f1": metrics["f1"]
            })

        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': metrics['top1']
            }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

        if metrics['top1'] > best_val_acc:
            best_val_acc = metrics['top1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, os.path.join(args.checkpoint_dir, "best_model.pt"))
            print(f"New best model saved into check-points with validation score: {best_val_acc:.2f}%")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()