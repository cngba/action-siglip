# train.py
# Author: Cong
# Streamlined Training Engine with Custom Profiling & Target Metrics Strategy
# Adapted for Level 5 Dual-Tower LoRA and Hybrid Temporal Modeling
# Configured for Unified Mode Matrix Parsers (ucf101.yaml)

import os
import sys
import logging
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
import random
import numpy as np

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

def setup_logger(log_file):
    """Configures logging to record outputs to both a file and the console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def run_train_epoch(epoch, model, dataloader, criterion, optimizer, scheduler, device):
    """Runs a single training epoch optimization pass."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Initialize standard automatic mixed precision gradient scaler
    scaler = torch.amp.GradScaler()

    progress_bar = tqdm(dataloader, desc=f"Training (Epoch {epoch})", file=sys.stdout)
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["label_id"].to(device)
        
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        
        # Backward pass using the gradient scaler to prevent underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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

    # Step the learning rate scheduler once per epoch
    scheduler.step()
    return epoch_loss, epoch_acc


def main():
    import datetime
    log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Parse unified config options from CLI arguments
    parser = argparse.ArgumentParser(description="SigLIP 2 PEFT Ablation Engine")
    parser.add_argument('--config', '-cfg', default='configs/ucf101.yaml', help='Path to the unified YAML file')
    parser.add_argument('--mode', '-m', default='train_full_peft', help='Target mode matrix selection from YAML')
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Missing configuration file at: {args.config}")
        
    with open(args.config, 'r') as f:
        raw_yaml = yaml.safe_load(f)
    
    
    if "modes" not in raw_yaml or args.mode not in raw_yaml["modes"]:
        raise KeyError(f"Selected target mode configuration option '{args.mode}' not discovered in the YAML map matrix.")

    # 2. Extract shared global keys into a flat runtime dictionary configuration
    config = {
        "model_name": raw_yaml.get("model_name", "google/siglip2-base-patch16-224"),
        "seed": raw_yaml.get("seed", 1024),
        "base_dir": raw_yaml["data"]["base_dir"],
        "annotation_dir": raw_yaml["data"]["annotation_dir"],
        "split": raw_yaml["data"]["split"],
        "num_frames": raw_yaml["data"]["num_segments"],
        "num_workers": raw_yaml["data"]["workers"]
    }
    
    # 3. Dynamically merge current mode specific hyper-parameters onto the flat runtime dictionary
    mode_specific_config = raw_yaml["modes"][args.mode]
    config.update(mode_specific_config)
    
    target_log = config.get("log_file", f"logs/{args.mode}_{log_time}.log")
    logger = setup_logger(target_log)
    logger.info(f"Loaded master configuration profile from {args.config}")
    
    logger.info(f"Successfully compiled configuration vectors for operational target matrix: {args.mode}")

    # Enforce standard asset validation guards early
    if not os.path.exists(config["annotation_dir"]):
        raise FileNotFoundError(f"Missing core storage index alignment folder metadata at: {config['annotation_dir']}")

    # Set seed for repeatability checks

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing optimization loop target context device state: {device}")

    processor = AutoProcessor.from_pretrained(config["model_name"])

    # Construct execution directories safely
    run_name = f"{log_time}_{args.mode}_{config['num_frames']}f"
    run_dir = os.path.join(config["checkpoint_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Backup script layouts and configs inside unique run folders for reproducibility
    shutil.copy(args.config, os.path.join(run_dir, "ucf101_config_snapshot.yaml"))
    shutil.copy("model.py", os.path.join(run_dir, "model_arch_backup.py"))

    if wandb is not None:
        wandb.init(
            project="action-siglip2-peft",
            name=run_name,
            config=config
        )

    logger.info("Setting up Training Split and Data Loaders pipeline configurations...")
    train_dataset = UCF101VideoDataset(
        base_dir=config["base_dir"],
        annotation_dir=config["annotation_dir"],
        processor=processor,
        split=config["split"],
        num_frames=config["num_frames"],
        mode='train'
    )
    val_dataset = UCF101VideoDataset(
        base_dir=config["base_dir"],
        annotation_dir=config["annotation_dir"],
        processor=processor,
        split=config["split"],
        num_frames=config["num_frames"],
        mode='val'
    )

    g = torch.Generator()
    g.manual_seed(config["seed"])

    def worker_init_fn(worker_id):
        # Spreads a unique, non-overlapping seed sequence across all 8 parallel workers
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    num_workers = min(config["num_workers"], os.cpu_count()) if os.name != 'nt' else 0

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers, 
    pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, # <--- Pass it here
        generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers, 
    pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, # <--- Pass it here
        generator=g)

    class_names_list = train_dataset.unique_labels
    logger.info(f"Extracted unique tokens count elements targets: {len(class_names_list)}")

    # Initialize model with configuration settings directly from the flat configuration matrix mapping
    model = Siglip2FullLoRATemporalBridge(
        model_name=config["model_name"],
        class_names=class_names_list,
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        use_vision_lora=config["vision_lora"],
        use_text_lora=config["text_lora"],
        use_temporal_adapter=config["temporal_module"]
    ).to(device)

    if wandb is not None:
        wandb.watch(model, log="gradients", log_freq=100)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        if wandb is not None:
            wandb.finish()
        raise ValueError(
            f"No trainable parameters found for mode '{args.mode}'! "
            f"train.py is strictly for training. For Zero-Shot evaluation, please use test.py instead."
        )
    
    logger.info(f"Active Trainable Tensors layers identified for backpropagation processing: {len(trainable_params)}")
    logger.info("--- Parameter Audit Matrix Summary ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable: {name} | Shape: {list(param.shape)}")

    criterion = nn.CrossEntropyLoss()
    temporal_params = [p for n, p in model.named_parameters() if p.requires_grad and "temporal" in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora" in n]

    optimizer = optim.AdamW([
        {"params": lora_params, "lr": config["lr"]},
        {"params": temporal_params, "lr": config["lr"] * 2}  # Slightly higher for new weights
    ], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("scheduler_t_max", config["num_epochs"]))

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    start_epoch = 1
    best_val_acc = 0.0

    if config.get("resume") and os.path.isfile(config["resume"]):
        logger.info(f"Restoring optimization execution checkpoint data state vectors from: {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Successfully restored learning rate scheduler state configuration.")

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        logger.info(f"--- Starting Epoch {epoch}/{config['num_epochs']} ---")
        train_loss, train_acc = run_train_epoch(epoch, model, train_loader, criterion, optimizer, scheduler, device)
        
        metrics = test.validate(epoch, val_loader, model, device)
        
        logger.info(f"Epoch Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Summary   -> Top-1: {metrics['top1']:.2f}% | Top-5: {metrics['top5']:.2f}% | F1: {metrics['f1']:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']
        if wandb is not None:
            wandb.log({
                "epoch": epoch,
                "lr": current_lr,
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
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': metrics['top1']
            }, os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt"))

        if metrics['top1'] > best_val_acc:
            best_val_acc = metrics['top1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_val_acc
            }, os.path.join(config["checkpoint_dir"], "best_model.pt"))
            logger.info(f"New best model saved into check-points with validation score: {best_val_acc:.2f}%")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()