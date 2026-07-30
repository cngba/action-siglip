# action-siglip/train.py
# Author: Cong

"""
Main training script for Siglip2 action recognition experiments.

This file wires configuration, data loading, model initialization,
the training loop, validation calls, checkpointing and optional
experiment logging (Weights & Biases).
"""

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
import time
import datetime
from peft import LoraConfig

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

try:
    import wandb
except ImportError:
    wandb = None

from model import Siglip2ActionModel
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

class EarlyStopping:
    """Dừng huấn luyện sớm nếu điểm validation không cải thiện sau số epoch chỉ định."""
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

    def __call__(self, score, logger):
        if self.best_score is None:
            self.best_score = score
        # Nếu đang đo Loss (min), score không được lớn hơn best_score - min_delta
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # Nếu đang đo Acc (max), score không được nhỏ hơn best_score + min_delta
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def run_train_epoch(epoch, model, dataloader, criterion, optimizer, scheduler, device, accumulation_steps=1):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    base_model = getattr(model, "module", model)
    target_classes = base_model.class_names
    class_to_idx = {name: idx for idx, name in enumerate(target_classes)}

    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc=f"Training - Epoch {epoch}", file=sys.stdout)
    for i, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        
        if "label_name" in batch:
            raw_labels = batch["label_name"]
            labels = torch.tensor([class_to_idx[lbl] for lbl in raw_labels], dtype=torch.long, device=device)
        else:
            labels = batch["label_id"].to(device)
            
        B = pixel_values.size(0)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 1. Get normalized video representations (B, D)
            v_norm, t_features, logit_scale, logit_bias = model(pixel_values, return_features=True)
            
            # For contrastive training on a batch, we extract text features corresponding *only* to the samples in the batch
            # t_features shape is typically (B, K, D) or we can select the text features matching the batch labels
            batch_text_feats = t_features[torch.arange(B), labels] # Shape: (B, D)
            
            # 2. Compute Cosine Similarities scaled by SigLIP's logit_scale
            # Symmetric Contrastive Loss (Image-to-Text and Text-to-Image)
            logits_per_image = logit_scale * torch.matmul(v_norm, batch_text_feats.t()) + logit_bias
            logits_per_text = logits_per_image.t()
            
            # Ground truth targets are matching indices along the batch diagonal (0, 1, 2, ..., B-1)
            gt_targets = torch.arange(B, device=device)
            
            loss_i2t = criterion(logits_per_image, gt_targets)
            loss_t2i = criterion(logits_per_text, gt_targets)
            loss = (loss_i2t + loss_t2i) / 2.0
            
            loss = loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += (loss.item() * accumulation_steps)
        
        # Track accuracy using the batch similarity matrix argmax
        _, predicted = torch.max(logits_per_image, 1)
        total += labels.size(0)
        correct += (predicted == gt_targets).sum().item()

        progress_bar.set_postfix({
            "Loss": f"{(loss.item() * accumulation_steps):.4f}", 
            "Acc": f"{100 * correct / total:.2f}%"
        })
        
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    scheduler.step()
    return epoch_loss, epoch_acc, peak_vram
    
def main():
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
    
    # 3. Dynamically merge current mode specific hyper-parameters onto flat runtime config
    mode_specific_config = raw_yaml["modes"][args.mode]
    config.update(mode_specific_config)
    
    target_log = config.get("log_file", f"logs/{args.mode}_{log_time}.log")
    logger = setup_logger(target_log)
    logger.info(f"Loaded master configuration profile from {args.config}")
    logger.info(f"Successfully compiled configuration vectors for operational target matrix: {args.mode}")

    # Enforce standard asset validation guards early
    if not os.path.exists(config["annotation_dir"]):
        raise FileNotFoundError(f"Missing core storage index alignment folder metadata at: {config['annotation_dir']}")

    # Set seeds for repeatability
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
    
    # Backup configurations and architecture scripts
    shutil.copy(args.config, os.path.join(run_dir, "ucf101_config_snapshot.yaml"))
    model_py_path = os.path.join(os.path.dirname(__file__), "model.py")
    if os.path.exists(model_py_path):
        shutil.copy(model_py_path, os.path.join(run_dir, "model_arch_backup.py"))

    if wandb is not None:
        wandb.init(
            project="action-siglip2-peft",
            name=run_name,
            config=config
        )

    dataset_name = config.get("dataset", "ucf101").lower()
    if dataset_name == "hmdb51":
        from datasets import HMDB51VideoDataset as ActionDataset
    else:
        from datasets import UCF101VideoDataset as ActionDataset

    # Đọc danh sách class SEEN từ config
    seen_classes = raw_yaml.get("zero_shot_splits", {}).get("seen_class_names", None)

    logger.info("Setting up Training Split and Data Loaders pipeline configurations...")
    train_dataset = ActionDataset(
        base_dir=config["base_dir"],
        annotation_dir=config["annotation_dir"],
        processor=processor,
        split=config["split"],
        num_frames=config["num_frames"],
        mode='train',
        allowed_classes=seen_classes
    )
    val_dataset = ActionDataset(
        base_dir=config["base_dir"],
        annotation_dir=config["annotation_dir"],
        processor=processor,
        split=config["split"],
        num_frames=config["num_frames"],
        mode='val',
        allowed_classes=seen_classes
    )

    g_train = torch.Generator().manual_seed(config["seed"])
    g_val = torch.Generator().manual_seed(config["seed"])

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # FIX: Use sched_getaffinity to get actual Slurm-allocated CPUs on Linux
    if hasattr(os, 'sched_getaffinity'):
        allocated_cpus = len(os.sched_getaffinity(0)) #type:ignore
        num_workers = min(config["num_workers"], allocated_cpus)
    else:
        num_workers = min(config["num_workers"], os.cpu_count()) if os.name != 'nt' else 0

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(), 
        worker_init_fn=worker_init_fn,
        generator=g_train
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(), 
        worker_init_fn=worker_init_fn,
        generator=g_val
    )

    class_names_list = train_dataset.unique_labels
    logger.info(f"Extracted unique tokens count elements targets: {len(class_names_list)}")

    # Construct PEFT LoraConfig object directly from parsed YAML options
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # Instantiate model with explicit mode options
    model = Siglip2ActionModel(
        model_name=config["model_name"],
        class_names=class_names_list,
        prompt_type=config["prompt_type"],
        manual_prompt_template=config["manual_prompt_template"],
        cocoop_hidden_dim=config["cocoop_hidden_dim"],
        lora_config=lora_config,
        unfreeze_backbone=config.get("unfreeze_backbone", False)
    )
    
    # NEW: Automatically utilize multiple A100s if allocated by Slurm
    if torch.cuda.device_count() > 1:
        logger.info(f"Multi-GPU detected! Wrapping model in DataParallel using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
        
    model = model.to(device)

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

    # Đếm tổng số lượng tham số học được (đơn vị: triệu tham số - M)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Total Parameters: {total_params / 1e6:.2f} M")
    logger.info(f"Trainable Parameters: {total_trainable_params / 1e6:.2f} M ({(total_trainable_params/total_params)*100:.2f}%)")

    criterion = nn.CrossEntropyLoss()

    # Dynamic parameter grouping for multi-rate optimization
    lora_params = []
    custom_head_params = [] 
    other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in n:
            lora_params.append(p)
        elif "temporal" in n or "meta_net" in n or "gamma" in n: # FIX: Added gamma
            custom_head_params.append(p)
        else:
            other_params.append(p)

    # Đọc trực tiếp lr_base và lr_head từ config
    param_groups = [
        {"params": lora_params, "lr": config.get("lr_base", 1e-4)},
        {"params": custom_head_params, "lr": config.get("lr_head", 1e-3)},
    ]
    if other_params:
        # FFT sẽ rơi vào other_params, dùng lr_base
        param_groups.append({"params": other_params, "lr": config.get("lr_base", 1e-4)})

    optimizer = optim.AdamW(param_groups, weight_decay=config["weight_decay"])    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get("scheduler_t_max", config["num_epochs"]))

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    start_epoch = 1
    best_val_loss = float('inf')

    # KHỞI TẠO EARLY STOPPING TẠI ĐÂY
    patience = config.get("early_stopping_patience", 5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0, mode='min') 
    logger.info(f"Early Stopping enabled with patience: {patience}")

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

    accumulation_steps = config.get("accumulation_steps", 1)

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        logger.info(f"--- Starting Epoch {epoch}/{config['num_epochs']} ---")

        start_time = time.time() # Bắt đầu bấm giờ

        train_loss, train_acc, peak_vram = run_train_epoch(
            epoch, model, train_loader, criterion, optimizer, scheduler, device, accumulation_steps
        )
        
        metrics = test.validate(epoch, val_loader, model, device)
        val_loss = metrics.get('val_loss', 0.0)

        epoch_time = time.time() - start_time # Kết thúc bấm giờ (giây)

        logger.info(f"Epoch Time: {epoch_time:.2f}s | Peak VRAM: {peak_vram:.2f} GB")
        logger.info(f"Epoch Summary -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Summary   -> Val Loss: {val_loss:.4f} | Top-1: {metrics['top1']:.2f}% | Top-5: {metrics['top5']:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']
        if wandb is not None:
            wandb.log({
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_top1": metrics["top1"],
                "val_top5": metrics["top5"],
                "val_f1": metrics["f1"],
                "epoch_time_seconds": epoch_time,
                "peak_vram_gb": peak_vram
            })

        state_dict_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        # 1. Save Lightweight periodic checkpoints (weights only)
        if epoch % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict_to_save,
                'val_acc': metrics['top1']
            }, os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt"))

        # 2. Save full state to last_checkpoint.pt for quick job resume recovery
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': metrics['top1']
        }, os.path.join(run_dir, "last_checkpoint.pt"))

        # 3. Preserve full state for best validation model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': metrics['top1']
            }, os.path.join(run_dir, "best_model.pt"))
            logger.info(f"New best model saved into check-points with Validation Loss: {best_val_loss:.4f}")
        # -----------------------------------------

        # Truyền val_loss vào Early Stopping
        early_stopping(val_loss, logger)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered! Đã {early_stopping.patience} epochs không có sự cải thiện nào.")
            break

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()