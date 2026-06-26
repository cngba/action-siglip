# Author: Cong
# Decoupled Inference Validation Script - Tracking Explicit Multi-Class Configurations
# Optimized for Level 5 Dual-Tower Co-Alignment and Hybrid Temporal Bridge Performance

import os
import sys
import time
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Prevent diagnostic network verification calls over HTTPS layers
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import wandb
except ImportError:
    wandb = None

from model import Siglip2FullLoRATemporalBridge
from datasets import UCF101VideoDataset


def setup_logging(output_dir):
    """Sets up localized file tracking logging outputs."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "evaluation_log.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Silence external core diagnostic network trackers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    return log_file


def validate(epoch, dataloader, model, device):
    """Clean evaluation checking loop isolating your final target list over to wandb."""
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []
    
    start_time = time.time()
    total_videos = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation (Epoch {epoch})", file=sys.stdout):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label_id"].to(device)
            
            total_videos += labels.size(0)
            logits = model(pixel_values)
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
            
    end_time = time.time()
    inference_duration = end_time - start_time
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 1. Top-1 Target Calculation
    top1_acc = accuracy_score(all_labels, all_preds) * 100
    
    # 2. Top-5 Target Calculation
    num_classes = all_logits.shape[1]
    k = min(5, num_classes)
    _, topk_indices = torch.topk(all_logits, k, dim=1)
    topk_correct = topk_indices.eq(torch.tensor(all_labels).view(-1, 1)).sum().item()
    top5_acc = (topk_correct / total_videos) * 100

    # 3. Macro Target Precision, Recall, and F1 calculations
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    precision_pct = precision * 100
    recall_pct = recall * 100
    f1_pct = f1 * 100

    # Console prints and logging tracking elements configurations
    report_header = "\n" + "="*50 + f"\n            DETAILED PERFORMANCE REPORT (EPOCH {epoch})           \n" + "="*50
    print(report_header)
    print(f"Top-1 Accuracy:  {top1_acc:.2f}%")
    print(f"Top-5 Accuracy:  {top5_acc:.2f}%")
    print(f"Macro F1-Score:  {f1_pct:.2f}%")
    print(f"Macro Precision: {precision_pct:.2f}%")
    print(f"Macro Recall:    {recall_pct:.2f}%")
    print("-"*50)
    print(f"Inference Time:  {inference_duration:.2f} seconds total")
    print(f"Throughput:      {(total_videos / max(0.001, inference_duration)):.2f} clips/sec")
    print("="*50 + "\n")

    # Sync only your strict target checklist parameters to your WandB Web Portal
    if wandb and wandb.run:
        try:
            wandb.log({
                "epoch": epoch,
                "val_top1": top1_acc,
                "val_top5": top5_acc,
                "val_f1": f1_pct,
                "val_precision": precision_pct,
                "val_recall": recall_pct
            })
        except Exception as e:
            print(f"[Wandb Telemetry Warning] Connection stream skipped: {e}")

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "f1": f1_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Target Script")
    parser.add_argument("--weights", type=str, default="./checkpoints/best_model.pt", help="Path to checkpoint state .pt file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch sizing layer step processing constraint")
    parser.add_argument("--output_dir", type=str, default="./logs", help="Directory location where evaluations write")
    
    # LoRA hyperparameter variables tracking shapes dimensions alignments
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    args = parser.parse_args()

    # Local environment path configurations matching Windows kagglehub caching metrics structure
    kaggle_root = "C:\\Users\\CONG\\.cache\\kagglehub\\datasets\\matthewjansen\\ucf101-action-recognition\\versions\\4"
    base_dir = kaggle_root
    annotation_dir = kaggle_root

    setup_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Targeting active processing evaluation hardware context: {device}")

    model_name = "google/siglip2-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_name)
    
    logging.info("Constructing video datasets validation loader configurations...")
    val_dataset = UCF101VideoDataset(
        base_dir=base_dir, 
        annotation_dir=annotation_dir, 
        processor=processor, 
        num_frames=8, 
        mode='val'
    )
    
    num_workers = 0 if os.name == 'nt' else 4
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )

    class_list = val_dataset.unique_labels
    
    # Initialize the matching Level 5 structural architecture
    model = Siglip2FullLoRATemporalBridge(
        model_name=model_name, 
        class_names=class_list,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    ).to(device)

    if args.weights and os.path.isfile(args.weights):
        logging.info(f"Loading checkpoint weights directly from: {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        current_epoch = checkpoint.get('epoch', 0)
    else:
        logging.warning("No checkpoint metrics matched. Running baseline testing evaluations over randomized layers.")
        current_epoch = 0

    if wandb:
        wandb.init(
            project="action-siglip",
            name="level-5-standalone-eval",
            config=vars(args)
        )

    validate(current_epoch, val_loader, model, device)
    
    if wandb:
        wandb.finish()