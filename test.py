# action-siglip/test.py
# Author: Cong

import os
import sys
import time
import yaml
import argparse
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import LoraConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Prevent diagnostic network verification calls over HTTPS layers
os.environ["HF_HUB_OFFLINE"] = "1"

try:
    import wandb
except ImportError:
    wandb = None

from model import Siglip2ActionModel
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


@torch.no_grad()
def validate(epoch, dataloader, model, device, unseen_class_names=None, is_zero_shot=False):
    """
    Clean evaluation checking loop computing metrics based on L2-normalized cosine similarities.
    Supports both traditional validation and Zero-Shot Learning scenarios.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_logits = []

    total_val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    total_videos = 0

    # SAFEGUARD: Extract underlying model if wrapped in DataParallel
    base_model = getattr(model, "module", model)

    # 1. Xác định chính xác danh sách class đang đánh giá
    target_classes = unseen_class_names if (is_zero_shot and unseen_class_names is not None) else base_model.class_names
    
    # 2. Bắt buộc map tên lớp nguyên bản (VD: 'ApplyEyeMakeup') sang index [0 ... K-1] của Logits Matrix
    class_to_idx = {name: idx for idx, name in enumerate(target_classes)}
    # Target class selection based on scenario
    desc_msg = f"Zero-Shot Evaluation" if is_zero_shot else f"Validation (Epoch {epoch})"
    
    for batch in tqdm(dataloader, desc=desc_msg, file=sys.stdout):
        pixel_values = batch["pixel_values"].to(device)
        
        # FIX CỐT LÕI: Chắc chắn rằng label index thu được trùng khớp 100% với vị trí cột Logits
        if "label_name" in batch:
            raw_labels = batch["label_name"]
            labels = torch.tensor([class_to_idx[lbl] for lbl in raw_labels], dtype=torch.long, device=device)
        else:
            labels = batch["label_id"].to(device)
            
        total_videos += labels.size(0)
        
        # Modern PyTorch Autocast API
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            # FIX: Truyền target_classes vào unseen_class_names để Model và Validation dùng chung 1 List
            logits = model(pixel_values, unseen_class_names=target_classes, is_zero_shot=is_zero_shot)

            loss = criterion(logits, labels)
            total_val_loss += loss.item() * labels.size(0)
        
        _, predicted = torch.max(logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())
            
    end_time = time.time()
    inference_duration = end_time - start_time
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    avg_val_loss = total_val_loss / max(1, total_videos)
    
    # 1. Top-1 Target Calculation
    top1_acc = accuracy_score(all_labels, all_preds) * 100
    
    # 2. Top-5 Target Calculation (dynamically bounded by class count)
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

    # Console output and logging
    report_title = "DETAILED ZERO-SHOT REPORT" if is_zero_shot else "DETAILED PERFORMANCE REPORT"
    report_header = "\n" + "="*50 + f"\n            {report_title}            \n" + "="*50
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

    if wandb and wandb.run:
        try:
            prefix = "zero_shot_" if is_zero_shot else "val_"
            wandb.log({
                "epoch": epoch,
                f"{prefix}top1": top1_acc,
                f"{prefix}top5": top5_acc,
                f"{prefix}f1": f1_pct,
                f"{prefix}precision": precision_pct,
                f"{prefix}recall": recall_pct
            })
        except Exception as e:
            print(f"[Wandb Telemetry Warning] Connection stream skipped: {e}")

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "f1": f1_pct,
        "val_loss": avg_val_loss
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Target Script")
    parser.add_argument('--config', '-cfg', default='configs/ucf101.yaml', help='Path to the unified YAML file')
    parser.add_argument('--mode', '-m', default='train_full_peft', help='Target mode matrix selection from YAML')
    parser.add_argument("--weights", type=str, default="", help="Optional path to checkpoint weights")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Missing configuration file at: {args.config}")
        
    with open(args.config, 'r') as f:
        raw_yaml = yaml.safe_load(f)

    if "modes" not in raw_yaml or args.mode not in raw_yaml["modes"]:
        raise KeyError(f"Selected target mode configuration option '{args.mode}' not discovered in the YAML map matrix.")

    # Flatten runtime configuration
    config = {
        "model_name": raw_yaml.get("model_name", "google/siglip2-base-patch16-224"),
        "seed": raw_yaml.get("seed", 1024),
        "base_dir": raw_yaml["data"]["base_dir"],
        "annotation_dir": raw_yaml["data"]["annotation_dir"],
        "split": raw_yaml["data"]["split"],
        "num_frames": raw_yaml["data"]["num_segments"],
        "num_workers": raw_yaml["data"]["workers"]
    }
    
    mode_specific_config = raw_yaml["modes"][args.mode]
    config.update(mode_specific_config)
    
    setup_logging(os.path.dirname(config.get("log_file", f"logs/{args.mode}_eval.log")))
    logging.info(f"Loaded master configuration profile for evaluation mode: {args.mode}")

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Targeting active processing evaluation hardware context: {device}")

    processor = AutoProcessor.from_pretrained(config["model_name"])
    
    logging.info("Constructing video datasets validation loader configurations...")

    # 1. Định tuyến Dataset linh hoạt từ file cấu hình YAML
    dataset_name = config.get("dataset", "ucf101").lower()
    if dataset_name == "hmdb51":
        from datasets.hmdb51 import HMDB51VideoDataset as ActionDataset
        logging.info("=> Routing to HMDB51 Data Pipeline")
    else:
        from datasets.ucf101 import UCF101VideoDataset as ActionDataset
        logging.info("=> Routing to UCF101 Data Pipeline")
    
    # 2. Nhận diện các mode Zero-Shot (zero_shot, zero_shot_hmdb, zero_shot_fft, v.v.)
    is_zs_mode = ("zero_shot" in args.mode)

    # 3. Nạp danh sách class tương ứng
    if is_zs_mode:
        target_classes = raw_yaml.get("zero_shot_splits", {}).get("unseen_class_names", None)
    else:
        target_classes = raw_yaml.get("zero_shot_splits", {}).get("seen_class_names", None)
    
    val_dataset = ActionDataset(
        base_dir=config["base_dir"], 
        annotation_dir=config["annotation_dir"], 
        processor=processor, 
        split=config["split"],
        num_frames=config["num_frames"], 
        mode='test',
        prompt_template=config.get("manual_prompt_template", "A video of a person performing {}"),
        allowed_classes=target_classes
    )
    
    g_val = torch.Generator()
    g_val.manual_seed(config["seed"])

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if hasattr(os, 'sched_getaffinity'):
        allocated_cpus = len(os.sched_getaffinity(0)) #type:ignore
        num_workers = min(config["num_workers"], allocated_cpus)
    else:
        num_workers = min(config["num_workers"], os.cpu_count()) if os.name != 'nt' else 0

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=g_val
    )

    class_list = val_dataset.unique_labels
    
    # Construct PEFT LoraConfig matching model/train setup
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    # Initialize model matching model.py signature
    model = Siglip2ActionModel(
        model_name=config["model_name"], 
        class_names=class_list,
        prompt_type=config["prompt_type"],
        manual_prompt_template=config["manual_prompt_template"],
        cocoop_hidden_dim=config["cocoop_hidden_dim"],
        lora_config=lora_config,
        unfreeze_backbone=config.get("unfreeze_backbone", False)
    ).to(device)

    current_epoch = 0
    unseen_classes = None

    if is_zs_mode:
        logging.info("Executing pure Zero-Shot baseline performance evaluation over unseen action domains.")
        unseen_classes = config.get("unseen_class_names", class_list)
    
    checkpoint_target = args.weights if args.weights else os.path.join(config["checkpoint_dir"], "best_model.pt")
    if checkpoint_target and os.path.isfile(checkpoint_target):
        logging.info(f"Loading fine-tuned checkpoint weights directly from: {checkpoint_target}")
        checkpoint = torch.load(checkpoint_target, map_location=device, weights_only=False)
        
        # Lấy state_dict từ checkpoint an toàn
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Xử lý prefix 'module.' và ánh xạ lại tên biến 'temporal_module'
        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            
            # Dịch tên biến cũ sang kiến trúc mới
            if name.startswith('temporal_') and not name.startswith('temporal_module.'):
                name = name.replace('temporal_', 'temporal_module.')
                
            clean_state_dict[name] = v
        
        model.load_state_dict(clean_state_dict, strict=False)
        current_epoch = checkpoint.get('epoch', 0)
    else:
        logging.warning(f"No checkpoint matched at destination: '{checkpoint_target}'. "
                        f"Evaluating vanilla pre-trained weights instead.")

    # Accelerate inference if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logging.info(f"Multi-GPU detected! Wrapping model in DataParallel for faster evaluation.")
        model = nn.DataParallel(model)


    if wandb:
        project_name = "action-siglip2-zeroshot" if is_zs_mode else "action-siglip2-peft-eval"
        wandb.init(
            project=project_name,
            name=f"eval_{args.mode}_{config['num_frames']}f",
            config=config
        )

    # Execute evaluation pass
    validate(
        epoch=current_epoch, 
        dataloader=val_loader, 
        model=model, 
        device=device, 
        unseen_class_names=unseen_classes, 
        is_zero_shot=is_zs_mode
    )
    
    if wandb:
        wandb.finish()